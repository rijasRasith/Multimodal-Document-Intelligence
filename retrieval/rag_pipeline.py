from __future__ import annotations

import io
from typing import Optional

from PIL import Image

from retrieval.embeddings import embed_text
from retrieval.vector_store import similarity_search, hybrid_search


def retrieve(
    conn,
    question: str,
    top_k: int = 5,
    use_hybrid: bool = True,
    doc_id_filter: Optional[str] = None,
    content_type_filter: Optional[str] = None,
) -> list[dict]:
    query_embedding = embed_text(question)

    if use_hybrid:
        results = hybrid_search(conn, query_embedding, keyword=question, top_k=top_k)
        if not results:
            results = similarity_search(
                conn, query_embedding, top_k=top_k,
                content_type_filter=content_type_filter,
                doc_id_filter=doc_id_filter,
            )
    else:
        results = similarity_search(
            conn, query_embedding, top_k=top_k,
            content_type_filter=content_type_filter,
            doc_id_filter=doc_id_filter,
        )

    return results


def compose_answer(
    question: str,
    retrieved_chunks: list[dict],
    vlm_model,
    processor,
    conn=None,
) -> dict:
    import torch

    text_context = []
    page_images = []
    citations = []

    for chunk in retrieved_chunks:
        citation = f"{chunk['source_file']} (page {chunk['page_number']})"
        if chunk["content_type"] in ("text", "table"):
            text_context.append(f"[Source: {citation}]\n{chunk['content']}")
            citations.append(citation)
        elif chunk["content_type"] == "page_image" and chunk.get("image_data"):
            img = Image.open(io.BytesIO(chunk["image_data"])).convert("RGB")
            page_images.append(img)
            citations.append(citation)

    context_str = "\n\n".join(text_context) if text_context else "No text context retrieved."
    prompt = (
        f"Answer the following question using the provided context. "
        f"If the answer is not found in the context, respond with: 'I do not have enough information to answer this question.'\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    if page_images:
        primary_image = page_images[0]
        inputs = processor(text=prompt, images=primary_image, return_tensors="pt")
    else:
        inputs = processor(text=prompt, return_tensors="pt")

    inputs = {k: v.to(vlm_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = vlm_model.generate(**inputs, max_new_tokens=400, do_sample=False)

    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    answer_text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    unique_citations = list(dict.fromkeys(citations))

    return {
        "answer": answer_text,
        "citations": unique_citations,
        "retrieved_chunks": len(retrieved_chunks),
        "avg_retrieval_score": round(
            sum(c.get("similarity_score", 0.0) for c in retrieved_chunks) / max(len(retrieved_chunks), 1), 4
        ),
    }


def evaluate_retrieval(
    conn,
    qa_pairs: list[dict],
    top_k_values: list[int] = [1, 3, 5],
    use_hybrid: bool = True,
) -> dict:
    results = {f"recall@{k}": 0 for k in top_k_values}
    results["mrr"] = 0.0
    reciprocal_ranks = []

    for qa in qa_pairs:
        question = qa["question"]
        expected_source = qa["source_file"]
        expected_page = qa["page_number"]

        max_k = max(top_k_values)
        retrieved = retrieve(conn, question, top_k=max_k, use_hybrid=use_hybrid)

        found_at = None
        for rank, chunk in enumerate(retrieved, start=1):
            if (chunk["source_file"] == expected_source
                    and chunk["page_number"] == expected_page):
                found_at = rank
                break

        for k in top_k_values:
            if found_at is not None and found_at <= k:
                results[f"recall@{k}"] += 1

        reciprocal_ranks.append(1.0 / found_at if found_at else 0.0)

    n = len(qa_pairs)
    if n > 0:
        for k in top_k_values:
            results[f"recall@{k}"] = round(results[f"recall@{k}"] / n, 4)
        results["mrr"] = round(sum(reciprocal_ranks) / n, 4)

    return results
