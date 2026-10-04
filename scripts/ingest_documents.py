from __future__ import annotations

import argparse
import json
from pathlib import Path

from retrieval.vector_store import get_connection, initialize_schema, upsert_document, insert_chunks
from retrieval.embeddings import embed_batch
from retrieval.ingest import ingest_pdf, ingest_image


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG index.")
    parser.add_argument("--input", required=True, help="Path to a file or directory of documents.")
    parser.add_argument("--dsn", default=None, help="PostgreSQL DSN string.")
    parser.add_argument("--no_vlm_summaries", action="store_true",
                        help="Skip VLM page summarization (faster, lower quality).")
    args = parser.parse_args()

    conn = get_connection(args.dsn)
    initialize_schema(conn)

    input_path = Path(args.input)
    files = []
    if input_path.is_dir():
        files.extend(input_path.glob("*.pdf"))
        files.extend(input_path.glob("*.png"))
        files.extend(input_path.glob("*.jpg"))
        files.extend(input_path.glob("*.jpeg"))
    elif input_path.is_file():
        files = [input_path]
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    vlm_model = None
    processor = None
    if not args.no_vlm_summaries:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        import torch, os
        model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2-VL-2B-Instruct")
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        vlm_model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        vlm_model.eval()

    total_chunks = 0
    for file_path in files:
        print(f"Ingesting: {file_path.name}")
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            nodes = ingest_pdf(file_path, vlm_model=vlm_model, processor=processor)
        else:
            nodes = ingest_image(file_path, vlm_model=vlm_model, processor=processor)

        texts = [n["content"] for n in nodes]
        embeddings = embed_batch(texts)
        for node, emb in zip(nodes, embeddings):
            node["embedding"] = emb

        doc_id = nodes[0]["doc_id"] if nodes else None
        if doc_id:
            upsert_document(conn, doc_id, file_path.name)
            insert_chunks(conn, nodes)
            total_chunks += len(nodes)
            print(f"  Indexed {len(nodes)} chunks (doc_id: {doc_id})")

    print(f"\nIngestion complete. Total chunks indexed: {total_chunks}")
    conn.close()


if __name__ == "__main__":
    main()
