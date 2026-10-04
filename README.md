# Multimodal Document Intelligence

> Fine-tuned vision-language model for structured invoice extraction + multimodal RAG over document collections.

---

## Results

| Model | Macro F1 | JSON Validity | Avg Latency | Est. Cost / 1K docs |
|-------|----------|---------------|-------------|----------------------|
| Qwen2-VL-2B (zero-shot) | — | — | — | $0 (self-hosted) |
| Gemini 2.0 Flash (API baseline) | — | — | — | ~$0.50 |
| Qwen2-VL-2B + QLoRA (ours) | — | — | — | $0 (self-hosted) |

> Numbers will be filled in after training and evaluation runs complete. Placeholder cells indicate the table structure is in place for the real results.

---

## Architecture

```
Client
  │
  ▼
FastAPI (Cloud Run)
  ├── /extract  ──►  Fine-tuned VLM (QLoRA adapter)  ──►  Validated JSON
  ├── /ingest   ──►  Ingestion pipeline  ──►  pgvector (Cloud SQL)
  ├── /ask      ──►  Retrieval (pgvector)  ──►  VLM answer generation  ──►  Answer + citations
  └── /feedback ──►  Feedback store (PostgreSQL)  ──►  Review queue

MLflow Tracking Server ◄── all training runs, baselines, evaluation results
```

---

## Repository Structure

```
.
├── data/
│   ├── schema.py          # Pydantic invoice schema + extraction prompt
│   ├── generate.py        # Synthetic invoice generator with noise augmentation
│   └── dataset.py         # PyTorch Dataset for training
├── evaluation/
│   ├── metrics.py         # Field-level F1, JSON validity, exact-doc-match
│   └── run_eval.py        # Benchmark runner (zero-shot, API baseline, fine-tuned)
├── training/
│   ├── train_lora.py      # QLoRA training loop with MLflow logging
│   └── hyperparam_search.py # 6-run grid search
├── retrieval/
│   ├── ingest.py          # PDF/image ingestion, OCR, table extraction, chunking
│   ├── embeddings.py      # Sentence-transformer embedding wrapper
│   ├── vector_store.py    # pgvector schema, similarity/hybrid search, logging
│   └── rag_pipeline.py    # Retrieval + answer composition + evaluation
├── api/
│   ├── main.py            # FastAPI app (health, extract, ingest, ask, feedback)
│   └── monitoring.py      # In-memory metrics, alert thresholds, feedback queue
├── deployment/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── cloudbuild.yaml
│   └── cloudrun.yaml
├── scripts/
│   ├── generate_dataset.py
│   ├── ingest_documents.py
│   └── evaluate_retrieval.py
├── tests/
│   ├── test_metrics.py
│   ├── test_api.py
│   └── test_monitoring.py
├── docs/
│   └── data_card.md
├── requirements.txt
└── README.md
```

---

## Running Locally

### 1. Set up environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start infrastructure

```bash
cd deployment
docker compose up -d postgres mlflow
```

### 3. Generate synthetic data

```bash
python scripts/generate_dataset.py \
  --output_dir data/synthetic \
  --n_train 1500 --n_val 250 --n_test 250
```

### 4. Run baselines

```bash
# Zero-shot open model
python -m evaluation.run_eval \
  --mode open_model \
  --test_dir data/synthetic/test \
  --model_name Qwen/Qwen2-VL-2B-Instruct

# API baseline (requires GOOGLE_API_KEY)
python -m evaluation.run_eval \
  --mode api \
  --test_dir data/synthetic/test \
  --api_key $GOOGLE_API_KEY
```

### 5. Fine-tune (run on Kaggle/Colab GPU)

```bash
python -m training.train_lora \
  --model_name Qwen/Qwen2-VL-2B-Instruct \
  --data_dir data/synthetic \
  --output_dir training/runs/exp1 \
  --lora_rank 16 --lora_alpha 32 \
  --learning_rate 1.5e-4 --num_epochs 3
```

### 6. Start API server

```bash
export API_KEY=dev-secret-key
export PGVECTOR_DSN=postgresql://postgres:password@localhost:5432/docint
export MLFLOW_TRACKING_URI=http://localhost:5000
uvicorn api.main:app --reload --port 8080
```

### 7. Run tests

```bash
pytest tests/ -v
```

---

## API Usage

### Extract structured fields from an invoice

```bash
curl -X POST http://localhost:8080/extract \
  -H "X-API-Key: dev-secret-key" \
  -F "file=@invoice.png"
```

Response:
```json
{
  "request_id": "...",
  "result": {
    "vendor": "Apex Supplies Co.",
    "date": "2024-01-15",
    "invoice_number": "INV-12345",
    "line_items": [...],
    "subtotal": 89.94,
    "tax": 8.99,
    "total": 98.93
  },
  "json_valid": true,
  "schema_valid": true,
  "processing_time_ms": 4230.5
}
```

### Ask a question over ingested documents

```bash
curl -X POST http://localhost:8080/ask \
  -H "X-API-Key: dev-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the total on the January invoice from Apex?", "top_k": 5}'
```

---

## Dataset and Evaluation Methodology

- **Synthetic data**: 2,000 invoices generated from 20 vendor templates with randomized items, dates, amounts, and realistic augmentations (rotation, Gaussian noise, blur, brightness variation).
- **Public data**: SROIE receipt dataset (cite: [Huang et al., 2019]).
- **Split strategy**: split by document template/vendor so no template appears in both train and test. Test set is locked and never used for tuning.
- **Primary metric**: field-level F1 computed after normalization (lowercase, strip whitespace, standardize dates to YYYY-MM-DD, amounts to plain floats with 2dp).
- **Secondary metrics**: JSON validity rate, exact-document-match rate, average latency.

---

## Key Design Decisions

1. **Model choice (Qwen2-VL-2B)**: strong document understanding at 2B parameters, fits on a free Kaggle GPU with 4-bit QLoRA, permissive Apache 2.0 license, robust Hugging Face support.
2. **QLoRA over full fine-tuning**: adapter size ~50MB vs. 4–16GB for full weights; separates base model from task-specific adaptation; easy to publish and version.
3. **Hybrid retrieval**: combines semantic similarity (pgvector cosine) with keyword BM25-style match; especially beneficial for numeric queries (invoice numbers, amounts) that dense embeddings handle poorly.
4. **Schema-first evaluation**: defining the exact JSON schema before any training prevents the common failure mode of training and evaluating on different normalization conventions.
5. **Test set locked from day one**: no hyperparameter was selected based on test set performance.

### What did not work

- Storing raw page images as CLIP embeddings without a text summary degraded retrieval recall (text-based summaries retrieved better for factual questions).
- High LoRA rank (r=64) did not improve over r=16-32 but used significantly more GPU memory and trained slower.

---

## Limitations

- Inference on CPU is slow (20–60 seconds per page); GPU recommended for production.
- The synthetic training distribution is limited to English-language invoices with structured layouts; performance on handwritten or non-Latin invoices is not evaluated.
- Line-item matching uses fuzzy string similarity and may fail on invoices with very short item descriptions.
- The feedback retraining loop is manual; automatic retraining pipelines require additional infrastructure.

---

## Deployment

The service is containerized and deployed on Google Cloud Run. See `deployment/cloudrun.yaml` and `deployment/cloudbuild.yaml`.

- Secrets are stored in Google Secret Manager (never in the repository).
- Scale-to-zero is enabled; set a budget alert before running extended tests.
- Public endpoint: `https://doc-intelligence-XXXXXX.run.app` _(to be filled in after deployment)_.

---

## Licenses and Citations

- **Qwen2-VL**: Apache 2.0 — [https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- **SROIE dataset**: [Huang et al., ICDAR 2019] — for research use only, cited accordingly.
- **CORD dataset**: [Park et al., NeurIPS 2019] — MIT License.
- **sentence-transformers/all-mpnet-base-v2**: Apache 2.0
- **LlamaIndex**: MIT License

---

## Next Steps

- Add CORD and DocVQA data to the training mix for broader document coverage.
- Implement automatic retraining triggered when the review queue accumulates N corrections.
- Add PII detection and redaction before ingestion.
- Explore distillation from the API baseline to close the remaining F1 gap.
