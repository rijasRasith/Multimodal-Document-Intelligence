# Model Card: Invoice Extraction QLoRA Adapter

## Model Details

- **Base model**: Qwen/Qwen2-VL-2B-Instruct
- **Adapter type**: QLoRA (4-bit quantized LoRA)
- **Task**: Structured field extraction from invoice/receipt images → JSON
- **Language**: English

## Training

- **Dataset**: 1,500 synthetic + SROIE training examples
- **LoRA rank**: 16, alpha: 32
- **Learning rate**: 1.5e-4 (cosine schedule with 5% warmup)
- **Epochs**: 3
- **Precision**: 4-bit (NF4) base, BF16 compute
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

## Evaluation

Evaluated on a locked 250-sample test set. Results (to be filled after training):

| Metric | Base (zero-shot) | API Baseline | Fine-tuned |
|--------|-----------------|--------------|------------|
| Macro F1 | — | — | — |
| JSON Validity | — | — | — |
| Avg Latency | — | — | — |

## Intended Use

Automated extraction of structured data from scanned invoices and receipts. Intended for document processing pipelines where cost and data privacy make self-hosted models preferable to commercial APIs.

## Out-of-Scope Uses

- Handwritten documents
- Non-English invoices
- Medical or financial compliance use cases without additional validation

## Limitations

- May hallucinate vendor names for low-quality or blurry images
- Performance degrades on non-Western layouts and non-ASCII characters
- Inference on CPU takes 20–60 seconds per image

## Ethical Considerations

- Do not use with documents containing real personal data without appropriate anonymization
- Model outputs should be validated before use in automated payment or financial workflows

## License

Adapter weights follow the base model license: Apache 2.0.
