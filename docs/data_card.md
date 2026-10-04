# Data Card: Multimodal Invoice Extraction Dataset

## Overview

A dataset for structured field extraction from invoice and receipt images.

## Sources

| Source | Type | Count | License | Notes |
|--------|------|-------|---------|-------|
| Synthetic (this project) | Generated | 2,000 | None (original) | Perfect GT, controlled noise |
| SROIE | Real receipts | ~1,000 | Research only | Huang et al., ICDAR 2019 |

## Schema

Each sample: one image (PNG) + one JSON label.

```json
{
  "vendor": "string or null",
  "date": "YYYY-MM-DD or null",
  "invoice_number": "string or null",
  "line_items": [
    {"description": "string", "quantity": number|null, "unit_price": number|null, "total": number|null}
  ],
  "subtotal": "number or null",
  "tax": "number or null",
  "total": "number or null"
}
```

Missing fields are represented as `null`, never guessed or omitted.

## Splits

| Split | Count | Notes |
|-------|-------|-------|
| train | 1,500 | With noise augmentation |
| val | 250 | With noise augmentation |
| test | 250 | Clean + 50 noisy images; LOCKED |

Split strategy: by document template/vendor. No template appears in both train and test.

## Known Biases

- Synthetic invoices are English-only with Western date and currency formats.
- All vendors are fictional; real-world vendor name diversity is not represented.
- Layout variety is limited to the 20 templates defined in `data/generate.py`.
- SROIE covers Southeast Asian receipt formats which may differ from Western invoices.

## Augmentations Applied (train/val only)

- Random rotation: ±3°
- Gaussian noise
- Gaussian blur
- Brightness variation: 75%–110%

## Citation

If you use the SROIE portion:
```
Huang, Z., et al. "ICDAR2019 competition on scanned receipt OCR and information extraction."
ICDAR 2019.
```
