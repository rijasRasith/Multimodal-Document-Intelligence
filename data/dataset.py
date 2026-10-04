from __future__ import annotations

import json
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image


class InvoiceDataset(Dataset):
    def __init__(self, split_dir: Path, processor, prompt: str):
        self.split_dir = Path(split_dir)
        self.processor = processor
        self.prompt = prompt
        self.samples = self._load_samples()

    def _load_samples(self) -> list[dict]:
        img_dir = self.split_dir / "images"
        lbl_dir = self.split_dir / "labels"
        samples = []
        for img_path in sorted(img_dir.glob("*.png")):
            lbl_path = lbl_dir / img_path.with_suffix(".json").name
            if not lbl_path.exists():
                continue
            with open(lbl_path) as f:
                label = json.load(f)
            samples.append({"image_path": str(img_path), "label": label})
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        target = json.dumps(sample["label"], separators=(",", ":"))
        inputs = self.processor(
            text=self.prompt,
            images=image,
            return_tensors="pt",
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = self.processor.tokenizer(
            target,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512,
        ).input_ids.squeeze(0)
        return inputs
