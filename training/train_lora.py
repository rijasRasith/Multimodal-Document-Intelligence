from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

from data.dataset import InvoiceDataset
from data.schema import EXTRACTION_PROMPT
from evaluation.metrics import compute_field_f1, parse_model_output


def build_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def build_lora_config(rank: int, alpha: int, dropout: float = 0.05) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


def load_model_and_processor(model_name: str, use_4bit: bool = True):
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    if use_4bit and torch.cuda.is_available():
        bnb_config = build_bnb_config()
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    return model, processor


def collate_fn(batch: list[dict]) -> dict:
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        values = [item[key] for item in batch]
        if key == "labels":
            max_len = max(v.shape[0] for v in values)
            padded = torch.full((len(values), max_len), -100, dtype=torch.long)
            for i, v in enumerate(values):
                padded[i, : v.shape[0]] = v
            collated[key] = padded
        elif isinstance(values[0], torch.Tensor):
            try:
                collated[key] = torch.stack(values)
            except RuntimeError:
                max_len = max(v.shape[0] for v in values)
                padded = torch.zeros(len(values), max_len, dtype=values[0].dtype)
                for i, v in enumerate(values):
                    padded[i, : v.shape[0]] = v
                collated[key] = padded
        else:
            collated[key] = values
    return collated


def validate_epoch(model, processor, val_dir: Path, device: torch.device, max_samples: int = 100) -> dict:
    model.eval()
    img_dir = val_dir / "images"
    lbl_dir = val_dir / "labels"

    predictions = []
    ground_truths = []
    img_paths = sorted(img_dir.glob("*.png"))[:max_samples]

    from PIL import Image

    for img_path in img_paths:
        lbl_path = lbl_dir / img_path.with_suffix(".json").name
        if not lbl_path.exists():
            continue
        with open(lbl_path) as f:
            gt = json.load(f)
        ground_truths.append(gt)

        image = Image.open(img_path).convert("RGB")
        inputs = processor(text=EXTRACTION_PROMPT, images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        raw = processor.batch_decode(generated, skip_special_tokens=True)[0]
        parsed = parse_model_output(raw)
        predictions.append(parsed if parsed else {})

    model.train()
    return compute_field_f1(predictions, ground_truths)


def train(
    model_name: str,
    data_dir: Path,
    output_dir: Path,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    learning_rate: float = 1.5e-4,
    num_epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    warmup_ratio: float = 0.05,
    mlflow_experiment: str = "invoice_qlora_training",
    use_4bit: bool = True,
) -> Path:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, processor = load_model_and_processor(model_name, use_4bit=use_4bit)
    lora_config = build_lora_config(rank=lora_rank, alpha=lora_alpha)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = InvoiceDataset(data_dir / "train", processor, EXTRACTION_PROMPT)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01,
    )

    total_steps = math.ceil(len(train_loader) / gradient_accumulation_steps) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run(run_name=f"qlora_r{lora_rank}_a{lora_alpha}_lr{learning_rate}"):
        mlflow.log_params({
            "model_name": model_name,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "warmup_ratio": warmup_ratio,
            "use_4bit": use_4bit,
        })

        global_step = 0
        best_val_f1 = 0.0
        best_adapter_path = output_dir / "best_adapter"

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                pixel_values = batch.get("pixel_values")
                if pixel_values is not None and isinstance(pixel_values, torch.Tensor):
                    pixel_values = pixel_values.to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                )
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()
                epoch_loss += loss.item() * gradient_accumulation_steps

                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    mlflow.log_metric("train_loss_step", epoch_loss / (step + 1), step=global_step)

            avg_epoch_loss = epoch_loss / len(train_loader)
            mlflow.log_metric("train_loss_epoch", avg_epoch_loss, step=epoch)
            print(f"Epoch {epoch + 1}/{num_epochs} — Loss: {avg_epoch_loss:.4f}")

            val_metrics = validate_epoch(model, processor, data_dir / "val", device)
            val_f1 = val_metrics.get("macro_f1", 0.0)
            mlflow.log_metric("val_macro_f1", val_f1, step=epoch)
            print(f"  Val Macro F1: {val_f1:.4f}")

            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}"
            model.save_pretrained(str(checkpoint_path))
            processor.save_pretrained(str(checkpoint_path))

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                model.save_pretrained(str(best_adapter_path))
                processor.save_pretrained(str(best_adapter_path))
                print(f"  New best adapter saved (val F1: {val_f1:.4f})")

        mlflow.log_metric("best_val_f1", best_val_f1)
        mlflow.log_artifact(str(best_adapter_path))

    return best_adapter_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1.5e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--mlflow_experiment", default="invoice_qlora_training")
    args = parser.parse_args()

    train(
        model_name=args.model_name,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_4bit=not args.no_4bit,
        mlflow_experiment=args.mlflow_experiment,
    )
