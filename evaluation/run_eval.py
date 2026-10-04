from __future__ import annotations

import json
import time
import argparse
from pathlib import Path

import mlflow
from PIL import Image
from tqdm import tqdm

from evaluation.metrics import compute_field_f1, parse_model_output, print_metrics_table


def run_open_model_baseline(
    test_dir: Path,
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    mlflow_experiment: str = "invoice_extraction_baselines",
) -> dict:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    import torch

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    from data.schema import EXTRACTION_PROMPT
    return _evaluate_model(
        model=model,
        processor=processor,
        test_dir=test_dir,
        run_name="base_model_zero_shot",
        model_name=model_name,
        prompt=EXTRACTION_PROMPT,
        mlflow_experiment=mlflow_experiment,
    )


def run_api_baseline(
    test_dir: Path,
    api_key: str,
    model_name: str = "gemini-2.0-flash",
    mlflow_experiment: str = "invoice_extraction_baselines",
) -> dict:
    import google.generativeai as genai
    import base64

    genai.configure(api_key=api_key)
    client = genai.GenerativeModel(model_name)

    from data.schema import EXTRACTION_PROMPT

    test_dir = Path(test_dir)
    img_dir = test_dir / "images"
    lbl_dir = test_dir / "labels"

    predictions = []
    ground_truths = []
    latencies = []

    for img_path in tqdm(sorted(img_dir.glob("*.png")), desc="API Baseline"):
        lbl_path = lbl_dir / img_path.with_suffix(".json").name
        if not lbl_path.exists():
            continue

        with open(lbl_path) as f:
            gt = json.load(f)
        ground_truths.append(gt)

        image = Image.open(img_path).convert("RGB")
        start = time.time()
        try:
            response = client.generate_content([EXTRACTION_PROMPT, image])
            raw = response.text
        except Exception:
            raw = ""
        latency = time.time() - start

        parsed = parse_model_output(raw)
        predictions.append(parsed if parsed else {})
        latencies.append(latency)

    metrics = compute_field_f1(predictions, ground_truths)
    metrics["avg_latency_s"] = round(sum(latencies) / len(latencies), 4) if latencies else 0.0

    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run(run_name=f"api_baseline_{model_name}"):
        mlflow.log_param("model", model_name)
        mlflow.log_param("baseline_type", "api")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)

    print_metrics_table(metrics, model_name=f"API Baseline ({model_name})")
    return metrics


def _evaluate_model(
    model,
    processor,
    test_dir: Path,
    run_name: str,
    model_name: str,
    prompt: str,
    mlflow_experiment: str,
) -> dict:
    import torch

    test_dir = Path(test_dir)
    img_dir = test_dir / "images"
    lbl_dir = test_dir / "labels"

    predictions = []
    ground_truths = []
    latencies = []

    for img_path in tqdm(sorted(img_dir.glob("*.png")), desc=run_name):
        lbl_path = lbl_dir / img_path.with_suffix(".json").name
        if not lbl_path.exists():
            continue

        with open(lbl_path) as f:
            gt = json.load(f)
        ground_truths.append(gt)

        image = Image.open(img_path).convert("RGB")
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        start = time.time()
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        latency = time.time() - start

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        raw = processor.batch_decode(generated, skip_special_tokens=True)[0]

        parsed = parse_model_output(raw)
        predictions.append(parsed if parsed else {})
        latencies.append(latency)

    metrics = compute_field_f1(predictions, ground_truths)
    metrics["avg_latency_s"] = round(sum(latencies) / len(latencies), 4) if latencies else 0.0

    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model", model_name)
        mlflow.log_param("baseline_type", "open_model_zero_shot")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)

    print_metrics_table(metrics, model_name=f"Open Model Zero-Shot ({model_name})")
    return metrics


def run_finetuned_evaluation(
    test_dir: Path,
    base_model_name: str,
    adapter_path: str,
    mlflow_experiment: str = "invoice_extraction_baselines",
    run_name: str = "finetuned_lora",
) -> dict:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from peft import PeftModel
    import torch

    processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    from data.schema import EXTRACTION_PROMPT
    metrics = _evaluate_model(
        model=model,
        processor=processor,
        test_dir=test_dir,
        run_name=run_name,
        model_name=f"{base_model_name}+LoRA",
        prompt=EXTRACTION_PROMPT,
        mlflow_experiment=mlflow_experiment,
    )
    print_metrics_table(metrics, model_name=f"Fine-tuned LoRA ({base_model_name})")
    return metrics


def perform_error_analysis(
    test_dir: Path,
    predictions: list[dict],
    ground_truths: list[dict],
    output_file: Path,
    max_samples: int = 50,
) -> None:
    failures = []
    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        if pred == gt:
            continue
        diff = {}
        for field in ["vendor", "date", "invoice_number", "subtotal", "tax", "total"]:
            pv = pred.get(field) if isinstance(pred, dict) else None
            gv = gt.get(field)
            if str(pv) != str(gv):
                diff[field] = {"predicted": pv, "ground_truth": gv}
        if diff:
            failures.append({"index": i, "differences": diff})
        if len(failures) >= max_samples:
            break

    categories = {
        "wrong_amount": 0,
        "wrong_date": 0,
        "wrong_vendor": 0,
        "hallucinated_field": 0,
        "missing_field": 0,
        "malformed_json": 0,
    }

    for failure in failures:
        if not isinstance(predictions[failure["index"]], dict):
            categories["malformed_json"] += 1
            continue
        for field, vals in failure.get("differences", {}).items():
            pv, gv = vals["predicted"], vals["ground_truth"]
            if field in ("subtotal", "tax", "total"):
                categories["wrong_amount"] += 1
            elif field == "date":
                categories["wrong_date"] += 1
            elif field == "vendor":
                categories["wrong_vendor"] += 1
            elif pv is not None and gv is None:
                categories["hallucinated_field"] += 1
            elif pv is None and gv is not None:
                categories["missing_field"] += 1

    analysis = {"total_failures": len(failures), "categories": categories, "samples": failures[:20]}
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Error analysis saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["open_model", "api", "finetuned"], required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--mlflow_experiment", default="invoice_extraction_baselines")
    args = parser.parse_args()

    if args.mode == "open_model":
        run_open_model_baseline(Path(args.test_dir), args.model_name, args.mlflow_experiment)
    elif args.mode == "api":
        run_api_baseline(Path(args.test_dir), args.api_key, args.model_name, args.mlflow_experiment)
    elif args.mode == "finetuned":
        run_finetuned_evaluation(Path(args.test_dir), args.model_name, args.adapter_path, args.mlflow_experiment)
