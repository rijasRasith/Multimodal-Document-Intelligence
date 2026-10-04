from __future__ import annotations

import argparse
from pathlib import Path

import mlflow


HYPERPARAMETER_GRID = [
    {"lora_rank": 8,  "lora_alpha": 16, "learning_rate": 1e-4,  "num_epochs": 2},
    {"lora_rank": 8,  "lora_alpha": 16, "learning_rate": 2e-4,  "num_epochs": 3},
    {"lora_rank": 16, "lora_alpha": 32, "learning_rate": 1e-4,  "num_epochs": 3},
    {"lora_rank": 16, "lora_alpha": 32, "learning_rate": 1.5e-4, "num_epochs": 3},
    {"lora_rank": 32, "lora_alpha": 64, "learning_rate": 1e-4,  "num_epochs": 2},
    {"lora_rank": 32, "lora_alpha": 64, "learning_rate": 1.5e-4, "num_epochs": 3},
]


def run_grid_search(
    model_name: str,
    data_dir: Path,
    output_dir: Path,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    use_4bit: bool = True,
    mlflow_experiment: str = "invoice_qlora_training",
) -> None:
    from training.train_lora import train

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    best_f1 = 0.0
    best_config = None

    results = []
    for i, config in enumerate(HYPERPARAMETER_GRID):
        print(f"\n{'='*60}")
        print(f"Experiment {i + 1}/{len(HYPERPARAMETER_GRID)}: {config}")
        print(f"{'='*60}")

        run_output = output_dir / f"run_{i + 1}_r{config['lora_rank']}_lr{config['learning_rate']}"

        try:
            adapter_path = train(
                model_name=model_name,
                data_dir=data_dir,
                output_dir=run_output,
                lora_rank=config["lora_rank"],
                lora_alpha=config["lora_alpha"],
                learning_rate=config["learning_rate"],
                num_epochs=config["num_epochs"],
                batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                use_4bit=use_4bit,
                mlflow_experiment=mlflow_experiment,
            )

            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(mlflow_experiment)
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )
            val_f1 = float(runs[0].data.metrics.get("best_val_f1", 0.0)) if runs else 0.0

            results.append({**config, "val_f1": val_f1, "adapter_path": str(adapter_path)})
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_config = config

            print(f"  Val F1: {val_f1:.4f}")

        except Exception as e:
            print(f"  Run failed: {e}")
            results.append({**config, "val_f1": 0.0, "error": str(e)})

    print(f"\n{'='*60}")
    print(f"Grid Search Complete")
    print(f"Best Val F1: {best_f1:.4f}")
    print(f"Best Config: {best_config}")
    print(f"{'='*60}")

    import json
    results_path = output_dir / "grid_search_results.json"
    with open(results_path, "w") as f:
        json.dump({"best_f1": best_f1, "best_config": best_config, "all_runs": results}, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--mlflow_experiment", default="invoice_qlora_training")
    args = parser.parse_args()

    run_grid_search(
        model_name=args.model_name,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_4bit=not args.no_4bit,
        mlflow_experiment=args.mlflow_experiment,
    )
