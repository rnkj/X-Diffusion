from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate run artifacts")
    parser.add_argument("--run_dir", required=True, help="Run directory to validate")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_metrics(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def summarize_run(run_dir: Path) -> Dict[str, Any]:
    metrics = load_metrics(run_dir / "metrics.jsonl")
    data_audit = load_json(run_dir / "data_audit.json")
    dataset_summary = load_json(run_dir / "dataset_summary.json")
    reload_report = load_json(run_dir / "reload_report.json")
    if not metrics:
        raise ValueError("metrics.jsonl is empty")

    first_train = metrics[0]["train_loss"]
    min_train = min(row["train_loss"] for row in metrics)
    first_val = metrics[0]["val_loss"]
    max_val = max(row["val_loss"] for row in metrics)
    finite_losses = all(math.isfinite(row["train_loss"]) and math.isfinite(row["val_loss"]) for row in metrics)
    train_decreased = min_train < first_train
    val_stable = math.isfinite(first_val) and math.isfinite(max_val) and max_val <= max(first_val * 5.0, first_val + 10.0)

    mode = dataset_summary["mode"]
    class_dist = dataset_summary["train_class_distribution"]
    robot_count = class_dist.get("robot", 0)
    human_count = class_dist.get("human", 0)
    naive_has_both = mode != "naive" or (robot_count > 0 and human_count > 0)
    no_masking = not dataset_summary.get("integrate_classifier", True) and not reload_report.get("integrate_classifier", True)

    checks = {
        "no_skipped_or_corrupt_episodes": bool(data_audit["no_skipped_or_corrupt_episodes"]),
        "finite_losses": finite_losses,
        "train_loss_decreased": train_decreased,
        "val_not_catastrophic": val_stable,
        "checkpoint_reload_success": bool(reload_report["reload_success"]),
        "inference_success": bool(reload_report["inference_success"]),
        "naive_includes_both_robot_and_human": naive_has_both,
        "no_classifier_masking": no_masking,
    }
    valid = all(checks.values())
    return {
        "run_dir": str(run_dir),
        "mode": mode,
        "task": dataset_summary["task"],
        "checks": checks,
        "valid": valid,
        "metrics_summary": {
            "epochs": len(metrics),
            "first_train_loss": first_train,
            "final_train_loss": metrics[-1]["train_loss"],
            "min_train_loss": min_train,
            "first_val_loss": first_val,
            "final_val_loss": metrics[-1]["val_loss"],
            "max_val_loss": max_val,
        },
        "class_distribution": class_dist,
    }


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    report = summarize_run(run_dir)
    (run_dir / "validation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
