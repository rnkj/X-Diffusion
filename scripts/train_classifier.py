from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))
from scripts.evaluate_classifier_noise_sweep import evaluate_noise_sweep
DEFAULT_DATA_ROOT = REPO_ROOT.parent / "X-Diffusion-Data"
PHYSICAL_REQUIRED_KEYS = ["ee_pos", "ee_euler", "gripper_open", "3d_tracks"]


@dataclass
class ClassifierTrainConfig:
    task: str = "mug_on_rack"
    run_name: str = "m5_classifier"
    seed: int = 42
    device: str = "cpu"
    epochs: int = 3
    steps_per_epoch: int = 8
    max_val_batches: int = 8
    batch_size: int = 4
    num_workers: int = 0
    lr: float = 1e-4
    grad_clip: float = 5.0
    obs_horizon: int = 1
    action_horizon: int = 8
    pred_horizon: int = 8
    use_ee_data: bool = True
    balanced_sampling_weights: List[float] = field(default_factory=lambda: [0.5, 0.5])
    train_demo_ids: List[str] = field(default_factory=lambda: [
        "demo00000",
        "demo00001",
        "demo00002",
        "demo00003",
        "demo00004",
    ])
    val_demo_ids: List[str] = field(default_factory=lambda: ["demo00005"])
    data_root: Optional[str] = None
    torch_compile: bool = False
    num_train_timesteps: int = 101
    add_noise_to_sample: bool = True
    add_noise_to_state_cond: bool = True
    threshold: float = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classifier training")
    parser.add_argument("--config_path", required=True, help="Path to YAML config file")
    return parser.parse_args()


def load_config(config_path: Path) -> ClassifierTrainConfig:
    payload = yaml.safe_load(config_path.read_text()) or {}
    return ClassifierTrainConfig(**payload)


def resolve_data_root(raw_root: Optional[str]) -> Path:
    return Path(raw_root).expanduser().resolve() if raw_root else DEFAULT_DATA_ROOT.resolve()


def resolve_run_dir(cfg: ClassifierTrainConfig, data_root: Path) -> Path:
    return (data_root / "_private" / "runs" / cfg.run_name).resolve()


def ensure_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        if not link_path.is_symlink() or link_path.resolve() != target_path.resolve():
            raise RuntimeError(f"Wrapper path already exists with unexpected target: {link_path}")
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target_path, link_path, target_is_directory=target_path.is_dir())


def build_dataset_dirs(cfg: ClassifierTrainConfig, data_root: Path, run_dir: Path) -> Dict[str, Path]:
    del run_dir
    return {
        "robot": (data_root / "retargeted" / cfg.task / "robot").resolve(),
        "human": (data_root / "retargeted" / cfg.task / "human").resolve(),
    }


def write_fixed_split(run_dir: Path, cfg: ClassifierTrainConfig) -> Path:
    split = {
        "train": {"robot": list(cfg.train_demo_ids), "human": list(cfg.train_demo_ids)},
        "val": {"robot": list(cfg.val_demo_ids), "human": list(cfg.val_demo_ids)},
    }
    split_path = run_dir / "fixed_split.json"
    split_path.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")
    return split_path


def audit_demo_file(path: Path, pred_horizon: int) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "path": str(path.resolve()) if path.exists() else str(path),
        "exists": path.exists(),
        "missing_keys": [],
        "num_frames": None,
        "error": None,
    }
    if not path.exists():
        return info
    try:
        with h5py.File(path, "r") as handle:
            for key in PHYSICAL_REQUIRED_KEYS:
                if key not in handle:
                    info["missing_keys"].append(key)
            if "ee_pos" in handle:
                info["num_frames"] = int(handle["ee_pos"].shape[0])
            if info["num_frames"] is not None and info["num_frames"] < pred_horizon:
                info["error"] = f"episode shorter than pred_horizon={pred_horizon}"
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def audit_selected_demos(wrapper_dirs: Dict[str, Path], cfg: ClassifierTrainConfig) -> Dict[str, Any]:
    requested = cfg.train_demo_ids + cfg.val_demo_ids
    report: Dict[str, Any] = {
        "task": cfg.task,
        "mode": "classifier",
        "train_demo_ids": list(cfg.train_demo_ids),
        "val_demo_ids": list(cfg.val_demo_ids),
        "datasets": {},
        "missing_files": [],
        "corrupt_files": [],
        "modalities_used": ["physical"],
        "no_skipped_or_corrupt_episodes": True,
    }
    for dataset_type, wrapper_dir in wrapper_dirs.items():
        entries = []
        for demo_id in requested:
            info = audit_demo_file(wrapper_dir / f"{demo_id}.h5", cfg.pred_horizon)
            entries.append(info)
            if not info["exists"]:
                report["missing_files"].append(info["path"])
            if info["missing_keys"] or info["error"]:
                report["corrupt_files"].append(info)
        report["datasets"][dataset_type] = {
            "wrapper_dir": str(wrapper_dir),
            "real_dir": str(wrapper_dir.resolve()),
            "episodes": entries,
        }
    report["no_skipped_or_corrupt_episodes"] = not report["missing_files"] and not report["corrupt_files"]
    return report


def save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def write_command_log(run_dir: Path) -> None:
    (run_dir / "command.txt").write_text(" ".join([sys.executable, *sys.argv]) + "\n")


def build_resolved_config(cfg: ClassifierTrainConfig, data_root: Path, run_dir: Path, split_path: Path, dataset_dirs: Dict[str, Path]) -> Dict[str, Any]:
    payload = asdict(cfg)
    payload.update(
        {
            "data_root_resolved": str(data_root),
            "run_dir": str(run_dir),
            "split_file": str(split_path),
            "dataset_dirs": {key: str(value) for key, value in dataset_dirs.items()},
            "classifier_checkpoint_path": str((run_dir / "checkpoints" / "latest.pth").resolve()),
            "integrate_classifier": False,
        }
    )
    return payload


def instantiate_classifier(obs_horizon: int, state_cond_dim: int, action_dim: int, num_train_timesteps: int, add_noise_to_state_cond: bool, add_noise_to_sample: bool):
    from models.xdiffusion.hr_classifier import HumanRobotClassifier, HumanRobotClassifierConfig

    cfg = HumanRobotClassifierConfig(
        num_train_timesteps=num_train_timesteps,
        add_noise_to_sample=add_noise_to_sample,
        add_noise_to_state_cond=add_noise_to_state_cond,
    )
    classifier = HumanRobotClassifier(
        obs_horizon=obs_horizon,
        state_cond_dim=state_cond_dim,
        action_dim=action_dim,
        cfg=cfg,
    )
    return classifier, cfg


def run_training(cfg: ClassifierTrainConfig) -> int:
    import numpy as np
    import torch

    from common_utils.checkpointer import Checkpointer
    from dataset_utils.h5_dataset import H5DatasetConfig, create_h5_dataloader
    from models import train_utils

    data_root = resolve_data_root(cfg.data_root)
    run_dir = resolve_run_dir(cfg, data_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_command_log(run_dir)

    dataset_dirs = build_dataset_dirs(cfg, data_root, run_dir)
    split_path = write_fixed_split(run_dir, cfg)
    audit_report = audit_selected_demos(dataset_dirs, cfg)
    (run_dir / "data_audit.json").write_text(json.dumps(audit_report, indent=2, sort_keys=True) + "\n")
    if not audit_report["no_skipped_or_corrupt_episodes"]:
        raise RuntimeError("Selected classifier episodes are missing or corrupt; see data_audit.json")

    resolved = build_resolved_config(cfg, data_root, run_dir, split_path, dataset_dirs)
    save_yaml(run_dir / "config_resolved.yaml", resolved)
    save_yaml(run_dir / "config.yaml", resolved)
    train_utils.set_seed(cfg.seed)

    dataset_cfg = H5DatasetConfig(
        split_file_name=str(split_path),
        use_ee_data=cfg.use_ee_data,
        add_grasp_info_to_tracks=True,
        load_images=False,
        obs_horizon=cfg.obs_horizon,
        pred_horizon=cfg.pred_horizon,
        action_horizon=cfg.action_horizon,
        balanced_sampling_weights=tuple(cfg.balanced_sampling_weights),
    )
    physical_paths = [str(dataset_dirs["robot"]), str(dataset_dirs["human"])]
    dataloader_kwargs = dict(
        log_dir=str(run_dir),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        physical_data_paths=physical_paths,
        rgb_seg_data_paths=physical_paths,
        dataset_cfg=dataset_cfg,
        human_only_train=False,
        use_image=False,
        camera_views=None,
        return_class_distribution=True,
    )
    train_loader, _, train_class_dist = create_h5_dataloader("train", **dataloader_kwargs)
    val_loader, _, val_class_dist = create_h5_dataloader("val", no_transforms=True, **dataloader_kwargs)
    if train_loader is None or val_loader is None:
        raise RuntimeError("Failed to create classifier dataloaders")

    dataset_summary = {
        "train_dataset_windows": len(train_loader.dataset),
        "val_dataset_windows": len(val_loader.dataset),
        "train_class_distribution": train_class_dist,
        "val_class_distribution": val_class_dist,
        "mode": "classifier",
        "integrate_classifier": False,
        "task": cfg.task,
        "device": cfg.device,
        "batch_size": cfg.batch_size,
        "obs_horizon": cfg.obs_horizon,
        "action_horizon": cfg.action_horizon,
        "pred_horizon": cfg.pred_horizon,
        "epochs": cfg.epochs,
        "steps_per_epoch": cfg.steps_per_epoch,
        "max_val_batches": cfg.max_val_batches,
        "num_train_timesteps": cfg.num_train_timesteps,
    }
    (run_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2, sort_keys=True) + "\n")

    classifier, classifier_cfg = instantiate_classifier(
        obs_horizon=cfg.obs_horizon,
        state_cond_dim=train_loader.dataset.state_cond_dim,
        action_dim=train_loader.dataset.action_dim,
        num_train_timesteps=cfg.num_train_timesteps,
        add_noise_to_state_cond=cfg.add_noise_to_state_cond,
        add_noise_to_sample=cfg.add_noise_to_sample,
    )
    classifier = classifier.to(cfg.device)
    if cfg.torch_compile:
        classifier = torch.compile(classifier)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=cfg.lr, weight_decay=0.0)
    checkpointer = Checkpointer(save_dir=run_dir / "checkpoints")
    metrics_path = run_dir / "metrics.jsonl"
    train_epoch_losses = []
    val_epoch_losses = []
    train_iter = iter(train_loader)

    for epoch_idx in range(cfg.epochs):
        classifier.train()
        step_losses = []
        grad_norms = []
        epoch_start = time.time()
        for _ in range(cfg.steps_per_epoch):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            batch = train_utils.process_namedtuple_batch(batch, cfg.device)
            loss = classifier.loss(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite classifier training loss at epoch {epoch_idx}: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()
            step_losses.append(float(loss.item()))
            grad_norms.append(float(grad_norm.item()))

        train_loss = float(np.mean(step_losses))
        train_epoch_losses.append(train_loss)

        classifier.eval()
        val_losses = []
        val_probabilities = []
        val_labels = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= cfg.max_val_batches:
                    break
                batch = train_utils.process_namedtuple_batch(batch, cfg.device)
                logits = classifier.forward(batch)
                probs = torch.sigmoid(logits.squeeze(-1))
                if not torch.isfinite(probs).all():
                    raise RuntimeError(f"Non-finite classifier probabilities at epoch {epoch_idx}")
                loss = classifier.loss(batch)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite classifier validation loss at epoch {epoch_idx}: {loss.item()}")
                val_losses.append(float(loss.item()))
                val_probabilities.append(probs.cpu())
                val_labels.append(batch.label.detach().cpu())
        if not val_losses:
            raise RuntimeError("Classifier validation loader produced zero batches for the fixed split")
        val_loss = float(np.mean(val_losses))
        val_epoch_losses.append(val_loss)
        checkpointer.save(classifier, optimizer, epoch_idx, val_loss)

        row = {
            "epoch": epoch_idx,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "grad_norm": float(np.mean(grad_norms)),
            "epoch_seconds": time.time() - epoch_start,
            "train_batches": cfg.steps_per_epoch,
            "val_batches": len(val_losses),
            "probability_min": float(torch.cat(val_probabilities).min().item()),
            "probability_max": float(torch.cat(val_probabilities).max().item()),
            "val_human_count": int((torch.cat(val_labels) == 0).sum().item()),
            "val_robot_count": int((torch.cat(val_labels) == 1).sum().item()),
            "loss_finite": True,
        }
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    train_utils.plot_losses(train_epoch_losses, val_epoch_losses, str(run_dir))
    latest_ckpt = run_dir / "checkpoints" / "latest.pth"
    if not latest_ckpt.exists():
        raise RuntimeError("Classifier checkpoint save failed: latest.pth not found")

    reloaded, _ = instantiate_classifier(
        obs_horizon=cfg.obs_horizon,
        state_cond_dim=train_loader.dataset.state_cond_dim,
        action_dim=train_loader.dataset.action_dim,
        num_train_timesteps=cfg.num_train_timesteps,
        add_noise_to_state_cond=cfg.add_noise_to_state_cond,
        add_noise_to_sample=cfg.add_noise_to_sample,
    )
    reloaded = reloaded.to(cfg.device)
    checkpoint = torch.load(latest_ckpt, map_location=cfg.device, weights_only=False)
    reloaded.load_state_dict(checkpoint["model_state_dict"], strict=True)
    reloaded.eval()
    val_batch = next(iter(val_loader))
    val_batch = train_utils.process_namedtuple_batch(val_batch, cfg.device)
    with torch.no_grad():
        logits = reloaded.forward(val_batch)
        probs = torch.sigmoid(logits.squeeze(-1))
    reload_report = {
        "checkpoint_path": str(latest_ckpt.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_val_loss": float(checkpoint["val_loss"]),
        "reload_success": True,
        "inference_success": bool(torch.isfinite(probs).all().item()),
        "probability_min": float(probs.min().item()),
        "probability_max": float(probs.max().item()),
        "num_train_timesteps": classifier_cfg.num_train_timesteps,
        "mode": "classifier",
        "integrate_classifier": False,
    }
    if reload_report["probability_min"] < 0.0 or reload_report["probability_max"] > 1.0:
        raise RuntimeError("Reloaded classifier produced probabilities outside [0,1]")
    (run_dir / "reload_report.json").write_text(json.dumps(reload_report, indent=2, sort_keys=True) + "\n")
    sweep_summary = evaluate_noise_sweep(
        run_dir=run_dir,
        checkpoint=latest_ckpt,
        device=cfg.device,
        batch_size=max(cfg.batch_size, 128),
        timestep_start=0,
        timestep_stop=100,
        timestep_step=1,
        output_prefix="noise_sweep_p_robot",
        seed=cfg.seed,
    )
    (run_dir / "noise_sweep_invocation.json").write_text(json.dumps(sweep_summary, indent=2, sort_keys=True) + "\n")
    return 0


def main() -> int:
    args = parse_args()
    cfg = load_config(Path(args.config_path).resolve())
    return run_training(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
