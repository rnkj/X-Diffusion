from __future__ import annotations

import argparse
import hashlib
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
DEFAULT_DATA_ROOT = REPO_ROOT.parent / "X-Diffusion-Data"
PHYSICAL_REQUIRED_KEYS = ["ee_pos", "ee_euler", "gripper_open", "3d_tracks"]
FILTERED_TASKS = {"mug_on_rack", "pan_on_plate", "push_plate"}


@dataclass
class PolicyTrainConfig:
    task: str = "mug_on_rack"
    mode: str = "robot_only"
    run_name: str = "m4_robot_only"
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
    balanced_sampling_weights: Optional[List[float]] = None
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
    integrate_classifier: bool = False
    threshold: float = 0.5
    num_train_timesteps: int = 101
    classifier_checkpoint_path: Optional[str] = None
    classifier_run_dir: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Policy training")
    parser.add_argument("--config_path", required=True, help="Path to YAML config file")
    return parser.parse_args()


def load_config(config_path: Path) -> PolicyTrainConfig:
    payload = yaml.safe_load(config_path.read_text()) or {}
    return PolicyTrainConfig(**payload)


def resolve_data_root(raw_root: Optional[str]) -> Path:
    return Path(raw_root).expanduser().resolve() if raw_root else DEFAULT_DATA_ROOT.resolve()


def resolve_run_dir(cfg: PolicyTrainConfig, data_root: Path) -> Path:
    return (data_root / "_private" / "runs" / cfg.run_name).resolve()


def validate_mode_task(cfg: PolicyTrainConfig) -> None:
    if cfg.mode not in {"robot_only", "naive", "xdiffusion", "filtered"}:
        raise ValueError(f"Unsupported mode: {cfg.mode}")
    if cfg.mode == "filtered" and cfg.task not in FILTERED_TASKS:
        allowed = ", ".join(sorted(FILTERED_TASKS))
        raise ValueError(f"Filtered mode is only supported for: {allowed}. Got {cfg.task}")


def get_human_source_embodiment(cfg: PolicyTrainConfig) -> Optional[str]:
    if cfg.mode == "filtered":
        return "human_filtered"
    if cfg.mode in {"naive", "xdiffusion"}:
        return "human"
    return None


def resolve_source_dirs(cfg: PolicyTrainConfig, data_root: Path) -> Dict[str, Path]:
    sources = {
        "robot": (data_root / "retargeted" / cfg.task / "robot").resolve(),
    }
    human_source = get_human_source_embodiment(cfg)
    if human_source is not None:
        sources[human_source] = (data_root / "retargeted" / cfg.task / human_source).resolve()
    return sources


def ensure_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        if not link_path.is_symlink() or link_path.resolve() != target_path.resolve():
            raise RuntimeError(f"Wrapper path already exists with unexpected target: {link_path}")
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target_path, link_path, target_is_directory=target_path.is_dir())


def build_dataset_dirs(cfg: PolicyTrainConfig, data_root: Path, run_dir: Path) -> Dict[str, Path]:
    del run_dir
    return resolve_source_dirs(cfg, data_root)


def write_fixed_split(run_dir: Path, cfg: PolicyTrainConfig) -> Path:
    include_human = get_human_source_embodiment(cfg) is not None
    split = {
        "train": {"robot": list(cfg.train_demo_ids)},
        "val": {"robot": list(cfg.val_demo_ids)},
    }
    if include_human:
        human_key = get_human_source_embodiment(cfg)
        split["train"][human_key] = list(cfg.train_demo_ids)
        split["val"][human_key] = list(cfg.val_demo_ids)
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


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def audit_selected_demos(wrapper_dirs: Dict[str, Path], source_dirs: Dict[str, Path], cfg: PolicyTrainConfig) -> Dict[str, Any]:
    requested = cfg.train_demo_ids + cfg.val_demo_ids
    human_source = get_human_source_embodiment(cfg)
    report: Dict[str, Any] = {
        "task": cfg.task,
        "mode": cfg.mode,
        "train_demo_ids": list(cfg.train_demo_ids),
        "val_demo_ids": list(cfg.val_demo_ids),
        "datasets": {},
        "missing_files": [],
        "corrupt_files": [],
        "selected_source_mismatches": [],
        "modalities_used": ["physical"],
        "human_source_embodiment": human_source,
        "filtered_human_only": cfg.mode == "filtered",
        "all_selected_human_from_human_filtered": cfg.mode != "filtered",
        "no_skipped_or_corrupt_episodes": True,
    }
    for dataset_type, wrapper_dir in wrapper_dirs.items():
        source_dir = source_dirs[dataset_type]
        entries = []
        for demo_id in requested:
            info = audit_demo_file(wrapper_dir / f"{demo_id}.h5", cfg.pred_horizon)
            entries.append(info)
            if not info["exists"]:
                report["missing_files"].append(info["path"])
            if info["missing_keys"] or info["error"]:
                report["corrupt_files"].append(info)
            if cfg.mode == "filtered" and dataset_type == "human_filtered" and info["exists"]:
                real_path = Path(info["path"])
                if not _is_relative_to(real_path, source_dir) or source_dir.name != "human_filtered":
                    report["selected_source_mismatches"].append(
                        {
                            "demo_id": demo_id,
                            "expected_source_dir": str(source_dir),
                            "resolved_path": info["path"],
                        }
                    )
        report["datasets"][dataset_type] = {
            "wrapper_dir": str(wrapper_dir),
            "real_dir": str(wrapper_dir.resolve()),
            "source_dir": str(source_dir),
            "source_embodiment": source_dir.name,
            "episodes": entries,
        }
    if cfg.mode == "filtered":
        report["all_selected_human_from_human_filtered"] = not report["selected_source_mismatches"]
    report["no_skipped_or_corrupt_episodes"] = (
        not report["missing_files"]
        and not report["corrupt_files"]
        and not report["selected_source_mismatches"]
    )
    return report


def write_command_log(run_dir: Path) -> None:
    (run_dir / "command.txt").write_text(" ".join([sys.executable, *sys.argv]) + "\n")


def save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def build_resolved_config(
    cfg: PolicyTrainConfig,
    data_root: Path,
    run_dir: Path,
    split_path: Path,
    dataset_dirs: Dict[str, Path],
) -> Dict[str, Any]:
    payload = asdict(cfg)
    payload.update(
        {
            "data_root_resolved": str(data_root),
            "run_dir": str(run_dir),
            "split_file": str(split_path),
            "dataset_dirs": {key: str(value) for key, value in dataset_dirs.items()},
            "human_source_embodiment": get_human_source_embodiment(cfg),
            "integrate_classifier": False if cfg.mode == "xdiffusion" else cfg.integrate_classifier,
            "xdiffusion_masking_enabled": bool(cfg.mode == "xdiffusion"),
        }
    )
    return payload


def instantiate_policy(action_dim: int, state_cond_dim: int, num_points: int, cfg: PolicyTrainConfig):
    from models.xdiffusion.xDP import DiffusionPolicy, DiffusionPolicyConfig

    policy_cfg = DiffusionPolicyConfig()
    policy_cfg.ddim.num_train_timesteps = cfg.num_train_timesteps
    policy = DiffusionPolicy(
        action_dim=action_dim,
        state_cond_dim=state_cond_dim,
        num_points=num_points,
        obs_horizon=cfg.obs_horizon,
        pred_horizon=cfg.pred_horizon,
        action_horizon=cfg.action_horizon,
        cfg=policy_cfg,
        use_image=False,
    )
    return policy


def resolve_classifier_checkpoint(cfg: PolicyTrainConfig) -> Path:
    if not cfg.classifier_checkpoint_path:
        raise RuntimeError("Classifier checkpoint provenance is missing from config")
    ckpt = Path(cfg.classifier_checkpoint_path).resolve()
    if not ckpt.exists():
        raise RuntimeError(f"Classifier checkpoint path does not exist: {ckpt}")
    if cfg.classifier_run_dir:
        run_dir = Path(cfg.classifier_run_dir).resolve()
        reload_report = run_dir / "reload_report.json"
        if not reload_report.exists():
            raise RuntimeError(f"Classifier checkpoint provenance report missing: {reload_report}")
    return ckpt


def load_frozen_classifier(cfg: PolicyTrainConfig, state_cond_dim: int, action_dim: int, device: str):
    import torch
    from models.xdiffusion.hr_classifier import HumanRobotClassifier, HumanRobotClassifierConfig

    checkpoint_path = resolve_classifier_checkpoint(cfg)
    classifier_cfg = HumanRobotClassifierConfig(num_train_timesteps=cfg.num_train_timesteps)
    classifier = HumanRobotClassifier(
        obs_horizon=cfg.obs_horizon,
        state_cond_dim=state_cond_dim,
        action_dim=action_dim,
        cfg=classifier_cfg,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classifier.load_state_dict(checkpoint["model_state_dict"], strict=True)
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False
    provenance = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_val_loss": float(checkpoint["val_loss"]),
        "frozen": all(not p.requires_grad for p in classifier.parameters()),
        "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    }
    return classifier, provenance


def compute_xdiffusion_loss_masks(batch, probabilities, threshold: float):
    import torch

    human_mask = batch.label == 0
    robot_mask = batch.label == 1
    include_mask = torch.zeros_like(batch.label, dtype=torch.float32)
    include_mask[robot_mask] = 1.0
    predicted_robot = probabilities > threshold
    include_mask[human_mask & predicted_robot] = 1.0
    stats = {
        "human_total": int(human_mask.sum().item()),
        "human_included": int((human_mask & predicted_robot).sum().item()),
        "robot_total": int(robot_mask.sum().item()),
        "robot_included": int(include_mask[robot_mask].sum().item()),
    }
    return include_mask, stats


def run_training(cfg: PolicyTrainConfig) -> int:
    import numpy as np
    import torch

    from common_utils.checkpointer import Checkpointer
    from dataset_utils.h5_dataset import H5DatasetConfig, create_h5_dataloader
    from models import train_utils

    validate_mode_task(cfg)

    data_root = resolve_data_root(cfg.data_root)
    run_dir = resolve_run_dir(cfg, data_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_command_log(run_dir)

    dataset_dirs = build_dataset_dirs(cfg, data_root, run_dir)
    split_path = write_fixed_split(run_dir, cfg)
    audit_report = audit_selected_demos(dataset_dirs, dataset_dirs, cfg)
    (run_dir / "data_audit.json").write_text(json.dumps(audit_report, indent=2, sort_keys=True) + "\n")
    if not audit_report["no_skipped_or_corrupt_episodes"]:
        raise RuntimeError("Selected episodes are missing, corrupt, or use the wrong human source; see data_audit.json")

    save_yaml(
        run_dir / "config_resolved.yaml",
        build_resolved_config(cfg, data_root, run_dir, split_path, dataset_dirs),
    )
    train_utils.set_seed(cfg.seed)

    dataset_cfg = H5DatasetConfig(
        split_file_name=str(split_path),
        use_ee_data=cfg.use_ee_data,
        add_grasp_info_to_tracks=True,
        load_images=False,
        obs_horizon=cfg.obs_horizon,
        pred_horizon=cfg.pred_horizon,
        action_horizon=cfg.action_horizon,
        balanced_sampling_weights=tuple(cfg.balanced_sampling_weights) if cfg.balanced_sampling_weights else None,
    )
    physical_paths = [str(dataset_dirs["robot"])]
    human_source = get_human_source_embodiment(cfg)
    if human_source is not None:
        physical_paths.append(str(dataset_dirs[human_source]))

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
        raise RuntimeError("Failed to create dataloaders")
    if cfg.mode == "filtered" and train_class_dist.get("human", 0) <= 0:
        raise RuntimeError("Filtered mode did not load any feasible-only human samples")

    dataset_summary = {
        "train_dataset_windows": len(train_loader.dataset),
        "val_dataset_windows": len(val_loader.dataset),
        "train_class_distribution": train_class_dist,
        "val_class_distribution": val_class_dist,
        "mode": cfg.mode,
        "integrate_classifier": False if cfg.mode == "xdiffusion" else cfg.integrate_classifier,
        "xdiffusion_masking_enabled": bool(cfg.mode == "xdiffusion"),
        "task": cfg.task,
        "device": cfg.device,
        "batch_size": cfg.batch_size,
        "obs_horizon": cfg.obs_horizon,
        "action_horizon": cfg.action_horizon,
        "pred_horizon": cfg.pred_horizon,
        "epochs": cfg.epochs,
        "steps_per_epoch": cfg.steps_per_epoch,
        "max_val_batches": cfg.max_val_batches,
        "classifier_checkpoint_path": str(cfg.classifier_checkpoint_path) if cfg.classifier_checkpoint_path else None,
        "classifier_run_dir": str(cfg.classifier_run_dir) if cfg.classifier_run_dir else None,
        "human_source_embodiment": get_human_source_embodiment(cfg),
        "source_dirs": {key: str(value) for key, value in dataset_dirs.items()},
        "filtered_human_only": cfg.mode == "filtered",
        "selected_human_from_human_filtered": audit_report["all_selected_human_from_human_filtered"],
    }
    (run_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2, sort_keys=True) + "\n")

    policy = instantiate_policy(
        action_dim=train_loader.dataset.action_dim,
        state_cond_dim=train_loader.dataset.state_cond_dim,
        num_points=dataset_cfg.num_points,
        cfg=cfg,
    )
    policy = policy.to(cfg.device)
    if cfg.torch_compile:
        policy = torch.compile(policy)

    classifier = None
    classifier_provenance = None
    if cfg.mode == "xdiffusion":
        classifier, classifier_provenance = load_frozen_classifier(
            cfg,
            state_cond_dim=train_loader.dataset.state_cond_dim,
            action_dim=train_loader.dataset.action_dim,
            device=cfg.device,
        )
        (run_dir / "classifier_provenance.json").write_text(json.dumps(classifier_provenance, indent=2, sort_keys=True) + "\n")

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr, weight_decay=0.0)
    checkpointer = Checkpointer(save_dir=run_dir / "checkpoints")
    metrics_path = run_dir / "metrics.jsonl"
    train_epoch_losses: List[float] = []
    val_epoch_losses: List[float] = []
    train_loader_iter = iter(train_loader)
    masking_rows: List[Dict[str, Any]] = []

    for epoch_idx in range(cfg.epochs):
        policy.train()
        step_losses: List[float] = []
        grad_norms: List[float] = []
        epoch_start = time.time()
        epoch_masking = {"human_total": 0, "human_included": 0, "robot_total": 0, "robot_included": 0}
        timestep_min = None
        timestep_max = None

        for _ in range(cfg.steps_per_epoch):
            try:
                batch = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(train_loader)
                batch = next(train_loader_iter)
            batch = train_utils.process_namedtuple_batch(batch, cfg.device)

            if cfg.mode == "xdiffusion":
                timesteps = torch.randint(low=0, high=cfg.num_train_timesteps, size=(batch.label.shape[0],), device=cfg.device).long()
                with torch.no_grad():
                    logits = classifier.unified_forward(batch, timesteps=timesteps)
                    probabilities = torch.sigmoid(logits.squeeze(-1))
                if not torch.isfinite(probabilities).all():
                    raise RuntimeError("Non-finite classifier probabilities during X-Diffusion training")
                loss_masks, mask_stats = compute_xdiffusion_loss_masks(batch, probabilities, cfg.threshold)
                if mask_stats["robot_included"] != mask_stats["robot_total"]:
                    raise RuntimeError("Robot actions were not all included in policy loss")
                loss, _ = policy.unified_loss(batch, timesteps=timesteps, loss_masks=loss_masks)
                for key in epoch_masking:
                    epoch_masking[key] += mask_stats[key]
                current_min = int(timesteps.min().item())
                current_max = int(timesteps.max().item())
                timestep_min = current_min if timestep_min is None else min(timestep_min, current_min)
                timestep_max = current_max if timestep_max is None else max(timestep_max, current_max)
            else:
                loss, _ = policy.loss(batch, action_mse=True)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch_idx}: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()
            step_losses.append(float(loss.item()))
            grad_norms.append(float(grad_norm.item()))

        train_loss = float(np.mean(step_losses))
        train_epoch_losses.append(train_loss)

        policy.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= cfg.max_val_batches:
                    break
                batch = train_utils.process_namedtuple_batch(batch, cfg.device)
                if cfg.mode == "xdiffusion":
                    timesteps = torch.randint(low=0, high=cfg.num_train_timesteps, size=(batch.label.shape[0],), device=cfg.device).long()
                    logits = classifier.unified_forward(batch, timesteps=timesteps)
                    probabilities = torch.sigmoid(logits.squeeze(-1))
                    if not torch.isfinite(probabilities).all():
                        raise RuntimeError("Non-finite classifier probabilities during X-Diffusion validation")
                    loss_masks, mask_stats = compute_xdiffusion_loss_masks(batch, probabilities, cfg.threshold)
                    if mask_stats["robot_included"] != mask_stats["robot_total"]:
                        raise RuntimeError("Robot actions were not all included in X-Diffusion validation loss")
                    loss, _ = policy.unified_loss(batch, timesteps=timesteps, loss_masks=loss_masks)
                else:
                    loss, _ = policy.loss(batch, action_mse=True)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite validation loss at epoch {epoch_idx}: {loss.item()}")
                val_losses.append(float(loss.item()))
        if not val_losses:
            raise RuntimeError("Validation loader produced zero batches for the fixed split")
        val_loss = float(np.mean(val_losses))
        val_epoch_losses.append(val_loss)
        checkpointer.save(policy, optimizer, epoch_idx, val_loss)

        row = {
            "epoch": epoch_idx,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "grad_norm": float(np.mean(grad_norms)),
            "epoch_seconds": time.time() - epoch_start,
            "train_batches": cfg.steps_per_epoch,
            "val_batches": len(val_losses),
            "loss_finite": True,
            "mode": cfg.mode,
        }
        if cfg.mode == "xdiffusion":
            row.update(
                {
                    "human_total": epoch_masking["human_total"],
                    "human_included": epoch_masking["human_included"],
                    "robot_total": epoch_masking["robot_total"],
                    "robot_included": epoch_masking["robot_included"],
                    "timestep_min": timestep_min,
                    "timestep_max": timestep_max,
                    "classifier_frozen": bool(classifier_provenance["frozen"]),
                    "classifier_checkpoint_path": classifier_provenance["checkpoint_path"],
                    "human_classified_at_sampled_timestep": True,
                    "robot_always_included": epoch_masking["robot_total"] == epoch_masking["robot_included"],
                }
            )
            masking_rows.append(dict(row))
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    train_utils.plot_losses(train_epoch_losses, val_epoch_losses, str(run_dir))
    latest_ckpt = run_dir / "checkpoints" / "latest.pth"
    if not latest_ckpt.exists():
        raise RuntimeError("Checkpoint save failed: latest.pth not found")

    reloaded_policy = instantiate_policy(
        action_dim=train_loader.dataset.action_dim,
        state_cond_dim=train_loader.dataset.state_cond_dim,
        num_points=dataset_cfg.num_points,
        cfg=cfg,
    )
    reloaded_policy = reloaded_policy.to(cfg.device)
    checkpoint = torch.load(latest_ckpt, map_location=cfg.device, weights_only=False)
    reloaded_policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    reloaded_policy.eval()

    val_batch = next(iter(val_loader))
    val_batch = train_utils.process_namedtuple_batch(val_batch, cfg.device)
    with torch.no_grad():
        predicted = reloaded_policy.act(val_batch.state_cond, first_action_abs=val_batch.action[:, :1, :], cpu=True)
    reload_report = {
        "checkpoint_path": str(latest_ckpt.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_val_loss": float(checkpoint["val_loss"]),
        "reload_success": True,
        "inference_success": bool(torch.isfinite(predicted).all().item()),
        "predicted_shape": list(predicted.shape),
        "mode": cfg.mode,
        "integrate_classifier": False if cfg.mode == "xdiffusion" else cfg.integrate_classifier,
        "xdiffusion_masking_enabled": bool(cfg.mode == "xdiffusion"),
        "human_source_embodiment": get_human_source_embodiment(cfg),
        "filtered_human_only": cfg.mode == "filtered",
    }
    (run_dir / "reload_report.json").write_text(json.dumps(reload_report, indent=2, sort_keys=True) + "\n")

    if cfg.mode == "xdiffusion":
        totals = {
            "human_total": sum(row["human_total"] for row in masking_rows),
            "human_included": sum(row["human_included"] for row in masking_rows),
            "robot_total": sum(row["robot_total"] for row in masking_rows),
            "robot_included": sum(row["robot_included"] for row in masking_rows),
            "classifier_checkpoint_path": classifier_provenance["checkpoint_path"],
            "classifier_frozen": bool(classifier_provenance["frozen"]),
            "human_classified_at_sampled_timestep": True,
            "only_human_predicted_robot_included": True,
            "robot_always_included": sum(row["robot_total"] for row in masking_rows) == sum(row["robot_included"] for row in masking_rows),
            "threshold": cfg.threshold,
            "timestep_range": [0, cfg.num_train_timesteps - 1],
        }
        (run_dir / "masking_report.json").write_text(json.dumps(totals, indent=2, sort_keys=True) + "\n")
    return 0


def main() -> int:
    args = parse_args()
    cfg = load_config(Path(args.config_path).resolve())
    return run_training(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
