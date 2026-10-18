import json
import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages saving and loading of model checkpoints."""

    def __init__(
        self,
        output_dir: str,
        max_checkpoints: int = 5,
        save_optimizer: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_optimizer = save_optimizer
        self._checkpoint_registry: list[str] = []

    def save(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler=None,
        ema_model: torch.nn.Module | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        ckpt_path = self.checkpoint_dir / f"checkpoint-{step}"
        ckpt_path.mkdir(exist_ok=True)

        # Save model state
        model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save(model_state, ckpt_path / "model.pt")

        # Save EMA model
        if ema_model is not None:
            ema_state = (
                ema_model.module.state_dict()
                if hasattr(ema_model, "module")
                else ema_model.state_dict()
            )
            torch.save(ema_state, ckpt_path / "ema_model.pt")

        # Save optimizer
        if self.save_optimizer and optimizer is not None:
            torch.save(optimizer.state_dict(), ckpt_path / "optimizer.pt")

        # Save scheduler
        if scheduler is not None:
            torch.save(scheduler.state_dict(), ckpt_path / "scheduler.pt")

        # Save metadata
        meta = {"step": step, **(metadata or {})}
        with open(ckpt_path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved checkpoint at step {step} → {ckpt_path}")

        # Rotation
        self._checkpoint_registry.append(str(ckpt_path))
        if len(self._checkpoint_registry) > self.max_checkpoints:
            oldest = self._checkpoint_registry.pop(0)
            self._remove_checkpoint(oldest)

        # Update latest pointer
        with open(self.checkpoint_dir / "latest.txt", "w") as f:
            f.write(str(ckpt_path))

    def load(
        self,
        checkpoint_path: str | None = None,
        model: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler=None,
        ema_model: torch.nn.Module | None = None,
        strict: bool = True,
    ) -> dict[str, Any]:
        if checkpoint_path is None:
            latest_file = self.checkpoint_dir / "latest.txt"
            if not latest_file.exists():
                logger.warning("No checkpoint found; starting from scratch.")
                return {}
            checkpoint_path = latest_file.read_text().strip()

        ckpt_path = Path(checkpoint_path)
        logger.info(f"Loading checkpoint from {ckpt_path}")

        if model is not None and (ckpt_path / "model.pt").exists():
            state = torch.load(ckpt_path / "model.pt", map_location="cpu")
            target = model.module if hasattr(model, "module") else model
            target.load_state_dict(state, strict=strict)

        if ema_model is not None and (ckpt_path / "ema_model.pt").exists():
            state = torch.load(ckpt_path / "ema_model.pt", map_location="cpu")
            target = ema_model.module if hasattr(ema_model, "module") else ema_model
            target.load_state_dict(state, strict=strict)

        if optimizer is not None and (ckpt_path / "optimizer.pt").exists():
            optimizer.load_state_dict(torch.load(ckpt_path / "optimizer.pt", map_location="cpu"))

        if scheduler is not None and (ckpt_path / "scheduler.pt").exists():
            scheduler.load_state_dict(torch.load(ckpt_path / "scheduler.pt", map_location="cpu"))

        meta = {}
        if (ckpt_path / "metadata.json").exists():
            with open(ckpt_path / "metadata.json") as f:
                meta = json.load(f)

        return meta

    def _remove_checkpoint(self, path: str):
        import shutil

        try:
            shutil.rmtree(path)
            logger.info(f"Removed old checkpoint: {path}")
        except Exception as e:
            logger.warning(f"Failed to remove checkpoint {path}: {e}")
