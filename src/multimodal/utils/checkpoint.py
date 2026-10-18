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
        extra_states: dict[str, Any] | None = None,
    ):
        """Persist one checkpoint.

        `extra_states` names additional state dicts to store alongside the
        primary model — e.g. the GAN trainer saves the discriminator and its
        optimizer this way. Each entry lands at `<name>.pt` in the checkpoint
        directory and is restored by `load(extra_states=...)`.
        """
        ckpt_path = self.checkpoint_dir / f"checkpoint-{step}"
        ckpt_path.mkdir(exist_ok=True)

        # Save model state (unwrap DDP when present)
        model_state = self._unwrap(model).state_dict()
        torch.save(model_state, ckpt_path / "model.pt")

        # Save EMA model
        if ema_model is not None:
            torch.save(self._unwrap(ema_model).state_dict(), ckpt_path / "ema_model.pt")

        # Save optimizer
        if self.save_optimizer and optimizer is not None:
            torch.save(optimizer.state_dict(), ckpt_path / "optimizer.pt")

        # Save scheduler
        if scheduler is not None:
            torch.save(scheduler.state_dict(), ckpt_path / "scheduler.pt")

        # Save auxiliary state dicts (discriminator, secondary optimizers, ...)
        for name, state in (extra_states or {}).items():
            torch.save(state, ckpt_path / f"{name}.pt")

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
        extra_states: dict[str, torch.nn.Module | torch.optim.Optimizer] | None = None,
    ) -> dict[str, Any]:
        """Restore a checkpoint saved by `save`.

        `extra_states` maps the saved `<name>.pt` back onto the live object
        (module or optimizer) that should receive it.
        """
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
            self._unwrap(model).load_state_dict(state, strict=strict)

        if ema_model is not None and (ckpt_path / "ema_model.pt").exists():
            state = torch.load(ckpt_path / "ema_model.pt", map_location="cpu")
            self._unwrap(ema_model).load_state_dict(state, strict=strict)

        if optimizer is not None and (ckpt_path / "optimizer.pt").exists():
            optimizer.load_state_dict(torch.load(ckpt_path / "optimizer.pt", map_location="cpu"))

        if scheduler is not None and (ckpt_path / "scheduler.pt").exists():
            scheduler.load_state_dict(torch.load(ckpt_path / "scheduler.pt", map_location="cpu"))

        for name, target in (extra_states or {}).items():
            path = ckpt_path / f"{name}.pt"
            if path.exists():
                target.load_state_dict(torch.load(path, map_location="cpu"))

        meta = {}
        if (ckpt_path / "metadata.json").exists():
            with open(ckpt_path / "metadata.json") as f:
                meta = json.load(f)

        return meta

    @staticmethod
    def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
        """Peel DistributedDataParallel wrappers (checked at the type level,
        not via hasattr: Module attribute lookup is dynamically typed)."""
        inner = getattr(model, "module", None)
        return inner if isinstance(inner, torch.nn.Module) else model

    def _remove_checkpoint(self, path: str):
        import shutil

        try:
            shutil.rmtree(path)
            logger.info(f"Removed old checkpoint: {path}")
        except Exception as e:
            logger.warning(f"Failed to remove checkpoint {path}: {e}")
