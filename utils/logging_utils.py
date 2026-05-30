import logging
import sys
import os
from typing import Optional, Dict, Any
from rich.console import Console
from rich.logging import RichHandler
import wandb
import torch


console = Console()


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Rich handler for terminal
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_path=False,
    )
    rich_handler.setLevel(level)
    logger.addHandler(rich_handler)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(file_handler)

    return logger


class MetricLogger:
    """Unified metric logging to W&B + TensorBoard + console."""

    def __init__(
        self,
        project: str,
        name: str,
        config: Dict[str, Any],
        use_wandb: bool = True,
        output_dir: str = "./outputs",
    ):
        self.use_wandb = use_wandb
        self.step = 0
        self.logger = setup_logger(
            name,
            log_file=os.path.join(output_dir, "train.log"),
        )

        if use_wandb:
            wandb.init(
                project=project,
                name=name,
                config=config,
                dir=output_dir,
            )

        # TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(
                log_dir=os.path.join(output_dir, "tensorboard")
            )
        except ImportError:
            self.tb_writer = None

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        step = step or self.step
        if self.use_wandb:
            wandb.log(metrics, step=step)
        if self.tb_writer:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, step)
        # Console log
        msg = " | ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in metrics.items()
        )
        self.logger.info(f"[Step {step}] {msg}")

    def log_image(self, key: str, images, step: Optional[int] = None):
        step = step or self.step
        if self.use_wandb:
            wandb.log({key: [wandb.Image(img) for img in images]}, step=step)

    def log_video(self, key: str, video_path: str, step: Optional[int] = None):
        step = step or self.step
        if self.use_wandb:
            wandb.log({key: wandb.Video(video_path)}, step=step)

    def increment_step(self):
        self.step += 1

    def finish(self):
        if self.use_wandb:
            wandb.finish()
        if self.tb_writer:
            self.tb_writer.close()