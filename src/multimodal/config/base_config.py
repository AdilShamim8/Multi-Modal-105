from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import os


@dataclass
class BaseConfig:
    # ── Experiment ────────────────────────────────────────────────────────────
    experiment_name: str = "multimodal_agent"
    output_dir: str = "./outputs"
    seed: int = 42
    debug: bool = False

    # ── Hardware ─────────────────────────────────────────────────────────────
    device: str = "cuda"
    num_gpus: int = 8
    mixed_precision: str = "bf16"          # "no" | "fp16" | "bf16"
    gradient_checkpointing: bool = True
    compile_model: bool = False            # torch.compile

    # ── Logging ───────────────────────────────────────────────────────────────
    log_every_n_steps: int = 50
    save_every_n_steps: int = 1000
    eval_every_n_steps: int = 5000
    use_wandb: bool = True
    wandb_project: str = "multimodal-agent"
    wandb_entity: Optional[str] = None

    # ── Distributed ───────────────────────────────────────────────────────────
    distributed_backend: str = "nccl"
    find_unused_parameters: bool = False

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class OptimizerConfig:
    optimizer: str = "adamw"              # "adam" | "adamw" | "lion"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Scheduler
    scheduler: str = "cosine"            # "cosine" | "linear" | "constant"
    warmup_steps: int = 500
    num_training_steps: int = 100_000
    min_lr_ratio: float = 0.0