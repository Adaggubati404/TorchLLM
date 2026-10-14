"""TorchLLM: GPT-style decoder-only language models in pure PyTorch."""

from torchllm.config import Config, ModelConfig, TrainConfig, load_config
from torchllm.model import GPT

__all__ = ["Config", "ModelConfig", "TrainConfig", "GPT", "load_config"]
__version__ = "0.1.0"
