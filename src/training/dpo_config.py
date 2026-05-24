from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class DPOConfig:
    """Configuration for DPO training."""
    
    batch_size: int = 1
    gradient_accumulation: int = 4
    learning_rate: float = 1e-5
    training_steps: int = 1000
    checkpoint_interval: int = 500
    eval_interval: int = 100
    max_seq_length: int = 2048
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list = None
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    output_dir: str = "output/dpo_training"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


class DPOConfigManager:
    """Manages DPO training configuration."""
    
    DEFAULT_CONFIGS = {
        "quick_test": DPOConfig(
            batch_size=1,
            gradient_accumulation=4,
            learning_rate=1e-5,
            training_steps=100,
            checkpoint_interval=50,
            eval_interval=25,
            lora_rank=4,
            output_dir="output/dpo_quick_test"
        ),
        "standard": DPOConfig(
            batch_size=2,
            gradient_accumulation=8,
            learning_rate=1e-5,
            training_steps=1000,
            checkpoint_interval=500,
            eval_interval=100,
            lora_rank=8,
            output_dir="output/dpo_standard"
        ),
        "production": DPOConfig(
            batch_size=4,
            gradient_accumulation=16,
            learning_rate=5e-6,
            training_steps=5000,
            checkpoint_interval=1000,
            eval_interval=250,
            lora_rank=16,
            output_dir="output/dpo_production"
        )
    }
    
    @classmethod
    def get_config(cls, config_name: str = "standard") -> DPOConfig:
        """Get a predefined configuration by name."""
        if config_name in cls.DEFAULT_CONFIGS:
            return cls.DEFAULT_CONFIGS[config_name]
        else:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(cls.DEFAULT_CONFIGS.keys())}")
    
    @classmethod
    def create_from_dict(cls, config_dict: Dict) -> DPOConfig:
        """Create a configuration from a dictionary."""
        return DPOConfig(**config_dict)
    
    @classmethod
    def save_config(cls, config: DPOConfig, output_path: Path):
        """Save configuration to a JSON file."""
        import json
        
        config_dict = {
            "batch_size": config.batch_size,
            "gradient_accumulation": config.gradient_accumulation,
            "learning_rate": config.learning_rate,
            "training_steps": config.training_steps,
            "checkpoint_interval": config.checkpoint_interval,
            "eval_interval": config.eval_interval,
            "max_seq_length": config.max_seq_length,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "target_modules": config.target_modules,
            "warmup_steps": config.warmup_steps,
            "weight_decay": config.weight_decay,
            "max_grad_norm": config.max_grad_norm,
            "output_dir": config.output_dir
        }
        
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_config(cls, config_path: Path) -> DPOConfig:
        """Load configuration from a JSON file."""
        import json
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.create_from_dict(config_dict)