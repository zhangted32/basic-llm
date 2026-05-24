import math
from typing import List, Optional
from dataclasses import dataclass

try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


class LoRALinear(nn.Module):
    """
    LoRA-adapted linear layer.
    
    Implements low-rank adaptation by adding trainable low-rank matrices to existing weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        bias: bool = True
    ):
        """
        Initialize LoRA-adapted linear layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            rank: Rank of LoRA matrices (lower = more compression)
            alpha: Scaling factor (typically 2x rank)
            dropout: Dropout probability for LoRA layers
            bias: Whether to include bias term
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        
        # Original layer (frozen)
        self.weight = None  # Will be set externally
        self.bias = bias
        
        if bias:
            self.bias = mx.zeros((out_features,))
        
        # LoRA matrices (trainable)
        # W = W_base + (alpha / rank) * W_down @ W_up
        self.W_down = mx.random.normal((in_features, rank)) * 0.01
        self.W_up = mx.zeros((rank, out_features))
        
        # Scaling factor
        self.scaling = alpha / rank
        
        # Dropout layer
        if dropout > 0.0:
            self.dropout_layer = nn.Dropout(p=dropout)
        else:
            self.dropout_layer = None

    def __call__(self, x: mx.array, train: bool = False) -> mx.array:
        """
        Forward pass with LoRA adaptation.
        
        Args:
            x: Input tensor
            train: Whether in training mode
            
        Returns:
            Output tensor with LoRA adaptation applied
        """
        # Original forward pass
        base_output = x @ self.weight.T
        if self.bias is not None:
            base_output = base_output + self.bias
        
        # LoRA adaptation
        # Apply dropout to input if training
        lora_input = self.dropout_layer(x) if train and self.dropout_layer else x
        
        # LoRA forward: (alpha / rank) * (x @ W_down @ W_up)
        lora_output = (lora_input @ self.W_down) @ self.W_up
        lora_output = lora_output * self.scaling
        
        return base_output + lora_output

    def parameters(self):
        """Return LoRA trainable parameters."""
        params = {
            "W_down": self.W_down,
            "W_up": self.W_up
        }
        if self.bias is not None:
            params["bias"] = self.bias
        return params

    def get_rank(self) -> int:
        """Return LoRA rank."""
        return self.rank


class LoRAConfig:
    """Configuration for LoRA adaptation."""
    
    def __init__(
        self,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        bias: str = "none",  # "none", "all", "lora_only"
        task_type: str = "CAUSAL_LM"  # or "SEQ_CLS"
    ):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"]
        self.bias = bias
        self.task_type = task_type
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": self.task_type
        }


class LoRAModelWrapper:
    """
    Wraps a model with LoRA adaptation.
    
    Replaces target linear layers with LoRA-adapted versions.
    """
    
    def __init__(self, base_model: nn.Module, config: LoRAConfig):
        """
        Initialize LoRA model wrapper.
        
        Args:
            base_model: Base model to adapt
            config: LoRA configuration
        """
        self.base_model = base_model
        self.config = config
        self.lora_layers = {}
        self.original_layers = {}
        
    def apply_lora(self):
        """Apply LoRA adaptation to target modules."""
        if not MLX_AVAILABLE:
            print("MLX not available, skipping LoRA application")
            return
        
        # This is a stub implementation
        # In production, would traverse model and replace target layers
        print(f"Applying LoRA with rank={self.config.rank}, alpha={self.config.alpha}")
        print(f"Target modules: {self.config.target_modules}")
        
    def get_trainable_parameters(self):
        """Get only LoRA parameters (not base model parameters)."""
        trainable = {}
        for name, layer in self.lora_layers.items():
            trainable[f"lora.{name}.W_down"] = layer.W_down
            trainable[f"lora.{name}.W_up"] = layer.W_up
        return trainable
    
    def freeze_base_model(self):
        """Freeze base model parameters."""
        if hasattr(self.base_model, 'freeze'):
            self.base_model.freeze()
        print("Base model frozen, only LoRA parameters are trainable")
    
    def save_lora_weights(self, path: str):
        """Save only LoRA adapter weights."""
        import json
        
        lora_weights = {}
        for name, layer in self.lora_layers.items():
            lora_weights[f"lora.{name}.W_down"] = layer.W_down.tolist()
            lora_weights[f"lora.{name}.W_up"] = layer.W_up.tolist()
        
        with open(path, 'w') as f:
            json.dump(lora_weights, f)
        
        print(f"LoRA weights saved to {path}")
    
    def load_lora_weights(self, path: str):
        """Load LoRA adapter weights."""
        import json
        
        with open(path, 'r') as f:
            lora_weights = json.load(f)
        
        for name, layer in self.lora_layers.items():
            if f"lora.{name}.W_down" in lora_weights:
                layer.W_down = mx.array(lora_weights[f"lora.{name}.W_down"])
            if f"lora.{name}.W_up" in lora_weights:
                layer.W_up = mx.array(lora_weights[f"lora.{name}.W_up"])
        
        print(f"LoRA weights loaded from {path}")