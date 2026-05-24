"""
MLX Implementation of Tiny-LLM Model (mirrors PyTorch k_model.py)

This module provides an Apple MLX implementation of the Transformer model,
designed for learning purposes to compare with the PyTorch implementation.

Key Differences between PyTorch and MLX:
1. Import: mlx.core / mlx.nn vs torch
2. Device: Automatic on Apple Silicon vs explicit cuda/cpu
3. Autograd: mlx.core.value_and_grad vs torch.autograd
4. Operations: Mostly identical, some naming differences

Author: Adapted from PyTorch implementation for MLX/Apple Silicon
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class ModelConfig:
    """Configuration class for model hyperparameters - mirrors PyTorch version"""
    model_type = "Tiny-K"

    def __init__(
            self,
            dim: int = 768,
            n_layers: int = 12,
            n_heads: int = 16,
            n_kv_heads: int = 8,
            vocab_size: int = 6144,
            hidden_dim: int = None,
            multiple_of: int = 64,
            norm_eps: float = 1e-5,
            max_seq_len: int = 512,
            dropout: float = 0.0,
            flash_attn: bool = True,
            pad_token_id: int = 0,
            **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        self.pad_token_id = pad_token_id


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    PyTorch: torch.nn.Parameter(torch.ones(dim))
    MLX: Just use mx.array (no Parameter wrapper needed)

    Formula: RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # Weight - in MLX, just store as array (not wrapped in Parameter)
        # PyTorch: nn.Parameter(torch.ones(dim))
        # MLX: mx.ones(dim) directly
        self.weight = mx.ones(dim)

    def __call__(self, x: mx.array) -> mx.array:
        # Compute RMS: sqrt(mean(x^2) + eps)
        # PyTorch: x.pow(2).mean(-1, keepdim=True)
        # MLX: mx.square(x).mean(axis=-1, keepdims=True)
        rms = mx.sqrt(mx.square(x).mean(axis=-1, keepdims=True) + self.eps)
        # Normalize and scale
        return x * (1.0 / rms) * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tuple[mx.array, mx.array]:
    """
    Precompute cosine and sine frequencies for Rotary Position Embedding (RoPE)

    PyTorch vs MLX:
    - torch.arange -> mx.arange
    - torch.outer -> mx.outer (or simple broadcasting)
    - torch.cos/torch.sin -> mx.cos/mx.sin (same API)

    Args:
        dim: Head dimension (should be dim // n_heads)
        end: Sequence length
        theta: Scaling factor (default 10000.0)

    Returns:
        freqs_cos: Cosine frequencies
        freqs_sin: Sine frequencies
    """
    # Generate frequency sequence
    # PyTorch: torch.arange(0, dim, 2)[: (dim // 2)].float()
    # MLX: mx.arange(0, dim, 2)[:dim // 2]
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2)[: dim // 2] / dim))

    # Generate time steps
    # PyTorch: torch.arange(end, device=freqs.device)
    # MLX: mx.arange(end) - device is automatic
    t = mx.arange(end)

    # Compute outer product
    # PyTorch: torch.outer(t, freqs)
    # MLX: mx.triu? No, just broadcast: t[:, None] * freqs[None, :]
    freqs = t[:, None] * freqs[None, :]

    # Compute real and imaginary parts
    freqs_cos = mx.cos(freqs)
    freqs_sin = mx.sin(freqs)

    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis: mx.array, x: mx.array) -> mx.array:
    """
    Reshape frequencies for broadcasting with query/key tensors

    PyTorch vs MLX:
    - x.shape -> x.shape (same)
    - List comprehension for shape -> same approach

    Args:
        freqs_cis: Frequency tensor of shape [seq_len, head_dim/2]
        x: Query/Key tensor

    Returns:
        Reshaped frequency tensor
    """
    ndim = len(x.shape)
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])

    # Create broadcast-compatible shape
    # e.g., [1, seq_len, 1, head_dim/2] for 4D tensor
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.reshape(shape)


def apply_rotary_emb(
    xq: mx.array,
    xk: mx.array,
    freqs_cos: mx.array,
    freqs_sin: mx.array
) -> Tuple[mx.array, mx.array]:
    """
    Apply Rotary Position Embedding to query and key tensors

    PyTorch vs MLX:
    - x.float().reshape(...) -> x.astype(mx.float32).reshape(...) (explicit type cast)
    - .unbind(-1) -> mx.split(..., axis=-1, num_split=2) or indexing
    - torch.stack -> mx.stack
    - .flatten(3) -> x.reshape(..., -1)

    The core operation is complex number rotation:
    Real = x_r * cos(freq) - x_i * sin(freq)
    Imag = x_r * sin(freq) + x_i * cos(freq)
    """
    # Convert to float and reshape to separate real/imag parts
    # PyTorch: xq.float().reshape(xq.shape[:-1] + (-1, 2))
    # MLX: Similar but need explicit type conversion
    xq_r, xq_i = mx.split(xq.astype(mx.float32).reshape(xq.shape[:-1] + (-1, 2)), 2, axis=-1)
    xk_r, xk_i = mx.split(xk.astype(mx.float32).reshape(xk.shape[:-1] + (-1, 2)), 2, axis=-1)

    # Reshape frequencies for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # Apply rotary transformation
    # Real part: x_r * cos - x_i * sin
    # Imag part: x_r * sin + x_i * cos
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # Stack and flatten back to original shape
    # PyTorch: torch.stack([...], dim=-1).flatten(3)
    # MLX: mx.stack([...], axis=-1).reshape(..., -1)
    xq_out = mx.stack([xq_out_r, xq_out_i], axis=-1).reshape(xq.shape)
    xk_out = mx.stack([xk_out_r, xk_out_i], axis=-1).reshape(xk.shape)

    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)


def repeat_kv(x: mx.array, n_rep: int) -> mx.array:
    """
    Repeat key/value tensors to match query heads (for Grouped Query Attention)

    PyTorch vs MLX:
    - x[:, :, :, None, :] expand -> Similar approach with reshape
    - .expand(bs, slen, n_kv_heads, n_rep, head_dim) -> direct shape manipulation
    - .reshape(bs, slen, n_kv_heads * n_rep, head_dim) -> .reshape(..., -1, head_dim)

    Args:
        x: Key/Value tensor of shape [batch, seq_len, n_kv_heads, head_dim]
        n_rep: Number of repetitions

    Returns:
        Repeated tensor of shape [batch, seq_len, n_heads, head_dim]
    """
    if n_rep == 1:
        return x

    bs, slen, n_kv_heads, head_dim = x.shape

    # Reshape to add repetition dimension, then broadcast and flatten
    # PyTorch: x[:, :, :, None, :].expand(...).reshape(...)
    # MLX: Similar approach
    x = x.reshape(bs, slen, n_kv_heads, 1, head_dim)
    x = mx.broadcast_to(x, (bs, slen, n_kv_heads, n_rep, head_dim))
    x = x.reshape(bs, slen, n_kv_heads * n_rep, head_dim)

    return x


class Attention(nn.Module):
    """
    Multi-Head Attention with RoPE and Grouped Query Attention (GQA)

    PyTorch vs MLX:
    - nn.Linear -> same API
    - nn.Dropout -> same API (but dropout is only applied during training)
    - self.register_buffer -> No buffer concept in MLX, just store as attributes
    - Attention mask handling is similar
    """
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0

        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.dropout = args.dropout

        # Weight matrices for Q, K, V, and output
        # PyTorch: nn.Linear(dim, num_heads * head_dim)
        # MLX: Same API
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # Dropout (only used during training)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

    def __call__(
        self,
        x: mx.array,
        freqs_cos: mx.array,
        freqs_sin: mx.array,
        attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        """
        Forward pass of attention mechanism

        Args:
            x: Input tensor [batch, seq_len, dim]
            freqs_cos: Precomputed cosine frequencies
            freqs_sin: Precomputed sine frequencies
            attention_mask: Optional attention mask

        Returns:
            Output tensor [batch, seq_len, dim]
        """
        bsz, seqlen, _ = x.shape

        # Compute Q, K, V projections
        xq = self.wq(x).reshape(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = self.wk(x).reshape(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = self.wv(x).reshape(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # Apply rotary embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # Repeat K, V for grouped query attention
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # Transpose for attention computation: [batch, heads, seq, dim]
        xq = xq.transpose(0, 2, 1, 3)  # PyTorch: xq.transpose(1, 2)
        xk = xk.transpose(0, 2, 1, 3)
        xv = xv.transpose(0, 2, 1, 3)

        # Compute attention scores
        # PyTorch: torch.matmul(xq, xk.transpose(2, 3)) / sqrt(head_dim)
        # MLX: mx.matmul with transpose or use @ operator
        scores = (xq @ xk.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)

        # Apply causal mask (triangular)
        mask = mx.tril(mx.ones((seqlen, seqlen)))
        mask = mx.where(mask, 0.0, float("-inf"))
        scores = scores + mask

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask to match scores shape
            mask_expanded = attention_mask[:, None, :, None]  # Broadcast
            mask_expanded = mx.where(mask_expanded, 0.0, float("-inf"))
            scores = scores + mask_expanded

        # Softmax and dropout
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(xq.dtype)
        scores = self.attn_dropout(scores)

        # Compute output
        output = scores @ xv

        # Reshape back: [batch, seq, heads, dim] -> [batch, seq, dim]
        output = output.transpose(0, 2, 1, 3).reshape(bsz, seqlen, -1)

        # Final projection and residual dropout
        output = self.wo(output)
        output = self.resid_dropout(output)

        return output


class MLP(nn.Module):
    """
    Feed-Forward Network with SiLU activation (Swish)

    PyTorch vs MLX:
    - nn.Linear -> same API
    - F.silu -> mx.nn.silu or use torch.nn.functional.silu equivalent
    - Element-wise multiplication is the same (* or mx.multiply)

    Architecture: w1(x) * silu(w3(x)) @ w2
    This is the SwiGLU activation variant used in LLaMA
    """
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        # SiLU/Swish: x * sigmoid(x)
        # PyTorch: F.silu(x)
        # MLX: mx.nn.silu(x) or define manually
        return self.dropout(self.w2(nn.gelu(self.w1(x)) * self.w3(x)))


class DecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer

    Combines attention and feed-forward sublayers with RMSNorm and residual connections

    PyTorch vs MLX:
    - Submodules stored as attributes (same pattern)
    - Forward pass is nearly identical
    """
    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = Attention(args)
        self.feed_forward = MLP(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )

        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def __call__(
        self,
        x: mx.array,
        freqs_cos: mx.array,
        freqs_sin: mx.array,
        attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        # Attention with pre-norm and residual
        h = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin, attention_mask)
        # FFN with pre-norm and residual
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    """
    Full Transformer/LLaMA model

    PyTorch vs MLX:
    - nn.Embedding -> same API
    - ModuleList -> nn.ModuleList (same)
    - register_buffer not needed - just store as regular attributes
    - Model loading: torch.load vs mx.load
    - Weight initialization: similar approach

    Key features:
    - Weight tying between embedding and output layer
    - Precomputed RoPE frequencies
    - Causal language modeling head
    """
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        # Token embeddings
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.dropout = nn.Dropout(args.dropout)

        # Decoder layers
        # PyTorch: nn.ModuleList() - MLX uses regular list
        # In MLX, modules are stored differently
        self.layers = []
        for layer_id in range(args.n_layers):
            self.layers.append(DecoderLayer(layer_id, args))

        # Output norm and projection
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Tie weights between embedding and output
        self.output.weight = self.tok_embeddings.weight

        # Precompute RoPE frequencies
        # PyTorch: register_buffer for non-trainable tensors
        # MLX: Just store as instance attributes
        freqs_cos, freqs_sin = precompute_freqs_cis(
            self.args.dim // self.args.n_heads,
            self.args.max_seq_len
        )
        self.freqs_cos = freqs_cos
        self.freqs_sin = freqs_sin

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights - mirrors PyTorch approach"""
        # Linear layers: normal init with std=0.02
        # Embedding layers: normal init with std=0.02
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # PyTorch: torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                # MLX: Manual initialization needed
                if hasattr(module, 'weight') and module.weight is not None:
                    # Initialize with normal distribution
                    std = 0.02
                    module.weight = mx.random.normal(
                        module.weight.shape,
                        dtype=module.weight.dtype
                    ) * std

            elif isinstance(module, nn.Embedding):
                if hasattr(module, 'weight') and module.weight is not None:
                    std = 0.02
                    module.weight = mx.random.normal(
                        module.weight.shape,
                        dtype=module.weight.dtype
                    ) * std

    def __call__(
        self,
        tokens: mx.array,
        targets: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None
    ) -> dict:
        """
        Forward pass

        Args:
            tokens: Input token IDs [batch, seq_len]
            targets: Target token IDs for training [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Dictionary with logits and optional loss
        """
        _bsz, seqlen = tokens.shape

        # Embedding lookup with dropout
        h = self.dropout(self.tok_embeddings(tokens))

        # Get precomputed frequencies for this sequence length
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # Pass through decoder layers
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin, attention_mask)

        # Final norm
        h = self.norm(h)

        # Output projection to vocabulary
        logits = self.output(h)

        result = {"logits": logits}

        if targets is not None:
            # Compute cross-entropy loss
            ignore_index = self.args.pad_token_id if self.args.pad_token_id is not None else 0
            if mx.any(targets == -100):
                ignore_index = -100

            # PyTorch: F.cross_entropy(logits.view(-1, ...), targets.view(-1), ...)
            # MLX: mx.losses.cross_entropy expects different format
            # Reshape for loss computation
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = targets.flatten()

            # Filter out masked positions
            valid_mask = targets_flat != ignore_index
            if mx.any(valid_mask).item():
                logits_valid = logits_flat[valid_mask]
                targets_valid = targets_flat[valid_mask]
                loss = mx.mean(
                    nn.losses.cross_entropy(
                        logits_valid[None, ...],
                        targets_valid[None, ...]
                    )
                )
                result["loss"] = loss
            else:
                result["loss"] = mx.array(0.0)

        return result

    def generate(
        self,
        tokens: mx.array,
        eos_token_id: int,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> mx.array:
        """
        Generate text autoregressively

        Args:
            tokens: Input tokens [batch, seq_len]
            eos_token_id: End-of-sequence token ID
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter

        Returns:
            Generated token IDs
        """
        self.eval()

        generated = tokens
        for _ in range(max_new_tokens):
            # Forward pass
            result = self.__call__(generated)
            logits = result["logits"]

            # Get last position logits
            next_token_logits = logits[:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Top-k filtering
            if top_k > 0:
                # Get top-k indices
                top_k_vals = mx.topk(next_token_logits, min(top_k, next_token_logits.shape[-1]))
                indices_to_remove = next_token_logits < top_k_vals[-1]
                next_token_logits = mx.where(indices_to_remove, -float("inf"), next_token_logits)

            # Sample from distribution
            probs = mx.softmax(next_token_logits, axis=-1)
            next_token = mx.random.categorical(probs)

            # Append to sequence
            generated = mx.concatenate([generated, next_token[:, None]], axis=1)

            # Check for EOS
            if mx.any(next_token == eos_token_id).item():
                break

        return generated


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model - MLX version"""
    total = 0
    for key, value in model.__dict__.items():
        if isinstance(value, mx.array):
            total += value.size
    return total


def load_model_weights(
    model: Transformer,
    weights_path: str,
    device: str = "apple"
) -> Transformer:
    """
    Load model weights from PyTorch checkpoint

    PyTorch: torch.load(path, map_location=...)
    MLX: mx.load(path) or need conversion

    This function handles the conversion from PyTorch to MLX format.

    Args:
        model: MLX Transformer model
        weights_path: Path to PyTorch .pth file
        device: Target device (only 'apple' for MLX)

    Returns:
        Model with loaded weights
    """
    import torch

    # Load PyTorch weights
    pt_weights = torch.load(weights_path, map_location="cpu")

    # Convert to MLX format
    mlx_weights = {}
    for key, value in pt_weights.items():
        # Remove 'module.' prefix if present (from DataParallel)
        if key.startswith("module."):
            key = key[7:]

        # Convert PyTorch tensor to MLX array
        # PyTorch: torch.tensor -> MLX: mx.array
        mlx_weights[key] = mx.array(value.numpy())

    # Load into model using MLX's tree_map for updating
    def update_leaf(path, val):
        key = ".".join(path)
        if key in mlx_weights:
            return mlx_weights[key]
        return val

    model = mx.tree_map(update_leaf, model)

    return model


if __name__ == "__main__":
    print("=" * 60)
    print("MLX Transformer Model - Learning Comparison with PyTorch")
    print("=" * 60)

    # Create model config (same as PyTorch version)
    config = ModelConfig(
        dim=512,       # Smaller for demo
        n_layers=8,   # Fewer layers for demo
        n_heads=8,
        n_kv_heads=4,
        vocab_size=6144,
        max_seq_len=256,
        dropout=0.0,
    )

    print("\n1. Creating MLX Transformer model...")
    model = Transformer(config)
    num_params = count_parameters(model)
    print(f"   Model parameters: {num_params / 1e6:.2f} M")

    print("\n2. Model structure (compare with PyTorch):")
    print(f"   - Vocab size: {model.vocab_size}")
    print(f"   - Hidden dim: {config.dim}")
    print(f"   - Layers: {model.n_layers}")
    print(f"   - Heads: {config.n_heads}")
    print(f"   - Head dim: {config.dim // config.n_heads}")

    print("\n3. Testing forward pass...")
    # Create sample input
    batch_size = 2
    seq_len = 32
    dummy_tokens = mx.randint(0, config.vocab_size, (batch_size, seq_len))

    print(f"   Input shape: {dummy_tokens.shape}")
    print(f"   Device: Apple Silicon (MLX)")

    # Forward pass
    result = model(dummy_tokens)
    print(f"   Output logits shape: {result['logits'].shape}")

    print("\n4. Testing generation...")
    # Test generation with a prompt
    prompt_tokens = mx.array([[1, 2, 3, 4, 5]])  # Simple prompt
    generated = model.generate(
        prompt_tokens,
        eos_token_id=0,
        max_new_tokens=20,
        temperature=0.8,
        top_k=40
    )
    print(f"   Generated shape: {generated.shape}")

    print("\n5. Comparing PyTorch vs MLX (for learning):")
    print("   | Aspect          | PyTorch           | MLX               |")
    print("   |-----------------|-------------------|-------------------|")
    print("   | Import          | import torch      | import mlx.core   |")
    print("   | Neural Networks | torch.nn          | mlx.nn           |")
    print("   | Device          | cuda, cpu, mps    | automatic         |")
    print("   | Forward pass    | .forward()        | __call__()       |")
    print("   | Gradients       | autograd           | value_and_grad    |")
    print("   | Loading         | torch.load()      | mx.load()        |")
    print("   | Saving          | torch.save()      | mx.save()        |")
    print("   | Array creation  | torch.tensor()    | mx.array()       |")
    print("   | Attention       | scaled_dot_product| manual or optim  |")

    print("\n" + "=" * 60)
    print("MLX model verification completed successfully!")
    print("=" * 60)
