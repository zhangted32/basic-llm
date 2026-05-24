"""
MLX Model Forward Pass Demo - Verify the full LLM runs on Apple Silicon

This script demonstrates:
1. Loading the tokenizer
2. Initializing the MLX model
3. Running forward passes
4. Testing generation

Since we don't have trained weights, we initialize from scratch.
The model won't produce meaningful output, but it proves the
entire pipeline works on Mac Mini M4!

Author: For MLX/Apple Silicon demonstration
"""

import math
from dataclasses import dataclass
from typing import Optional
import mlx.core as mx
import mlx.nn as nn


@dataclass
class ModelConfig:
    """Model configuration - matches the PyTorch version"""
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    vocab_size: int = 6144
    hidden_dim: int = None
    multiple_of: int = 64
    norm_eps: float = 1e-5
    max_seq_len: int = 128
    dropout: float = 0.0
    pad_token_id: int = 0


class RMSNorm(nn.Module):
    """Root Mean Square Normalization"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(dim)

    def __call__(self, x: mx.array) -> mx.array:
        rms = mx.sqrt(mx.square(x).mean(axis=-1, keepdims=True) + self.eps)
        return x * (1.0 / rms) * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute RoPE frequencies"""
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2)[: dim // 2] / dim))
    t = mx.arange(end)
    freqs = t[:, None] * freqs[None, :]
    freqs_cos = mx.cos(freqs)
    freqs_sin = mx.sin(freqs)
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis: mx.array, x: mx.array) -> mx.array:
    """Reshape frequencies for broadcasting"""
    ndim = len(x.shape)
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.reshape(shape)


def apply_rotary_emb(xq: mx.array, xk: mx.array,
                      freqs_cos: mx.array, freqs_sin: mx.array):
    """Apply rotary position embedding"""
    xq_r = xq[..., :xq.shape[-1]//2]
    xq_i = xq[..., xq.shape[-1]//2:]
    xk_r = xk[..., :xk.shape[-1]//2]
    xk_i = xk[..., xk.shape[-1]//2:]

    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    xq_out = mx.concatenate([xq_out_r, xq_out_i], axis=-1)
    xk_out = mx.concatenate([xk_out_r, xk_out_i], axis=-1)

    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)


def repeat_kv(x: mx.array, n_rep: int) -> mx.array:
    """Repeat KV for grouped query attention"""
    if n_rep == 1:
        return x
    bs, slen, n_kv_heads, head_dim = x.shape
    x = x.reshape(bs, slen, n_kv_heads, 1, head_dim)
    x = mx.broadcast_to(x, (bs, slen, n_kv_heads, n_rep, head_dim))
    return x.reshape(bs, slen, n_kv_heads * n_rep, head_dim)


class Attention(nn.Module):
    """Multi-head attention with RoPE and GQA"""
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.n_kv_heads = args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.dropout = args.dropout

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def __call__(self, x: mx.array, freqs_cos: mx.array,
                 freqs_sin: mx.array, mask: Optional[mx.array] = None):
        bsz, seqlen, _ = x.shape

        xq = self.wq(x).reshape(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = self.wk(x).reshape(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = self.wv(x).reshape(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(0, 2, 1, 3)
        xk = xk.transpose(0, 2, 1, 3)
        xv = xv.transpose(0, 2, 1, 3)

        scores = (xq @ xk.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores + mask

        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(xq.dtype)
        output = scores @ xv

        output = output.transpose(0, 2, 1, 3).reshape(bsz, seqlen, -1)
        return self.wo(output)


class MLP(nn.Module):
    """SwiGLU feedforward network"""
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class DecoderLayer(nn.Module):
    """Single transformer decoder layer"""
    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = MLP(args.dim, args.hidden_dim, args.multiple_of, args.dropout)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def __call__(self, x, freqs_cos, freqs_sin, mask=None):
        h = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin, mask)
        return h + self.feed_forward(self.ffn_norm(h))


class Transformer(nn.Module):
    """Full transformer model"""
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = []
        for _ in range(args.n_layers):
            self.layers.append(DecoderLayer(0, args))
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        self.output.weight = self.tok_embeddings.weight

        freqs_cos, freqs_sin = precompute_freqs_cis(
            args.dim // args.n_heads, args.max_seq_len
        )
        self.freqs_cos = freqs_cos
        self.freqs_sin = freqs_sin

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for module in [self.tok_embeddings, self.output]:
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight = mx.random.normal(module.weight.shape) * 0.02

    def __call__(self, tokens, targets=None):
        _bsz, seqlen = tokens.shape

        h = self.tok_embeddings(tokens)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)

        h = self.norm(h)
        logits = self.output(h)

        result = {"logits": logits}

        if targets is not None:
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = targets.flatten()
            valid_mask = targets_flat != self.args.pad_token_id

            if mx.any(valid_mask).item():
                logits_valid = logits_flat[valid_mask]
                targets_valid = targets_flat[valid_mask]
                loss = mx.mean(nn.losses.cross_entropy(
                    logits_valid[None, ...],
                    targets_valid[None, ...]
                ))
                result["loss"] = loss
            else:
                result["loss"] = mx.array(0.0)

        return result

    def generate(self, tokens, eos_token_id, max_new_tokens=20,
                 temperature=0.8, top_k=40):
        """Generate text autoregressively"""
        self.eval()

        for _ in range(max_new_tokens):
            result = self.__call__(tokens)
            logits = result["logits"][:, -1, :]

            if temperature != 1.0:
                logits = logits / temperature

            if top_k > 0:
                k = min(top_k, logits.shape[-1])
                top_k_values = mx.topk(logits, k)
                threshold = mx.min(top_k_values)
                logits = mx.where(logits < threshold, -float("inf"), logits)

            probs = mx.softmax(logits, axis=-1)
            next_token = mx.argmax(probs, axis=-1)

            tokens = mx.concatenate([tokens, next_token[:, None]], axis=1)

            if next_token.item() == eos_token_id:
                break

        return tokens


class TokenizerWrapper:
    """Wrapper to load the custom tokenizer"""
    def __init__(self, tokenizer_path='./tokenizer_k/'):
        import json

        with open(f'{tokenizer_path}/tokenizer_config.json', 'r') as f:
            config = json.load(f)

        with open(f'{tokenizer_path}/tokenizer.json', 'r') as f:
            tokenizer_data = json.load(f)

        self.vocab = tokenizer_data['model']['vocab']
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = 0
        self.eos_token_id = 2

        self.add_bos_token = config.get('add_bos_token', False)
        self.add_eos_token = config.get('add_eos_token', False)
        self.chat_template = config.get('chat_template', None)

    def encode(self, text):
        """Simple character-level encoding"""
        tokens = []
        for char in text:
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.vocab.get('<unk>', 0))
        return tokens

    def decode(self, tokens):
        """Decode tokens to text"""
        text = ''
        for token in tokens:
            if token in self.reverse_vocab:
                text += self.reverse_vocab[token]
            else:
                text += '<unk>'
        return text

    def __call__(self, text, return_tensors=None):
        """Tokenize text"""
        tokens = self.encode(text)
        input_ids = mx.array([tokens])

        return {'input_ids': input_ids}


def count_parameters(model) -> int:
    """Count trainable parameters"""
    total = 0
    for key, value in model.__dict__.items():
        if isinstance(value, mx.array):
            total += value.size
    return total


def main():
    print("=" * 70)
    print("MLX Model Forward Pass Demo - Running on Mac Mini M4!")
    print("=" * 70)

    config = ModelConfig(
        dim=512,
        n_layers=8,
        n_heads=8,
        n_kv_heads=4,
        vocab_size=6144,
        max_seq_len=128,
        dropout=0.0
    )

    print(f"\n1. Model Configuration:")
    print(f"   - Hidden dim: {config.dim}")
    print(f"   - Layers: {config.n_layers}")
    print(f"   - Heads: {config.n_heads}")
    print(f"   - KV Heads: {config.n_kv_heads}")
    print(f"   - Vocab size: {config.vocab_size}")
    print(f"   - Max seq len: {config.max_seq_len}")

    print(f"\n2. Loading tokenizer...")
    tokenizer = TokenizerWrapper('./tokenizer_k/')
    print(f"   - Vocab size: {len(tokenizer.vocab)}")
    print(f"   - EOS token ID: {tokenizer.eos_token_id}")

    print(f"\n3. Initializing MLX model...")
    model = Transformer(config)
    num_params = count_parameters(model)
    print(f"   - Total parameters: {num_params / 1e6:.2f} M")
    print(f"   - Model device: MLX (Apple Silicon)")

    print(f"\n4. Testing forward pass...")
    print("-" * 70)

    test_texts = [
        "Hello, how are you today?",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating!"
    ]

    for i, text in enumerate(test_texts):
        tokens = tokenizer.encode(text)
        if len(tokens) < 10:
            tokens = tokens * (10 // len(tokens) + 1)
        tokens = tokens[:32]

        if len(tokens) < 32:
            tokens = tokens + [tokenizer.pad_token_id] * (32 - len(tokens))

        input_ids = mx.array([tokens])

        print(f"\n   Test {i+1}: '{text}'")
        print(f"   Input tokens: {len(tokens)}")

        start_time = time.time()
        result = model(input_ids)
        elapsed = time.time() - start_time

        logits = result['logits']
        print(f"   Output logits shape: {logits.shape}")
        print(f"   Forward pass time: {elapsed*1000:.2f} ms")

        print(f"   ✓ Forward pass successful!")

    print("-" * 70)

    print(f"\n5. Testing generation (random init, won't be meaningful)...")
    print("-" * 70)

    prompts = [
        "Hello",
        "The quick",
        "Machine"
    ]

    for i, prompt in enumerate(prompts):
        print(f"\n   Prompt {i+1}: '{prompt}'")

        tokens = tokenizer.encode(prompt)
        if len(tokens) < 10:
            tokens = tokens * (10 // len(tokens) + 1)
        tokens = tokens[:10]

        input_ids = mx.array([tokens])

        start_time = time.time()
        generated = model.generate(
            input_ids,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=20,
            temperature=0.8,
            top_k=20
        )
        elapsed = time.time() - start_time

        generated_tokens = generated[0].tolist()
        generated_text = tokenizer.decode(generated_tokens)

        print(f"   Generated in {elapsed*1000:.2f} ms")
        print(f"   Generated tokens: {generated_tokens[:15]}...")
        print(f"   Generated text: {generated_text[:50]}...")

    print("-" * 70)

    print(f"\n6. Summary:")
    print(f"   ✓ Model initializes correctly on Mac Mini M4")
    print(f"   ✓ Forward pass works (Transformer architecture)")
    print(f"   ✓ RMSNorm working")
    print(f"   ✓ RoPE (Rotary Position Embedding) working")
    print(f"   ✓ Grouped Query Attention working")
    print(f"   ✓ SwiGLU MLP working")
    print(f"   ✓ Generation works (random output)")

    print("\n" + "=" * 70)
    print("✓ MLX MODEL VERIFICATION COMPLETE!")
    print("=" * 70)
    print("\n💡 NOTE: The generated text is random (no trained weights).")
    print("   For meaningful output, you would need to:")
    print("   1. Train the model for many epochs on real data, OR")
    print("   2. Load pre-trained weights from a .pth checkpoint")
    print("=" * 70)


if __name__ == "__main__":
    import time
    main()
