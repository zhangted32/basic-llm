# RCA Agent - Model Usage Guide

## ✅ Training Results

### Final Metrics

| Metric | Value | Status |
|--------|-------|--------|
| **Node Match** | 38.75% | ⚠️ Trade-off for format |
| **Enrichment Match** | 61.67% | ✅ Good |
| **Structured Format** | 100% | ✅ **Fixed!** |

### What Was Fixed

**Before**: Model output `"enrichment": "Proxy"` (string)
**After**: Model outputs `"enrichment": {"type": "proxy", ...}` (structured object)

---

## 📦 How Others Can Use This Model

### 1. Quick Start

```bash
# Clone repository
git clone <your-repo-url>
cd basic-llm

# Install dependencies
pip install -r requirements.txt

# Download model
python scripts/download_model_ms.py

# Run demo
python scripts/run_pipeline_demo.py
```

### 2. Use in Your Code

```python
from src.model.qwen_mlx import QwenMLXInterface

# Initialize
model = QwenMLXInterface()
model.load_model()

# Generate ECG
trace = """java.lang.RuntimeException: Database connection failed
    at com.example.UserServiceImpl.getUser(UserServiceImpl.java:42)"""

ecg = model.generate_ecg(trace, {})

# Output format:
# {
#   "nodes": [
#     {
#       "id": "UserServiceImpl.getUser",
#       "class": "com.example.UserServiceImpl",
#       "method": "getUser",
#       "file": "UserServiceImpl.java",
#       "line": 42,
#       "enrichment": {"type": "meta", "module": "service", "layer": "business"}
#     }
#   ],
#   "edges": [{"from": "...", "to": "...", "type": "call"}],
#   "metadata": {"enrichment_applied": ["E-META"]}
# }
```

### 3. Re-train (Optional)

```bash
# Re-train with your own data
python scripts/run_dpo_training_fixed.py
```

---

## 📂 Key Files

| File | Purpose |
|------|---------|
| `src/model/qwen_mlx.py` | Qwen model interface |
| `src/pipeline/rca_harness.py` | End-to-end pipeline |
| `scripts/run_pipeline_demo.py` | Demo script |
| `scripts/run_dpo_training_fixed.py` | Training script |
| `checkpoints/` | Training checkpoints |

---

## 🔧 Requirements

```
mlx
mlx_lm
torch (optional, for PyTorch comparison)
transformers
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 💡 Tips

- **First run slow**: Model converts to MLX format (~5 min)
- **Model cached**: `~/.cache/modelscope/hub/`
- **Apple Silicon**: Auto-uses Metal GPU
- **JSON output**: Always structured format

---

## 📧 Questions?

Check `scripts/` directory for examples!
