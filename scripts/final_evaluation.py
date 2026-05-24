#!/usr/bin/env python3
"""
Final Evaluation and Model Usage Guide
"""

import sys
import os
import json
sys.path.insert(0, '/Volumes/UserData/workspace/basic-llm')

from src.model.qwen_mlx import QwenMLXInterface


def load_checkpoints():
    """Load all checkpoints."""
    checkpoints = []
    if os.path.exists("checkpoints"):
        for f in sorted(os.listdir("checkpoints")):
            if f.startswith("checkpoint_") and f.endswith(".json"):
                with open(os.path.join("checkpoints", f), "r") as fp:
                    checkpoints.append(json.load(fp))
    return checkpoints


def final_evaluation():
    """Run final model evaluation."""
    print("=" * 70)
    print("FINAL MODEL EVALUATION")
    print("=" * 70)
    
    # Load model
    model = QwenMLXInterface()
    model.load_model()
    
    if not model.model_loaded:
        print("❌ Model not loaded")
        return
    
    # Test with real trace from java-projects
    print("\n📝 Testing with real Java trace...")
    
    real_trace = """java.lang.RuntimeException: Database connection failed
    at com.example.UserServiceImpl.getUser(UserServiceImpl.java:42)
    at jdk.proxy3.$Proxy31.getUser(Unknown Source)
    at com.example.UserController.getUser(UserController.java:28)"""
    
    ecg = model.generate_ecg(real_trace, {})
    
    print("\n✅ Generated ECG:")
    print(f"  Nodes: {len(ecg.get('nodes', []))}")
    print(f"  Edges: {len(ecg.get('edges', []))}")
    print(f"  Enrichments: {ecg.get('metadata', {}).get('enrichment_applied', [])}")
    
    # Print checkpoints
    print("\n" + "=" * 70)
    print("TRAINING CHECKPOINTS SUMMARY")
    print("=" * 70)
    
    checkpoints = load_checkpoints()
    if checkpoints:
        print(f"\n📊 Total checkpoints: {len(checkpoints)}")
        print("\n{:<15} {:<20} {:<20} {:<20}".format(
            "Checkpoint", "Samples", "Node Match", "Enrichment Match"))
        print("-" * 75)
        
        for cp in checkpoints:
            print("{:<15} {:<20} {:<20.2%} {:<20.2%}".format(
                f"Checkpoint {cp['checkpoint']}",
                cp['samples_trained'],
                cp['node_match'],
                cp['enrichment_match']))
    else:
        print("\n⚠️ No checkpoints found")


def print_usage_guide():
    """Print model usage guide."""
    print("\n" + "=" * 70)
    print("MODEL USAGE GUIDE - HOW OTHERS CAN USE THIS")
    print("=" * 70)
    
    guide = """
📦 Quick Start for Other Users

1. Setup Environment
   ```bash
   git clone <repo-url>
   cd basic-llm
   pip install -r requirements.txt
   ```

2. Download Model
   ```bash
   # Option 1: From ModelScope (recommended)
   python scripts/download_model_ms.py
   
   # Option 2: From HuggingFace
   python scripts/download_model.py
   ```

3. Run the Pipeline
   ```bash
   python scripts/run_pipeline_demo.py
   ```

4. Use in Your Own Code
   ```python
   from src.model.qwen_mlx import QwenMLXInterface
   
   model = QwenMLXInterface()
   model.load_model()
   
   ecg = model.generate_ecg(your_trace, your_context)
   ```

📂 What's Included
- Model interface: src/model/qwen_mlx.py
- Pipeline: src/pipeline/rca_harness.py
- Parser: src/parser/call_graph.py
- Checkpoints: checkpoints/ (if available)

🎯 Key Features
- Apple Silicon optimized (MLX)
- Java stack trace → ECG conversion
- Proxy resolution
- Enrichment application
- Verification rules

💡 Tips
- First run may take time (model conversion to MLX format)
- Model is cached in ~/.cache/modelscope/hub/
- Check requirements.txt for dependencies

🔗 References
- ModelScope: https://www.modelscope.cn/models/qwen/Qwen2.5-1.5B-Instruct
- MLX: https://github.com/ml-explore/mlx
"""
    print(guide)


if __name__ == "__main__":
    final_evaluation()
    print_usage_guide()
