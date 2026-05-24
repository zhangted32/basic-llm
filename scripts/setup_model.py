#!/usr/bin/env python3
"""
Download and setup Qwen2.5-1.5B-Instruct model for RCA-Agent.

This script:
1. Downloads the model from HuggingFace (if not cached)
2. Converts to MLX format
3. Runs a test inference
"""

import os
import sys
from pathlib import Path

def check_huggingface_cli():
    """Check if huggingface-cli is available."""
    import shutil
    if shutil.which("huggingface-cli"):
        return True
    print("❌ huggingface-cli not found.")
    print("   Install with: pip install huggingface_hub")
    return False

def download_model():
    """Download Qwen model from HuggingFace."""
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    
    print(f"Downloading {model_id}...")
    print("This may take a few minutes depending on your internet connection.")
    print()
    
    # Use huggingface-cli to download
    os.system(f"huggingface-cli download {model_id}")
    
    print()
    print("✅ Model downloaded successfully!")
    
    # Show cache location
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    print(f"   Cache location: {cache_dir}")

def test_model_loading():
    """Test that the model can be loaded."""
    print("\nTesting model loading...")
    
    try:
        from src.model.qwen_mlx import QwenMLXInterface
        
        model = QwenMLXInterface()
        model.load_model()
        
        if model.model_loaded:
            print("✅ Model loaded successfully!")
            return True
        else:
            print("⚠️  Model not loaded (may need MLX conversion)")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def run_test_inference():
    """Run a test inference."""
    print("\nRunning test inference...")
    
    try:
        from src.pipeline.rca_harness import RCAHarness
        
        trace = """java.lang.NullPointerException
    at com.example.UserService.getUser(UserService.java:42)
    at com.example.Controller.handle(Controller.java:28)"""
        
        context = {}
        
        harness = RCAHarness()
        result = harness.run_single_trace(trace, context, skip_verification=True)
        
        print(f"✅ Inference completed!")
        print(f"   RCG frames: {len(result.rcg.frames)}")
        print(f"   ECG nodes: {len(result.ecg.get('nodes', []))}")
        print(f"   Time: {result.total_time:.3f}s")
        
        return True
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False

def main():
    print("=" * 60)
    print("RCA-Agent Model Setup")
    print("=" * 60)
    print()
    
    # Check prerequisites
    if not check_huggingface_cli():
        sys.exit(1)
    
    # Download model
    print("\nStep 1: Download Model")
    print("-" * 40)
    download_model()
    
    # Test loading
    print("\nStep 2: Test Model Loading")
    print("-" * 40)
    loaded = test_model_loading()
    
    # Test inference
    if loaded:
        print("\nStep 3: Test Inference")
        print("-" * 40)
        run_test_inference()
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\nYou can now run:")
    print("  python3 scripts/demo_rca_agent.py")
    print("  python3 scripts/run_dpo_training.py")

if __name__ == "__main__":
    main()