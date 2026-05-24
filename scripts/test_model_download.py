#!/usr/bin/env python3
"""
Script to download and test the Qwen2.5-1.5B-Instruct model with MLX.
"""

import os
import sys
sys.path.insert(0, '/Volumes/UserData/workspace/basic-llm')

from src.model.qwen_mlx import QwenMLXInterface


def main():
    print("=" * 60)
    print("Qwen Model Download and Test")
    print("=" * 60)
    print()
    
    # Create model interface
    model = QwenMLXInterface()
    
    # Load the model (this will download if not cached)
    print("Attempting to load Qwen2.5-1.5B-Instruct model...")
    print("This may take several minutes on first run.")
    print()
    
    model.load_model()
    
    print()
    if model.model_loaded:
        print("✅ Model loaded successfully!")
        print(f"   Model: {model.model_name}")
        print(f"   Model object: {type(model.model)}")
        print(f"   Tokenizer: {type(model.tokenizer)}")
        
        # Test inference
        print()
        print("Testing inference...")
        
        test_trace = """java.lang.NullPointerException
    at com.example.service.UserService.getUser(UserService.java:42)
    at com.example.controller.UserController.handle(UserController.java:28)"""
        
        test_context = {}
        
        result = model.generate_ecg(test_trace, test_context)
        
        print(f"✅ Inference completed!")
        print(f"   ECG Nodes: {len(result.get('nodes', []))}")
        print(f"   ECG Edges: {len(result.get('edges', []))}")
        print(f"   Generation Time: {result.get('metadata', {}).get('generation_time_ms', 0)}ms")
        
    else:
        print("❌ Model loading failed. Using mock mode.")
        print("Please check your internet connection and try again.")


if __name__ == "__main__":
    main()