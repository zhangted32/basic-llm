#!/usr/bin/env python3
"""
Test script to load Qwen model from ModelScope and run inference.
"""

import sys
sys.path.insert(0, '/Volumes/UserData/workspace/basic-llm')

def test_model_load():
    print("=" * 60)
    print("Testing Qwen Model Loading from ModelScope")
    print("=" * 60)
    print()
    
    try:
        from mlx_lm import load, generate
        from pathlib import Path
        
        # Check ModelScope cache
        modelscope_path = Path.home() / ".cache" / "modelscope" / "hub" / "qwen" / "Qwen2.5-1.5B-Instruct"
        
        if not modelscope_path.exists():
            print(f"❌ ModelScope path not found: {modelscope_path}")
            return False
        
        print(f"✅ ModelScope cache found at: {modelscope_path}")
        print()
        
        # Load the model
        print("Loading model... (this may take 1-2 minutes)")
        model, tokenizer = load(str(modelscope_path))
        
        print("✅ Model loaded successfully!")
        print()
        
        # Test inference (without temperature parameter)
        print("Testing inference...")
        prompt = "Hello, how are you?"
        response = generate(model, tokenizer, prompt=prompt, max_tokens=50)
        
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print()
        print("✅ Inference successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_model_load()
    sys.exit(0 if success else 1)