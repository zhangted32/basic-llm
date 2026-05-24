#!/usr/bin/env python3
"""
Debug script to see what the model is actually outputting.
"""

import sys
sys.path.insert(0, '/Volumes/UserData/workspace/basic-llm')

import json
from src.model.qwen_mlx import QwenMLXInterface
from mlx_lm import generate


def test_model_output():
    print("=" * 60)
    print("Debugging Model Output")
    print("=" * 60)
    print()
    
    # Load model
    model_interface = QwenMLXInterface()
    model_interface.load_model()
    
    if not model_interface.model_loaded:
        print("❌ Model not loaded")
        return
    
    # Test trace
    trace = """java.lang.RuntimeException: Database connection failed
    at com.example.service.UserServiceImpl.getUser(UserServiceImpl.java:42)
    at jdk.proxy3.$Proxy31.getUser(Unknown Source)
    at com.example.controller.UserController.getUser(UserController.java:28)"""
    
    context = {
        "proxy_mappings": {"$Proxy31": "com.example.service.UserServiceImpl"}
    }
    
    # Get raw model output
    print("Getting raw model output...")
    prompt = model_interface._build_prompt(trace, context)
    print(f"\nPrompt length: {len(prompt)} characters")
    print(f"Prompt preview:\n{prompt[:500]}...")
    print()
    
    # Generate output
    output = generate(
        model_interface.model,
        model_interface.tokenizer,
        prompt=prompt,
        max_tokens=1024,
        verbose=False
    )
    
    print("=" * 60)
    print("RAW MODEL OUTPUT:")
    print("=" * 60)
    print(repr(output))  # Show raw output with escaped characters
    print()
    
    print("=" * 60)
    print("FORMATTED OUTPUT:")
    print("=" * 60)
    print(output)
    print()
    
    # Try to parse
    print("=" * 60)
    print("PARSING ATTEMPT:")
    print("=" * 60)
    try:
        # Try to find JSON in output
        import re
        json_match = re.search(r'\{.*\}', output, re.DOTALL)
        
        if json_match:
            print("Found JSON-like content:")
            print(json_match.group())
            print()
            
            try:
                ecg = json.loads(json_match.group())
                print("✅ Successfully parsed JSON!")
                print(f"Nodes: {len(ecg.get('nodes', []))}")
                print(f"Edges: {len(ecg.get('edges', []))}")
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
        else:
            print("❌ No JSON found in output")
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_model_output()