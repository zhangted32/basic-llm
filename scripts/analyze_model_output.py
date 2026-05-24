#!/usr/bin/env python3
"""
Analyze model output format vs expected format
"""

import sys
import json
sys.path.insert(0, '/Volumes/UserData/workspace/basic-llm')

from src.model.qwen_mlx import QwenMLXInterface


def analyze_output():
    model = QwenMLXInterface()
    model.load_model()
    
    trace = """java.lang.RuntimeException: Database connection failed
    at com.example.service.UserServiceImpl.getUser(UserServiceImpl.java:42)
    at jdk.proxy3.$Proxy31.getUser(Unknown Source)
    at com.example.controller.UserController.getUser(UserController.java:28)"""
    
    ecg = model.generate_ecg(trace, {})
    
    print("=" * 70)
    print("MODEL OUTPUT ANALYSIS")
    print("=" * 70)
    
    print("\n✅ Model generated ECG:")
    print(json.dumps(ecg, indent=2))
    
    print("\n" + "=" * 70)
    print("FORMAT COMPARISON")
    print("=" * 70)
    
    print("\n📊 Node enrichment format:")
    for node in ecg.get('nodes', [])[:2]:
        print(f"  ID: {node.get('id')}")
        print(f"  Enrichment type: {type(node.get('enrichment'))}")
        print(f"  Enrichment value: {node.get('enrichment')}")
        print()
    
    print("\n📊 Metadata enrichment_applied format:")
    print(f"  Type: {type(ecg.get('metadata', {}).get('enrichment_applied'))}")
    print(f"  Value: {ecg.get('metadata', {}).get('enrichment_applied')}")
    
    print("\n" + "=" * 70)
    print("EXPECTED FORMAT")
    print("=" * 70)
    print("""
Expected node enrichment:
  "enrichment": {
    "type": "meta",
    "module": "service",
    "layer": "business"
  }

Expected metadata:
  "metadata": {
    "enrichment_applied": ["E-META", "E-PROXY"]
  }
""")


if __name__ == "__main__":
    analyze_output()
