#!/usr/bin/env python3
"""
RCA-Agent Demo Script

This script demonstrates the full RCA-Agent pipeline:
1. Parse a Java stack trace → RCG
2. Generate ECG using the model
3. Verify the ECG
4. Output results

Usage:
    python3 demo_rca_agent.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.rca_harness import RCAHarness, run_trace
from src.evaluation.evaluator import Evaluator, GroundTruthGenerator
from src.evaluation.evaluation_report import TestScenario


def demo_single_trace():
    """Demo: Process a single stack trace."""
    print("=" * 60)
    print("DEMO 1: Single Trace Processing")
    print("=" * 60)
    
    # Sample Java stack trace with proxy
    trace = """java.lang.RuntimeException: User not found
    at com.example.service.UserServiceImpl.getUser(UserServiceImpl.java:42)
    at jdk.proxy3.$Proxy31.getUser(Unknown Source)
    at com.example.controller.UserController.getUser(UserController.java:28)
    at org.springframework.web.servlet.DispatcherServlet.doDispatch(DispatcherServlet.java:1040)
    at org.springframework.web.servlet.FrameworkServlet.service(FrameworkServlet.java:973)
    at javax.servlet.http.HttpServlet.service(HttpServlet.java:750)"""
    
    # Static analysis context
    context = {
        "classes": {
            "com.example.service.UserServiceImpl": {
                "annotations": ["@Service", "@Transactional"]
            },
            "com.example.controller.UserController": {
                "annotations": ["@RestController", "@RequestMapping"]
            }
        },
        "proxy_mappings": {
            "$Proxy31": "com.example.service.UserServiceImpl"
        }
    }
    
    print("\nInput Stack Trace:")
    print("-" * 40)
    print(trace)
    
    print("\nContext:")
    print("-" * 40)
    print(f"  Proxy mappings: {context['proxy_mappings']}")
    print(f"  Classes: {list(context['classes'].keys())}")
    
    # Run the pipeline
    print("\nRunning RCA Pipeline...")
    print("-" * 40)
    
    harness = RCAHarness()
    result = harness.run_single_trace(trace, context, skip_verification=True)
    
    # Display results
    print("\nResults:")
    print("-" * 40)
    print(f"  RCG Frames: {len(result.rcg.frames)}")
    print(f"  ECG Nodes: {len(result.ecg.get('nodes', []))}")
    print(f"  ECG Edges: {len(result.ecg.get('edges', []))}")
    print(f"  Inference Time: {result.inference_time:.3f}s")
    print(f"  Total Time: {result.total_time:.3f}s")
    print(f"  Verification: {'✓ PASSED' if result.verification.is_valid else '✗ FAILED'}")
    
    if result.ecg.get("metadata", {}).get("enrichment_applied"):
        print(f"  Enrichments: {result.ecg['metadata']['enrichment_applied']}")
    
    # Show ECG nodes
    print("\nECG Nodes:")
    for node in result.ecg.get("nodes", [])[:5]:
        enrichment_type = node.get("enrichment", {}).get("type", "none")
        print(f"  - {node['id']} [{enrichment_type}]")
    
    return result


def demo_async_trace():
    """Demo: Process an async stack trace."""
    print("\n" + "=" * 60)
    print("DEMO 2: Async Boundary Detection")
    print("=" * 60)
    
    trace = """java.util.concurrent.ExecutionException: Task failed
    at java.util.concurrent.FutureTask.get(FutureTask.java:155)
    at com.example.service.DataService.fetchData(DataService.java:25)
    at com.example.service.DataService$$Lambda$123/0x123.run(Unknown Source)
    at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1128)
    at com.example.controller.DataController.getData(DataController.java:30)"""
    
    context = {
        "classes": {
            "com.example.service.DataService": {
                "annotations": ["@Service", "@Async"]
            }
        }
    }
    
    print("\nInput Stack Trace (with async):")
    print("-" * 40)
    print(trace)
    
    harness = RCAHarness()
    result = harness.run_single_trace(trace, context, skip_verification=True)
    
    print("\nResults:")
    print("-" * 40)
    print(f"  ECG Nodes: {len(result.ecg.get('nodes', []))}")
    print(f"  Async boundaries detected: ", end="")
    
    async_nodes = [n for n in result.ecg.get("nodes", []) 
                   if n.get("enrichment", {}).get("type") == "async"]
    print(len(async_nodes))
    
    return result


def demo_evaluation():
    """Demo: Run evaluation on test scenarios."""
    print("\n" + "=" * 60)
    print("DEMO 3: Evaluation Harness")
    print("=" * 60)
    
    # Create test scenarios
    scenarios = [
        TestScenario(name="proxy", pattern="E-PROXY", description="Proxy resolution test"),
        TestScenario(name="async", pattern="E-ASYNC", description="Async boundary test"),
        TestScenario(name="feign", pattern="E-REMOTE", description="Remote service test"),
    ]
    
    print(f"\nTest Scenarios: {len(scenarios)}")
    for s in scenarios:
        print(f"  - {s.name}: {s.pattern}")
    
    # Run evaluation
    evaluator = Evaluator()
    report = evaluator.evaluate_all(scenarios)
    
    # Print report
    report.print_summary()
    
    return report


def demo_dpo_training():
    """Demo: Show DPO training setup (mock run)."""
    print("\n" + "=" * 60)
    print("DEMO 4: DPO Training Setup")
    print("=" * 60)
    
    from src.training.dpo_config import DPOConfig, DPOConfigManager
    from src.training.dpo_trainer import DPOTrainer
    
    # Get quick test config
    config = DPOConfigManager.get_config("quick_test")
    
    print("\nDPO Training Configuration:")
    print("-" * 40)
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Gradient Accumulation: {config.gradient_accumulation}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Training Steps: {config.training_steps}")
    print(f"  LoRA Rank: {config.lora_rank}")
    print(f"  LoRA Alpha: {config.lora_alpha}")
    print(f"  Output Dir: {config.output_dir}")
    
    print("\nTo run actual training:")
    print("-" * 40)
    print("  1. Download Qwen model from HuggingFace:")
    print("     huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct")
    print("  2. Prepare DPO dataset:")
    print("     python3 scripts/prepare_dpo_dataset.py")
    print("  3. Run training:")
    print("     python3 scripts/run_dpo_training.py")
    
    return config


def demo_ground_truth():
    """Demo: Show ground truth generation."""
    print("\n" + "=" * 60)
    print("DEMO 5: Ground Truth Generation")
    print("=" * 60)
    
    # Generate proxy scenario
    proxy_data = GroundTruthGenerator.generate_proxy_scenario()
    
    print("\nProxy Scenario Ground Truth:")
    print("-" * 40)
    print(f"  Scenario: {proxy_data['scenario']}")
    print(f"  Pattern: {proxy_data['pattern']}")
    print(f"  Test Cases: {len(proxy_data['test_cases'])}")
    
    if proxy_data['test_cases']:
        tc = proxy_data['test_cases'][0]
        print(f"\n  Example Trace:")
        print(f"    Exception: {tc['trace'].split(chr(10))[0]}")
        print(f"  Ground Truth ECG:")
        print(f"    Nodes: {len(tc['ground_truth_ecg']['nodes'])}")
        print(f"    Edges: {len(tc['ground_truth_ecg']['edges'])}")
    
    return proxy_data


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("RCA-Agent Demo Suite")
    print("=" * 60)
    print("\nThis demo shows the RCA-Agent pipeline capabilities.")
    print("Note: Model is running in mock mode for demonstration.")
    print("For actual model inference, download Qwen2.5-1.5B-Instruct.")
    
    # Run demos
    demo_single_trace()
    demo_async_trace()
    demo_evaluation()
    demo_dpo_training()
    demo_ground_truth()
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nNext Steps:")
    print("  1. Download Qwen model for actual inference")
    print("  2. Prepare DPO training dataset")
    print("  3. Run DPO training to align the model")
    print("  4. Evaluate on test scenarios")
    print("\nSee docs/RCA-Agent/implementation_design.md for details.")


if __name__ == "__main__":
    main()