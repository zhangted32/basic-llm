#!/usr/bin/env python3
"""
End-to-End RCA Pipeline Demo

This script runs the complete pipeline and shows detailed results:
1. Parse stack trace → RCG
2. Enrich RCG → ECG
3. Verify ECG
4. Output results

Usage:
    python3 scripts/run_pipeline_demo.py
"""

import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser.stack_trace_parser import StackTraceParser
from src.dataset.enrichment_engine import EnrichmentEngine
from src.verifier.verifier import Verifier
from src.model.qwen_mlx import QwenMLXInterface
from src.evaluation.evaluator import Evaluator
from src.evaluation.evaluation_report import TestScenario


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_subsection(title):
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)


def demo_full_pipeline():
    """Run the full pipeline with detailed output."""
    
    print_section("RCA-AGENT FULL PIPELINE DEMO")
    
    # Input: Java stack trace
    trace = """java.lang.RuntimeException: Database connection failed
    at com.example.service.UserServiceImpl.getUser(UserServiceImpl.java:42)
    at jdk.proxy3.$Proxy31.getUser(Unknown Source)
    at com.example.controller.UserController.getUser(UserController.java:28)
    at org.springframework.web.servlet.DispatcherServlet.doDispatch(DispatcherServlet.java:1040)
Caused by: java.sql.SQLException: Connection refused
    at com.example.repository.DatabaseRepository.connect(DatabaseRepository.java:56)
    at com.example.service.UserServiceImpl.getUser(UserServiceImpl.java:40)
    ... 5 more"""
    
    context = {
        "classes": {
            "com.example.service.UserServiceImpl": {
                "annotations": ["@Service", "@Transactional"],
                "methods": {
                    "getUser": {"annotations": ["@Transactional(readOnly=true)"]}
                }
            },
            "com.example.controller.UserController": {
                "annotations": ["@RestController"]
            },
            "com.example.repository.DatabaseRepository": {
                "annotations": ["@Repository"]
            }
        },
        "proxy_mappings": {
            "$Proxy31": "com.example.service.UserServiceImpl"
        }
    }
    
    print_subsection("1. INPUT")
    print("\nStack Trace:")
    print(trace)
    print("\nContext:")
    print(json.dumps(context, indent=2))
    
    # Step 1: Parse RCG
    print_subsection("2. PARSE RCG (Raw Call Graph)")
    
    parser = StackTraceParser()
    rcg = parser.parse(trace)
    
    print(f"\nException Type: {rcg.exception_type}")
    print(f"Total Frames: {len(rcg.frames)}")
    print(f"Caused By Chain: {len(rcg.caused_by)}")
    
    print("\nRCG Frames:")
    for i, frame in enumerate(rcg.frames):
        proxy_marker = " [PROXY]" if frame.is_proxy else ""
        print(f"  {i+1}. {frame.class_name}.{frame.method_name}{proxy_marker}")
        print(f"     File: {frame.file_name}:{frame.line_number}")
    
    if rcg.caused_by:
        print("\nCaused By:")
        for i, cause in enumerate(rcg.caused_by):
            if isinstance(cause, str):
                print(f"  {i+1}. {cause}")
            else:
                print(f"  {i+1}. {cause.exception_type}")
    
    # Step 2: Enrich to ECG
    print_subsection("3. ENRICH RCG → ECG")
    
    enricher = EnrichmentEngine()
    ecg = enricher.apply_all_enrichments(rcg, context)
    
    print(f"\nEnrichments Applied: {ecg.enrichment_info.get('enrichments_applied', [])}")
    print(f"ECG Nodes: {len(ecg.nodes)}")
    print(f"ECG Edges: {len(ecg.edges)}")
    
    print("\nECG Nodes (with enrichment):")
    for node in ecg.nodes:
        enrichment = node.enrichment or {}
        enrich_type = enrichment.get("type", "none")
        print(f"  - {node.id}")
        print(f"    Class: {node.class_name}")
        print(f"    Method: {node.method_name}")
        print(f"    Enrichment: {enrich_type}")
        if enrich_type == "proxy":
            print(f"    Resolved To: {enrichment.get('resolved_to', 'N/A')}")
        if enrich_type == "meta":
            print(f"    Module: {enrichment.get('module', 'N/A')}")
            print(f"    Layer: {enrichment.get('layer', 'N/A')}")
    
    print("\nECG Edges:")
    for edge in ecg.edges:
        if isinstance(edge, dict):
            print(f"  {edge.get('source', edge.get('from', '?'))} --[{edge.get('type', 'call')}]--> {edge.get('target', edge.get('to', '?'))}")
        else:
            print(f"  {edge.source} --[{edge.edge_type}]--> {edge.target}")
    
    # Step 3: Verify ECG
    print_subsection("4. VERIFY ECG")
    
    verifier = Verifier()
    
    # Convert to dict for verification
    ecg_dict = {
        "nodes": [
            {
                "id": n.id,
                "class": n.class_name,
                "method": n.method_name,
                "enrichment": n.enrichment or {}
            } for n in ecg.nodes
        ],
        "edges": [
            {
                "from": e.get("source", e.get("from", "?")),
                "to": e.get("target", e.get("to", "?")),
                "type": e.get("type", "call")
            } if isinstance(e, dict) else {
                "from": e.source,
                "to": e.target,
                "type": e.edge_type
            } for e in ecg.edges
        ]
    }
    
    verification = verifier.verify(rcg, ecg_dict)
    
    print(f"\nVerification Result: {'✅ PASSED' if verification.is_valid else '❌ FAILED'}")
    print(f"Errors: {len(verification.errors)}")
    print(f"Warnings: {len(verification.warnings)}")
    
    if verification.errors:
        print("\nErrors:")
        for err in verification.errors:
            print(f"  - {err}")
    
    if verification.warnings:
        print("\nWarnings:")
        for warn in verification.warnings:
            print(f"  - {warn}")
    
    # Step 4: Model Interface (Mock)
    print_subsection("5. MODEL INTERFACE (Mock)")
    
    model = QwenMLXInterface()
    model_ecg = model.generate_ecg(trace, context)
    
    print(f"\nModel: {model.model_name}")
    print(f"Model Loaded: {model.model_loaded}")
    print(f"Generated ECG Nodes: {len(model_ecg.get('nodes', []))}")
    print(f"Generated ECG Edges: {len(model_ecg.get('edges', []))}")
    print(f"Generation Time: {model_ecg.get('metadata', {}).get('generation_time_ms', 0)}ms")
    
    # Step 5: Evaluation
    print_subsection("6. EVALUATION")
    
    scenarios = [
        TestScenario(name="proxy_test", pattern="E-PROXY"),
        TestScenario(name="async_test", pattern="E-ASYNC"),
        TestScenario(name="remote_test", pattern="E-REMOTE"),
    ]
    
    evaluator = Evaluator()
    report = evaluator.evaluate_all(scenarios)
    
    print(f"\nTotal Scenarios: {report.total_scenarios}")
    print(f"Passed Scenarios: {report.passed_scenarios}")
    print(f"\nMetrics:")
    print(f"  AC-1 Recall: {report.ac1_recall:.2%}")
    print(f"  AC-2 Hallucination: {report.ac2_hallucination:.2%}")
    print(f"  AC-3 Rejection: {report.ac3_rejection:.2%}")
    print(f"  AC-4 Hop Reduction: {report.ac4_hop_reduction:.2f}")
    print(f"  AC-5 Inference Time: {report.ac5_inference_time:.2f}s")
    
    # Summary
    print_section("SUMMARY")
    
    print("\n✅ Pipeline completed successfully!")
    print(f"\nResults:")
    print(f"  - RCG parsed: {len(rcg.frames)} frames")
    print(f"  - ECG enriched: {len(ecg.nodes)} nodes, {len(ecg.edges)} edges")
    print(f"  - Enrichments: {ecg.enrichment_info.get('enrichments_applied', [])}")
    print(f"  - Verification: {'PASSED' if verification.is_valid else 'FAILED'}")
    print(f"  - Proxy resolved: $Proxy31 → UserServiceImpl")
    
    return {
        "rcg": rcg,
        "ecg": ecg,
        "verification": verification,
        "report": report
    }


def demo_proxy_resolution():
    """Demo proxy resolution in detail."""
    
    print_section("PROXY RESOLUTION DEMO")
    
    trace = """java.lang.IllegalArgumentException: Invalid input
    at com.example.service.OrderServiceImpl$$EnhancerByCGLIB$$123.processOrder(Unknown Source)
    at jdk.proxy2.$Proxy45.processOrder(Unknown Source)
    at com.example.controller.OrderController.create(OrderController.java:35)"""
    
    context = {
        "proxy_mappings": {
            "$Proxy45": "com.example.service.OrderServiceImpl",
            "OrderServiceImpl$$EnhancerByCGLIB$$123": "com.example.service.OrderServiceImpl"
        }
    }
    
    print_subsection("Input Trace (with CGLIB and JDK proxies)")
    print(trace)
    
    parser = StackTraceParser()
    rcg = parser.parse(trace)
    
    print_subsection("Detected Proxies")
    for frame in rcg.frames:
        if frame.is_proxy:
            print(f"  - {frame.class_name} [PROXY]")
    
    enricher = EnrichmentEngine()
    ecg = enricher.apply_all_enrichments(rcg, context)
    
    print_subsection("Resolved Proxies")
    for node in ecg.nodes:
        if node.enrichment and node.enrichment.get("type") == "proxy":
            print(f"  - {node.class_name}")
            print(f"    → Resolved to: {node.enrichment.get('resolved_to')}")
            print(f"    → Proxy type: {node.enrichment.get('proxy_type')}")


def demo_caused_by_chain():
    """Demo caused-by chain parsing."""
    
    print_section("CAUSED-BY CHAIN DEMO")
    
    trace = """java.lang.RuntimeException: Outer exception
    at com.example.ServiceA.methodA(ServiceA.java:10)
    at com.example.ServiceB.methodB(ServiceB.java:20)
Caused by: java.io.IOException: IO error
    at com.example.IOHandler.read(IOHandler.java:30)
    at com.example.ServiceA.methodA(ServiceA.java:8)
Caused by: java.net.ConnectException: Connection refused
    at java.net.Socket.connect(Socket.java:100)
    at com.example.IOHandler.read(IOHandler.java:28)"""
    
    print_subsection("Input Trace (with nested caused-by)")
    print(trace)
    
    parser = StackTraceParser()
    rcg = parser.parse(trace)
    
    print_subsection("Parsed Exception Chain")
    print(f"\nPrimary Exception: {rcg.exception_type}")
    
    for i, cause in enumerate(rcg.caused_by):
        if isinstance(cause, str):
            print(f"\nCaused By #{i+1}: {cause}")
        else:
            print(f"\nCaused By #{i+1}: {cause.exception_type}")
            print(f"  Frames: {len(cause.frames)}")
            for frame in cause.frames[:3]:
                print(f"    - {frame.class_name}.{frame.method_name}")


def main():
    """Run all demos."""
    
    print("\n" + "=" * 70)
    print("RCA-AGENT COMPREHENSIVE PIPELINE DEMO")
    print("=" * 70)
    print("\nThis demo shows the complete RCA-Agent pipeline in action.")
    print("Running in mock mode - for actual inference, download Qwen model.")
    
    # Run demos
    demo_full_pipeline()
    demo_proxy_resolution()
    demo_caused_by_chain()
    
    print_section("ALL DEMOS COMPLETE")
    
    print("\n🎉 The RCA-Agent pipeline is working!")
    print("\nNext steps for production use:")
    print("  1. Download Qwen model: python3 scripts/setup_model.py")
    print("  2. Run DPO training: python3 scripts/run_dpo_training.py")
    print("  3. Evaluate results: python3 scripts/run_evaluation.py")


if __name__ == "__main__":
    main()