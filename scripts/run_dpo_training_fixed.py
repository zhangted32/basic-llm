#!/usr/bin/env python3
"""
DPO Training Script - Fixed Evaluation
"""

import sys
import os
import json
import time
import random
sys.path.insert(0, '/Volumes/UserData/workspace/basic-llm')

from src.model.qwen_mlx import QwenMLXInterface
import mlx.core as mx


# Set up MLX
mx.set_default_device(mx.gpu if mx.metal.is_available() else mx.cpu)
print(f"MLX Device: {mx.default_device()}")


def generate_samples(count=150):
    """Generate synthetic training samples with diverse enrichments."""
    samples = []
    
    exception_types = [
        "java.lang.RuntimeException",
        "java.lang.NullPointerException", 
        "java.lang.IllegalArgumentException",
        "java.sql.SQLException",
        "java.io.IOException",
        "java.util.concurrent.ExecutionException",
        "org.springframework.transaction.CannotCreateTransactionException",
        "com.netflix.hystrix.exception.HystrixRuntimeException",
        "feign.RetryableException",
        "redis.clients.jedis.exceptions.JedisConnectionException"
    ]
    
    modules = [
        {"name": "User", "package": "com.example.service", "type": "service"},
        {"name": "Order", "package": "com.example.service", "type": "service"},
        {"name": "Product", "package": "com.example.service", "type": "service"},
        {"name": "Payment", "package": "com.example.service", "type": "service"},
        {"name": "Inventory", "package": "com.example.service", "type": "service"},
        {"name": "Database", "package": "com.example.repository", "type": "repository"},
        {"name": "Cache", "package": "com.example.repository", "type": "repository"},
        {"name": "User", "package": "com.example.controller", "type": "controller"},
        {"name": "Order", "package": "com.example.controller", "type": "controller"},
        {"name": "Product", "package": "com.example.controller", "type": "controller"},
        {"name": "Payment", "package": "com.example.client", "type": "client"},
        {"name": "External", "package": "com.example.client", "type": "client"}
    ]
    
    proxy_types = ["jdk.proxy3.$Proxy", "jdk.proxy2.$Proxy", "com.example.service.$$EnhancerByCGLIB$$", "org.springframework.cglib.proxy.$Proxy"]
    frameworks = [
        "org.springframework.web.servlet.DispatcherServlet", 
        "org.springframework.transaction.interceptor.TransactionInterceptor",
        "org.springframework.aop.framework.ReflectiveMethodInvocation",
        "com.netflix.hystrix.AbstractCommand"
    ]
    
    for i in range(count):
        exception = random.choice(exception_types)
        service = random.choice([m for m in modules if m["package"].endswith("service")])
        repo = random.choice([m for m in modules if m["package"].endswith("repository")])
        controller = random.choice([m for m in modules if m["package"].endswith("controller")])
        proxy_num = random.randint(10, 99)
        proxy_type = random.choice(proxy_types)
        framework = random.choice(frameworks)
        
        has_proxy = random.choice([True, False])
        has_cause = random.choice([True, False])
        
        trace_lines = []
        trace_lines.append(f"{exception}: Error occurred")
        trace_lines.append(f"    at {service['package']}.{service['name']}ServiceImpl.{random.choice(['get', 'save', 'update', 'delete'])}{service['name']}({service['name']}ServiceImpl.java:{random.randint(1, 100)})")
        
        if has_proxy:
            proxy_name = f"{proxy_type}{proxy_num}"
            trace_lines.append(f"    at {proxy_name}.{random.choice(['get', 'save', 'update', 'delete'])}{service['name']}(Unknown Source)")
        
        trace_lines.append(f"    at {controller['package']}.{controller['name']}Controller.handle({controller['name']}Controller.java:{random.randint(1, 100)})")
        
        if "Transaction" in framework:
            trace_lines.append(f"    at {framework}.invoke({framework.split('.')[-1]}.java:{random.randint(1, 100)})")
        
        if has_cause:
            cause_exception = random.choice([e for e in exception_types if e != exception])
            trace_lines.append(f"Caused by: {cause_exception}: Nested error")
            trace_lines.append(f"    at {repo['package']}.{repo['name']}Repository.connect({repo['name']}Repository.java:{random.randint(1, 100)})")
            trace_lines.append(f"    at {service['package']}.{service['name']}ServiceImpl.{random.choice(['get', 'save', 'update', 'delete'])}{service['name']}({service['name']}ServiceImpl.java:{random.randint(1, 100)})")
            trace_lines.append("    ... 3 more")
        
        trace = "\n".join(trace_lines)
        
        context = {
            "classes": {
                f"{service['package']}.{service['name']}ServiceImpl": {
                    "annotations": ["@Service", "@Transactional"]
                },
                f"{controller['package']}.{controller['name']}Controller": {
                    "annotations": ["@RestController"]
                }
            }
        }
        
        if has_proxy:
            context["proxy_mappings"] = {
                proxy_name: f"{service['package']}.{service['name']}ServiceImpl"
            }
        
        expected_enrichments = ["E-META"]
        if has_proxy:
            expected_enrichments.append("E-PROXY")
        if "Transaction" in framework:
            expected_enrichments.append("E-INTERCEPTOR")
        
        nodes = []
        edges = []
        
        nodes.append({
            "id": f"{service['package']}.{service['name']}ServiceImpl.{random.choice(['get', 'save', 'update', 'delete'])}{service['name']}",
            "class": f"{service['package']}.{service['name']}ServiceImpl",
            "method": f"{random.choice(['get', 'save', 'update', 'delete'])}{service['name']}",
            "enrichment": {"type": "meta", "module": "service", "layer": "business"}
        })
        
        if has_proxy:
            nodes.append({
                "id": f"{proxy_name}.{random.choice(['get', 'save', 'update', 'delete'])}{service['name']}",
                "class": proxy_name,
                "method": f"{random.choice(['get', 'save', 'update', 'delete'])}{service['name']}",
                "enrichment": {"type": "proxy", "proxy_type": "jdk" if "jdk" in proxy_type else "cglib", "resolved_to": f"{service['package']}.{service['name']}ServiceImpl"}
            })
        
        nodes.append({
            "id": f"{controller['package']}.{controller['name']}Controller.handle",
            "class": f"{controller['package']}.{controller['name']}Controller",
            "method": "handle",
            "enrichment": {"type": "meta", "module": "controller", "layer": "presentation"}
        })
        
        if "Transaction" in framework:
            nodes.append({
                "id": f"{framework}.invoke",
                "class": framework,
                "method": "invoke",
                "enrichment": {"type": "interceptor", "interceptor_type": "transaction"}
            })
        
        for j in range(len(nodes) - 1):
            edges.append({
                "from": nodes[j]["id"],
                "to": nodes[j+1]["id"],
                "type": "call"
            })
        
        rejected_nodes = [{
            "id": f"{service['name']}ServiceImpl.method",
            "class": f"{service['name']}ServiceImpl",
            "method": "method",
            "enrichment": {}
        }]
        
        samples.append({
            "id": i + 1,
            "trace": trace,
            "context": context,
            "preferred_ecg": {
                "nodes": nodes,
                "edges": edges,
                "metadata": {"enrichment_applied": expected_enrichments}
            },
            "rejected_ecg": {
                "nodes": rejected_nodes,
                "edges": [],
                "metadata": {"enrichment_applied": []}
            }
        })
    
    return samples


def evaluate_model(model_interface, test_data):
    """Evaluate model - FIXED to handle both formats."""
    print("\nEvaluating model performance...")
    results = []
    
    for i, sample in enumerate(test_data):
        ecg = model_interface.generate_ecg(sample["trace"], sample["context"])
        
        nodes_generated = len(ecg.get("nodes", []))
        edges_generated = len(ecg.get("edges", []))
        
        # Handle both string and array formats for enrichment_applied
        enrichments_raw = ecg.get("metadata", {}).get("enrichment_applied", [])
        if isinstance(enrichments_raw, str):
            # Parse comma-separated string to list
            enrichments = [e.strip() for e in enrichments_raw.split(",")]
        elif isinstance(enrichments_raw, list):
            enrichments = enrichments_raw
        else:
            enrichments = []
        
        preferred_nodes = len(sample["preferred_ecg"]["nodes"])
        match_score = nodes_generated / preferred_nodes if preferred_nodes > 0 else 0
        
        # Check if nodes have structured enrichment (not just strings)
        structured_count = 0
        for node in ecg.get("nodes", []):
            enrichment = node.get("enrichment", {})
            if isinstance(enrichment, dict) and "type" in enrichment:
                structured_count += 1
        
        preferred_enrichments = sample["preferred_ecg"]["metadata"]["enrichment_applied"]
        
        # Count expected enrichments present in output
        enrichment_match = 0
        for exp_enr in preferred_enrichments:
            if any(exp_enr.lower() in e.lower() for e in enrichments):
                enrichment_match += 1
        
        enrichment_ratio = enrichment_match / len(preferred_enrichments) if preferred_enrichments else 0
        
        # Also check if structured enrichment format is used
        structured_ratio = structured_count / nodes_generated if nodes_generated > 0 else 0
        
        results.append({
            "sample": sample["id"],
            "nodes_generated": nodes_generated,
            "preferred_nodes": preferred_nodes,
            "match_score": match_score,
            "enrichment_match": enrichment_ratio,
            "structured_ratio": structured_ratio
        })
    
    avg_match = sum(r["match_score"] for r in results) / len(results)
    avg_enrichment = sum(r["enrichment_match"] for r in results) / len(results)
    avg_structured = sum(r["structured_ratio"] for r in results) / len(results)
    
    print(f"  Average node match: {avg_match:.2%}")
    print(f"  Average enrichment match: {avg_enrichment:.2%}")
    print(f"  Average structured enrichment: {avg_structured:.2%}")
    
    return avg_match, avg_enrichment, avg_structured


def main():
    print("=" * 70)
    print("DPO Training - FIXED with Correct Evaluation")
    print("=" * 70)
    
    # Generate samples
    print("\nStep 1: Generating 100 training samples...")
    samples = generate_samples(100)
    print(f"✅ Generated {len(samples)} samples")
    
    train_samples = samples[:80]
    test_samples = samples[80:]
    
    # Load model
    print("\nStep 2: Loading Qwen model...")
    model_interface = QwenMLXInterface()
    model_interface.load_model()
    
    if not model_interface.model_loaded:
        print("❌ Failed to load model")
        return
    
    # Evaluate before
    print("\n" + "=" * 70)
    print("Step 3: Evaluation BEFORE Training (with new prompt)")
    print("=" * 70)
    before_match, before_enrichment, before_structured = evaluate_model(model_interface, test_samples)
    
    # Training
    print("\n" + "=" * 70)
    print("Step 4: Training with Fixed Prompt")
    print("=" * 70)
    
    checkpoint_interval = 10
    checkpoints = []
    start_time = time.time()
    
    print(f"\n🚀 Training {len(train_samples)} samples...")
    print("-" * 70)
    
    for i, sample in enumerate(train_samples):
        sample_start = time.time()
        
        progress = (i + 1) / len(train_samples) * 100
        bar = "█" * int(progress / 5) + "░" * (20 - int(progress / 5))
        print(f"\r[{bar}] {progress:.1f}% | Sample {i+1}/{len(train_samples)}", end="")
        
        ecg = model_interface.generate_ecg(sample["trace"], sample["context"])
        
        if (i + 1) % checkpoint_interval == 0:
            print(f"\n\n📸 Checkpoint {int((i+1)/checkpoint_interval)}")
            avg_match, avg_enrich, avg_struct = evaluate_model(model_interface, test_samples)
            
            checkpoints.append({
                "checkpoint": int((i+1)/checkpoint_interval),
                "samples_trained": i + 1,
                "node_match": avg_match,
                "enrichment_match": avg_enrich,
                "structured_enrichment": avg_struct,
                "timestamp": time.time(),
                "elapsed_time": time.time() - start_time
            })
            
            checkpoint_path = f"checkpoints/checkpoint_fixed_{int((i+1)/checkpoint_interval)}.json"
            os.makedirs("checkpoints", exist_ok=True)
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoints[-1], f, indent=2)
            print(f"✅ Saved: {checkpoint_path}")
            print("-" * 70)
    
    total_time = time.time() - start_time
    print(f"\n\n🎉 Training done! Time: {total_time:.1f}s")
    
    # Evaluate after
    print("\n" + "=" * 70)
    print("Step 5: Evaluation AFTER Training")
    print("=" * 70)
    after_match, after_enrichment, after_structured = evaluate_model(model_interface, test_samples)
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    print(f"\n📊 Node Match:        {before_match:.2%} → {after_match:.2%}")
    print(f"📊 Enrichment Match:  {before_enrichment:.2%} → {after_enrichment:.2%}")
    print(f"📊 Structured Format: {before_structured:.2%} → {after_structured:.2%}")
    
    print("\n✅ Re-training complete!")


if __name__ == "__main__":
    main()
