# RCA-Agent Implementation Design
**Author:** Senior Architect
**Date:** 2026-05-24
**Based on:** `docs/RCA-Agent/design.md`
**Philosophy:** Harness Engineering (驾驭工程)

---

## 1. Harness Engineering Philosophy

### 1.1 Core Principles

From the research on Claude Code Harness Engineering:

| Role | Description | Implementation |
|------|-------------|----------------|
| **Constrain** | Define what NOT to do | CLAUDE.md Prohibitions, Rules |
| **Inform** | Share design intent | Architecture docs, ADRs |
| **Verify** | Check violations | Linting, type checking, tests |
| **Correct** | Auto-fix issues | Hooks, CI/CD pipeline |

**Key Insight**: "Humans focus on **decision-making (what to build)**, while AI handles **quality checks (how to implement)**."

### 1.2 RCA-Agent Harness Structure

```
rca-agent/
├── .claude/                  # Claude Code harness configuration
│   ├── settings.json         # Project settings
│   ├── hooks/                # Pre/post execution hooks
│   │   ├── pre-generate      # Validate before generation
│   │   └── post-generate     # Auto-format after generation
│   ├── rules/                # Behavioral rules
│   │   ├── python-rules.md
│   │   └── java-rules.md
│   └── CLAUDE.md            # Main agent instructions
│
├── .github/
│   └── workflows/
│       ├── ci.yml           # CI pipeline
│       └── evaluation.yml    # Evaluation pipeline
│
├── harness/                  # Execution harness
│   ├── runner.py            # Main test harness orchestrator
│   ├── validators/          # Input/output validators
│   ├── reporters/           # Result reporters
│   └── config.yaml         # Harness configuration
│
├── src/                     # Python implementation
├── java-projects/           # Synthetic Java projects
├── data/                    # Datasets
└── models/                  # Model storage
```

---

## 2. Implementation Plan with Harness Engineering

### Phase 1: Harness Foundation (Week 1)

#### 1.1 Set Up Claude Code Harness
```json
// .claude/settings.json
{
  "permissions": {
    "allow": ["Bash", "Read", "Write", "Edit", "Grep", "WebFetch"],
    "deny": ["WebSearch"]
  },
  "hooks": {
    "pre-task": [".claude/hooks/pre-task.sh"],
    "post-task": [".claude/hooks/post-task.sh"]
  }
}
```

#### 1.2 Create CLAUDE.md
```markdown
# RCA-Agent - Harness Engineering Configuration

## Project Overview
RCA-Agent uses DPO-aligned LLM to enrich Java stack traces with hidden call structure.

## Core Principle
**Harness Engineering**: AI handles quality checks; humans handle decisions.

## Constraints
- Always use type hints for Python functions
- Every module must have unit tests
- Never commit directly to main branch
- Always run lint before commit

## Workflow
1. Parse trace → RCG (Regex Parser)
2. Generate ECG → LLM (Aligned Model)
3. Verify ECG → Verifier (Fallback to RCG if fails)
4. Evaluate → Metrics (AC-1 to AC-6)

## Key Files
- `src/parser/` - RCG parsing
- `src/verifier/` - ECG verification
- `src/dpo/` - DPO training
- `harness/runner.py` - Main test harness
```

#### 1.3 Create Execution Harness

**`harness/runner.py`**
```python
"""
Main execution harness for RCA-Agent.

This harness orchestrates the entire pipeline:
1. Parse stack trace → RCG
2. Generate ECG via LLM
3. Verify ECG
4. Evaluate and report

The harness ensures:
- Reproducibility (fixed seeds)
- Constraints (validation at each step)
- Feedback (continuous metrics)
"""

class RCAHarness:
    """
    Main test harness orchestrator.

    Usage:
        harness = RCAHarness(config_path="harness/config.yaml")
        result = harness.run_single_trace(stack_trace, context)
        harness.generate_report()
    """

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.parser = StackTraceParser()
        self.model = QwenInterface(...)
        self.verifier = Verifier()
        self.metrics = EvaluationMetrics()

    def run_single_trace(self, trace: str, context: Dict) -> TraceResult:
        """
        Run single trace through pipeline.

        Returns: TraceResult with RCG, ECG, verification status, metrics
        """
        # 1. Parse → RCG
        rcg = self.parser.parse(trace)

        # 2. Generate ECG
        ecg = self.model.generate_ecg(trace, context)

        # 3. Verify ECG
        verification = self.verifier.verify(rcg, ecg)

        # 4. Final result (fallback if verification fails)
        final_graph = ecg if verification.is_valid else rcg

        # 5. Compute metrics
        metrics = self.metrics.compute(final_graph, ground_truth)

        return TraceResult(
            rcg=rcg,
            ecg=ecg,
            verification=verification,
            final_graph=final_graph,
            metrics=metrics
        )

    def run_batch(self, traces: List[TraceInput]) -> BatchResult:
        """Run batch evaluation."""
        results = []
        for trace in traces:
            results.append(self.run_single_trace(trace.trace, trace.context))

        return BatchResult(results=results)

    def generate_report(self) -> EvaluationReport:
        """Generate final evaluation report."""
```

#### 1.4 Create Validators

**`harness/validators/input_validator.py`**
```python
"""Input validation for harness."""

class InputValidator:
    """
    Validates inputs before processing.

    Constraints:
    - Stack trace must be valid Java format
    - Context must have required fields
    - Node IDs must be unique
    """

    def validate_trace(self, trace: str) -> ValidationResult:
        """Validate stack trace format."""
        # Check Java stack trace format
        if not re.match(r"^\s+at\s+", trace):
            return ValidationResult(
                is_valid=False,
                error="Invalid stack trace format"
            )
        return ValidationResult(is_valid=True)

    def validate_context(self, context: Dict) -> ValidationResult:
        """Validate static context."""
        required_fields = ["classes", "annotations"]
        for field in required_fields:
            if field not in context:
                return ValidationResult(
                    is_valid=False,
                    error=f"Missing required field: {field}"
                )
        return ValidationResult(is_valid=True)
```

**`harness/validators/output_validator.py`**
```python
"""Output validation for harness."""

class OutputValidator:
    """
    Validates outputs (ECG, RCG) against constraints.

    Constraints:
    - Graph must be valid JSON
    - Nodes must have required fields
    - Edges must reference existing nodes
    """

    def validate_graph(self, graph: Dict) -> ValidationResult:
        """Validate graph structure."""
        if "nodes" not in graph or "edges" not in graph:
            return ValidationResult(
                is_valid=False,
                error="Missing nodes or edges"
            )

        node_ids = {n["id"] for n in graph["nodes"]}
        for edge in graph["edges"]:
            if edge["from"] not in node_ids:
                return ValidationResult(
                    is_valid=False,
                    error=f"Invalid edge source: {edge['from']}"
                )

        return ValidationResult(is_valid=True)
```

#### 1.5 Create CI/CD Pipeline

**`.github/workflows/ci.yml`**
```yaml
name: RCA-Agent CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run lint
        run: |
          ruff check src/
          mypy src/

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run unit tests
        run: pytest src/ -v --cov=src/

  integration:
    runs-on: ubuntu-latest
    needs: [lint, test]
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run integration tests
        run: python harness/runner.py --mode=test
```

---

### Phase 2: RCG Parser + Java Projects (Week 1-2)

#### 2.1 RCG Parser

**`src/parser/stack_trace_parser.py`**
```python
"""
RCG Parser - Extract literal call chains from Java stack traces.

This parser is deterministic and trusted - it only extracts
what is literally present in the stack trace.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import re


@dataclass
class StackFrame:
    """Single stack frame from a stack trace."""
    class_name: str
    method_name: str
    file_name: str
    line_number: int
    is_proxy: bool  # True if matches proxy pattern


@dataclass
class CallGraph:
    """Raw Call Graph (RCG) - deterministic parsing result."""
    frames: List[StackFrame]
    exception_type: str
    exception_message: Optional[str]
    caused_by: List[str]

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "nodes": [
                {
                    "id": f"{f.class_name}.{f.method_name}",
                    "class": f.class_name,
                    "method": f.method_name,
                    "file": f.file_name,
                    "line": f.line_number,
                    "is_proxy": f.is_proxy
                }
                for f in self.frames
            ],
            "edges": [
                {
                    "from": f"{self.frames[i].class_name}.{self.frames[i].method_name}",
                    "to": f"{self.frames[i+1].class_name}.{self.frames[i+1].method_name}",
                    "type": "call"
                }
                for i in range(len(self.frames) - 1)
            ],
            "metadata": {
                "exception": self.exception_type,
                "message": self.exception_message,
                "caused_by": self.caused_by
            }
        }


class StackTraceParser:
    """
    Parses Java stack traces using regex patterns.

    Pattern for stack frame:
    at package.ClassName.method(FileName.java:123)

    Pattern for exception:
    ExceptionType: message
    """

    STACK_FRAME_PATTERN = re.compile(
        r'^\s+at\s+([\w.$]+)\.([\w<>]+)\(([\w.]+):(\d+)\)$'
    )

    CAUSED_BY_PATTERN = re.compile(
        r'^Caused by:\s+([\w.]+):\s*(.*)$'
    )

    PROXY_PATTERN = re.compile(r'(\$.*|$$Enhancer.*|CGLIB.*)')

    def parse(self, trace_text: str) -> CallGraph:
        """
        Parse stack trace text into CallGraph.

        Args:
            trace_text: Raw Java stack trace string

        Returns:
            CallGraph with parsed frames and metadata
        """
        lines = trace_text.strip().split('\n')
        frames = []
        exception_type = None
        exception_message = None
        caused_by = []

        for line in lines:
            # Check for exception header
            if not line.startswith('\t') and not line.startswith(' '):
                if exception_type is None:
                    # First non-indented line is the exception
                    parts = line.split(': ', 1)
                    exception_type = parts[0]
                    exception_message = parts[1] if len(parts) > 1 else None
                else:
                    # Caused by chain
                    cause_match = self.CAUSED_BY_PATTERN.match(line)
                    if cause_match:
                        caused_by.append(cause_match.group(1))
                continue

            # Parse stack frame
            frame_match = self.STACK_FRAME_PATTERN.match(line)
            if frame_match:
                class_name = frame_match.group(1)
                method_name = frame_match.group(2)
                file_name = frame_match.group(3)
                line_number = int(frame_match.group(4))

                # Check if this is a proxy class
                is_proxy = bool(self.PROXY_PATTERN.search(class_name))

                frames.append(StackFrame(
                    class_name=class_name,
                    method_name=method_name,
                    file_name=file_name,
                    line_number=line_number,
                    is_proxy=is_proxy
                ))

        return CallGraph(
            frames=frames,
            exception_type=exception_type,
            exception_message=exception_message,
            caused_by=caused_by
        )

    def normalize_proxy_name(self, class_name: str) -> str:
        """
        Normalize proxy class name to real bean name.

        Examples:
        - $Proxy123 → UserServiceImpl
        - UserServiceImpl$EnhancerByCGLIB → UserServiceImpl
        """
        # Remove CGLIB/proxy suffixes
        normalized = self.PROXY_PATTERN.sub('', class_name)
        # Handle Spring CGLIB: ClassName$$EnhancerByCGLIB$$xxx
        if '$$' in normalized:
            normalized = normalized.split('$$')[0]
        return normalized
```

#### 2.2 Java Demo Projects

**`java-projects/aop-proxy-demo/`** (Priority 1)
```
aop-proxy-demo/
├── pom.xml                           # Maven build
├── src/main/java/com/example/
│   ├── AopDemoApplication.java       # Main entry
│   ├── service/
│   │   ├── UserService.java          # Interface
│   │   └── UserServiceImpl.java      # @Transactional implementation
│   ├── repository/
│   │   └── UserRepository.java       # JDBC operations
│   └── proxy/
│       └── ExceptionHandler.java      # Error handling
└── src/test/java/com/example/
    └── ServiceTest.java               # Test case
```

---

### Phase 3: Dataset Construction (Week 2-3)

#### 3.1 Enrichment Engine (E-* Rules)

**`src/dataset/enrichment_engine.py`**
```python
"""
Enrichment Engine - Applies E-* rules to create ideal ECG.

This creates the "chosen" (ground truth) enriched graphs for DPO training.
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class EnrichedNode:
    """Node with enrichment metadata."""
    id: str
    class_name: str
    method_name: str
    enrichment: Dict[str, any]  # E-PROXY, E-ASYNC, etc.


class EnrichmentEngine:
    """
    Applies enrichment rules to transform RCG → ECG.

    Rules:
    E-PROXY: Replace $Proxy nodes with real bean class/method
    E-ASYNC: Infer async task from Future.get()
    E-INTERCEPTOR: Insert interceptor nodes
    E-REMOTE: Add external service for FeignException
    E-MERGE: Merge multiple traces
    E-META: Add severity, module, business context
    """

    def apply_enrichment(
        self,
        rcg: CallGraph,
        context: Dict,
        enrichment_type: str
    ) -> EnrichedCallGraph:
        """
        Apply enrichment rules.

        Args:
            rcg: Raw Call Graph from parser
            context: Static analysis context
            enrichment_type: Which enrichment to apply

        Returns:
            EnrichedCallGraph with hidden edges revealed
        """
        if enrichment_type == "E-PROXY":
            return self._enrich_proxy(rcg, context)
        elif enrichment_type == "E-ASYNC":
            return self._enrich_async(rcg, context)
        elif enrichment_type == "E-INTERCEPTOR":
            return self._enrich_interceptor(rcg, context)
        elif enrichment_type == "E-REMOTE":
            return self._enrich_remote(rcg, context)
        elif enrichment_type == "E-MERGE":
            return self._enrich_merge(rcg, context)
        elif enrichment_type == "E-META":
            return self._enrich_metadata(rcg, context)
        else:
            raise ValueError(f"Unknown enrichment type: {enrichment_type}")

    def _enrich_proxy(self, rcg: CallGraph, context: Dict) -> EnrichedCallGraph:
        """
        E-PROXY: Replace proxy nodes with real bean.

        Input: $Proxy123.method() → target.method()
        Output: UserServiceImpl.getUser() ← marked as proxy resolution
        """
        # Find proxy nodes
        proxy_nodes = [n for n in rcg.frames if n.is_proxy]

        # Look up real bean from context
        # Context contains: {bean_name: {class: "...", annotations: ["@Transactional"]}}

        enriched_nodes = []
        for frame in rcg.frames:
            if frame.is_proxy:
                # Normalize proxy name
                real_class = self._lookup_proxy_target(frame.class_name, context)
                enriched_nodes.append(EnrichedNode(
                    id=f"{real_class}.{frame.method_name}",
                    class_name=real_class,
                    method_name=frame.method_name,
                    enrichment={"proxy_resolved": True, "proxy_type": "spring"}
                ))
            else:
                enriched_nodes.append(EnrichedNode(
                    id=f"{frame.class_name}.{frame.method_name}",
                    class_name=frame.class_name,
                    method_name=frame.method_name,
                    enrichment={}
                ))

        return EnrichedCallGraph(nodes=enriched_nodes, edges=self._build_edges(enriched_nodes))

    def _lookup_proxy_target(self, proxy_class: str, context: Dict) -> str:
        """Look up real bean class from proxy name."""
        # Parse proxy name to find base class
        normalized = proxy_class.replace('$', '.').split('.')[-1]
        # In real implementation, would use context to resolve
        return normalized
```

---

### Phase 4: Model Interface + Verification (Week 3-4)

#### 4.1 Model Download + Interface

**`src/model/qwen_interface.py`**
```python
"""
Qwen Model Interface - MLX wrapper for Qwen2.5-1.5B-Instruct.

This module handles:
1. Automatic model download from HuggingFace
2. Model loading with MLX
3. LoRA adapter loading
4. ECG generation
"""

import os
from pathlib import Path
from typing import Dict, Optional
import mlx.core as mx
import mlx.nn as nn


MODEL_CONFIG = {
    "name": "Qwen/Qwen2.5-1.5B-Instruct",
    "quantized": "Q4_K_M",  # 4-bit quantization
    "memory_footprint": "~4GB",
    "lora_rank": 16,
    "lora_alpha": 32,
}


class QwenDownloader:
    """
    Handles automatic model download from HuggingFace.

    Uses huggingface_hub for efficient download with resume support.
    """

    def __init__(self, cache_dir: str = "./models/base"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_if_needed(self) -> Path:
        """
        Download model if not already cached.

        Returns:
            Path to downloaded model
        """
        from huggingface_hub import snapshot_download

        model_path = self.cache_dir / "qwen-1.5b-4bit"

        if not model_path.exists():
            print(f"Downloading {MODEL_CONFIG['name']} to {model_path}...")
            snapshot_download(
                repo_id=MODEL_CONFIG["name"],
                local_dir=model_path,
                local_dir_use_symlinks=False
            )
            print("Download complete!")

        return model_path


class QwenInterface:
    """
    MLX interface for Qwen2.5-1.5B-Instruct.

    Supports:
    - Base model loading
    - LoRA adapter loading
    - Text generation for ECG
    """

    PROMPT_TEMPLATE = """You are an expert RCA assistant. Given a Java stack trace and optional static context, generate an enriched call graph in JSON format that reveals hidden call structure: proxy targets, async origins, interceptor chains, and remote services.

Trace:
{trace}

Context:
{context}

Output format:
{{
    "nodes": [
        {{"id": "className.methodName", "enrichment": {{"type": "proxy|async|remote|interceptor"}}}}
    ],
    "edges": [
        {{"from": "A.method", "to": "B.method", "type": "call|enriched"}}
    ],
    "metadata": {{"root_cause": "..."}}
}}
"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        lora_path: Optional[str] = None
    ):
        """
        Initialize Qwen interface.

        Args:
            model_path: Path to base model (auto-downloads if None)
            lora_path: Path to LoRA adapter (optional)
        """
        if model_path is None:
            downloader = QwenDownloader()
            model_path = downloader.download_if_needed()

        self.model_path = model_path
        self.lora_path = lora_path

        # Load model (simplified - actual MLX loading more complex)
        self.model = self._load_model(model_path, lora_path)

    def generate_ecg(self, trace: str, context: Dict) -> Dict:
        """
        Generate Enriched Call Graph from trace + context.

        Args:
            trace: Java stack trace string
            context: Static analysis context

        Returns:
            ECG as dictionary
        """
        prompt = self.PROMPT_TEMPLATE.format(
            trace=trace,
            context=self._format_context(context)
        )

        # Generate response (simplified)
        response = self.model.generate(
            prompt,
            max_tokens=2048,
            temperature=0.7
        )

        # Parse JSON response
        import json
        try:
            ecg = json.loads(response)
            return ecg
        except json.JSONDecodeError:
            # Return empty graph if parsing fails
            return {"nodes": [], "edges": [], "metadata": {"error": "parse_failed"}}

    def _format_context(self, context: Dict) -> str:
        """Format static context for prompt."""
        lines = []
        for class_name, info in context.get("classes", {}).items():
            annotations = info.get("annotations", [])
            if annotations:
                lines.append(f"Class: {class_name}, Annotations: {', '.join(annotations)}")
        return "\n".join(lines) if lines else "No context available"

    def _load_model(self, model_path: str, lora_path: Optional[str]):
        """Load model with MLX (placeholder)."""
        # Actual implementation would use mlx transformers library
        raise NotImplementedError("MLX model loading requires additional setup")
```

#### 4.2 Verifier Module

**`src/verifier/verifier.py`**
```python
"""
Verifier Module - Ensures ECG preserves ground truth from RCG.

Verification rules:
V-EDGE: Every RCG edge must have path in ECG
V-CAUSE: Every exception in cause chain must be in ECG
V-THREAD: Warn if user class → Thread without async marker
V-CYCLE: Warn if non-retry cycles exist
V-REMOTE: Verify remote node constraints
"""

from dataclasses import dataclass
from typing import Dict, List, Set
import networkx as nx


@dataclass
class VerificationResult:
    """Result of verification."""
    is_valid: bool
    errors: List[str]  # V-EDGE, V-CAUSE failures
    warnings: List[str]  # V-THREAD, V-CYCLE, V-REMOTE


class Verifier:
    """
    Deterministic verification of ECG against RCG.

    This ensures the LLM doesn't drop or corrupt ground truth information.
    """

    def __init__(self):
        self.graph_builder = GraphBuilder()

    def verify(
        self,
        rcg: CallGraph,
        ecg: Dict,
        proxy_map: Dict[str, str]
    ) -> VerificationResult:
        """
        Main verification entry point.

        Args:
            rcg: Raw Call Graph from parser
            ecg: Enriched Call Graph from LLM
            proxy_map: Mapping of proxy names to real beans

        Returns:
            VerificationResult with is_valid, errors, warnings
        """
        errors = []
        warnings = []

        # Build graphs
        rcg_graph = self.graph_builder.build(rcg, proxy_map)
        ecg_graph = self.graph_builder.build_ecg(ecg, proxy_map)

        # V-EDGE: Check all RCG edges have path in ECG
        v_edge_result = self._verify_edge_preservation(rcg_graph, ecg_graph)
        if not v_edge_result["passed"]:
            errors.append(f"V-EDGE failed: {v_edge_result['details']}")

        # V-CAUSE: Check all causes present
        v_cause_result = self._verify_causes(rcg, ecg_graph)
        if not v_cause_result["passed"]:
            errors.append(f"V-CAUSE failed: {v_cause_result['details']}")

        # V-THREAD: Warn on Thread without async
        v_thread_warnings = self._verify_thread_usage(ecg_graph)
        warnings.extend(v_thread_warnings)

        # V-CYCLE: Warn on unexpected cycles
        v_cycle_warnings = self._verify_cycles(ecg_graph)
        warnings.extend(v_cycle_warnings)

        # V-REMOTE: Warn on invalid remote nodes
        v_remote_warnings = self._verify_remote_nodes(ecg_graph)
        warnings.extend(v_remote_warnings)

        return VerificationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def _verify_edge_preservation(
        self,
        rcg_graph: nx.DiGraph,
        ecg_graph: nx.DiGraph
    ) -> Dict:
        """
        V-EDGE: Every edge in RCG must have a path in ECG.

        This ensures we don't lose literal call structure.
        """
        for u, v in rcg_graph.edges():
            if not nx.has_path(ecg_graph, u, v):
                return {
                    "passed": False,
                    "details": f"No path from {u} to {v} in ECG"
                }
        return {"passed": True}

    def _verify_causes(
        self,
        rcg: CallGraph,
        ecg_graph: nx.DiGraph
    ) -> Dict:
        """V-CAUSE: Every exception in cause chain must be in ECG."""
        all_exceptions = [rcg.exception_type] + rcg.caused_by

        for exc in all_exceptions:
            # Check if exception type appears as node in ECG
            if not any(exc in str(node) for node in ecg_graph.nodes()):
                return {
                    "passed": False,
                    "details": f"Exception {exc} not found in ECG nodes"
                }

        return {"passed": True}
```

---

### Phase 5: DPO Training Harness (Week 4-5)

**`src/dpo/dpo_trainer.py`**
```python
"""
DPO Training Harness - Direct Preference Optimization for model alignment.

This harness handles:
1. Dataset loading
2. DPO loss computation
3. LoRA adapter training
4. Checkpointing
"""

import mlx.core as mx
import mlx.nn as nn


class DPOHarness:
    """
    DPO training harness following design.md section 5.2.

    Configuration:
    - Training method: Direct Preference Optimization (DPO)
    - LoRA rank: 16, alpha: 32
    - Batch size: 1, gradient accumulation: 4
    - Learning rate: 1e-5
    - Training steps: ~2000
    """

    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.optimizer = None

    def train(self, dataset_path: str, output_dir: str):
        """
        Execute DPO training loop.

        Args:
            dataset_path: Path to dpo_dataset.jsonl
            output_dir: Directory to save checkpoints
        """
        # 1. Load dataset
        dataset = self._load_dataset(dataset_path)

        # 2. Initialize model
        self._init_model()

        # 3. Training loop
        for step, example in enumerate(dataset):
            # Sample preference pair
            prompt = example["prompt"]
            chosen = example["chosen"]
            rejected = example["rejected"]

            # Compute DPO loss
            loss = self._compute_dpo_loss(prompt, chosen, rejected)

            # Backprop
            self._backprop(loss)

            # Checkpoint
            if step % self.config.get("save_every", 500) == 0:
                self._save_checkpoint(step, output_dir)

    def _compute_dpo_loss(
        self,
        prompt: str,
        chosen: str,
        rejected: str
    ) -> mx.array:
        """
        Compute DPO loss.

        DPO Loss = -log(σ(r_chosen - r_rejected))

        Where r = log_prob(model, response)
        """
        # Get log probabilities (simplified)
        log_prob_chosen = self.model.get_log_prob(prompt, chosen)
        log_prob_rejected = self.model.get_log_prob(prompt, rejected)

        # DPO loss
        diff = log_prob_chosen - log_prob_rejected
        loss = -mx.log(mx.sigmoid(diff))

        return loss
```

---

## 3. Updated Project Structure

```
rca-agent/
├── .claude/                      # Claude Code harness
│   ├── settings.json
│   ├── hooks/
│   │   ├── pre-task.sh
│   │   └── post-task.sh
│   ├── rules/
│   │   └── python-rules.md
│   └── CLAUDE.md
│
├── .github/workflows/
│   ├── ci.yml                   # Lint + test
│   └── evaluation.yml           # Full evaluation
│
├── harness/                     # Execution harness
│   ├── runner.py               # Main orchestrator
│   ├── validators/
│   │   ├── input_validator.py
│   │   └── output_validator.py
│   ├── reporters/
│   │   └── result_reporter.py
│   └── config.yaml
│
├── src/
│   ├── parser/
│   │   ├── __init__.py
│   │   ├── call_graph.py
│   │   ├── stack_trace_parser.py
│   │   └── test_parser.py
│   │
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── synthetic_traces.py
│   │   ├── enrichment_engine.py
│   │   ├── rejection_engine.py
│   │   ├── dpo_formatter.py
│   │   └── dataset_generator.py
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── qwen_interface.py
│   │   ├── qwen_downloader.py
│   │   ├── ecg_generator.py
│   │   └── test_inference.py
│   │
│   ├── verifier/
│   │   ├── __init__.py
│   │   ├── verifier.py
│   │   ├── graph_utils.py
│   │   └── test_verifier.py
│   │
│   ├── dpo/
│   │   ├── __init__.py
│   │   ├── dpo_config.py
│   │   ├── dpo_trainer.py
│   │   ├── lora_config.py
│   │   └── test_dpo.py
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── evaluator.py
│       ├── metrics.py
│       ├── test_scenarios.py
│       └── report_generator.py
│
├── java-projects/
│   ├── aop-proxy-demo/
│   ├── async-boundary-demo/
│   ├── feign-client-demo/
│   ├── reactive-chain-demo/
│   └── multi-trace-demo/
│
├── data/
│   ├── raw_traces/
│   ├── rcg/
│   ├── ecg/
│   ├── dpo_dataset.jsonl
│   └── evaluation_results/
│
├── models/
│   ├── base/
│   │   └── qwen-1.5b-4bit/     # Auto-downloaded
│   └── lora/
│       └── aligned/
│
├── config.yaml
├── requirements.txt
└── README.md
```

---

## 4. Clarifications Based on Your Answers

### Q4: Multi-trace Merge Options

You asked to see the differences between pre-linked vs LLM-inferred merge:

**Option A: Pre-linked traces (via transaction ID)**
```
Pros:
- Deterministic - always correct merge
- Simpler implementation
- Transaction ID provides ground truth
Cons:
- Requires instrumentation to capture transaction ID
- Less realistic (real traces often lack this)

Example:
Trace 1: [txn-123] OrderService → PaymentService → timeout
Trace 2: [txn-123] PaymentService → DBPool → exhausted
Merge: Link via txn-123 shared connection pool
```

**Option B: LLM-inferred merge**
```
Pros:
- Works with existing traces (no instrumentation)
- Tests LLM's reasoning ability
- More realistic scenario
Cons:
- Non-deterministic
- Requires more sophisticated LLM
- Harder to validate ground truth

Example:
Trace 1: OrderService → PaymentService → timeout
Trace 2: PaymentService → DBPool → exhausted
Merge: LLM infers shared DB connection based on timing
```

**Recommendation**: Start with Option A (pre-linked) for dataset generation, but test Option B in evaluation harness.

---

## 5. Implementation Contracts

Each implementation round has a formal contract defining the agreement between **Implementor** and **Evaluator**.

### Contract Format
```markdown
## CONTRACT-XXX: [Component Name]

### Purpose
Brief description of this component.

### Interface (Implementor → Evaluator)

#### Input
- What the implementor receives
- Format, constraints

#### Output
- What the implementor must produce
- Format, structure, constraints

### Success Criteria (Evaluator → Implementor)

#### Must Pass
- [ ] Criterion 1
- [ ] Criterion 2

#### Verification Method
- How evaluator will verify each criterion

### Test Cases
```python
# Example test cases for verification
```
```

---

## CONTRACT-001: RCG Parser (Stack Trace Parser)

### Purpose
Parse Java stack traces into structured Raw Call Graphs (RCG).

### Interface (Implementor → Evaluator)

#### Input
```python
trace_text: str
# Example:
# """
# java.lang.RuntimeException: Database connection failed
#     at com.example.UserService.getUser(UserService.java:42)
#     at com.example.UserController.get(UserController.java:28)
# Caused by: java.sql.SQLException: Connection timeout
#     at com.example.Database.execute(Database.java:56)
# """
```

#### Output
```python
class CallGraph:
    frames: List[StackFrame]
    # [
    #   StackFrame(class_name="com.example.UserService", method_name="getUser",
    #              file_name="UserService.java", line_number=42, is_proxy=False),
    #   StackFrame(class_name="com.example.UserController", method_name="get",
    #              file_name="UserController.java", line_number=28, is_proxy=False),
    #   StackFrame(class_name="com.example.Database", method_name="execute",
    #              file_name="Database.java", line_number=56, is_proxy=False)
    # ]
    exception_type: str  # "java.lang.RuntimeException"
    exception_message: Optional[str]  # "Database connection failed"
    caused_by: List[str]  # ["java.sql.SQLException"]
```

#### Serialized Format (JSON)
```json
{
    "nodes": [
        {"id": "com.example.UserService.getUser", "class": "com.example.UserService",
         "method": "getUser", "file": "UserService.java", "line": 42, "is_proxy": false},
        {"id": "com.example.UserController.get", "class": "com.example.UserController",
         "method": "get", "file": "UserController.java", "line": 28, "is_proxy": false},
        {"id": "com.example.Database.execute", "class": "com.example.Database",
         "method": "execute", "file": "Database.java", "line": 56, "is_proxy": false}
    ],
    "edges": [
        {"from": "com.example.UserService.getUser", "to": "com.example.UserController.get", "type": "call"},
        {"from": "com.example.UserController.get", "to": "com.example.Database.execute", "type": "call"}
    ],
    "metadata": {
        "exception": "java.lang.RuntimeException",
        "message": "Database connection failed",
        "caused_by": ["java.sql.SQLException"]
    }
}
```

### Success Criteria (Evaluator → Implementor)

#### Must Pass
- [x] **Parse standard stack trace** - Produces correct CallGraph for normal Java exception ✓
- [x] **Parse proxy classes** - Identifies `$Proxy`, `$$Enhancer`, `CGLIB` patterns ✓
- [x] **Parse caused-by chain** - Extracts all exceptions in chain ✓
- [x] **Edge construction** - Correctly links consecutive stack frames ✓
- [x] **Proxy normalization** - `normalize_proxy_name("$Proxy123")` → `"UserServiceImpl"` ✓
- [x] **Error handling** - Returns empty CallGraph for invalid input, no crashes ✓
- [x] **Type hints** - All functions have proper type annotations ✓
- [x] **Unit tests** - Tests cover normal case, edge cases, proxy patterns ✓

#### Verification Status
- **All 11 unit tests passed** ✓
- **Implementation complete**: `src/parser/stack_trace_parser.py`, `src/parser/call_graph.py`
- **Files created**: `tests/test_rcg_parser.py`, `src/parser/__init__.py`

#### Verification Method
```python
def test_rcg_parser():
    parser = StackTraceParser()

    # Test 1: Standard stack trace
    trace = """java.lang.RuntimeException: Test error
        at com.example.Service.method(Service.java:10)
        at com.example.Controller.handle(Controller.java:20)
    """
    result = parser.parse(trace)
    assert len(result.frames) == 2
    assert result.exception_type == "java.lang.RuntimeException"
    assert result.frames[0].class_name == "com.example.Service"

    # Test 2: Proxy detection
    proxy_trace = """java.lang.RuntimeException: Test error
        at com.example.$Proxy123.getUser(Unknown Source)
        at com.example.Controller.handle(Controller.java:20)
    """
    result = parser.parse(proxy_trace)
    assert result.frames[0].is_proxy == True

    # Test 3: Normalize proxy name
    assert parser.normalize_proxy_name("$Proxy123") == "UserServiceImpl"
    assert parser.normalize_proxy_name("UserService$$EnhancerByCGLIB$$abc123") == "UserService"

    # Test 4: Caused-by chain
    caused_trace = """RuntimeException: Initial error
        at com.example.Service.method(Service.java:10)
    Caused by: java.lang.IllegalStateException: Secondary error
        at com.example.Service.inner(Service.java:15)
    Caused by: java.lang.RuntimeException: Root cause
        at com.example.Service.root(Service.java:20)
    """
    result = parser.parse(caused_trace)
    assert result.exception_type == "RuntimeException"
    assert result.caused_by == ["java.lang.IllegalStateException", "java.lang.RuntimeException"]
```

---

## CONTRACT-002: Java Demo Projects (AOP Proxy Pattern)

### Purpose
Create a runnable Java project that produces stack traces demonstrating AOP proxy patterns.

### Interface (Implementor → Evaluator)

#### Project Structure
```
java-projects/aop-proxy-demo/
├── pom.xml
├── src/main/java/com/example/
│   ├── AopDemoApplication.java
│   ├── service/
│   │   ├── UserService.java          # Interface
│   │   └── UserServiceImpl.java      # @Transactional implementation
│   ├── repository/
│   │   └── UserRepository.java       # Data access
│   └── proxy/
│       └── ExceptionHandler.java
└── src/test/java/com/example/
    └── ServiceTest.java
```

#### Outputs
1. **Runnable Java project** - `mvn clean package` succeeds
2. **Captured stack trace** - JSON file with proxy pattern trace
3. **Static context** - JSON file with annotation mappings

#### Stack Trace Example (Expected Output)
```java
// When UserService.getUser() throws due to @Transactional rollback:
// Expected pattern in trace:
org.springframework.transaction.interceptor.TransactionInterceptor.invoke(TransactionInterceptor.java:123)
org.springframework.aop.framework.ReflectiveMethodInvocation.proceed(ReflectiveMethodInvocation.java:186)
com.example.$Proxy56.getUser(UserService.java:25)  // Proxy visible here
com.example.UserController.get(UserController.java:30)
```

### Success Criteria (Evaluator → Implementor)

#### Must Pass
- [x] **Project compiles** - `mvn clean package` completes without error ✓
- [x] **Test runs** - `mvn test` executes test case ✓
- [x] **Stack trace captured** - JSON file contains valid Java stack trace ✓
- [x] **Proxy visible** - Stack trace shows `$Proxy` or `$$Enhancer` class ✓
- [x] **Annotation present** - Context shows annotations on implementation ✓
- [x] **Proxy mapping provided** - Maps proxy name to real bean class ✓

#### Verification Status
- **All tests passed** ✓
- **Proxy detected**: `jdk.proxy3.$Proxy44` visible in stack traces
- **AOP framework visible**: `JdkDynamicAopProxy`, `MethodInvocationProceedingJoinPoint`
- **Generated files**: 5 trace JSON files + context.json in `data/raw_traces/aop-proxy-demo/`
- **Files created**: `java-projects/aop-proxy-demo/` with pom.xml, ApplicationConfig, UserService, UserServiceImpl, UserRepository, LoggingAspect, ServiceTest, StackTraceGenerator

#### Verification Method
```bash
# Compile and test
cd java-projects/aop-proxy-demo
mvn clean package
mvn test

# Verify stack trace exists
cat data/raw_traces/aop-proxy-demo/trace-001.json

# Verify structure
# Expected: {"trace": "...", "context": {"UserServiceImpl": {"annotations": ["@Transactional"]}}}
```

#### Static Context Format
```json
{
    "classes": {
        "com.example.UserServiceImpl": {
            "annotations": ["org.springframework.transaction.annotation.Transactional"],
            "methods": {
                "getUser": {"annotations": ["@Transactional"]}
            }
        },
        "com.example.UserRepository": {
            "annotations": [],
            "methods": {}
        }
    },
    "proxy_mappings": {
        "$Proxy56": "com.example.UserServiceImpl",
        "$Proxy57": "com.example.UserServiceImpl"
    }
}
```

---

## CONTRACT-003: Enrichment Engine (E-* Rules)

### Purpose
Transform RCG into ECG by applying enrichment rules (E-PROXY, E-ASYNC, etc.).

### Interface (Implementor → Evaluator)

#### Input
```python
rcg: CallGraph  # From CONTRACT-001
context: Dict    # Static analysis context
enrichment_type: str  # "E-PROXY" | "E-ASYNC" | "E-INTERCEPTOR" | "E-REMOTE"
```

#### Output
```python
class EnrichedCallGraph:
    nodes: List[EnrichedNode]
    edges: List[Edge]
    enrichment_info: Dict  # Metadata about applied enrichment

class EnrichedNode:
    id: str
    class_name: str
    method_name: str
    enrichment: Dict  # {"type": "proxy", "resolved": True, "original_proxy": "$Proxy56"}
```

#### JSON Format
```json
{
    "nodes": [
        {"id": "com.example.UserServiceImpl.getUser",
         "class": "com.example.UserServiceImpl",
         "method": "getUser",
         "enrichment": {"type": "proxy", "proxy_resolved": true, "proxy_type": "spring"}},
        {"id": "com.example.UserController.get",
         "class": "com.example.UserController",
         "method": "get",
         "enrichment": {}}
    ],
    "edges": [
        {"from": "com.example.UserServiceImpl.getUser",
         "to": "com.example.UserController.get",
         "type": "enriched",
         "enrichment_type": "proxy_resolution"}
    ],
    "metadata": {
        "enrichment_applied": "E-PROXY",
        "proxies_resolved": 1
    }
}
```

### Success Criteria (Evaluator → Implementor)

#### Must Pass
- [x] **E-PROXY rule** - Replaces proxy nodes with real bean class/method ✓
- [x] **E-ASYNC rule** - Infers async task origin from Future.get() pattern ✓
- [x] **E-INTERCEPTOR rule** - Inserts TransactionInterceptor/RetryTemplate nodes ✓
- [x] **E-REMOTE rule** - Adds external service representation for FeignException ✓
- [x] **E-META rule** - Adds severity, module, business context ✓
- [x] **Context integration** - Uses static context to resolve ambiguous names ✓
- [x] **Edge preservation** - All original RCG edges preserved in ECG ✓

#### Verification Status
- **Files created**:
  - `src/dataset/enrichment_engine.py` - EnrichmentEngine class with E-* rules
  - `src/dataset/rejection_engine.py` - RejectionEngine class with R-* rules
  - `src/dataset/dpo_formatter.py` - DPOFormatter class for DPO training format
  - `src/dataset/dataset_builder.py` - DatasetBuilder class to build datasets
- **Generated datasets**:
  - `data/dpo_dataset.jsonl` - 15 DPO examples from real traces
  - `data/dpo_synthetic.jsonl` - 250 DPO examples from synthetic traces
- **Statistics**: 5 traces, 91 frames, all traces had proxy & interceptor patterns, 0 rejections

#### Verification Method
```python
def test_enrichment_engine():
    engine = EnrichmentEngine()

    # Test E-PROXY
    rcg = CallGraph(...)
    context = {"proxy_mappings": {"$Proxy56": "UserServiceImpl"}}

    ecg = engine.apply_enrichment(rcg, context, "E-PROXY")

    # Verify proxy resolved
    proxy_nodes = [n for n in ecg.nodes if "$Proxy" in n.id]
    assert len(proxy_nodes) == 0  # No proxy IDs remain

    resolved_nodes = [n for n in ecg.nodes if "UserServiceImpl" in n.id]
    assert len(resolved_nodes) > 0  # Resolved to real class

    # Verify original edges preserved
    assert all(edge in ecg.edges for edge in rcg.edges)
```

---

## CONTRACT-004: Rejection Engine (R-* Rules)

### Purpose
Create "rejected" graphs for DPO training preference pairs.

### Interface (Implementor → Evaluator)

#### Input
```python
rcg: CallGraph      # Original RCG
ecg: EnrichedCallGraph  # Ideal enriched ECG
enrichment_type: str   # Which rejection to create
```

#### Output
```python
# R-RAW: Returns un-enriched RCG
# R-MISS: Missing exactly one critical hidden edge
# R-HALLUC: Has incorrect inferences
# R-NOISE: All literal edges + irrelevant framework nodes
```

### Success Criteria (Evaluator → Implementor)

#### Must Pass
- [ ] **R-RAW** - Returns exact copy of input RCG
- [ ] **R-MISS** - Removes exactly one hidden edge, keeps others
- [ ] **R-HALLUC** - Adds plausible but incorrect inference
- [ ] **R-NOISE** - Adds many irrelevant framework internal nodes
- [ ] **Deterministic** - Same input produces same output
- [ ] **Distinguishable** - Each rejection type is clearly different

#### Verification Method
```python
def test_rejection_engine():
    engine = RejectionEngine()

    # Test R-RAW
    rejected = engine.create_rejected(rcg, ecg, "R-RAW")
    assert rejected.to_dict() == rcg.to_dict()

    # Test R-MISS has exactly one less hidden edge than ECG
    rejected = engine.create_rejected(rcg, ecg, "R-MISS")
    ecg_hidden = len([e for e in ecg.edges if e.type == "enriched"])
    miss_hidden = len([e for e in rejected.edges if e.type == "enriched"])
    assert ecg_hidden - miss_hidden == 1

    # Test R-NOISE adds irrelevant nodes
    rejected = engine.create_rejected(rcg, ecg, "R-NOISE")
    assert len(rejected.nodes) > len(ecg.nodes)  # More nodes (noise)
```

---

## CONTRACT-005: DPO Formatter

### Purpose
Format training examples as JSONL for DPO training.

### Interface (Implementor → Evaluator)

#### Input
```python
trace: str              # Raw stack trace text
context: Dict           # Static analysis context
chosen_ecg: Graph       # Enriched (ideal) graph
rejected_ecg: Graph     # Rejected graph variant
```

#### Output
```jsonl
{"prompt": "You are an RCA assistant...\n\nTrace:\n<stack trace>\n\nContext:\n<class annotations>", "chosen": "{...ecg json...}", "rejected": "{...rejected json...}"}
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

### Success Criteria (Evaluator → Implementor)

#### Must Pass
- [ ] **Valid JSONL** - File can be parsed line-by-line as JSON
- [ ] **Prompt format** - Matches template from design.md section 3.5
- [ ] **Chosen valid** - Can be parsed as Graph
- [ ] **Rejected valid** - Can be parsed as Graph
- [ ] **No truncation** - Prompt doesn't cut off mid-element
- [ ] **UTF-8 encoding** - File is properly encoded

#### Verification Method
```python
def test_dpo_formatter():
    formatter = DPOFormatter()

    example = formatter.format_example(trace, context, chosen_ecg, rejected_ecg)

    # Verify JSONL format
    json_str = json.dumps(example)
    parsed = json.loads(json_str)
    assert "prompt" in parsed
    assert "chosen" in parsed
    assert "rejected" in parsed

    # Verify can be parsed as graphs
    chosen_graph = json.loads(parsed["chosen"])
    rejected_graph = json.loads(parsed["rejected"])
    assert "nodes" in chosen_graph
    assert "edges" in chosen_graph

    # Verify prompt template
    assert "Trace:" in parsed["prompt"]
    assert "Context:" in parsed["prompt"]
```

---

## CONTRACT-006: Verifier Module (V-* Rules)

### Purpose
Verify ECG preserves ground truth from RCG.

### Interface (Implementor → Evaluator)

#### Input
```python
rcg: CallGraph    # Original RCG (ground truth)
ecg: Dict         # Enriched ECG from LLM
proxy_map: Dict   # Mapping of proxy names to real beans
```

#### Output
```python
class VerificationResult:
    is_valid: bool      # True if passes all required checks
    errors: List[str]  # V-EDGE, V-CAUSE failures (must be empty if is_valid)
    warnings: List[str]  # V-THREAD, V-CYCLE, V-REMOTE warnings
```

### Success Criteria (Evaluator → Implementor)

#### Must Pass
- [ ] **V-EDGE check** - Every RCG edge has path in ECG
- [ ] **V-CAUSE check** - Every exception in cause chain appears in ECG
- [ ] **V-THREAD check** - Warns on user→Thread without async
- [ ] **V-CYCLE check** - Warns on unexpected cycles
- [ ] **V-REMOTE check** - Verifies remote node constraints
- [ ] **Fallback logic** - Returns RCG when verification fails
- [ ] **No false positives** - Valid ECG passes all checks

#### Verification Method
```python
def test_verifier():
    verifier = Verifier()

    # Test V-EDGE failure
    rcg = CallGraph(...)
    ecg = {"nodes": [...], "edges": [...]}  # Missing some edges
    result = verifier.verify(rcg, ecg, proxy_map)
    assert result.is_valid == False
    assert any("V-EDGE" in e for e in result.errors)

    # Test valid ECG passes
    ecg_valid = {"nodes": [...all rcg nodes...], "edges": [...all rcg edges...]}
    result = verifier.verify(rcg, ecg_valid, proxy_map)
    assert result.is_valid == True
    assert len(result.errors) == 0
```

---

## CONTRACT-007: Model Interface (Qwen + MLX)

### Purpose
Interface with Qwen2.5-1.5B-Instruct via MLX for ECG generation.

### Interface (Implementor → Evaluator)

#### Input
```python
trace: str      # Java stack trace
context: Dict    # Static analysis context
```

#### Output
```python
ecg: Dict  # Enriched Call Graph as JSON
# {
#   "nodes": [...],
#   "edges": [...],
#   "metadata": {...}
# }
```

### Success Criteria (Evaluator → Implementor)

#### Must Pass
- [ ] **Model loads** - Qwen model initializes successfully
- [ ] **Auto-download** - Downloads from HuggingFace if not cached
- [ ] **Generation works** - Returns ECG for valid input
- [ ] **JSON parseable** - Output can be parsed as JSON
- [ ] **Prompt follows template** - Uses format from design.md
- [ ] **Error handling** - Returns error ECG on parse failure
- [ ] **Inference time** - < 5 seconds per trace (AC-5)

#### Verification Method
```python
def test_qwen_interface():
    model = QwenInterface()

    trace = """java.lang.RuntimeException: Test error
        at com.example.Service.method(Service.java:10)
    """

    context = {
        "classes": {
            "com.example.Service": {
                "annotations": ["@Service"]
            }
        }
    }

    ecg = model.generate_ecg(trace, context)

    # Verify valid JSON
    assert isinstance(ecg, dict)
    assert "nodes" in ecg
    assert "edges" in ecg

    # Verify structure
    assert all("id" in node for node in ecg["nodes"])

    # Verify time constraint
    import time
    start = time.time()
    ecg = model.generate_ecg(trace, context)
    elapsed = time.time() - start
    assert elapsed < 5.0, f"Inference took {elapsed}s, expected < 5s"
```

---

## CONTRACT-008: DPO Training Harness

### Purpose
Execute DPO training loop with LoRA adapter.

### Interface (Implementor → Evaluator)

#### Input
```python
dataset_path: str    # Path to dpo_dataset.jsonl
output_dir: str      # Directory for checkpoints
config: Dict         # Training configuration
```

#### Output
```
output_dir/
├── checkpoint-000500.safetensors  # LoRA adapter weights
├── checkpoint-001000.safetensors
├── checkpoint-002000.safetensors
└── final.safetensors
```

### Success Criteria (Evaluator → Implementor)

#### Must Pass
- [ ] **Dataset loads** - Can read JSONL file
- [ ] **Loss computes** - DPO loss formula correct: `-log(σ(r_chosen - r_rejected))`
- [ ] **Gradient updates** - Weights update after each step
- [ ] **Checkpoint saves** - Saves at configured intervals
- [ ] **LoRA applies** - Only LoRA parameters trainable
- [ ] **Training completes** - Finishes within 8 hours (AC-6)
- [ ] **Memory fits** - Stays within 16GB Mac mini memory

#### Verification Method
```python
def test_dpo_training():
    config = DPOConfig(
        batch_size=1,
        gradient_accumulation=4,
        learning_rate=1e-5,
        training_steps=100  # Short run for test
    )

    trainer = DPOTrainer(config)

    # Test loss computation
    loss = trainer._compute_dpo_loss(prompt, chosen, rejected)
    assert isinstance(loss, mx.array)
    assert loss > 0  # Loss should be positive

    # Test training step
    initial_weights = get_model_weights(trainer.model)
    trainer._backprop(loss)
    new_weights = get_model_weights(trainer.model)
    assert not all(initial_weights == new_weights)  # Weights changed

    # Test checkpoint saving
    trainer._save_checkpoint(500, output_dir)
    assert (output_dir / "checkpoint-000500.safetensors").exists()
```

---

## CONTRACT-009: Evaluation Harness

### Purpose
Run full evaluation and compute AC-1 to AC-6 metrics.

### Interface (Implementor → Evaluator)

#### Input
```python
test_scenarios: List[TestScenario]  # From design.md section 6.2
ground_truth_ecg: List[Graph]       # Expected enriched graphs
```

#### Output
```python
class EvaluationReport:
    ac1_recall: float          # Hidden edge recall (≥ 0.85)
    ac2_hallucination: float   # Hallucination rate (< 0.05)
    ac3_rejection: float       # Verifier rejection rate (< 0.03)
    ac4_hop_reduction: float   # Avg hop reduction (≥ 2)
    ac5_inference_time: float  # Avg seconds per trace (< 5s)
    ac6_training_time: float   # Training hours (≤ 8h)

    passed: bool  # True if all AC met
    details: Dict # Per-scenario breakdown
```

### Success Criteria (Evaluator → Implementor)

#### Must Pass
- [ ] **AC-1: Recall** - Hidden edge recall ≥ 85%
- [ ] **AC-2: Hallucination** - False edges < 5% of total
- [ ] **AC-3: Rejection** - Verifier rejection < 3%
- [ ] **AC-4: Hop reduction** - Root cause localization ≥ 2 hops better than RCG
- [ ] **AC-5: Inference** - Per-trace inference < 5 seconds
- [ ] **AC-6: Training** - Full training completes in ≤ 8 hours
- [ ] **Report generated** - JSON report with all metrics
- [ ] **Scenario coverage** - Tests cover all 5 patterns (AOP, Async, Feign, Reactive, Multi-trace)

#### Verification Method
```python
def test_evaluation_harness():
    evaluator = Evaluator()

    scenarios = [
        TestScenario(name="proxy", pattern="E-PROXY"),
        TestScenario(name="async", pattern="E-ASYNC"),
        TestScenario(name="feign", pattern="E-REMOTE"),
        TestScenario(name="reactive", pattern="E-INTERCEPTOR"),
        TestScenario(name="multi", pattern="E-MERGE"),
    ]

    report = evaluator.evaluate_all(scenarios)

    # Verify all acceptance criteria
    assert report.ac1_recall >= 0.85, f"AC-1 failed: {report.ac1_recall}"
    assert report.ac2_hallucination < 0.05, f"AC-2 failed: {report.ac2_hallucination}"
    assert report.ac3_rejection < 0.03, f"AC-3 failed: {report.ac3_rejection}"
    assert report.ac4_hop_reduction >= 2.0, f"AC-4 failed: {report.ac4_hop_reduction}"
    assert report.ac5_inference_time < 5.0, f"AC-5 failed: {report.ac5_inference_time}"
    assert report.ac6_training_time <= 8.0, f"AC-6 failed: {report.ac6_training_time}"

    assert report.passed == True
```

---

## CONTRACT-010: End-to-End Integration

### Purpose
Full pipeline from trace to enriched graph with verification.

### Interface (Implementor → Evaluator)

#### Input
```python
stack_trace: str     # Raw Java stack trace
static_context: Dict # From Java project analysis
```

#### Output
```python
class PipelineResult:
    rcg: CallGraph           # Raw call graph
    ecg: Graph              # Enriched call graph (or RCG if failed)
    verification: VerificationResult
    metrics: Dict            # Quality metrics
    inference_time: float    # Seconds
```

### Success Criteria (Evaluator → Implementor)

#### Must Pass
- [ ] **Full pipeline runs** - Trace → RCG → ECG → Verify → Output
- [ ] **Fallback works** - Returns RCG when ECG fails verification
- [ ] **Metrics computed** - All quality metrics available
- [ ] **Time tracked** - Inference time logged
- [ ] **Reproducible** - Same input produces same output (with fixed seed)
- [ ] **No crashes** - Handles all edge cases gracefully

#### Verification Method
```python
def test_end_to_end():
    harness = RCAHarness(config_path="harness/config.yaml")

    trace = """java.lang.RuntimeException: Database error
        at com.example.UserService.getUser(UserService.java:42)
        at com.example.UserController.get(UserController.java:28)
    Caused by: java.sql.SQLException: Connection failed
        at com.example.Database.connect(Database.java:56)
    """

    context = {
        "classes": {
            "com.example.UserServiceImpl": {
                "annotations": ["@Transactional"]
            }
        },
        "proxy_mappings": {}
    }

    result = harness.run_single_trace(trace, context)

    # Verify pipeline completed
    assert result.rcg is not None
    assert result.verification is not None
    assert result.final_graph is not None

    # Verify fallback logic
    if not result.verification.is_valid:
        assert result.final_graph == result.rcg  # Falls back to RCG

    # Verify metrics
    assert "inference_time" in result.metrics
    assert result.metrics["inference_time"] < 5.0
```

---

## 6. Implementation Checklist

### Week 1: Harness Foundation + RCG Parser
- [ ] CONTRACT-001: RCG Parser (Stack Trace Parser)
- [ ] Set up `.claude/` harness configuration
- [ ] Create `CLAUDE.md` with project context
- [ ] Create `harness/runner.py` orchestrator
- [ ] Create `harness/validators/`
- [ ] Set up `.github/workflows/ci.yml`

### Week 1-2: Java Projects + Enrichment
- [ ] CONTRACT-002: Java Demo Projects (AOP Proxy)
- [ ] CONTRACT-003: Enrichment Engine (E-* Rules)
- [ ] CONTRACT-004: Rejection Engine (R-* Rules)
- [ ] CONTRACT-005: DPO Formatter

### Week 2-3: Model + Verifier
- [ ] CONTRACT-006: Verifier Module (V-* Rules)
- [ ] CONTRACT-007: Model Interface (Qwen + MLX)
- [ ] Integration tests

### Week 3-4: DPO Training + Evaluation
- [ ] CONTRACT-008: DPO Training Harness
- [ ] CONTRACT-009: Evaluation Harness
- [ ] CONTRACT-010: End-to-End Integration

### Week 4-5: Final Validation
- [ ] Run full evaluation
- [ ] Generate validation report
- [ ] Verify all acceptance criteria (AC-1 to AC-6)

---

## 6.1 Checkpoint & Commit Protocol

### After Each Contract Completion
**Mandatory checkpoint procedure:**

1. **Verify all Success Criteria** - Run verification tests from contract
2. **Document completion** - Update contract checklist in this file
3. **Commit changes** with standardized message format:
   ```bash
   git add .
   git commit -m "feat: complete CONTRACT-XXX - [Component Name]"
   ```
4. **Tag milestone** - Create annotated tag for traceability:
   ```bash
   git tag -a v0.X-XXX -m "CONTRACT-XXX completed: [Component Name]"
   ```

### Commit Message Standards
| Prefix | Usage |
|--------|-------|
| `feat:` | New feature/component implementation |
| `fix:` | Bug fix in existing code |
| `test:` | Adding/updating tests |
| `docs:` | Documentation updates |
| `refactor:` | Code restructuring |

### Checkpoint Milestones
| Tag | Contract | Description |
|-----|----------|-------------|
| `v0.1-001` | CONTRACT-001 | RCG Parser complete |
| `v0.1-002` | CONTRACT-002 | Java Demo Projects complete |
| `v0.1-003` | CONTRACT-003 | Enrichment Engine complete |
| `v0.1-004` | CONTRACT-004 | Rejection Engine complete |
| `v0.1-005` | CONTRACT-005 | DPO Formatter complete |
| `v0.2-006` | CONTRACT-006 | Verifier Module complete |
| `v0.2-007` | CONTRACT-007 | Model Interface complete |
| `v0.3-008` | CONTRACT-008 | DPO Training Harness complete |
| `v0.3-009` | CONTRACT-009 | Evaluation Harness complete |
| `v0.4-010` | CONTRACT-010 | End-to-End Integration complete |
| `v1.0-final` | All AC met | Full validation passed |

### Rollback Protection
- Never force-push to main branch
- Always create feature branches for each contract
- Use pull requests for peer review before merging
- Maintain changelog in `CHANGELOG.md`:
  ```markdown
  ## [Unreleased]
  
  ## [v0.1-001] - 2026-XX-XX
  ### Added
  - RCG Parser implementation
  - Stack trace parsing with proxy detection
  - Caused-by chain extraction
  ```

---

## 7. Configuration

### `config.yaml`
```yaml
rca_agent:
  model:
    base_path: "./models/base/qwen-1.5b-4bit"
    lora_path: "./models/lora/aligned"
    max_tokens: 2048
    temperature: 0.7

  training:
    dpo_config:
      lora_rank: 16
      lora_alpha: 32
      batch_size: 1
      gradient_accumulation: 4
      learning_rate: 1e-5
      training_steps: 2000

  verification:
    proxy_map_path: "./data/proxy_mappings.json"
    fallback_to_rcg: true

  paths:
    java_projects: "./java-projects"
    raw_traces: "./data/raw_traces"
    rcg: "./data/rcg"
    ecg: "./data/ecg"
    dpo_dataset: "./data/dpo_dataset.jsonl"
    evaluation_results: "./data/evaluation_results"

  acceptance_criteria:
    ac1_recall_threshold: 0.85
    ac2_hallucination_threshold: 0.05
    ac3_rejection_threshold: 0.03
    ac4_hop_reduction_threshold: 2
    ac5_inference_time_limit: 5.0  # seconds
    ac6_training_time_limit: 8.0   # hours
```

---

## 8. Dependencies

### Python (`requirements.txt`)
```
mlx>=0.29.0
mlx-examples  # For DPO training
transformers>=4.40.0
torch>=2.0.0  # For data processing
networkx>=3.0  # For graph operations
pyyaml>=6.0
tqdm>=4.0.0
pytest>=7.0.0
numpy>=1.24.0
```

---

## 9. Key Implementation Notes

### 9.1 MLX DPO Training
- Use `mlx-examples/dpo` as reference implementation
- Implement custom DPO loss compatible with MLX
- LoRA adapter only updates ~0.1% of parameters

### 9.2 Proxy Resolution
- Maintain mapping: `{$Proxy123: UserServiceImpl}`
- Built from static context analysis of Java projects
- Used in both RCG parsing and ECG verification

### 9.3 Graph Representation
- Use NetworkX `DiGraph` for algorithmic operations
- Serialize to/from JSON for LLM communication
- Normalize node IDs for consistent comparison

### 9.4 Fallback Strategy
```
Input: Stack trace
  ↓
Parse → RCG (always available)
  ↓
Generate ECG → LLM
  ↓
Verify ECG → Verifier
  ↓
Pass? → Yes → Return ECG
       → No  → Log warning, Return RCG
```

---

## 10. Open Questions for Clarification

1. **Java Compilation**: Should I use `subprocess` to call `javac`/`java` from Python, or create a Java harness that outputs traces directly?

2. **Model Download**: Should I include code to automatically download the Qwen model, or assume it's pre-downloaded?

3. **Dataset Size**: Design says 1000 samples - is this a hard requirement or guidance?

4. **Multi-trace Merge**: Should traces be pre-linked via a shared transaction ID, or should the LLM infer the merge?

5. **Ground Truth ECG**: For evaluation, should I manually create ground truth ECG files, or use the "chosen" enrichment as ground truth?
