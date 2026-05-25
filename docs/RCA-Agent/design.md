# Enriched Call Graph Generation via DPO‑Aligned LLM for RCA  
**Design Document**  
**Author:** Senior Architect  
**Date:** 2026-05-24  
**Target Hardware:** Mac mini (10‑core, 16 GB unified memory)  
**Constraint:** No access to the production Java repository; verification must use simple, self‑contained Java projects.

---

## 1. Background & Motivation

A regex parser can reliably extract the **literal call chain** from a Java exception stack trace and produce a raw call graph (RCG). However, modern frameworks (Spring Boot, Micronaut, etc.) hide critical runtime structure:

- **AOP proxies** (`$Proxy123`, CGLIB enhancers) mask real bean methods.
- **Asynchronous boundaries** (`@Async`, `CompletableFuture`) split the logical call path across threads.
- **Interceptors** (`@Transactional`, `@Retryable`) swallow and re‑wrap exceptions.
- **Remote calls** (Feign, WebClient) appear as opaque framework internals.
- **Multi‑trace cascades** require merging several stack traces to find a shared root cause.

An aligned LLM can **infer** these hidden connections and output an **enriched call graph (ECG)** that dramatically reduces the on‑call engineer’s cognitive load and shortens MTTR.

**Why RL alignment (DPO)?**  
Supervised fine‑tuning alone would teach the output format, but not optimise for *diagnostic usefulness*. By training with Direct Preference Optimization (DPO) on pairs of “good” and “bad” graphs, the model learns to prioritise actionable, accurate inferences.

---

## 2. High‑Level Architecture
Java stack trace
│
▼
[Regex Parser] ──> Raw Call Graph (RCG) ← deterministic, trusted
│
▼
[Context Retriever] ← static analysis from the project (class‑annotation map, proxy targets)
│
▼
[Aligned LLM] ──> Enriched Call Graph (ECG)
│
▼
[Verifier] ──> Final Graph (fallback to RCG if verification fails)

text

- The regex handles the ground‑truth parsing.
- The LLM adds hidden structure and metadata.
- The verifier ensures no ground‑truth information is lost.

---

## 3. Dataset Construction Rules

We will build a synthetic dataset using **small, illustrative Spring Boot projects** that are completely independent of the real product. These projects will be designed to exhibit specific patterns.

### 3.1 Reference Java Projects (Patterns)

| Pattern               | How to Trigger                                                                 |
|-----------------------|--------------------------------------------------------------------------------|
| AOP Proxy             | `@Transactional` service method throwing `RuntimeException`.                   |
| Async Boundary        | `@Async` method returning `Future`; caller `.get()` throws `ExecutionException`. |
| Feign/RestTemplate    | REST call to an invalid endpoint, producing `FeignException`.                  |
| Reactive Chain        | WebFlux `Mono` pipeline with an error signal.                                  |
| Multi‑Trace Cascade   | Service A times out calling B; service A’s scheduled job fails on a dead pool. |

Trace capture can be done via a custom error handler or logging.

### 3.2 Building the Raw Call Graph (RCG)

Apply the existing regex to each trace. The output is a minimal JSON graph containing only the literal edges (e.g., `classA.method1 → classB.method2`). Save as `rcg.json`.

### 3.3 Enrichment Rules (for the “Chosen” ECG)

For each trace a human developer creates the ideal enriched graph. The following **enrichment rules** must be followed:

| Rule ID | Description | ECG Representation |
|---------|-------------|-------------------|
| **E‑PROXY** | Replace `$ProxyX`/`$$Enhancer` nodes with the actual bean class and method. | Add edge from proxy to real method with property `"proxy": true`. |
| **E‑ASYNC** | If the trace ends at `Future.get()` and contains `ExecutionException`, infer the async task that actually failed. | Add a node for the lambda/runnable and mark the edge `"async": true`. |
| **E‑INTERCEPTOR** | Detect well‑known interceptors (`TransactionInterceptor`, `RetryTemplate`). | Insert interceptor nodes and show flow through them with `"interceptor": "@Transactional"` etc. |
| **E‑REMOTE** | For `FeignException`/`WebClientRequestException`, add a representation of the downstream service. | Add a node for the external endpoint, marked `"remote": true`. |
| **E‑MERGE** | When multiple traces belong to the same incident, merge them into a single graph. | Link traces through shared resources (e.g., connection pool) and highlight the earliest failure as root cause. |
| **E‑META** | Annotate nodes with metadata inferable from the stack trace (severity, module, business context). | `severity: high`, `module: user-service`, etc. |

### 3.4 Rejected Graph Construction Rules

For each trace, generate at least one **rejected** graph to form the DPO preference pair. Use these rules:

| Rule ID | Rejected Candidate |
|---------|--------------------|
| **R‑RAW** | The un‑enriched RCG itself. |
| **R‑MISS** | A graph missing exactly one critical hidden edge (e.g., no proxy resolution) but otherwise correct. |
| **R‑HALLUC** | A graph with plausible but incorrect inferences (e.g., wrong proxy target, a remote call where none exists). This can be produced by the base (unaligned) LLM or crafted manually. |
| **R‑NOISE** | A graph that includes all literal edges but adds many irrelevant framework internal nodes, making it too verbose. |

The final DPO dataset is a JSONL file with entries:  
`{"prompt": "<trace + static context>", "chosen": "<enriched ECG>", "rejected": "<rejected graph>"}`

### 3.5 Prompt Template
You are an expert RCA assistant. Given a Java stack trace and optional static context, generate an enriched call graph in JSON format that reveals hidden call structure: proxy targets, async origins, interceptor chains, and remote services. Use the provided context to resolve ambiguous names.

Trace:
<full stack trace>

Context:

Class: UserService, Method: getUser → @Transactional

Class: OrderClient, Method: placeOrder → @FeignClient("order-service")
... (any static analysis info available)

text

---

## 4. Verifier Design & Verification Rules

The verifier is a deterministic guard that ensures the LLM does not drop or corrupt ground‑truth information from the regex RCG.

### 4.1 Verification Rules

| Rule ID | Check | Failure Action |
|---------|-------|----------------|
| **V‑EDGE** | For every edge `A→B` in the RCG, there must exist a directed path from `A` to `B` in the ECG, after normalising proxy names (using a provided mapping). | Reject ECG, fall back to RCG. |
| **V‑CAUSE** | Every exception in the “caused by” chain must have at least one corresponding node in the ECG. | Reject ECG, fall back to RCG. |
| **V‑THREAD** | No edge from a user class directly to `java.lang.Thread` unless explicitly marked `"async": true`. | Accept ECG but log a warning. |
| **V‑CYCLE** | No cycles that are not marked as retry/recursion boundaries. | Accept ECG but log a warning. |
| **V‑REMOTE** | A node with `"remote": true` must have an incoming edge from an HTTP client (e.g., `FeignClient`) and no outgoing edges to internal methods. | Accept ECG but log a warning. |

### 4.2 Implementation Skeleton

```python
def verify(rcg: dict, ecg: dict, proxy_map: dict) -> bool:
    rcg_graph = build_graph(rcg, proxy_map)
    ecg_graph = build_graph(ecg, proxy_map)
    
    # V-EDGE
    for u, v in rcg_graph.edges():
        if not nx.has_path(ecg_graph, u, v):
            return False
    
    # V-CAUSE
    if not all_causes_present(rcg, ecg_graph):
        return False
    
    # V-THREAD, V-CYCLE, V-REMOTE (log warnings)
    sanity_checks(ecg_graph)
    return True
5. Model Selection & DPO Alignment (Mac Mini Constraints)
5.1 Base Model
Qwen2.5‑1.5B‑Instruct quantised to 4‑bit (Q4_K_M) using Apple’s MLX framework.
Memory footprint with LoRA: ~4 GB, well within 16 GB.

5.2 Training Configuration
Parameter	Value
Training method	Direct Preference Optimization (DPO)
Framework	MLX (mlx-examples/dpo)
LoRA rank	16
LoRA alpha	32
Batch size	1
Gradient accumulation	4
Learning rate	1e-5
Optimizer	AdamW
Training steps	~2000 (1–2 epochs for 1000 samples)
Hardware	Mac mini, 10 CPU cores, 16 GB unified memory
5.3 Training Command (illustrative)
bash
python dpo.py --model ./qwen-1.5b-4bit \
              --dataset ./data/dpo_triples.jsonl \
              --lora-layers 16 --batch-size 1 --grad-accum 4 \
              --iters 2000 --lr 1e-5 --save-every 500
6. Verification Process Using Simple Java Projects
Because we cannot use the production repository, all validation is performed on the synthetic projects described in Section 3.1.

6.1 Test Harness
Deploy the demo apps locally or on a test machine.

Trigger exceptions by calling the designated endpoints.

Capture traces via a logging file or custom error handler.

Generate static context by running a simple script (e.g., javap -v filtered for annotations) on the compiled classes.

6.2 Evaluation Scenarios
Single‑trace tests: proxy crash, async failure, Feign timeout, reactive error.

Multi‑trace test: simulate a complex incident with two traces sharing a database pool exhaustion root cause.

For each scenario, a ground‑truth enriched call graph is manually created (following the enrichment rules) and the “correct root cause” is documented.

6.3 Testing Procedure
Parse trace → RCG (regex).

Feed trace + static context to the aligned LLM → ECG.

Verify ECG with the verifier. If it fails, fall back to RCG.

Measure:

Hidden edge recall: percentage of manually added edges (proxy, async, remote) that appear in the ECG.

Hallucination rate: percentage of ECG‑added edges that are incorrect.

Graph edit distance to ground truth ECG.

Diagnosis efficiency: number of graph hops from the exception origin node to the root cause node (lower is better).

Compare against the RCG baseline.

6.4 Iterative Refinement Loop
Evaluate on a held‑out set.

Collect failure cases (e.g., missing proxy resolution).

Generate new preference triples using the rejected graphs from those failures.

Re‑train DPO on the augmented dataset.

Re‑evaluate until acceptance criteria are met.

7. Acceptance Criteria (AC)
ID	Criterion	Target Value	Measurement
AC‑1	Hidden edge recall (proxy + async + remote)	≥ 85%	Compare against ground truth ECG on test set of ≥30 traces.
AC‑2	Hallucination rate (false added edges)	< 5% of total edges in ECG	Manual inspection / automated comparison with ground truth.
AC‑3	Verifier rejection rate due to V‑EDGE or V‑CAUSE	< 3%	Runtime log over all test traces.
AC‑4	Root cause localisation improvement	≥ 2‑hop reduction on average vs. RCG	Measure shortest path from exception origin to root cause node in ECG vs. RCG.
AC‑5	Model inference time	< 5 seconds per trace on Mac mini	Timed benchmark.
AC‑6	Training completion	Full DPO run finishes within 8 hours on the specified hardware	Wall‑clock measurement.
All criteria must be met using only the synthetic Java projects – no access to the production codebase is required.

8. Deliverables
DPO dataset (JSONL) – preference triples for the synthetic traces.

Trained LoRA adapter – weights that can be merged with the base Qwen model.

Verifier module – Python script with the checks defined in Section 4.

Evaluation harness – script that triggers exceptions, runs the pipeline, and computes metrics.

Validation report – demonstration that all acceptance criteria are satisfied.

9. Summary
This design leverages a small, locally trained LLM to bridge the gap between a stack trace’s literal content and the mental model needed for fast incident resolution. By enforcing strict rules for enrichment, dataset creation, and output verification, the system remains reliable and auditable. The entire approach can be built, trained, and proven on a single Mac mini using only small, non‑production Java projects – making it a low‑risk, high‑value addition to any RCA agent suite.