import json
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path

from src.parser.stack_trace_parser import StackTraceParser
from src.parser.call_graph import CallGraph, StackFrame
from src.model.model_interface import ModelInterface
from src.model.qwen_mlx import QwenMLXInterface
from src.verifier.verifier import Verifier, VerificationResult
from src.evaluation.evaluation_report import TestScenario


@dataclass
class PipelineResult:
    """Result of the full RCA pipeline."""
    
    # Raw call graph
    rcg: CallGraph
    
    # Enriched call graph (or RCG if enrichment failed)
    ecg: Dict[str, Any]
    
    # Verification result
    verification: VerificationResult
    
    # Quality metrics
    metrics: Dict[str, Any]
    
    # Timing information
    inference_time: float
    total_time: float
    
    # Error information if any
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rcg": {
                "frames": [
                    {
                        "class_name": f.class_name,
                        "method_name": f.method_name,
                        "file_name": f.file_name,
                        "line_number": f.line_number
                    } for f in self.rcg.frames
                ],
                "exception_type": self.rcg.exception_type,
                "caused_by": self.rcg.caused_by
            },
            "ecg": self.ecg,
            "verification": {
                "is_valid": self.verification.is_valid,
                "errors": self.verification.errors,
                "warnings": self.verification.warnings
            },
            "metrics": self.metrics,
            "inference_time": self.inference_time,
            "total_time": self.total_time,
            "error": self.error
        }


class RCAHarness:
    """
    RCA Agent Harness - End-to-End Integration Pipeline.
    
    Orchestrates the full pipeline:
    1. Parse stack trace → RCG
    2. Generate ECG from RCG + context → ECG
    3. Verify ECG → VerificationResult
    4. Fallback to RCG if verification fails
    """
    
    def __init__(
        self,
        model: Optional[ModelInterface] = None,
        verifier: Optional[Verifier] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize RCA Harness.
        
        Args:
            model: Model interface for ECG generation
            verifier: Verifier for ECG validation
            config: Optional configuration dictionary
        """
        self.parser = StackTraceParser()
        self.model = model or QwenMLXInterface()
        self.verifier = verifier or Verifier()
        self.config = config or {}
        
        self._load_model()
    
    def _load_model(self):
        """Load the model if not already loaded."""
        if not self.model.model_loaded:
            print("Loading model...")
            self.model.load_model()
    
    def run_single_trace(
        self,
        trace: str,
        context: Dict[str, Any],
        skip_verification: bool = False
    ) -> PipelineResult:
        """
        Run the full pipeline for a single trace.
        
        Args:
            trace: Java stack trace string
            context: Static analysis context
            skip_verification: Skip verification step (for testing)
            
        Returns:
            PipelineResult with RCG, ECG, verification, and metrics
        """
        total_start = time.time()
        
        try:
            # Step 1: Parse RCG
            rcg = self.parser.parse(trace)
            
            # Step 2: Generate ECG
            inference_start = time.time()
            ecg = self.model.generate_ecg(trace, context)
            inference_time = time.time() - inference_start
            
            # Step 3: Verify ECG
            if skip_verification:
                verification = VerificationResult(is_valid=True, errors=[], warnings=[], details={})
            else:
                verification = self.verifier.verify(ecg)
            
            # Step 4: Fallback to RCG if verification failed
            final_ecg = ecg
            if not verification.is_valid:
                print(f"Verification failed, falling back to RCG: {verification.errors}")
                final_ecg = self._rcg_to_ecg(rcg)
                verification = VerificationResult(
                    is_valid=True,
                    errors=["Fallback to RCG due to ECG verification failure"],
                    warnings=verification.errors,
                    details={}
                )
            
            # Compute metrics
            metrics = self._compute_metrics(rcg, final_ecg, verification, inference_time)
            
            total_time = time.time() - total_start
            
            return PipelineResult(
                rcg=rcg,
                ecg=final_ecg,
                verification=verification,
                metrics=metrics,
                inference_time=inference_time,
                total_time=total_time
            )
            
        except Exception as e:
            total_time = time.time() - total_start
            return PipelineResult(
                rcg=CallGraph(frames=[], exception_type="", caused_by=[]),
                ecg={},
                verification=VerificationResult(is_valid=False, errors=[str(e)], warnings=[], details={}),
                metrics={},
                inference_time=0.0,
                total_time=total_time,
                error=str(e)
            )
    
    def _rcg_to_ecg(self, rcg: CallGraph) -> Dict[str, Any]:
        """
        Convert RCG to ECG format.
        
        Args:
            rcg: Raw call graph
            
        Returns:
            ECG dictionary
        """
        nodes = []
        edges = []
        
        for i, frame in enumerate(rcg.frames):
            node_id = f"{frame.class_name}.{frame.method_name}"
            
            node = {
                "id": node_id,
                "class": frame.class_name,
                "method": frame.method_name,
                "file": frame.file_name,
                "line": frame.line_number,
                "enrichment": {
                    "type": "meta",
                    "module": self._infer_module(frame.class_name),
                    "severity": "low",
                    "layer": self._infer_layer(frame.class_name)
                }
            }
            nodes.append(node)
            
            # Create edges
            if i > 0:
                prev_frame = rcg.frames[i-1]
                edges.append({
                    "from": f"{prev_frame.class_name}.{prev_frame.method_name}",
                    "to": node_id,
                    "type": "call"
                })
        
        # Add exception node
        if rcg.exception_type:
            exception_id = rcg.exception_type
            nodes.append({
                "id": exception_id,
                "class": rcg.exception_type,
                "method": "",
                "file": None,
                "line": None,
                "enrichment": {
                    "type": "meta",
                    "severity": "critical",
                    "module": "exception"
                }
            })
            
            if nodes:
                edges.append({
                    "from": nodes[-2]["id"],
                    "to": exception_id,
                    "type": "throws"
                })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "enrichment_applied": ["E-META"],
                "is_fallback": True
            }
        }
    
    def _infer_module(self, class_name: str) -> str:
        """Infer module from class name."""
        if 'controller' in class_name.lower():
            return "controller"
        elif 'service' in class_name.lower():
            return "service"
        elif 'repository' in class_name.lower() or 'dao' in class_name.lower():
            return "repository"
        return "other"
    
    def _infer_layer(self, class_name: str) -> str:
        """Infer layer from class name."""
        if 'controller' in class_name.lower():
            return "presentation"
        elif 'service' in class_name.lower():
            return "business"
        elif 'repository' in class_name.lower() or 'dao' in class_name.lower():
            return "data"
        return "infrastructure"
    
    def _compute_metrics(
        self,
        rcg: CallGraph,
        ecg: Dict[str, Any],
        verification: VerificationResult,
        inference_time: float
    ) -> Dict[str, Any]:
        """
        Compute quality metrics for the result.
        
        Args:
            rcg: Raw call graph
            ecg: Enriched call graph
            verification: Verification result
            inference_time: Time taken for inference
            
        Returns:
            Metrics dictionary
        """
        metrics = {
            "rcg_hops": len(rcg.frames),
            "ecg_hops": len(ecg.get("nodes", [])),
            "ecg_edges": len(ecg.get("edges", [])),
            "enrichment_count": len(ecg.get("metadata", {}).get("enrichment_applied", [])),
            "verification_passed": verification.is_valid,
            "verification_errors": len(verification.errors),
            "verification_warnings": len(verification.warnings),
            "inference_time": inference_time,
            "hop_reduction": len(rcg.frames) - len(ecg.get("nodes", []))
        }
        
        # Add specific enrichment types
        if "metadata" in ecg and "enrichment_applied" in ecg["metadata"]:
            metrics["enrichments"] = ecg["metadata"]["enrichment_applied"]
        
        return metrics
    
    def run_batch(
        self,
        traces: list,
        contexts: list,
        output_dir: Optional[str] = None
    ) -> list:
        """
        Run pipeline for multiple traces.
        
        Args:
            traces: List of Java stack traces
            contexts: List of context dictionaries
            output_dir: Optional output directory for results
            
        Returns:
            List of PipelineResult dictionaries
        """
        results = []
        
        for i, (trace, context) in enumerate(zip(traces, contexts)):
            print(f"Processing trace {i+1}/{len(traces)}...")
            result = self.run_single_trace(trace, context)
            results.append(result.to_dict())
            
            # Save intermediate results
            if output_dir:
                output_path = Path(output_dir) / f"result_{i:04d}.json"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2)
        
        return results


# Convenience function for quick testing
def run_trace(trace: str, context: Dict[str, Any] = None) -> PipelineResult:
    """
    Run a single trace through the RCA pipeline.
    
    Args:
        trace: Java stack trace
        context: Optional context dictionary
        
    Returns:
        PipelineResult
    """
    harness = RCAHarness()
    return harness.run_single_trace(trace, context or {})