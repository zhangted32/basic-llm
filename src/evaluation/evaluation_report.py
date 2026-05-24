from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class TestScenario:
    """Represents a test scenario for evaluation."""
    name: str
    pattern: str  # e.g., "E-PROXY", "E-ASYNC", "E-REMOTE"
    description: str = ""
    difficulty: str = "medium"  # easy, medium, hard
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.description:
            self.description = f"Test scenario for {self.pattern}"


@dataclass
class EvaluationReport:
    """Container for evaluation metrics and results."""
    
    # Acceptance Criteria Metrics (AC-1 to AC-6)
    ac1_recall: float = 0.0  # Hidden edge recall (≥ 0.85)
    ac2_hallucination: float = 0.0  # Hallucination rate (< 0.05)
    ac3_rejection: float = 0.0  # Verifier rejection rate (< 0.03)
    ac4_hop_reduction: float = 0.0  # Avg hop reduction (≥ 2)
    ac5_inference_time: float = 0.0  # Avg seconds per trace (< 5s)
    ac6_training_time: float = 0.0  # Training hours (≤ 8h)
    
    # Overall pass/fail
    passed: bool = False
    
    # Detailed breakdown
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Per-scenario results
    scenario_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    total_scenarios: int = 0
    passed_scenarios: int = 0
    evaluation_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "ac1_recall": self.ac1_recall,
            "ac2_hallucination": self.ac2_hallucination,
            "ac3_rejection": self.ac3_rejection,
            "ac4_hop_reduction": self.ac4_hop_reduction,
            "ac5_inference_time": self.ac5_inference_time,
            "ac6_training_time": self.ac6_training_time,
            "passed": self.passed,
            "details": self.details,
            "scenario_results": self.scenario_results,
            "total_scenarios": self.total_scenarios,
            "passed_scenarios": self.passed_scenarios,
            "evaluation_time_seconds": self.evaluation_time_seconds
        }
    
    def print_summary(self):
        """Print a human-readable summary."""
        print("\n" + "="*60)
        print("EVALUATION REPORT SUMMARY")
        print("="*60)
        print(f"\nAcceptance Criteria (AC-1 to AC-6):")
        print(f"  AC-1 Recall:            {self.ac1_recall:.2%} (≥ 85%)")
        print(f"  AC-2 Hallucination:     {self.ac2_hallucination:.2%} (< 5%)")
        print(f"  AC-3 Rejection:        {self.ac3_rejection:.2%} (< 3%)")
        print(f"  AC-4 Hop Reduction:    {self.ac4_hop_reduction:.2f} (≥ 2.0)")
        print(f"  AC-5 Inference Time:    {self.ac5_inference_time:.2f}s (< 5s)")
        print(f"  AC-6 Training Time:     {self.ac6_training_time:.2f}h (≤ 8h)")
        print(f"\nOverall: {'✓ PASSED' if self.passed else '✗ FAILED'}")
        print(f"Scenarios: {self.passed_scenarios}/{self.total_scenarios} passed")
        print(f"Evaluation Time: {self.evaluation_time_seconds:.2f}s")
        print("="*60)


class MetricThresholds:
    """Thresholds for acceptance criteria."""
    AC1_RECALL_MIN = 0.85
    AC2_HALLUCINATION_MAX = 0.05
    AC3_REJECTION_MAX = 0.03
    AC4_HOP_REDUCTION_MIN = 2.0
    AC5_INFERENCE_TIME_MAX = 5.0  # seconds
    AC6_TRAINING_TIME_MAX = 8.0  # hours
    
    @classmethod
    def check_all(cls, report: EvaluationReport) -> bool:
        """Check if all AC thresholds are met."""
        checks = [
            report.ac1_recall >= cls.AC1_RECALL_MIN,
            report.ac2_hallucination < cls.AC2_HALLUCINATION_MAX,
            report.ac3_rejection < cls.AC3_REJECTION_MAX,
            report.ac4_hop_reduction >= cls.AC4_HOP_REDUCTION_MIN,
            report.ac5_inference_time < cls.AC5_INFERENCE_TIME_MAX,
            report.ac6_training_time <= cls.AC6_TRAINING_TIME_MAX
        ]
        return all(checks)
    
    @classmethod
    def get_failures(cls, report: EvaluationReport) -> List[str]:
        """Get list of failed AC checks."""
        failures = []
        if report.ac1_recall < cls.AC1_RECALL_MIN:
            failures.append(f"AC-1 Recall: {report.ac1_recall:.2%} < {cls.AC1_RECALL_MIN:.2%}")
        if report.ac2_hallucination >= cls.AC2_HALLUCINATION_MAX:
            failures.append(f"AC-2 Hallucination: {report.ac2_hallucination:.2%} >= {cls.AC2_HALLUCINATION_MAX:.2%}")
        if report.ac3_rejection >= cls.AC3_REJECTION_MAX:
            failures.append(f"AC-3 Rejection: {report.ac3_rejection:.2%} >= {cls.AC3_REJECTION_MAX:.2%}")
        if report.ac4_hop_reduction < cls.AC4_HOP_REDUCTION_MIN:
            failures.append(f"AC-4 Hop Reduction: {report.ac4_hop_reduction:.2f} < {cls.AC4_HOP_REDUCTION_MIN:.2f}")
        if report.ac5_inference_time >= cls.AC5_INFERENCE_TIME_MAX:
            failures.append(f"AC-5 Inference Time: {report.ac5_inference_time:.2f}s >= {cls.AC5_INFERENCE_TIME_MAX:.2f}s")
        if report.ac6_training_time > cls.AC6_TRAINING_TIME_MAX:
            failures.append(f"AC-6 Training Time: {report.ac6_training_time:.2f}h > {cls.AC6_TRAINING_TIME_MAX:.2f}h")
        return failures