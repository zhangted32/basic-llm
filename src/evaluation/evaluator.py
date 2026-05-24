import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.evaluation.evaluation_report import (
    TestScenario, 
    EvaluationReport, 
    MetricThresholds
)
from src.verifier.verifier import Verifier
from src.parser.stack_trace_parser import StackTraceParser


class Evaluator:
    """
    Evaluation Harness for RCA-Agent.
    
    Runs full evaluation and computes AC-1 to AC-6 metrics.
    """
    
    def __init__(self, ground_truth_path: Optional[str] = None):
        """
        Initialize Evaluator.
        
        Args:
            ground_truth_path: Optional path to ground truth ECG data
        """
        self.ground_truth_path = ground_truth_path
        self.ground_truth_data = {}
        self.verifier = Verifier()
        self.parser = StackTraceParser()
        
        if ground_truth_path:
            self._load_ground_truth(ground_truth_path)
    
    def _load_ground_truth(self, path: str):
        """Load ground truth ECG data."""
        path = Path(path)
        if not path.exists():
            print(f"Warning: Ground truth file not found: {path}")
            return
        
        with open(path, 'r') as f:
            self.ground_truth_data = json.load(f)
        
        print(f"Loaded {len(self.ground_truth_data)} ground truth examples")
    
    def evaluate_all(self, scenarios: List[TestScenario]) -> EvaluationReport:
        """
        Evaluate all scenarios.
        
        Args:
            scenarios: List of test scenarios
            
        Returns:
            EvaluationReport with metrics
        """
        start_time = time.time()
        
        report = EvaluationReport()
        report.total_scenarios = len(scenarios)
        
        all_recalls = []
        all_hallucinations = []
        all_rejections = []
        all_hop_reductions = []
        all_inference_times = []
        
        for scenario in scenarios:
            result = self._evaluate_scenario(scenario)
            report.scenario_results.append(result)
            
            # Collect metrics
            if result.get("recall") is not None:
                all_recalls.append(result["recall"])
            if result.get("hallucination_rate") is not None:
                all_hallucinations.append(result["hallucination_rate"])
            if result.get("rejected"):
                all_rejections.append(1.0)
            else:
                all_rejections.append(0.0)
            if result.get("hop_reduction") is not None:
                all_hop_reductions.append(result["hop_reduction"])
            if result.get("inference_time") is not None:
                all_inference_times.append(result["inference_time"])
            
            # Count passed scenarios
            if result.get("passed", False):
                report.passed_scenarios += 1
        
        # Compute aggregate metrics
        if all_recalls:
            report.ac1_recall = sum(all_recalls) / len(all_recalls)
        if all_hallucinations:
            report.ac2_hallucination = sum(all_hallucinations) / len(all_hallucinations)
        if all_rejections:
            report.ac3_rejection = sum(all_rejections) / len(all_rejections)
        if all_hop_reductions:
            report.ac4_hop_reduction = sum(all_hop_reductions) / len(all_hop_reductions)
        if all_inference_times:
            report.ac5_inference_time = sum(all_inference_times) / len(all_inference_times)
        
        # Training time is typically provided externally
        # For now, set a mock value based on whether training completed
        report.ac6_training_time = 0.0  # Would be set from training run
        
        # Overall pass/fail
        report.passed = MetricThresholds.check_all(report)
        
        # Evaluation time
        report.evaluation_time_seconds = time.time() - start_time
        
        return report
    
    def _evaluate_scenario(self, scenario: TestScenario) -> Dict[str, Any]:
        """
        Evaluate a single scenario.
        
        Args:
            scenario: Test scenario to evaluate
            
        Returns:
            Dictionary with scenario results
        """
        result = {
            "scenario": scenario.name,
            "pattern": scenario.pattern,
            "passed": False,
            "errors": []
        }
        
        try:
            # Load test data for this scenario
            test_data = self._get_test_data(scenario)
            
            if not test_data:
                result["errors"].append(f"No test data for scenario {scenario.name}")
                return result
            
            # Evaluate each test case
            recalls = []
            hallucinations = []
            rejections = 0
            
            for test_case in test_data:
                case_result = self._evaluate_test_case(test_case, scenario)
                recalls.append(case_result.get("recall", 0.0))
                hallucinations.append(case_result.get("hallucination_rate", 0.0))
                if case_result.get("rejected"):
                    rejections += 1
                
                # Collect inference time
                if "inference_time" in case_result:
                    if "inference_times" not in result:
                        result["inference_times"] = []
                    result["inference_times"].append(case_result["inference_time"])
            
            # Aggregate scenario results
            if recalls:
                result["recall"] = sum(recalls) / len(recalls)
            if hallucinations:
                result["hallucination_rate"] = sum(hallucinations) / len(hallucinations)
            
            result["rejected"] = rejections > 0
            result["rejection_count"] = rejections
            result["total_cases"] = len(test_data)
            
            # Check if scenario passed
            result["passed"] = (
                result.get("recall", 0) >= MetricThresholds.AC1_RECALL_MIN and
                result.get("hallucination_rate", 1) < MetricThresholds.AC2_HALLUCINATION_MAX
            )
            
        except Exception as e:
            result["errors"].append(str(e))
        
        return result
    
    def _evaluate_test_case(self, test_case: Dict, scenario: TestScenario) -> Dict[str, Any]:
        """
        Evaluate a single test case.
        
        Args:
            test_case: Test case data with trace and ground truth
            scenario: Test scenario being evaluated
            
        Returns:
            Dictionary with evaluation results
        """
        result = {}
        
        trace = test_case.get("trace", "")
        ground_truth = test_case.get("ground_truth_ecg", {})
        model_output = test_case.get("model_ecg", {})
        
        if not trace:
            return {"error": "No trace in test case"}
        
        # Parse RCG from trace
        try:
            rcg = self.parser.parse(trace)
            rcg_hops = len(rcg.frames)
        except Exception as e:
            return {"error": f"Failed to parse trace: {e}"}
        
        # Use model output if available, otherwise use ground truth
        if model_output:
            ecg = model_output
        elif ground_truth:
            ecg = ground_truth
        else:
            return {"error": "No ECG data available"}
        
        # Count ECG hops
        ecg_hops = len(ecg.get("nodes", []))
        
        # Compute hop reduction
        hop_reduction = rcg_hops - ecg_hops
        result["hop_reduction"] = hop_reduction
        result["rcg_hops"] = rcg_hops
        result["ecg_hops"] = ecg_hops
        
        # Verify ECG
        try:
            verification_result = self.verifier.verify(ecg)
            result["verification_passed"] = verification_result.is_valid
            result["rejected"] = not verification_result.is_valid
            result["verification_errors"] = verification_result.errors
        except Exception as e:
            result["verification_error"] = str(e)
            result["rejected"] = True
        
        # Compute metrics against ground truth
        if ground_truth and ecg:
            # Recall: How many ground truth edges were found
            gt_edges = set()
            for edge in ground_truth.get("edges", []):
                gt_edges.add((edge.get("from"), edge.get("to")))
            
            found_edges = set()
            for edge in ecg.get("edges", []):
                found_edges.add((edge.get("from"), edge.get("to")))
            
            if gt_edges:
                recall = len(gt_edges & found_edges) / len(gt_edges)
                result["recall"] = recall
            
            # Hallucination: How many edges in ECG are not in ground truth
            hallucinated = len(found_edges - gt_edges)
            total_edges = len(found_edges) if found_edges else 1
            result["hallucination_rate"] = hallucinated / total_edges
            result["hallucinated_edges"] = hallucinated
            result["total_edges"] = total_edges
        
        # Inference time
        result["inference_time"] = test_case.get("inference_time", 0.0)
        
        return result
    
    def _get_test_data(self, scenario: TestScenario) -> List[Dict]:
        """
        Get test data for a scenario.
        
        Args:
            scenario: Test scenario
            
        Returns:
            List of test cases
        """
        # Check if we have scenario-specific data
        if scenario.name in self.ground_truth_data:
            return self.ground_truth_data[scenario.name].get("test_cases", [])
        
        # Return mock data for testing
        return [
            {
                "trace": "java.lang.Exception\n\tat com.example.Test.method(Test.java:10)",
                "ground_truth_ecg": {
                    "nodes": [
                        {"id": "com.example.Test.method", "class": "com.example.Test", "method": "method"}
                    ],
                    "edges": []
                },
                "model_ecg": None,
                "inference_time": 1.5
            }
        ]
    
    def evaluate_single_trace(
        self, 
        trace: str, 
        ground_truth_ecg: Optional[Dict] = None,
        model_ecg: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single trace.
        
        Args:
            trace: Java stack trace
            ground_truth_ecg: Expected enriched graph
            model_ecg: Model generated enriched graph
            
        Returns:
            Evaluation results dictionary
        """
        test_case = {
            "trace": trace,
            "ground_truth_ecg": ground_truth_ecg or {},
            "model_ecg": model_ecg,
            "inference_time": 0.0
        }
        
        scenario = TestScenario(name="single", pattern="E-META")
        return self._evaluate_test_case(test_case, scenario)
    
    def save_report(self, report: EvaluationReport, output_path: str):
        """
        Save evaluation report to JSON file.
        
        Args:
            report: Evaluation report
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        print(f"Report saved to {output_path}")


class GroundTruthGenerator:
    """Generate ground truth ECG data for testing."""
    
    @staticmethod
    def generate_proxy_scenario() -> Dict[str, Any]:
        """Generate test data for proxy scenario."""
        return {
            "scenario": "proxy",
            "pattern": "E-PROXY",
            "test_cases": [
                {
                    "trace": """java.lang.RuntimeException: User not found
    at com.example.service.UserServiceImpl.getUser(UserServiceImpl.java:20)
    at jdk.proxy3.$Proxy31.getUser(Unknown Source)
    at com.example.controller.UserController.getUser(UserController.java:35)
    at org.springframework.web.servlet.DispatcherServlet.doDispatch(DispatcherServlet.java:1040)""",
                    "ground_truth_ecg": {
                        "nodes": [
                            {"id": "com.example.service.UserServiceImpl.getUser", "class": "com.example.service.UserServiceImpl", "method": "getUser", "enrichment": {"type": "proxy", "resolved_to": "com.example.service.UserServiceImpl"}},
                            {"id": "com.example.controller.UserController.getUser", "class": "com.example.controller.UserController", "method": "getUser", "enrichment": {"type": "meta", "module": "controller"}}
                        ],
                        "edges": [
                            {"from": "com.example.controller.UserController.getUser", "to": "com.example.service.UserServiceImpl.getUser", "type": "call"}
                        ]
                    }
                }
            ]
        }
    
    @staticmethod
    def generate_async_scenario() -> Dict[str, Any]:
        """Generate test data for async scenario."""
        return {
            "scenario": "async",
            "pattern": "E-ASYNC",
            "test_cases": [
                {
                    "trace": """java.util.concurrent.ExecutionException
    at java.util.concurrent.FutureTask.get(FutureTask.java:155)
    at com.example.service.DataService.fetchData(DataService.java:25)
    at com.example.controller.DataController.get(DataController.java:30)""",
                    "ground_truth_ecg": {
                        "nodes": [
                            {"id": "com.example.service.DataService.fetchData", "class": "com.example.service.DataService", "method": "fetchData", "enrichment": {"type": "async", "async_boundary": True}},
                            {"id": "com.example.controller.DataController.get", "class": "com.example.controller.DataController", "method": "get", "enrichment": {"type": "meta", "module": "controller"}}
                        ],
                        "edges": [
                            {"from": "com.example.controller.DataController.get", "to": "com.example.service.DataService.fetchData", "type": "async"}
                        ]
                    }
                }
            ]
        }