import unittest
import json
from src.evaluation.evaluation_report import (
    TestScenario, 
    EvaluationReport, 
    MetricThresholds
)
from src.evaluation.evaluator import Evaluator, GroundTruthGenerator


class TestEvaluationReport(unittest.TestCase):
    
    def test_report_creation(self):
        """Test creating an evaluation report."""
        report = EvaluationReport(
            ac1_recall=0.90,
            ac2_hallucination=0.02,
            ac3_rejection=0.01,
            ac4_hop_reduction=2.5,
            ac5_inference_time=2.5,
            ac6_training_time=6.0,
            passed=True
        )
        
        self.assertEqual(report.ac1_recall, 0.90)
        self.assertEqual(report.ac2_hallucination, 0.02)
        self.assertTrue(report.passed)
    
    def test_report_to_dict(self):
        """Test converting report to dictionary."""
        report = EvaluationReport(ac1_recall=0.85)
        report_dict = report.to_dict()
        
        self.assertIn("ac1_recall", report_dict)
        self.assertEqual(report_dict["ac1_recall"], 0.85)
    
    def test_report_print_summary(self):
        """Test printing report summary."""
        report = EvaluationReport(
            ac1_recall=0.90,
            ac2_hallucination=0.02,
            ac3_rejection=0.01,
            ac4_hop_reduction=2.5,
            ac5_inference_time=2.5,
            ac6_training_time=6.0,
            total_scenarios=5,
            passed_scenarios=4
        )
        
        # Should not raise an exception
        report.print_summary()


class TestMetricThresholds(unittest.TestCase):
    
    def test_all_thresholds_pass(self):
        """Test checking all thresholds when passed."""
        report = EvaluationReport(
            ac1_recall=0.90,  # ≥ 0.85
            ac2_hallucination=0.02,  # < 0.05
            ac3_rejection=0.01,  # < 0.03
            ac4_hop_reduction=2.5,  # ≥ 2.0
            ac5_inference_time=3.0,  # < 5.0
            ac6_training_time=6.0  # ≤ 8.0
        )
        
        self.assertTrue(MetricThresholds.check_all(report))
    
    def test_ac1_fails(self):
        """Test AC-1 recall failure."""
        report = EvaluationReport(
            ac1_recall=0.80,  # < 0.85
            ac2_hallucination=0.02,
            ac3_rejection=0.01,
            ac4_hop_reduction=2.5,
            ac5_inference_time=3.0,
            ac6_training_time=6.0
        )
        
        self.assertFalse(MetricThresholds.check_all(report))
        failures = MetricThresholds.get_failures(report)
        self.assertTrue(any("AC-1" in f for f in failures))
    
    def test_ac2_fails(self):
        """Test AC-2 hallucination failure."""
        report = EvaluationReport(
            ac1_recall=0.90,
            ac2_hallucination=0.10,  # ≥ 0.05
            ac3_rejection=0.01,
            ac4_hop_reduction=2.5,
            ac5_inference_time=3.0,
            ac6_training_time=6.0
        )
        
        self.assertFalse(MetricThresholds.check_all(report))
        failures = MetricThresholds.get_failures(report)
        self.assertTrue(any("AC-2" in f for f in failures))
    
    def test_get_failures(self):
        """Test getting failure list."""
        report = EvaluationReport(
            ac1_recall=0.80,
            ac2_hallucination=0.10,
            ac3_rejection=0.01,
            ac4_hop_reduction=1.5,
            ac5_inference_time=3.0,
            ac6_training_time=6.0
        )
        
        failures = MetricThresholds.get_failures(report)
        self.assertGreaterEqual(len(failures), 3)  # At least AC-1, AC-2, AC-4


class TestTestScenario(unittest.TestCase):
    
    def test_scenario_creation(self):
        """Test creating a test scenario."""
        scenario = TestScenario(
            name="proxy_test",
            pattern="E-PROXY",
            description="Test proxy enrichment",
            difficulty="hard"
        )
        
        self.assertEqual(scenario.name, "proxy_test")
        self.assertEqual(scenario.pattern, "E-PROXY")
        self.assertEqual(scenario.difficulty, "hard")
    
    def test_scenario_default_description(self):
        """Test default description."""
        scenario = TestScenario(name="test", pattern="E-META")
        
        self.assertIn("E-META", scenario.description)


class TestEvaluator(unittest.TestCase):
    
    def setUp(self):
        self.evaluator = Evaluator()
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        self.assertIsNotNone(self.evaluator.verifier)
        self.assertIsNotNone(self.evaluator.parser)
    
    def test_evaluate_all_empty(self):
        """Test evaluating with no scenarios."""
        report = self.evaluator.evaluate_all([])
        
        self.assertEqual(report.total_scenarios, 0)
        self.assertEqual(report.passed_scenarios, 0)
    
    def test_evaluate_single_trace(self):
        """Test evaluating a single trace."""
        trace = """java.lang.Exception
    at com.example.Test.method(Test.java:10)
    at com.example.Main.main(Main.java:20)"""
        
        model_ecg = {
            "nodes": [
                {"id": "com.example.Test.method", "class": "com.example.Test", "method": "method"}
            ],
            "edges": []
        }
        
        result = self.evaluator.evaluate_single_trace(trace, model_ecg=model_ecg)
        
        self.assertIn("rcg_hops", result)
        self.assertIn("ecg_hops", result)
        self.assertIn("hop_reduction", result)


class TestGroundTruthGenerator(unittest.TestCase):
    
    def test_generate_proxy_scenario(self):
        """Test generating proxy scenario."""
        data = GroundTruthGenerator.generate_proxy_scenario()
        
        self.assertEqual(data["scenario"], "proxy")
        self.assertEqual(data["pattern"], "E-PROXY")
        self.assertIn("test_cases", data)
        self.assertGreater(len(data["test_cases"]), 0)
    
    def test_generate_async_scenario(self):
        """Test generating async scenario."""
        data = GroundTruthGenerator.generate_async_scenario()
        
        self.assertEqual(data["scenario"], "async")
        self.assertEqual(data["pattern"], "E-ASYNC")
        self.assertIn("test_cases", data)


if __name__ == "__main__":
    unittest.main()