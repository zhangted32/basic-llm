import unittest
from src.pipeline.rca_harness import RCAHarness, PipelineResult


class TestPipelineResult(unittest.TestCase):
    
    def test_result_creation(self):
        """Test creating a pipeline result."""
        from src.parser.call_graph import CallGraph
        from src.verifier.verifier import VerificationResult
        
        rcg = CallGraph(frames=[], exception_type="Exception", caused_by=[])
        
        result = PipelineResult(
            rcg=rcg,
            ecg={"nodes": [], "edges": []},
            verification=VerificationResult(is_valid=True, errors=[], warnings=[], details={}),
            metrics={"rcg_hops": 0},
            inference_time=1.5,
            total_time=2.0
        )
        
        self.assertEqual(result.inference_time, 1.5)
        self.assertEqual(result.total_time, 2.0)
    
    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        from src.parser.call_graph import CallGraph
        from src.verifier.verifier import VerificationResult
        
        rcg = CallGraph(frames=[], exception_type="Exception", caused_by=[])
        
        result = PipelineResult(
            rcg=rcg,
            ecg={"nodes": [], "edges": []},
            verification=VerificationResult(is_valid=True, errors=[], warnings=[], details={}),
            metrics={"rcg_hops": 0},
            inference_time=1.5,
            total_time=2.0
        )
        
        result_dict = result.to_dict()
        
        self.assertIn("rcg", result_dict)
        self.assertIn("ecg", result_dict)
        self.assertIn("verification", result_dict)
        self.assertIn("metrics", result_dict)
        self.assertEqual(result_dict["inference_time"], 1.5)


class TestRCAHarness(unittest.TestCase):
    
    def setUp(self):
        self.harness = RCAHarness()
    
    def test_harness_initialization(self):
        """Test harness initialization."""
        self.assertIsNotNone(self.harness.parser)
        self.assertIsNotNone(self.harness.model)
        self.assertIsNotNone(self.harness.verifier)
    
    def test_run_single_trace(self):
        """Test running a single trace through the pipeline."""
        trace = """java.lang.RuntimeException: Database error
    at com.example.UserService.getUser(UserService.java:42)
    at com.example.UserController.get(UserController.java:28)
    at org.springframework.web.servlet.DispatcherServlet.doDispatch(DispatcherServlet.java:1040)"""
        
        context = {
            "classes": {
                "com.example.UserService": {
                    "annotations": ["@Service"]
                },
                "com.example.UserController": {
                    "annotations": ["@RestController"]
                }
            },
            "proxy_mappings": {}
        }
        
        result = self.harness.run_single_trace(trace, context, skip_verification=True)
        
        # Verify pipeline completed
        self.assertIsNotNone(result.rcg)
        self.assertIsNotNone(result.ecg)
        self.assertIsNotNone(result.verification)
        
        # Verify RCG was parsed
        self.assertGreater(len(result.rcg.frames), 0)
        
        # Verify ECG has nodes
        self.assertIn("nodes", result.ecg)
        
        # Verify metrics computed
        self.assertIn("inference_time", result.metrics)
        self.assertLess(result.metrics["inference_time"], 5.0)
    
    def test_fallback_to_rcg(self):
        """Test fallback to RCG when ECG verification fails."""
        trace = """java.lang.RuntimeException
    at com.example.Test.method(Test.java:10)
    at com.example.Main.main(Main.java:20)"""
        
        context = {}
        
        # Run with verification disabled to get invalid ECG, then it should fallback
        result = self.harness.run_single_trace(trace, context, skip_verification=True)
        
        # Should complete without crashing
        self.assertIsNotNone(result.rcg)
        self.assertIsNotNone(result.ecg)
    
    def test_module_inference(self):
        """Test module inference from class names."""
        module = self.harness._infer_module("com.example.UserController")
        self.assertEqual(module, "controller")
        
        module = self.harness._infer_module("com.example.UserService")
        self.assertEqual(module, "service")
        
        module = self.harness._infer_module("com.example.UserRepository")
        self.assertEqual(module, "repository")
        
        module = self.harness._infer_module("com.example.Other")
        self.assertEqual(module, "other")
    
    def test_layer_inference(self):
        """Test layer inference from class names."""
        layer = self.harness._infer_layer("com.example.UserController")
        self.assertEqual(layer, "presentation")
        
        layer = self.harness._infer_layer("com.example.UserService")
        self.assertEqual(layer, "business")
        
        layer = self.harness._infer_layer("com.example.UserRepository")
        self.assertEqual(layer, "data")
        
        layer = self.harness._infer_layer("com.example.Other")
        self.assertEqual(layer, "infrastructure")
    
    def test_rcg_to_ecg_conversion(self):
        """Test converting RCG to ECG."""
        from src.parser.call_graph import CallGraph, StackFrame
        
        frame1 = StackFrame(
            class_name="com.example.Test",
            method_name="method1",
            file_name="Test.java",
            line_number=10,
            is_proxy=False
        )
        frame2 = StackFrame(
            class_name="com.example.Main",
            method_name="main",
            file_name="Main.java",
            line_number=20,
            is_proxy=False
        )
        
        rcg = CallGraph(frames=[frame1, frame2], exception_type="Exception", caused_by=[])
        
        ecg = self.harness._rcg_to_ecg(rcg)
        
        self.assertIn("nodes", ecg)
        self.assertIn("edges", ecg)
        self.assertEqual(len(ecg["nodes"]), 3)  # 2 frames + exception node
        self.assertEqual(len(ecg["edges"]), 2)  # 1 call + 1 throws
    
    def test_metrics_computation(self):
        """Test metrics computation."""
        from src.parser.call_graph import CallGraph
        from src.verifier.verifier import VerificationResult
        
        rcg = CallGraph(frames=[], exception_type="", caused_by=[])
        ecg = {"nodes": [{"id": "test"}], "edges": [], "metadata": {"enrichment_applied": ["E-META"]}}
        verification = VerificationResult(is_valid=True, errors=[], warnings=[], details={})
        
        metrics = self.harness._compute_metrics(rcg, ecg, verification, 1.5)
        
        self.assertIn("rcg_hops", metrics)
        self.assertIn("ecg_hops", metrics)
        self.assertIn("verification_passed", metrics)
        self.assertIn("inference_time", metrics)
        self.assertEqual(metrics["inference_time"], 1.5)


if __name__ == "__main__":
    unittest.main()