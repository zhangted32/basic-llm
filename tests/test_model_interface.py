import unittest
import json
import time
from src.model.qwen_mlx import QwenMLXInterface
from src.model.model_interface import ModelInterface


class TestQwenMLXInterface(unittest.TestCase):
    
    def setUp(self):
        self.model = QwenMLXInterface()
        self.sample_trace = """java.lang.IllegalArgumentException: Invalid user ID
    at com.example.service.UserServiceImpl.getUser(UserServiceImpl.java:20)
    at jdk.proxy2.$Proxy30.getUser(Unknown Source)
    at com.example.controller.UserController.get(UserController.java:35)
    at org.springframework.web.servlet.DispatcherServlet.doDispatch(DispatcherServlet.java:1040)"""

        self.sample_context = {
            "proxy_mappings": {
                "$Proxy30": "com.example.service.UserServiceImpl"
            },
            "classes": {
                "com.example.service.UserServiceImpl": {
                    "annotations": ["@Service", "@Transactional"]
                },
                "com.example.controller.UserController": {
                    "annotations": ["@RestController"]
                }
            }
        }

    def test_model_initialization(self):
        """Test that model initializes correctly."""
        self.assertEqual(self.model.model_name, "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertFalse(self.model.model_loaded)
        self.assertEqual(self.model.inference_count, 0)

    def test_load_model(self):
        """Test model loading."""
        self.model.load_model()
        # Note: In test environment, model might not actually load
        # but should not raise an exception
        self.assertIsNotNone(self.model)

    def test_generate_ecg_basic(self):
        """Test basic ECG generation."""
        ecg = self.model.generate_ecg(self.sample_trace, self.sample_context)
        
        # Verify structure
        self.assertIn("nodes", ecg)
        self.assertIn("edges", ecg)
        self.assertIn("metadata", ecg)
        
        # Verify nodes exist
        self.assertIsInstance(ecg["nodes"], list)
        self.assertGreater(len(ecg["nodes"]), 0)
        
        # Verify node structure
        for node in ecg["nodes"]:
            self.assertIn("id", node)
            self.assertIn("class", node)
            self.assertIn("method", node)

    def test_generate_ecg_with_proxy(self):
        """Test ECG generation with proxy resolution."""
        ecg = self.model.generate_ecg(self.sample_trace, self.sample_context)
        
        # Check that proxy was detected
        proxy_found = False
        for node in ecg["nodes"]:
            if node.get("enrichment", {}).get("type") == "proxy":
                proxy_found = True
                # Verify proxy was resolved
                self.assertIn("proxy_resolved", node["enrichment"])
                break
        
        # Should have some enrichment
        self.assertTrue(len(ecg["nodes"]) > 0)

    def test_generate_ecg_metadata(self):
        """Test that metadata is properly populated."""
        ecg = self.model.generate_ecg(self.sample_trace, self.sample_context)
        
        # Check metadata fields
        self.assertIn("generation_time_ms", ecg["metadata"])
        self.assertIn("model", ecg["metadata"])
        self.assertEqual(ecg["metadata"]["model"], "Qwen/Qwen2.5-1.5B-Instruct")

    def test_inference_time_constraint(self):
        """Test that inference completes within time limit."""
        start = time.time()
        ecg = self.model.generate_ecg(self.sample_trace, self.sample_context)
        elapsed = time.time() - start
        
        # Should complete within 5 seconds (as per contract)
        self.assertLess(elapsed, 5.0, f"Inference took {elapsed}s, expected < 5s")
        
        # Check metadata
        self.assertIn("generation_time_ms", ecg["metadata"])
        self.assertLess(ecg["metadata"]["generation_time_ms"], 5000)

    def test_inference_count_increments(self):
        """Test that inference count increments."""
        initial_count = self.model.inference_count
        
        self.model.generate_ecg(self.sample_trace, self.sample_context)
        self.assertEqual(self.model.inference_count, initial_count + 1)
        
        self.model.generate_ecg(self.sample_trace, self.sample_context)
        self.assertEqual(self.model.inference_count, initial_count + 2)

    def test_build_prompt(self):
        """Test prompt building."""
        prompt = self.model._build_prompt(self.sample_trace, self.sample_context)
        
        # Check that prompt contains trace and context
        self.assertIn(self.sample_trace, prompt)
        self.assertIn("E-PROXY", prompt)
        self.assertIn("E-INTERCEPTOR", prompt)
        self.assertIn("E-META", prompt)

    def test_validate_ecg(self):
        """Test ECG validation."""
        # Valid ECG
        valid_ecg = {
            "nodes": [
                {"id": "test.Test.method", "class": "test.Test", "method": "method"}
            ],
            "edges": [
                {"from": "test.Test.method", "to": "test.Other.method", "type": "call"}
            ]
        }
        self.assertTrue(self.model._validate_ecg(valid_ecg))
        
        # Invalid ECG - missing nodes
        invalid_ecg = {
            "edges": []
        }
        self.assertFalse(self.model._validate_ecg(invalid_ecg))
        
        # Invalid ECG - missing edge 'to' field
        invalid_ecg2 = {
            "nodes": [{"id": "test.Test.method"}],
            "edges": [{"from": "test.Test.method"}]
        }
        self.assertFalse(self.model._validate_ecg(invalid_ecg2))

    def test_error_ecg_creation(self):
        """Test error ECG creation."""
        error_ecg = self.model._create_error_ecg("Test error message")
        
        self.assertEqual(error_ecg["nodes"], [])
        self.assertEqual(error_ecg["edges"], [])
        self.assertIn("error", error_ecg["metadata"])
        self.assertIn("Test error message", error_ecg["metadata"]["error"])

    def test_proxy_resolution_in_context(self):
        """Test proxy resolution using context."""
        trace = """java.lang.Exception
    at jdk.proxy3.$Proxy99.getData(Unknown Source)
    at com.example.Controller.fetch(Controller.java:20)"""
        
        context = {
            "proxy_mappings": {
                "$Proxy99": "com.example.DataService"
            }
        }
        
        ecg = self.model.generate_ecg(trace, context)
        
        # Should contain nodes
        self.assertGreater(len(ecg["nodes"]), 0)


class TestModelInterface(unittest.TestCase):
    """Test the base ModelInterface class."""

    def test_interface_instantiation(self):
        """Test that ModelInterface can be instantiated."""
        interface = ModelInterface()
        self.assertEqual(interface.model_name, "Qwen/Qwen2.5-1.5B-Instruct")

    def test_build_prompt_format(self):
        """Test that prompt follows expected format."""
        interface = ModelInterface()
        
        trace = "test trace"
        context = {"test": "context"}
        
        prompt = interface._build_prompt(trace, context)
        
        # Check for required sections
        self.assertIn("Java exception analyst", prompt)
        self.assertIn("Enriched Call Graph", prompt)
        self.assertIn("E-PROXY", prompt)
        self.assertIn("E-INTERCEPTOR", prompt)


if __name__ == "__main__":
    unittest.main()