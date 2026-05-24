import unittest
from src.parser.call_graph import CallGraph, StackFrame
from src.verifier.verifier import Verifier, VerificationResult


class TestVerifier(unittest.TestCase):
    
    def setUp(self):
        self.verifier = Verifier()

    def test_v_edge_valid(self):
        """Test V-EDGE with valid ECG that preserves all edges."""
        rcg = CallGraph(
            frames=[
                StackFrame(class_name="com.example.Controller", method_name="getUser", line_number=30),
                StackFrame(class_name="com.example.Service", method_name="getUser", line_number=20),
                StackFrame(class_name="com.example.Repository", method_name="findById", line_number=10),
            ],
            exception_type="java.lang.NullPointerException"
        )
        
        ecg = {
            "nodes": [
                {"id": "com.example.Controller.getUser", "class": "com.example.Controller", "method": "getUser"},
                {"id": "com.example.Service.getUser", "class": "com.example.Service", "method": "getUser"},
                {"id": "com.example.Repository.findById", "class": "com.example.Repository", "method": "findById"},
                {"id": "java.lang.NullPointerException", "class": "java.lang.NullPointerException", "method": ""},
            ],
            "edges": [
                {"from": "com.example.Controller.getUser", "to": "com.example.Service.getUser", "type": "call"},
                {"from": "com.example.Service.getUser", "to": "com.example.Repository.findById", "type": "call"},
                {"from": "com.example.Repository.findById", "to": "java.lang.NullPointerException", "type": "throws"},
            ]
        }
        
        result = self.verifier.verify(rcg, ecg)
        
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(result.details["v_edge"]["total_rcg_edges"], 2)
        self.assertEqual(result.details["v_edge"]["verified_edges"], 2)

    def test_v_edge_missing_path(self):
        """Test V-EDGE with missing edge in ECG."""
        rcg = CallGraph(
            frames=[
                StackFrame(class_name="com.example.Controller", method_name="getUser", line_number=30),
                StackFrame(class_name="com.example.Service", method_name="getUser", line_number=20),
                StackFrame(class_name="com.example.Repository", method_name="findById", line_number=10),
            ],
            exception_type="java.lang.NullPointerException"
        )
        
        # ECG missing the edge from Service to Repository
        ecg = {
            "nodes": [
                {"id": "com.example.Controller.getUser", "class": "com.example.Controller", "method": "getUser"},
                {"id": "com.example.Service.getUser", "class": "com.example.Service", "method": "getUser"},
                {"id": "com.example.Repository.findById", "class": "com.example.Repository", "method": "findById"},
            ],
            "edges": [
                {"from": "com.example.Controller.getUser", "to": "com.example.Service.getUser", "type": "call"},
                # Missing edge: Service -> Repository
            ]
        }
        
        result = self.verifier.verify(rcg, ecg)
        
        self.assertFalse(result.is_valid)
        self.assertTrue(any("V-EDGE" in e for e in result.errors))

    def test_v_cause_valid(self):
        """Test V-CAUSE with exception in cause chain present in ECG."""
        rcg = CallGraph(
            frames=[
                StackFrame(class_name="com.example.Service", method_name="process", line_number=50),
            ],
            exception_type="java.lang.RuntimeException",
            caused_by=["java.sql.SQLException", "java.io.IOException"]
        )
        
        ecg = {
            "nodes": [
                {"id": "com.example.Service.process", "class": "com.example.Service", "method": "process"},
                {"id": "java.sql.SQLException", "class": "java.sql.SQLException", "method": ""},
                {"id": "java.io.IOException", "class": "java.io.IOException", "method": ""},
                {"id": "java.lang.RuntimeException", "class": "java.lang.RuntimeException", "method": ""},
            ],
            "edges": []
        }
        
        result = self.verifier.verify(rcg, ecg)
        
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)

    def test_v_cause_missing(self):
        """Test V-CAUSE with missing exception in cause chain."""
        rcg = CallGraph(
            frames=[
                StackFrame(class_name="com.example.Service", method_name="process", line_number=50),
            ],
            exception_type="java.lang.RuntimeException",
            caused_by=["java.sql.SQLException", "java.io.IOException"]
        )
        
        # ECG missing IOException
        ecg = {
            "nodes": [
                {"id": "com.example.Service.process", "class": "com.example.Service", "method": "process"},
                {"id": "java.sql.SQLException", "class": "java.sql.SQLException", "method": ""},
                {"id": "java.lang.RuntimeException", "class": "java.lang.RuntimeException", "method": ""},
            ],
            "edges": []
        }
        
        result = self.verifier.verify(rcg, ecg)
        
        self.assertFalse(result.is_valid)
        self.assertTrue(any("V-CAUSE" in e and "IOException" in e for e in result.errors))

    def test_v_thread_with_async(self):
        """Test V-THREAD with async enrichment (no warning expected)."""
        rcg = CallGraph(
            frames=[
                StackFrame(class_name="com.example.Controller", method_name="asyncGet", line_number=30),
                StackFrame(class_name="java.util.concurrent.CompletableFuture", method_name="get", line_number=1),
            ],
            exception_type="java.lang.Exception"
        )
        
        ecg = {
            "nodes": [
                {"id": "com.example.Controller.asyncGet", "class": "com.example.Controller", "method": "asyncGet", "enrichment": {"type": "meta"}},
                {"id": "java.util.concurrent.CompletableFuture.get", "class": "java.util.concurrent.CompletableFuture", "method": "get", "enrichment": {"type": "async"}},
            ],
            "edges": [
                {"from": "com.example.Controller.asyncGet", "to": "java.util.concurrent.CompletableFuture.get", "type": "call"},
            ]
        }
        
        result = self.verifier.verify(rcg, ecg)
        
        self.assertEqual(len(result.warnings), 0)

    def test_v_thread_without_async(self):
        """Test V-THREAD without async enrichment (warning expected)."""
        rcg = CallGraph(
            frames=[
                StackFrame(class_name="com.example.Controller", method_name="getData", line_number=30),
                StackFrame(class_name="java.lang.Thread", method_name="run", line_number=1),
            ],
            exception_type="java.lang.Exception"
        )
        
        ecg = {
            "nodes": [
                {"id": "com.example.Controller.getData", "class": "com.example.Controller", "method": "getData", "enrichment": {}},
                {"id": "java.lang.Thread.run", "class": "java.lang.Thread", "method": "run", "enrichment": {}},
            ],
            "edges": [
                {"from": "com.example.Controller.getData", "to": "java.lang.Thread.run", "type": "call"},
            ]
        }
        
        result = self.verifier.verify(rcg, ecg)
        
        self.assertTrue(len(result.warnings) > 0)
        self.assertTrue(any("V-THREAD" in w for w in result.warnings))

    def test_v_remote_with_external_service(self):
        """Test V-REMOTE with external service representation (no warning expected)."""
        rcg = CallGraph(
            frames=[
                StackFrame(class_name="com.example.Service", method_name="callExternal", line_number=30),
                StackFrame(class_name="feign.FeignException", method_name="errorStatus", line_number=1),
            ],
            exception_type="feign.FeignException"
        )
        
        ecg = {
            "nodes": [
                {"id": "com.example.Service.callExternal", "class": "com.example.Service", "method": "callExternal", "enrichment": {"type": "remote"}},
                {"id": "ExternalFeignService", "class": "ExternalService", "method": "", "enrichment": {"type": "remote"}},
            ],
            "edges": [
                {"from": "com.example.Service.callExternal", "to": "ExternalFeignService", "type": "call"},
            ]
        }
        
        result = self.verifier.verify(rcg, ecg)
        
        self.assertEqual(len(result.warnings), 0)

    def test_v_remote_without_external_service(self):
        """Test V-REMOTE without external service representation (warning expected)."""
        rcg = CallGraph(
            frames=[
                StackFrame(class_name="com.example.Service", method_name="callExternal", line_number=30),
                StackFrame(class_name="feign.FeignException", method_name="errorStatus", line_number=1),
            ],
            exception_type="feign.FeignException"
        )
        
        ecg = {
            "nodes": [
                {"id": "com.example.Service.callExternal", "class": "com.example.Service", "method": "callExternal", "enrichment": {}},
                {"id": "feign.FeignException.errorStatus", "class": "feign.FeignException", "method": "errorStatus", "enrichment": {}},
            ],
            "edges": [
                {"from": "com.example.Service.callExternal", "to": "feign.FeignException.errorStatus", "type": "call"},
            ]
        }
        
        result = self.verifier.verify(rcg, ecg)
        
        self.assertTrue(len(result.warnings) > 0)
        self.assertTrue(any("V-REMOTE" in w for w in result.warnings))

    def test_verify_with_fallback_valid(self):
        """Test verify_with_fallback returns ECG when valid."""
        rcg = CallGraph(
            frames=[
                StackFrame(class_name="com.example.Controller", method_name="get", line_number=30),
                StackFrame(class_name="com.example.Service", method_name="get", line_number=20),
            ],
            exception_type="java.lang.Exception"
        )
        
        ecg = {
            "nodes": [
                {"id": "com.example.Controller.get", "class": "com.example.Controller", "method": "get"},
                {"id": "com.example.Service.get", "class": "com.example.Service", "method": "get"},
                {"id": "java.lang.Exception", "class": "java.lang.Exception", "method": ""},
            ],
            "edges": [
                {"from": "com.example.Controller.get", "to": "com.example.Service.get", "type": "call"},
                {"from": "com.example.Service.get", "to": "java.lang.Exception", "type": "throws"},
            ]
        }
        
        result, returned_ecg = self.verifier.verify_with_fallback(rcg, ecg)
        
        self.assertTrue(result.is_valid)
        self.assertEqual(returned_ecg, ecg)

    def test_verify_with_fallback_invalid(self):
        """Test verify_with_fallback returns RCG converted to ECG when invalid."""
        rcg = CallGraph(
            frames=[
                StackFrame(class_name="com.example.Controller", method_name="get", line_number=30),
                StackFrame(class_name="com.example.Service", method_name="get", line_number=20),
            ],
            exception_type="java.lang.Exception"
        )
        
        # Invalid ECG (missing edge)
        ecg = {
            "nodes": [
                {"id": "com.example.Controller.get", "class": "com.example.Controller", "method": "get"},
            ],
            "edges": []
        }
        
        result, returned_ecg = self.verifier.verify_with_fallback(rcg, ecg)
        
        self.assertFalse(result.is_valid)
        # Should return RCG as fallback
        self.assertEqual(len(returned_ecg["nodes"]), 2)
        self.assertEqual(returned_ecg["metadata"]["enrichment_applied"], "fallback")

    def test_proxy_resolution(self):
        """Test V-EDGE with proxy resolution."""
        rcg = CallGraph(
            frames=[
                StackFrame(class_name="com.example.Controller", method_name="get", line_number=20),
                StackFrame(class_name="jdk.proxy2.$Proxy30", method_name="getUser", line_number=30, is_proxy=True, proxy_type="jdk"),
            ],
            exception_type="java.lang.Exception"
        )
        
        proxy_map = {"$Proxy30": "com.example.service.UserServiceImpl"}
        
        ecg = {
            "nodes": [
                {"id": "com.example.Controller.get", "class": "com.example.Controller", "method": "get"},
                {"id": "com.example.service.UserServiceImpl.getUser", "class": "com.example.service.UserServiceImpl", "method": "getUser"},
                {"id": "java.lang.Exception", "class": "java.lang.Exception", "method": ""},
            ],
            "edges": [
                {"from": "com.example.Controller.get", "to": "com.example.service.UserServiceImpl.getUser", "type": "call"},
                {"from": "com.example.service.UserServiceImpl.getUser", "to": "java.lang.Exception", "type": "throws"},
            ]
        }
        
        result = self.verifier.verify(rcg, ecg, proxy_map)
        
        self.assertTrue(result.is_valid)


if __name__ == "__main__":
    unittest.main()