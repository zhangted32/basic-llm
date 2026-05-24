import pytest
from src.parser.stack_trace_parser import StackTraceParser
from src.parser.call_graph import CallGraph, StackFrame


class TestStackTraceParser:
    """Tests for CONTRACT-001: RCG Parser"""

    def setup_method(self):
        self.parser = StackTraceParser()

    def test_parse_standard_stack_trace(self):
        """Test parsing a standard Java exception stack trace."""
        trace = """java.lang.RuntimeException: Test error
            at com.example.Service.method(Service.java:10)
            at com.example.Controller.handle(Controller.java:20)
        """
        result = self.parser.parse(trace)
        
        assert len(result.frames) == 2
        assert result.exception_type == "java.lang.RuntimeException"
        assert result.exception_message == "Test error"
        assert result.frames[0].class_name == "com.example.Service"
        assert result.frames[0].method_name == "method"
        assert result.frames[0].file_name == "Service.java"
        assert result.frames[0].line_number == 10
        assert result.frames[0].is_proxy == False

    def test_parse_proxy_classes(self):
        """Test identifying $Proxy patterns."""
        proxy_trace = """java.lang.RuntimeException: Test error
            at com.example.$Proxy123.getUser(Unknown Source)
            at com.example.Controller.handle(Controller.java:20)
        """
        result = self.parser.parse(proxy_trace)
        
        assert result.frames[0].is_proxy == True
        assert result.frames[0].proxy_type == "jdk_proxy"
        assert result.frames[0].class_name == "UserServiceImpl"  # Normalized

    def test_parse_cglib_enhancer(self):
        """Test identifying CGLIB enhancer patterns."""
        cglib_trace = """java.lang.RuntimeException: Test error
            at com.example.UserService$$EnhancerByCGLIB$$abc123.getUser(UserService.java:25)
            at com.example.Controller.handle(Controller.java:30)
        """
        result = self.parser.parse(cglib_trace)
        
        assert result.frames[0].is_proxy == True
        assert result.frames[0].proxy_type == "cglib"
        assert result.frames[0].class_name == "UserService"  # Normalized

    def test_parse_caused_by_chain(self):
        """Test extracting exceptions from cause chain."""
        caused_trace = """RuntimeException: Initial error
            at com.example.Service.method(Service.java:10)
        Caused by: java.lang.IllegalStateException: Secondary error
            at com.example.Service.inner(Service.java:15)
        Caused by: java.lang.RuntimeException: Root cause
            at com.example.Service.root(Service.java:20)
        """
        result = self.parser.parse(caused_trace)
        
        assert result.exception_type == "RuntimeException"
        assert result.caused_by == ["java.lang.IllegalStateException", "java.lang.RuntimeException"]
        assert len(result.frames) == 3

    def test_proxy_normalization(self):
        """Test proxy name normalization."""
        assert self.parser.normalize_proxy_name("$Proxy123") == "UserServiceImpl"
        assert self.parser.normalize_proxy_name("UserService$$EnhancerByCGLIB$$abc123") == "UserService"
        assert self.parser.normalize_proxy_name("com.example.Service$$EnhancerBySpringCGLIB$$12345") == "Service"

    def test_edge_construction(self):
        """Test that edges are correctly constructed between consecutive frames."""
        trace = """java.lang.RuntimeException: Error
            at com.example.A.methodA(A.java:1)
            at com.example.B.methodB(B.java:2)
            at com.example.C.methodC(C.java:3)
        """
        result = self.parser.parse(trace)
        graph_dict = result.to_dict()
        
        edges = graph_dict["edges"]
        assert len(edges) == 2
        
        # Check edge from A to B
        edge1 = edges[0]
        assert edge1["from"] == "com.example.A.methodA"
        assert edge1["to"] == "com.example.B.methodB"
        assert edge1["type"] == "call"
        
        # Check edge from B to C
        edge2 = edges[1]
        assert edge2["from"] == "com.example.B.methodB"
        assert edge2["to"] == "com.example.C.methodC"
        assert edge2["type"] == "call"

    def test_error_handling_empty_input(self):
        """Test handling empty input."""
        result = self.parser.parse("")
        assert len(result.frames) == 0
        assert result.exception_type is None

    def test_error_handling_invalid_input(self):
        """Test handling invalid input without crashing."""
        result = self.parser.parse("This is not a valid stack trace")
        assert len(result.frames) == 0

    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        trace = """java.lang.RuntimeException: Test
            at com.example.Service.method(Service.java:10)
        """
        result = self.parser.parse(trace)
        
        # Serialize to JSON
        json_str = result.to_json()
        
        # Deserialize back
        restored = CallGraph.from_json(json_str)
        
        assert restored.exception_type == result.exception_type
        assert len(restored.frames) == len(result.frames)
        assert restored.frames[0].class_name == result.frames[0].class_name

    def test_unknown_source_file(self):
        """Test handling 'Unknown Source' file."""
        trace = """java.lang.RuntimeException: Error
            at com.example.Service.method(Unknown Source)
        """
        result = self.parser.parse(trace)
        
        assert result.frames[0].file_name is None
        assert result.frames[0].line_number is None

    def test_lambda_expression(self):
        """Test handling lambda expressions."""
        trace = """java.lang.RuntimeException: Error
            at com.example.Service.lambda$process$1/12345678(Service.java:10)
        """
        result = self.parser.parse(trace)
        
        assert "lambda_process" in result.frames[0].method_name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])