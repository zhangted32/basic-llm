import json
import time
from typing import Dict, Any, Optional
from pathlib import Path


class ModelInterface:
    """
    Base class for Model Interface.
    Provides common functionality for ECG generation.
    """

    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        self.model_loaded = False
        self.inference_count = 0

    def load_model(self):
        """Load the model (to be implemented by subclasses)."""
        raise NotImplementedError

    def generate_ecg(self, trace: str, context: Dict) -> Dict[str, Any]:
        """
        Generate ECG from stack trace and context.
        
        Args:
            trace: Java stack trace string
            context: Static analysis context
            
        Returns:
            ECG dictionary with nodes, edges, and metadata
        """
        raise NotImplementedError

    def _build_prompt(self, trace: str, context: Dict) -> str:
        """
        Build prompt from trace and context.
        
        Args:
            trace: Java stack trace string
            context: Static analysis context
            
        Returns:
            Formatted prompt string
        """
        context_str = json.dumps(context, indent=2)
        
        prompt = f"""You are an expert Java exception analyst. Given a Java stack trace and context information, generate an Enriched Call Graph (ECG).

## Input Stack Trace:
```
{trace}
```

## Context Information:
```json
{context_str}
```

## ECG Format:
Generate an Enriched Call Graph (ECG) as JSON with the following structure:
{{
  "nodes": [
    {{
      "id": "className.methodName",
      "class": "fully.qualified.ClassName",
      "method": "methodName",
      "file": "FileName.java",
      "line": 123,
      "enrichment": {{
        "type": "proxy|async|interceptor|remote|meta",
        "proxy_resolved": true,
        "proxy_type": "spring|jdk|cglib",
        "resolved_to": "RealClassName.methodName",
        "async_boundary": true,
        "interceptor_type": "transaction|retry|circuit_breaker",
        "remote_type": "feign|rest|http",
        "external_service": "ServiceName",
        "module": "controller|service|repository",
        "severity": "low|medium|high|critical",
        "layer": "presentation|business|data|infrastructure"
      }}
    }}
  ],
  "edges": [
    {{
      "from": "ClassName.methodName",
      "to": "ClassName.methodName",
      "type": "call|throws|async"
    }}
  ],
  "metadata": {{
    "enrichment_applied": ["E-PROXY", "E-INTERCEPTOR", "E-META"],
    "generation_time_ms": 123,
    "model": "Qwen2.5-1.5B-Instruct"
  }}
}}

## Enrichment Rules:
1. **E-PROXY**: Replace proxy class names (e.g., `$Proxy123`, `UserService$$EnhancerByCGLIB$$abc123`) with real implementation names
2. **E-INTERCEPTOR**: Insert framework interceptor nodes (TransactionInterceptor, RetryTemplate, etc.) based on annotations
3. **E-ASYNC**: Mark async boundaries for Future.get(), CompletableFuture, @Async methods
4. **E-REMOTE**: Add external service nodes for Feign client calls, REST clients, etc.
5. **E-META**: Add module (controller/service/repository), severity (low/medium/high/critical), and layer (presentation/business/data)

Generate the ECG now. Only output valid JSON, no other text."""
        
        return prompt

    def _parse_model_output(self, output: str) -> Dict[str, Any]:
        """
        Parse model output into ECG format.
        
        Args:
            output: Raw model output string
            
        Returns:
            Parsed ECG dictionary
        """
        try:
            ecg = json.loads(output)
            return ecg
        except json.JSONDecodeError:
            return self._create_error_ecg(f"Failed to parse model output as JSON: {output[:200]}")

    def _create_error_ecg(self, error_message: str) -> Dict[str, Any]:
        """
        Create an error ECG when generation fails.
        
        Args:
            error_message: Description of the error
            
        Returns:
            Error ECG dictionary
        """
        return {
            "nodes": [],
            "edges": [],
            "metadata": {
                "error": error_message,
                "model": self.model_name
            }
        }

    def _validate_ecg(self, ecg: Dict[str, Any]) -> bool:
        """
        Validate ECG structure.
        
        Args:
            ecg: ECG dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(ecg, dict):
            return False
        
        if "nodes" not in ecg or "edges" not in ecg:
            return False
        
        if not isinstance(ecg["nodes"], list) or not isinstance(ecg["edges"], list):
            return False
        
        for node in ecg["nodes"]:
            if not isinstance(node, dict):
                return False
            if "id" not in node:
                return False
        
        for edge in ecg["edges"]:
            if not isinstance(edge, dict):
                return False
            if "from" not in edge or "to" not in edge:
                return False
        
        return True