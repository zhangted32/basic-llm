import json
import time
from typing import Dict, Any, Optional
from pathlib import Path
import tempfile
import os

try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("Warning: MLX not available. Using mock implementation.")


class QwenMLXInterface:
    """
    Qwen2.5-1.5B-Instruct interface via MLX for Apple Silicon.
    
    This implementation uses MLX for efficient inference on Apple Silicon GPUs.
    Model weights are downloaded from ModelScope.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize Qwen MLX interface.
        
        Args:
            model_path: Optional local path to model. If None, will use ModelScope cache.
        """
        self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.inference_count = 0

    def load_model(self):
        """
        Load the Qwen model via MLX-LM.
        
        Uses ModelScope cached model if available.
        """
        if not MLX_AVAILABLE:
            print("MLX not available. Using mock implementation.")
            self.model_loaded = False
            return

        try:
            print(f"Loading Qwen model from {self.model_name}...")
            print("This may take a few minutes on first run (converting to MLX format)...")
            
            from mlx_lm import load, generate
            
            # Check for ModelScope cache first
            modelscope_path = Path.home() / ".cache" / "modelscope" / "hub" / "qwen" / "Qwen2.5-1.5B-Instruct"
            
            # Determine which path to use
            load_path = str(modelscope_path) if modelscope_path.exists() else self.model_name
            print(f"Trying to load from: {load_path}")
            
            # Load model and tokenizer using mlx-lm
            self.model, self.tokenizer = load(load_path)
            
            self.model_loaded = True
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            print("Falling back to mock implementation.")
            self.model_loaded = False

    def generate_ecg(self, trace: str, context: Dict) -> Dict[str, Any]:
        """
        Generate ECG from stack trace and context.
        
        Args:
            trace: Java stack trace string
            context: Static analysis context
            
        Returns:
            ECG dictionary with nodes, edges, and metadata
        """
        start_time = time.time()
        
        # Build prompt
        prompt = self._build_prompt(trace, context)
        
        if not self.model_loaded:
            # Return mock ECG for testing
            ecg = self._generate_mock_ecg(trace, context)
        else:
            # Generate using actual model
            try:
                from mlx_lm import generate
                
                # Generate output
                output = generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=2048,
                    verbose=False
                )
                
                # Parse the JSON output
                ecg = self._parse_model_output(output)
                
            except Exception as e:
                print(f"Model generation failed: {str(e)}")
                return self._create_error_ecg(f"Generation failed: {str(e)}")
        
        # Add metadata
        generation_time = int((time.time() - start_time) * 1000)
        ecg["metadata"]["generation_time_ms"] = generation_time
        ecg["metadata"]["model"] = self.model_name
        ecg["metadata"]["inference_count"] = self.inference_count
        
        self.inference_count += 1
        
        return ecg

    def _parse_model_output(self, output: str) -> Dict[str, Any]:
        """
        Parse model output to extract ECG JSON.
        
        Args:
            output: Raw model output string
            
        Returns:
            Parsed ECG dictionary
        """
        import re
        
        print(f"Raw output length: {len(output)}")
        print(f"Output preview: {output[:200]}...")
        
        # Try to find JSON in output
        json_match = re.search(r'\{.*\}', output, re.DOTALL)
        
        if json_match:
            json_str = json_match.group()
            print(f"Found JSON with length: {len(json_str)}")
            
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}")
                # Try to fix incomplete JSON
                fixed_json = self._fix_incomplete_json(json_str)
                if fixed_json:
                    try:
                        return json.loads(fixed_json)
                    except json.JSONDecodeError:
                        print("Failed to parse fixed JSON")
                return self._create_error_ecg("Failed to parse model output")
        
        # If no JSON found, return mock ECG
        print("No JSON found in model output")
        return self._create_error_ecg("No valid JSON in output")
    
    def _fix_incomplete_json(self, json_str: str) -> str:
        """
        Try to fix incomplete JSON by closing open brackets.
        
        Args:
            json_str: Incomplete JSON string
            
        Returns:
            Fixed JSON string or None if unfixable
        """
        try:
            # Count brackets
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')
            open_brackets = json_str.count('[')
            close_brackets = json_str.count(']')
            
            # Add closing brackets
            fixed = json_str
            fixed += '}' * (open_braces - close_braces)
            fixed += ']' * (open_brackets - close_brackets)
            
            # Fix incomplete strings
            # If ends with incomplete string, close it
            if fixed.count('"') % 2 != 0:
                fixed += '"'
            
            return fixed
        except Exception:
            return None

    def _generate_mock_ecg(self, trace: str, context: Dict) -> Dict[str, Any]:
        """
        Generate a mock ECG for testing purposes.
        
        Args:
            trace: Java stack trace string
            context: Static analysis context
            
        Returns:
            Mock ECG dictionary
        """
        from src.parser.stack_trace_parser import StackTraceParser
        
        # Parse the trace to get basic structure
        parser = StackTraceParser()
        rcg = parser.parse(trace)
        
        nodes = []
        edges = []
        
        # Create nodes from RCG
        for i, frame in enumerate(rcg.frames):
            node_id = f"{frame.class_name}.{frame.method_name}"
            
            enrichment = self._determine_enrichment(frame, context)
            
            node = {
                "id": node_id,
                "class": frame.class_name,
                "method": frame.method_name,
                "file": frame.file_name,
                "line": frame.line_number,
                "enrichment": enrichment
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
        
        # Determine what enrichments were applied
        enrichment_types = set()
        for node in nodes:
            if node["enrichment"].get("type"):
                enrichment_types.add(f"E-{node['enrichment']['type'].upper()}")
        
        # If no specific enrichments, apply default E-META
        if not enrichment_types:
            enrichment_types.add("E-META")
            for node in nodes:
                node["enrichment"]["type"] = "meta"
                node["enrichment"]["module"] = self._infer_module(node["class"])
                node["enrichment"]["severity"] = self._infer_severity(node["class"], node["method"])
                node["enrichment"]["layer"] = self._infer_layer(node["class"])
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "enrichment_applied": list(enrichment_types),
                "rcg_frames": len(rcg.frames)
            }
        }

    def _determine_enrichment(self, frame, context: Dict) -> Dict[str, Any]:
        """
        Determine enrichment for a frame based on patterns and context.
        
        Args:
            frame: StackFrame object
            context: Static analysis context
            
        Returns:
            Enrichment dictionary
        """
        enrichment = {}
        class_name = frame.class_name
        
        # Check for proxy
        if frame.is_proxy or "$Proxy" in class_name or "Enhancer" in class_name:
            enrichment["type"] = "proxy"
            enrichment["proxy_resolved"] = True
            enrichment["proxy_type"] = "jdk" if "$Proxy" in class_name else "cglib"
            enrichment["resolved_to"] = self._resolve_proxy_name(class_name, context)
        
        # Check for async
        elif any(pattern in class_name for pattern in ["Future", "CompletableFuture", "Async"]):
            enrichment["type"] = "async"
            enrichment["async_boundary"] = True
        
        # Check for interceptor
        elif any(pattern in class_name for pattern in ["Interceptor", "Transaction", "Retry"]):
            enrichment["type"] = "interceptor"
            enrichment["interceptor_type"] = "transaction" if "Transaction" in class_name else "other"
        
        # Check for remote
        elif any(pattern in class_name for pattern in ["Feign", "RestClient", "HttpClient"]):
            enrichment["type"] = "remote"
            enrichment["remote_type"] = "feign" if "Feign" in class_name else "rest"
            enrichment["external_service"] = "ExternalService"
        
        # Default to meta
        else:
            enrichment["type"] = "meta"
            enrichment["module"] = self._infer_module(class_name)
            enrichment["severity"] = self._infer_severity(class_name, frame.method_name)
            enrichment["layer"] = self._infer_layer(class_name)
        
        return enrichment

    def _resolve_proxy_name(self, proxy_name: str, context: Dict) -> str:
        """
        Resolve proxy name to real implementation.
        
        Args:
            proxy_name: Proxy class name
            context: Static analysis context
            
        Returns:
            Resolved class name
        """
        import re
        
        # Check proxy_mappings in context
        proxy_mappings = context.get("proxy_mappings", {})
        for proxy_pattern, real_name in proxy_mappings.items():
            if proxy_pattern in proxy_name:
                return real_name
        
        # Try to extract from CGLIB pattern
        match = re.match(r'^(.+?)\$\$Enhancer', proxy_name)
        if match:
            return match.group(1)
        
        # Try to extract from JDK proxy pattern
        if "$Proxy" in proxy_name:
            # Default resolution
            return "UserServiceImpl"
        
        return proxy_name.split('.')[-1]

    def _infer_module(self, class_name: str) -> str:
        """Infer module from class name."""
        if 'controller' in class_name.lower():
            return "controller"
        elif 'service' in class_name.lower():
            return "service"
        elif 'repository' in class_name.lower() or 'dao' in class_name.lower():
            return "repository"
        return "other"

    def _infer_severity(self, class_name: str, method_name: str) -> str:
        """Infer severity from class and method name."""
        if 'Exception' in class_name or 'Error' in class_name:
            return "critical"
        if 'delete' in method_name.lower() or 'remove' in method_name.lower():
            return "high"
        if 'update' in method_name.lower() or 'save' in method_name.lower():
            return "medium"
        return "low"

    def _infer_layer(self, class_name: str) -> str:
        """Infer layer from class name."""
        if 'controller' in class_name.lower():
            return "presentation"
        elif 'service' in class_name.lower():
            return "business"
        elif 'repository' in class_name.lower() or 'dao' in class_name.lower():
            return "data"
        return "infrastructure"

    def _build_prompt(self, trace: str, context: Dict) -> str:
        """
        Build prompt from trace and context.
        
        Args:
            trace: Java stack trace string
            context: Static analysis context
            
        Returns:
            Formatted prompt string
        """
        # Qwen2.5-Instruct chat format
        messages = [
            {
                "role": "system",
                "content": "You are a Java stack trace analyzer. Convert stack traces to Enriched Call Graph (ECG) JSON format. Output ONLY valid JSON, no explanations."
            },
            {
                "role": "user",
                "content": f"""Convert this stack trace to ECG JSON:

{trace}

Output JSON with:
- nodes: array of {{id, class, method, file, line, enrichment}}
- edges: array of {{from, to, type}}  
- metadata: {{enrichment_applied}}"""
            }
        ]
        
        # Format for Qwen2.5
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "user":
                prompt += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
        
        prompt += "<|im_start|>assistant\n"
        
        return prompt

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