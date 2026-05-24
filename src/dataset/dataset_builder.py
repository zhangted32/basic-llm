import json
import os
from typing import List, Dict, Any, Optional
from src.parser.stack_trace_parser import StackTraceParser
from src.dataset.enrichment_engine import EnrichmentEngine
from src.dataset.rejection_engine import RejectionEngine
from src.dataset.dpo_formatter import DPOFormatter, DPOExample


class DatasetBuilder:
    """
    Builds DPO training dataset from raw stack traces.
    
    Workflow:
    1. Load raw traces from data/raw_traces/
    2. Parse each trace into RCG
    3. Check rejection rules
    4. Apply enrichments
    5. Format as DPO examples
    6. Save to output file
    """

    def __init__(self):
        self.parser = StackTraceParser()
        self.enrichment_engine = EnrichmentEngine()
        self.rejection_engine = RejectionEngine()
        self.dpo_formatter = DPOFormatter()
        
    def build_dataset(self, input_dir: str = "data/raw_traces", 
                     output_file: str = "data/dpo_dataset.jsonl",
                     max_examples: int = 1000) -> int:
        """
        Build the complete DPO dataset.
        
        Args:
            input_dir: Directory containing raw trace JSON files
            output_file: Output JSONL file for DPO training
            max_examples: Maximum number of DPO examples to generate
            
        Returns:
            Number of DPO examples generated
        """
        all_examples: List[DPOExample] = []
        
        # Load all trace files
        trace_files = self._load_trace_files(input_dir)
        
        for trace_file in trace_files:
            if len(all_examples) >= max_examples:
                break
                
            try:
                # Load trace and context
                trace_data = self._load_trace(trace_file)
                context = self._load_context(trace_file)
                
                if not trace_data:
                    continue
                
                # Parse into RCG
                rcg = self.parser.parse(trace_data.get("trace", ""))
                
                if not rcg or len(rcg.frames) == 0:
                    continue
                
                # Check rejection rules
                rejection_result = self.rejection_engine.check_rejection(rcg, context)
                
                if rejection_result.rejected:
                    print(f"Rejected {trace_file}: {rejection_result.reason}")
                    continue
                
                # Generate DPO examples
                examples = self.dpo_formatter.format_dpo_examples(rcg, context)
                all_examples.extend(examples)
                
                print(f"Processed {trace_file}: {len(examples)} examples")
                
            except Exception as e:
                print(f"Error processing {trace_file}: {str(e)}")
        
        # Save to output file
        self.dpo_formatter.format_to_jsonl(all_examples[:max_examples], output_file)
        
        print(f"\nDataset built successfully!")
        print(f"Total DPO examples: {len(all_examples[:max_examples])}")
        print(f"Output file: {output_file}")
        
        return len(all_examples[:max_examples])

    def _load_trace_files(self, input_dir: str) -> List[str]:
        """Load all trace files from input directory."""
        trace_files = []
        
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.startswith("trace_") and file.endswith(".json"):
                    trace_files.append(os.path.join(root, file))
        
        return sorted(trace_files)

    def _load_trace(self, trace_file: str) -> Optional[Dict]:
        """Load a single trace file."""
        try:
            with open(trace_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {trace_file}: {str(e)}")
            return None

    def _load_context(self, trace_file: str) -> Dict:
        """Load context file for a trace."""
        context_dir = os.path.dirname(trace_file)
        context_file = os.path.join(context_dir, "context.json")
        
        if os.path.exists(context_file):
            try:
                with open(context_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading {context_file}: {str(e)}")
        
        return {}

    def build_synthetic_dataset(self, output_file: str = "data/dpo_synthetic.jsonl",
                               num_examples: int = 100) -> int:
        """
        Build a synthetic DPO dataset for testing.
        
        Args:
            output_file: Output JSONL file
            num_examples: Number of synthetic examples to generate
            
        Returns:
            Number of examples generated
        """
        examples: List[DPOExample] = []
        
        # Generate synthetic traces with different patterns
        synthetic_traces = self._generate_synthetic_traces(num_examples)
        
        for trace_data in synthetic_traces:
            rcg = self.parser.parse(trace_data.get("trace", ""))
            context = trace_data.get("context", {})
            
            if not rcg or len(rcg.frames) == 0:
                continue
            
            # Generate DPO examples
            dpo_examples = self.dpo_formatter.format_dpo_examples(rcg, context)
            examples.extend(dpo_examples)
        
        # Save to output file
        self.dpo_formatter.format_to_jsonl(examples, output_file)
        
        print(f"\nSynthetic dataset built successfully!")
        print(f"Total DPO examples: {len(examples)}")
        print(f"Output file: {output_file}")
        
        return len(examples)

    def _generate_synthetic_traces(self, count: int) -> List[Dict]:
        """Generate synthetic traces with various patterns."""
        traces = []
        
        # Proxy pattern trace
        proxy_trace = {
            "trace": """java.lang.IllegalArgumentException: Invalid user ID
    at com.example.service.UserServiceImpl.getUser(UserServiceImpl.java:20)
    at jdk.proxy2.$Proxy30.getUser(Unknown Source)
    at com.example.controller.UserController.getUser(UserController.java:35)
    at org.springframework.web.servlet.DispatcherServlet.doDispatch(DispatcherServlet.java:1040)""",
            "context": {
                "proxy_mappings": {"$Proxy30": "com.example.service.UserServiceImpl"},
                "annotations": {
                    "com.example.service.UserServiceImpl": {
                        "annotations": ["@Service", "@Transactional"]
                    }
                }
            }
        }
        
        # Async pattern trace
        async_trace = {
            "trace": """java.util.concurrent.ExecutionException: java.lang.RuntimeException
    at java.util.concurrent.CompletableFuture.reportGet(CompletableFuture.java:395)
    at java.util.concurrent.CompletableFuture.get(CompletableFuture.java:1999)
    at com.example.service.UserService.lambda$asyncGetUser$0(UserService.java:45)
    at java.util.concurrent.CompletableFuture$AsyncSupply.run(CompletableFuture.java:1768)""",
            "context": {
                "annotations": {
                    "com.example.service.UserService": {
                        "annotations": ["@Async"]
                    }
                }
            }
        }
        
        # Remote call trace
        remote_trace = {
            "trace": """feign.FeignException$NotFound: [404] during [GET] to [http://external-service/api/users/123]
    at feign.FeignException.errorStatus(FeignException.java:178)
    at feign.FeignException.errorStatus(FeignException.java:142)
    at feign.codec.ErrorDecoder$Default.decode(ErrorDecoder.java:92)
    at com.example.client.UserFeignClient.getUser(UserFeignClient.java:25)
    at com.example.service.UserService.getUserFromRemote(UserService.java:60)""",
            "context": {
                "remote_services": {
                    "http://external-service": "UserService"
                }
            }
        }
        
        # Mix pattern trace
        mixed_trace = {
            "trace": """java.lang.NullPointerException
    at com.example.service.OrderServiceImpl.processOrder(OrderServiceImpl.java:55)
    at org.springframework.transaction.interceptor.TransactionInterceptor.invoke(TransactionInterceptor.java:123)
    at jdk.proxy2.$Proxy45.processOrder(Unknown Source)
    at com.example.controller.OrderController.createOrder(OrderController.java:42)""",
            "context": {
                "proxy_mappings": {"$Proxy45": "com.example.service.OrderServiceImpl"},
                "annotations": {
                    "com.example.service.OrderServiceImpl": {
                        "annotations": ["@Service", "@Transactional"]
                    }
                }
            }
        }
        
        # Add multiple copies with variations
        templates = [proxy_trace, async_trace, remote_trace, mixed_trace]
        
        for i in range(count):
            template = templates[i % len(templates)]
            traces.append({
                "trace": template["trace"].replace("123", str(i)),
                "context": template["context"]
            })
        
        return traces

    def get_dataset_statistics(self, input_dir: str = "data/raw_traces") -> Dict[str, Any]:
        """Get statistics about the raw traces."""
        stats = {
            "total_traces": 0,
            "total_frames": 0,
            "traces_with_proxy": 0,
            "traces_with_interceptor": 0,
            "traces_with_async": 0,
            "traces_with_remote": 0,
            "avg_frames_per_trace": 0,
            "rejected_count": 0
        }
        
        trace_files = self._load_trace_files(input_dir)
        total_frames = 0
        traces_with_proxy = 0
        traces_with_interceptor = 0
        traces_with_async = 0
        traces_with_remote = 0
        rejected_count = 0
        
        for trace_file in trace_files:
            try:
                trace_data = self._load_trace(trace_file)
                context = self._load_context(trace_file)
                
                if not trace_data:
                    continue
                
                rcg = self.parser.parse(trace_data.get("trace", ""))
                
                if not rcg:
                    continue
                
                stats["total_traces"] += 1
                total_frames += len(rcg.frames)
                
                # Check patterns
                has_proxy = any(frame.is_proxy or "$Proxy" in frame.class_name for frame in rcg.frames)
                has_interceptor = any("Interceptor" in frame.class_name for frame in rcg.frames)
                has_async = any("Future" in frame.class_name or "Async" in frame.class_name for frame in rcg.frames)
                has_remote = any("Feign" in frame.class_name or "Remote" in frame.class_name for frame in rcg.frames)
                
                if has_proxy:
                    traces_with_proxy += 1
                if has_interceptor:
                    traces_with_interceptor += 1
                if has_async:
                    traces_with_async += 1
                if has_remote:
                    traces_with_remote += 1
                
                # Check rejection
                rejection_result = self.rejection_engine.check_rejection(rcg, context)
                if rejection_result.rejected:
                    rejected_count += 1
                    
            except Exception as e:
                print(f"Error processing {trace_file}: {str(e)}")
        
        stats["total_frames"] = total_frames
        stats["traces_with_proxy"] = traces_with_proxy
        stats["traces_with_interceptor"] = traces_with_interceptor
        stats["traces_with_async"] = traces_with_async
        stats["traces_with_remote"] = traces_with_remote
        stats["rejected_count"] = rejected_count
        
        if stats["total_traces"] > 0:
            stats["avg_frames_per_trace"] = total_frames / stats["total_traces"]
        
        return stats


if __name__ == "__main__":
    builder = DatasetBuilder()
    
    # Build dataset from real traces
    builder.build_dataset(max_examples=100)
    
    # Build synthetic dataset for additional training data
    builder.build_synthetic_dataset(num_examples=100)
    
    # Print statistics
    stats = builder.get_dataset_statistics()
    print("\n=== Dataset Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")