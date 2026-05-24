from typing import List, Dict, Any, Optional
from src.parser.call_graph import CallGraph
from src.dataset.enrichment_engine import EnrichedCallGraph, EnrichmentEngine


class DPOExample:
    """Represents a DPO training example."""
    
    def __init__(self, prompt: str, chosen: str, rejected: str):
        self.prompt = prompt
        self.chosen = chosen
        self.rejected = rejected

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected
        }


class DPOFormatter:
    """
    Formats enriched call graphs into DPO training examples.
    
    DPO (Direct Preference Optimization) requires pairs of responses where:
    - chosen: The preferred (correctly enriched) version
    - rejected: The non-preferred (raw or incorrectly enriched) version
    
    The formatter creates multiple types of training examples:
    1. Proxy Resolution: Raw proxy class vs resolved real class
    2. Interceptor Detection: Raw stack vs with inferred interceptors
    3. Async Boundary: Raw trace vs with async boundaries marked
    4. Remote Call: Raw trace vs with external service nodes
    5. Meta Enrichment: Basic trace vs with severity/module/layer info
    """

    def __init__(self):
        self.enrichment_engine = EnrichmentEngine()

    def format_dpo_examples(self, rcg: CallGraph, context: Dict) -> List[DPOExample]:
        """
        Generate multiple DPO examples from a single trace.
        
        Args:
            rcg: Raw Call Graph from parser
            context: Static analysis context
            
        Returns:
            List of DPO examples for different enrichment types
        """
        examples = []
        
        # Generate examples for each enrichment type
        examples.extend(self._generate_proxy_examples(rcg, context))
        examples.extend(self._generate_interceptor_examples(rcg, context))
        examples.extend(self._generate_async_examples(rcg, context))
        examples.extend(self._generate_remote_examples(rcg, context))
        examples.extend(self._generate_meta_examples(rcg, context))
        
        return examples

    def _generate_proxy_examples(self, rcg: CallGraph, context: Dict) -> List[DPOExample]:
        """Generate DPO examples for proxy resolution."""
        examples = []
        
        # Check if there are any proxy frames
        has_proxy = any(frame.is_proxy or "$Proxy" in frame.class_name or "Enhancer" in frame.class_name 
                       for frame in rcg.frames)
        
        if not has_proxy:
            return examples

        # Get enriched version with proxy resolution
        ecg = self.enrichment_engine.apply_enrichment(rcg, context, "E-PROXY")
        
        # Build raw trace text
        raw_trace_text = self._format_trace_raw(rcg)
        
        # Build enriched trace text
        enriched_trace_text = self._format_trace_enriched(ecg, highlight="proxy")
        
        # Create prompt
        prompt = self._build_prompt("proxy", raw_trace_text)
        
        examples.append(DPOExample(
            prompt=prompt,
            chosen=enriched_trace_text,
            rejected=raw_trace_text
        ))
        
        return examples

    def _generate_interceptor_examples(self, rcg: CallGraph, context: Dict) -> List[DPOExample]:
        """Generate DPO examples for interceptor detection."""
        examples = []
        
        # Check if there are any interceptor patterns
        has_interceptor = any("Interceptor" in frame.class_name or "Transaction" in frame.class_name
                           for frame in rcg.frames)
        
        # Also check context for transactional annotations
        if not has_interceptor:
            annotations = context.get("annotations", {})
            for class_info in annotations.values():
                if "Transactional" in str(class_info.get("annotations", [])):
                    has_interceptor = True
                    break
        
        if not has_interceptor:
            return examples

        # Get enriched version with interceptor detection
        ecg = self.enrichment_engine.apply_enrichment(rcg, context, "E-INTERCEPTOR")
        
        # Build raw trace text
        raw_trace_text = self._format_trace_raw(rcg)
        
        # Build enriched trace text
        enriched_trace_text = self._format_trace_enriched(ecg, highlight="interceptor")
        
        # Create prompt
        prompt = self._build_prompt("interceptor", raw_trace_text)
        
        examples.append(DPOExample(
            prompt=prompt,
            chosen=enriched_trace_text,
            rejected=raw_trace_text
        ))
        
        return examples

    def _generate_async_examples(self, rcg: CallGraph, context: Dict) -> List[DPOExample]:
        """Generate DPO examples for async boundary detection."""
        examples = []
        
        # Check if there are any async patterns
        async_patterns = ['Future', 'CompletableFuture', 'Async', '@Async']
        has_async = any(any(pattern in frame.class_name or pattern in frame.method_name 
                           for pattern in async_patterns) for frame in rcg.frames)
        
        if not has_async:
            return examples

        # Get enriched version with async detection
        ecg = self.enrichment_engine.apply_enrichment(rcg, context, "E-ASYNC")
        
        # Build raw trace text
        raw_trace_text = self._format_trace_raw(rcg)
        
        # Build enriched trace text
        enriched_trace_text = self._format_trace_enriched(ecg, highlight="async")
        
        # Create prompt
        prompt = self._build_prompt("async", raw_trace_text)
        
        examples.append(DPOExample(
            prompt=prompt,
            chosen=enriched_trace_text,
            rejected=raw_trace_text
        ))
        
        return examples

    def _generate_remote_examples(self, rcg: CallGraph, context: Dict) -> List[DPOExample]:
        """Generate DPO examples for remote call detection."""
        examples = []
        
        # Check if there are any remote patterns
        remote_patterns = ['Feign', 'RestClient', 'HttpClient', 'Remote', 'Exception']
        has_remote = any(any(pattern in frame.class_name for pattern in remote_patterns) 
                       for frame in rcg.frames)
        
        if not has_remote:
            return examples

        # Get enriched version with remote detection
        ecg = self.enrichment_engine.apply_enrichment(rcg, context, "E-REMOTE")
        
        # Build raw trace text
        raw_trace_text = self._format_trace_raw(rcg)
        
        # Build enriched trace text
        enriched_trace_text = self._format_trace_enriched(ecg, highlight="remote")
        
        # Create prompt
        prompt = self._build_prompt("remote", raw_trace_text)
        
        examples.append(DPOExample(
            prompt=prompt,
            chosen=enriched_trace_text,
            rejected=raw_trace_text
        ))
        
        return examples

    def _generate_meta_examples(self, rcg: CallGraph, context: Dict) -> List[DPOExample]:
        """Generate DPO examples for meta enrichment."""
        examples = []
        
        # Always generate meta examples as they add value to any trace
        # Get enriched version with meta info
        ecg = self.enrichment_engine.apply_enrichment(rcg, context, "E-META")
        
        # Build raw trace text
        raw_trace_text = self._format_trace_raw(rcg)
        
        # Build enriched trace text
        enriched_trace_text = self._format_trace_enriched(ecg, highlight="meta")
        
        # Create prompt
        prompt = self._build_prompt("meta", raw_trace_text)
        
        examples.append(DPOExample(
            prompt=prompt,
            chosen=enriched_trace_text,
            rejected=raw_trace_text
        ))
        
        return examples

    def _build_prompt(self, enrichment_type: str, raw_trace: str) -> str:
        """Build a prompt for the given enrichment type."""
        prompts = {
            "proxy": f"""Enrich the following Java stack trace by resolving proxy classes to their real implementations:

<trace>
{raw_trace}
</trace>

Provide the enriched call graph with proxy resolution.""",
            
            "interceptor": f"""Enrich the following Java stack trace by identifying and inserting interceptor nodes (TransactionInterceptor, RetryTemplate, etc.):

<trace>
{raw_trace}
</trace>

Provide the enriched call graph with interceptor information.""",
            
            "async": f"""Enrich the following Java stack trace by identifying async boundaries:

<trace>
{raw_trace}
</trace>

Provide the enriched call graph with async boundary markers.""",
            
            "remote": f"""Enrich the following Java stack trace by identifying remote calls and external service representations:

<trace>
{raw_trace}
</trace>

Provide the enriched call graph with remote service information.""",
            
            "meta": f"""Enrich the following Java stack trace by adding meta information (module, severity, layer):

<trace>
{raw_trace}
</trace>

Provide the enriched call graph with meta annotations."""
        }
        
        return prompts.get(enrichment_type, f"""Enrich the following Java stack trace:

<trace>
{raw_trace}
</trace>

Provide the enriched call graph.""")

    def _format_trace_raw(self, rcg: CallGraph) -> str:
        """Format raw call graph as text."""
        lines = []
        if rcg.exception_type:
            lines.append(f"{rcg.exception_type}: {rcg.exception_message or ''}")
        
        for i, frame in enumerate(rcg.frames):
            line_info = f"at {frame.class_name}.{frame.method_name}"
            if frame.file_name:
                line_info += f"({frame.file_name}"
                if frame.line_number:
                    line_info += f":{frame.line_number}"
                line_info += ")"
            lines.append(f"    {line_info}")
        
        return "\n".join(lines)

    def _format_trace_enriched(self, ecg: EnrichmentEngine, highlight: str) -> str:
        """Format enriched call graph as text with specific highlight."""
        lines = []
        
        for node in ecg.nodes:
            base_info = f"{node.class_name}.{node.method_name}"
            
            # Add enrichment markers
            enrichment = node.enrichment
            if enrichment:
                if highlight == "proxy" and enrichment.get("type") == "proxy":
                    base_info += f" [PROXY→{enrichment.get('resolved_to', '?')}]"
                elif highlight == "interceptor" and enrichment.get("type") == "interceptor":
                    base_info += f" [INTERCEPTOR:{enrichment.get('interceptor_type')}]"
                elif highlight == "async" and enrichment.get("type") == "async":
                    base_info += f" [ASYNC:{enrichment.get('pattern')}]"
                elif highlight == "remote" and enrichment.get("type") == "remote":
                    base_info += f" [REMOTE:{enrichment.get('remote_type')}]"
                elif highlight == "meta" and enrichment.get("type") == "meta":
                    base_info += f" [MODULE:{enrichment.get('module')}, SEVERITY:{enrichment.get('severity')}, LAYER:{enrichment.get('layer')}]"
            
            lines.append(f"    {base_info}")
        
        return "\n".join(lines)

    def format_to_jsonl(self, examples: List[DPOExample], output_file: str):
        """Write DPO examples to JSONL file."""
        import json
        
        with open(output_file, 'w') as f:
            for example in examples:
                f.write(json.dumps(example.to_dict()) + '\n')

    def format_to_hf_dataset(self, examples: List[DPOExample]) -> Dict[str, List]:
        """Format examples for Hugging Face DPO trainer."""
        return {
            "prompt": [ex.prompt for ex in examples],
            "chosen": [ex.chosen for ex in examples],
            "rejected": [ex.rejected for ex in examples]
        }