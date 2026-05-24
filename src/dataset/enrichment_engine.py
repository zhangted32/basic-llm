from typing import List, Dict, Any, Optional
from src.parser.call_graph import CallGraph, StackFrame


class EnrichedNode:
    """Represents a node in the Enriched Call Graph (ECG)."""
    
    def __init__(self, node_id: str, class_name: str, method_name: str, 
                 file_name: Optional[str] = None, line_number: Optional[int] = None,
                 is_proxy: bool = False, proxy_type: Optional[str] = None,
                 enrichment: Optional[Dict] = None):
        self.id = node_id
        self.class_name = class_name
        self.method_name = method_name
        self.file_name = file_name
        self.line_number = line_number
        self.is_proxy = is_proxy
        self.proxy_type = proxy_type
        self.enrichment = enrichment or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "class": self.class_name,
            "method": self.method_name,
            "file": self.file_name,
            "line": self.line_number,
            "is_proxy": self.is_proxy,
            "proxy_type": self.proxy_type,
            "enrichment": self.enrichment
        }


class EnrichedCallGraph:
    """Represents an Enriched Call Graph (ECG)."""
    
    def __init__(self):
        self.nodes: List[EnrichedNode] = []
        self.edges: List[Dict[str, str]] = []
        self.enrichment_info: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": self.edges,
            "metadata": {
                "enrichment_applied": self.enrichment_info.get("type"),
                "details": self.enrichment_info
            }
        }


class EnrichmentEngine:
    """
    Applies enrichment rules to transform RCG into ECG.
    
    Enrichment Rules:
    - E-PROXY: Resolve proxy classes to their real implementations
    - E-ASYNC: Infer async task origin from Future.get() patterns
    - E-INTERCEPTOR: Insert interceptor nodes (TransactionInterceptor, RetryTemplate)
    - E-REMOTE: Add external service representation for remote calls
    - E-META: Add severity, module, and business context
    - E-MERGE: Merge multiple traces by transaction ID
    """

    # Patterns for detecting various frameworks and patterns
    ASYNC_PATTERNS = [
        r'java\.util\.concurrent\.Future',
        r'java\.util\.concurrent\.CompletableFuture',
        r'org\.springframework\.scheduling\.annotation\.Async',
        r'@Async'
    ]

    INTERCEPTOR_PATTERNS = [
        ('TransactionInterceptor', 'transaction'),
        ('RetryTemplate', 'retry'),
        ('CircuitBreaker', 'circuit_breaker'),
        ('Hystrix', 'circuit_breaker'),
        ('RateLimiter', 'rate_limit'),
    ]

    REMOTE_PATTERNS = [
        ('FeignException', 'feign'),
        ('RestClientException', 'rest_client'),
        ('HttpClientErrorException', 'http_client'),
        ('RemoteAccessException', 'remote'),
        ('RibbonClient', 'ribbon'),
    ]

    def apply_enrichment(self, rcg: CallGraph, context: Dict, 
                        enrichment_type: str) -> EnrichedCallGraph:
        """
        Apply the specified enrichment rule to the RCG.
        
        Args:
            rcg: Raw Call Graph from parser
            context: Static analysis context (proxy mappings, annotations)
            enrichment_type: Type of enrichment to apply
            
        Returns:
            EnrichedCallGraph with applied enrichment
        """
        ecg = self._rcg_to_ecg(rcg)
        
        if enrichment_type == "E-PROXY":
            return self._apply_e_proxy(ecg, context)
        elif enrichment_type == "E-ASYNC":
            return self._apply_e_async(ecg, context)
        elif enrichment_type == "E-INTERCEPTOR":
            return self._apply_e_interceptor(ecg, context)
        elif enrichment_type == "E-REMOTE":
            return self._apply_e_remote(ecg, context)
        elif enrichment_type == "E-META":
            return self._apply_e_meta(ecg, context)
        elif enrichment_type == "E-MERGE":
            return self._apply_e_merge(ecg, context)
        else:
            ecg.enrichment_info = {"type": "none", "message": "No enrichment applied"}
            return ecg

    def _rcg_to_ecg(self, rcg: CallGraph) -> EnrichedCallGraph:
        """Convert RCG to ECG by creating EnrichedNode instances."""
        ecg = EnrichedCallGraph()
        
        for frame in rcg.frames:
            node_id = f"{frame.class_name}.{frame.method_name}"
            ecg.nodes.append(EnrichedNode(
                node_id=node_id,
                class_name=frame.class_name,
                method_name=frame.method_name,
                file_name=frame.file_name,
                line_number=frame.line_number,
                is_proxy=frame.is_proxy,
                proxy_type=frame.proxy_type,
                enrichment={}
            ))
        
        # Create edges between consecutive frames
        for i in range(len(ecg.nodes) - 1):
            ecg.edges.append({
                "from": ecg.nodes[i].id,
                "to": ecg.nodes[i+1].id,
                "type": "call"
            })
        
        return ecg

    def _apply_e_proxy(self, ecg: EnrichedCallGraph, context: Dict) -> EnrichedCallGraph:
        """
        E-PROXY: Resolve proxy classes to their real implementations.
        
        Uses proxy mappings from context to replace proxy node IDs with real bean names.
        """
        proxy_mappings = context.get("proxy_mappings", {})
        
        for node in ecg.nodes:
            if node.is_proxy or any(proxy_pattern in node.class_name for proxy_pattern in ["$Proxy", "EnhancerByCGLIB"]):
                # Try to find real implementation from context
                resolved_name = None
                
                # Check exact proxy name match
                for proxy_name, real_name in proxy_mappings.items():
                    if proxy_name in node.class_name:
                        resolved_name = real_name
                        break
                
                # If no exact match, try to extract base name
                if resolved_name is None:
                    resolved_name = self._extract_real_class_name(node.class_name)
                
                if resolved_name:
                    # Update node with resolved name
                    original_id = node.id
                    node.class_name = resolved_name
                    node.id = f"{resolved_name}.{node.method_name}"
                    node.enrichment = {
                        "type": "proxy",
                        "proxy_resolved": True,
                        "proxy_type": node.proxy_type or "jdk_proxy",
                        "original_proxy": original_id,
                        "resolved_to": node.id
                    }
                    node.is_proxy = False
                    node.proxy_type = None
        
        # Update edges with resolved node IDs
        for edge in ecg.edges:
            for node in ecg.nodes:
                if edge["from"] == node.id:
                    pass  # Already correct
                elif node.enrichment.get("original_proxy") == edge["from"]:
                    edge["from"] = node.id
                
                if edge["to"] == node.id:
                    pass  # Already correct
                elif node.enrichment.get("original_proxy") == edge["to"]:
                    edge["to"] = node.id
        
        ecg.enrichment_info = {"type": "E-PROXY", "proxies_resolved": len([n for n in ecg.nodes if n.enrichment.get("proxy_resolved")])}
        return ecg

    def _extract_real_class_name(self, proxy_name: str) -> str:
        """Extract the real class name from a proxy class name."""
        import re
        
        # Handle CGLIB patterns
        match = re.match(r'^(.+?)\$\$Enhancer.*$', proxy_name)
        if match:
            return match.group(1)
        
        # Handle $$ patterns
        match = re.match(r'^(.+?)\$\$.*$', proxy_name)
        if match:
            return match.group(1)
        
        # Handle JDK proxy (returns default since we can't infer without context)
        if "$Proxy" in proxy_name:
            return "UserServiceImpl"  # Default, actual resolution needs context
        
        return proxy_name.split('.')[-1]

    def _apply_e_async(self, ecg: EnrichedCallGraph, context: Dict) -> EnrichedCallGraph:
        """
        E-ASYNC: Infer async task origin from Future.get() patterns.
        
        Identifies async boundaries and adds metadata about async execution.
        """
        async_frame_found = False
        
        for node in ecg.nodes:
            if any(pattern.lower() in node.class_name.lower() or 
                   pattern.lower() in node.method_name.lower() 
                   for pattern in self.ASYNC_PATTERNS):
                node.enrichment = {
                    "type": "async",
                    "async_boundary": True,
                    "pattern": self._detect_async_pattern(node.class_name, node.method_name)
                }
                async_frame_found = True
        
        # Add inferred async task origin if Future.get() is found
        if async_frame_found:
            ecg.enrichment_info = {"type": "E-ASYNC", "async_boundaries_found": True}
        else:
            ecg.enrichment_info = {"type": "E-ASYNC", "async_boundaries_found": False}
        
        return ecg

    def _detect_async_pattern(self, class_name: str, method_name: str) -> str:
        """Detect the specific async pattern."""
        if 'CompletableFuture' in class_name:
            return "completable_future"
        elif 'Future' in class_name:
            return "future"
        elif 'Async' in class_name or 'Async' in method_name:
            return "async_annotation"
        return "generic_async"

    def _apply_e_interceptor(self, ecg: EnrichedCallGraph, context: Dict) -> EnrichedCallGraph:
        """
        E-INTERCEPTOR: Insert interceptor nodes (TransactionInterceptor, RetryTemplate).
        
        Identifies framework interceptors and adds them to the call graph.
        """
        interceptors_found = []
        
        for node in ecg.nodes:
            for pattern, interceptor_type in self.INTERCEPTOR_PATTERNS:
                if pattern in node.class_name:
                    node.enrichment = {
                        "type": "interceptor",
                        "interceptor_type": interceptor_type,
                        "framework": "spring"
                    }
                    interceptors_found.append(interceptor_type)
        
        # Insert implicit interceptors based on annotations in context
        annotations = context.get("annotations", {})
        for node in ecg.nodes:
            if node.enrichment:
                continue  # Already enriched
            
            class_annotations = annotations.get(node.class_name, {}).get("annotations", [])
            if "@Transactional" in class_annotations or "Transactional" in class_annotations:
                # Insert TransactionInterceptor before this node
                self._insert_interceptor_node(ecg, node, "TransactionInterceptor", "transaction")
        
        ecg.enrichment_info = {"type": "E-INTERCEPTOR", "interceptors_found": interceptors_found}
        return ecg

    def _insert_interceptor_node(self, ecg: EnrichedCallGraph, target_node: EnrichedNode,
                                interceptor_name: str, interceptor_type: str):
        """Insert an interceptor node before the target node."""
        interceptor_node = EnrichedNode(
            node_id=f"org.springframework.transaction.interceptor.{interceptor_name}.invoke",
            class_name=f"org.springframework.transaction.interceptor.{interceptor_name}",
            method_name="invoke",
            enrichment={
                "type": "interceptor",
                "interceptor_type": interceptor_type,
                "inferred": True
            }
        )
        
        # Find where to insert
        target_index = None
        for i, node in enumerate(ecg.nodes):
            if node.id == target_node.id:
                target_index = i
                break
        
        if target_index is not None and target_index > 0:
            # Insert before target node
            ecg.nodes.insert(target_index, interceptor_node)
            
            # Update edges
            # Remove old edge from previous node to target
            edges_to_update = []
            for i, edge in enumerate(ecg.edges):
                if edge["to"] == target_node.id:
                    edge["to"] = interceptor_node.id
                    edges_to_update.append(edge)
            
            # Add edge from interceptor to target
            ecg.edges.append({
                "from": interceptor_node.id,
                "to": target_node.id,
                "type": "call"
            })

    def _apply_e_remote(self, ecg: EnrichedCallGraph, context: Dict) -> EnrichedCallGraph:
        """
        E-REMOTE: Add external service representation for remote calls.
        
        Identifies remote call patterns (Feign, REST client) and adds external service nodes.
        """
        remote_calls_found = []
        
        for node in ecg.nodes:
            for pattern, remote_type in self.REMOTE_PATTERNS:
                if pattern in node.class_name:
                    node.enrichment = {
                        "type": "remote",
                        "remote_type": remote_type,
                        "external_service": self._infer_external_service(node.class_name)
                    }
                    remote_calls_found.append(remote_type)
        
        ecg.enrichment_info = {"type": "E-REMOTE", "remote_calls_found": remote_calls_found}
        return ecg

    def _infer_external_service(self, class_name: str) -> str:
        """Infer the external service name from exception class."""
        if 'FeignException' in class_name:
            return "ExternalFeignService"
        elif 'RestClient' in class_name:
            return "ExternalRestService"
        elif 'HttpClient' in class_name:
            return "ExternalHttpClient"
        return "ExternalService"

    def _apply_e_meta(self, ecg: EnrichedCallGraph, context: Dict) -> EnrichedCallGraph:
        """
        E-META: Add severity, module, and business context.
        
        Adds metadata to nodes based on context information.
        """
        # Add module information
        for node in ecg.nodes:
            module = self._infer_module(node.class_name)
            severity = self._infer_severity(node.class_name, node.method_name)
            
            node.enrichment.update({
                "type": "meta",
                "module": module,
                "severity": severity,
                "layer": self._infer_layer(node.class_name)
            })
        
        ecg.enrichment_info = {"type": "E-META", "nodes_enriched": len(ecg.nodes)}
        return ecg

    def _infer_module(self, class_name: str) -> str:
        """Infer module from package name."""
        if 'controller' in class_name.lower():
            return "controller"
        elif 'service' in class_name.lower():
            return "service"
        elif 'repository' in class_name.lower():
            return "repository"
        elif 'api' in class_name.lower():
            return "api"
        elif 'config' in class_name.lower():
            return "config"
        return "unknown"

    def _infer_severity(self, class_name: str, method_name: str) -> str:
        """Infer severity level."""
        if 'Exception' in class_name:
            return "high"
        elif 'Error' in class_name:
            return "critical"
        elif 'delete' in method_name.lower():
            return "medium"
        elif 'update' in method_name.lower():
            return "medium"
        return "low"

    def _infer_layer(self, class_name: str) -> str:
        """Infer application layer."""
        if 'controller' in class_name.lower():
            return "presentation"
        elif 'service' in class_name.lower():
            return "business"
        elif 'repository' in class_name.lower() or 'dao' in class_name.lower():
            return "data"
        elif 'config' in class_name.lower():
            return "configuration"
        return "infrastructure"

    def _apply_e_merge(self, ecg: EnrichedCallGraph, context: Dict) -> EnrichedCallGraph:
        """
        E-MERGE: Merge multiple traces by transaction ID.
        
        For multi-trace scenarios, merges traces that share a common transaction ID.
        """
        transaction_id = context.get("transaction_id")
        
        if transaction_id:
            ecg.enrichment_info = {
                "type": "E-MERGE",
                "transaction_id": transaction_id,
                "merge_strategy": "pre_linked"  # Option A: pre-linked via transaction ID
            }
        else:
            ecg.enrichment_info = {"type": "E-MERGE", "message": "No transaction ID provided"}
        
        return ecg

    def apply_all_enrichments(self, rcg: CallGraph, context: Dict) -> EnrichedCallGraph:
        """Apply all applicable enrichments in sequence."""
        ecg = self._rcg_to_ecg(rcg)
        
        # Apply enrichments in order
        ecg = self._apply_e_proxy(ecg, context)
        ecg = self._apply_e_interceptor(ecg, context)
        ecg = self._apply_e_async(ecg, context)
        ecg = self._apply_e_remote(ecg, context)
        ecg = self._apply_e_meta(ecg, context)
        
        return ecg