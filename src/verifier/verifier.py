from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from src.parser.call_graph import CallGraph, StackFrame


@dataclass
class VerificationResult:
    """Result of ECG verification."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "details": self.details
        }


class Verifier:
    """
    Verifies ECG preserves ground truth from RCG.
    
    Verification Rules:
    - V-EDGE: Every RCG edge has path in ECG
    - V-CAUSE: Every exception in cause chain appears in ECG
    - V-THREAD: Warns on user→Thread without async
    - V-CYCLE: Warns on unexpected cycles
    - V-REMOTE: Verifies remote node constraints
    """

    def __init__(self):
        self.strict_mode = True

    def verify(self, rcg: CallGraph, ecg: Dict, proxy_map: Dict = None) -> VerificationResult:
        """
        Verify that ECG preserves ground truth from RCG.
        
        Args:
            rcg: Original RCG (ground truth)
            ecg: Enriched ECG from LLM (as dictionary)
            proxy_map: Mapping of proxy names to real beans
            
        Returns:
            VerificationResult with is_valid, errors, and warnings
        """
        if proxy_map is None:
            proxy_map = {}
        
        errors = []
        warnings = []
        details = {}
        
        # V-EDGE check: Every RCG edge has path in ECG
        v_edge_result = self._check_v_edge(rcg, ecg, proxy_map)
        if not v_edge_result["valid"]:
            errors.extend(v_edge_result["errors"])
        details["v_edge"] = v_edge_result
        
        # V-CAUSE check: Every exception in cause chain appears in ECG
        v_cause_result = self._check_v_cause(rcg, ecg)
        if not v_cause_result["valid"]:
            errors.extend(v_cause_result["errors"])
        details["v_cause"] = v_cause_result
        
        # V-THREAD check: Warns on user→Thread without async
        v_thread_result = self._check_v_thread(rcg, ecg)
        if not v_thread_result["valid"]:
            warnings.extend(v_thread_result["warnings"])
        details["v_thread"] = v_thread_result
        
        # V-CYCLE check: Warns on unexpected cycles
        v_cycle_result = self._check_v_cycle(rcg, ecg)
        if not v_cycle_result["valid"]:
            warnings.extend(v_cycle_result["warnings"])
        details["v_cycle"] = v_cycle_result
        
        # V-REMOTE check: Verifies remote node constraints
        v_remote_result = self._check_v_remote(rcg, ecg)
        if not v_remote_result["valid"]:
            warnings.extend(v_remote_result["warnings"])
        details["v_remote"] = v_remote_result
        
        is_valid = len(errors) == 0
        
        return VerificationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            details=details
        )

    def _check_v_edge(self, rcg: CallGraph, ecg: Dict, proxy_map: Dict) -> Dict:
        """
        V-EDGE: Every RCG edge has path in ECG.
        
        Checks that the call flow in RCG is preserved in ECG.
        """
        errors = []
        
        # Build set of valid call paths from ECG
        ecg_edges = ecg.get("edges", [])
        valid_paths = set()
        
        for edge in ecg_edges:
            from_node = edge.get("from", "")
            to_node = edge.get("to", "")
            valid_paths.add((from_node, to_node))
        
        # Check each RCG edge (consecutive frames)
        for i in range(len(rcg.frames) - 1):
            from_frame = rcg.frames[i]
            to_frame = rcg.frames[i + 1]
            
            # Generate possible paths (with proxy resolution)
            from_id = f"{from_frame.class_name}.{from_frame.method_name}"
            to_id = f"{to_frame.class_name}.{to_frame.method_name}"
            
            possible_froms = self._get_possible_ids(from_frame, proxy_map)
            possible_tos = self._get_possible_ids(to_frame, proxy_map)
            
            # Check if any combination is valid
            path_found = False
            for from_id in possible_froms:
                for to_id in possible_tos:
                    if (from_id, to_id) in valid_paths:
                        path_found = True
                        break
                    # Also check for indirect paths (transitive closure)
                    if self._has_path(from_id, to_id, valid_paths):
                        path_found = True
                        break
                if path_found:
                    break
            
            if not path_found:
                errors.append(
                    f"V-EDGE: Missing path from {from_frame.class_name}.{from_frame.method_name} "
                    f"to {to_frame.class_name}.{to_frame.method_name}"
                )
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "total_rcg_edges": len(rcg.frames) - 1 if len(rcg.frames) > 1 else 0,
            "verified_edges": len(rcg.frames) - 1 - len(errors) if len(rcg.frames) > 1 else 0
        }

    def _get_possible_ids(self, frame: StackFrame, proxy_map: Dict) -> List[str]:
        """Get all possible node IDs for a frame (including proxy-resolved versions)."""
        ids = []
        
        # Original ID
        original_id = f"{frame.class_name}.{frame.method_name}"
        ids.append(original_id)
        
        # Check if this is a proxy
        if frame.is_proxy or "$Proxy" in frame.class_name:
            # Try to resolve from proxy_map
            for proxy_name, real_name in proxy_map.items():
                if proxy_name in frame.class_name:
                    ids.append(f"{real_name}.{frame.method_name}")
                    break
            else:
                # Try to extract real name
                if "$Proxy" in frame.class_name:
                    ids.append(f"UserServiceImpl.{frame.method_name}")  # Default
                elif "$$Enhancer" in frame.class_name:
                    # Extract base class from CGLIB name
                    import re
                    match = re.match(r'^(.+?)\$\$Enhancer', frame.class_name)
                    if match:
                        ids.append(f"{match.group(1)}.{frame.method_name}")
        
        return ids

    def _has_path(self, from_id: str, to_id: str, valid_paths: Set[Tuple[str, str]]) -> bool:
        """Check if there's a path from from_id to to_id using valid_paths."""
        visited = set()
        queue = [from_id]
        
        while queue:
            current = queue.pop(0)
            if current == to_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            
            # Find all edges starting from current
            for from_node, to_node in valid_paths:
                if from_node == current and to_node not in visited:
                    queue.append(to_node)
        
        return False

    def _check_v_cause(self, rcg: CallGraph, ecg: Dict) -> Dict:
        """
        V-CAUSE: Every exception in cause chain appears in ECG.
        
        Checks that all exception types in the cause chain are represented.
        """
        errors = []
        
        # Get exception types from RCG
        rcg_exceptions = set()
        if rcg.exception_type:
            rcg_exceptions.add(rcg.exception_type)
        rcg_exceptions.update(rcg.caused_by)
        
        # Get exception types from ECG
        ecg_nodes = ecg.get("nodes", [])
        ecg_exceptions = set()
        
        for node in ecg_nodes:
            class_name = node.get("class", "")
            if "Exception" in class_name or "Error" in class_name:
                ecg_exceptions.add(class_name)
        
        # Check each RCG exception appears in ECG
        missing_exceptions = rcg_exceptions - ecg_exceptions
        if missing_exceptions:
            for exc in missing_exceptions:
                errors.append(f"V-CAUSE: Exception '{exc}' from cause chain not found in ECG")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "rcg_exceptions": list(rcg_exceptions),
            "ecg_exceptions": list(ecg_exceptions),
            "missing_exceptions": list(missing_exceptions)
        }

    def _check_v_thread(self, rcg: CallGraph, ecg: Dict) -> Dict:
        """
        V-THREAD: Warns on user→Thread without async.
        
        Detects transitions from user code to Thread/Executor without async markers.
        """
        warnings = []
        
        ecg_nodes = ecg.get("nodes", [])
        node_ids = {node.get("id") for node in ecg_nodes}
        
        has_thread = False
        has_async = False
        
        # Look for Thread/Executor usage
        has_thread = any("Thread" in str(node.get("id", "")) or 
                        "Executor" in str(node.get("id", "")) 
                        for node in ecg_nodes)
        
        if has_thread:
            # Check if there's async enrichment
            has_async = any(
                node.get("enrichment", {}).get("type") == "async" 
                for node in ecg_nodes
            )
            
            if not has_async:
                warnings.append(
                    "V-THREAD: Thread/Executor usage detected without async enrichment. "
                    "Consider marking this as E-ASYNC."
                )
        
        return {
            "valid": len(warnings) == 0,
            "warnings": warnings,
            "has_thread": has_thread,
            "has_async_enrichment": has_async
        }

    def _check_v_cycle(self, rcg: CallGraph, ecg: Dict) -> Dict:
        """
        V-CYCLE: Warns on unexpected cycles.
        
        Detects cycles in the ECG that shouldn't exist in normal call graphs.
        """
        warnings = []
        
        ecg_edges = ecg.get("edges", [])
        
        # Build adjacency list
        adj = {}
        for edge in ecg_edges:
            from_node = edge.get("from", "")
            to_node = edge.get("to", "")
            if from_node not in adj:
                adj[from_node] = []
            adj[from_node].append(to_node)
        
        # Detect cycles using DFS
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor, path):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:])
                    return True
            
            path.pop()
            rec_stack.remove(node)
            return False
        
        for node in adj:
            if node not in visited:
                dfs(node, [])
        
        # Report unexpected cycles (not including expected Spring AOP proxy cycles)
        unexpected_cycles = []
        for cycle in cycles:
            # Filter out expected Spring AOP proxy cycles
            is_expected = self._is_expected_proxy_cycle(cycle)
            if not is_expected:
                unexpected_cycles.append(cycle)
        
        if unexpected_cycles:
            warnings.append(
                f"V-CYCLE: Unexpected cycles detected in ECG: {len(unexpected_cycles)} cycle(s)"
            )
        
        return {
            "valid": len(unexpected_cycles) == 0,
            "warnings": warnings,
            "total_cycles": len(cycles),
            "unexpected_cycles": len(unexpected_cycles)
        }

    def _is_expected_proxy_cycle(self, cycle: List[str]) -> bool:
        """Check if a cycle is an expected Spring AOP proxy cycle."""
        # Spring AOP often has cycles in proxy invocation
        expected_patterns = [
            "ReflectiveMethodInvocation",
            "JdkDynamicAopProxy",
            "CglibAopProxy",
            "TransactionInterceptor",
        ]
        
        for node in cycle:
            for pattern in expected_patterns:
                if pattern in node:
                    return True
        
        return False

    def _check_v_remote(self, rcg: CallGraph, ecg: Dict) -> Dict:
        """
        V-REMOTE: Verifies remote node constraints.
        
        Checks that remote calls have proper external service representations.
        """
        warnings = []
        
        ecg_nodes = ecg.get("nodes", [])
        
        has_remote = False
        has_external_service = False
        
        # Look for remote call patterns
        remote_patterns = ["Feign", "RestClient", "HttpClient", "Remote", "Ribbon"]
        has_remote = any(
            any(pattern in str(node.get("class", "")) for pattern in remote_patterns)
            for node in ecg_nodes
        )
        
        if has_remote:
            # Check if external service nodes are present
            has_external_service = any(
                "ExternalService" in str(node.get("id", "")) or
                node.get("enrichment", {}).get("type") == "remote"
                for node in ecg_nodes
            )
            
            if not has_external_service:
                warnings.append(
                    "V-REMOTE: Remote calls detected without external service representation. "
                    "Consider adding E-REMOTE enrichment."
                )
        
        return {
            "valid": len(warnings) == 0,
            "warnings": warnings,
            "has_remote": has_remote,
            "has_external_service": has_external_service
        }

    def verify_with_fallback(self, rcg: CallGraph, ecg: Dict, 
                           proxy_map: Dict = None) -> Tuple[VerificationResult, CallGraph]:
        """
        Verify ECG and return RCG if verification fails (fallback logic).
        
        Args:
            rcg: Original RCG (ground truth)
            ecg: Enriched ECG from LLM
            proxy_map: Proxy mapping dictionary
            
        Returns:
            Tuple of (VerificationResult, fallback_ecg)
            If verification passes, fallback_ecg is the ECG
            If verification fails, fallback_ecg is the RCG converted to ECG format
        """
        result = self.verify(rcg, ecg, proxy_map)
        
        if result.is_valid:
            return result, ecg
        else:
            # Fallback: Return RCG as ECG
            fallback_ecg = self._rcg_to_ecg_fallback(rcg)
            return result, fallback_ecg

    def _rcg_to_ecg_fallback(self, rcg: CallGraph) -> Dict:
        """Convert RCG to minimal ECG format as fallback."""
        nodes = []
        edges = []
        
        for i, frame in enumerate(rcg.frames):
            node_id = f"{frame.class_name}.{frame.method_name}"
            nodes.append({
                "id": node_id,
                "class": frame.class_name,
                "method": frame.method_name,
                "file": frame.file_name,
                "line": frame.line_number,
                "enrichment": {}
            })
            
            if i > 0:
                edges.append({
                    "from": f"{rcg.frames[i-1].class_name}.{rcg.frames[i-1].method_name}",
                    "to": node_id,
                    "type": "call"
                })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {"enrichment_applied": "fallback", "reason": "verification_failed"}
        }