from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json


@dataclass
class StackFrame:
    class_name: str
    method_name: str
    file_name: Optional[str] = None
    line_number: Optional[int] = None
    is_proxy: bool = False
    proxy_type: Optional[str] = None


@dataclass
class CallGraph:
    frames: List[StackFrame] = field(default_factory=list)
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    caused_by: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert CallGraph to dictionary for JSON serialization."""
        nodes = []
        edges = []
        
        for i, frame in enumerate(self.frames):
            node_id = f"{frame.class_name}.{frame.method_name}"
            nodes.append({
                "id": node_id,
                "class": frame.class_name,
                "method": frame.method_name,
                "file": frame.file_name,
                "line": frame.line_number,
                "is_proxy": frame.is_proxy,
                "proxy_type": frame.proxy_type
            })
            
            if i < len(self.frames) - 1:
                next_node_id = f"{self.frames[i+1].class_name}.{self.frames[i+1].method_name}"
                edges.append({
                    "from": node_id,
                    "to": next_node_id,
                    "type": "call"
                })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "exception": self.exception_type,
                "message": self.exception_message,
                "caused_by": self.caused_by
            }
        }

    def to_json(self) -> str:
        """Convert CallGraph to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CallGraph':
        """Create CallGraph from dictionary."""
        frames = []
        for node in data.get("nodes", []):
            frames.append(StackFrame(
                class_name=node["class"],
                method_name=node["method"],
                file_name=node.get("file"),
                line_number=node.get("line"),
                is_proxy=node.get("is_proxy", False),
                proxy_type=node.get("proxy_type")
            ))
        
        metadata = data.get("metadata", {})
        return cls(
            frames=frames,
            exception_type=metadata.get("exception"),
            exception_message=metadata.get("message"),
            caused_by=metadata.get("caused_by", [])
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'CallGraph':
        """Create CallGraph from JSON string."""
        return cls.from_dict(json.loads(json_str))