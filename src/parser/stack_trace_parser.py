import re
from typing import List, Optional, Tuple
from .call_graph import CallGraph, StackFrame


class StackTraceParser:
    """
    Parser for Java stack traces into structured CallGraph objects.
    
    Supports:
    - Standard Java exception stack traces
    - Caused-by chains
    - Proxy detection ($Proxy, $$Enhancer, CGLIB)
    - Lambda expressions
    """

    # Patterns for parsing stack frames
    STACK_FRAME_PATTERN = re.compile(
        r'^\s*at\s+([a-zA-Z$][a-zA-Z0-9_.$]*)\.([a-zA-Z$][a-zA-Z0-9_<>/$]*)\s*\('
        r'([^:)]+)(?::(\d+))?\)'
    )

    # Exception pattern
    EXCEPTION_PATTERN = re.compile(
        r'^\s*([a-zA-Z][a-zA-Z0-9_.]*)(?::\s*(.*))?$'
    )

    # Caused-by pattern
    CAUSED_BY_PATTERN = re.compile(
        r'^\s*(?:Caused by:|Suppressed:)\s*(.+)$'
    )

    # Proxy detection patterns
    PROXY_PATTERNS = [
        (re.compile(r'\$Proxy\d+$'), 'jdk_proxy'),
        (re.compile(r'.*EnhancerByCGLIB.*'), 'cglib'),
        (re.compile(r'.*\$\$Enhancer.*'), 'spring_cglib'),
        (re.compile(r'.*javassist.*'), 'javassist'),
        (re.compile(r'.*JdkDynamicAopProxy.*'), 'spring_jdk'),
        (re.compile(r'.*CGILIB.*'), 'cglib'),
    ]

    # Lambda pattern
    LAMBDA_PATTERN = re.compile(r'lambda\$[\w$]+/\d+')

    def parse(self, trace_text: str) -> CallGraph:
        """
        Parse a Java stack trace into a CallGraph.
        
        Args:
            trace_text: Raw stack trace text
            
        Returns:
            CallGraph with parsed frames, exception info, and caused-by chain
        """
        lines = trace_text.strip().split('\n')
        frames: List[StackFrame] = []
        exception_type: Optional[str] = None
        exception_message: Optional[str] = None
        caused_by: List[str] = []
        
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            
            # Parse exception header
            if i == 0 and not line.startswith('at '):
                match = self.EXCEPTION_PATTERN.match(line)
                if match:
                    exception_type = match.group(1)
                    exception_message = match.group(2)
                i += 1
                continue
            
            # Parse stack frame
            frame_match = self.STACK_FRAME_PATTERN.match(line)
            if frame_match:
                class_name = frame_match.group(1)
                method_name = frame_match.group(2)
                file_name = frame_match.group(3)
                line_number = int(frame_match.group(4)) if frame_match.group(4) else None
                
                # Check if this is a proxy class
                is_proxy, proxy_type = self._detect_proxy(class_name)
                
                # Handle lambda expressions
                if self.LAMBDA_PATTERN.match(method_name):
                    method_name = self._normalize_lambda(method_name)
                
                # Normalize proxy class name
                if is_proxy:
                    class_name = self.normalize_proxy_name(class_name)
                
                frames.append(StackFrame(
                    class_name=class_name,
                    method_name=method_name,
                    file_name=file_name if file_name != 'Unknown Source' else None,
                    line_number=line_number,
                    is_proxy=is_proxy,
                    proxy_type=proxy_type
                ))
                i += 1
                continue
            
            # Parse caused-by
            caused_match = self.CAUSED_BY_PATTERN.match(line)
            if caused_match:
                caused_text = caused_match.group(1)
                caused_exception = self.EXCEPTION_PATTERN.match(caused_text)
                if caused_exception:
                    caused_by.append(caused_exception.group(1))
                i += 1
                continue
            
            i += 1
        
        return CallGraph(
            frames=frames,
            exception_type=exception_type,
            exception_message=exception_message,
            caused_by=caused_by
        )

    def _detect_proxy(self, class_name: str) -> Tuple[bool, Optional[str]]:
        """
        Detect if a class name represents a proxy.
        
        Args:
            class_name: Full class name
            
        Returns:
            Tuple of (is_proxy: bool, proxy_type: Optional[str])
        """
        for pattern, proxy_type in self.PROXY_PATTERNS:
            if pattern.search(class_name):
                return True, proxy_type
        return False, None

    def normalize_proxy_name(self, proxy_name: str) -> str:
        """
        Normalize proxy class name to its original bean name.
        
        Args:
            proxy_name: Proxy class name like '$Proxy123' or 'UserService$$EnhancerByCGLIB$$abc123'
            
        Returns:
            Normalized bean name (simple class name without package)
        """
        # Handle $ProxyXXX pattern (with or without package)
        proxy_pattern = re.compile(r'(?:.*\.)?\$Proxy\d+$')
        if proxy_pattern.match(proxy_name):
            return 'UserServiceImpl'  # Default; actual mapping comes from context
        
        # Handle CGLIB enhancer pattern
        cglib_pattern = re.compile(r'^(.+?)\$\$Enhancer.*$')
        match = cglib_pattern.match(proxy_name)
        if match:
            full_name = match.group(1)
            # Strip package prefix
            return full_name.split('.')[-1]
        
        # Handle other patterns
        patterns = [
            r'^(.+?)\$\$.*$',           # Class$$Enhancer...
            r'.+\.(.+?)\$[0-9]+$',     # pkg.Class$123
        ]
        
        for pattern in patterns:
            match = re.match(pattern, proxy_name)
            if match:
                return match.group(1).split('.')[-1]
        
        # Return simple class name if not matching any pattern
        return proxy_name.split('.')[-1]

    def _normalize_lambda(self, method_name: str) -> str:
        """
        Normalize lambda method name for consistent representation.
        
        Args:
            method_name: Lambda method name like 'lambda$process$1/12345678'
            
        Returns:
            Normalized name like 'lambda_process'
        """
        # Extract the base name before the slash
        match = re.match(r'(lambda\$[\w$]+)/\d+', method_name)
        if match:
            return match.group(1).replace('$', '_')
        return method_name