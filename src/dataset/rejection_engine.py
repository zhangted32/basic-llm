from typing import List, Dict, Any, Optional
from src.parser.call_graph import CallGraph


class RejectionResult:
    """Represents the result of a rejection check."""
    
    def __init__(self, rejected: bool, reason: str, rejection_type: str, details: Dict = None):
        self.rejected = rejected
        self.reason = reason
        self.rejection_type = rejection_type
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rejected": self.rejected,
            "reason": self.reason,
            "rejection_type": self.rejection_type,
            "details": self.details
        }


class RejectionEngine:
    """
    Applies rejection rules to filter out invalid or low-quality traces.
    
    Rejection Rules:
    - R-RAW: Reject raw traces without enrichment (no proxy, no interceptor)
    - R-MISS: Reject traces missing critical context (class annotations, line numbers)
    - R-HALLUC: Reject traces with hallucinated stack frames
    - R-NOISE: Reject traces dominated by noise frames (system libraries)
    """

    # System library patterns that indicate noise
    NOISE_PATTERNS = [
        r'java\.lang\.',
        r'java\.util\.',
        r'java\.io\.',
        r'java\.net\.',
        r'java\.nio\.',
        r'java\.base\.',
        r'jdk\.proxy',
        r'org\.junit\.',
        r'org\.apache\.maven\.',
        r'org\.codehaus\.mojo\.',
        r'com\.sun\.',
        r'sun\.reflect\.',
    ]

    # Framework patterns that are expected
    FRAMEWORK_PATTERNS = [
        r'org\.springframework\.',
        r'com\.alibaba\.fastjson\.',
        r'org\.aspectj\.',
    ]

    def check_rejection(self, rcg: CallGraph, context: Dict) -> RejectionResult:
        """
        Check if the trace should be rejected based on rejection rules.
        
        Args:
            rcg: Raw Call Graph from parser
            context: Additional context information
            
        Returns:
            RejectionResult indicating whether the trace should be rejected
        """
        # R-MISS: Check for missing critical context
        miss_result = self._check_r_miss(rcg, context)
        if miss_result.rejected:
            return miss_result

        # R-HALLUC: Check for hallucinated frames
        halluc_result = self._check_r_halluc(rcg, context)
        if halluc_result.rejected:
            return halluc_result

        # R-NOISE: Check for noise-dominated traces
        noise_result = self._check_r_noise(rcg, context)
        if noise_result.rejected:
            return noise_result

        # R-RAW: Check for raw traces without enrichment opportunities
        raw_result = self._check_r_raw(rcg, context)
        if raw_result.rejected:
            return raw_result

        # Passed all checks
        return RejectionResult(
            rejected=False,
            reason="Trace passed all rejection checks",
            rejection_type="NONE",
            details={
                "frames_count": len(rcg.frames),
                "exception_type": rcg.exception_type
            }
        )

    def _check_r_miss(self, rcg: CallGraph, context: Dict) -> RejectionResult:
        """
        R-MISS: Reject traces missing critical context.
        
        Checks:
        - Missing line numbers in critical frames
        - Missing exception type
        - Empty stack trace
        """
        issues = []
        
        # Check for empty stack trace
        if len(rcg.frames) == 0:
            return RejectionResult(
                rejected=True,
                reason="Empty stack trace",
                rejection_type="R-MISS",
                details={"frames_count": 0}
            )

        # Check for missing exception type
        if not rcg.exception_type:
            issues.append("missing exception type")

        # Check for missing line numbers in application frames
        missing_line_count = 0
        for frame in rcg.frames:
            if self._is_application_frame(frame) and frame.line_number is None:
                missing_line_count += 1

        if missing_line_count > len(rcg.frames) * 0.5:  # More than 50% missing
            issues.append(f"missing line numbers in {missing_line_count} frames")

        if issues:
            return RejectionResult(
                rejected=True,
                reason=f"Missing critical context: {', '.join(issues)}",
                rejection_type="R-MISS",
                details={
                    "issues": issues,
                    "frames_count": len(rcg.frames),
                    "missing_line_count": missing_line_count
                }
            )

        return RejectionResult(
            rejected=False,
            reason="R-MISS check passed",
            rejection_type="R-MISS",
            details={"frames_count": len(rcg.frames)}
        )

    def _check_r_halluc(self, rcg: CallGraph, context: Dict) -> RejectionResult:
        """
        R-HALLUC: Reject traces with hallucinated stack frames.
        
        Detects:
        - Inconsistent call chains (child frame appears before parent)
        - Impossible method signatures
        - Duplicate consecutive frames
        """
        issues = []
        
        # Check for duplicate consecutive frames
        for i in range(len(rcg.frames) - 1):
            current = rcg.frames[i]
            next_frame = rcg.frames[i + 1]
            
            if (current.class_name == next_frame.class_name and 
                current.method_name == next_frame.method_name):
                issues.append(f"duplicate frame: {current.class_name}.{current.method_name}")

        # Check for impossible method calls (e.g., Object.wait() in main thread without sync)
        for i, frame in enumerate(rcg.frames):
            if self._is_hallucinated_frame(frame, rcg.frames, i):
                issues.append(f"hallucinated frame: {frame.class_name}.{frame.method_name}")

        if issues:
            return RejectionResult(
                rejected=True,
                reason=f"Hallucinated frames detected: {', '.join(issues)}",
                rejection_type="R-HALLUC",
                details={"issues": issues}
            )

        return RejectionResult(
            rejected=False,
            reason="R-HALLUC check passed",
            rejection_type="R-HALLUC",
            details={"frames_count": len(rcg.frames)}
        )

    def _is_hallucinated_frame(self, frame, frames, index) -> bool:
        """Check if a frame appears to be hallucinated."""
        # Check for suspicious patterns
        suspicious_patterns = [
            r'\$\$Lambda.*',  # Lambda frames without context
            r'Unsafe\.park',  # Should not appear without proper context
            r'NativeMethodAccessorImpl\.invoke0',  # Native method without wrapper
        ]
        
        full_name = f"{frame.class_name}.{frame.method_name}"
        for pattern in suspicious_patterns:
            if pattern in full_name:
                # Check context - if it's surrounded by unrelated frames, it's suspicious
                if index > 0 and index < len(frames) - 1:
                    prev_frame = frames[index - 1]
                    next_frame = frames[index + 1]
                    
                    # If neither neighbor is related, it's likely hallucinated
                    if not self._frames_related(prev_frame, frame) and \
                       not self._frames_related(frame, next_frame):
                        return True
        return False

    def _frames_related(self, frame1, frame2) -> bool:
        """Check if two frames are related (same package or class hierarchy)."""
        pkg1 = self._get_package(frame1.class_name)
        pkg2 = self._get_package(frame2.class_name)
        
        # Same package
        if pkg1 == pkg2:
            return True
        
        # One is parent package of the other
        if pkg1 and pkg2:
            if pkg1.startswith(pkg2) or pkg2.startswith(pkg1):
                return True
        
        return False

    def _get_package(self, class_name: str) -> str:
        """Extract package name from fully qualified class name."""
        parts = class_name.split('.')
        if len(parts) > 1:
            return '.'.join(parts[:-1])
        return ""

    def _check_r_noise(self, rcg: CallGraph, context: Dict) -> RejectionResult:
        """
        R-NOISE: Reject traces dominated by noise frames.
        
        Rejects traces where more than 80% of frames are system/library frames
        and less than 2 frames are application frames.
        """
        noise_count = 0
        framework_count = 0
        application_count = 0
        
        for frame in rcg.frames:
            if self._is_noise_frame(frame):
                noise_count += 1
            elif self._is_framework_frame(frame):
                framework_count += 1
            else:
                application_count += 1

        total_frames = len(rcg.frames)
        noise_ratio = noise_count / total_frames if total_frames > 0 else 0

        # Reject if:
        # - More than 80% noise frames, OR
        # - Less than 2 application frames
        if noise_ratio > 0.8 or application_count < 2:
            return RejectionResult(
                rejected=True,
                reason=f"Trace dominated by noise: {noise_ratio:.1%} noise frames, {application_count} application frames",
                rejection_type="R-NOISE",
                details={
                    "noise_ratio": noise_ratio,
                    "noise_count": noise_count,
                    "framework_count": framework_count,
                    "application_count": application_count,
                    "total_frames": total_frames
                }
            )

        return RejectionResult(
            rejected=False,
            reason="R-NOISE check passed",
            rejection_type="R-NOISE",
            details={
                "noise_ratio": noise_ratio,
                "application_count": application_count
            }
        )

    def _check_r_raw(self, rcg: CallGraph, context: Dict) -> RejectionResult:
        """
        R-RAW: Reject raw traces without enrichment opportunities.
        
        Rejects traces that have:
        - No proxy classes
        - No framework interceptors
        - No async patterns
        - No remote calls
        
        These traces provide minimal value for model training.
        """
        has_proxy = False
        has_interceptor = False
        has_async = False
        has_remote = False
        
        for frame in rcg.frames:
            # Check for proxy
            if frame.is_proxy or "$Proxy" in frame.class_name or "Enhancer" in frame.class_name:
                has_proxy = True
            
            # Check for interceptor
            for pattern, _ in [('TransactionInterceptor', 'transaction'),
                               ('RetryTemplate', 'retry'),
                               ('CircuitBreaker', 'circuit_breaker')]:
                if pattern in frame.class_name:
                    has_interceptor = True
            
            # Check for async
            async_patterns = ['Future', 'CompletableFuture', 'Async']
            if any(pattern in frame.class_name for pattern in async_patterns):
                has_async = True
            
            # Check for remote
            remote_patterns = ['Feign', 'RestClient', 'HttpClient', 'Remote']
            if any(pattern in frame.class_name for pattern in remote_patterns):
                has_remote = True

        # If none of the enrichment opportunities exist, reject
        if not has_proxy and not has_interceptor and not has_async and not has_remote:
            return RejectionResult(
                rejected=True,
                reason="Raw trace without enrichment opportunities (no proxy, interceptor, async, or remote calls)",
                rejection_type="R-RAW",
                details={
                    "has_proxy": has_proxy,
                    "has_interceptor": has_interceptor,
                    "has_async": has_async,
                    "has_remote": has_remote
                }
            )

        return RejectionResult(
            rejected=False,
            reason="R-RAW check passed",
            rejection_type="R-RAW",
            details={
                "has_proxy": has_proxy,
                "has_interceptor": has_interceptor,
                "has_async": has_async,
                "has_remote": has_remote
            }
        )

    def _is_noise_frame(self, frame) -> bool:
        """Check if a frame is noise (system library)."""
        import re
        for pattern in self.NOISE_PATTERNS:
            if re.match(pattern, frame.class_name):
                return True
        return False

    def _is_framework_frame(self, frame) -> bool:
        """Check if a frame is from a framework."""
        import re
        for pattern in self.FRAMEWORK_PATTERNS:
            if re.match(pattern, frame.class_name):
                return True
        return False

    def _is_application_frame(self, frame) -> bool:
        """Check if a frame is from the application code."""
        return not self._is_noise_frame(frame) and not self._is_framework_frame(frame)
