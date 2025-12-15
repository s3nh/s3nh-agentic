"""
Unified Guardrails Engine
Orchestrates all security checks and enforces policies
"""

import secrets
import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import asyncio

from .injection_detector import PromptInjectionDetector, InjectionCheckResult
from .input_sanitizer import InputSanitizer, SanitizationResult
from .output_filter import OutputFilter, FilterResult

logger = logging.getLogger(__name__)


@dataclass
class GuardrailResult:
    """Comprehensive guardrail check result"""
    passed: bool
    request_id: str
    timestamp:  datetime
    
    # Component results
    injection_check:  Optional[InjectionCheckResult] = None
    input_sanitization: Optional[SanitizationResult] = None
    output_filter: Optional[FilterResult] = None
    
    # Rate limiting
    rate_limited: bool = False
    rate_limit_message: Optional[str] = None
    
    # Aggregated info
    all_warnings: list[str] = field(default_factory=list)
    all_errors: list[str] = field(default_factory=list)
    
    # Session info
    session_id: Optional[str] = None
    boundary_token: Optional[str] = None
    
    # Audit trail
    processing_time_ms: float = 0.0


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, config: dict):
        rl_config = config.get("rate_limiting", {})
        self.enabled = rl_config.get("enabled", True)
        self.rpm = rl_config.get("requests_per_minute", 60)
        self.rph = rl_config.get("requests_per_hour", 500)
        self.tpm = rl_config.get("tokens_per_minute", 100000)
        self.concurrent = rl_config.get("concurrent_requests", 10)
        
        # Tracking
        self.requests: dict[str, list[float]] = defaultdict(list)
        self.tokens: dict[str, list[tuple[float, int]]] = defaultdict(list)
        self.active:  dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()
    
    async def check(
        self, 
        user_id: str, 
        estimated_tokens: int = 0
    ) -> tuple[bool, Optional[str]]:
        """Check if request is allowed under rate limits"""
        if not self.enabled:
            return True, None
        
        async with self._lock:
            now = time.time()
            minute_ago = now - 60
            hour_ago = now - 3600
            
            # Clean old entries
            self. requests[user_id] = [
                t for t in self. requests[user_id] if t > minute_ago
            ]
            self.tokens[user_id] = [
                (t, n) for t, n in self.tokens[user_id] if t > minute_ago
            ]
            
            # Check concurrent
            if self.active[user_id] >= self. concurrent:
                return False, f"Concurrent request limit ({self.concurrent}) exceeded"
            
            # Check RPM
            recent_requests = len(self.requests[user_id])
            if recent_requests >= self.rpm:
                return False, f"Rate limit ({self.rpm}/min) exceeded"
            
            # Check TPM
            recent_tokens = sum(n for _, n in self.tokens[user_id])
            if recent_tokens + estimated_tokens > self.tpm:
                return False, f"Token limit ({self.tpm}/min) exceeded"
            
            # Record request
            self.requests[user_id].append(now)
            if estimated_tokens > 0:
                self.tokens[user_id]. append((now, estimated_tokens))
            self.active[user_id] += 1
            
            return True, None
    
    async def release(self, user_id: str):
        """Release a concurrent request slot"""
        async with self._lock:
            self.active[user_id] = max(0, self.active[user_id] - 1)


class AuditLogger:
    """Security audit logger"""
    
    def __init__(self, config: dict):
        audit_config = config.get("audit_logging", {})
        self.enabled = audit_config.get("enabled", True)
        self.log_inputs = audit_config.get("log_inputs", True)
        self.log_outputs = audit_config.get("log_outputs", True)
        self.log_blocked = audit_config.get("log_blocked_requests", True)
        self.mask_sensitive = audit_config.get("sensitive_field_masking", True)
        
        self.logger = logging.getLogger("guardrails. audit")
    
    def log_request(
        self, 
        request_id: str,
        user_id: Optional[str],
        action: str,
        input_preview: Optional[str] = None,
        result: Optional[GuardrailResult] = None
    ):
        """Log a request for audit purposes"""
        if not self.enabled:
            return
        
        log_data = {
            "request_id": request_id,
            "user_id": self._mask(user_id) if self.mask_sensitive else user_id,
            "action": action,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        if self. log_inputs and input_preview:
            log_data["input_preview"] = input_preview[: 200] + "..." if len(input_preview) > 200 else input_preview
        
        if result:
            log_data["passed"] = result.passed
            log_data["warnings_count"] = len(result.all_warnings)
            log_data["errors_count"] = len(result.all_errors)
            log_data["processing_time_ms"] = result.processing_time_ms
            
            if result.injection_check:
                log_data["risk_score"] = result.injection_check.risk_score
            
            if not result.passed and self.log_blocked:
                log_data["block_reasons"] = result.all_errors
        
        if result and not result.passed:
            self.logger.warning(f"BLOCKED: {log_data}")
        else:
            self.logger. info(f"REQUEST: {log_data}")
    
    def _mask(self, value: Optional[str]) -> Optional[str]:
        """Mask
