"""
Output Filtering Module
Filters and sanitizes LLM outputs before returning to user
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FilterAction(Enum):
    """Action to take when pattern matches"""
    REDACT = "redact"
    FLAG = "flag"
    BLOCK = "block"


@dataclass
class PIIMatch:
    """Detected PII match"""
    pii_type: str
    value: str
    position: tuple[int, int]
    action: FilterAction
    replacement: Optional[str]


@dataclass
class FilterResult:
    """Result of output filtering"""
    is_safe: bool
    filtered_output: str
    original_output:  str
    pii_detected: list[PIIMatch] = field(default_factory=list)
    blocked_patterns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    was_modified: bool = False


class OutputFilter:
    """
    Filters LLM outputs for sensitive information and policy violations
    
    Features: 
    - PII detection and redaction
    - System prompt leak detection
    - Credential leak prevention
    - Content policy enforcement
    """
    
    def __init__(self, config: dict):
        self.config = config.get("output_filtering", {})
        self.enabled = self.config.get("enabled", True)
        
        # Compile PII patterns
        self.pii_patterns = self._compile_pii_patterns(
            self.config.get("pii_patterns", {})
        )
        
        # Compile blocked output patterns
        self.blocked_patterns = self._compile_blocked_patterns(
            self.config.get("blocked_outputs", [])
        )
    
    def _compile_pii_patterns(self, patterns: dict) -> list[dict]:
        """Compile PII detection patterns"""
        compiled = []
        for name, config in patterns.items():
            try:
                compiled.append({
                    "name": name,
                    "regex": re.compile(config["pattern"]),
                    "action": FilterAction(config. get("action", "flag")),
                    "replacement":  config.get("replacement"),
                })
            except (re.error, KeyError) as e:
                logger.warning(f"Invalid PII pattern '{name}': {e}")
        return compiled
    
    def _compile_blocked_patterns(self, patterns: list[dict]) -> list[dict]:
        """Compile blocked output patterns"""
        compiled = []
        for p in patterns:
            try: 
                compiled.append({
                    "regex": re.compile(p["pattern"], re.IGNORECASE),
                    "description": p.get("description", "Blocked pattern"),
                })
            except (re.error, KeyError) as e:
                logger.warning(f"Invalid blocked pattern:  {e}")
        return compiled
    
    def filter(
        self, 
        output: str, 
        redact_pii: bool = True,
        check_leaks: bool = True
    ) -> FilterResult:
        """
        Filter LLM output
        
        Args:
            output: The LLM output to filter
            redact_pii: Whether to redact detected PII
            check_leaks: Whether to check for prompt/credential leaks
        """
        if not self.enabled:
            return FilterResult(
                is_safe=True,
                filtered_output=output,
                original_output=output,
            )
        
        filtered = output
        pii_detected = []
        blocked = []
        warnings = []
        was_modified = False
        
        # Check for blocked patterns first (these block the response)
        if check_leaks: 
            for pattern_info in self.blocked_patterns:
                if pattern_info["regex"].search(filtered):
                    blocked.append(pattern_info["description"])
        
        if blocked:
            return FilterResult(
                is_safe=False,
                filtered_output="[OUTPUT BLOCKED:  Security policy violation detected]",
                original_output=output,
                blocked_patterns=blocked,
                was_modified=True,
            )
        
        # Detect and handle PII
        for pattern_info in self.pii_patterns:
            matches = list(pattern_info["regex"].finditer(filtered))
            
            for match in reversed(matches):  # Reverse to preserve positions
                pii_match = PIIMatch(
                    pii_type=pattern_info["name"],
                    value=match.group(),
                    position=(match.start(), match.end()),
                    action=pattern_info["action"],
                    replacement=pattern_info["replacement"],
                )
                pii_detected.append(pii_match)
                
                if redact_pii and pattern_info["action"] == FilterAction.REDACT:
                    replacement = pattern_info["replacement"] or f"[{pattern_info['name']. upper()} REDACTED]"
                    filtered = filtered[:match.start()] + replacement + filtered[match.end():]
                    was_modified = True
                elif pattern_info["action"] == FilterAction.FLAG:
                    warnings.append(
                        f"PII detected ({pattern_info['name']}): "
                        f"position {match.start()}-{match.end()}"
                    )
        
        # Additional leak detection
        leak_warnings = self._detect_potential_leaks(filtered)
        warnings.extend(leak_warnings)
        
        return FilterResult(
            is_safe=True,
            filtered_output=filtered,
            original_output=output,
            pii_detected=pii_detected,
            warnings=warnings,
            was_modified=was_modified,
        )
    
    def _detect_potential_leaks(self, text: str) -> list[str]:
        """Detect potential information leaks"""
        warnings = []
        
        # Check for common credential patterns
        credential_patterns = [
            (r'(?i)api[_-]? key\s*[=: ]\s*[\'"]?[\w-]{20,}', "Potential API key"),
            (r'(?i)password\s*[=:]\s*[\'"]?[^\s\'"]{8,}', "Potential password"),
            (r'(?i)secret\s*[=:]\s*[\'"]?[\w-]{16,}', "Potential secret"),
            (r'(?i)token\s*[=:]\s*[\'"]?[\w.-]{20,}', "Potential token"),
            (r'(?i)bearer\s+[\w.-]{20,}', "Potential bearer token"),
        ]
        
        for pattern, description in credential_patterns:
            if re.search(pattern, text):
                warnings.append(f"{description} detected in output")
        
        # Check for system prompt indicators
        system_indicators = [
            r'(?i)my (system )?instructions (are|say|tell)',
            r'(?i)i was (programmed|instructed|told) to',
            r'(? i)my (initial|original|base) prompt',
        ]
        
        for pattern in system_indicators:
            if re.search(pattern, text):
                warnings.append("Potential system prompt disclosure detected")
                break
        
        return warnings
    
    def add_pii_pattern(
        self, 
        name: str, 
        pattern:  str, 
        action: FilterAction = FilterAction.FLAG,
        replacement: Optional[str] = None
    ):
        """Add a custom PII pattern at runtime"""
        try:
            self.pii_patterns.append({
                "name": name,
                "regex":  re.compile(pattern),
                "action": action,
                "replacement": replacement,
            })
            logger.info(f"Added PII pattern:  {name}")
        except re.error as e:
            logger.error(f"Invalid PII pattern:  {pattern} - {e}")
            raise ValueError(f"Invalid regex pattern: {e}")
