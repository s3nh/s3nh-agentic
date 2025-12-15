"""
Prompt Injection Detection System
Multi-layered detection using pattern matching, heuristics, and ML-ready hooks
"""

import re
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ThreatSeverity(Enum):
    """Severity levels for detected threats"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types of injection threats"""
    INSTRUCTION_OVERRIDE = "instruction_override"
    ROLE_MANIPULATION = "role_manipulation"
    PROMPT_EXTRACTION = "prompt_extraction"
    CODE_EXECUTION = "code_execution"
    ENCODING_BYPASS = "encoding_bypass"
    TAG_INJECTION = "tag_injection"
    JAILBREAK_ATTEMPT = "jailbreak_attempt"
    DATA_EXFILTRATION = "data_exfiltration"
    CONTEXT_MANIPULATION = "context_manipulation"


@dataclass
class ThreatIndicator:
    """Individual threat indicator"""
    threat_type: ThreatType
    severity: ThreatSeverity
    pattern_matched: str
    description: str
    position: Optional[tuple[int, int]] = None  # start, end positions
    confidence: float = 1.0


@dataclass
class InjectionCheckResult:
    """Result of injection detection check"""
    is_safe: bool
    risk_score: float  # 0.0 to 1.0
    threats: list[ThreatIndicator] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sanitized_input: Optional[str] = None
    check_id: str = ""
    
    def __post_init__(self):
        if not self.check_id:
            self.check_id = hashlib.sha256(str(id(self)).encode()).hexdigest()[:12]


class PromptInjectionDetector:
    """
    Multi-layered prompt injection detection system
    
    Implements: 
    - Pattern-based detection (regex)
    - Phrase matching
    - Heuristic analysis
    - Entropy analysis for encoded content
    - Hooks for ML-based detection
    """
    
    def __init__(self, config: dict):
        self.config = config. get("injection_detection", {})
        self.enabled = self.config.get("enabled", True)
        self.sensitivity = self.config.get("sensitivity", "high")
        
        # Load patterns from config
        self.blocked_phrases = [
            p.lower() for p in self.config.get("blocked_phrases", [])
        ]
        self.regex_patterns = self._compile_patterns(
            self.config.get("regex_patterns", [])
        )
        self.warning_patterns = [
            p.lower() for p in self.config.get("warning_patterns", [])
        ]
        
        # Sensitivity thresholds
        self.thresholds = {
            "low": {"block": 0.8, "warn": 0.6},
            "medium": {"block":  0.6, "warn":  0.4},
            "high": {"block": 0.4, "warn": 0.2},
        }
    
    def _compile_patterns(self, patterns: list[dict]) -> list[dict]:
        """Compile regex patterns from config"""
        compiled = []
        for p in patterns:
            try:
                compiled.append({
                    "regex": re.compile(p["pattern"], re.IGNORECASE | re.MULTILINE),
                    "severity":  ThreatSeverity(p. get("severity", "medium")),
                    "description": p. get("description", "Unknown pattern"),
                })
            except re.error as e:
                logger.warning(f"Invalid regex pattern: {p['pattern']} - {e}")
        return compiled
    
    def check(self, text: str, context: Optional[dict] = None) -> InjectionCheckResult:
        """
        Perform comprehensive injection detection check
        
        Args:
            text: The text to check (user input/prompt)
            context: Optional context for contextual analysis
            
        Returns: 
            InjectionCheckResult with safety assessment
        """
        if not self.enabled:
            return InjectionCheckResult(is_safe=True, risk_score=0.0)
        
        threats:  list[ThreatIndicator] = []
        warnings: list[str] = []
        
        # Normalize text for checking
        normalized = text.lower().strip()
        
        # 1. Exact phrase matching
        phrase_threats = self._check_blocked_phrases(normalized, text)
        threats.extend(phrase_threats)
        
        # 2. Regex pattern matching
        regex_threats = self._check_regex_patterns(text)
        threats.extend(regex_threats)
        
        # 3. Warning pattern check
        for pattern in self.warning_patterns:
            if pattern in normalized:
                warnings.append(f"Suspicious term detected: '{pattern}'")
        
        # 4. Heuristic checks
        heuristic_threats = self._heuristic_analysis(text)
        threats.extend(heuristic_threats)
        
        # 5. Entropy analysis (detect encoded content)
        entropy_warnings = self._entropy_analysis(text)
        warnings.extend(entropy_warnings)
        
        # 6. Structure analysis
        structure_threats = self._structure_analysis(text)
        threats.extend(structure_threats)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(threats, warnings)
        
        # Determine safety based on sensitivity threshold
        threshold = self.thresholds. get(self.sensitivity, self.thresholds["medium"])
        is_safe = risk_score < threshold["block"]
        
        # Add warnings for elevated risk
        if risk_score >= threshold["warn"] and is_safe:
            warnings.append(f"Elevated risk score: {risk_score:.2f}")
        
        return InjectionCheckResult(
            is_safe=is_safe,
            risk_score=risk_score,
            threats=threats,
            warnings=warnings,
        )
    
    def _check_blocked_phrases(self, normalized: str, original: str) -> list[ThreatIndicator]:
        """Check for blocked phrases"""
        threats = []
        for phrase in self.blocked_phrases:
            if phrase in normalized:
                # Find position in original
                pos = normalized.find(phrase)
                threats.append(ThreatIndicator(
                    threat_type=ThreatType.INSTRUCTION_OVERRIDE,
                    severity=ThreatSeverity. CRITICAL,
                    pattern_matched=phrase,
                    description=f"Blocked phrase detected: '{phrase}'",
                    position=(pos, pos + len(phrase)),
                ))
        return threats
    
    def _check_regex_patterns(self, text: str) -> list[ThreatIndicator]: 
        """Check regex patterns"""
        threats = []
        for pattern_info in self.regex_patterns:
            matches = pattern_info["regex"].finditer(text)
            for match in matches:
                threats.append(ThreatIndicator(
                    threat_type=self._infer_threat_type(pattern_info["description"]),
                    severity=pattern_info["severity"],
                    pattern_matched=match.group(),
                    description=pattern_info["description"],
                    position=(match.start(), match.end()),
                ))
        return threats
    
    def _heuristic_analysis(self, text: str) -> list[ThreatIndicator]:
        """Apply heuristic rules for detection"""
        threats = []
        
        # Check for unusual character sequences
        if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', text):
            threats.append(ThreatIndicator(
                threat_type=ThreatType.ENCODING_BYPASS,
                severity=ThreatSeverity.HIGH,
                pattern_matched="[control characters]",
                description="Suspicious control characters detected",
            ))
        
        # Check for excessive special characters
        special_ratio = len(re.findall(r'[^a-zA-Z0-9\s.,!?;:\'"()-]', text)) / max(len(text), 1)
        if special_ratio > 0.3:
            threats. append(ThreatIndicator(
                threat_type=ThreatType.ENCODING_BYPASS,
                severity=ThreatSeverity. MEDIUM,
                pattern_matched=f"special_char_ratio={special_ratio:.2f}",
                description="Unusually high ratio of special characters",
                confidence=min(special_ratio * 2, 1.0),
            ))
        
        # Check for markdown/formatting abuse
        if text.count('```') > 6 or text.count('---') > 10:
            threats.append(ThreatIndicator(
                threat_type=ThreatType. CONTEXT_MANIPULATION,
                severity=ThreatSeverity. MEDIUM,
                pattern_matched="[excessive formatting]",
                description="Potential formatting-based injection attempt",
            ))
        
        # Check for repeated instruction patterns
        instruction_words = ['must', 'always', 'never', 'ignore', 'forget', 'instead']
        instruction_count = sum(text.lower().count(word) for word in instruction_words)
        if instruction_count > 5:
            threats.append(ThreatIndicator(
                threat_type=ThreatType. INSTRUCTION_OVERRIDE,
                severity=ThreatSeverity.MEDIUM,
                pattern_matched=f"instruction_word_count={instruction_count}",
                description="High concentration of instruction-related words",
                confidence=min(instruction_count / 10, 1.0),
            ))
        
        return threats
    
    def _entropy_analysis(self, text: str) -> list[str]:
        """Analyze text entropy to detect encoded content"""
        warnings = []
        
        # Simple entropy calculation
        if len(text) > 50:
            char_freq = {}
            for char in text:
                char_freq[char] = char_freq.get(char, 0) + 1
            
            import math
            entropy = -sum(
                (freq/len(text)) * math.log2(freq/len(text)) 
                for freq in char_freq.values()
            )
            
            # High entropy might indicate encoded content
            if entropy > 5. 5:
                warnings.append(
                    f"High text entropy ({entropy:.2f}) - possible encoded content"
                )
        
        # Check for base64-like patterns
        if re.search(r'[A-Za-z0-9+/]{50,}={0,2}', text):
            warnings.append("Potential base64 encoded content detected")
        
        return warnings
    
    def _structure_analysis(self, text: str) -> list[ThreatIndicator]: 
        """Analyze prompt structure for manipulation"""
        threats = []
        
        # Check for fake conversation markers
        conversation_markers = [
            r'(?i)^(user|assistant|system|human|ai):\s*',
            r'(? i)\n(user|assistant|system|human|ai):\s*',
            r'(?i)<\|?(user|assistant|system|im_start|im_end)\|?>',
        ]
        
        for pattern in conversation_markers:
            if re.search(pattern, text):
                threats.append(ThreatIndicator(
                    threat_type=ThreatType. CONTEXT_MANIPULATION,
                    severity=ThreatSeverity.HIGH,
                    pattern_matched=pattern,
                    description="Fake conversation marker detected",
                ))
                break
        
        # Check for system prompt simulation
        if re.search(r'(? i)^system\s*prompt|system\s*message|<system>', text):
            threats.append(ThreatIndicator(
                threat_type=ThreatType.CONTEXT_MANIPULATION,
                severity=ThreatSeverity. CRITICAL,
                pattern_matched="system prompt simulation",
                description="Attempt to simulate system prompt",
            ))
        
        return threats
    
    def _infer_threat_type(self, description: str) -> ThreatType:
        """Infer threat type from description"""
        description_lower = description.lower()
        if "override" in description_lower or "instruction" in description_lower:
            return ThreatType.INSTRUCTION_OVERRIDE
        elif "role" in description_lower or "manipulation" in description_lower:
            return ThreatType.ROLE_MANIPULATION
        elif "prompt" in description_lower or "extract" in description_lower:
            return ThreatType.PROMPT_EXTRACTION
        elif "code" in description_lower or "exec" in description_lower:
            return ThreatType.CODE_EXECUTION
        elif "encod" in description_lower or "bypass" in description_lower: 
            return ThreatType. ENCODING_BYPASS
        elif "tag" in description_lower: 
            return ThreatType. TAG_INJECTION
        else:
            return ThreatType.JAILBREAK_ATTEMPT
    
    def _calculate_risk_score(
        self, 
        threats: list[ThreatIndicator], 
        warnings: list[str]
    ) -> float:
        """Calculate overall risk score from threats and warnings"""
        if not threats and not warnings:
            return 0.0
        
        # Severity weights
        severity_weights = {
            ThreatSeverity.LOW: 0.2,
            ThreatSeverity.MEDIUM: 0.4,
            ThreatSeverity.HIGH: 0.7,
            ThreatSeverity.CRITICAL: 1.0,
        }
        
        # Calculate threat score
        threat_score = sum(
            severity_weights[t.severity] * t.confidence 
            for t in threats
        )
        
        # Add warning contribution
        warning_score = len(warnings) * 0.1
        
        # Normalize to 0-1 range with diminishing returns
        import math
        total = threat_score + warning_score
        risk_score = 1 - math.exp(-total)
        
        return min(risk_score, 1.0)
    
    def add_custom_pattern(
        self, 
        pattern:  str, 
        severity: ThreatSeverity, 
        description: str
    ):
        """Add a custom detection pattern at runtime"""
        try:
            compiled = re.compile(pattern, re. IGNORECASE | re.MULTILINE)
            self.regex_patterns.append({
                "regex": compiled,
                "severity":  severity,
                "description": description,
            })
            logger.info(f"Added custom pattern: {description}")
        except re.error as e:
            logger.error(f"Invalid custom pattern: {pattern} - {e}")
            raise ValueError(f"Invalid regex pattern: {e}")
