"""
Input Sanitization Module
Cleans and validates input before processing
"""

import re
import html
from dataclasses import dataclass, field
from typing import Optional, Any
from pathlib import Path
import mimetypes
import logging

logger = logging.getLogger(__name__)


@dataclass
class SanitizationResult:
    """Result of input sanitization"""
    is_valid: bool
    sanitized_value: Any
    original_value: Any
    modifications: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class InputSanitizer: 
    """
    Sanitizes and validates all inputs before processing
    
    Features:
    - Text sanitization (encoding, length limits)
    - File validation (type, size)
    - Parameter validation
    - XSS prevention
    """
    
    def __init__(self, config: dict):
        self.config = config.get("input_validation", {})
        self.max_prompt_length = self.config.get("max_prompt_length", 10000)
        self.max_document_size_mb = self.config.get("max_document_size_mb", 50)
        self.max_documents = self.config.get("max_documents_per_request", 20)
        self.allowed_mime_types = set(self.config.get("allowed_mime_types", []))
    
    def sanitize_text(
        self, 
        text:  str, 
        max_length: Optional[int] = None,
        strip_html: bool = True,
        normalize_whitespace: bool = True
    ) -> SanitizationResult:
        """
        Sanitize text input
        
        Args:
            text: Input text to sanitize
            max_length: Maximum allowed length (uses config default if None)
            strip_html: Whether to escape HTML entities
            normalize_whitespace: Whether to normalize whitespace
        """
        modifications = []
        errors = []
        warnings = []
        
        if not isinstance(text, str):
            return SanitizationResult(
                is_valid=False,
                sanitized_value=None,
                original_value=text,
                errors=["Input must be a string"],
            )
        
        sanitized = text
        max_len = max_length or self.max_prompt_length
        
        # Check length
        if len(sanitized) > max_len:
            sanitized = sanitized[:max_len]
            modifications.append(f"Truncated from {len(text)} to {max_len} characters")
            warnings.append("Input was truncated to maximum length")
        
        # Normalize unicode
        import unicodedata
        normalized = unicodedata.normalize('NFKC', sanitized)
        if normalized != sanitized:
            sanitized = normalized
            modifications.append("Normalized unicode characters")
        
        # Remove null bytes and control characters (except newlines/tabs)
        clean = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', sanitized)
        if clean != sanitized:
            sanitized = clean
            modifications.append("Removed control characters")
        
        # Escape HTML if requested
        if strip_html: 
            escaped = html.escape(sanitized)
            if escaped != sanitized:
                sanitized = escaped
                modifications.append("Escaped HTML entities")
        
        # Normalize whitespace if requested
        if normalize_whitespace:
            # Normalize line endings
            sanitized = sanitized.replace('\r\n', '\n').replace('\r', '\n')
            # Collapse multiple spaces (but preserve newlines)
            sanitized = re.sub(r'[^\S\n]+', ' ', sanitized)
            # Remove trailing whitespace from lines
            sanitized = '\n'.join(line.rstrip() for line in sanitized.split('\n'))
            if sanitized != text:
                modifications.append("Normalized whitespace")
        
        return SanitizationResult(
            is_valid=True,
            sanitized_value=sanitized. strip(),
            original_value=text,
            modifications=modifications,
            warnings=warnings,
        )
    
    def validate_document(
        self, 
        content: bytes, 
        filename: Optional[str] = None,
        mime_type: Optional[str] = None
    ) -> SanitizationResult: 
        """
        Validate document content
        
        Args:
            content: Document bytes
            filename: Optional filename for type detection
            mime_type: Optional explicit MIME type
        """
        errors = []
        warnings = []
        
        # Check size
        size_mb = len(content) / (1024 * 1024)
        if size_mb > self.max_document_size_mb:
            errors.append(
                f"Document size ({size_mb:.2f}MB) exceeds limit "
                f"({self.max_document_size_mb}MB)"
            )
        
        # Determine MIME type
        if mime_type is None and filename:
            mime_type, _ = mimetypes.guess_type(filename)
        
        if mime_type is None: 
            # Try to detect from content
            mime_type = self._detect_mime_type(content)
        
        # Validate MIME type
        if self.allowed_mime_types and mime_type not in self.allowed_mime_types:
            errors. append(f"MIME type '{mime_type}' is not allowed")
        
        # Additional validation for specific types
        type_validation = self._validate_content_type(content, mime_type)
        if type_validation: 
            warnings.extend(type_validation)
        
        return SanitizationResult(
            is_valid=len(errors) == 0,
            sanitized_value=content if not errors else None,
            original_value=content,
            errors=errors,
            warnings=warnings,
        )
    
    def validate_documents_batch(
        self, 
        documents: list[tuple[bytes, Optional[str], Optional[str]]]
    ) -> SanitizationResult:
        """
        Validate a batch of documents
        
        Args:
            documents: List of (content, filename, mime_type) tuples
        """
        errors = []
        warnings = []
        valid_docs = []
        
        if len(documents) > self.max_documents:
            errors.append(
                f"Number of documents ({len(documents)}) exceeds limit "
                f"({self.max_documents})"
            )
            return SanitizationResult(
                is_valid=False,
                sanitized_value=None,
                original_value=documents,
                errors=errors,
            )
        
        for i, (content, filename, mime_type) in enumerate(documents):
            result = self.validate_document(content, filename, mime_type)
            if result.is_valid:
                valid_docs.append((content, filename, mime_type))
            else:
                errors. extend([f"Document {i + 1}:  {e}" for e in result.errors])
            warnings.extend([f"Document {i + 1}: {w}" for w in result.warnings])
        
        return SanitizationResult(
            is_valid=len(errors) == 0,
            sanitized_value=valid_docs if not errors else None,
            original_value=documents,
            errors=errors,
            warnings=warnings,
        )
    
    def sanitize_parameters(self, params: dict) -> SanitizationResult:
        """Sanitize and validate processing parameters"""
        errors = []
        warnings = []
        sanitized = {}
        
        # Validate temperature
        if "temperature" in params:
            temp = params["temperature"]
            if not isinstance(temp, (int, float)) or not 0 <= temp <= 2: 
                errors.append("Temperature must be a number between 0 and 2")
            else:
                sanitized["temperature"] = float(temp)
        
        # Validate max_output_tokens
        if "max_output_tokens" in params: 
            tokens = params["max_output_tokens"]
            if not isinstance(tokens, int) or tokens < 1 or tokens > 65536:
                errors.append("max_output_tokens must be between 1 and 65536")
            else:
                sanitized["max_output_tokens"] = tokens
        
        # Validate output_format
        valid_formats = {"json", "markdown", "text", "csv", "html"}
        if "output_format" in params: 
            fmt = params["output_format"]. lower()
            if fmt not in valid_formats:
                warnings.append(f"Unknown output format '{fmt}', using 'text'")
                sanitized["output_format"] = "text"
            else:
                sanitized["output_format"] = fmt
        
        # Pass through other parameters
        for key, value in params.items():
            if key not in sanitized:
                sanitized[key] = value
        
        return SanitizationResult(
            is_valid=len(errors) == 0,
            sanitized_value=sanitized,
            original_value=params,
            errors=errors,
            warnings=warnings,
        )
    
    def _detect_mime_type(self, content: bytes) -> str:
        """Detect MIME type from content magic bytes"""
        # Common magic bytes
        magic_bytes = {
            b'%PDF':  'application/pdf',
            b'\x89PNG': 'image/png',
            b'\xff\xd8\xff': 'image/jpeg',
            b'GIF87a': 'image/gif',
            b'GIF89a': 'image/gif',
            b'RIFF':  'audio/wav',  # Also could be webp/avi
            b'\x00\x00\x00':  'video/mp4',  # Simplified
            b'ID3': 'audio/mpeg',
            b'\xff\xfb': 'audio/mpeg',
        }
        
        for magic, mime in magic_bytes.items():
            if content.startswith(magic):
                return mime
        
        # Try to decode as text
        try:
            content[: 1000]. decode('utf-8')
            return 'text/plain'
        except UnicodeDecodeError:
            pass
        
        return 'application/octet-stream'
    
    def _validate_content_type(
        self, 
        content:  bytes, 
        mime_type: Optional[str]
    ) -> list[str]:
        """Additional validation for specific content types"""
        warnings = []
        
        if mime_type == 'application/pdf':
            if not content.startswith(b'%PDF'):
                warnings.append("File claims to be PDF but lacks PDF header")
        
        if mime_type == 'application/json':
            try:
                import json
                json.loads(content.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                warnings.append("File claims to be JSON but is not valid JSON")
        
        return warnings
