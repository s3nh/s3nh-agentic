"""
Document classes and utilities
Universal document representation for processing
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Union, Optional
import mimetypes
import logging

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Supported document MIME types"""
    PDF = "application/pdf"
    IMAGE_PNG = "image/png"
    IMAGE_JPEG = "image/jpeg"
    IMAGE_WEBP = "image/webp"
    IMAGE_GIF = "image/gif"
    TEXT = "text/plain"
    MARKDOWN = "text/markdown"
    HTML = "text/html"
    CSV = "text/csv"
    JSON = "application/json"
    AUDIO_MP3 = "audio/mpeg"
    AUDIO_WAV = "audio/wav"
    VIDEO_MP4 = "video/mp4"
    VIDEO_WEBM = "video/webm"


@dataclass
class Document: 
    """
    Universal document representation
    
    Supports:
    - File-based documents (local files)
    - URL-based documents (remote resources)
    - In-memory documents (bytes/strings)
    """
    content: Union[bytes, str, Path]
    doc_type: DocumentType
    filename: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    
    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> "Document":
        """
        Load document from file path
        
        Args:
            filepath:  Path to the document file
            
        Returns: 
            Document instance
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type is not supported
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Guess MIME type from file extension
        mime_type, _ = mimetypes.guess_type(str(path))
        
        # Map mime type to DocumentType
        type_mapping = {v. value: v for v in DocumentType}
        doc_type = type_mapping.get(mime_type, DocumentType. TEXT)
        
        # Read file content
        try:
            content = path.read_bytes()
        except Exception as e:
            logger.error(f"Failed to read file {filepath}: {e}")
            raise IOError(f"Cannot read file: {e}")
        
        return cls(
            content=content,
            doc_type=doc_type,
            filename=path.name,
            metadata={
                "source": str(path. absolute()),
                "size_bytes": len(content),
                "mime_type": mime_type,
            }
        )
    
    @classmethod
    def from_url(cls, url: str, doc_type: DocumentType) -> "Document":
        """
        Create document reference from URL
        
        Args: 
            url: URL to the document
            doc_type: Type of document
            
        Returns: 
            Document instance with URL reference
        """
        return cls(
            content=url,
            doc_type=doc_type,
            metadata={
                "source": url,
                "is_url": True,
            }
        )
    
    @classmethod
    def from_bytes(
        cls, 
        content: bytes, 
        doc_type:  DocumentType,
        filename: Optional[str] = None
    ) -> "Document":
        """
        Create document from bytes
        
        Args: 
            content: Document content as bytes
            doc_type: Type of document
            filename: Optional filename
            
        Returns:
            Document instance
        """
        return cls(
            content=content,
            doc_type=doc_type,
            filename=filename,
            metadata={
                "source":  "bytes",
                "size_bytes": len(content),
            }
        )
    
    @classmethod
    def from_text(
        cls, 
        text: str, 
        doc_type: DocumentType = DocumentType.TEXT,
        filename: Optional[str] = None
    ) -> "Document":
        """
        Create document from text string
        
        Args:
            text: Document content as text
            doc_type: Type of document (default: TEXT)
            filename:  Optional filename
            
        Returns: 
            Document instance
        """
        return cls(
            content=text,
            doc_type=doc_type,
            filename=filename,
            metadata={
                "source": "text",
                "size_chars": len(text),
            }
        )
    
    def get_size(self) -> int:
        """Get document size in bytes"""
        if isinstance(self.content, bytes):
            return len(self. content)
        elif isinstance(self.content, str):
            return len(self.content.encode('utf-8'))
        elif isinstance(self.content, Path):
            return self.content.stat().st_size if self.content.exists() else 0
        else:
            return 0
    
    def is_url(self) -> bool:
        """Check if document is URL-based"""
        return self.metadata.get("is_url", False)
    
    def is_text_based(self) -> bool:
        """Check if document is text-based"""
        text_types = {
            DocumentType.TEXT,
            DocumentType.MARKDOWN,
            DocumentType.HTML,
            DocumentType.CSV,
            DocumentType.JSON,
        }
        return self.doc_type in text_types
    
    def is_image(self) -> bool:
        """Check if document is an image"""
        image_types = {
            DocumentType.IMAGE_PNG,
            DocumentType.IMAGE_JPEG,
            DocumentType.IMAGE_WEBP,
            DocumentType.IMAGE_GIF,
        }
        return self.doc_type in image_types
    
    def is_audio(self) -> bool:
        """Check if document is audio"""
        audio_types = {
            DocumentType.AUDIO_MP3,
            DocumentType.AUDIO_WAV,
        }
        return self.doc_type in audio_types
    
    def is_video(self) -> bool:
        """Check if document is video"""
        video_types = {
            DocumentType. VIDEO_MP4,
            DocumentType.VIDEO_WEBM,
        }
        return self. doc_type in video_types
    
    def __repr__(self) -> str:
        """String representation of document"""
        size = self.get_size()
        size_str = f"{size: ,} bytes" if size > 0 else "unknown size"
        filename_str = f"'{self.filename}'" if self.filename else "unnamed"
        return f"Document({filename_str}, {self.doc_type.value}, {size_str})"
