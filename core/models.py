"""
Model configuration and management
Defines available Gemini models and their configurations
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class GeminiModel(Enum):
    """
    Available Gemini model placeholders
    
    Update this enum as new models are released by Google
    """
    # Flash models - Fast and cost-effective
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"
    
    # Pro models - Balanced performance
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_PRO = "gemini-pro"
    
    # Experimental / Preview models (placeholders for future releases)
    GEMINI_3_0_PRO = "gemini-3.0-pro"  # Placeholder for future
    GEMINI_ULTRA = "gemini-ultra"      # Placeholder for future
    
    # Vision-specific (legacy)
    GEMINI_PRO_VISION = "gemini-pro-vision"
    
    @classmethod
    def get_flash_models(cls) -> list['GeminiModel']:
        """Get all Flash tier models"""
        return [
            cls.GEMINI_2_5_FLASH,
            cls.GEMINI_2_0_FLASH,
            cls. GEMINI_1_5_FLASH,
        ]
    
    @classmethod
    def get_pro_models(cls) -> list['GeminiModel']: 
        """Get all Pro tier models"""
        return [
            cls.GEMINI_2_5_PRO,
            cls. GEMINI_1_5_PRO,
            cls. GEMINI_PRO,
            cls.GEMINI_3_0_PRO,
        ]
    
    @classmethod
    def get_latest_flash(cls) -> 'GeminiModel':
        """Get the latest Flash model"""
        return cls.GEMINI_2_5_FLASH
    
    @classmethod
    def get_latest_pro(cls) -> 'GeminiModel':
        """Get the latest Pro model"""
        return cls.GEMINI_2_5_PRO
    
    @classmethod
    def from_string(cls, model_str: str) -> Optional['GeminiModel']:
        """Get model enum from string"""
        for model in cls:
            if model. value == model_str:
                return model
        return None


@dataclass
class ModelConfig:
    """
    Configuration for Gemini model parameters
    
    Attributes: 
        model: The Gemini model to use
        temperature: Controls randomness (0.0 to 2.0)
        max_output_tokens: Maximum tokens in response
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling parameter
        stop_sequences: Optional stop sequences
    """
    model: GeminiModel = GeminiModel.GEMINI_2_5_FLASH
    temperature: float = 0.7
    max_output_tokens:  int = 8192
    top_p: float = 0.95
    top_k: int = 40
    stop_sequences: Optional[list[str]] = None
    
    def __post_init__(self):
        """Validate configuration parameters"""
        # Validate temperature
        if not 0.0 <= self. temperature <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        
        # Validate max_output_tokens
        if self. max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1")
        
        # Validate top_p
        if not 0.0 <= self. top_p <= 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")
        
        # Validate top_k
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        
        logger.debug(f"ModelConfig initialized:  {self. model.value}, temp={self.temperature}")
    
    def to_generation_config(self) -> dict:
        """
        Convert to Gemini API generation config format
        
        Returns:
            Dictionary compatible with google.genai GenerateContentConfig
        """
        config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "top_p": self.top_p,
            "top_k":  self.top_k,
        }
        
        if self. stop_sequences:
            config["stop_sequences"] = self.stop_sequences
        
        return config
    
    @classmethod
    def for_task(cls, task: str) -> 'ModelConfig':
        """
        Get recommended configuration for specific task
        
        Args: 
            task: Task name (summarize, extract, analyze, etc.)
            
        Returns:
            Optimized ModelConfig for the task
        """
        task_configs = {
            "summarize":  cls(
                model=GeminiModel.GEMINI_2_5_FLASH,
                temperature=0.5,
                max_output_tokens=4096,
            ),
            "extract": cls(
                model=GeminiModel.GEMINI_2_5_PRO,
                temperature=0.2,
                max_output_tokens=8192,
            ),
            "analyze": cls(
                model=GeminiModel. GEMINI_2_5_PRO,
                temperature=0.4,
                max_output_tokens=8192,
            ),
            "qa": cls(
                model=GeminiModel.GEMINI_2_5_FLASH,
                temperature=0.3,
                max_output_tokens=2048,
            ),
            "classify": cls(
                model=GeminiModel.GEMINI_2_5_FLASH,
                temperature=0.1,
                max_output_tokens=1024,
            ),
            "translate": cls(
                model=GeminiModel.GEMINI_2_5_PRO,
                temperature=0.3,
                max_output_tokens=8192,
            ),
            "compare": cls(
                model=GeminiModel.GEMINI_2_5_PRO,
                temperature=0.3,
                max_output_tokens=8192,
            ),
        }
        
        return task_configs.get(task, cls())
    
    @classmethod
    def conservative(cls) -> 'ModelConfig':
        """Get conservative (low temperature) configuration"""
        return cls(temperature=0.2, max_output_tokens=4096)
    
    @classmethod
    def creative(cls) -> 'ModelConfig':
        """Get creative (high temperature) configuration"""
        return cls(temperature=1.2, max_output_tokens=8192)
    
    @classmethod
    def balanced(cls) -> 'ModelConfig':
        """Get balanced configuration (default)"""
        return cls()
    
    def with_model(self, model: GeminiModel) -> 'ModelConfig':
        """Create a copy with a different model"""
        return ModelConfig(
            model=model,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            stop_sequences=self.stop_sequences,
        )
    
    def with_temperature(self, temperature: float) -> 'ModelConfig':
        """Create a copy with a different temperature"""
        return ModelConfig(
            model=self.model,
            temperature=temperature,
            max_output_tokens=self.max_output_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            stop_sequences=self.stop_sequences,
        )
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"ModelConfig(model={self.model.value}, "
            f"temp={self.temperature}, "
            f"max_tokens={self.max_output_tokens})"
        )


@dataclass
class ModelCapabilities:
    """
    Model capabilities and limitations
    
    Used for validation and selection
    """
    model:  GeminiModel
    context_window: int
    max_output_tokens: int
    supports_vision: bool
    supports_audio:  bool
    supports_video: bool
    cost_tier: str  # "low", "medium", "high"
    
    @classmethod
    def get_capabilities(cls, model: GeminiModel) -> 'ModelCapabilities':
        """Get capabilities for a specific model"""
        capabilities_map = {
            GeminiModel.GEMINI_2_5_FLASH: cls(
                model=model,
                context_window=1_000_000,
                max_output_tokens=8192,
                supports_vision=True,
                supports_audio=True,
                supports_video=True,
                cost_tier="low",
            ),
            GeminiModel.GEMINI_2_5_PRO: cls(
                model=model,
                context_window=2_000_000,
                max_output_tokens=16384,
                supports_vision=True,
                supports_audio=True,
                supports_video=True,
                cost_tier="medium",
            ),
            GeminiModel.GEMINI_1_5_PRO: cls(
                model=model,
                context_window=2_000_000,
                max_output_tokens=8192,
                supports_vision=True,
                supports_audio=True,
                supports_video=True,
                cost_tier="medium",
            ),
        }
        
        # Default capabilities for unknown models
        return capabilities_map. get(
            model,
            cls(
                model=model,
                context_window=1_000_000,
                max_output_tokens=8192,
                supports_vision=True,
                supports_audio=True,
                supports_video=True,
                cost_tier="medium",
            )
        )
