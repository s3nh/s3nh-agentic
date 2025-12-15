"""
Document processing agents
Implements various agent types for document analysis and processing
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union
from pathlib import Path
import asyncio
import logging
import secrets

from .documents import Document
from .models import GeminiModel, ModelConfig

logger = logging.getLogger(__name__)


class AgentAction(Enum):
    """Available agent actions for document processing"""
    SUMMARIZE = "summarize"
    EXTRACT_DATA = "extract_data"
    ANALYZE = "analyze"
    QUESTION_ANSWER = "question_answer"
    TRANSLATE = "translate"
    CLASSIFY = "classify"
    COMPARE = "compare"
    CONVERT_FORMAT = "convert_format"
    EXTRACT_TABLES = "extract_tables"
    EXTRACT_IMAGES = "extract_images"
    OCR = "ocr"
    CUSTOM = "custom"


@dataclass
class ProcessingTask:
    """
    Defines a document processing task
    
    Attributes:
        action: The action to perform
        documents: List of documents to process
        prompt: Optional custom prompt
        output_format:  Desired output format (json, markdown, text, etc.)
        additional_context: Additional context or instructions
        session_id: Optional session identifier for tracking
        boundary_token: Security boundary token for prompt injection prevention
    """
    action: AgentAction
    documents: list[Document]
    prompt: Optional[str] = None
    output_format: Optional[str] = "markdown"
    additional_context: Optional[str] = None
    session_id: Optional[str] = None
    boundary_token: Optional[str] = None
    
    def __post_init__(self):
        """Initialize session and security tokens if not provided"""
        if self.session_id is None:
            self.session_id = secrets.token_hex(8)
        
        if self.boundary_token is None:
            self.boundary_token = secrets.token_hex(16)
        
        # Validate documents list
        if not self.documents:
            raise ValueError("At least one document is required")


@dataclass
class ProcessingResult:
    """
    Result from document processing
    
    Attributes:
        success: Whether processing was successful
        content: The processed output content
        model_used: Model identifier used for processing
        tokens_used: Number of tokens consumed (if available)
        error: Error message if processing failed
        metadata: Additional metadata about the processing
    """
    success: bool
    content: Any
    model_used: str
    tokens_used: Optional[int] = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert result to dictionary"""
        return {
            "success": self.success,
            "content": self.content,
            "model_used": self.model_used,
            "tokens_used": self.tokens_used,
            "error": self.error,
            "metadata": self. metadata,
        }


class BaseDocumentAgent(ABC):
    """
    Abstract base class for document processing agents
    
    All agent implementations must inherit from this class
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model_config: Optional[ModelConfig] = None
    ):
        """
        Initialize the agent
        
        Args:
            api_key:  Google API key (if None, uses environment variable)
            model_config:  Model configuration to use
        """
        try:
            from google import genai
            self.client = genai.Client(api_key=api_key) if api_key else genai. Client()
        except ImportError: 
            raise ImportError(
                "google-genai package is required.  "
                "Install with: pip install google-genai"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise
        
        self.config = model_config or ModelConfig()
        logger.info(f"Agent initialized with model: {self.config.model. value}")
    
    @abstractmethod
    async def process(self, task: ProcessingTask) -> ProcessingResult:
        """
        Process the given task
        
        Args:
            task: The processing task to execute
            
        Returns:
            ProcessingResult with the output
        """
        pass
    
    def _prepare_document_parts(self, documents: list[Document]) -> list:
        """
        Convert documents to Gemini API parts
        
        Args:
            documents: List of Document objects
            
        Returns: 
            List of Gemini API Part objects
        """
        from google.genai import types
        
        parts = []
        for doc in documents:
            try:
                if isinstance(doc. content, bytes):
                    parts.append(types.Part. from_bytes(
                        data=doc.content,
                        mime_type=doc. doc_type. value
                    ))
                elif doc.is_url():
                    # For URLs, fetch content first
                    import httpx
                    response = httpx.get(str(doc.content), timeout=30.0)
                    response.raise_for_status()
                    parts.append(types. Part.from_bytes(
                        data=response. content,
                        mime_type=doc.doc_type.value
                    ))
                elif isinstance(doc.content, str):
                    parts.append(types.Part.from_text(doc.content))
                elif isinstance(doc.content, Path):
                    # Read from path
                    content = doc.content.read_bytes()
                    parts.append(types.Part.from_bytes(
                        data=content,
                        mime_type=doc. doc_type.value
                    ))
                else:
                    logger.warning(f"Unsupported content type for document: {type(doc.content)}")
                    
            except Exception as e:
                logger.error(f"Failed to prepare document part: {e}")
                raise ValueError(f"Could not process document: {e}")
        
        return parts
    
    def _build_prompt(
        self,
        task: ProcessingTask,
        template: Optional[str] = None
    ) -> str:
        """
        Build the prompt from task and template
        
        Args: 
            task: Processing task
            template: Optional prompt template
            
        Returns: 
            Formatted prompt string
        """
        if template:
            # Replace placeholders in template
            prompt = template.format(
                boundary_token=task.boundary_token or "",
                additional_context=task.additional_context or "",
                output_format=task.output_format or "markdown",
            )
        else:
            # Build basic prompt
            prompt = f"""
Task: {task.action.value}

{task.additional_context or ''}

Please provide the output in {task.output_format or 'markdown'} format. 
"""
        
        # Add custom prompt if provided
        if task. prompt:
            prompt = f"{prompt}\n\nAdditional instructions:  {task.prompt}"
        
        return prompt


class UniversalDocumentAgent(BaseDocumentAgent):
    """
    Universal agent that can handle any document type and action
    
    This is the main workhorse agent for document processing
    """
    
    async def process(self, task: ProcessingTask) -> ProcessingResult: 
        """
        Process any document with any action
        
        Args: 
            task: The processing task
            
        Returns:
            ProcessingResult with the output
        """
        try:
            # Prepare content parts
            parts = self._prepare_document_parts(task.documents)
            
            # Build prompt
            prompt = self._build_prompt(task)
            parts.append(self.client.models.generate_content.__self__.types.Part.from_text(prompt))
            
            # Import types here to avoid circular imports
            from google.genai import types
            parts.append(types.Part.from_text(prompt))
            
            # Generate response
            logger.info(f"Processing {len(task.documents)} document(s) with action:  {task.action.value}")
            
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.config.model.value,
                contents=parts,
                config=types.GenerateContentConfig(**self.config.to_generation_config())
            )
            
            # Extract token usage if available
            tokens_used = None
            if hasattr(response, 'usage_metadata'):
                tokens_used = getattr(response. usage_metadata, 'total_token_count', None)
            
            logger.info(f"Processing completed successfully. Tokens used: {tokens_used}")
            
            return ProcessingResult(
                success=True,
                content=response.text,
                model_used=self.config.model.value,
                tokens_used=tokens_used,
                metadata={
                    "action": task.action.value,
                    "session_id": task.session_id,
                    "document_count": len(task.documents),
                }
            )
            
        except Exception as e:
            logger. error(f"Processing failed: {e}", exc_info=True)
            return ProcessingResult(
                success=False,
                content=None,
                model_used=self.config.model.value,
                error=str(e),
                metadata={"action": task.action.value}
            )


class MultiStepAgent(BaseDocumentAgent):
    """
    Agent that performs multi-step reasoning and processing
    
    Breaks down complex tasks into multiple steps with intermediate reasoning
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conversation_history:  list[dict] = []
    
    async def process(self, task: ProcessingTask) -> ProcessingResult:
        """
        Multi-step processing with reasoning
        
        Args:
            task: The processing task
            
        Returns:
            ProcessingResult with final output and step metadata
        """
        steps = [
            ("analyze", "First, analyze the document structure and content type. "),
            ("extract", "Extract the most relevant information based on the task."),
            ("synthesize", "Synthesize findings and generate the final output.")
        ]
        
        results = []
        parts = self._prepare_document_parts(task.documents)
        
        for step_name, step_prompt in steps:
            full_prompt = f"""
Step:  {step_name}
{step_prompt}

Task:  {task.action.value}
{task.additional_context or ''}

Previous steps results:  {results[-1]['result'] if results else 'None'}

Please provide output for this step in {task.output_format} format.
"""
            
            try:
                from google.genai import types
                
                logger.info(f"Executing step: {step_name}")
                
                response = await asyncio.to_thread(
                    self.client. models.generate_content,
                    model=self.config.model.value,
                    contents=parts + [types.Part.from_text(full_prompt)],
                    config=types.GenerateContentConfig(**self.config.to_generation_config())
                )
                
                results.append({
                    "step": step_name,
                    "result": response.text
                })
                
            except Exception as e:
                logger.error(f"Step {step_name} failed: {e}")
                results.append({
                    "step": step_name,
                    "error": str(e)
                })
        
        # Check if all steps succeeded
        all_success = all("error" not in r for r in results)
        final_content = results[-1]. get("result") if results and all_success else None
        
        return ProcessingResult(
            success=all_success,
            content=final_content,
            model_used=self.config.model.value,
            metadata={
                "steps": results,
                "action": task.action.value,
                "multi_step":  True,
            }
        )


class BatchDocumentProcessor:
    """
    Process multiple documents in batch with different configurations
    
    Supports parallel and sequential processing
    """
    
    def __init__(
        self,
        default_model: GeminiModel = GeminiModel.GEMINI_2_5_FLASH,
        api_key: Optional[str] = None
    ):
        """
        Initialize batch processor
        
        Args: 
            default_model: Default model to use
            api_key:  Google API key
        """
        self.default_config = ModelConfig(model=default_model)
        self.api_key = api_key
        self.agents:  dict[str, BaseDocumentAgent] = {}
    
    def register_agent(self, name: str, agent: BaseDocumentAgent):
        """Register a custom agent"""
        self.agents[name] = agent
        logger.info(f"Registered agent: {name}")
    
    async def process_batch(
        self,
        tasks: list[ProcessingTask],
        agent_name: Optional[str] = None,
        parallel: bool = True
    ) -> list[ProcessingResult]:
        """
        Process multiple tasks
        
        Args:
            tasks: List of processing tasks
            agent_name: Optional specific agent to use
            parallel: Whether to process in parallel
            
        Returns: 
            List of ProcessingResults
        """
        agent = self.agents.get(agent_name) or UniversalDocumentAgent(
            api_key=self.api_key,
            model_config=self.default_config
        )
        
        if parallel:
            logger.info(f"Processing {len(tasks)} tasks in parallel")
            results = await asyncio.gather(*[agent.process(task) for task in tasks])
        else:
            logger.info(f"Processing {len(tasks)} tasks sequentially")
            results = []
            for task in tasks:
                result = await agent.process(task)
                results.append(result)
        
        return results
