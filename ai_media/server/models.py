"""Pydantic models for API requests and responses."""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


# --- Response Models ---

class SystemInfo(BaseModel):
    """System information response."""
    device: str
    dtype: str
    cuda_available: bool
    mps_available: bool
    gpu_name: Optional[str] = None
    vram_total_gb: Optional[float] = None
    ram_total_gb: float
    platform: str
    python_version: str


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    model_id: str
    category: str
    vram_required: Optional[int] = None
    ram_required: Optional[int] = None
    max_resolution: Optional[tuple] = None
    max_duration: Optional[int] = None


class ResourceStats(BaseModel):
    """Real-time resource statistics."""
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    vram_used_gb: float
    vram_total_gb: float
    gpu_percent: float
    timestamp: str


class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    status: str
    progress: int
    phase: str
    message: str
    result_path: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


# --- Request Models ---

class ConfigUpdateRequest(BaseModel):
    """Configuration update request."""
    theme: Optional[str] = None


class ImageGenerateRequest(BaseModel):
    """Image generation request."""
    prompt: str = Field(..., description="Text prompt for image generation")
    model: str = Field("default", description="Model name (e.g., 'flux', 'sdxl', 'sd-1.5')")
    width: int = Field(1024, description="Image width")
    height: int = Field(1024, description="Image height")
    steps: int = Field(30, description="Number of inference steps")
    guidance_scale: float = Field(7.5, description="Guidance scale for generation")
    negative_prompt: Optional[str] = Field("", description="Negative prompt (what to avoid)")
    output_filename: Optional[str] = Field(None, description="Custom output filename")
    force: bool = Field(False, description="Force execution, skipping confirmations (overwrites and warnings)")
    bypass_warning: bool = Field(False, description="Specifically skip resource warning prompts")


class VideoGenerateRequest(BaseModel):
    """Video generation request."""
    prompt: str = Field(..., description="Text prompt for video generation")
    model: str = Field("default", description="Model name")
    width: int = Field(512, description="Video width")
    height: int = Field(512, description="Video height")
    duration: float = Field(3.0, description="Duration in seconds")
    fps: int = Field(8, description="Frames per second")
    input_image: Optional[str] = Field(None, description="Path to input image for img2vid")
    output_filename: Optional[str] = Field(None, description="Custom output filename")
    force: bool = Field(False, description="Force execution, skipping confirmations (overwrites and warnings)")
    bypass_warning: bool = Field(False, description="Specifically skip resource warning prompts")


class AudioGenerateRequest(BaseModel):
    """Audio generation request."""
    prompt: str = Field(..., description="Text prompt for audio generation")
    model: str = Field("default", description="Model name")
    duration: float = Field(10.0, description="Duration in seconds")
    output_filename: Optional[str] = Field(None, description="Custom output filename")
    force: bool = Field(False, description="Force execution, skipping confirmations (overwrites and warnings)")
    bypass_warning: bool = Field(False, description="Specifically skip resource warning prompts")


class ArticleGenerateRequest(BaseModel):
    """Article generation request."""
    topic: str = Field(..., description="Topic for article generation")
    model: str = Field("default", description="LLM model name")
    format: str = Field("md", description="Output format (md, html, pdf, docx)")
    length: str = Field("standard", description="quick, standard, or detailed")
    online: bool = Field(False, description="Enable online research")
    research_iterations: int = Field(3, description="Number of research iterations (if online)")
    max_images: int = Field(5, description="Maximum number of images to fetch (if online)")
    output_filename: Optional[str] = Field(None, description="Custom output filename")
    bypass_warning: bool = Field(False, description="Specifically skip resource warning prompts")


class CodeGenerateRequest(BaseModel):
    """Code generation request."""
    prompt: str = Field(..., description="Code generation prompt")
    model: str = Field("default", description="LLM model name")
    output_name: str = Field("", description="Output name (optional). Empty = auto-generated. Multi-file = zip filename")
    bypass_warning: bool = Field(False, description="Specifically skip resource warning prompts")


class TransformRequest(BaseModel):
    """Image transformation request."""
    input_path: str = Field(..., description="Path to input image")
    instruction: str = Field(..., description="Transformation instruction (or 'remove-bg')")
    model: str = Field("default", description="Model name (e.g., 'instruct-pix2pix', 'remove-bg')")
    guidance_scale: float = Field(7.5, description="Text guidance scale")
    image_guidance_scale: float = Field(1.5, description="Image guidance scale")
    silhouette: bool = Field(False, description="Create black silhouette instead of transparent (for rembg)")
    output_filename: Optional[str] = Field(None, description="Custom output filename")
    bypass_warning: bool = Field(False, description="Specifically skip resource warning prompts")


class ConvertRequest(BaseModel):
    """Media conversion request."""
    input_path: str = Field(..., description="Path to input file")
    target_format: str = Field(..., description="Target format (e.g., 'mp4', 'gif', 'pdf')")
    ocr_enabled: bool = Field(False, description="Enable OCR for images/scanned PDFs")
    ocr_model: str = Field("qwen-vl", description="OCR model to use ('qwen-vl', 'florence')")
    output_filename: Optional[str] = Field(None, description="Custom output filename")


class UpscaleRequest(BaseModel):
    """Upscale request."""
    input_path: str = Field(..., description="Path to input media")
    factor: float = Field(2.0, description="Upscale factor (2.0 or 4.0)")
    method: str = Field("fast", description="Method: 'fast' (Real-ESRGAN), 'ai' (Latent), 'simple' (Lanczos)")
    strength: float = Field(0.3, description="Denoising strength for AI upscale (0.0-1.0)")
    output_filename: Optional[str] = Field(None, description="Custom output filename")
    force: bool = Field(False, description="Force execution, skipping confirmations (overwrites and warnings)")
    bypass_warning: bool = Field(False, description="Specifically skip resource warning prompts")


class TextExportRequest(BaseModel):
    """Text export request (MD, TXT, HTML, JSON, PDF, RTF, DOCX)."""
    content: str = Field(..., description="Content to export")
    format: str = Field(..., description="Target format")
    filename: str = Field(..., description="Target filename")
