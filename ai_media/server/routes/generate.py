"""Image, video, and audio generation routes."""

from datetime import datetime

from fastapi import APIRouter

from ..models import ImageGenerateRequest, VideoGenerateRequest, AudioGenerateRequest
from ..jobs import create_job
from ..process_manager import spawn_job_process
from ..tasks import image as image_tasks
from ..tasks import video as video_tasks
from ..tasks import audio as audio_tasks

router = APIRouter(tags=["Generation"])


@router.post("/api/generate/image")
async def generate_image(request: ImageGenerateRequest):
    """Start an image generation job."""
    job = create_job(
        "image",
        prompt=request.prompt,
        model=request.model,
        params={
            "width": request.width,
            "height": request.height,
            "steps": request.steps,
            "guidance_scale": request.guidance_scale,
        }
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = request.output_filename or f"image_{timestamp}.png"
    output_path = f"output/{filename}"
    
    spawn_job_process(
        job["job_id"],
        image_tasks.run_image_generation,
        (
            job["job_id"],
            request.prompt,
            output_path,
            request.width,
            request.height,
            request.model,
            request.steps,
            request.guidance_scale,
        ),
    )
    
    return {"job_id": job["job_id"], "status": "pending", "output_path": output_path}


@router.post("/api/generate/video")
async def generate_video(request: VideoGenerateRequest):
    """Start a video generation job."""
    job = create_job(
        "video",
        prompt=request.prompt,
        model=request.model,
        params={
            "width": request.width,
            "height": request.height,
            "duration": request.duration,
            "fps": request.fps,
        }
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = request.output_filename or f"video_{timestamp}.mp4"
    output_path = f"output/{filename}"
    
    spawn_job_process(
        job["job_id"],
        video_tasks.run_video_generation,
        (
            job["job_id"],
            request.prompt,
            output_path,
            request.width,
            request.height,
            request.duration,
            request.fps,
            request.model,
            request.input_image,
        ),
    )
    
    return {"job_id": job["job_id"], "status": "pending", "output_path": output_path}


@router.post("/api/generate/audio")
async def generate_audio(request: AudioGenerateRequest):
    """Start an audio generation job."""
    job = create_job(
        "audio",
        prompt=request.prompt,
        model=request.model,
        params={"duration": request.duration}
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = request.output_filename or f"audio_{timestamp}.wav"
    output_path = f"output/{filename}"
    
    spawn_job_process(
        job["job_id"],
        audio_tasks.run_audio_generation,
        (
            job["job_id"],
            request.prompt,
            output_path,
            request.duration,
            request.model,
        ),
    )
    
    return {"job_id": job["job_id"], "status": "pending", "output_path": output_path}

