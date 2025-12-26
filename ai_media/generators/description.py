"""
Description/caption generation module for AI-Media.

Supports: Florence-2 (SOTA) and BLIP for image and video captioning.
"""


def generate_caption(input_path, device, quiet=False, model_type="florence"):
    """
    Generate a text description for an image or video.
    
    Models: 
      - 'florence' (Microsoft Florence-2-Large, SOTA)
      - 'blip' (Salesforce BLIP-Large, Classic)
      
    Args:
        input_path: Path to image or video file
        device: Torch device to use
        quiet: Suppress progress output
        model_type: 'florence' or 'blip'
        
    Returns:
        Caption string or None on failure
    """
    try:

        
        # ----------------------------------------------------------------
        # MODEL: BLIP
        # ----------------------------------------------------------------
        if model_type == "blip":
            from transformers import BlipProcessor, BlipForConditionalGeneration
            from diffusers.utils import load_image
            import torch
            from PIL import Image
            import cv2
            import numpy as np
            from ..utils.system import is_bfloat16_supported

            caption_model_id = "Salesforce/blip-image-captioning-large"
            if not quiet:
                print(f"   Loading Caption Model: {caption_model_id}...")
            
            # Determine dtype (prefer bf16 on CUDA)
            if device.type == "cuda":
                dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
            else:
                dtype = torch.float32
            
            processor = BlipProcessor.from_pretrained(caption_model_id, use_fast=True)
            model = BlipForConditionalGeneration.from_pretrained(caption_model_id, torch_dtype=dtype).to(device)
            
            if not quiet:
                dtype_name = str(dtype).replace("torch.", "")
                print(f"   Platform: {device.type.upper()} | Dtype: {dtype_name}")
            
            # Check if video
            ext = input_path.lower().split('.')[-1]
            is_video = ext in ['mp4', 'mov', 'avi', 'mkv', 'webm', 'flv', 'gif']
            
            if is_video:
                # Video Logic
                cap = cv2.VideoCapture(input_path)
                if not cap.isOpened():
                    return "Unknown video content"
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                num_samples = 10
                indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
                captions = []
                for i, idx in enumerate(indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    
                    inputs = processor(pil_image, return_tensors="pt").to(device)
                    out = model.generate(**inputs)
                    frame_caption = processor.decode(out[0], skip_special_tokens=True)
                    captions.append(frame_caption)
                cap.release()
                return ", ".join(captions)
            else:
                # Image Logic
                raw_image = load_image(input_path).convert('RGB')
                inputs = processor(raw_image, return_tensors="pt").to(device)
                out = model.generate(**inputs)
                caption = processor.decode(out[0], skip_special_tokens=True)
                if not quiet:
                    print(f"   Detected: '{caption}'")
                return caption

        # ----------------------------------------------------------------
        # MODEL: Florence-2 (Default/SOTA)
        # ----------------------------------------------------------------
        else:
            from transformers import AutoProcessor, AutoModelForCausalLM
            from diffusers.utils import load_image
            import cv2
            import numpy as np
            import torch
            from PIL import Image
            from ..utils.system import is_bfloat16_supported
            
            # Load Florence-2 (SOTA Captioning, ~1.5GB)
            caption_model_id = "microsoft/Florence-2-large"
            
            if not quiet:
                print(f"   Loading Caption Model: {caption_model_id}...")
            
            # Determine dtype (prefer bf16 on CUDA)
            if device.type == "cuda":
                dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
            else:
                dtype = torch.float32
            
            processor = AutoProcessor.from_pretrained(caption_model_id, trust_remote_code=True)
            
            # Use eager attention to avoid SDPA crashes on MPS/Mac with recent transformers
            model = AutoModelForCausalLM.from_pretrained(
                caption_model_id, 
                trust_remote_code=True,
                attn_implementation="eager",
                torch_dtype=dtype
            ).to(device)
            
            if not quiet:
                dtype_name = str(dtype).replace("torch.", "")
                print(f"   Platform: {device.type.upper()} | Dtype: {dtype_name}") 
            
            # Task prompt for Florence-2
            task_prompt = "<MORE_DETAILED_CAPTION>"
            
            # Check if video
            ext = input_path.lower().split('.')[-1]
            is_video = ext in ['mp4', 'mov', 'avi', 'mkv', 'webm', 'flv', 'gif']
            
            if is_video:
                cap = cv2.VideoCapture(input_path)
                if not cap.isOpened():
                    if not quiet:
                        print(f"⚠️  Could not open video: {input_path}")
                    return "Unknown video content"
                    
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = total_frames / fps if fps > 0 else 0
                
                # Select 10 evenly distributed frames
                num_samples = 10
                indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
                
                captions = []
                if not quiet:
                    print(f"   Analyzing {num_samples} frames from video ({duration:.1f}s)...")
                
                for i, idx in enumerate(indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    # Convert BGR (OpenCV) to RGB (PIL)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    
                    # Generate caption (Florence-2)
                    inputs = processor(text=task_prompt, images=pil_image, return_tensors="pt")
                    
                    # Ensure pixel_values are correct dtype
                    if device.type == "mps":
                        inputs["pixel_values"] = inputs["pixel_values"].to(device, torch.float32)
                        inputs["input_ids"] = inputs["input_ids"].to(device)
                    else:
                        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype, device=device)
                        inputs["input_ids"] = inputs["input_ids"].to(device=device)

                    # Disable cache to avoid MPS past_key_values crash
                    generated_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=1024,
                        do_sample=False,
                        num_beams=3,
                        use_cache=False,
                    )
                    
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                    parsed_answer = processor.post_process_generation(
                        generated_text, task=task_prompt, 
                        image_size=(pil_image.width, pil_image.height)
                    )
                    frame_caption = parsed_answer[task_prompt]
                    
                    timestamp = idx / fps if fps > 0 else 0
                    captions.append(f"[{timestamp:.1f}s]: {frame_caption}")
                    if not quiet:
                        print(f"   Frame {i+1}/{num_samples} ({timestamp:.1f}s): {frame_caption}")
                    
                cap.release()
                
                # Consolidated description
                summary = ", ".join([c.split(": ")[1] for c in captions])
                return summary 
                
            else:
                # Image handling
                raw_image = load_image(input_path).convert('RGB')
                
                inputs = processor(text=task_prompt, images=raw_image, return_tensors="pt")
                
                # Ensure pixel_values are correct dtype
                if device.type == "mps":
                    inputs["pixel_values"] = inputs["pixel_values"].to(device, torch.float32)
                    inputs["input_ids"] = inputs["input_ids"].to(device)
                else:
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype, device=device)
                    inputs["input_ids"] = inputs["input_ids"].to(device=device)

                # Disable cache to avoid MPS past_key_values crash
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    do_sample=False,
                    num_beams=3,
                    use_cache=False,
                )
                
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed_answer = processor.post_process_generation(
                    generated_text, task=task_prompt, 
                    image_size=(raw_image.width, raw_image.height)
                )
                caption = parsed_answer[task_prompt]
                
                if not quiet:
                    print(f"   Detected: '{caption}'")
                return caption
                
    except ImportError as e:
        print(f"❌ Error: Missing dependencies for captioning. {e}")
        return None
    except Exception as e:
        print(f"❌ Caption generation failed: {e}")
        return None
