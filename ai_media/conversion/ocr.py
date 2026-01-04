
"""
OCR module using Microsoft Florence-2 (VLM).
Supports high-accuracy full-page text extraction.
"""

import sys
import torch
from PIL import Image
from diffusers.utils import load_image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForImageTextToText
import time

# Global cache for model and processor to avoid reloading
_processor = None
_model = None
_current_model_type = None

def load_ocr_model(model_type="qwen-vl"):
    """Load OCR model and processor (cached)."""
    global _processor, _model, _current_model_type
    
    if _model is not None and _current_model_type == model_type:
        return _processor, _model
        
    print(f"⏳ Loading OCR model ({model_type})...")
    
    try:
        if model_type == "florence":
            model_name = "microsoft/Florence-2-large"
        elif model_type == "qwen-vl":
            model_name = "Qwen/Qwen3-VL-8B-Instruct"
        else:
            raise ValueError(f"Unsupported OCR model: {model_type}")
        
        # Select device
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            
        # Determine dtype
        # Check for bfloat16 support helper if available, else default rules
        try:
            from ai_media.utils.system import is_bfloat16_supported
            if device.type == "cuda":
                dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
            else:
                dtype = torch.float32
        except ImportError:
             dtype = torch.float32
             
        if model_type == "florence":
            _processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            _model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                trust_remote_code=True,
                attn_implementation="eager",
                torch_dtype=dtype
            ).to(device)
        else: # qwen-vl
            _model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                device_map="auto" if device.type != "mps" else None,
                trust_remote_code=True,
                torch_dtype="auto" 
            )
            if device.type == "mps":
                _model = _model.to(device)
            _processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            
        _current_model_type = model_type
        print(f"✅ OCR model loaded on {device.type.upper()}")
        
        return _processor, _model
    except Exception as e:
        print(f"❌ Failed to load OCR model: {e}")
        return None, None

def image_to_text(image_path, model_type="qwen-vl"):
    """
    Extract text from an image using chosen OCR model.
    
    Args:
        image_path (str): Path to image file
        model_type (str): 'florence' or 'qwen-vl'
        
    Returns:
        str: Extracted text
    """
    processor, model = load_ocr_model(model_type)
    if not model:
        raise RuntimeError("OCR model failed to load")
        
    try:
        # Load image
        image = load_image(image_path).convert("RGB")
        
        if model_type == "florence":
            # Task prompt
            task_prompt = "<OCR>"
            
            # Prepare inputs
            inputs = processor(text=task_prompt, images=image, return_tensors="pt")
            
            # Move inputs to device and cast correct dtype
            device = model.device
            dtype = model.dtype
            
            if device.type == "mps":
                 # MPS often needs float32 for input_ids/processing
                inputs["pixel_values"] = inputs["pixel_values"].to(device, torch.float32)
                inputs["input_ids"] = inputs["input_ids"].to(device)
            else:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype, device=device)
                inputs["input_ids"] = inputs["input_ids"].to(device=device)
                
            # Generate
            # Disable cache to avoid MPS past_key_values crash (common issue on Mac)
            t0 = time.time()
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    do_sample=False,
                    num_beams=1,
                    use_cache=False, 
                    early_stopping=False,
                )
            t1 = time.time()
            
            # Decode and Post-process
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = processor.post_process_generation(
                generated_text, task=task_prompt, 
                image_size=(image.width, image.height)
            )
            result_text = parsed_answer[task_prompt]
        else: # qwen-vl
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Read all the text in this image verbatim."},
                    ],
                }
            ]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt",
            ).to(model.device)
            
            t0 = time.time()
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=1024)
            t1 = time.time()
            
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            result_text = output_text[0]

        print(f"   (⏱️  OCR Inference Time: {t1 - t0:.2f}s)")
        return result_text
        
    except Exception as e:
        print(f"❌ OCR extraction failed: {e}")
        # Improve error message if it's likely a dependency issue
        if "diffusers" in str(e):
             print("   (HINT: Ensure 'diffusers' is installed: pip install diffusers)")
        raise e
