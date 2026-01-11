
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

def load_ocr_model(model_type="florence"):
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
                # Florence-2 can be unstable with bfloat16 on some setups/versions
                # Force float16 for Florence, allow bfloat16 for others
                if model_type == "florence":
                    dtype = torch.float16
                else:
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

def image_to_text(image_path, model_type="florence"):
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

def unload_ocr_model():
    """Unload OCR model from VRAM/RAM."""
    global _processor, _model, _current_model_type
    
    if _model is not None:
        print("🧹 Unloading OCR Model...")
        del _model
        del _processor
        
        _model = None
        _processor = None
        _current_model_type = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        elif torch.backends.mps.is_available():
             torch.mps.empty_cache()
             
        import gc
        gc.collect()
        print("✅ OCR Model unloaded.")

def image_to_text_with_coords(image_path, model_type="florence", on_ready=None):
    """
    Extract text AND coordinates using Florence-2 (<OCR_WITH_REGION>).
    
    Args:
        image_path: Input image
        model_type: Must be 'florence' (Qwen-VL doesn't support easy coords)
        on_ready: Optional callback to trigger when model is loaded.
        
    Returns:
        List of dicts: [{'text': str, 'box': [x1, y1, x2, y2]}, ...]
    """
    # Support both models now
    # if model_type != "florence": ... (Removed hard switch)
        
    processor, model = load_ocr_model(model_type)
    if not model:
        raise RuntimeError("OCR model failed to load")
        
    if on_ready:
        try:
            on_ready()
        except Exception as e:
            print(f"⚠️  on_ready callback failed: {e}")

    try:
        image = load_image(image_path).convert("RGB")
        
        if model_type == "florence":
            task_prompt = "<OCR_WITH_REGION>"
            
            inputs = processor(text=task_prompt, images=image, return_tensors="pt")
            
            # Florence-specific device handling...
            device = model.device
            dtype = model.dtype
            
            if device.type == "mps":
                 inputs["pixel_values"] = inputs["pixel_values"].to(device, torch.float32)
                 inputs["input_ids"] = inputs["input_ids"].to(device)
            else:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype, device=device)
                inputs["input_ids"] = inputs["input_ids"].to(device=device)

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    do_sample=False,
                    num_beams=1,
                    use_cache=False 
                )

            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = processor.post_process_generation(
                generated_text, 
                task=task_prompt, 
                image_size=(image.width, image.height)
            )
            
            data = parsed_answer.get(task_prompt, {})
            quad_boxes = data.get('quad_boxes', [])
            labels = data.get('labels', [])
            
            results = []
            for box, label in zip(quad_boxes, labels):
                xs = [box[0], box[2], box[4], box[6]]
                ys = [box[1], box[3], box[5], box[7]]
                results.append({
                    "text": label,
                    "box": [min(xs), min(ys), max(xs), max(ys)]
                })
            return results

        else: # Qwen-VL (Experimental)
            # Qwen-VL requires specific prompting for grounding/detection
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Read the text in the image. Return each text line followed by its bounding box in format [x1, y1, x2, y2]. Do not include the word 'box' or 'ref' in the output. Just text and coordinates."},
                    ],
                }
            ]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=2048)
                
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # Qwen-VL parsing
            # Clean up response first to avoid "box" artifacts in text
            clean_text = output_text.replace("<box>", "").replace("</box>", "").replace("<ref>", "").replace("</ref>", "")
            
            results = []
            lines = clean_text.split('\n')
            
            import re
            for line in lines:
                # Look for box pattern like [100, 200, 300, 400] or (100,200),(300,400)
                # Qwen often outputs: text [x1,y1,x2,y2]
                box_match = re.search(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', line)
                if not box_match:
                    # Try parenthesis format (x1,y1),(x2,y2) from older Qwen-VL
                    box_match = re.search(r'\((\d+),(\d+)\),\s*\((\d+),(\d+)\)', line)
                
                if box_match:
                    coords = [int(g) for g in box_match.groups()]
                    
                    # Extract text: everything before the box
                    text_content = line[:box_match.start()].strip()
                    
                    # Remove "box" prefix if present (common artifact)
                    if text_content.lower().startswith("box"):
                         text_content = text_content[3:].strip()
                    
                    # Remove any straggling special chars
                    text_content = text_content.strip('[](), ')
                    
                    # Normalize if coords are in 1000 scale
                    if all(c <= 1000 for c in coords):
                        w, h = image.width, image.height
                        coords = [
                            coords[0] / 1000 * w,
                            coords[1] / 1000 * h,
                            coords[2] / 1000 * w,
                            coords[3] / 1000 * h
                        ]
                    
                    if text_content and len(text_content) > 0:
                        results.append({"text": text_content, "box": coords})
            
            if not results:
                # If structured parsing failed, try to just return the full text if it looks like one block
                # Only if it's not full of coordinate garbage
                if "[" not in clean_text and "(" not in clean_text:
                     # Just return full text as one big box covering image
                     results.append({
                        "text": clean_text.strip(),
                        "box": [0, 0, image.width, image.height]
                     })
                else:
                    print(f"⚠️  Qwen-VL Layout extraction returned complex text, naive parse failed: {output_text[:100]}...")
                
            return results
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Layout OCR failed: {e}")
        return []
