
import logging
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ai_media.utils.system import get_optimal_device_and_dtype
from ai_media.generators.description import generate_ocr

logger = logging.getLogger(__name__)

class ImageTranslationGenerator:
    def __init__(self):
        self.device, self.dtype = get_optimal_device_and_dtype()
        # Use ArticleGenerator for robust Text Translation (NLLB, LLMs)
        from ai_media.generators.text import ArticleGenerator
        self.translator = ArticleGenerator()

    def run(self, input_path: str, target_lang: str, render_method: str = "smart", translate_model: str = "nllb-200-3.3b", ocr_model: str = "florence", output_path: str = None, on_ready: callable = None):
        """
        Run Image-to-Image translation.
        
        Args:
            input_path: Path to input image.
            target_lang: Target language code.
            render_method: Render method (currently only "smart" is supported).
            translate_model: Model ID for text translation step.
            ocr_model: OCR model to use (florence/qwen-vl).
            output_path: Output file path.
            on_ready: Optional callback called when translation model is ready.
        """
        logger.info(f"Running Image Translation (Smart Logic) -> {target_lang}")

        # 1. OCR (Get text + boxes)
        from ai_media.conversion.ocr import image_to_text_with_coords, unload_ocr_model
        
        try:
            ocr_data = image_to_text_with_coords(input_path, model_type=ocr_model, on_ready=on_ready)
        finally:
            unload_ocr_model()
        
        if not ocr_data:
            raise RuntimeError("OCR failed to detect text regions.")

        # 2. Translate extracted text
        translated_data = []
        
        for item in ocr_data:
            original_text = item['text']
            if len(original_text.strip()) < 1:
                continue

            try:
                translated_text = self.translator.translate_text(
                    content=original_text, 
                    target_lang=target_lang, 
                    source_lang="auto",
                    model_id=translate_model,
                    keep_loaded=True,  # optimization
                )
                    
                if not translated_text:
                    translated_text = original_text
            except Exception as e:
                logger.warning(f"Translation failed for '{original_text}': {e}")
                translated_text = original_text  # Fallback
                if not translated_text:
                    translated_text = original_text
            except Exception as e:
                logger.warning(f"Translation failed for '{original_text}': {e}")
                translated_text = original_text  # Fallback

            item['translated_text'] = translated_text
            translated_data.append(item)

        # 3. Render using Smart Logic (Pillow/OpenCV inpainting)
        return self._render_smart(input_path, translated_data, output_path)

    def _render_smart(self, input_path, data, output_path):
        """
        Use Pillow/OpenCV to inpaint text regions and overlay new text.
        """
        # Load image
        img_pil = Image.open(input_path).convert("RGB")
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        draw = ImageDraw.Draw(img_pil)

        for item in data:
            box = item['box']  # [x1, y1, x2, y2]
            translated_text = item['translated_text']
            
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            
            # Simple Inpainting (Remove old text)
            # Create mask for this box
            mask = np.zeros(img_cv.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)
            
            # Inpaint (Telea algorithm for speed)
            img_cv = cv2.inpaint(img_cv, mask, 3, cv2.INPAINT_TELEA)

        # Convert back to PIL for text drawing
        img_result = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_result)

        # Load font (with fallback)
        try:
            font = ImageFont.truetype("arial.ttf", 20) 
        except:
            font = ImageFont.load_default()

        for item in data:
            box = item['box']
            translated_text = item['translated_text']
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            
            # Auto-size font based on box height
            font_size = max(10, int(h * 0.8))
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                pass
            
            # Center text in box
            text_length = draw.textlength(translated_text, font=font)
            text_x = x1 + (w - text_length) / 2
            text_y = y1 + (h - font_size) / 2
             
            draw.text((text_x, text_y), translated_text, fill=(0, 0, 0), font=font)

        if output_path:
            img_result.save(output_path)
        
        return output_path
