
import logging
import torch
from pathlib import Path
from ai_media.models import TRANSLATION_MODELS, get_model_id
from ai_media.utils.system import get_optimal_device_and_dtype

logger = logging.getLogger(__name__)

class TranslationGenerator:
    def __init__(self):
        self.device, self.dtype = get_optimal_device_and_dtype()
        self.processor = None
        self.model = None
        self.current_model_id = None

    def load_model(self, model_key="default_audio"):
        model_id = get_model_id(model_key, TRANSLATION_MODELS)
        
        if self.model and self.current_model_id == model_id:
            return

        logger.info(f"Loading Translation model: {model_id}")
        
        # Free previous model
        if self.model:
            del self.model
            del self.processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        try:
            from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToSpeech
            
            if "seamless" in model_id.lower():
                self.processor = AutoProcessor.from_pretrained(model_id)
                self.model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(
                    model_id, 
                    torch_dtype=self.dtype
                ).to(self.device)
            else:
                 # Standard NLLB or LLM loading here if strictly needed by this class
                 # But we might offload Text translation to existing ArticleGenerator 
                 # or create a dedicated LLM translator. For now, focus on Seamless.
                 pass

            self.current_model_id = model_id
            logger.info("Translation model loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load translation model {model_id}: {e}")
            raise RuntimeError(f"Could not load model {model_id}") from e

    def run(self, input_path: str, target_lang: str, task: str = "s2st", output_path: str = None):
        """
        Run translation task.
        task: 's2st' (Speech-to-Speech), 's2tt' (Speech-to-Text), 't2st' (Text-to-Speech), etc.
        """
        if not self.model:
            self.load_model()

        # Prepare inputs based on task
        if task == "t2tt":
            # Text-to-Text: input_path is the text string
            # No audio loading needed
            text_inputs = self.processor(text=input_path, src_lang="eng", return_tensors="pt").to(self.device)
            
            # Map target lang
            tgt_lang_3 = self._map_lang_code(target_lang)
            
            output_tokens = self.model.generate(
                **text_inputs,
                tgt_lang=tgt_lang_3,
                generate_speech=False
            )
            txt = self.processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
            return txt

        # Audio Tasks (S2ST, S2TT)
        import torchaudio
        
        try:
            audio_input, sample_rate = torchaudio.load(input_path)
            
            # Resample if needed
            if sample_rate != self.processor.feature_extractor.sampling_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.processor.feature_extractor.sampling_rate)
                audio_input = resampler(audio_input)
                sample_rate = self.processor.feature_extractor.sampling_rate

            processed_inputs = self.processor(
                audios=audio_input, 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            ).to(self.device)

            tgt_lang_3 = self._map_lang_code(target_lang)

            # Generate
            # S2ST returns audio
            if task == "s2st":
                output = self.model.generate(
                    **processed_inputs,
                    tgt_lang=tgt_lang_3,
                    return_intermediate_token_ids=False
                )
                waveform = output[0].cpu()
                sample_rate = self.model.config.sampling_rate
                
                if output_path:
                    torchaudio.save(output_path, waveform, sample_rate)
                return output_path

            # S2TT returns text tokens -> decode
            elif task == "s2tt":
                output_tokens = self.model.generate(
                    **processed_inputs,
                    tgt_lang=tgt_lang_3,
                    generate_speech=False
                )
                txt = self.processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
                return txt

        except Exception as e:
            logger.error(f"Translation generation failed: {e}")
            raise

    def _map_lang_code(self, code: str) -> str:
        # Simple mapping for common codes. seamless supports ~100.
        # Standard iso 639-3 usually works but Seamless has specifics like "eng", "spa", "fra", "deu", "zho" (CMN?)
        # Refer to Seamless m4t docs for precise list. 
        # For now, simplistic mapping:
        mapping = {
            "es": "spa", "en": "eng", "fr": "fra", "de": "deu", "it": "ita", 
            "pt": "por", "zh": "cmn", "ja": "jpn", "ko": "kor", "ru": "rus",
            "hi": "hin", "ar": "arb"
        }
        return mapping.get(code, "eng")
