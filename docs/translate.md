# Translation & Subtitles

AI-Media provides powerful AI-based translation capabilities for both text and media (audio/video).

## Interactive Menu

Select **Translate** from the main menu (Option 11) or start directly:
```bash
python ai-media.py -I translate
```

### Modes
1. **Text**: Enter text directly to translate it.
2. **File**: Provide an audio file or text document for translation.

### Supported Models
- **Seamless M4T v2** (Default): Multimodal model (Speech-to-Speech, Speech-to-Text, Text-to-Text). accurate and fast.
- **NLLB** (No Language Left Behind): Specialized Text-to-Text model supporting 200+ languages.
- **ALMA** (Advanced Language Model-based Translator): Fine-tuned LLM optimized for professional-grade translation (12 core languages).

## CLI Usage

### Text / File Translation
Translate text or a file to a target language.

```bash
# Translate text
python ai-media.py --translate -p "Hello world" --target-language fr

# Translate audio file (Speech-to-Text / Speech-to-Speech)
python ai-media.py --translate -ii interview.mp3 --target-language es

# Specify Model
python ai-media.py --translate -p "Tech article..." -tl de --translation-model alma
```

### Subtitle Generation (with Translation)
Generate subtitles for a video and optionally translate them.

```bash
# Generate subtitles (English by default)
python ai-media.py -gs -ii video.mp4

# Generate and translate to French and Spanish
python ai-media.py -gs -ii video.mp4 --translate-to "fr,es"
```

## Supported Languages

Common language codes (ISO 639):
- `en`: English
- `fr`: French
- `es`: Spanish
- `de`: German
- `it`: Italian
- `pt`: Portuguese
- `zh`: Chinese (Mandarin)
- `ja`: Japanese
- `ko`: Korean
- `ru`: Russian
- `hi`: Hindi
- `ar`: Arabic

*Note: NLLB supports over 200 languages. ALMA is optimized for the set above.*
