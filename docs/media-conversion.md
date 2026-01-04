# Media & Document Conversion

Instantly convert images, videos, audio, and documents between formats (no AI).

← [Back to Main README](../README.md)

## Media Conversion Options

Convert between image formats (PNG, JPG, WebP, GIF, TIFF, BMP) using PIL or FFmpeg.

| Option | Description |
| :--- | :--- |
| `-ci, --convert-image` | Convert image format (e.g., gif→png). |
| `-cit, --convert-image-to` | Output format (png, .webp, out.jpg). |
| `--convert-image-engine` | pil (default) or ffmpeg. |

Convert between video formats (MP4, MOV, WebM, AVI, MKV, GIF) using FFmpeg.

| Option | Description |
| :--- | :--- |
| `-cv, --convert-video` | Convert video (mov→mp4). |
| `-cvt, --convert-video-to` | Output format (mp4, .webm, out.avi). |

Convert between audio formats (MP3, WAV, AAC, FLAC, OGG, M4A) using FFmpeg.

| Option | Description |
| :--- | :--- |
| `-ca, --convert-audio` | Convert audio (wav→mp3). |
| `-cat, --convert-audio-to` | Output format (mp3, .flac, out.ogg). |

## Document Conversion Options

Convert between document formats (MD, HTML, PDF, DOCX, RTF, TXT, JSON).

| Option | Description |
| :--- | :--- |
| `-cd, --convert-document` | Input document file (e.g., report.docx). |
| `-cdt, --convert-document-to` | Output format: md, html, pdf, docx, rtf, txt, json. |
| `-om, --ocr-model` | OCR Model: 'qwen-vl' (default, precise) or 'florence' (fast, ~1.5GB RAM). |

## Conversion Matrix

| From → | MD | HTML | PDF | DOCX | RTF | TXT | JSON | Images |
|--------|:--:|:----:|:---:|:----:|:---:|:---:|:----:|:------:|
| **MD** | - | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **HTML** | ✅ | - | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **PDF** | ⚠️ | ⚠️ | - | ⚠️ | ⚠️ | ✅ | ⚠️ | ❌ |
| **DOCX** | ✅ | ✅ | ✅ | - | ✅ | ✅ | ✅ | ❌ |
| **RTF** | ⚠️ | ⚠️ | ⚠️ | ⚠️ | - | ✅ | ⚠️ | ❌ |
| **TXT** | ✅ | ✅ | ✅ | ✅ | ✅ | - | ✅ | ❌ |
| **JSON** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | - | ❌ |
| **Images** *(JPG/PNG/WebP/GIF/TIFF/BMP)* | 📷 | 📷 | 📷 | 📷 | 📷 | 📷 | 📷 | ✅* |

✅ = Full support | ⚠️ = Text extraction only (lost formatting) | 📷 = OCR (Image-to-Text)

> **Note on OCR**: Images (JPG, PNG, WebP, GIF, TIFF, BMP) and scanned PDFs can be converted to text documents using AI-powered Vision-Language Models (VLMs). 
> 
> | Model | Speed | Accuracy | Req. RAM | Best For |
> | :--- | :--- | :--- | :--- | :--- |
> | **Qwen-VL** | Standard (~55s) | **High (SOTA)** | **~30GB** | **Default.** Best for code, exact paths, symbols, emojis. |
> | **Florence-2** | Fast (~25s) | Moderate | ~1.5GB | Quick scans, high-contrast text. |

This happens automatically for scanned PDFs or when explicitly selected for images. Use `-om qwen-vl` if accuracy is critical.


## Examples

### Image Conversion

```bash
python ai-media.py -ci photo.gif -cit png
python ai-media.py -ci input.png -cit output.webp --convert-image-engine ffmpeg
```

### Video Conversion

```bash
python ai-media.py -cv clip.mov -cvt mp4
```

### Audio Conversion

```bash
python ai-media.py -ca song.wav -cat mp3
```

### Document & OCR Conversion

```bash
# Convert DOCX to PDF
python ai-media.py -cd report.docx -cdt pdf

# Extract Text from Image (OCR - Default)
python ai-media.py -cd photo_of_text.jpg -cdt txt

# Extract Text from Image (OCR - High Precision)
python ai-media.py -cd code_snippet.png -cdt txt -om qwen-vl

# Extract Text from Scanned PDF (OCR)
python ai-media.py -cd scanned_doc.pdf -cdt md -om qwen-vl
```

← [Back to Main README](../README.md)
