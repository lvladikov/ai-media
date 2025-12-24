# Media & Document Conversion

Instantly convert images, videos, audio, and documents between formats (no AI).

← [Back to Main README](../README.md)

## Media Conversion Options

| Option | Description |
| :--- | :--- |
| `-ci, --convert-image` | Convert image format (e.g., gif→png). |
| `-cit, --convert-image-to` | Output format (png, .webp, out.jpg). |
| `-cv, --convert-video` | Convert video (mov→mp4). |
| `-cvt, --convert-video-to` | Output format (mp4, .webm, out.avi). |
| `-ca, --convert-audio` | Convert audio (wav→mp3). |
| `-cat, --convert-audio-to` | Output format (mp3, .flac, out.ogg). |
| `--convert-image-engine` | pil (default) or ffmpeg. |

## Document Conversion Options

Convert between document formats (MD, HTML, PDF, DOCX, RTF, TXT, JSON).

| Option | Description |
| :--- | :--- |
| `-cd, --convert-document` | Input document file (e.g., report.docx). |
| `-cdt, --convert-document-to` | Output format: md, html, pdf, docx, rtf, txt, json. |

## Conversion Matrix

| From → | MD | HTML | PDF | DOCX | RTF | TXT | JSON |
|--------|:--:|:----:|:---:|:----:|:---:|:---:|:----:|
| **MD** | - | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **HTML** | ✅ | - | ✅ | ✅ | ✅ | ✅ | ✅ |
| **PDF** | ⚠️ | ⚠️ | - | ⚠️ | ⚠️ | ✅ | ⚠️ |
| **DOCX** | ✅ | ✅ | ✅ | - | ✅ | ✅ | ✅ |
| **RTF** | ⚠️ | ⚠️ | ⚠️ | ⚠️ | - | ✅ | ⚠️ |
| **TXT** | ✅ | ✅ | ✅ | ✅ | ✅ | - | ✅ |
| **JSON** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | - |

✅ = Full support | ⚠️ = Text extraction only (formatting/images may be lost)

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

← [Back to Main README](../README.md)
