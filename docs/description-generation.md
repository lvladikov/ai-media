# Description Generation

Use Vision-Language Models (VLMs) to give eyes to your AI. This module analyzes images and videos and returns text descriptions of what it sees.

### 1. Image Captioning
Generate a detailed text description of any image file.
*   **Uses**: Auto-generating alt-text, organizing photo libraries, or getting prompts for "remixing" an image.
*   **Models**: **Florence-2** (Detailed, spatial awareness) or **BLIP** (Short, concise captions).

### 2. Video Analysis
Summarize the content of a video file.
*   **How it works**: The script intelligently samples 10 evenly-spaced frames from the video, analyzes each one, and synthesizes a summary of the action, setting, and characters.
*   **Uses**: summarizing meeting recordings, analyzing security footage, or generating metadata for video archives.

← [Back to Main README](../README.md)

## Options

| Option | Description |
| :--- | :--- |
| `-gd, --generate-description` | Generate caption/description for input image/video. For videos, 10 evenly-spaced frames are sampled and described. |
| `-cm, --caption-model` | Model: `florence` (default), `blip`. See [Models](#models). |
| `-ii, --input-image` | Path to the image or video file to describe. |
| `-o, --output` | Output text filename (optional). |

See [Description Generation Examples](#examples) and [Models](#models).

## Models

| Model | Code | Download | VRAM | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **Florence-2 Large** | `florence` | ~1.5GB | ~3GB | **(Default)** SOTA details, rich descriptions, "seeing" the scene. |
| **BLIP Large** | `blip` | ~1GB | ~2GB | Simple, concise captions. Faster but less detailed. |

## Examples

## Examples

### Basic Usage

```bash
# Describe an Image
python ai-media.py -gd -ii photo.jpg
python ai-media.py --generate-description --input-image photo.jpg

# Describe a Video (Samples 10 frames)
# Generates a summary based on analyzing frames throughout the video.
python ai-media.py -gd -ii clip.mp4
```

### Model Selection

```bash
# Florence-2 (Default) - Rich Detail
python ai-media.py -gd -ii scene.jpg -cm florence

# BLIP (Legacy) - Simple Captions
# Faster, less VRAM, but descriptions are very short ("A dog sitting on a bench").
python ai-media.py -gd -ii scene.jpg -cm blip
python ai-media.py -gd -ii scene.jpg --caption-model blip
```

> [!TIP]
> If you are interested in producing a subtitle file based on Audio or Video using AI, see the [auto-subtitles project](https://github.com/lvladikov/auto-subtitles).

← [Back to Main README](../README.md)
