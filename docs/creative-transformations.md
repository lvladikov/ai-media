# Creative Image Transformations

Go beyond simple generation. This module allows you to edit, modify, and process existing images using AI-powered instructional commands or rigorous computer vision tasks.

### 1. Instructional Editing (InstructPix2Pix)
Edit images using natural language instructions. The AI understands the context of the image and applies your change while preserving the structure.
*   **Examples**: "Make it snow", "Add a red cape to the man", "Turn the sketch into a realistic photo".
*   **Key Concept**: It uses structural guidance (edges/depth) combined with text prompts to "repaint" only what you asked for.

### 2. Background Removal & Silhouettes
Instantly extract subjects from their background.
*   **Background Removal**: Uses a specialized model (RMBG-1.4) to create clean, transparent PNGs.
*   **Silhouettes**: Converts the extracted subject into a solid black shape (useful for vectors, icons, or design assets).

← [Back to Main README](../README.md)

## Options

| Argument | Description |
| :--- | :--- |
| `-ti`, `--transform-image` | Path to the image file to transform. |
| `-p`, `--prompt` | Edit instruction (works for standalone transformations). |
| `-tp`, `--transform-prompt` | Edit instruction for chaining with generation (e.g., `-i -p "..." -ti file -tp "..."`). |
| `--remove-background`, `-rb` | Remove background (outputs transparent PNG). |
| `--silhouette` | Create a black silhouette (requires `--remove-background`). |
| `--image-guidance` | Image guidance scale (default: `1.5`). Higher = closer to original structure. |

> [!NOTE]
> **`-p` vs `-tp`**: For standalone transformations (`-ti` only), use `-p`. When chaining generation (`-i`) with transformation (`-ti`), use `-p` for the **generation prompt** and `-tp` for the **edit instruction**.

See [Creative Transformation Examples](#examples) and [Models](#models).

## Models

| Model | Code | Download | VRAM | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **InstructPix2Pix** | `instruct-pix2pix` | ~4GB | ~8GB (High Precision) | Instructional image editing (e.g., "Make it anime"). |
| **RMBG-1.4** | `remove-bg` | ~0.2GB | ~2GB | Background removal and silhouette creation. |

## Transformation Recipe Book 🪄

Here are prompt examples for common editing tasks.

### Styles

| Goal | Command Pattern |
| :--- | :--- |
| Anime / Manga | `-tp "Turn the subject into an anime character"` |
| Disney / Pixar | `-tp "Make the subject look like a 3D Pixar character"` |
| Studio Ghibli | `-tp "Make it look like a Studio Ghibli movie"` |
| Oil Painting | `-tp "Make it look like an oil painting"` |
| Watercolor | `-tp "Turn this into a watercolor painting"` |
| Pencil Sketch | `-tp "Turn this into a pencil sketch"` |
| Cartoon | `-tp "Turn this into a flat cartoon"` |
| Coloring Page | `-tp "Make it a black and white coloring page"` |
| Sticker | `-tp "Turn this into a sticker with a white outline"` |

### Photo Manipulations

| Goal | Command Pattern |
| :--- | :--- |
| Remove Beard | `-tp "Remove the beard"` |
| Change Hairstyle | `-tp "Give the subject a mohawk hairstyle"` |
| Facial Expressions | `-tp "Make the subject smile"`, `-tp "Make the subject look surprised"` |
| Age / Baby | `-tp "Make the subject look like a baby"` |
| Caricature | `-tp "Turn this into a funny caricature"` |
| Recolor | `-tp "Change the red dress to blue"` |
| Colorize B&W | `-tp "Colorize this photo"` |
| Sketch to Image | `-tp "Turn this sketch into a photo of an apple"` |

### Removal

| Goal | Command Pattern |
| :--- | :--- |
| Background | `--remove-background` (No prompt needed) |
| Silhouette | `--remove-background --silhouette` |
| Text / Objects | `-tp "Remove the text"`, `-tp "Remove the cup"` (Experimental) |

## Examples

## Examples

### Instructional Edits (InstructPix2Pix)

Modify images using natural language instructions.

```bash
# Simple Edit
python ai-media.py -ti photo.jpg -tp "Make it look like an anime drawing"
python ai-media.py --transform-image photo.jpg --transform-prompt "Make it look like an anime drawing"

# Specific Feature Editing
python ai-media.py -ti portrait.jpg -tp "Add a pair of sunglasses"
python ai-media.py -ti room.jpg -tp "Replace the chair with a sofa"
```

### Guidance Scale

Control how much the image deviates from the original structure.
*   **Lower (<1.2)**: More creative, less adherence to original shapes.
*   **Higher (>1.5)**: Stricter, closer to original.

```bash
# Strong Edit (More Creative change)
python ai-media.py -ti sketch.png -tp "Turn into a realistic photo" --image-guidance 1.0

# Subtle Edit (Keep structure)
python ai-media.py -ti face.jpg -tp "Add smile" --image-guidance 1.8
```

### Background Removal

```bash
# Remove Background (Transparent PNG)
python ai-media.py -ti product.jpg -rb
python ai-media.py --transform-image product.jpg --remove-background

# Create Silhouette
python ai-media.py -ti icon.png -rb --silhouette
```

### Operation Chaining

Combine Generation, Transformation, and Utility in one go.

```bash
# 1. Edit -> Then Remove Background
python ai-media.py -ti dog.jpg -tp "Make the dog wear a hat" -rb

# 2. Generate -> Then Edit -> Then Remove Background
# -p: Generation Prompt
# -tp: Edit Prompt
python ai-media.py -i -p "A knight" -ti -tp "Add a red cape" -rb
```

← [Back to Main README](../README.md)
