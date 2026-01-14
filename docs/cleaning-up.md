# Cleanup Options

AI-Media caches downloaded models and generates output files that can consume significant disk space. This guide covers all cleanup options available.

## Interactive Menu

Access cleanup options via the interactive menu:

```bash
python ai-media.py -I
# Navigate to: Cleanup / Maintenance
```

### Available Options

| Option | Description |
|--------|-------------|
| Clear testing/data/outputs | Removes test output files |
| Clear media_output | Removes generated media files |
| Clear All Output Data | Clears both of the above |
| Clear Hub Model | Browse and delete cached HuggingFace models |

---

## CLI Commands

### Clear Test Outputs

```bash
python ai-media.py --clear-data-output
```

Clears the `ai_media/testing/data/outputs/` folder. Safe to run anytime.

### Clear Media Output

```bash
python ai-media.py --clear-media-output
```

Clears the configured `media_output` folder (default: `media-output/`).

> [!WARNING]
> This will delete all your generated images, videos, and audio files permanently.

### Clear All Outputs

```bash
python ai-media.py --clear-all-outputs
```

Combines both `--clear-data-output` and `--clear-media-output`.

### Clear Hub Model

```bash
python ai-media.py --clear-hub-model "models--org--model-name"
```

Deletes a specific model from the HuggingFace hub cache.

**Example:**
```bash
# Delete a specific model
python ai-media.py --clear-hub-model "models--black-forest-labs--FLUX.1-schnell"
```

> [!CAUTION]
> Deleted models will need to be re-downloaded (2-30GB each) the next time they are used.

---

## Hub Model Management

The **Clear Hub Model** interactive menu provides:

1. **Loading indicator** while scanning the hub folder
2. **Size information** for each cached model (sorted largest first)
3. **Total size** summary panel
4. **Confirmation prompt** before deletion

### Finding Large Models

Use the interactive menu to identify which models are consuming the most space. Models are sorted by size with the largest at the top.

### Hub Path Configuration

The hub cache location is configured in `config.json`:

```json
{
  "paths": {
    "hf_home": "/path/to/huggingface"
  }
}
```

The actual model cache is stored in `{hf_home}/hub/`.

---

## Tips

- **Test outputs** can be safely cleared without affecting your work
- **Media outputs** contain your generated content - back up what you need first
- **Hub models** are re-downloadable but may take time depending on model size and connection speed
- Hidden folders like `.locks` are automatically skipped during hub cleanup
