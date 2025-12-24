# Audio Generation

Compose music, create sound effects, and generate realistic speech entirely offline. This module integrates specialized models for different audio needs:

### 1. Music Composition (MusicGen)
Generate high-fidelity music tracks from text descriptions.
*   **Usage**: "Lo-fi hip hop beat for studying", "Epic orchestral trailer music".
*   **Capabilities**: Can generate infinite loops or specific durations.

### 2. Sound Effects & Foley (AudioLDM 2)
Create realistic environmental sounds, foley, and soundscapes.
*   **Usage**: "Heavy rain on a tin roof", "Footsteps on gravel", "Laser gun sounds".
*   **Quality**: Supports high sample rates (48kHz) and bit depths (24-bit) for professional use.

### 3. Speech & Acting (Bark)
A text-to-audio model that can speak in multiple languages with emotion, laughter, and non-speech sounds.
*   **Usage**: "Hello world [laughs]", podcasts, storytelling, or character voices.
*   **Unique Feature**: Unlike standard TTS, Bark understands mood tags like `[sighs]`, `[music]`, and `[gasps]`.

### 4. Visual-to-Audio (Media scoring)
Automatically generate a soundtrack for an image or video.
*   **How it works**: The AI analyzes the visual content (e.g., sees a "Beach"), and automatically generates matching audio (e.g., "Ocean waves and seagulls").

← [Back to Main README](../README.md)

## Options

| Option | Description |
| :--- | :--- |
| `--audio-model` | Model: `musicgen-medium` (default), `musicgen-small`, `musicgen-large`, `audioldm2`, `stable-audio`, `bark`. See [Models](#models). |
| `-l, --length` | Duration. Supports "15s", "1m", "1h30m", `{m:1, s:30}`. Default: 15s. |
| `-ii, --input-image` | Source image/video for **Image-to-Audio** or **Video-to-Audio** (auto-captions then generates audio). |
| `-m, --sampling-rate` | Sample rate in Hz (e.g. `44100`, `48k`, `32000`). Default: 32000. |
| `-b, --bit-depth` | Bit depth (16, 24, 32). Default: 16. |
| `-r, --bit-rate` | Target bitrate (e.g. `128k`, `320kbps`). |
| `--voice-preset` | Bark voice preset (e.g. `v2/en_speaker_6`, `v2/fr_speaker_1`). Default: v2/en_speaker_6. |
| `-p, --prompt` | Text description of content to generate. |
| `-o, --output` | Output filename/path. Default: mp3. |

See [Audio Generation Examples](#examples) and [Models](#models).

### Supported Durations (`-l` or `--length`)
- **Strings**: `15s`, `1m`, `1h30m5s`
- **Objects**: `{m: 1, s: 30}`, `{hours: 1, minutes: 15}`
- **Numeric**: `30` (interpreted as seconds)

## Models

| Model | Code | Download | VRAM | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **MusicGen Small** | `musicgen-small` | ~2GB | ~4GB | Fast, lightweight. Good for quick sketches. |
| **MusicGen Medium** | `musicgen-medium` | ~6GB | ~8GB | **(Default)** Balanced quality/speed. |
| **MusicGen Large** | `musicgen-large` | ~10GB | ~16GB | High fidelity. Slower. |
| **AudioLDM2** | `audioldm2` | ~4GB | ~6GB | Specialized in Sound Effects (SFX), foley, environmental. |
| **Stable Audio** | `stable-audio` | ~10GB | ~10GB | 🔒 **Gated**. Variable-length, high-quality music/SFX. Top-tier. |
| **Bark** | `bark` | ~4GB | ~6GB | Realistic speech, music, sound effects. Multi-language TTS. |

## Bark Configuration

Bark is a transformer-based model that can generate highly realistic speech as well as other audio (music, background noise, etc.).

### Special Tokens / Sound Effects

To generate non-speech audio, use these tags in your prompt:
*   `[laughter]`, `[cheers]`, `[music]`, `[sighs]`, `[gasps]`, `[clears throat]`
*   `—` or `...` (hesitations)
*   `♪` (wrap lyrics for singing, e.g. `♪ Hello World ♪`)

> [!TIP]
> **Token Reliability**: These sound effects are probabilistic and may not work with every voice or seed.
> *   **Try different voices**: Some speakers "laugh" better than others.
> *   **Context matters**: A prompt like *"That was funny! [laughter]"* works better than just `[laughter]`.
> *   **Singing**: Lyrics wrapping `♪` works best with short, rhythmic lines.

### Voice Presets (`--voice-preset`)

You can change the speaker using the `--voice-preset` flag (default: `v2/en_speaker_6`).
*   **Format**: `v2/{lang}_speaker_{0-9}`
*   **Languages**: `en` (English), `fr` (French), `de` (German), `es` (Spanish), `it` (Italian), `ja` (Japanese), `zh` (Chinese), `pt` (Portuguese), `ru` (Russian).
*   **Reference**: [Bark Speaker Library (Audio Samples)](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc8cd1ed101043facc93a945395850fb)

> **Example**: `python ai-media.py -a -am bark -p "♪ In the jungle ♪ [laughter]" --voice-preset v2/it_speaker_2`

### Auto-Chunking & Unlimited Length ♾️

By default, the Bark model can only generate ~14 seconds of audio per pass. This script includes an **automatic long-form generation** feature.
*   **Triggers**: This mode activates automatically if your text is long (>150 characters) or if you explicitly set a long duration (e.g. `--length 20s`).
*   **Audio Length**: The final audio length depends **entirely on your text**. (The `--length` flag effectively serves as a "force split" switch for Bark).
*   **Process**: The script splits your text into sentences and generates them in independent, stable chunks to ensure voice consistency.
*   **Usage**: Just provide a long text prompt.
    *   `python ai-media.py -a -am bark -p "This is a very long story..."`
    *   `python ai-media.py -a -am bark --voice-preset v2/en_speaker_6 -p "This is the first sentence. And this is the second one! Now we can go on forever without the model cutting us off. Like I am continuing here for a long long time [laughter]. Oh no [gasp], why did I do that!"`

## Examples

```bash
## Examples

### Basic Usage (Text-to-Music)

```bash
# 15s Music Clip (Default: MusicGen Medium)
python ai-media.py -a -p "Lo-fi hip hop beat"
python ai-media.py --generate-audio --prompt "Lo-fi hip hop beat"

# 1 Minute Clip
python ai-media.py -a -p "Piano concerto" -l 1m
python ai-media.py -a -p "Piano concerto" --length 1m
```

### Sound Effects & Ambience (AudioLDM 2)

AudioLDM 2 is often better for environmental sounds than MusicGen.

```bash
# High-Quality Rain Sounds (48kHz, 24-bit)
python ai-media.py -a -p "Heavy rain on roof, thunder" -am audioldm2 -m 48000 -b 24
python ai-media.py -a -p "Heavy rain" --audio-model audioldm2 --sampling-rate 48000 --bit-depth 24
```

### Speech Generation (Bark)

Generate realistic speech with emotion and non-speech sounds.

```bash
# Simple Speech
python ai-media.py -a -am bark -p "Hello, this is a test of the emergency broadcast system."

# Multilingual & Emotional (using Voice Presets)
# Use tags like [laughter], [sighs], [music]
python ai-media.py -a -am bark --voice-preset v2/fr_speaker_1 -p "C'est magnifique! [laughter] I love it."

# Long-Form Speech (Auto-Chunking)
# Bark naturally fails after ~14s. The script auto-splits long prompts to fix this.
python ai-media.py -a -am bark -p "Here is a very long story that would normally crash the model. The script splits it into sentences. It generates each sentence individually. Then it stitches them back together seamlessly."
```

### Image/Video-to-Audio

AI "looks" at your media, captions it, then generates matching audio.

```bash
# Soundtrack for an image
python ai-media.py -a -ii "./beach.jpg" -o beach_vibes.mp3
python ai-media.py --generate-audio --input-image "./beach.jpg"

# Soundtrack for a video clip
python ai-media.py -a -ii "./clip.mp4" -l 10s
```

← [Back to Main README](../README.md)
