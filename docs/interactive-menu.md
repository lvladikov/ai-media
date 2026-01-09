# Interactive Mode

The interactive mode offers a guided menu system for all features. It runs automatically if no arguments are provided, or explicitly via `--interactive`.

## Options

| Option              | Description                                                                                                |
| :------------------ | :--------------------------------------------------------------------------------------------------------- |
| `-I, --interactive` | Launch the interactive menu. Runs automatically if no arguments are provided.                              |
| `-I <jump>`         | Jump directly to a submenu (e.g., `image/sdxl`, `chat`, `5/2`). See [Fast Jump Points](#fast-jump-points). |

← [Back to Main README](../README.md)

## Mouse Support 🖱️ (macOS/Linux only)

The interactive menu supports full mouse interaction on **macOS and Linux** terminals (iTerm2, Terminal.app, GNOME Terminal, etc.):

- **Click to Select**: Click any menu item to select it instantly.
- **Scroll Wheel**: Scroll up/down through long lists.
- **Navigation**: Click "⬆️ ... more above" or "⬇️ ... more below" to jump pages.
- **Click Back**: Click the "⬅️ Back" button to return.

> [!NOTE] > **Windows**: Mouse support is not available on Windows due to terminal limitations (`msvcrt` only captures keyboard input). Use keyboard navigation instead.

## Keyboard Navigation ⌨️

- **Arrow Keys**: Navigate Up/Down.
- **PageUp/PageDown/Home/End**: Fast navigation.
- **vim keys**: Use `g` (Top) and `G` (Bottom) if your terminal intercepts Home/End.
- **0**: Quick Back/Exit.

## Usage

```bash
# Run interactive menu
python ai-media.py
# OR
python ai-media.py --interactive
# OR
python ai-media.py -I
```

![Interactive Menu](../screenshots/interactive-menu.png)

![Video Generation Menu](../screenshots/interactive-menu-video-gen.png)

## Random Prompts 🎲

> [!TIP] > **Need inspiration?** When prompted for a text prompt (image, article, code, etc.), type one of these magic phrases to get a random creative prompt:
>
> - `rndPr`
> - `rndPrompt`
> - `randomPrompt`
> - `random prompt`
>
> Example: For image generation, type `rndPr` and the tool will automatically use a random prompt like _"A serene Japanese garden with cherry blossoms falling into a koi pond"_.

## Fast Jump Points

You can jump directly to specific submenus or models using shortcut paths with `--interactive`:

| Menu # | Task            | Jump Point             | Description              |
| :----- | :-------------- | :--------------------- | :----------------------- |
| `1`    | **Image**       | `image`                | Image Menu               |
| `1/1`  |                 | `image/sdxl`           | SDXL Turbo (Fast)        |
| `1/2`  |                 | `image/sd15`           | SD 1.5 (Regular)         |
| `1/3`  |                 | `image/flux`           | Flux Schnell             |
| `1/4`  |                 | `image/flux-dev`       | Flux Dev                 |
| `2`    | **Video**       | `video`                | Video Menu               |
| `2/1`  |                 | `video/zeroscope`      | Zeroscope (No Watermark) |
| `2/2`  |                 | `video/modelscope`     | ModelScope (General)     |
| `2/3`  |                 | `video/cogvideox`      | CogVideoX                |
| `2/4`  |                 | `video/svd`            | Stable Video Diffusion   |
| `3`    | **Audio**       | `audio`                | Audio Menu               |
| `3/1`  |                 | `audio/musicgen`       | MusicGen Medium          |
| `3/2`  |                 | `audio/musicgen-small` | MusicGen Small (Fast)    |
| `3/3`  |                 | `audio/musicgen-large` | MusicGen Large (Quality) |
| `3/4`  |                 | `audio/audioldm2`      | AudioLDM2 (SFX)          |
| `3/5`  |                 | `audio/bark`           | Bark (TTS)               |
| `4`    | **Analysis**    | `analysis`             | Analysis Menu            |
| `5`    | **Article**     | `article`              | Article/Research Menu    |
| `5/1`  |                 | `article/offline`      | Offline Article          |
| `5/2`  |                 | `article/online`       | Online Research          |
| `6`    | **Code**        | `code`                 | Generate Code            |
| `7`    | **Chat**        | `chat`                 | Interactive Chat         |
| `8`    | **Edit**        | `transform`            | Transform Menu           |
| `8/1`  |                 | `transform/edit`       | Creative Edit            |
| `8/2`  |                 | `transform/rembg`      | Background Removal       |
| `8/3`  |                 | `transform/silhouette` | Silhouette               |
| `9`    | **Convert**     | `convert`              | Convert Menu             |
| `10`   | **Doc Convert** | `doc_convert`          | Convert Document         |
| `11`   | **Translate**   | `translate`            | Translate Menu           |
| `12`   | **Upscale**     | `upscale`              | Upscale Menu             |
| `13`   | **Test**        | `test`                 | Run Tests                |
| `13/1` |                 | `test/unit`            | Unit Tests               |
| `13/2` |                 | `test/integration`     | Integration Tests        |
| `13/3` |                 | `test/codec`           | Codec Limits Test        |
| `14`   | **Sysinfo**     | `sysinfo`              | System Information       |
| `15`   | **Web Server**  | `web`                  | Web Server Mode Menu     |
| `16`   | **Cleanup**     | `cleanup`              | Cleanup                  |

```bash
python ai-media.py --interactive "image/sdxl"
python ai-media.py --interactive "audio/bark"
python ai-media.py --interactive code
python ai-media.py --interactive chat
python ai-media.py --interactive "5/2"
python ai-media.py --interactive 12
python ai-media.py --interactive 13
```

← [Back to Main README](../README.md)
