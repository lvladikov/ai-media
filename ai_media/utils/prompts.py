"""
Random prompts utility for AI-Media.

Provides curated prompts for different generation types (image, video, audio, article, code).
Used by CLI, interactive menus, and inference server.

Single source of truth: ai_media/web/src/data/prompts.json
"""

import json
import random
import os

# Trigger patterns for random prompt (case-insensitive match)
RANDOM_PROMPT_TRIGGERS = ["rndpr", "rndprompt", "randomprompt", "random prompt"]

# Path to shared prompts JSON (relative to this file)
_PROMPTS_JSON_PATH = os.path.join(
    os.path.dirname(__file__), 
    "..", "data", "prompts.json"
)

# Lazy-loaded prompts cache
_prompts_cache = None


def _load_prompts():
    """Load prompts from shared JSON file."""
    global _prompts_cache
    if _prompts_cache is None:
        try:
            with open(_PROMPTS_JSON_PATH, "r", encoding="utf-8") as f:
                _prompts_cache = json.load(f)
        except FileNotFoundError:
            # Fallback minimal prompts if JSON not found
            print(f"Warning: prompts.json not found at {_PROMPTS_JSON_PATH}, using fallback prompts")
            _prompts_cache = {
                "image": ["A beautiful sunset over mountains"],
                "video": ["A time-lapse of clouds moving"],
                "audio": ["Relaxing ambient music"],
                "article": ["The Future of AI"],
                "code": ["Create a simple to-do app"]
            }
    return _prompts_cache


def is_random_prompt_trigger(text: str) -> bool:
    """Check if the text matches any random prompt trigger."""
    clean = text.strip().lower()
    return clean in RANDOM_PROMPT_TRIGGERS


def get_random_prompt(prompt_type: str = "image") -> str:
    """Get a random prompt for the specified type."""
    prompts = _load_prompts()
    prompt_list = prompts.get(prompt_type, prompts.get("image", ["A beautiful landscape"]))
    return random.choice(prompt_list)


def maybe_replace_with_random(prompt: str, prompt_type: str = "image") -> tuple[str, bool]:
    """
    Check if prompt is a random trigger and replace if so.

    Returns:
        (final_prompt, was_replaced)
    """
    if is_random_prompt_trigger(prompt):
        return get_random_prompt(prompt_type), True
    return prompt, False
