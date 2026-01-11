"""
Configuration routes for AI-Media Server.
Exposes backend configuration and constants to the frontend.
"""

from fastapi import APIRouter
from ai_media.utils.prompts import RANDOM_PROMPT_TRIGGERS

router = APIRouter()

@router.get("/api/config/prompts")
async def get_prompts_config():
    """
    Get configuration for prompts, specifically random prompt triggers.
    Used by frontend to know which keywords trigger random prompt substitution.
    """
    return {
        "triggers": RANDOM_PROMPT_TRIGGERS
    }
