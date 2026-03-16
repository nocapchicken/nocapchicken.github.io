# AI-assisted (Claude Code, claude.ai) — https://claude.ai
"""Shared constants between build_features.py and app/inference.py.

Single source of truth for text-derived feature patterns, ensuring
training-time and inference-time feature construction stay in sync.
"""

# Safety-adjacent keywords that may correlate with inspection outcomes
SAFETY_PATTERN = r"\b(dirty|filthy|sick|roach|bug|rat|mouse|health|violation|gross|smell|mold|expired|undercooked|raw|contaminated)\b"

# Strongly negative phrases (sentiment proxy)
NEGATIVE_PATTERN = r"\b(worst|terrible|horrible|disgusting|awful|never again|food poisoning|threw up|diarrhea)\b"
