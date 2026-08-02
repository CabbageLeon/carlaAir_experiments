"""Prompt templates read from config.json — each model can have its own wording."""

from __future__ import annotations

from . import load_config

# LLaMA-2 chat template wrapper (matches training distribution)
_SYS_PROMPT = (
    "You are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language."
)

_CHAT_TEMPLATE = (
    "[INST] <<SYS>>\n{system}\n<</SYS>>\n\n"
    "What action should the robot take to {instruction}? [/INST]"
)


def _get_instruction(key: str) -> str:
    cfg = load_config()
    return cfg["prompts"][key]


def _wrap_chat(policy: str, instruction: str) -> str:
    """Wrap instruction in LLaMA-2 chat template (only for LLM-based policies like OpenFly).
    SPF uses raw text through its own _prompt() wrapper in spf_policy.py."""
    if policy == "openfly":
        return _CHAT_TEMPLATE.format(system=_SYS_PROMPT, instruction=instruction.lower())
    return instruction


def landing_prompt(policy: str, mode: str, direction: str, motion: str, phase: str) -> str:
    key = f"landing_{mode.lower()}"
    template = _get_instruction(key)
    instruction = template.format(direction=direction, motion=motion, phase=phase)
    return _wrap_chat(policy, instruction)


def escort_prompt(policy: str, mode: str, occluded: bool, direction: str, motion: str, phase: str) -> str:
    key = f"escort_{mode.lower()}"
    template = _get_instruction(key)
    if "{state}" in template:
        if occluded:
            state = f"temporarily occluded. The truck {motion} and will reappear on the {direction} side"
        else:
            state = f"visible. The truck {motion}"
        instruction = template.format(state=state, phase=phase)
    else:
        instruction = template
    return _wrap_chat(policy, instruction)
