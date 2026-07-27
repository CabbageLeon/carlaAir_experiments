"""Prompt templates specified by Table 10 of experiment.md."""

from __future__ import annotations


LANDING_INSTRUCTION = (
    "Follow the moving truck, keep it in view, align above its rear cargo bed, "
    "and land safely on the cargo bed."
)
ESCORT_INSTRUCTION = "Follow the moving truck and keep it in view."


def landing_prompt(mode: str, direction: str, motion: str, phase: str) -> str:
    """Return the C0/C1/C2 landing prompt without adding unapproved information."""
    if mode == "C0":
        return LANDING_INSTRUCTION
    if mode not in {"C1", "C2"}:
        raise ValueError(f"unsupported landing mode: {mode}")
    return (
        f"{LANDING_INSTRUCTION}\n\n"
        f"Assistant hint: the cargo bed is {direction}. The truck is {motion}. "
        f"Current phase: {phase}. Use the hint only to choose your next UAV action."
    )


def escort_prompt(mode: str, occluded: bool, direction: str, motion: str, phase: str) -> str:
    """Return the C0/C1 escort prompt without exposing metric-state coordinates."""
    if mode == "C0":
        return ESCORT_INSTRUCTION
    if mode != "C1":
        raise ValueError(f"unsupported escort mode: {mode}")
    if occluded:
        state = f"temporarily occluded. The truck {motion} and will reappear on the {direction} side"
    else:
        state = f"visible. The truck {motion}"
    return (
        f"{ESCORT_INSTRUCTION}\n\n"
        f"Assistant hint: the truck is {state}. Current phase: {phase}. "
        "Use the hint only to recover visual contact."
    )
