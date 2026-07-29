"""API provider configuration for SPF evaluation.

Presets for common VLM providers.  Override any field with an environment variable.

Environment variables (all optional, see presets for defaults):
  OPENAI_API_KEY   – API key (required; set outside version control)
  OPENAI_BASE_URL  – OpenAI-compatible base URL
  OPENAI_MODEL     – model name
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# ── Preset providers ──────────────────────────────────────────
PROVIDERS: dict[str, dict[str, str]] = {
    "bailian": {
        "base_url": "https://ws-9uve2kqdj44bdrwr.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3-vl-flash",
        "description": "阿里云百炼 MaaS 工作空间",
    },
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-vl-max",
        "description": "阿里云 DashScope 灵积平台",
    },
    "qianwenai": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-vl-max",
        "description": "千问 AI 开放平台",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "description": "OpenAI 官方",
    },
    "custom": {
        "base_url": "",
        "model": "",
        "description": "手动通过 OPENAI_BASE_URL / OPENAI_MODEL 指定",
    },
}

# ── Tunable defaults ──────────────────────────────────────────
# VLM inference
DEFAULT_TEMPERATURE: float = 0.2
DEFAULT_MAX_TOKENS: int = 4096
DEFAULT_PROJECTION_DISTANCE_MAX_M: float = 28.0
DEFAULT_HORIZONTAL_FOV_DEG: float = 108.0

# Controller
DEFAULT_HORIZONTAL_GAIN: float = 5.0
DEFAULT_VERTICAL_GAIN: float = 0.8

# Scenario
DEFAULT_TRUCK_SPEED: float = 4.0


@dataclass
class SPFConfig:
    """Resolved configuration for one SPF experiment."""

    # ── API ──
    model: str
    base_url: str
    api_key: str
    provider: str = ""

    # ── VLM inference ──
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    horizontal_fov_deg: float = DEFAULT_HORIZONTAL_FOV_DEG
    projection_distance_max_m: float = DEFAULT_PROJECTION_DISTANCE_MAX_M

    # ── Controller ──
    horizontal_gain: float = DEFAULT_HORIZONTAL_GAIN
    vertical_gain: float = DEFAULT_VERTICAL_GAIN

    # ── Scenario ──
    truck_speed: float = DEFAULT_TRUCK_SPEED

    extra_headers: dict[str, str] = field(default_factory=dict)


def get_config(model_override: str | None = None) -> SPFConfig:
    """Resolve SPF configuration from environment variables and presets.

    Resolution order (later wins):
      1. Provider preset (SPF_PROVIDER, default: "qianwenai")
      2. Explicit env vars: OPENAI_BASE_URL, OPENAI_MODEL
      3. Function argument ``model_override``

    OPENAI_API_KEY is always required from the environment.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required. Set it in your environment:\n"
            "  export OPENAI_API_KEY='your-key'"
        )

    provider_name = os.environ.get("SPF_PROVIDER", "qianwenai").lower()
    if provider_name not in PROVIDERS:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(f"Unknown SPF_PROVIDER '{provider_name}'. Available: {available}")
    preset = PROVIDERS[provider_name]

    base_url = os.environ.get("OPENAI_BASE_URL") or preset["base_url"]
    model = model_override or os.environ.get("OPENAI_MODEL") or preset["model"]

    if not base_url:
        raise RuntimeError(
            "No base_url configured. Set OPENAI_BASE_URL or choose a provider preset "
            f"(SPF_PROVIDER={provider_name} has no default base_url)."
        )
    if not model:
        raise RuntimeError(
            "No model configured. Set OPENAI_MODEL or choose a provider preset "
            f"(SPF_PROVIDER={provider_name} has no default model)."
        )

    return SPFConfig(
        model=model,
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        provider=provider_name,
    )


def print_config() -> None:
    """Print the resolved configuration (useful for debugging)."""
    try:
        cfg = get_config()
    except RuntimeError as e:
        print(f"Configuration error: {e}")
        return
    _mask = lambda k: k if len(k) <= 8 else f"{k[:6]}...{k[-4:]}"
    print(f"Provider:     {cfg.provider} ({PROVIDERS.get(cfg.provider, {}).get('description', '')})")
    print(f"Base URL:     {cfg.base_url}")
    print(f"Model:        {cfg.model}")
    print(f"API Key:      {_mask(cfg.api_key)}" if len(cfg.api_key) > 20 else f"API Key:      {cfg.api_key}")
    print(f"Temperature:  {cfg.temperature}")
    print(f"Max tokens:   {cfg.max_tokens}")
    print(f"FOV:          {cfg.horizontal_fov_deg}°")
    print(f"Proj. max:    {cfg.projection_distance_max_m} m")
    print(f"H. gain:      {cfg.horizontal_gain}")
    print(f"V. gain:      {cfg.vertical_gain}")
    print(f"Truck speed:  {cfg.truck_speed} m/s")


if __name__ == "__main__":
    print_config()
