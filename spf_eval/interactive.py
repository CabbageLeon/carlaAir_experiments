"""Interactive provider/model/key selection for SPF evaluation.

Called by runner.py before launching an experiment so the user does not
need to manage environment variables manually.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .config import (
    PROVIDERS,
    SPFConfig,
    DEFAULT_HORIZONTAL_GAIN,
    DEFAULT_VERTICAL_GAIN,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROJECTION_DISTANCE_MAX_M,
    DEFAULT_TEMPERATURE,
    DEFAULT_TRUCK_SPEED,
    get_config,
)


STATE_DIR = Path.home() / ".carlaair"
STATE_FILE = STATE_DIR / "state.json"


def _load_state() -> dict[str, str]:
    """Read saved state (last provider, model, masked key)."""
    if STATE_FILE.is_file():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_state(provider: str, model: str) -> None:
    """Persist provider & model choices for next run (never persist the raw key)."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = _load_state()
    state["provider"] = provider
    state["model"] = model
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _mask(key: str) -> str:
    return key if len(key) <= 8 else f"{key[:6]}...{key[-4:]}"


def _read_key(prompt: str, default: str | None) -> str:
    """Read a single line; empty input falls back to *default*."""
    try:
        value = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(0)
    return value if value else (default or "")


# ── Model variants per provider ──────────────────────────────
# Each tuple is (model_id, label).  The first entry is the default.
MODEL_VARIANTS: dict[str, list[tuple[str, str]]] = {
    "bailian": [
        ("qwen3-vl-flash", "qwen3-vl-flash  (快速)"),
        ("qwen3-vl-plus", "qwen3-vl-plus   (均衡)"),
        ("qwen3-vl-max", "qwen3-vl-max     (最强)"),
        ("qwen-vl-max", "qwen-vl-max       (上一代)"),
    ],
    "dashscope": [
        ("qwen-vl-max", "qwen-vl-max       (旗舰)"),
        ("qwen-vl-plus", "qwen-vl-plus      (均衡)"),
        ("qwen3-vl-flash", "qwen3-vl-flash  (快速)"),
        ("qwen3-vl-max", "qwen3-vl-max     (最新旗舰)"),
    ],
    "qianwenai": [
        ("qwen3-vl-flash", "qwen3-vl-flash   (快速，推荐)"),
        ("qwen3-vl-plus", "qwen3-vl-plus    (均衡)"),
        ("qwen-vl-max", "qwen-vl-max      (旗舰)"),
        ("qwen-vl-plus", "qwen-vl-plus     (上一代均衡)"),
        ("qwen-vl-ocr", "qwen-vl-ocr      (OCR专用)"),
    ],
    "openai": [
        ("gpt-4o", "gpt-4o            (旗舰)"),
        ("gpt-4o-mini", "gpt-4o-mini       (轻量)"),
        ("gpt-4.1", "gpt-4.1           (最新)"),
    ],
}


def interactive_config(model_arg: str | None = None) -> SPFConfig:
    """Run the interactive selection flow.  Returns a ready-to-use SPFConfig.

    If the user has already exported OPENAI_API_KEY + SPF_PROVIDER in the
    environment *and* passed a model on the CLI, we skip the prompts entirely
    (non-interactive fallback).
    """

    # ── Fast path: API key set AND provider explicitly configured ──
    if os.environ.get("OPENAI_API_KEY") and os.environ.get("SPF_PROVIDER"):
        return get_config(model_override=model_arg)

    state = _load_state()
    saved_provider = state.get("provider", "qianwenai")

    # ── Step 1: Select provider ───────────────────────────────
    provider_names = list(PROVIDERS.keys())
    print()
    print("=" * 56)
    print("  SPF 评估 — 选择模型提供商")
    print("=" * 56)
    for i, name in enumerate(provider_names, 1):
        desc = PROVIDERS[name]["description"]
        default_mark = " (*)" if name == saved_provider else ""
        print(f"  [{i}] {name:<14} {desc}{default_mark}")
    print()

    choice = _read_key(
        f"选择提供商 [1-{len(provider_names)}，默认 {saved_provider}]: ",
        "",
    )
    if choice.strip():
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(provider_names):
                saved_provider = provider_names[idx]
            else:
                print(f"无效选项，使用默认: {saved_provider}")
        except ValueError:
            print(f"无效输入，使用默认: {saved_provider}")
    print(f"  -> {PROVIDERS[saved_provider]['description']}")
    print()

    # ── Step 2: Select model ──────────────────────────────────
    variants = MODEL_VARIANTS.get(saved_provider, [])
    saved_model = state.get("model") or PROVIDERS[saved_provider].get("model", "")
    if variants:
        print("─" * 56)
        print("  选择模型版本")
        print("─" * 56)
        for i, (mid, label) in enumerate(variants, 1):
            default_mark = " (*)" if mid == saved_model else ""
            print(f"  [{i}] {label}{default_mark}")
        print(f"  [0] 自定义模型名")
        print()

        choice = _read_key(
            f"选择模型 [0-{len(variants)}，默认 {saved_model or variants[0][0]}]: ",
            "",
        )
        if choice.strip():
            try:
                idx = int(choice)
                if 1 <= idx <= len(variants):
                    model = variants[idx - 1][0]
                elif idx == 0:
                    model = _read_key("输入自定义模型名: ", saved_model or "")
                else:
                    model = saved_model or variants[0][0]
            except ValueError:
                model = saved_model or variants[0][0]
        else:
            model = saved_model or variants[0][0]
    else:
        model = model_arg or saved_model or PROVIDERS[saved_provider].get("model", "")
        if not model:
            model = _read_key("输入模型名: ", "")

    if model_arg and model_arg != model:
        print(f"  (命令行 --model={model_arg} 覆盖为: {model})")

    print(f"  -> 模型: {model}")
    print()

    # ── Step 3: API Key ───────────────────────────────────────
    saved_key = state.get("_api_key", "")
    env_key = os.environ.get("OPENAI_API_KEY", "")
    # Priority: saved > env, but user can always override
    default_key = saved_key or env_key

    print("─" * 56)
    print("  设置 API Key")
    print("─" * 56)
    if default_key:
        source = "上次保存" if saved_key else "环境变量 OPENAI_API_KEY"
        print(f"  {source}: {_mask(default_key)}")
        print("  (直接回车使用，输入新 key 替换，输入 'clear' 清除)")
    else:
        print("  (输入 API Key，之后会记住)")
    print()

    api_key = _read_key("API Key: ", "")
    if api_key.lower() == "clear":
        state.pop("_api_key", None)
        api_key = ""
        print("  已清除。下次运行需重新输入。")
    elif not api_key and default_key:
        api_key = default_key
        print(f"  -> {_mask(api_key)}")
    elif api_key:
        state["_api_key"] = api_key
    print()

    if not api_key:
        print("错误: 未提供 API Key。")
        print("设置环境变量 OPENAI_API_KEY 或重新运行并输入 key。")
        sys.exit(1)

    # ── Persist choices ───────────────────────────────────────
    _save_state(saved_provider, model)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Step 4: Advanced settings (optional) ────────────────────
    saved_temperature = state.get("temperature", str(DEFAULT_TEMPERATURE))
    saved_truck_speed = state.get("truck_speed", str(DEFAULT_TRUCK_SPEED))
    saved_h_gain = state.get("horizontal_gain", str(DEFAULT_HORIZONTAL_GAIN))
    saved_v_gain = state.get("vertical_gain", str(DEFAULT_VERTICAL_GAIN))
    saved_proj_dist = state.get("projection_distance_max_m", str(DEFAULT_PROJECTION_DISTANCE_MAX_M))

    print("-" * 56)
    print("  高级参数 (直接回车用默认，输入数字覆盖)")
    print("-" * 56)
    print(f"  [1] VLM 温度         : {saved_temperature}   (越低越确定，0.1-1.0)")
    print(f"  [2] 卡车速度         : {saved_truck_speed} m/s (越慢越容易跟踪)")
    print(f"  [3] 水平跟踪增益     : {saved_h_gain}     (越大追得越快)")
    print(f"  [4] 垂直跟踪增益     : {saved_v_gain}     (越大下降越快)")
    print(f"  [5] 深度投影最大距离 : {saved_proj_dist} m  (depth=10 对应的实际距离)")
    print()

    adv = _read_key("修改参数 [1-5]，多个用逗号分隔，回车跳过: ", "")
    if adv.strip():
        for item in adv.split(","):
            item = item.strip()
            try:
                num, val = item.split()
                val_f = float(val)
                if num == "1":
                    state["temperature"] = str(max(0.1, min(1.0, val_f)))
                elif num == "2":
                    state["truck_speed"] = str(max(1.0, min(6.0, val_f)))
                elif num == "3":
                    state["horizontal_gain"] = str(max(1.0, min(8.0, val_f)))
                elif num == "4":
                    state["vertical_gain"] = str(max(0.3, min(3.0, val_f)))
                elif num == "5":
                    state["projection_distance_max_m"] = str(max(3.0, min(20.0, val_f)))
            except (ValueError, IndexError):
                pass
    print()

    temperature = float(state.get("temperature", DEFAULT_TEMPERATURE))
    truck_speed = float(state.get("truck_speed", DEFAULT_TRUCK_SPEED))
    horizontal_gain = float(state.get("horizontal_gain", DEFAULT_HORIZONTAL_GAIN))
    vertical_gain = float(state.get("vertical_gain", DEFAULT_VERTICAL_GAIN))
    projection_distance_max_m = float(state.get("projection_distance_max_m", DEFAULT_PROJECTION_DISTANCE_MAX_M))

    # ── Build resolved config ─────────────────────────────────
    preset = PROVIDERS[saved_provider]
    base_url = os.environ.get("OPENAI_BASE_URL") or preset["base_url"]
    if not base_url:
        print(f"错误: 提供商 '{saved_provider}' 没有默认 base_url。")
        base_url = _read_key("请输入 OpenAI-compatible base URL: ", "")
        if not base_url:
            sys.exit(1)

    print("=" * 56)
    print(f"  提供商     : {saved_provider}")
    print(f"  Base URL   : {base_url}")
    print(f"  模型       : {model}")
    print(f"  API Key    : {_mask(api_key)}")
    print(f"  VLM 温度   : {temperature}")
    print(f"  卡车速度   : {truck_speed} m/s")
    print(f"  水平增益   : {horizontal_gain}")
    print(f"  垂直增益   : {vertical_gain}")
    print(f"  深度最大距 : {projection_distance_max_m} m")
    print("=" * 56)
    print()

    return SPFConfig(
        model=model,
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        provider=saved_provider,
        temperature=temperature,
        truck_speed=truck_speed,
        horizontal_gain=horizontal_gain,
        vertical_gain=vertical_gain,
        projection_distance_max_m=projection_distance_max_m,
    )


if __name__ == "__main__":
    cfg = interactive_config()
    print(f"Resolved: provider={cfg.provider} model={cfg.model} base_url={cfg.base_url}")
