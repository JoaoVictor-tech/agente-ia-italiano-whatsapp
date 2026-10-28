"""Catalogo de modelos e calculo de custo por provedor, a partir de precos
pesquisados em julho/2026 (nao de memoria de treino - modelos e precos mudam
com frequencia). Revisar periodicamente contra a pagina de precos de cada
provedor.
"""

from __future__ import annotations

from dataclasses import dataclass

# input/output em USD por milhao de tokens. cache_read (quando informado) e o preco
# absoluto de tokens servidos via cache (nao um multiplicador). Os tres provedores fazem cache
# automatico, sem custo de escrita separado.
MODEL_CATALOG: dict[str, dict[str, dict]] = {
    "gemini": {
        "gemini-3.1-pro-preview": {"label": "Gemini 3.1 Pro (preview)", "input": 2.00, "output": 12.00},
        "gemini-3.5-flash": {"label": "Gemini 3.5 Flash", "input": 1.50, "output": 9.00, "cache_read": 0.15},
        "gemini-3-flash-preview": {"label": "Gemini 3 Flash (preview)", "input": 0.50, "output": 3.00},
        "gemini-3.1-flash-lite": {"label": "Gemini 3.1 Flash-Lite (mais barato)", "input": 0.25, "output": 1.50},
        "gemini-2.5-flash": {"label": "Gemini 2.5 Flash", "input": 0.30, "output": 2.50},
    },
    "openai": {
        "gpt-5.6-sol": {"label": "GPT-5.6 Sol (topo de linha)", "input": 5.00, "output": 30.00, "cache_read": 0.50},
        "gpt-5.6-terra": {"label": "GPT-5.6 Terra (equilibrado)", "input": 2.50, "output": 15.00, "cache_read": 0.25},
        "gpt-5.6-luna": {"label": "GPT-5.6 Luna (mais barato)", "input": 1.00, "output": 6.00, "cache_read": 0.10},
        "gpt-5.4-mini": {"label": "GPT-5.4 mini", "input": 0.75, "output": 4.50, "cache_read": 0.075},
    },
    "deepseek": {
        "deepseek-v4-flash": {"label": "DeepSeek V4 Flash (mais barato)", "input": 0.14, "output": 0.28, "cache_read": 0.0028},
        "deepseek-v4-pro": {"label": "DeepSeek V4 Pro", "input": 0.435, "output": 0.87, "cache_read": 0.003625},
    },
}

PROVIDER_LABELS = {
    "openai": "OpenAI (GPT)",
    "gemini": "Google Gemini",
    "deepseek": "DeepSeek",
}


@dataclass
class NormalizedUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class UsageCost:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float


def get_model_info(provider: str, model: str) -> dict:
    return MODEL_CATALOG.get(provider, {}).get(model) or {"label": model, "input": 0.0, "output": 0.0}


def calculate_cost(provider: str, model: str, usage: NormalizedUsage) -> UsageCost:
    info = get_model_info(provider, model)
    input_price = info["input"]
    output_price = info["output"]

    cache_read_price = info.get("cache_read", input_price * 0.1)
    cache_write_price = info.get("cache_write", 0.0)

    cost = (
        (usage.input_tokens / 1_000_000) * input_price
        + (usage.output_tokens / 1_000_000) * output_price
        + (usage.cache_read_tokens / 1_000_000) * cache_read_price
        + (usage.cache_write_tokens / 1_000_000) * cache_write_price
    )

    return UsageCost(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        cost_usd=round(cost, 8),
    )
