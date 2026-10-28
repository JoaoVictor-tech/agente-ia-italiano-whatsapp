"""Utilitarios compartilhados pelas chamadas de LLM.

Ate 10/2026 este arquivo tambem servia ao loop de tool-calling do motor antigo (removido).
"""

from __future__ import annotations

from app.pricing_llm import NormalizedUsage

# Teto de tokens da resposta. Folga maior que o padrao: a mensagem inicial do Matteo e' a
# mais longa (lista dos 5 dados com emoji) e resposta cortada no meio e' pior do que gastar
# alguns tokens. O raciocinio do Gemini fica no minimo (chat_models._thinking) para nao comer
# este teto.
MAX_OUTPUT_TOKENS = 2048


def sum_usage(a: NormalizedUsage, b: NormalizedUsage) -> NormalizedUsage:
    return NormalizedUsage(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        cache_read_tokens=a.cache_read_tokens + b.cache_read_tokens,
        cache_write_tokens=a.cache_write_tokens + b.cache_write_tokens,
    )
