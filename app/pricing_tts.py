"""Catalogo de modelos de voz (Gemini TTS) usados na resposta em audio, com preco por milhao de
tokens (tabela publica do Google, set/2026). O Gemini cobra o TEXTO enviado como entrada e o
AUDIO gerado como saida (~25 tokens por segundo de fala) - os dois numeros vem no retorno da API
(usageMetadata), entao o custo gravado em app/tts.py e o real, nao uma estimativa.

Revisar estes valores se o Google mudar a tabela.
"""

from __future__ import annotations

TTS_MODEL_CATALOG: dict[str, dict] = {
    "gemini-2.5-flash-preview-tts": {
        "label": "Gemini 2.5 Flash TTS (padrão, rápido)",
        "usd_input_per_million": 0.50,
        "usd_output_per_million": 10.00,
    },
    "gemini-2.5-pro-preview-tts": {
        "label": "Gemini 2.5 Pro TTS (mais natural, mais caro)",
        "usd_input_per_million": 1.00,
        "usd_output_per_million": 20.00,
    },
}

DEFAULT_TTS_MODEL = "gemini-2.5-flash-preview-tts"


def calcular_custo(model_id: str, tokens_entrada: int, tokens_saida: int) -> float:
    tabela = TTS_MODEL_CATALOG.get(model_id) or TTS_MODEL_CATALOG[DEFAULT_TTS_MODEL]
    return (
        tokens_entrada / 1_000_000 * tabela["usd_input_per_million"]
        + tokens_saida / 1_000_000 * tabela["usd_output_per_million"]
    )
