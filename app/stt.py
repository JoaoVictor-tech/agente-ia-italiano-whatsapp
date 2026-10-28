"""Transcricao do audio recebido - substitui o node "Transcreve áudio" (OpenAI Whisper) do fluxo
n8n. Neste agente tudo roda com a chave Gemini (decisao do cliente, 30/09/2026), entao o audio vai
direto para a API nativa do Gemini como parte inline do generateContent.

A API OpenAI-compativel do Gemini NAO expoe /audio/transcriptions (404, conferido num agente
anterior em 02/09/2026) - por isso a chamada e pela API nativa, nao pelo SDK da OpenAI.
"""

from __future__ import annotations

import base64
import logging

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

PROVEDOR = "gemini_audio"

_PROMPT = (
    "Trascrivi questo audio. Il cliente parla italiano (può capitare qualche parola in un'altra "
    "lingua: trascrivila com'è). Rispondi SOLO con la trascrizione, senza commenti, senza "
    "virgolette e senza testo aggiuntivo. Scrivi le misure dei pneumatici in cifre "
    "(es. \"duecentocinque cinquantacinque sedici\" -> \"205/55 R16\")."
)

# O Gemini cobra audio como token de ENTRADA - e mais caro que texto. USD por milhao de tokens
# de audio do gemini-2.5-flash (tabela publica do Google, set/2026). O numero de tokens vem no
# proprio retorno (usageMetadata), entao nao precisamos estimar a duracao.
_USD_POR_MILHAO_TOKENS_AUDIO = 1.00
_USD_POR_MILHAO_TOKENS_SAIDA = 2.50


async def download_audio(file_url: str) -> bytes:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(file_url)
        response.raise_for_status()
        return response.content


async def transcribe(audio_bytes: bytes, mime_type: str | None = None) -> tuple[str, float, str]:
    """Retorna (texto, custo_usd, provedor). Levanta excecao se falhar - quem chama decide o
    que fazer (ver app/pipeline.py, que troca por um marcador e transfere)."""
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY nao configurada - sem transcricao de audio.")

    # WhatsApp manda voz como audio/ogg (opus); o Gemini aceita ogg/mp3/wav/aac/flac direto.
    mime = (mime_type or "audio/ogg").split(";")[0].strip() or "audio/ogg"
    corpo = {
        "contents": [
            {
                "parts": [
                    {"text": _PROMPT},
                    {"inline_data": {"mime_type": mime, "data": base64.b64encode(audio_bytes).decode()}},
                ]
            }
        ],
        # Sem raciocinio (mesmo motivo de app/llm_providers/gemini_provider.py): os tokens de
        # pensamento contam no teto de saida e nao ajudam numa transcricao.
        "generationConfig": {"temperature": 0, "thinkingConfig": {"thinkingBudget": 0}},
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.gemini_stt_model}:generateContent"
    async with httpx.AsyncClient(timeout=90.0) as client:
        resposta = await client.post(url, headers={"x-goog-api-key": settings.gemini_api_key}, json=corpo)
        resposta.raise_for_status()
        dados = resposta.json()

    partes = (dados.get("candidates") or [{}])[0].get("content", {}).get("parts") or []
    texto = "".join(p.get("text", "") for p in partes).strip()

    uso = dados.get("usageMetadata") or {}
    custo = (
        (uso.get("promptTokenCount") or 0) / 1_000_000 * _USD_POR_MILHAO_TOKENS_AUDIO
        + (uso.get("candidatesTokenCount") or 0) / 1_000_000 * _USD_POR_MILHAO_TOKENS_SAIDA
    )
    return texto, custo, PROVEDOR
