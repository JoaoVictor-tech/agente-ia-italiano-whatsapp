"""Texto-para-fala via Gemini TTS, usado quando a mensagem recebida foi audio e a resposta e
curta (< 80 palavras) - mesma regra do node "Responder por Audio?1" do fluxo n8n (que usava
ElevenLabs; neste agente tudo roda com a chave Gemini, decisao do cliente em 30/09/2026).

O Gemini devolve PCM cru (16 bits, mono, 24 kHz). A Helena espera um arquivo de audio de
verdade (o upload em app/helena_client.py:send_audio_reply assina a URL para audio/mpeg), entao
o PCM passa pelo ffmpeg (instalado no Dockerfile) e sai como mp3.
"""

from __future__ import annotations

import asyncio
import base64

import httpx

from app.config import get_settings
from app.pricing_tts import calcular_custo

settings = get_settings()

MAX_PALAVRAS_PARA_AUDIO = 80

_TAXA_AMOSTRAGEM = 24000

# Orientacao de estilo lida pelo modelo de voz (nao e falada). Italiano natural, tom de
# atendente de loja - o texto em si ja vem pronto do agente.
_ESTILO = "Leggi in italiano, con tono cordiale, naturale e sicuro, a ritmo di conversazione:"


def deve_responder_em_audio(texto: str) -> bool:
    return len(texto.split()) < MAX_PALAVRAS_PARA_AUDIO


def _texto_para_voz(texto: str) -> str:
    """Tira o que nao se fala: emojis viram ruido ou pausa estranha na sintese."""
    return "".join(ch for ch in texto if ord(ch) < 0x2190 or ch in "€").strip()


async def _pcm_para_mp3(pcm: bytes) -> bytes:
    processo = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "s16le", "-ar", str(_TAXA_AMOSTRAGEM), "-ac", "1", "-i", "pipe:0",
        "-f", "mp3", "-b:a", "64k", "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    saida, erro = await processo.communicate(pcm)
    if processo.returncode != 0 or not saida:
        raise RuntimeError(f"ffmpeg falhou ao converter o audio: {erro.decode(errors='ignore')[:300]}")
    return saida


async def synthesize(text: str, *, model_id: str | None = None) -> tuple[bytes, float]:
    """(mp3, custo_usd). `model_id` vem do modelo de voz ATIVO em /dashboard/settings (ver
    app/settings_service.py:get_active_tts_model) - so cai no .env (GEMINI_TTS_MODEL) se nada
    tiver sido salvo ainda no banco."""
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY nao configurada - sem resposta em audio.")
    model_id = model_id or settings.gemini_tts_model

    corpo = {
        "contents": [{"parts": [{"text": f"{_ESTILO}\n{_texto_para_voz(text)}"}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": settings.gemini_tts_voice}}
            },
        },
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"
    async with httpx.AsyncClient(timeout=90.0) as client:
        resposta = await client.post(url, headers={"x-goog-api-key": settings.gemini_api_key}, json=corpo)
        resposta.raise_for_status()
        dados = resposta.json()

    partes = (dados.get("candidates") or [{}])[0].get("content", {}).get("parts") or []
    pcm_b64 = next((p["inlineData"]["data"] for p in partes if p.get("inlineData")), None)
    if not pcm_b64:
        raise RuntimeError("Gemini TTS nao devolveu audio.")

    mp3 = await _pcm_para_mp3(base64.b64decode(pcm_b64))
    uso = dados.get("usageMetadata") or {}
    custo = calcular_custo(model_id, uso.get("promptTokenCount") or 0, uso.get("candidatesTokenCount") or 0)
    return mp3, custo
