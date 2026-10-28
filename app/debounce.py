"""Agrega mensagens rapidas do mesmo cliente antes de acionar o agente
(equivalente ao 'lastMessagesAggregated' que a propria plataforma ja tenta fazer,
mas reforcado aqui para cobrir mensagens de audio/texto que cheguem em sequencia).

Implementacao assume um unico processo/worker (ver docker-compose.yml - um
container `api`). O buffer fica no Redis para nao se perder num restart do
processo antes do flush disparar; o agendamento do flush em si e feito com
asyncio no processo local. Se no futuro for necessario rodar varios workers
em paralelo, essa parte precisa migrar para um agendador distribuido (ex: arq).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable

import redis.asyncio as redis

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_redis = redis.from_url(settings.redis_url, decode_responses=True)
_pending_flush_tasks: dict[str, asyncio.Task] = {}
# sessoes cujo flush ja passou do sleep e esta executando on_flush de verdade - nunca
# cancelar um flush nesse estado (ver enqueue_message).
_flushing: set[str] = set()

# on_flush(session_id, texto_agregado, contexto) - contexto e o ultimo snapshot
# passado em enqueue_message (dados do contato/canal daquele webhook).
OnFlush = Callable[[str, str, dict], Awaitable[None]]


def _buffer_key(session_id: str) -> str:
    return f"debounce:{session_id}:buffer"


def _context_key(session_id: str) -> str:
    return f"debounce:{session_id}:context"


async def tem_mensagem_pendente(session_id: str) -> bool:
    """O cliente mandou algo NOVO enquanto o turno atual ainda estava sendo processado (a
    mensagem ficou no buffer esperando este flush terminar). Ver pipeline: uma resposta que
    so pergunta, produzida antes dessa mensagem, chega atrasada e parece repetida."""
    try:
        return bool(await _redis.exists(_buffer_key(session_id)))
    except Exception:
        logger.exception("Falha ao consultar o buffer de debounce da sessao %s", session_id)
        return False


async def enqueue_message(
    session_id: str, text: str, context: dict[str, Any], on_flush: OnFlush,
    *, atraso_segundos: float | None = None,
) -> None:
    """`atraso_segundos`: espera diferente da padrao para ESTE agendamento (ver pipeline:
    abertura do botao do site). Mensagem nova reagenda com a espera padrao."""
    if not text:
        return

    # Mescla com o contexto ja pendente (se houver) em vez de sobrescrever - sem isso, se o
    # cliente mandar 2+ audios dentro da mesma janela de debounce, o custo de transcricao (STT)
    # do primeiro audio seria perdido quando o segundo chegasse (cada enqueue_message sobrescrevia
    # o contexto inteiro). "stt_cost_usd" e somado de proposito; os demais campos (telefone,
    # nome etc.) sao estaveis dentro da mesma sessao, entao o ultimo valor prevalece normalmente.
    raw_context_anterior = await _redis.get(_context_key(session_id))
    if raw_context_anterior:
        context_anterior = json.loads(raw_context_anterior)
        custo_acumulado = context_anterior.get("stt_cost_usd", 0.0) + context.get("stt_cost_usd", 0.0)
        context = {**context_anterior, **context, "stt_cost_usd": custo_acumulado}

    await _redis.rpush(_buffer_key(session_id), text)
    await _redis.expire(_buffer_key(session_id), 300)
    await _redis.set(_context_key(session_id), json.dumps(context), ex=300)

    if session_id in _flushing:
        # ja existe um flush rodando de verdade (passou do sleep, esta chamando on_flush) -
        # NUNCA cancelar esse: cancelar um flush em andamento mata o turno em silencio (o
        # CancelledError nao e pego pelo "except Exception" do proprio flush). A mensagem
        # nova so fica no buffer do Redis e o proprio flush em andamento reagenda outro
        # flush pra ela ao terminar (ver finally em _flush_after_delay).
        return

    existing = _pending_flush_tasks.get(session_id)
    if existing and not existing.done():
        existing.cancel()

    _pending_flush_tasks[session_id] = asyncio.create_task(
        _flush_after_delay(session_id, on_flush, atraso_segundos)
    )


async def _flush_after_delay(session_id: str, on_flush: OnFlush, atraso_segundos: float | None = None) -> None:
    try:
        await asyncio.sleep(atraso_segundos if atraso_segundos is not None else settings.debounce_seconds)
    except asyncio.CancelledError:
        return

    _pending_flush_tasks.pop(session_id, None)
    _flushing.add(session_id)
    try:
        buffer_key = _buffer_key(session_id)
        context_key = _context_key(session_id)

        parts = await _redis.lrange(buffer_key, 0, -1)
        raw_context = await _redis.get(context_key)
        await _redis.delete(buffer_key, context_key)

        if not parts:
            return

        aggregated = "\n".join(parts)
        context = json.loads(raw_context) if raw_context else {}
        try:
            await on_flush(session_id, aggregated, context)
        except BaseException:
            # BaseException (nao so Exception) de proposito: se esse turno for cancelado
            # por algum motivo externo, queremos o log mesmo assim - nada mais espera esta
            # task, entao engolir e logar aqui e seguro e evita uma falha 100% silenciosa.
            logger.exception("Falha ao processar mensagem agregada da sessao %s", session_id)
    finally:
        _flushing.discard(session_id)
        # chegou mensagem nova enquanto processavamos o flush anterior - agenda outro flush
        # pra ela agora que terminamos (ela nao foi cancelada, so ficou esperando no buffer).
        if await _redis.exists(_buffer_key(session_id)):
            _pending_flush_tasks[session_id] = asyncio.create_task(_flush_after_delay(session_id, on_flush))
