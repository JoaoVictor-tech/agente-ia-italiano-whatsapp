"""Retomada de mensagem orfa: responde mensagem de cliente que ficou sem resposta nenhuma.

Herdado do agente anterior (caso real 21/08/2026): o cliente mandou a medida e o container foi
reiniciado 8 segundos depois, por um deploy. A mensagem ja estava gravada no banco (o pipeline
commita antes de chamar o modelo), mas a tarefa que geraria a resposta morreu junto com o
processo, e ninguem respondeu.

O turno vive inteiro dentro de uma BackgroundTask do FastAPI (ver app/main.py), entao qualquer
queda - deploy, OOM, crash - descarta o atendimento em andamento. Aqui varremos o banco atras de
conversas cuja ULTIMA mensagem e do cliente e as respondemos.

Roda na subida (pegando inclusive o que a propria reinicializacao derrubou) e a cada
INTERVALO_VARREDURA_SEGUNDOS, o que tambem cobre webhook perdido pela Helena."""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy import text

from app import repository
from app.db import async_session

logger = logging.getLogger(__name__)

# Precisa ser maior que a janela de debounce (app/config.py: debounce_seconds), senao
# atropelariamos uma mensagem que ainda esta esperando agrupamento.
ATRASO_MINIMO_SEGUNDOS = 90
# Acima disso o cliente ja desistiu ou foi atendido por outro canal - responder do nada seria
# pior que ficar quieto.
JANELA_MAXIMA_MINUTOS = 30
# Teto por varredura, para um incidente maior nao virar uma enxurrada de mensagens de uma vez.
MAXIMO_POR_VARREDURA = 20
INTERVALO_VARREDURA_SEGUNDOS = 300
EVENTO_RETOMADA = "mensagem_orfa_retomada"

# Serializa varreduras no mesmo processo - a de 5 em 5 minutos pode pegar uma anterior ainda
# rodando quando o modelo demora, e as duas veriam a mesma orfa antes de qualquer evento ser
# gravado.
_trava_varredura = asyncio.Lock()

_SQL_ORFAS = text(
    """
    SELECT c.session_id, c.environment, c.contact_id, c.phone, c.name
    FROM conversations c
    JOIN LATERAL (
        SELECT m.role, m.created_at
        FROM messages m
        WHERE m.session_id = c.session_id
        ORDER BY m.created_at DESC
        LIMIT 1
    ) ultima ON true
    WHERE c.stage <> :stage_transferido
      AND c.paused_by_human = false
      AND ultima.role = 'user'
      AND ultima.created_at < now() - make_interval(secs => :atraso)
      AND ultima.created_at > now() - make_interval(mins => :janela)
      -- Ja retomada: existe um evento REGISTRADO DEPOIS da mensagem do cliente. Sem isto, duas
      -- varreduras sobrepostas respondem a mesma pessoa duas vezes (visto num agente anterior).
      AND NOT EXISTS (
          SELECT 1 FROM events e
          WHERE e.session_id = c.session_id
            AND e.type = :tipo_evento
            AND e.created_at > ultima.created_at
      )
    ORDER BY ultima.created_at
    LIMIT :limite
    """
)


async def _orfas() -> list[dict]:
    async with async_session() as db:
        resultado = await db.execute(
            _SQL_ORFAS,
            {
                "stage_transferido": repository.STAGE_TRANSFERIDO,
                "tipo_evento": EVENTO_RETOMADA,
                "atraso": ATRASO_MINIMO_SEGUNDOS,
                "janela": JANELA_MAXIMA_MINUTOS,
                "limite": MAXIMO_POR_VARREDURA,
            },
        )
        return [dict(linha) for linha in resultado.mappings().all()]


async def varrer_uma_vez() -> int:
    """Responde as orfas encontradas agora. Devolve quantas foram retomadas."""
    async with _trava_varredura:
        return await _varrer()


async def _varrer() -> int:
    # Import tardio: app/pipeline.py importa este modulo indiretamente via app/main.py, e o
    # import no topo criaria ciclo.
    from app.pipeline import _process_turn

    try:
        pendentes = await _orfas()
    except Exception:
        logger.exception("Falha ao procurar mensagens orfas.")
        return 0

    from app import debounce
    from app.pipeline import turnos_em_andamento

    retomadas = 0
    for conversa in pendentes:
        session_id = conversa["session_id"]
        if session_id in turnos_em_andamento or session_id in debounce._flushing or await debounce.tem_mensagem_pendente(session_id):
            # O turno original ainda esta rodando (provedor lento) - responder de novo daria duas
            # respostas no mesmo segundo. Nao e orfa.
            logger.info("Sessao %s tem turno em andamento/mensagem em processamento - nao e orfa.", session_id)
            continue
        contexto = {
            "environment": conversa["environment"],
            "contact_id": conversa["contact_id"],
            "phone": conversa["phone"],
            "name": conversa["name"],
            "message_type": "TEXT",
            "stt_cost_usd": 0.0,
        }
        try:
            logger.warning("Retomando mensagem orfa da sessao %s.", session_id)
            async with async_session() as db:
                await repository.save_event(
                    db, session_id=session_id, type_=EVENTO_RETOMADA, payload={}
                )
                await db.commit()
            # `mensagem_ja_salva=True`: a mensagem do cliente ja esta no banco - grava-la de novo
            # duplicaria a conversa no painel e no historico enviado ao modelo.
            await _process_turn(session_id, "", contexto, mensagem_ja_salva=True)
            retomadas += 1
        except Exception:
            logger.exception("Falha ao retomar mensagem orfa da sessao %s.", session_id)

    if retomadas:
        logger.warning("Mensagens orfas retomadas nesta varredura: %s", retomadas)
    return retomadas


async def loop_de_varredura() -> None:
    """Roda para sempre; chamado como task na subida da aplicacao (ver app/main.py)."""
    # Pequena espera na primeira volta: da tempo do banco/redis subirem junto com a API e evita
    # atropelar um turno que ainda esta terminando durante o restart.
    await asyncio.sleep(30)
    while True:
        try:
            await varrer_uma_vez()
        except Exception:
            logger.exception("Erro inesperado na varredura de mensagens orfas.")
        await asyncio.sleep(INTERVALO_VARREDURA_SEGUNDOS)
