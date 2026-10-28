from __future__ import annotations

import datetime

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Conversation, Event, Message

STAGE_ATIVO = "ativo"
STAGE_TRANSFERIDO = "transferido"

# Quantas mensagens o modelo enxerga da conversa. No n8n era 15 (node "Simple Memory1"); aqui
# segue o valor do agente anterior (40), que cobre a conversa inteira em praticamente todos os casos - a
# coleta do Matteo tem 5 dados, entao a conversa raramente passa de 20 mensagens.
HISTORY_WINDOW = 40


async def get_or_create_conversation(
    db: AsyncSession,
    *,
    session_id: str,
    environment: str,
    contact_id: str,
    phone: str | None,
    name: str | None,
) -> Conversation:
    conversation = await db.get(Conversation, session_id)
    if conversation is None:
        conversation = Conversation(
            session_id=session_id,
            environment=environment,
            contact_id=contact_id,
            phone=phone,
            name=name,
            slots={},
        )
        db.add(conversation)
        await db.flush()
    else:
        # mantem nome/telefone atualizados caso o contato edite o cadastro na plataforma
        conversation.contact_id = contact_id
        conversation.phone = phone or conversation.phone
        conversation.name = name or conversation.name
    return conversation


# Por quanto tempo duas sessoes do MESMO contato sao tratadas como um atendimento so.
# A Helena as vezes emite um session_id novo no meio da conversa (a sessao dela expira, ou o
# atendimento e reaberto) - sem isto o agente perdia a memoria e reperguntava o que ja sabia
# (visto no agente anterior, que deu origem a este codigo).
JANELA_MESMO_CONTATO_HORAS = 6


def _corte_mesmo_contato() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        hours=JANELA_MESMO_CONTATO_HORAS
    )


def _sessoes_irmas(*, session_id: str, contact_id: str, environment: str | None):
    """Outras sessoes do mesmo contato. O recorte de tempo e feito nas MENSAGENS, nao em
    conversations.updated_at: aquele campo so muda quando a linha da conversa e gravada de
    novo, entao nao serve como relogio confiavel de atividade."""
    irmas = select(Conversation.session_id).where(
        Conversation.contact_id == contact_id,
        Conversation.session_id != session_id,
    )
    if environment:
        # nunca misturar producao com o ambiente de teste do painel
        irmas = irmas.where(Conversation.environment == environment)
    return irmas


def _sessoes_do_contato(*, session_id: str, contact_id: str | None, environment: str | None):
    """Condicao que casa a sessao atual INTEIRA mais o que as sessoes irmas trocaram nas ultimas
    JANELA_MESMO_CONTATO_HORAS horas."""
    if not contact_id:
        return Message.session_id == session_id
    irmas = _sessoes_irmas(
        session_id=session_id, contact_id=contact_id, environment=environment
    )
    return or_(
        Message.session_id == session_id,
        and_(
            Message.session_id.in_(irmas),
            Message.created_at >= _corte_mesmo_contato(),
        ),
    )


async def get_recent_history(
    db: AsyncSession,
    *,
    session_id: str,
    limit: int = HISTORY_WINDOW,
    contact_id: str | None = None,
    environment: str | None = None,
) -> list[dict]:
    stmt = (
        select(Message)
        .where(
            _sessoes_do_contato(
                session_id=session_id, contact_id=contact_id, environment=environment
            ),
            Message.role.in_(["user", "assistant"]),
        )
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    messages = list(reversed(result.scalars().all()))
    return [{"role": message.role, "content": message.content, "session_id": message.session_id} for message in messages]


async def slots_do_contato(
    db: AsyncSession, *, session_id: str, contact_id: str | None, environment: str | None,
    combinar=None,
) -> dict:
    """Dados ja confirmados nas sessoes irmas recentes do mesmo contato (ver
    JANELA_MESMO_CONTATO_HORAS). Quem chama sobrescreve com os slots da sessao atual.
    `combinar(acumulado, slots_da_irma)`: como juntar cada irma (padrao: update)."""
    if not contact_id:
        return {}
    # so as irmas que tiveram mensagem dentro da janela
    ativas = select(Message.session_id).where(
        Message.created_at >= _corte_mesmo_contato()
    )
    stmt = (
        select(Conversation)
        .where(
            Conversation.session_id.in_(
                _sessoes_irmas(
                    session_id=session_id, contact_id=contact_id, environment=environment
                )
            ),
            Conversation.session_id.in_(ativas),
        )
        .order_by(Conversation.created_at)
    )
    juntos: dict = {}
    for conversa in (await db.execute(stmt)).scalars():
        if combinar is not None:
            combinar(juntos, conversa.slots or {})
        else:
            juntos.update(conversa.slots or {})
    return juntos


async def save_message(
    db: AsyncSession,
    *,
    session_id: str,
    role: str,
    content: str,
    model: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    cost_usd: float = 0.0,
    audio_cost_usd: float = 0.0,
    audio_provider: str | None = None,
) -> Message:
    message = Message(
        session_id=session_id,
        role=role,
        content=content,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        cost_usd=cost_usd,
        audio_cost_usd=audio_cost_usd,
        audio_provider=audio_provider,
    )
    db.add(message)
    await db.flush()
    return message


async def set_message_audio_cost(db: AsyncSession, *, message_id: int, cost_usd: float, provider: str) -> None:
    """Atualiza o custo de audio de uma mensagem ja salva - usado pela resposta em audio
    (TTS), cujo custo so e conhecido DEPOIS que a mensagem de texto ja foi gravada (a
    sintese/envio acontecem fora da transacao de banco, ver app/pipeline.py)."""
    message = await db.get(Message, message_id)
    if message is not None:
        message.audio_cost_usd = cost_usd
        message.audio_provider = provider
        await db.flush()


async def save_event(db: AsyncSession, *, session_id: str, type_: str, payload: dict) -> Event:
    event = Event(session_id=session_id, type=type_, payload=payload)
    db.add(event)
    await db.flush()
    return event


async def mark_transferred(db: AsyncSession, conversation: Conversation) -> None:
    """Chamado quando a transferencia executa com sucesso (ver
    app/agent_tools.py:executar_transferencia) - a partir daqui o bot fica em silencio nessa
    sessao (a Helena ja parou o bot do lado dela via `stopBotInExecution`, isto e so o espelho
    no nosso banco pro /dashboard mostrar o status certo)."""
    conversation.transferred_at = datetime.datetime.now(datetime.timezone.utc)
    conversation.stage = STAGE_TRANSFERIDO
    await db.flush()


async def ultima_mensagem(db: AsyncSession, *, session_id: str):
    """Ultima mensagem (de qualquer papel) desta sessao - None se nao houver."""
    stmt = select(Message).where(Message.session_id == session_id).order_by(Message.created_at.desc()).limit(1)
    return (await db.execute(stmt)).scalar_one_or_none()


async def ultima_resposta_em(db: AsyncSession, *, session_id: str):
    """Quando saiu a ultima mensagem do agente nesta sessao (None se nunca respondeu)."""
    stmt = (
        select(Message.created_at)
        .where(Message.session_id == session_id, Message.role == "assistant")
        .order_by(Message.created_at.desc())
        .limit(1)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


# Eventos que so registram o desfecho de um turno (observabilidade) e nao contam como "o que
# aconteceu por ultimo": sem isso, o desfecho_com_falha gravado depois de llm_call_failed escondia
# a falha anterior e a 2a falha seguida do modelo nunca transferia (bateria F2, 26/10/2026).
EVENTOS_DE_DESFECHO = ("desfecho_com_falha",)


async def ultimo_evento_foi(db: AsyncSession, *, session_id: str, type_: str) -> bool:
    """True quando o evento mais recente da sessao e do tipo dado (ex.: falha seguida do modelo).
    Ignora os eventos de desfecho, que vem sempre depois do evento do proprio turno."""
    stmt = (
        select(Event.type)
        .where(Event.session_id == session_id, Event.type.not_in(EVENTOS_DE_DESFECHO))
        .order_by(Event.created_at.desc()).limit(1)
    )
    ultimo = (await db.execute(stmt)).scalar_one_or_none()
    return ultimo == type_
