"""Orquestra o processamento de uma mensagem recebida do webhook, do debounce ate a resposta
enviada de volta para a Helena e a gravacao em banco - agente Matteo (Gomme Fenice).

Mapeamento com o fluxo n8n "agente original":

    Webhook / Coleta Texto / Divide arquivos / Switch  -> handle_incoming_webhook
    Baixa Audio / Transcreve áudio                      -> app/stt.py (Gemini)
    Agrupa Inputs / Monta JSON                          -> app/debounce.py (Redis)
    AI Agent1 + Simple Memory1                          -> app/llm.py + app/motor_v2.py + historico no Postgres
    criaResumo / transferirAtendimento                  -> app/agent_tools.py (executadas aqui, no fim do turno)
    Monta JSON Final (tira 【...】)                     -> app/response_grounding.py:limpar_resposta
    Responder por Audio?1 / If1 / ElevenLabs / upload   -> app/tts.py (Gemini TTS) + helena_client.send_audio_reply
    Retorno                                             -> helena_client.send_text_bubbles

Diferencas deliberadas em relacao ao n8n:
- Foto: no n8n o ramo IMAGEM do Switch nao ia a lugar nenhum e uma foto sem legenda era
  descartada pelo filtro "Coleta Texto" - o agente nunca sabia que a foto chegou, entao a regra
  "FOTO RICEVUTA" do prompt nunca disparava. Aqui a foto vira o marcador "[FOTO]" e segue o fluxo.
- Audio que nao da para transcrever e mensagem que a Helena nao consegue abrir nao viram
  silencio: o cliente recebe uma frase e vai para o operador, com nota.
- A transferencia so e executada depois de a resposta ao cliente estar pronta (ver
  _texto_da_passagem): quem decide que ela acontece e' a politica (app/politica.py), no motor.
- Todo o caminho - entrada, travas, motor, envio e transferencia - roda dentro de um trace de
  atendimento (app/observabilidade.py) que termina com um DESFECHO explicito. Sem ele, uma
  falha de envio pela Helena aparecia como turno bem-sucedido no painel.
"""

from __future__ import annotations

import logging
import re

from app import (
    agent_tools,
    atendimento_helena,
    helena_client,
    llm,
    observabilidade,
    repository,
    resumo_atendimento,
    stt,
    tts,
)
from app.config import get_settings
from app.db import async_session
from app.debounce import enqueue_message
from app.models import Conversation
from app.repository import STAGE_TRANSFERIDO
from app.settings_service import (
    get_active_knowledge_entries,
    get_active_llm_settings,
    get_active_tts_model,
    get_audio_replies_enabled,
    get_environment_credentials,
    get_llm_settings_for,
)
from app.webhook_schemas import HelenaWebhookPayload

logger = logging.getLogger(__name__)
settings = get_settings()

# Usado so quando o motor nao chega a produzir texto (duas falhas seguidas de provedor). A
# resposta barrada pela verificacao NAO cai aqui: ela vira o testo approvato da acao
# (app/redacao.py:texto_pronto), que e' muito melhor para o cliente.
FALLBACK_REPLY = "Scusami, ho avuto un piccolo problema tecnico. Puoi ripetere, per favore?"

# Marcadores gravados no lugar da midia - o painel mostra o que chegou, e o motor sabe o que
# fazer (a foto vira passagem ao operador; ver app/motor_v2.py:PREFIXO_FOTO).
MARCADOR_FOTO = "[FOTO] Il cliente ha inviato una foto."
MARCADOR_AUDIO_NAO_TRANSCRITO = "[audio ricevuto - trascrizione non riuscita]"
MARCADOR_NAO_SUPORTADA = "[contenuto non supportato dalla piattaforma]"

# Textos fixos (italiano, mesmo tom do prompt) para quando o SERVIDOR transfere.
TESTO_FOTO_RICEVUTA = (
    "Perfetto 👍 ho ricevuto la foto.\n"
    "Passo la tua richiesta a un operatore che la controlla e ti aiuta con la misura esatta."
)
TESTO_AUDIO_NON_TRASCRITTO = (
    "Ho ricevuto il tuo vocale, ma da qui non riesco ad ascoltarlo bene. "
    "Passo la tua richiesta a un esperto di Gomme Fenice che continua con te."
)
TESTO_NON_SUPPORTATO = (
    "Ho ricevuto il tuo messaggio, ma da qui non riesco ad aprire questo contenuto. "
    "Passo la tua richiesta a un esperto di Gomme Fenice che continua con te."
)
TESTO_PASSAGGIO = "Perfetto 👍 passo la tua richiesta a un esperto di Gomme Fenice, che continua con te."
# Mensagem padrao "DATI COMPLETI" (mesmo testo approvato de app/redacao.py).
TESTO_DATI_COMPLETI = (
    "Perfetto 👍 ho tutto!\n"
    "Ora verifichiamo la soluzione Gomme Fenice più adatta alle tue esigenze per disponibilità e prezzo."
)

# A plataforma entrega este texto no lugar de midia que ela nao consegue abrir (visto num agente anterior).
_REGEX_MENSAGEM_NAO_SUPORTADA = re.compile(
    r"n[ãa]o suportad[ao] pela api|message type unknown|mensagem ou rea[çc][ãa]o n[ãa]o suportad",
    re.IGNORECASE,
)


def _arquivos_do_lote(payload: HelenaWebhookPayload) -> list:
    candidatos = []
    if payload.lastMessage and payload.lastMessage.file:
        candidatos.append(payload.lastMessage.file)
    if payload.lastMessagesAggregated:
        candidatos.extend(payload.lastMessagesAggregated.files or [])
    return [a for a in candidatos if a and a.url]


def _audio_do_lote(payload: HelenaWebhookPayload):
    for arquivo in _arquivos_do_lote(payload):
        if (arquivo.mimeType or "").lower().startswith("audio"):
            return arquivo
    if payload.message_type == "AUDIO":
        arquivos = _arquivos_do_lote(payload)
        return arquivos[0] if arquivos else None
    return None


def _tem_imagem_no_lote(payload: HelenaWebhookPayload) -> bool:
    """Foto em qualquer posicao do lote (num agente anterior, foto + texto no mesmo lote vinham com
    lastMessage = TEXT e a foto se perdia)."""
    if payload.message_type == "IMAGE":
        return True
    return any((a.mimeType or "").lower().startswith("image") for a in _arquivos_do_lote(payload))


async def handle_incoming_webhook(payload: HelenaWebhookPayload, environment: str) -> None:
    session_id = payload.sessionId
    texto = (payload.aggregated_text or "").strip()
    stt_cost_usd = 0.0
    stt_provider = None

    audio = _audio_do_lote(payload)
    if audio is not None:
        try:
            audio_bytes = await stt.download_audio(audio.url)
            transcricao, stt_cost_usd, stt_provider = await stt.transcribe(audio_bytes, audio.mimeType)
            texto = "\n".join(t for t in (texto, transcricao.strip()) if t)
        except Exception:
            logger.exception("Falha ao transcrever audio da sessao %s", session_id)
        if not texto:
            logger.warning("Audio da sessao %s nao pode ser transcrito - transbordo.", session_id)
            texto = MARCADOR_AUDIO_NAO_TRANSCRITO
    elif _tem_imagem_no_lote(payload):
        # O Matteo nao le imagens (regra do prompt): a foto vira marcador e a legenda, se
        # houver, vai junto.
        texto = f"{MARCADOR_FOTO}\n{texto}".strip() if texto else MARCADOR_FOTO
    elif _REGEX_MENSAGEM_NAO_SUPORTADA.search(texto):
        texto = MARCADOR_NAO_SUPORTADA
    elif not texto:
        # PDF/arquivo sem texto nenhum: nada que o agente consiga usar - vai para o operador.
        texto = MARCADOR_NAO_SUPORTADA if _arquivos_do_lote(payload) else ""

    context = {
        "environment": environment,
        "contact_id": payload.contact.id,
        "phone": payload.contact.display_phonenumber or payload.contact.phonenumber,
        "name": payload.contact.first_name or payload.contact.name,
        # So "AUDIO" importa depois: define se a resposta pode sair em audio.
        "message_type": "AUDIO" if audio is not None and texto != MARCADOR_AUDIO_NAO_TRANSCRITO else payload.message_type,
        "stt_cost_usd": stt_cost_usd,
        "stt_provider": stt_provider,
    }
    await enqueue_message(session_id, texto, context, on_flush=_process_turn)


# Sessoes com turno em execucao NESTE processo - a varredura de mensagens orfas
# (app/orphan_recovery.py) consulta este conjunto e pula quem ainda esta sendo respondido.
turnos_em_andamento: set[str] = set()


async def _process_turn(
    session_id: str, aggregated_text: str, context: dict, *, mensagem_ja_salva: bool = False
) -> None:
    if session_id in turnos_em_andamento and mensagem_ja_salva:
        logger.warning("Sessao %s ja tem turno em andamento - retomada de orfa ignorada.", session_id)
        return
    turnos_em_andamento.add(session_id)
    try:
        with observabilidade.atendimento(
            session_id=session_id,
            contact_id=context.get("contact_id"),
            environment=context["environment"],
            entrada=aggregated_text,
            tipo_mensagem=context.get("message_type"),
        ) as trace:
            if context.get("stt_provider"):
                trace.registrar_entrada({
                    "tipo": "audio",
                    "modelo": context.get("stt_provider"),
                    "custo_usd": context.get("stt_cost_usd"),
                    "resultado": aggregated_text,
                    "erro": "transcricao nao retornou texto"
                    if aggregated_text.strip() == MARCADOR_AUDIO_NAO_TRANSCRITO else None,
                })
            try:
                await _process_turn_interno(
                    session_id, aggregated_text, context, trace=trace, mensagem_ja_salva=mensagem_ja_salva
                )
            except BaseException as erro:
                # O desfecho precisa estar marcado ANTES do finally abaixo: quem marca
                # falha_interna e' o proprio trace, mas so quando a excecao chega nele.
                if not trace.falhou:
                    trace.marcar(observabilidade.FALHA_INTERNA, f"{type(erro).__name__}: {erro}"[:300])
                raise
            finally:
                # Tambem no caminho de excecao: a falha de envio sobe daqui, e o evento com o
                # link do trace e' justamente o que faz alguem descobrir que ela aconteceu.
                await _gravar_falha_no_painel(session_id, trace)
    finally:
        turnos_em_andamento.discard(session_id)


async def _gravar_falha_no_painel(session_id: str, trace: observabilidade.Atendimento) -> None:
    """Desfecho de falha tambem vira evento no painel, com o link do trace - quem abre a
    conversa ve o que deu errado sem precisar procurar no Langfuse."""
    if not trace.falhou:
        return
    try:
        link = await trace.url()
        async with async_session() as db:
            await repository.save_event(
                db, session_id=session_id, type_="desfecho_com_falha",
                payload={"desfecho": trace.desfecho, "detalhe": trace.detalhe, "trace": link},
            )
            await db.commit()
    except Exception:
        logger.exception("Falha ao gravar o desfecho %s da sessao %s no painel.", trace.desfecho, session_id)


async def _responder_e_transferir(
    *, credentials, session_id: str, texto: str, motivo: str, ultima_mensagem: str, evento: str,
    trace: observabilidade.Atendimento,
) -> None:
    """Resposta fixa + nota automatica + transferencia, sem chamar o modelo."""
    async with async_session() as db:
        conversa = await db.get(Conversation, session_id)
        dados = dict(conversa.slots or {}) if conversa else {}
    try:
        with trace.etapa("envio", tipo="texto_fixo"):
            await helena_client.send_canned_text(credentials=credentials, session_id=session_id, text=texto)
    except Exception:
        trace.marcar(observabilidade.FALHA_ENVIO, "a Helena nao aceitou a mensagem fixa")
        raise
    trace.resposta = texto
    contexto = agent_tools.ToolContext(session_id=session_id, credentials=credentials)
    contexto.slots.update(dados)
    contexto.ultima_mensagem_do_cliente = ultima_mensagem
    with trace.etapa("transferencia", motivo=motivo):
        transferido = await agent_tools.executar_transferencia(contexto, motivo_sem_nota=motivo)
    if transferido:
        trace.marcar(observabilidade.RESPONDIDO, motivo)
        trace.avisar(f"transferencia sem passar pelo motor: {motivo}")
    else:
        trace.marcar(observabilidade.FALHA_TRANSFERENCIA, motivo)
    async with async_session() as db:
        conversa = await db.get(Conversation, session_id)
        await repository.save_message(db, session_id=session_id, role="assistant", content=texto)
        if transferido and conversa is not None:
            await repository.mark_transferred(db, conversa)
        await repository.save_event(db, session_id=session_id, type_=evento, payload={"transferido": transferido})
        await db.commit()


async def _process_turn_interno(
    session_id: str, aggregated_text: str, context: dict, *,
    trace: observabilidade.Atendimento, mensagem_ja_salva: bool = False,
) -> None:
    """`mensagem_ja_salva=True` vem da retomada de mensagem orfa (app/orphan_recovery.py): a
    mensagem do cliente ja esta gravada, entao nao a gravamos de novo e a tiramos do fim do
    historico para usa-la como `user_message` deste turno."""
    environment = context["environment"]
    stt_cost_usd = context.get("stt_cost_usd", 0.0)
    stt_provider = context.get("stt_provider")

    async with async_session() as db:
        conversation = await repository.get_or_create_conversation(
            db,
            session_id=session_id,
            environment=environment,
            contact_id=context["contact_id"],
            phone=context.get("phone"),
            name=context.get("name"),
        )

        if conversation.paused_by_human or conversation.stage == STAGE_TRANSFERIDO:
            # Um humano assumiu pelo painel, ou a conversa ja foi transferida (a Helena ja parou
            # o bot via stopBotInExecution - isto cobre um webhook atrasado). So registra.
            if not mensagem_ja_salva:
                await repository.save_message(
                    db, session_id=session_id, role="user", content=aggregated_text,
                    audio_cost_usd=stt_cost_usd, audio_provider=stt_provider,
                )
            await db.commit()
            trace.marcar(
                observabilidade.SILENCIO_INTENCIONAL,
                "operador assumiu pelo painel" if conversation.paused_by_human else "conversa ja transferida",
            )
            return

        history = await repository.get_recent_history(
            db, session_id=session_id, contact_id=conversation.contact_id, environment=environment,
        )
        history = [{"role": m["role"], "content": m["content"]} for m in history]
        dados_confirmados = await repository.slots_do_contato(
            db, session_id=session_id, contact_id=conversation.contact_id, environment=environment,
        )
        dados_confirmados.update(conversation.slots or {})

        if mensagem_ja_salva:
            pendentes = []
            while history and history[-1].get("role") == "user":
                pendentes.insert(0, history.pop().get("content") or "")
            aggregated_text = "\n".join(t for t in pendentes if t).strip()
            if not aggregated_text:
                logger.warning("Retomada de orfa sem texto de cliente na sessao %s - ignorando.", session_id)
                await db.commit()
                return
        else:
            await repository.save_message(
                db, session_id=session_id, role="user", content=aggregated_text,
                audio_cost_usd=stt_cost_usd, audio_provider=stt_provider,
            )

        llm_settings = await get_active_llm_settings(db)
        # Reserva (FALLBACK_LLM_PROVIDER/MODEL). Pode ser o mesmo provedor com OUTRO modelo -
        # com uma chave Gemini so, e o que cobre um modelo sobrecarregado.
        llm_settings_fallback = None
        cfg = get_settings()
        if cfg.fallback_llm_provider and cfg.fallback_llm_model and (
            cfg.fallback_llm_provider, cfg.fallback_llm_model
        ) != (llm_settings.provider, llm_settings.model):
            llm_settings_fallback = await get_llm_settings_for(
                db, provider=cfg.fallback_llm_provider, model=cfg.fallback_llm_model
            )
        avisos = [e.content for e in await get_active_knowledge_entries(db)]
        credentials = await get_environment_credentials(db, environment)
        tts_model_id = await get_active_tts_model(db)
        audio_replies_enabled = await get_audio_replies_enabled(db)
        # commit aqui - a mensagem do cliente fica salva e visivel no painel mesmo que a
        # chamada ao modelo logo abaixo falhe.
        await db.commit()

    # A conversa ainda e do bot? Um operador pode ter puxado a conversa direto na Helena, e a
    # plataforma nao nos avisa. DESCONHECIDO segue respondendo de proposito.
    if await atendimento_helena.quem_atende(
        credentials=credentials, session_id=session_id, environment=environment
    ) == atendimento_helena.HUMANO:
        logger.info("Sessao %s ja esta com um operador na Helena - o agente nao responde.", session_id)
        async with async_session() as db:
            conversa = await db.get(Conversation, session_id)
            if conversa is not None and conversa.stage != STAGE_TRANSFERIDO:
                await repository.mark_transferred(db, conversa)
                await repository.save_event(db, session_id=session_id, type_="atendimento_assumido_por_humano", payload={})
            await db.commit()
        trace.marcar(observabilidade.SILENCIO_INTENCIONAL, "operador ja assumiu na Helena")
        return

    # Midia que nao virou texto: resposta fixa e operador, sem gastar modelo.
    if aggregated_text.strip() == MARCADOR_AUDIO_NAO_TRANSCRITO:
        await _responder_e_transferir(
            credentials=credentials, session_id=session_id, texto=TESTO_AUDIO_NON_TRASCRITTO,
            motivo=resumo_atendimento.MOTIVO_AUDIO_NON_TRASCRITTO, ultima_mensagem=aggregated_text,
            evento="audio_nao_transcrito", trace=trace,
        )
        return
    if aggregated_text.strip() == MARCADOR_NAO_SUPORTADA:
        await _responder_e_transferir(
            credentials=credentials, session_id=session_id, texto=TESTO_NON_SUPPORTATO,
            motivo=resumo_atendimento.MOTIVO_MESSAGGIO_NON_SUPPORTATO, ultima_mensagem=aggregated_text,
            evento="mensagem_nao_suportada", trace=trace,
        )
        return

    tool_context = agent_tools.ToolContext(session_id=session_id, credentials=credentials)

    async def _rodar():
        return await llm.run_turn(
            history=history,
            user_message=aggregated_text,
            llm_settings=llm_settings,
            tool_context=tool_context,
            avisos=avisos,
            dados_confirmados=dados_confirmados,
            llm_settings_fallback=llm_settings_fallback,
        )

    try:
        result = await _rodar()
    except Exception:
        # Uma segunda tentativa resolve a maior parte das falhas de rede/provedor.
        logger.warning("1a tentativa falhou na sessao %s - tentando de novo.", session_id, exc_info=True)
        tool_context = agent_tools.ToolContext(session_id=session_id, credentials=credentials)
        try:
            result = await _rodar()
        except Exception:
            logger.exception("Falha ao chamar o modelo para sessao %s (2 tentativas)", session_id)
            async with async_session() as db:
                ja_falhou_antes = await repository.ultimo_evento_foi(db, session_id=session_id, type_="llm_call_failed")
                await repository.save_event(db, session_id=session_id, type_="llm_call_failed", payload={})
                await db.commit()
            if ja_falhou_antes:
                # Segunda falha seguida: operador com nota, em vez de repetir o pedido de desculpas.
                await _responder_e_transferir(
                    credentials=credentials, session_id=session_id, texto=TESTO_PASSAGGIO,
                    motivo=resumo_atendimento.MOTIVO_SISTEMA, ultima_mensagem=aggregated_text,
                    evento="transferido_por_falha_do_sistema", trace=trace,
                )
                return
            async with async_session() as db:
                await repository.save_message(db, session_id=session_id, role="assistant", content=FALLBACK_REPLY)
                await db.commit()
            trace.marcar(observabilidade.RESPONDIDO_COM_FALLBACK, "o motor falhou duas vezes seguidas")
            with trace.etapa("envio", tipo="fallback"):
                await helena_client.send_canned_text(credentials=credentials, session_id=session_id, text=FALLBACK_REPLY)
            trace.resposta = FALLBACK_REPLY
            return

    reply_text, transferir = _texto_da_passagem(result, tool_context, aggregated_text)
    if not result.grounding_valido:
        trace.avisar(f"verificacao barrou a 1a redacao: {result.grounding_motivo}")
    if result.fallback_usado:
        trace.avisar(f"modelo reserva usado: {result.modelo_usado}")

    async with async_session() as db:
        conversation = await db.get(Conversation, session_id)
        assistant_message = await repository.save_message(
            db,
            session_id=session_id,
            role="assistant",
            content=reply_text,
            model=result.modelo_usado or f"{llm_settings.provider}:{llm_settings.model}",
            input_tokens=result.usage_cost.input_tokens,
            output_tokens=result.usage_cost.output_tokens,
            cache_read_tokens=result.usage_cost.cache_read_tokens,
            cache_write_tokens=result.usage_cost.cache_write_tokens,
            cost_usd=result.usage_cost.cost_usd,
        )
        if result.fallback_usado:
            await repository.save_event(db, session_id=session_id, type_="llm_fallback_usado", payload={"modelo": result.modelo_usado})
        if result.motivos_retry:
            await repository.save_event(db, session_id=session_id, type_="resposta_refeita", payload={"motivos": result.motivos_retry})
        # O que o motor entendeu e decidiu, para o painel (mesmo conteudo que vai ao Langfuse).
        if tool_context.trace:
            await repository.save_event(db, session_id=session_id, type_="decisao_do_turno", payload=tool_context.trace)
        # Atribuicao nova (nao mutacao): o SQLAlchemy so marca o JSONB como sujo assim.
        if conversation is not None and tool_context.slots != (conversation.slots or {}):
            conversation.slots = dict(tool_context.slots)
        await db.commit()

    # Envio fora da transacao de banco. Audio so quando o cliente mandou audio e a resposta e
    # curta (regra do node "Responder por Audio?1" do n8n).
    enviado = False
    if audio_replies_enabled and context.get("message_type") == "AUDIO" and tts.deve_responder_em_audio(reply_text):
        try:
            with trace.etapa("envio", tipo="audio"):
                audio_bytes, tts_cost_usd = await tts.synthesize(reply_text, model_id=tts_model_id)
                await helena_client.send_audio_reply(credentials=credentials, session_id=session_id, audio_bytes=audio_bytes)
            enviado = True
            trace.registrar_voz(f"gemini_tts:{tts_model_id}", tts_cost_usd, len(reply_text))
            async with async_session() as db:
                await repository.set_message_audio_cost(
                    db, message_id=assistant_message.id, cost_usd=tts_cost_usd, provider=f"gemini_tts:{tts_model_id}",
                )
                await db.commit()
        except Exception:
            logger.exception("Falha ao responder em audio na sessao %s - enviando texto", session_id)
            trace.avisar("a resposta em audio falhou; o cliente recebeu o texto")
    if not enviado:
        try:
            with trace.etapa("envio", tipo="texto"):
                await helena_client.send_canned_text(credentials=credentials, session_id=session_id, text=reply_text)
        except Exception:
            # O cliente NAO recebeu a resposta. Sem este desfecho o atendimento aparecia como
            # bem-sucedido no painel e no trace, e ninguem descobria o "ha dato errore in chat".
            trace.marcar(observabilidade.FALHA_ENVIO, "a Helena nao aceitou a resposta")
            raise
    trace.resposta = reply_text
    trace.marcar(observabilidade.RESPONDIDO, tool_context.motivo_transferencia if transferir else None)

    # Transferencia DEPOIS da resposta: o cliente le a mensagem de passagem, e so entao o
    # atendimento muda de equipe (nota interna antes do PUT /transfer - ordem do prompt).
    if transferir:
        with trace.etapa("transferencia", motivo=tool_context.motivo_transferencia):
            transferido = await agent_tools.executar_transferencia(tool_context)
        if not transferido:
            # A Helena responde 200 e nao transfere em alguns casos (bug conhecido da
            # plataforma): o lead ficaria preso na fila da IA sem ninguem perceber.
            trace.marcar(observabilidade.FALHA_TRANSFERENCIA, tool_context.motivo_transferencia)
        async with async_session() as db:
            conversation = await db.get(Conversation, session_id)
            if transferido and conversation is not None:
                await repository.mark_transferred(db, conversation)
            await repository.save_event(
                db, session_id=session_id,
                type_="transferido" if transferido else "transfer_failed",
                payload={"motivo": tool_context.motivo_transferencia},
            )
            await db.commit()


def _texto_da_passagem(result, tool_context: agent_tools.ToolContext, mensagem: str) -> tuple[str, bool]:
    """(texto_final, transferir_agora).

    No motor v2 quem decide a passagem ao operador e' a politica (app/politica.py), dentro do
    turno, e a verificacao (app/response_grounding.py) ja garante que o texto avisa o cliente
    - entao aqui nao ha mais o que conferir. Restam duas redes de seguranca:

    - texto vazio (provedor devolvendo string vazia depois de uma resposta longa): sai o testo
      approvato do motivo, nunca silencio;
    - foto recebida sem a politica ter transferido: o Matteo nao le imagens, entao a foto
      SEMPRE vai para o operador (regra "FOTO RICEVUTA" do prompt).
    """
    texto = (result.final_text or "").strip()
    transferir = result.transferido

    if not transferir and MARCADOR_FOTO in mensagem:
        logger.warning("Sessao %s: foto recebida sem transferencia do motor - transferindo pelo servidor.", tool_context.session_id)
        tool_context.transferido = True
        tool_context.motivo_transferencia = tool_context.motivo_transferencia or "foto_misura"
        transferir = True
        if not texto:
            texto = TESTO_FOTO_RICEVUTA

    if not texto:
        if transferir:
            texto = TESTO_FOTO_RICEVUTA if MARCADOR_FOTO in mensagem else (
                TESTO_DATI_COMPLETI if not agent_tools.campos_faltando(tool_context.slots) else TESTO_PASSAGGIO
            )
        else:
            texto = FALLBACK_REPLY
    return texto, transferir
