import asyncio
import logging
import re
import secrets
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.models import Conversation, Event, Message, TestRun
from app.helena_client import list_channels, send_as_customer, send_canned_text
from app.persona_llm import generate_persona_message, generate_persona_prompt
from app.pricing_llm import MODEL_CATALOG, PROVIDER_LABELS, calculate_cost
from app.pricing_tts import TTS_MODEL_CATALOG
from app.repository import STAGE_TRANSFERIDO, get_recent_history, save_message
from app.settings_service import (
    PROVIDERS,
    create_knowledge_entry,
    default_system_prompt,
    get_active_llm_settings,
    get_configured_key_status,
    get_environment_credentials,
    get_helena_config_effective,
    get_integration_status,
    get_llm_settings_for,
    get_raw_settings,
    list_knowledge_entries,
    set_knowledge_entry_active,
    update_helena_config,
    update_settings,
)

TERMINAL_STAGES = (STAGE_TRANSFERIDO,)
_TESTE_RAPIDO_ENVIRONMENT = "teste"
# Tempo de espera apos enviar o codigo de ativacao ("#teste-XXX") antes de mandar a 1a fala da
# persona - da tempo da automacao da propria Helena responder "Teste do agente iniciado!"
# primeiro. Sem essa espera, a fala da persona as vezes chega ANTES dessa confirmacao no
# WhatsApp real (visto ao vivo, 2026-08-07 - print do cliente mostrando a ordem trocada).
_ATIVACAO_TESTE_DELAY_SECONDS = 3.0
# Confirmado ao vivo (2026-07-29) como o unico par estavel entre os canais da conta de teste
# compartilhada - outros numeros na Helena podem estar com a instancia Z-API desconectada e
# a mensagem fica presa em QUEUED sem nunca chegar ao webhook. Pre-selecionados por padrao no
# formulario (usuario ainda pode trocar, mas evita cair num canal quebrado por engano).
DEFAULT_SENDER_PHONE = "+55|11900000001"
DEFAULT_RECEIVER_PHONE = "+55|1130000002"


def _digits(phone: str | None) -> str:
    return re.sub(r"\D", "", phone or "")

settings = get_settings()
security = HTTPBasic()

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

QUALITY_EVENT_TYPES = (
    "schema_validation_failed",
    "empty_response",
    "llm_call_failed",
    "transfer_failed",
    "fallback_question_appended",
)


def require_auth(credentials: HTTPBasicCredentials = Depends(security)) -> None:
    valid_user = secrets.compare_digest(credentials.username, settings.dashboard_username)
    valid_pass = secrets.compare_digest(credentials.password, settings.dashboard_password)
    if not (valid_user and valid_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciais invalidas",
            headers={"WWW-Authenticate": "Basic"},
        )


@router.get("/", response_class=HTMLResponse)
async def inbox(request: Request, db: AsyncSession = Depends(get_session), _: None = Depends(require_auth)):
    conversations = (
        await db.execute(select(Conversation).order_by(Conversation.updated_at.desc()).limit(100))
    ).scalars().all()

    last_messages: dict[str, Message] = {}
    quality_flags: dict[str, int] = {}
    for conversation in conversations:
        last = (
            await db.execute(
                select(Message)
                .where(Message.session_id == conversation.session_id)
                .order_by(Message.created_at.desc())
                .limit(1)
            )
        ).scalars().first()
        if last:
            last_messages[conversation.session_id] = last

        flags_count = (
            await db.execute(
                select(func.count(Event.id)).where(
                    Event.session_id == conversation.session_id,
                    Event.type.in_(QUALITY_EVENT_TYPES),
                )
            )
        ).scalar_one()
        quality_flags[conversation.session_id] = flags_count

    return templates.TemplateResponse(
        "inbox.html",
        {
            "request": request,
            "conversations": conversations,
            "last_messages": last_messages,
            "quality_flags": quality_flags,
        },
    )


@router.get("/conversations/{session_id}", response_class=HTMLResponse)
async def conversation_detail(
    session_id: str,
    request: Request,
    db: AsyncSession = Depends(get_session),
    _: None = Depends(require_auth),
):
    conversation = await db.get(Conversation, session_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversa nao encontrada")

    messages = (
        await db.execute(
            select(Message).where(Message.session_id == session_id).order_by(Message.created_at.asc())
        )
    ).scalars().all()

    events = (
        await db.execute(select(Event).where(Event.session_id == session_id).order_by(Event.created_at.asc()))
    ).scalars().all()

    total_cost = sum(m.cost_usd + m.audio_cost_usd for m in messages)

    return templates.TemplateResponse(
        "conversation.html",
        {
            "request": request,
            "conversation": conversation,
            "messages": messages,
            "events": events,
            "total_cost": total_cost,
            "quality_event_types": QUALITY_EVENT_TYPES,
        },
    )


@router.post("/conversations/{session_id}/handoff")
async def take_over(
    session_id: str, db: AsyncSession = Depends(get_session), _: None = Depends(require_auth)
):
    conversation = await db.get(Conversation, session_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversa nao encontrada")
    conversation.paused_by_human = True
    db.add(Event(session_id=session_id, type="handoff_taken", payload={}))
    await db.commit()
    return RedirectResponse(url=f"/dashboard/conversations/{session_id}", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/conversations/{session_id}/release")
async def release(
    session_id: str, db: AsyncSession = Depends(get_session), _: None = Depends(require_auth)
):
    conversation = await db.get(Conversation, session_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversa nao encontrada")
    conversation.paused_by_human = False
    db.add(Event(session_id=session_id, type="handoff_released", payload={}))
    await db.commit()
    return RedirectResponse(url=f"/dashboard/conversations/{session_id}", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/conversations/{session_id}/send")
async def send_manual_message(
    session_id: str,
    text: str = Form(...),
    db: AsyncSession = Depends(get_session),
    _: None = Depends(require_auth),
):
    conversation = await db.get(Conversation, session_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversa nao encontrada")

    credentials = await get_environment_credentials(db, conversation.environment)
    await send_canned_text(credentials=credentials, session_id=session_id, text=text)
    await save_message(db, session_id=session_id, role="human_agent", content=text)
    await db.commit()
    return RedirectResponse(url=f"/dashboard/conversations/{session_id}", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/costs", response_class=HTMLResponse)
async def costs(request: Request, db: AsyncSession = Depends(get_session), _: None = Depends(require_auth)):
    by_day = (
        await db.execute(
            select(
                func.date(Message.created_at).label("day"),
                func.sum(Message.cost_usd).label("cost"),
                func.sum(Message.input_tokens).label("input_tokens"),
                func.sum(Message.output_tokens).label("output_tokens"),
                func.sum(Message.cache_read_tokens).label("cache_read_tokens"),
            )
            .group_by(func.date(Message.created_at))
            .order_by(func.date(Message.created_at).desc())
            .limit(30)
        )
    ).all()

    by_conversation = (
        await db.execute(
            select(Message.session_id, func.sum(Message.cost_usd).label("cost"))
            .group_by(Message.session_id)
            .order_by(func.sum(Message.cost_usd).desc())
            .limit(30)
        )
    ).all()

    # Custo por LLM nos atendimentos reais - Message.model ja vem gravado como
    # "provider:model" (ver app/pipeline.py) desde a resposta do agente de verdade.
    by_model_atendimentos = (
        await db.execute(
            select(
                Message.model,
                func.count(func.distinct(Message.session_id)).label("conversas"),
                func.sum(Message.cost_usd).label("cost"),
                func.sum(Message.input_tokens).label("input_tokens"),
                func.sum(Message.output_tokens).label("output_tokens"),
                func.sum(Message.cache_read_tokens).label("cache_read_tokens"),
            )
            .where(Message.model.is_not(None))
            .group_by(Message.model)
            .order_by(func.sum(Message.cost_usd).desc())
        )
    ).all()

    # Custo por LLM nos testes da aba "Teste Rapido" - llm_provider/llm_model sao fixados no
    # inicio de cada teste (ver TestRun.llm_provider/llm_model), nao o modelo "ativo" agora.
    by_model_testes = (
        await db.execute(
            select(
                TestRun.llm_provider,
                TestRun.llm_model,
                func.count(TestRun.id).label("testes"),
                func.sum(TestRun.turns_done).label("turnos"),
                func.sum(TestRun.total_cost_usd).label("cost"),
            )
            .where(TestRun.llm_provider != "")
            .group_by(TestRun.llm_provider, TestRun.llm_model)
            .order_by(func.sum(TestRun.total_cost_usd).desc())
        )
    ).all()

    # Custo de audio (transcricao Whisper nas mensagens do cliente + sintese ElevenLabs nas
    # respostas) - dimensao separada do custo de LLM acima, agrupada por provider em vez de
    # modelo (ver Message.audio_provider/audio_cost_usd, gravado em app/pipeline.py).
    by_audio_provider = (
        await db.execute(
            select(
                Message.audio_provider,
                func.count(Message.id).label("mensagens"),
                func.sum(Message.audio_cost_usd).label("cost"),
            )
            .where(Message.audio_provider.is_not(None))
            .group_by(Message.audio_provider)
            .order_by(func.sum(Message.audio_cost_usd).desc())
        )
    ).all()

    total_atendimentos_cost = sum((row.cost or 0) for row in by_model_atendimentos)
    total_testes_cost = sum((row.cost or 0) for row in by_model_testes)
    total_audio_cost = sum((row.cost or 0) for row in by_audio_provider)

    return templates.TemplateResponse(
        "costs.html",
        {
            "request": request,
            "by_day": by_day,
            "by_conversation": by_conversation,
            "by_model_atendimentos": by_model_atendimentos,
            "by_model_testes": by_model_testes,
            "by_audio_provider": by_audio_provider,
            "total_atendimentos_cost": total_atendimentos_cost,
            "total_testes_cost": total_testes_cost,
            "total_audio_cost": total_audio_cost,
        },
    )


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(
    request: Request, db: AsyncSession = Depends(get_session), _: None = Depends(require_auth)
):
    provider, model, custom_prompt, tts_model, audio_replies_enabled = await get_raw_settings(db)
    key_status = await get_configured_key_status(db)
    integration_status = get_integration_status()
    helena_config = await get_helena_config_effective(db)

    return templates.TemplateResponse(
        "settings.html",
        {
            "request": request,
            "active_choice": f"{provider}::{model}",
            "prompt_value": custom_prompt or default_system_prompt(),
            "using_custom_prompt": custom_prompt is not None,
            "key_status": key_status,
            "model_catalog": MODEL_CATALOG,
            "provider_labels": PROVIDER_LABELS,
            "providers": PROVIDERS,
            "tts_model_catalog": TTS_MODEL_CATALOG,
            "active_tts_model": tts_model,
            "audio_replies_enabled": audio_replies_enabled,
            "integration_status": integration_status,
            "helena_config": helena_config,
        },
    )


@router.post("/settings")
async def save_settings(
    model_choice: str = Form(...),
    tts_model: str = Form(...),
    audio_replies_enabled: str | None = Form(None),
    system_prompt: str = Form(""),
    reset_prompt: str | None = Form(None),
    openai_key: str = Form(""),
    gemini_key: str = Form(""),
    deepseek_key: str = Form(""),
    clear_openai: str | None = Form(None),
    clear_gemini: str | None = Form(None),
    clear_deepseek: str | None = Form(None),
    helena_producao_message_token: str = Form(""),
    helena_producao_tag_token: str = Form(""),
    helena_producao_file_token: str = Form(""),
    helena_producao_panel_id: str = Form(""),
    helena_producao_step_id_novo_lead: str = Form(""),
    helena_producao_step_id_em_atendimento: str = Form(""),
    helena_producao_department_id: str = Form(""),
    helena_teste_message_token: str = Form(""),
    helena_teste_tag_token: str = Form(""),
    helena_teste_file_token: str = Form(""),
    helena_teste_panel_id: str = Form(""),
    helena_teste_step_id_novo_lead: str = Form(""),
    helena_teste_step_id_em_atendimento: str = Form(""),
    helena_teste_department_id: str = Form(""),
    db: AsyncSession = Depends(get_session),
    _: None = Depends(require_auth),
):
    try:
        provider, model = model_choice.split("::", 1)
    except ValueError:
        raise HTTPException(status_code=400, detail="Selecao de modelo invalida")

    if provider not in MODEL_CATALOG or model not in MODEL_CATALOG[provider]:
        raise HTTPException(status_code=400, detail="Provedor/modelo invalido")

    if tts_model not in TTS_MODEL_CATALOG:
        raise HTTPException(status_code=400, detail="Modelo de voz invalido")

    new_keys = {
        "openai": openai_key.strip(),
        "gemini": gemini_key.strip(),
        "deepseek": deepseek_key.strip(),
    }
    clear_keys = [
        p
        for p, flag in (
            ("openai", clear_openai),
            ("gemini", clear_gemini),
            ("deepseek", clear_deepseek),
        )
        if flag
    ]

    await update_settings(
        db,
        provider=provider,
        model=model,
        new_keys=new_keys,
        clear_keys=clear_keys,
        system_prompt=None if reset_prompt else system_prompt,
        reset_prompt_to_default=bool(reset_prompt),
        tts_model=tts_model,
        audio_replies_enabled=bool(audio_replies_enabled),
    )

    helena_form_values = {
        "producao": {
            "message_token": helena_producao_message_token,
            "tag_token": helena_producao_tag_token,
            "file_token": helena_producao_file_token,
            "panel_id": helena_producao_panel_id,
            "step_id_novo_lead": helena_producao_step_id_novo_lead,
            "step_id_em_atendimento": helena_producao_step_id_em_atendimento,
            "department_id": helena_producao_department_id,
        },
        "teste": {
            "message_token": helena_teste_message_token,
            "tag_token": helena_teste_tag_token,
            "file_token": helena_teste_file_token,
            "panel_id": helena_teste_panel_id,
            "step_id_novo_lead": helena_teste_step_id_novo_lead,
            "step_id_em_atendimento": helena_teste_step_id_em_atendimento,
            "department_id": helena_teste_department_id,
        },
    }
    await update_helena_config(
        db, producao=helena_form_values["producao"], teste=helena_form_values["teste"]
    )

    await db.commit()
    return RedirectResponse(url="/dashboard/settings", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/base-conhecimento", response_class=HTMLResponse)
async def base_conhecimento_page(
    request: Request, db: AsyncSession = Depends(get_session), _: None = Depends(require_auth)
):
    entries = await list_knowledge_entries(db)
    await db.commit()

    return templates.TemplateResponse(
        "base_conhecimento.html",
        {"request": request, "entries": entries},
    )


@router.post("/base-conhecimento")
async def base_conhecimento_criar(
    content: str = Form(...),
    label: str = Form(""),
    db: AsyncSession = Depends(get_session),
    _: None = Depends(require_auth),
):
    if not content.strip():
        raise HTTPException(status_code=400, detail="O aviso nao pode ficar vazio")

    await create_knowledge_entry(db, content=content, label=label)
    await db.commit()
    return RedirectResponse(url="/dashboard/base-conhecimento", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/base-conhecimento/{entry_id}/toggle")
async def base_conhecimento_toggle(
    entry_id: int,
    active: str = Form(...),
    db: AsyncSession = Depends(get_session),
    _: None = Depends(require_auth),
):
    try:
        await set_knowledge_entry_active(db, entry_id, active == "1")
    except ValueError:
        raise HTTPException(status_code=404, detail="Aviso nao encontrado")
    await db.commit()
    return RedirectResponse(url="/dashboard/base-conhecimento", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/teste-rapido", response_class=HTMLResponse)
async def teste_rapido_form(
    request: Request, db: AsyncSession = Depends(get_session), _: None = Depends(require_auth)
):
    channels: list[dict] = []
    channels_error: str | None = None
    try:
        # Sem o token do sandbox (HELENA_TEST_*) a tela abre mesmo assim, com o aviso do que falta.
        credentials = await get_environment_credentials(db, _TESTE_RAPIDO_ENVIRONMENT)
        channels = await list_channels(credentials=credentials)
    except Exception as exc:
        channels_error = str(exc)

    runs = (
        await db.execute(select(TestRun).order_by(TestRun.created_at.desc()).limit(20))
    ).scalars().all()

    return templates.TemplateResponse(
        "teste_rapido.html",
        {
            "request": request,
            "channels": channels,
            "channels_error": channels_error,
            "runs": runs,
            "default_sender_phone": DEFAULT_SENDER_PHONE,
            "default_receiver_phone": DEFAULT_RECEIVER_PHONE,
        },
    )


@router.post("/teste-rapido/generate-persona", response_class=HTMLResponse)
async def teste_rapido_generate_persona(
    extra_instructions: str = Form(""),
    db: AsyncSession = Depends(get_session),
    _: None = Depends(require_auth),
):
    # Chamada exploratoria de antes do teste existir (nenhum TestRun criado ainda) - custo
    # unico e pequeno, nao atribuido a nenhum run (o rastreio de custo comeca no /start).
    llm_settings = await get_active_llm_settings(db)
    persona, _usage = await generate_persona_prompt(llm_settings, extra_instructions)
    return HTMLResponse(persona)


@router.post("/teste-rapido/start")
async def teste_rapido_start(
    sender_phone: str = Form(...),
    receiver_phone: str = Form(...),
    activation_code: str = Form(...),
    persona_prompt: str = Form(...),
    max_turns: int = Form(8),
    db: AsyncSession = Depends(get_session),
    _: None = Depends(require_auth),
):
    credentials = await get_environment_credentials(db, _TESTE_RAPIDO_ENVIRONMENT)
    llm_settings = await get_active_llm_settings(db)

    run = TestRun(
        sender_phone=sender_phone,
        receiver_phone=receiver_phone,
        activation_code=activation_code.strip().lstrip("#").removeprefix("teste-"),
        persona_prompt=persona_prompt,
        max_turns=max(1, max_turns),
        # Fixado agora e reaproveitado em todos os turnos do teste (ver teste_rapido_run) -
        # nao re-le o modelo "ativo" do agente a cada turno, pra custo/comportamento nao
        # mudarem no meio de um teste caso o usuario troque o modelo durante o teste.
        llm_provider=llm_settings.provider,
        llm_model=llm_settings.model,
    )
    db.add(run)
    await db.flush()

    # "#fim" primeiro: se o mesmo numero remetente ja tiver uma conversa antiga (encerrada ou
    # transferida) nesse canal, a Helena reaproveita a MESMA sessao em vez de abrir uma nova - o
    # agente so responderia com a mensagem enlatada de pos-transferencia (visto ao vivo,
    # 2026-07-29). "#fim" reseta isso antes do "#teste-XXX" reabrir do zero.
    await send_as_customer(
        credentials=credentials, from_phone=sender_phone, to_phone=receiver_phone, text="#fim"
    )
    await send_as_customer(
        credentials=credentials,
        from_phone=sender_phone,
        to_phone=receiver_phone,
        text=f"#teste-{run.activation_code}",
    )
    # Espera a Helena confirmar "Teste do agente iniciado!" antes de mandar a 1a fala da
    # persona - ver _ATIVACAO_TESTE_DELAY_SECONDS.
    await asyncio.sleep(_ATIVACAO_TESTE_DELAY_SECONDS)
    abertura, usage = await generate_persona_message(llm_settings, persona_prompt, [])
    await send_as_customer(
        credentials=credentials, from_phone=sender_phone, to_phone=receiver_phone, text=abertura
    )
    run.turns_done = 1
    run.total_cost_usd += calculate_cost(run.llm_provider, run.llm_model, usage).cost_usd

    await db.commit()
    return RedirectResponse(url=f"/dashboard/teste-rapido/{run.id}", status_code=status.HTTP_303_SEE_OTHER)


async def _resolve_target_conversation(db: AsyncSession, run: TestRun) -> Conversation | None:
    if run.target_session_id:
        return await db.get(Conversation, run.target_session_id)

    sender_digits = _digits(run.sender_phone)[-11:]
    candidates = (
        await db.execute(
            select(Conversation)
            .where(Conversation.created_at >= run.created_at)
            .order_by(Conversation.created_at.desc())
            .limit(50)
        )
    ).scalars().all()
    for conversation in candidates:
        if _digits(conversation.phone)[-11:] == sender_digits:
            run.target_session_id = conversation.session_id
            await db.flush()
            return conversation
    return None


@router.get("/teste-rapido/{run_id}", response_class=HTMLResponse)
async def teste_rapido_run(
    run_id: str,
    request: Request,
    db: AsyncSession = Depends(get_session),
    _: None = Depends(require_auth),
):
    run = await db.get(TestRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Teste nao encontrado")

    conversation = await _resolve_target_conversation(db, run)
    messages: list[Message] = []
    if conversation is not None:
        messages = (
            await db.execute(
                select(Message)
                .where(Message.session_id == conversation.session_id)
                .order_by(Message.created_at.asc())
            )
        ).scalars().all()

    turn_error: str | None = None
    if (
        run.status == "running"
        and conversation is not None
        and messages
        and messages[-1].role == "assistant"
        and run.turns_done < run.max_turns
        and conversation.stage not in TERMINAL_STAGES
    ):
        try:
            credentials = await get_environment_credentials(db, _TESTE_RAPIDO_ENVIRONMENT)
            llm_settings = await get_llm_settings_for(
                db, provider=run.llm_provider, model=run.llm_model
            )
            history = await get_recent_history(db, session_id=conversation.session_id)
            proxima_fala, usage = await generate_persona_message(
                llm_settings, run.persona_prompt, history
            )
            await send_as_customer(
                credentials=credentials,
                from_phone=run.sender_phone,
                to_phone=run.receiver_phone,
                text=proxima_fala,
            )
            run.turns_done += 1
            run.total_cost_usd += calculate_cost(run.llm_provider, run.llm_model, usage).cost_usd
        except Exception as exc:
            # Nunca deixa uma falha/demora da IA ou da Helena quebrar a tela - o turno so
            # avanca quando der certo; se falhar, o proximo polling do htmx tenta de novo
            # sozinho (nao trava, nao perde o teste).
            logging.getLogger(__name__).exception(
                "Falha ao avancar turno do Teste Rapido %s - tentando de novo no proximo polling", run.id
            )
            turn_error = str(exc)

    if conversation is not None and (
        conversation.stage in TERMINAL_STAGES or run.turns_done >= run.max_turns
    ):
        run.status = "finished"

    await db.commit()

    return templates.TemplateResponse(
        "teste_rapido_run.html",
        {
            "request": request,
            "run": run,
            "conversation": conversation,
            "messages": messages,
            "turn_error": turn_error,
        },
    )
