"""Configuracao do agente gerenciada em runtime (via /dashboard/settings), com as
chaves de API criptografadas no banco - nao em texto puro. As chaves do .env
(app/config.py) so servem de fallback de bootstrap, antes de qualquer
configuracao ter sido salva pelo painel.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import ENVIRONMENTS, EnvironmentCredentials, get_settings
from app.models import AppSettings, KnowledgeEntry

logger = logging.getLogger(__name__)
settings = get_settings()

_fernet = Fernet(settings.settings_encryption_key.encode())

PROVIDERS = ("gemini", "openai", "deepseek")

_FALLBACK_KEYS = {
    "openai": settings.openai_api_key,
    "gemini": settings.gemini_api_key,
    "deepseek": settings.deepseek_api_key,
}

# Prompt padrao do painel = a PERSONA do motor v2 (quem o Matteo e' e como ele escreve).
# As regras de fluxo nao estao mais no prompt: moram em app/politica.py. O prompt do motor
# antigo mandava chamar criarResumo/transferirAtendimento; se ele ainda estiver salvo no
# banco, app/redacao.py detecta e usa a persona do repositorio - ver `redacao.persona_ativa`.
_DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "persona.md"


def _encrypt_keys(keys: dict[str, str]) -> str:
    return _fernet.encrypt(json.dumps(keys).encode()).decode()


def _decrypt_keys(token: str | None) -> dict[str, str]:
    if not token:
        return {}
    try:
        return json.loads(_fernet.decrypt(token.encode()).decode())
    except (InvalidToken, ValueError, json.JSONDecodeError):
        logger.exception("Nao foi possivel descriptografar as chaves salvas - verifique SETTINGS_ENCRYPTION_KEY")
        return {}


@dataclass
class ActiveLLMSettings:
    provider: str
    model: str
    api_key: str | None
    system_prompt: str


async def _get_or_create_row(db: AsyncSession) -> AppSettings:
    row = await db.get(AppSettings, 1)
    if row is None:
        row = AppSettings(
            id=1,
            active_provider=settings.default_llm_provider,
            active_model=settings.default_llm_model,
        )
        db.add(row)
        await db.flush()
    return row


async def get_active_llm_settings(db: AsyncSession) -> ActiveLLMSettings:
    row = await db.get(AppSettings, 1)

    if row is None:
        provider = settings.default_llm_provider
        model = settings.default_llm_model
        keys: dict[str, str] = {}
        prompt = None
    else:
        provider = row.active_provider
        model = row.active_model
        keys = _decrypt_keys(row.provider_api_keys_encrypted)
        prompt = row.system_prompt_static

    api_key = keys.get(provider) or _FALLBACK_KEYS.get(provider)
    system_prompt = prompt or _DEFAULT_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

    return ActiveLLMSettings(provider=provider, model=model, api_key=api_key, system_prompt=system_prompt)


async def get_llm_settings_for(db: AsyncSession, *, provider: str, model: str) -> ActiveLLMSettings:
    """Como `get_active_llm_settings`, mas para um provider/model explicito (nao
    necessariamente o "ativo" agora) - usado pelo Teste Rapido, que fixa o provider/model no
    inicio do teste (`TestRun.llm_provider`/`llm_model`) e continua usando o mesmo em todos os
    turnos, mesmo que o usuario troque o modelo ativo do agente no meio do teste."""
    row = await db.get(AppSettings, 1)
    keys = _decrypt_keys(row.provider_api_keys_encrypted) if row else {}
    prompt = row.system_prompt_static if row else None

    api_key = keys.get(provider) or _FALLBACK_KEYS.get(provider)
    system_prompt = prompt or _DEFAULT_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

    return ActiveLLMSettings(provider=provider, model=model, api_key=api_key, system_prompt=system_prompt)


async def get_active_tts_model(db: AsyncSession) -> str:
    """Modelo de voz (Gemini TTS) ativo agora - so cai no default do .env
    (GEMINI_TTS_MODEL) se nada tiver sido salvo ainda no banco."""
    row = await db.get(AppSettings, 1)
    return (row.tts_model_id if row else None) or settings.gemini_tts_model


async def get_audio_replies_enabled(db: AsyncSession) -> bool:
    """Se False, o agente so responde em texto, mesmo recebendo audio do cliente - a
    transcricao do audio recebido (entendimento) nao e afetada por esta flag."""
    row = await db.get(AppSettings, 1)
    return row.audio_replies_enabled if row else True


async def get_configured_key_status(db: AsyncSession) -> dict[str, bool]:
    """Quais provedores ja tem chave configurada (no banco OU via fallback do .env) -
    usado so para exibir no formulario, nunca o valor em si."""
    row = await db.get(AppSettings, 1)
    keys = _decrypt_keys(row.provider_api_keys_encrypted) if row else {}
    return {p: bool(keys.get(p) or _FALLBACK_KEYS.get(p)) for p in PROVIDERS}


async def get_raw_settings(db: AsyncSession) -> tuple[str, str, str | None, str, bool]:
    """Retorna (provider, model, prompt_customizado_ou_None, tts_model, audio_replies_enabled) -
    usado pra pre-preencher o formulario."""
    row = await db.get(AppSettings, 1)
    if row is None:
        return settings.default_llm_provider, settings.default_llm_model, None, settings.gemini_tts_model, True
    return (
        row.active_provider,
        row.active_model,
        row.system_prompt_static,
        row.tts_model_id or settings.gemini_tts_model,
        row.audio_replies_enabled,
    )


async def update_settings(
    db: AsyncSession,
    *,
    provider: str | None = None,
    model: str | None = None,
    new_keys: dict[str, str] | None = None,
    clear_keys: list[str] | None = None,
    system_prompt: str | None = None,
    reset_prompt_to_default: bool = False,
    tts_model: str | None = None,
    audio_replies_enabled: bool | None = None,
) -> AppSettings:
    row = await _get_or_create_row(db)

    if provider:
        row.active_provider = provider
    if model:
        row.active_model = model
    if tts_model:
        row.tts_model_id = tts_model
    if audio_replies_enabled is not None:
        row.audio_replies_enabled = audio_replies_enabled

    if new_keys or clear_keys:
        current = _decrypt_keys(row.provider_api_keys_encrypted)
        for key, value in (new_keys or {}).items():
            if value:  # campo em branco = "nao alterar"
                current[key] = value
        for key in clear_keys or []:
            current.pop(key, None)
        row.provider_api_keys_encrypted = _encrypt_keys(current)

    if reset_prompt_to_default:
        row.system_prompt_static = None
    elif system_prompt is not None:
        row.system_prompt_static = system_prompt

    await db.flush()
    return row


def default_system_prompt() -> str:
    return _DEFAULT_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def get_integration_status() -> dict[str, bool]:
    """Status (configurado/nao) das integracoes que so existem no .env (sem override pelo
    painel) - so pra dar visibilidade visual em /dashboard/settings sem expor os valores. Os
    tokens/IDs da Helena NAO entram aqui - tem card proprio, editavel, em
    get_helena_config_effective."""
    return {
        # Transcricao (app/stt.py) e voz (app/tts.py) usam a GEMINI_API_KEY do .env.
        "gemini_audio": bool(settings.gemini_api_key),
    }


HELENA_FIELDS = (
    "message_token",
    "tag_token",
    "file_token",
    "panel_id",
    "step_id_novo_lead",
    "step_id_em_atendimento",
    "department_id",
)


def _helena_env_defaults(environment: str) -> dict[str, str]:
    prefix = "helena_prod_" if environment == "producao" else "helena_test_"
    return {field: getattr(settings, f"{prefix}{field}") or "" for field in HELENA_FIELDS}


def _decrypt_helena_config(token: str | None) -> dict[str, dict[str, str]]:
    if not token:
        return {}
    try:
        return json.loads(_fernet.decrypt(token.encode()).decode())
    except (InvalidToken, ValueError, json.JSONDecodeError):
        logger.exception("Nao foi possivel descriptografar a config da Helena salva - verifique SETTINGS_ENCRYPTION_KEY")
        return {}


def _encrypt_helena_config(config: dict[str, dict[str, str]]) -> str:
    return _fernet.encrypt(json.dumps(config).encode()).decode()


async def get_helena_config_effective(db: AsyncSession) -> dict[str, dict[str, str]]:
    """Retorna `{"producao": {...7 campos...}, "teste": {...}}` com os valores efetivos (o que
    foi salvo em /dashboard/settings sobrepondo o .env campo a campo) - usado tanto para
    resolver as credenciais de uma sessao (get_environment_credentials) quanto para
    pre-preencher o formulario de configuracoes."""
    row = await db.get(AppSettings, 1)
    saved = _decrypt_helena_config(row.helena_credentials_encrypted) if row else {}
    return {
        env: {
            field: (saved.get(env, {}).get(field) or "").strip() or _helena_env_defaults(env)[field]
            for field in HELENA_FIELDS
        }
        for env in ENVIRONMENTS
    }


async def get_environment_credentials(db: AsyncSession, environment: str) -> EnvironmentCredentials:
    if environment not in ENVIRONMENTS:
        raise ValueError(f"Ambiente desconhecido: {environment!r} (esperado 'teste' ou 'producao')")

    effective = (await get_helena_config_effective(db))[environment]
    if not effective["message_token"]:
        raise RuntimeError(
            f"Token de mensagens da Helena ({environment}) nao configurado - preencha em "
            "/dashboard/settings ou no .env do servidor"
        )

    return EnvironmentCredentials(
        environment=environment,
        message_token=effective["message_token"],
        tag_token=effective["tag_token"] or effective["message_token"],
        file_token=effective["file_token"] or effective["message_token"],
        panel_id=effective["panel_id"] or None,
        step_id_novo_lead=effective["step_id_novo_lead"] or None,
        step_id_em_atendimento=effective["step_id_em_atendimento"] or None,
        department_id=effective["department_id"] or None,
    )


async def update_helena_config(
    db: AsyncSession, *, producao: dict[str, str], teste: dict[str, str]
) -> AppSettings:
    row = await _get_or_create_row(db)
    row.helena_credentials_encrypted = _encrypt_helena_config(
        {
            "producao": {field: (producao.get(field) or "").strip() for field in HELENA_FIELDS},
            "teste": {field: (teste.get(field) or "").strip() for field in HELENA_FIELDS},
        }
    )
    await db.flush()
    return row


# --- Base de conhecimento (avisos: feriado/promocao) - ver /dashboard/base-conhecimento ---
# Diferente do prompt principal (system_prompt_static, editado em /dashboard/settings), aqui
# cada aviso e uma linha independente que so e ligada/desligada - nunca reescreve o prompt
# principal, entao nao ha risco de uma edicao de aviso alterar sutilmente uma regra de negocio
# do Matteo. Ver app/llm.py:_dynamic_system_block para onde os avisos ativos entram no prompt.


async def list_knowledge_entries(db: AsyncSession, limit: int = 100) -> list[KnowledgeEntry]:
    result = await db.execute(
        select(KnowledgeEntry).order_by(KnowledgeEntry.created_at.desc()).limit(limit)
    )
    return list(result.scalars().all())


async def get_active_knowledge_entries(db: AsyncSession) -> list[KnowledgeEntry]:
    """Avisos ativos agora, mais antigo primeiro (ordem natural de leitura no prompt) - usado
    por app/llm.py a cada turno."""
    result = await db.execute(
        select(KnowledgeEntry).where(KnowledgeEntry.active.is_(True)).order_by(KnowledgeEntry.created_at.asc())
    )
    return list(result.scalars().all())


async def create_knowledge_entry(db: AsyncSession, *, content: str, label: str | None) -> KnowledgeEntry:
    entry = KnowledgeEntry(content=content.strip(), label=(label or "").strip() or None, active=True)
    db.add(entry)
    await db.flush()
    return entry


async def set_knowledge_entry_active(db: AsyncSession, entry_id: int, active: bool) -> KnowledgeEntry:
    entry = await db.get(KnowledgeEntry, entry_id)
    if entry is None:
        raise ValueError(f"Aviso {entry_id} nao encontrado")
    entry.active = active
    await db.flush()
    return entry
