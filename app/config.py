from dataclasses import dataclass
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

ENVIRONMENTS = ("teste", "producao")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Chave unica do agente Gomme Fenice: Gemini faz a conversa, a transcricao do audio recebido e a
    # sintese da resposta em audio (decisao do cliente, 30/09/2026 - "a chave gemini que iremos
    # usar para tudo"). As chaves de LLM do .env sao so FALLBACK de bootstrap: depois de
    # configuradas em /dashboard/settings, as do banco (criptografadas) tem prioridade.
    gemini_api_key: str | None = None
    # Opcionais - so entram se alguem trocar o provedor da conversa no painel.
    openai_api_key: str = ""
    deepseek_api_key: str | None = None

    default_llm_provider: str = "gemini"
    default_llm_model: str = "gemini-2.5-flash"
    # Reserva automatica quando o modelo ativo falha por indisponibilidade (5xx, sobrecarga,
    # timeout) - ver app/llm.py. Pode ser o MESMO provedor com outro modelo: com uma chave so,
    # e o que protege o atendimento de um modelo sobrecarregado.
    fallback_llm_provider: str | None = "gemini"
    fallback_llm_model: str | None = "gemini-3.5-flash"

    # Chave usada para criptografar as chaves de API salvas no banco via /dashboard/settings.
    # Gere uma com: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    settings_encryption_key: str

    # Transcricao do audio recebido (app/stt.py) - Gemini nativo, audio como parte inline.
    gemini_stt_model: str = "gemini-2.5-flash"
    # Resposta em audio (app/tts.py) - Gemini TTS. Voz em /dashboard/settings pode trocar o
    # modelo; a voz fica aqui. "Charon" e masculina e neutra, combina com o Matteo.
    gemini_tts_model: str = "gemini-2.5-flash-preview-tts"
    gemini_tts_voice: str = "Charon"

    # Observabilidade (app/observabilidade.py). SEM as duas chaves o Langfuse fica desligado e
    # nada sai do processo - e' o padrao. O servidor precisa ser self-hosted: o trace leva a
    # fala do cliente (LGPD/GDPR). `langfuse_public_url` so e' preciso quando o endereco que o
    # navegador abre e' diferente do que o container usa (ex.: host interno no servidor).
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_url: str | None = None

    # Versao do agente, usada como tag dos traces (ex.: "gomme-fenice@2026-10-20").
    app_version: str = "dev"

    helena_base_url: str = "https://api.helena.run"
    # Endpoint usado so para encerrar/reativar o chatbot no atendimento (ver app/helena_client.py).
    helena_wts_base_url: str = "https://api.wts.chat"

    # Producao - numero real de WhatsApp da Gomme Fenice.
    helena_prod_message_token: str = ""
    helena_prod_tag_token: str = ""
    helena_prod_file_token: str = ""
    helena_prod_panel_id: str | None = None
    helena_prod_step_id_novo_lead: str | None = None
    helena_prod_step_id_em_atendimento: str | None = None
    # Equipe/departamento (core/v1/department) que recebe o atendimento apos a transferencia -
    # o mesmo `newDepartmentId` do node transferirAtendimento do fluxo n8n.
    helena_prod_department_id: str | None = None
    # Departamento em que o BOT trabalha. Serve para saber se a conversa AINDA e do Matteo
    # (ver app/atendimento_helena.py). So no .env, como todo id de integracao.
    helena_prod_bot_department_id: str | None = None

    # Teste - sandbox compartilhado da agencia (ambiente de testes padrao).
    helena_test_message_token: str | None = None
    helena_test_tag_token: str | None = None
    helena_test_file_token: str | None = None
    helena_test_panel_id: str | None = None
    helena_test_step_id_novo_lead: str | None = None
    helena_test_step_id_em_atendimento: str | None = None
    helena_test_department_id: str | None = None
    helena_test_bot_department_id: str | None = None

    database_url: str
    redis_url: str = "redis://localhost:6379/0"
    debounce_seconds: int = 5

    dashboard_username: str = "admin"
    dashboard_password: str = "changeme"


@lru_cache
def get_settings() -> Settings:
    return Settings()


@dataclass
class EnvironmentCredentials:
    environment: str
    message_token: str
    tag_token: str
    file_token: str
    panel_id: str | None
    step_id_novo_lead: str | None
    step_id_em_atendimento: str | None
    department_id: str | None


# A resolucao de credenciais (com override editavel em /dashboard/settings, banco tem
# prioridade sobre .env campo a campo) mora em app/settings_service.py:get_environment_credentials
# - precisa de acesso ao banco/Fernet, que este modulo (config.py) nao tem.
