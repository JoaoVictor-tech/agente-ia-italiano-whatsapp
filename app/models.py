import datetime
import uuid

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"

    session_id: Mapped[str] = mapped_column(String, primary_key=True)
    environment: Mapped[str] = mapped_column(String, nullable=False, default="producao")  # "teste" ou "producao"
    contact_id: Mapped[str] = mapped_column(String, nullable=False)
    phone: Mapped[str | None] = mapped_column(String, nullable=True)
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    # "ativo" | "transferido" (ver app/repository.py) - o Matteo conduz a coleta via prompt;
    # o unico estado deterministico relevante e se a sessao ja foi transferida para um
    # operador humano da Gomme Fenice.
    stage: Mapped[str] = mapped_column(String, nullable=False, default="ativo")
    slots: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    transferred_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    paused_by_human: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
        onupdate=lambda: datetime.datetime.now(datetime.timezone.utc),
    )

    messages: Mapped[list["Message"]] = relationship(back_populates="conversation", order_by="Message.created_at")
    events: Mapped[list["Event"]] = relationship(back_populates="conversation", order_by="Event.created_at")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("conversations.session_id"), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String, nullable=False)  # user | assistant | human_agent
    content: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str | None] = mapped_column(String, nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_read_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_write_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    # Custo de audio - transcricao (Gemini, mensagem role="user" que chegou como audio) OU
    # sintese (Gemini TTS, mensagem role="assistant" respondida em audio). Nunca os dois na
    # mesma linha (uma mensagem so tem uma direcao). Ver app/stt.py e app/tts.py.
    audio_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    audio_provider: Mapped[str | None] = mapped_column(String, nullable=True)  # "gemini_audio" | "gemini_tts:<modelo>"
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc), index=True
    )

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


class KnowledgeEntry(Base):
    """Um "aviso" da base de conhecimento (/dashboard/base-conhecimento) - feriado, promocao
    exclusiva, qualquer fato temporario que o Matteo deve saber sem mexer no prompt principal
    (app/prompts/persona.md, que fica intocado). Todos os avisos com `active=True` sao
    concatenados e injetados no briefing do turno (ver app/redacao.py:montar_bloco)
    a cada turno - religar/desligar um aviso e so um toggle, sem chamar LLM nenhuma pra
    reescrever nada. Nunca apagado de verdade (so desativado) - mantem o historico visivel."""

    __tablename__ = "knowledge_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Rotulo curto pra identificar o aviso na lista (ex: "Chiuso 15/08", "Promo invernali").
    # Se em branco, a tela mostra so o inicio do conteudo.
    label: Mapped[str | None] = mapped_column(String, nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc), index=True
    )


class AppSettings(Base):
    """Configuracao do agente editavel via /dashboard/settings - linha unica (id=1)."""

    __tablename__ = "app_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    active_provider: Mapped[str] = mapped_column(String, nullable=False, default="gemini")
    active_model: Mapped[str] = mapped_column(String, nullable=False, default="gemini-2.5-flash")
    # JSON {"gemini": "...", "openai": "...", "deepseek": "..."} criptografado
    # inteiro com Fernet (ver app/settings_service.py) - nunca fica em texto puro no banco.
    provider_api_keys_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Persona do Matteo (quem ele e' e como escreve; as regras de fluxo estao em
    # app/politica.py) - se None, usa o arquivo app/prompts/persona.md como padrao.
    system_prompt_static: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Modelo de voz ativo (Gemini TTS) usado na resposta em audio - se None, usa
    # GEMINI_TTS_MODEL do .env como padrao (ver app/settings_service.py).
    tts_model_id: Mapped[str | None] = mapped_column(String, nullable=True)
    # Se False, o agente nunca responde em audio (mesmo recebendo audio do cliente) - so
    # texto. A transcricao do audio recebido (entendimento) continua acontecendo sempre,
    # independente desta flag - ver app/pipeline.py. Comeca LIGADO, como no fluxo n8n (node
    # "Responder por Audio?1": cliente mandou audio e a resposta tem menos de 80 palavras).
    audio_replies_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    # JSON {"producao": {...7 campos...}, "teste": {...}} criptografado com Fernet (mesmo
    # padrao de provider_api_keys_encrypted) - tokens/IDs da plataforma Helena editaveis em
    # /dashboard/settings (ver app/settings_service.py:get_environment_credentials). Campo
    # vazio/ausente cai no .env (HELENA_PROD_*/HELENA_TEST_*) campo a campo.
    helena_credentials_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
        onupdate=lambda: datetime.datetime.now(datetime.timezone.utc),
    )


class TestRun(Base):
    """Uma execucao da tela "Teste Rapido" (dashboard) - simula uma conversa real via Helena
    (canal remetente finge ser o cliente, canal receptor e o proprio agente) guiada por uma
    persona gerada por IA. `target_session_id` so e preenchido quando a conversa correspondente
    aparecer em `conversations` (correlacionada por telefone - ver app/dashboard/routes.py)."""

    __tablename__ = "test_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    sender_phone: Mapped[str] = mapped_column(String, nullable=False)
    receiver_phone: Mapped[str] = mapped_column(String, nullable=False)
    activation_code: Mapped[str] = mapped_column(String, nullable=False)
    persona_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    target_session_id: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="running")  # running | finished
    turns_done: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_turns: Mapped[int] = mapped_column(Integer, nullable=False, default=8)
    # Provider/modelo fixados no inicio do teste (nao re-lidos do "ativo" a cada turno) - pra
    # custo e comportamento nao mudarem no meio de um teste caso o usuario troque o modelo
    # ativo do agente enquanto o teste ainda esta rodando.
    llm_provider: Mapped[str] = mapped_column(String, nullable=False, default="")
    llm_model: Mapped[str] = mapped_column(String, nullable=False, default="")
    total_cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc)
    )


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("conversations.session_id"), nullable=False, index=True)
    type: Mapped[str] = mapped_column(String, nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc), index=True
    )

    conversation: Mapped["Conversation"] = relationship(back_populates="events")
