from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field


class HelenaSession(BaseModel):
    id: str
    createdAt: str | None = None
    departmentId: str | None = None
    userId: str | None = None
    number: str | None = None


class HelenaChannel(BaseModel):
    id: str | None = None
    key: str | None = None
    platform: str | None = None
    displayName: str | None = None


class HelenaContact(BaseModel):
    id: str
    name: str | None = None
    first_name: str | None = Field(default=None, alias="first-name")
    phonenumber: str | None = None
    display_phonenumber: str | None = Field(default=None, alias="display-phonenumber")
    email: str | None = None
    tags: Any = None
    metadata: dict = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class HelenaFile(BaseModel):
    id: str | None = None
    # A Helena manda a URL do arquivo como "publicUrl" (confirmado em producao, 2026-07-29,
    # via log de diagnostico de uma mensagem de audio real) - o campo "url" nunca veio
    # preenchido de fato, so existia por suposicao inicial. Aceita os dois nomes por
    # seguranca, caso algum outro tipo de mensagem (IMAGE/PDF/FILE) use "url".
    url: str | None = Field(default=None, validation_alias=AliasChoices("url", "publicUrl"))
    mimeType: str | None = None
    name: str | None = None

    model_config = {"populate_by_name": True}


class HelenaLastMessage(BaseModel):
    id: str | None = None
    createdAt: str | None = None
    type: Literal["TEXT", "AUDIO", "IMAGE", "PDF", "FILE"] | str = "TEXT"
    text: str | None = None
    fileId: str | None = None
    file: HelenaFile | None = None


class HelenaLastMessagesAggregated(BaseModel):
    text: str | None = None
    files: list[HelenaFile] = Field(default_factory=list)


class HelenaWebhookPayload(BaseModel):
    """Corpo exato que a plataforma Helena envia no POST do webhook (sem wrapper extra)."""

    companyId: str | None = None
    sessionId: str
    callbackUrl: str | None = None
    callbackId: str | None = None
    session: HelenaSession | None = None
    channel: HelenaChannel | None = None
    contact: HelenaContact
    lastContactMessage: str | None = None
    # Opcional: a Helena manda eventos de webhook sem lastMessage (ex: evento de sistema ao
    # trocar o bot ativo via comando #teste-XXX) - nao ha mensagem de cliente pra processar
    # nesses casos (ver app/main.py, que ja descarta o evento antes de chegar aqui).
    lastMessage: HelenaLastMessage | None = None
    lastMessagesAggregated: HelenaLastMessagesAggregated | None = None
    senderId: str | None = None

    @property
    def session_id_str(self) -> str:
        return self.sessionId

    @property
    def aggregated_text(self) -> str:
        if self.lastMessagesAggregated and self.lastMessagesAggregated.text:
            return self.lastMessagesAggregated.text
        return self.lastMessage.text or "" if self.lastMessage else ""

    @property
    def message_type(self) -> str:
        return self.lastMessage.type if self.lastMessage else "TEXT"
