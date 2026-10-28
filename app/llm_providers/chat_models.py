"""Chat models do LangChain - um ponto so onde cada provedor de LLM e' configurado.

Provedores do agente Gomme Fenice (os mesmos de antes da migracao para o motor v2): Gemini (principal,
decisao do cliente de 30/09/2026), OpenAI e DeepSeek (opcionais, trocaveis no painel). Os
ajustes que o codigo antigo aprendeu em producao continuam aqui, como parametro do LangChain:

- Gemini: raciocinio no minimo. Sem isso as respostas chegavam CORTADAS no meio da frase (os
  tokens de raciocinio contam no mesmo teto de saida - medido no agente anterior, 21/08/2026).
  A familia 2.x aceita budget 0; a 3.x so aceita nivel ("minimal").
- OpenAI: a familia gpt-5.6 e' "raciocinadora" - reasoning_effort desligado.
- DeepSeek: `ChatDeepSeek`, e nao `ChatOpenAI` com base_url (o ChatOpenAI manda
  `max_completion_tokens`, que a API da DeepSeek nao conhece); "thinking mode" desligado.

Tambem mora aqui a classificacao de erro "provedor indisponivel" que aciona o modelo reserva.
"""

from __future__ import annotations

import copy

import httpx
import openai
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage

from app.llm_providers.base import MAX_OUTPUT_TOKENS
from app.pricing_llm import NormalizedUsage
from app.settings_service import ActiveLLMSettings


def _thinking_gemini(model: str) -> dict:
    if model.startswith(("gemini-3.", "gemini-3-")):
        return {"thinking_level": "minimal"}
    return {"thinking_budget": 0}


def criar_modelo(llm_settings: ActiveLLMSettings, *, max_tokens: int = MAX_OUTPUT_TOKENS) -> BaseChatModel:
    provider, model, key = llm_settings.provider, llm_settings.model, llm_settings.api_key
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model, google_api_key=key, max_output_tokens=max_tokens, **_thinking_gemini(model)
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        extra = {"reasoning_effort": "none"} if model.startswith("gpt-5.6") else {}
        return ChatOpenAI(model=model, api_key=key, max_tokens=max_tokens, **extra)
    if provider == "deepseek":
        from langchain_deepseek import ChatDeepSeek

        return ChatDeepSeek(
            model=model, api_key=key, max_tokens=max_tokens, extra_body={"thinking": {"type": "disabled"}}
        )
    raise ValueError(f"Provedor de LLM desconhecido: {provider}")


def system_message(provider: str, estatico: str, dinamico: str = "") -> SystemMessage:
    """O estatico vem primeiro: os tres provedores fazem cache automatico do prefixo."""
    return SystemMessage(content=f"{estatico}\n\n{dinamico}" if dinamico else estatico)


def schema_para(provider: str, schema: dict) -> dict:
    """O Gemini nao aceita `additionalProperties` no schema de funcao."""
    if provider != "gemini":
        return schema

    def limpar(no):
        if isinstance(no, dict):
            return {k: limpar(v) for k, v in no.items() if k != "additionalProperties"}
        if isinstance(no, list):
            return [limpar(v) for v in no]
        return no

    return limpar(copy.deepcopy(schema))


def uso(mensagem: AIMessage | None) -> NormalizedUsage:
    """`usage_metadata` do LangChain -> o formato de custo do projeto (app/pricing_llm.py).
    No LangChain `input_tokens` ja inclui o cache; no formato do projeto, nao - por isso a
    subtracao."""
    meta = getattr(mensagem, "usage_metadata", None) or {}
    detalhes = meta.get("input_token_details") or {}
    lido = detalhes.get("cache_read", 0) or 0
    escrito = detalhes.get("cache_creation", 0) or 0
    return NormalizedUsage(
        input_tokens=max((meta.get("input_tokens", 0) or 0) - lido - escrito, 0),
        output_tokens=meta.get("output_tokens", 0) or 0,
        cache_read_tokens=lido,
        cache_write_tokens=escrito,
    )


# Erros que significam "este provedor/modelo esta indisponivel agora" - e nao "o pedido esta
# errado". So estes acionam o reserva: um 400 de payload invalido aconteceria de novo no outro
# modelo. 404 entra porque um modelo descontinuado nao pode derrubar o atendimento enquanto
# alguem troca no painel.
_ERROS_OPENAI = (
    openai.RateLimitError,
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.NotFoundError,
    openai.APIConnectionError,
    openai.InternalServerError,
)
_CODIGOS_INDISPONIVEL = {401, 403, 404, 408, 429, 500, 502, 503, 504}


def erro_de_provedor(erro: BaseException) -> bool:
    """Percorre a cadeia de causas (o LangChain embrulha o erro do SDK)."""
    atual: BaseException | None = erro
    vistos = 0
    while atual is not None and vistos < 6:
        if isinstance(atual, (*_ERROS_OPENAI, httpx.TransportError, TimeoutError)):
            return True
        codigo = getattr(atual, "code", None) or getattr(atual, "status_code", None)
        if type(atual).__module__.startswith("google.") and codigo in _CODIGOS_INDISPONIVEL:
            return True
        atual = atual.__cause__ or atual.__context__
        vistos += 1
    return False
