"""Chamadas de LLM "soltas" (texto livre, sem tool forcada) para a persona simulada da tela de
Teste Rapido do dashboard - diferente de app/llm.py, que sempre forca o schema
RESPONDER_INPUT_SCHEMA usado pelo agente real. Aqui so precisamos de texto puro: a proxima fala
do "cliente" simulado, ou um system-prompt de persona gerado a partir do prompt real do agente.

Reaproveita o mesmo provedor/modelo/chave configurados para o agente (ActiveLLMSettings), pra a
persona custar e se comportar de forma equivalente ao que o proprio agente usa.
"""

from __future__ import annotations

import asyncio

from google import genai
from google.genai import types
from openai import AsyncOpenAI

from app.llm import DEEPSEEK_BASE_URL
from app.pricing_llm import NormalizedUsage
from app.settings_service import ActiveLLMSettings

FALLBACK_USER_MESSAGE = "Comece a conversa."

# Falha rapido em vez de deixar a chamada travar indefinidamente (ja observado ao vivo: uma
# chamada ao Gemini demorou o suficiente para estourar o timeout do proxy nginx, mesmo com o
# teste terminando certo por baixo - quem chama (app/dashboard/routes.py) trata o timeout
# deixando o turno para a proxima tentativa de polling, em vez de quebrar a pagina).
_COMPLETION_TIMEOUT_SECONDS = 45.0

_PERSONA_MESSAGE_SYSTEM_SUFFIX = (
    "\n\nRegra de formato desta simulacao: responda APENAS com a proxima mensagem que essa "
    "pessoa mandaria pelo WhatsApp - 1 a 2 frases curtas e naturais, sem aspas, sem prefixo "
    "como 'Cliente:', sem nunca revelar que isso e um teste ou que voce e uma IA."
)

_PERSONA_PROMPT_SYSTEM = (
    "Voce ajuda a criar personas de clientes para testar um agente de atendimento via WhatsApp. "
    "Dado o prompt real do agente (persona, negocio, regras) informado pelo usuario, escreva um "
    "system-prompt pronto para simular UM cliente especifico testando esse atendimento - inclua "
    "nome, personalidade/tom, e um objetivo concreto (ex: medida do pneu, quantos pneus, estacao, "
    "se sabe ou nao a medida, se pede o preco logo de cara). O cliente escreve no MESMO idioma do "
    "agente (italiano, no caso da Gomme Fenice). Responda SOMENTE com o texto do "
    "system-prompt da persona (sem comentarios extras, sem markdown, sem aspas envolvendo tudo)."
)


async def _complete_text(
    llm_settings: ActiveLLMSettings,
    system_prompt: str,
    history: list[dict],
    user_message: str | None,
) -> tuple[str, NormalizedUsage]:
    return await asyncio.wait_for(
        _complete_text_inner(llm_settings, system_prompt, history, user_message),
        timeout=_COMPLETION_TIMEOUT_SECONDS,
    )


async def _complete_text_inner(
    llm_settings: ActiveLLMSettings,
    system_prompt: str,
    history: list[dict],
    user_message: str | None,
) -> tuple[str, NormalizedUsage]:
    messages = [*history]
    if user_message is not None:
        messages.append({"role": "user", "content": user_message})
    if not messages:
        messages.append({"role": "user", "content": FALLBACK_USER_MESSAGE})

    provider = llm_settings.provider
    model = llm_settings.model
    api_key = llm_settings.api_key

    if provider == "gemini":
        client = genai.Client(api_key=api_key)
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part(text=m["content"])],
            )
            for m in messages
        ]
        response = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )
        text = (response.text or "").strip()
        usage_meta = response.usage_metadata
        usage = NormalizedUsage(
            input_tokens=getattr(usage_meta, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage_meta, "candidates_token_count", 0) or 0,
            cache_read_tokens=getattr(usage_meta, "cached_content_token_count", 0) or 0,
        )
        return text, usage

    # openai / deepseek - mesma API compativel, so muda o base_url
    base_url = DEEPSEEK_BASE_URL if provider == "deepseek" else None
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    chat_messages = [{"role": "system", "content": system_prompt}, *messages]
    extra_kwargs: dict = {}
    if provider == "openai" and model.startswith("gpt-5"):
        # gpt-5.x nao aceita mais "max_tokens" (erro 400 "Unsupported parameter").
        extra_kwargs["max_completion_tokens"] = 300
        if model.startswith("gpt-5.6"):
            # familia "raciocinadora" - desligamos por consistencia/custo (mesmo motivo
            # registrado em app/llm_providers/chat_models.py).
            extra_kwargs["reasoning_effort"] = "none"
    else:
        extra_kwargs["max_tokens"] = 300
    response = await client.chat.completions.create(model=model, messages=chat_messages, **extra_kwargs)
    text = (response.choices[0].message.content or "").strip()
    resp_usage = response.usage
    details = getattr(resp_usage, "prompt_tokens_details", None)
    cache_read = getattr(details, "cached_tokens", 0) or 0 if details is not None else 0
    if not cache_read:
        cache_read = getattr(resp_usage, "prompt_cache_hit_tokens", 0) or 0
    usage = NormalizedUsage(
        input_tokens=getattr(resp_usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(resp_usage, "completion_tokens", 0) or 0,
        cache_read_tokens=cache_read,
    )
    return text, usage


async def generate_persona_message(
    llm_settings: ActiveLLMSettings, persona_prompt: str, history: list[dict]
) -> tuple[str, NormalizedUsage]:
    """`history` entra na perspectiva do agente real (role="user" pro cliente, "assistant" pro
    agente - mesmo formato que `app.repository.get_recent_history` devolve). Só aqui dentro os
    papeis sao invertidos antes de mandar pro LLM: quem esta gerando a fala e o CLIENTE, entao os
    proprios turnos do cliente precisam ir como "assistant"/"model" (a fala que "eu" - o LLM -
    gerei antes) e os do agente como "user" (a fala da outra parte, a quem "eu" respondo agora).
    Passar sem inverter faz o LLM se confundir e comecar a responder como se fosse o proprio
    agente (persona perguntando coisas tipicas de atendente)."""
    inverted_history = [
        {"role": "assistant" if m["role"] == "user" else "user", "content": m["content"]}
        for m in history
    ]
    system_prompt = f"{persona_prompt}{_PERSONA_MESSAGE_SYSTEM_SUFFIX}"
    return await _complete_text(llm_settings, system_prompt, inverted_history, None)


async def generate_persona_prompt(
    llm_settings: ActiveLLMSettings, extra_instructions: str
) -> tuple[str, NormalizedUsage]:
    """Usa `llm_settings.system_prompt` (o prompt REAL do agente, ja resolvido pelo
    settings_service - arquivo ou override do banco) como contexto pra gerar uma persona de
    cliente pronta para `generate_persona_message`."""
    user_message = (
        f"PROMPT REAL DO AGENTE A TESTAR:\n{llm_settings.system_prompt}\n\n"
        f"INSTRUCAO EXTRA (pode estar vazia): {extra_instructions.strip() or '(nenhuma)'}"
    )
    return await _complete_text(llm_settings, _PERSONA_PROMPT_SYSTEM, [], user_message)
