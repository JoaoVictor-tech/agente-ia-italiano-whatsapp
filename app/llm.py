"""Porta de entrada de um turno do Matteo (Gomme Fenice).

`run_turn` abre o trace do turno no Langfuse (app/observabilidade.py) e roda o motor
(app/motor_v2.py: entender -> decidir -> redigir -> verificar). Pipeline, replay e testes
chamam daqui; ninguem chama o motor direto.

Divisao de responsabilidade, que e' o principio central desta arquitetura:
- CODIGO decide o que e' verdade e o que fazer: quais dos 5 dados o cliente ja deu, qual e' o
  proximo a pedir, SE ja da para passar ao operador e se a resposta cumpriu a tarefa
  (app/agent_tools.py, app/politica.py, app/response_grounding.py).
- LLM entende o que o cliente disse (app/entendimento.py) e escreve a mensagem
  (app/redacao.py).

Historico: ate 10/2026 o turno era um loop de tool-calling em que o modelo decidia e escrevia
na mesma chamada, contido por travas e retries (app/anti_stall.py). Foi removido junto com os
provedores proprios (app/llm_providers/gemini_provider.py, openai_compatible_provider.py):
quem fala com cada provedor agora sao os chat models do LangChain
(app/llm_providers/chat_models.py). Ver README-ARQUITETURA.md.

O provedor/modelo/chave e o prompt (persona, editavel e versionado no painel) vem da
configuracao em runtime (app/settings_service.py).
"""

from __future__ import annotations

import logging

from app import agent_tools, observabilidade
from app.pricing_llm import UsageCost
from app.settings_service import ActiveLLMSettings

logger = logging.getLogger(__name__)

# Cliente da API compativel da DeepSeek (usado pela geracao de persona do Teste Rapido).
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class TurnResult:
    def __init__(
        self,
        final_text: str,
        usage_cost: UsageCost,
        transferido: bool,
        tool_calls_used: int,
        modelo_usado: str | None = None,
        fallback_usado: bool = False,
        motivos_retry: list[str] | None = None,
        grounding_valido: bool = True,
        grounding_motivo: str | None = None,
    ):
        self.final_text = final_text
        self.usage_cost = usage_cost
        # A POLITICA decidiu passar ao operador (a execucao acontece em app/pipeline.py,
        # depois de a resposta ser enviada ao cliente).
        self.transferido = transferido
        self.tool_calls_used = tool_calls_used
        # Qual provedor:modelo de fato respondeu - pode NAO ser o ativo, se o reserva entrou.
        self.modelo_usado = modelo_usado
        self.fallback_usado = fallback_usado
        self.motivos_retry = list(motivos_retry or [])
        # A verificacao (app/response_grounding.py) aprovou a PRIMEIRA redacao?
        self.grounding_valido = grounding_valido
        self.grounding_motivo = grounding_motivo


async def run_turn(
    *,
    history: list[dict],
    user_message: str,
    llm_settings: ActiveLLMSettings,
    tool_context: agent_tools.ToolContext,
    avisos: list[str] | None = None,
    dados_confirmados: dict | None = None,
    llm_settings_fallback: ActiveLLMSettings | None = None,
) -> TurnResult:
    """Executa um turno completo dentro de um trace do Langfuse (no-op sem as chaves)."""
    from app import motor_v2  # import tardio: motor_v2 importa TurnResult deste modulo

    with observabilidade.turno(ctx=tool_context, user_message=user_message) as trace:
        resultado = await motor_v2.run_turn(
            history=history,
            user_message=user_message,
            llm_settings=llm_settings,
            tool_context=tool_context,
            avisos=avisos,
            dados_confirmados=dados_confirmados,
            llm_settings_fallback=llm_settings_fallback,
            callbacks=trace.callbacks,
        )
        trace.concluir(resultado, tool_context)
    return resultado
