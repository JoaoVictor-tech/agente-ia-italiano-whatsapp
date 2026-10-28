"""Motor v2 do turno do Matteo (Gomme Fenice), como grafo LangGraph:

    START -> entender -> decidir -> redigir -> verificar -> END
                                       ^           |
                                       +-----------+  (barrou: 1 reescrita com o motivo)

  entender  (app/entendimento.py) LLM com saida estruturada: o que o cliente disse. Antes
            dele roda a extracao deterministica (medida, quantidade, campos categoricos);
            depois, cada valor do LLM passa por `agent_tools.registrar`, que confere contra
            a fala real do cliente. O modelo nao grava dado nenhum sozinho.
  decidir   (app/politica.py) funcao pura: estado + entendimento -> acao. Sem LLM.
  redigir   (app/redacao.py) LLM sem ferramentas: escreve a acao ja decidida.
  verificar (app/response_grounding.py) proibicoes absolutas do prompt + a acao foi cumprida.
            Barrou: volta UMA vez para a redacao com o motivo; barrou de novo, sai o testo
            approvato da acao (redacao.texto_pronto) - nunca o fallback generico.

Chamado por `llm.run_turn` (que abre o trace do Langfuse). O efeito colateral sai pelo
`ToolContext` (slots, transferido, motivo, note_operatore), que o pipeline le depois do turno.

Motor antigo (loop de tool-calling em que o modelo decidia e escrevia na mesma chamada,
contido por app/anti_stall.py e retries): removido em 10/2026. Ver README-ARQUITETURA.md.
"""

from __future__ import annotations

import logging
import operator
from dataclasses import dataclass, field
from typing import Annotated, Awaitable, Callable, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from app import agent_tools, entendimento, politica, redacao, response_grounding
from app.llm_providers.base import MAX_OUTPUT_TOKENS, sum_usage
from app.llm_providers.chat_models import criar_modelo, erro_de_provedor, system_message, uso
from app.pricing_llm import NormalizedUsage
from app.settings_service import ActiveLLMSettings

logger = logging.getLogger(__name__)

# Marcador que o pipeline grava no lugar de uma imagem (app/pipeline.py:MARCADOR_FOTO). O
# Matteo nao le imagens: a foto e' motivo de passagem ao operador ("FOTO RICEVUTA").
PREFIXO_FOTO = "[FOTO]"

MAX_REESCRITAS = 1


@dataclass
class Turno:
    """Entrada do turno e dependencias. O modelo pode trocar no meio do turno: se o provedor
    ativo cair (5xx, cota, timeout), `trocar_para_reserva` passa tudo para o modelo reserva e
    o turno termina nele - o mesmo comportamento que o motor antigo tinha."""

    ctx: agent_tools.ToolContext
    history: list[dict]
    user_message: str
    persona: str
    llm_settings: ActiveLLMSettings
    llm_settings_fallback: ActiveLLMSettings | None = None
    avisos: list[str] = field(default_factory=list)
    modelo: BaseChatModel = field(init=False)
    provider: str = field(init=False)
    fallback_usado: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.modelo = criar_modelo(self.llm_settings, max_tokens=MAX_OUTPUT_TOKENS)
        self.provider = self.llm_settings.provider

    @property
    def ativo(self) -> ActiveLLMSettings:
        return self.llm_settings_fallback if self.fallback_usado else self.llm_settings

    def trocar_para_reserva(self) -> bool:
        reserva = self.llm_settings_fallback
        if self.fallback_usado or not reserva or not reserva.api_key:
            return False
        self.modelo = criar_modelo(reserva, max_tokens=MAX_OUTPUT_TOKENS)
        self.provider = reserva.provider
        self.fallback_usado = True
        return True


class Estado(TypedDict, total=False):
    """O que muda de no para no."""

    entendido: entendimento.Entendimento | None
    registro: dict
    registrou_algo: bool
    acao: politica.Acao
    texto: str
    cortada: bool
    reescritas: int
    correcao: str | None
    refazer: bool
    primeiro_bloqueio: str | None
    bloqueio_final: str | None
    usage: Annotated[NormalizedUsage, sum_usage]
    chamadas_estruturadas: Annotated[int, operator.add]


async def _com_reserva(t: Turno, chamada: Callable[[BaseChatModel, str], Awaitable]):
    """Executa a chamada ao LLM; se o provedor estiver indisponivel, repete no reserva."""
    try:
        return await chamada(t.modelo, t.provider)
    except Exception as exc:
        if not erro_de_provedor(exc) or not t.trocar_para_reserva():
            raise
        logger.warning(
            "Provedor %s indisponivel (%s: %s) - repetindo no reserva %s:%s.",
            t.llm_settings.provider, type(exc).__name__, exc,
            t.ativo.provider, t.ativo.model,
        )
        return await chamada(t.modelo, t.provider)


# --- Nos ------------------------------------------------------------------------------------


async def entender(estado: Estado, runtime: Runtime[Turno]) -> Estado:
    t = runtime.context
    t.ctx.ultima_mensagem_do_cliente = t.user_message

    # Extracao deterministica primeiro (medida, quantidade com unidade, campos categoricos
    # citados sem ambiguidade): barata, testada e nao alucina. O LLM complementa o resto.
    antes = agent_tools.slots_visiveis(t.ctx.slots)
    pre_registrado = agent_tools.pre_registrar_do_texto(t.ctx.slots, t.user_message)

    entendido, bruto = await _com_reserva(
        t,
        lambda modelo, provider: entendimento.entender(
            modelo=modelo, provider=provider, history=t.history,
            user_message=t.user_message, slots=t.ctx.slots,
        ),
    )

    registro = agent_tools.registrar(t.ctx.slots, dict(entendido.dados), t.user_message) if entendido.dados else {}
    if registro.get("nao_registrado"):
        logger.info("Dados descartados na validacao: %s", registro["nao_registrado"])
    # O que a extracao deterministica aceitou tambem e' "o que o cliente acabou de dizer", e a
    # redacao precisa saber: sem isto, uma correcao ("scusa, anzi no: invernali") recebia a
    # MESMA mensagem do turno anterior, como se o cliente tivesse sido ignorado - o dado era
    # registrado certo, mas o briefing nao o mencionava (pego na suite de conversas, 20/10/2026).
    aceitos = {**pre_registrado, **(registro.get("registrado") or {})}
    registro = {
        "registrado": aceitos,
        # Dado que o cliente TROCOU ("estive" -> "invernali"). Vale a ultima informacao, que e'
        # a regra do prompt, mas a redacao precisa confirmar a troca em poucas palavras: repetir
        # a mesma frase de antes faz o cliente achar que nao foi lido.
        "corrigido": {
            campo: (antes[campo], valor)
            for campo, valor in aceitos.items()
            if antes.get(campo) and str(antes[campo]) != str(valor)
        },
        "nao_registrado": registro.get("nao_registrado") or {},
    }

    # Preenchidos pelo CODIGO, nunca pelo modelo:
    entendido.foto_ricevuta = PREFIXO_FOTO in t.user_message
    # O cliente tentou dar a medida e ela nao passou na validacao (formato que nao existe, ou
    # numero que ele nao escreveu): a saida e' a foto do fianco, nao reperguntar a medida.
    entendido.misura_non_riconosciuta = bool(
        (registro.get("nao_registrado") or {}).get(agent_tools.LABEL_MISURA)
        and not t.ctx.slots.get(agent_tools.LABEL_MISURA)
    )
    # Rede de seguranca do entendimento: "voglio parlare con un operatore" escrito com todas
    # as letras nao pode depender de o modelo ter marcado a intencao certa.
    if response_grounding.pediu_operador(t.user_message):
        entendido.intencao = entendimento.RICHIESTA_OPERATORE

    t.ctx.note_operatore = entendido.note_per_operatore or None
    return {
        "entendido": entendido,
        "registro": registro,
        "registrou_algo": bool(pre_registrado or (registro.get("registrado") or {})),
        "usage": uso(bruto),
        "chamadas_estruturadas": 1,
    }


def decidir(estado: Estado, runtime: Runtime[Turno]) -> Estado:
    t = runtime.context
    entendido = estado.get("entendido") or entendimento.Entendimento()
    registrou_algo = bool(estado.get("registrou_algo"))
    primeira_resposta = not any(m.get("role") == "assistant" for m in t.history)

    acao = politica.decidir(
        t.ctx.slots, entendido,
        primeira_resposta=primeira_resposta,
        registrou_algo=registrou_algo,
    )
    logger.info("Decisao do turno: %s", acao.para_trace())
    t.ctx.slots.update(politica.atualizar_contadores(t.ctx.slots, acao, registrou_algo=registrou_algo))

    if acao.tipo == politica.TRANSFERIR:
        # Mesmo efeito que a ferramenta `transferirAtendimento` tinha no motor antigo: quem
        # executa a transferencia de fato e' o pipeline, DEPOIS de enviar a mensagem.
        t.ctx.transferido = True
        t.ctx.motivo_transferencia = acao.motivo
    return {"acao": acao}


async def redigir(estado: Estado, runtime: Runtime[Turno]) -> Estado:
    t = runtime.context
    bloco = redacao.montar_bloco(
        acao=estado["acao"],
        entendido=estado.get("entendido"),
        registro=estado.get("registro") or {},
        slots=t.ctx.slots,
        avisos=t.avisos,
        primeira_resposta=not any(m.get("role") == "assistant" for m in t.history),
        correcao=estado.get("correcao"),
    )
    resposta = await _com_reserva(
        t,
        lambda modelo, provider: modelo.ainvoke([
            system_message(provider, t.persona, bloco),
            *t.history,
            HumanMessage(content=t.user_message),
        ]),
    )
    # `.text` e' propriedade no langchain-core 1.x e metodo nas versoes antigas.
    bruto = getattr(resposta, "text", "") or ""
    texto = response_grounding.limpar_resposta(bruto() if callable(bruto) else bruto)
    return {"texto": texto, "cortada": parou_por_limite(resposta), "usage": uso(resposta)}


# Motivo de parada por limite de tokens, no nome que cada provedor usa.
_PAROU_POR_LIMITE = {"MAX_TOKENS", "length", "max_tokens"}


def parou_por_limite(resposta) -> bool:
    """A redacao bateu no teto de tokens? Entao o texto esta cortado no meio, mesmo que
    pareca normal ate ali (o raciocinio do modelo conta no mesmo teto de saida)."""
    meta = getattr(resposta, "response_metadata", None) or {}
    return any(meta.get(chave) in _PAROU_POR_LIMITE for chave in ("finish_reason", "stop_reason"))


def verificar(estado: Estado, runtime: Runtime[Turno]) -> Estado:
    t = runtime.context
    resultado = response_grounding.validar(estado["texto"], estado["acao"], t.ctx.slots)
    if resultado.valido and estado.get("cortada"):
        resultado = response_grounding.GroundingResult(
            False, "il messaggio è stato troncato a metà (limite di token); scrivine uno più corto"
        )

    reescritas = estado.get("reescritas", 0)
    saida: Estado = {"bloqueio_final": resultado.motivo, "refazer": False}
    if reescritas == 0:
        saida["primeiro_bloqueio"] = resultado.motivo

    if not resultado.valido and reescritas < MAX_REESCRITAS:
        # A redacao nao tem ferramenta e nao decide nada: refazer com o motivo e' correcao de
        # TEXTO, nao "trocar uma alucinacao por outra" (o risco que o retry do motor antigo
        # tinha, porque la o retry podia mudar a decisao de transferir).
        logger.warning("Verificacao barrou (%s) - redigindo de novo. Trechos: %s", resultado.motivo, resultado.trechos)
        return {**saida, "refazer": True, "reescritas": reescritas + 1, "correcao": resultado.motivo}

    if not resultado.valido:
        logger.warning("Verificacao barrou de novo (%s) - usando o testo approvato da acao.", resultado.motivo)
        saida["texto"] = redacao.texto_pronto(estado["acao"])
    return saida


def _depois_de_verificar(estado: Estado) -> str:
    return "redigir" if estado.get("refazer") else END


def _montar_grafo():
    grafo = StateGraph(Estado, context_schema=Turno)
    grafo.add_node("entender", entender)
    grafo.add_node("decidir", decidir)
    grafo.add_node("redigir", redigir)
    grafo.add_node("verificar", verificar)
    grafo.add_edge(START, "entender")
    grafo.add_edge("entender", "decidir")
    grafo.add_edge("decidir", "redigir")
    grafo.add_edge("redigir", "verificar")
    grafo.add_conditional_edges("verificar", _depois_de_verificar, ["redigir", END])
    return grafo.compile()


GRAFO = _montar_grafo()


# --- Entrada --------------------------------------------------------------------------------


async def run_turn(
    *,
    history: list[dict],
    user_message: str,
    llm_settings: ActiveLLMSettings,
    tool_context: agent_tools.ToolContext,
    avisos: list[str] | None = None,
    dados_confirmados: dict | None = None,
    llm_settings_fallback: ActiveLLMSettings | None = None,
    callbacks: list | None = None,
):
    """`callbacks`: handlers do LangChain para o grafo (o do Langfuse, quando ligado - ver
    app/observabilidade.py). Cada no e cada chamada ao LLM viram observacoes do trace."""
    from app.llm import TurnResult  # import tardio: llm.py importa este modulo

    if not llm_settings.api_key:
        raise RuntimeError(
            f"Nenhuma chave de API configurada para o provedor '{llm_settings.provider}' - "
            "configure em /dashboard/settings ou no .env (GEMINI_API_KEY)."
        )
    if dados_confirmados:
        tool_context.slots.update(dados_confirmados)

    t = Turno(
        ctx=tool_context,
        history=history,
        user_message=user_message,
        persona=redacao.persona_ativa(llm_settings.system_prompt),
        llm_settings=llm_settings,
        llm_settings_fallback=llm_settings_fallback,
        avisos=avisos or [],
    )

    final = await GRAFO.ainvoke(
        {"usage": NormalizedUsage(), "chamadas_estruturadas": 0, "reescritas": 0},
        config={"callbacks": callbacks or [], "run_name": "motor_v2"},
        context=t,
    )

    entendido = final.get("entendido")
    primeiro_bloqueio = final.get("primeiro_bloqueio")
    usado = t.ativo
    tool_context.trace = {
        "motor": "v2",
        "entendimento": entendido.para_trace() if entendido else None,
        "descartado_na_validacao": (final.get("registro") or {}).get("nao_registrado") or {},
        "acao": final["acao"].para_trace(),
        "verificacao": primeiro_bloqueio,
        "verificacao_final": final.get("bloqueio_final"),
    }
    return TurnResult(
        final_text=final["texto"],
        # O custo sai do provedor que de fato respondeu - se o reserva entrou, a tabela e' a dele.
        usage_cost=_custo(usado, final["usage"]),
        transferido=tool_context.transferido,
        tool_calls_used=final.get("chamadas_estruturadas", 0),
        modelo_usado=f"{usado.provider}:{usado.model}",
        fallback_usado=t.fallback_usado,
        # O painel grava isto como evento `resposta_refeita` (antes vinha dos retries do motor
        # antigo; agora e' a reescrita da verificacao, com o motivo que a barrou).
        motivos_retry=[f"verificacao: {primeiro_bloqueio}"] if primeiro_bloqueio else [],
        grounding_valido=primeiro_bloqueio is None,
        grounding_motivo=primeiro_bloqueio,
    )


def _custo(llm_settings: ActiveLLMSettings, usage: NormalizedUsage):
    from app.pricing_llm import calculate_cost

    return calculate_cost(llm_settings.provider, llm_settings.model, usage)
