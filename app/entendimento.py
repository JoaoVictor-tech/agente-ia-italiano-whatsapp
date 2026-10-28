"""Etapa 1 do motor v2: ENTENDIMENTO. Uma chamada de LLM com saida estruturada que le a
mensagem do cliente e devolve o que ele disse - e mais nada. Ela nao escreve ao cliente e nao
decide o que fazer.

Por que separar isto da redacao: no motor antigo a mesma chamada extraia os dados, decidia se
transferia E escrevia a resposta. Os desvios vinham da mistura - pedir de novo a medida escrita
na mesma frase (3 de 3 vezes no gemini-2.5-flash, 30/09/2026), anunciar a passagem sem chamar a
ferramenta, transferir junto com uma pergunta. Aqui a extracao nao tem para quem perguntar: a
unica saida possivel e' o JSON.

O que sai daqui NAO e' verdade ainda: cada valor passa pela conferencia de
`agent_tools.registrar` (contra o texto do cliente) antes de virar dado.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from app.agent_tools import (
    CAMPOS,
    LABEL_CONDIZIONE,
    LABEL_MISURA,
    LABEL_QUANTITA,
    LABEL_STAGIONE,
    LABEL_UTILIZZO,
    slots_visiveis,
)
from app.llm_providers.chat_models import system_message

logger = logging.getLogger(__name__)

PROMPT = (Path(__file__).parent / "prompts" / "entendimento.md").read_text(encoding="utf-8")

NOME = "registrar_entendimento"
DESCRICAO = "Registra o que o cliente disse na mensagem atual. Sempre chamada, uma vez."

RICHIESTA_OPERATORE = "richiesta_operatore"
DUBBIO_TECNICO = "dubbio_tecnico"
FUORI_FLUSSO = "fuori_flusso"
INCOMPRENSIBILE = "incomprensibile"
CONVERSA = "conversa"

INTENCOES = (RICHIESTA_OPERATORE, DUBBIO_TECNICO, FUORI_FLUSSO, INCOMPRENSIBILE, CONVERSA)

# Campo do JSON -> rotulo do dado (agent_tools.CAMPOS).
_CAMPO_DO_SLOT = {
    "misura": LABEL_MISURA,
    "quantita": LABEL_QUANTITA,
    "stagione": LABEL_STAGIONE,
    "condizione": LABEL_CONDIZIONE,
    "utilizzo": LABEL_UTILIZZO,
}

SCHEMA: dict = {
    "type": "object",
    "properties": {
        "misura": {"type": "string", "description": "Medida do pneu exatamente como o cliente escreveu. Vazio se não escreveu."},
        "quantita": {"type": "integer", "description": "Quantos pneus ele quer, só se disse. 0 se não disse."},
        "stagione": {"type": "string", "description": "Estivo, Invernale ou 4 stagioni - só se ele disse. Vazio se não."},
        "condizione": {"type": "string", "description": "Nuovo ou Usato - só se ele disse. Vazio se não."},
        "utilizzo": {"type": "string", "description": "Città, Misto ou Tanti km - só se ele disse. Vazio se não."},
        "intencao": {"type": "string", "enum": list(INTENCOES), "description": "A intenção principal desta mensagem (ver regras no prompt)."},
        "chiede_prezzo": {"type": "boolean", "description": "true se ele perguntou preço ou disponibilidade."},
        "non_sa_la_misura": {"type": "boolean", "description": "true se ele disse que não sabe a medida."},
        "domanda_del_cliente": {"type": "string", "description": "A pergunta que ele fez nesta mensagem, em uma frase. Vazio se não há."},
        "note_per_operatore": {"type": "string", "description": "1-2 frases em italiano para o operador, sobre o que não cabe nos 5 dados. Vazio se nada."},
    },
    "required": [
        "misura", "quantita", "stagione", "condizione", "utilizzo", "intencao",
        "chiede_prezzo", "non_sa_la_misura", "domanda_del_cliente", "note_per_operatore",
    ],
    "additionalProperties": False,
}

# Quantas mensagens do historico a extracao le. Ela precisa da ultima pergunta do atendente
# para entender respostas curtas ("4", "nuove"); nao precisa da conversa inteira.
JANELA_HISTORICO = 10


@dataclass
class Entendimento:
    dados: dict = field(default_factory=dict)  # so os campos preenchidos, por rotulo (CAMPOS)
    intencao: str = CONVERSA
    chiede_prezzo: bool = False
    non_sa_la_misura: bool = False
    domanda_del_cliente: str = ""
    note_per_operatore: str = ""
    # Preenchidos pelo CODIGO, nao pelo modelo (ver motor_v2.entender).
    foto_ricevuta: bool = False
    misura_non_riconosciuta: bool = False

    def para_trace(self) -> dict:
        return {
            "dados": self.dados,
            "intencao": self.intencao,
            "chiede_prezzo": self.chiede_prezzo,
            "non_sa_la_misura": self.non_sa_la_misura,
            "domanda_del_cliente": self.domanda_del_cliente,
            "foto_ricevuta": self.foto_ricevuta,
            "misura_non_riconosciuta": self.misura_non_riconosciuta,
        }


# Campo "vazio" escrito pelo modelo (visto no agente da Alpha com o Gemini: "-" no lugar de "").
_VAZIO = {"-", "--", "—", "–", ".", "n/a", "na", "nessuno", "nessuna", "null", "none", "vazio",
          "non indicato", "non raccolto", "nao informado", "não informado", "no", "nulla", "niente"}


def interpretar(bruto: dict) -> Entendimento:
    """JSON do modelo -> Entendimento. Tolerante: campo ausente ou com tipo errado vira o
    valor neutro, nunca excecao - um JSON torto nao pode derrubar o turno."""
    bruto = bruto if isinstance(bruto, dict) else {}

    def texto(chave: str) -> str:
        valor = bruto.get(chave)
        valor = valor.strip() if isinstance(valor, str) else ""
        return "" if valor.lower().strip(" .") in _VAZIO or not valor.strip(" .-—–") else valor

    dados: dict = {}
    for chave, rotulo in _CAMPO_DO_SLOT.items():
        if chave == "quantita":
            quantidade = bruto.get(chave)
            if isinstance(quantidade, float) and quantidade.is_integer():
                quantidade = int(quantidade)
            if isinstance(quantidade, int) and not isinstance(quantidade, bool) and quantidade > 0:
                dados[rotulo] = quantidade
        elif texto(chave):
            dados[rotulo] = texto(chave)

    intencao = texto("intencao")
    return Entendimento(
        dados=dados,
        intencao=intencao if intencao in INTENCOES else CONVERSA,
        chiede_prezzo=bruto.get("chiede_prezzo") is True,
        non_sa_la_misura=bruto.get("non_sa_la_misura") is True,
        domanda_del_cliente=texto("domanda_del_cliente"),
        note_per_operatore=texto("note_per_operatore"),
    )


def montar_entrada(*, history: list[dict], user_message: str, slots: dict) -> str:
    """O que muda a cada turno vai aqui, e nao no system: assim o prompt fica cacheavel."""
    linhas = ["DADOS JÁ REGISTRADOS NESTA CONVERSA:"]
    registrados = slots_visiveis(slots)
    linhas += [f"- {c}: {registrados[c]}" for c in CAMPOS if c in registrados] or ["- nenhum"]
    linhas += ["", "HISTÓRICO RECENTE:"]
    recentes = history[-JANELA_HISTORICO:]
    if recentes:
        for m in recentes:
            quem = "CLIENTE" if m.get("role") == "user" else "ATENDENTE"
            linhas.append(f"{quem}: {m.get('content') or ''}")
    else:
        linhas.append("(início da conversa)")
    linhas += ["", "MENSAGEM ATUAL DO CLIENTE:", user_message]
    return "\n".join(linhas)


async def entender(
    *,
    modelo: BaseChatModel,
    provider: str,
    history: list[dict],
    user_message: str,
    slots: dict,
) -> tuple[Entendimento, AIMessage | None]:
    """Uma chamada com a funcao `registrar_entendimento` forcada. Devolve tambem a mensagem
    crua, de onde sai o consumo de tokens.

    `bind_tools` e nao `with_structured_output`: o resultado e' o mesmo (o argumento da
    funcao), mas o segundo empilha etapas internas do LangChain em volta da chamada, e cada
    uma vira uma linha no trace do Langfuse."""
    extrator = modelo.bind_tools(
        [{"type": "function", "function": {"name": NOME, "description": DESCRICAO, "parameters": SCHEMA}}],
        tool_choice=NOME,
    )
    resposta = await extrator.ainvoke([
        system_message(provider, PROMPT),
        HumanMessage(content=montar_entrada(history=history, user_message=user_message, slots=slots)),
    ])
    chamadas = getattr(resposta, "tool_calls", None) or []
    bruto = (chamadas[0].get("args") or {}) if chamadas else {}
    if not chamadas:
        logger.warning("Entendimento sem chamada da funcao - tratado como vazio.")
    logger.info("Entendimento: %s", json.dumps(bruto, ensure_ascii=False)[:500])
    return interpretar(bruto), resposta
