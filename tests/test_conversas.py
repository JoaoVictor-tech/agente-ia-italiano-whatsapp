"""Conversas inteiras contra o LLM REAL, avaliadas com o DeepEval. Nada sai para a Helena e
nada e' gravado - nem banco e' preciso: o modelo e a chave vem do .env.

Cada caso de tests/conversas.json vira um `ConversationalTestCase` com duas metricas:

  RegrasDoCaso      deterministica: motivo da passagem, dados registrados, nada inventado,
                    termos proibidos, termina (ou nao) com pergunta. Nao precisa de juiz.
  Atendimento       ConversationalGEval com juiz LLM (Gemini por padrao): a conversa cumpre o
                    `resultado_esperado` do caso, de forma natural para o WhatsApp?

Modelo: DEFAULT_LLM_PROVIDER e DEFAULT_LLM_MODEL do ambiente/.env. Juiz: DEEPEVAL_JUDGE_MODEL
(padrao gemini-2.5-flash).

    deepeval test run tests/test_conversas.py
    deepeval test run tests/test_conversas.py -k prezzo

Regra de casa: cada incidente de producao vira um caso novo no JSON (anonimizado), em vez de
uma regex nova no codigo.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.metrics import BaseConversationalMetric, ConversationalGEval
from deepeval.models import GeminiModel
from deepeval.test_case import ConversationalTestCase, MultiTurnParams, Turn

from app import agent_tools, llm
from app.config import get_settings
from app.settings_service import ActiveLLMSettings, default_system_prompt
from tests.conftest import novo_contexto

pytestmark = pytest.mark.llm

CASOS = json.loads((Path(__file__).parent / "conversas.json").read_text(encoding="utf-8"))

_CHAVES = {
    "gemini": lambda s: s.gemini_api_key,
    "openai": lambda s: s.openai_api_key,
    "deepseek": lambda s: s.deepseek_api_key,
}


def _llm_settings() -> ActiveLLMSettings:
    s = get_settings()
    chave = _CHAVES.get(s.default_llm_provider, lambda _s: None)(s)
    if not chave:
        pytest.skip(f"sem chave de API para '{s.default_llm_provider}' no .env")
    return ActiveLLMSettings(
        provider=s.default_llm_provider,
        model=s.default_llm_model,
        api_key=chave,
        system_prompt=default_system_prompt(),
    )


def _juiz() -> GeminiModel | None:
    chave = get_settings().gemini_api_key
    if not chave:
        return None
    return GeminiModel(model=os.environ.get("DEEPEVAL_JUDGE_MODEL", "gemini-2.5-flash"), api_key=chave, temperature=0)


async def _conversar(caso: dict, llm_settings: ActiveLLMSettings) -> dict:
    """Roda o roteiro do caso turno a turno, como o pipeline faria."""
    historico: list[dict] = []
    slots: dict = {}
    turnos: list[Turn] = []
    motivo = None
    for mensagem in caso["mensagens"]:
        ctx = novo_contexto()
        resultado = await llm.run_turn(
            history=historico,
            user_message=mensagem,
            llm_settings=llm_settings,
            tool_context=ctx,
            avisos=[],
            dados_confirmados=slots,
        )
        slots = dict(ctx.slots)
        resposta = resultado.final_text
        turnos += [
            Turn(role="user", content=mensagem),
            # O que o motor entendeu e decidiu vai junto: e' por ali que se acha a causa de uma
            # resposta ruim (entendimento x decisao x redacao).
            Turn(role="assistant", content=resposta, metadata=ctx.trace),
        ]
        if ctx.transferido:
            motivo = ctx.motivo_transferencia
            break
        historico += [{"role": "user", "content": mensagem}, {"role": "assistant", "content": resposta}]
    return {"turnos": turnos, "motivo": motivo, "slots": slots}


class RegrasDoCaso(BaseConversationalMetric):
    """As expectativas objetivas do caso. Deterministica: nota 1 ou 0, motivo = o que falhou."""

    def __init__(self, esperado: dict):
        self.esperado = esperado
        self.threshold = 1.0
        self.include_reason = True
        self.async_mode = False

    def measure(self, test_case: ConversationalTestCase, *args, **kwargs) -> float:
        esperado, meta = self.esperado, test_case.metadata or {}
        respostas = [t.content for t in test_case.turns if t.role == "assistant"]
        ultima = respostas[-1] if respostas else ""
        slots = meta.get("slots") or {}
        falhas = []
        if "motivo" in esperado and meta.get("motivo") != esperado["motivo"]:
            falhas.append(f"motivo: esperado {esperado['motivo']!r}, obtido {meta.get('motivo')!r}")
        for chave, valor in (esperado.get("slots") or {}).items():
            if slots.get(chave) != valor:
                falhas.append(f"dado {chave}: esperado {valor!r}, obtido {slots.get(chave)!r}")
        for chave in esperado.get("slots_ausentes") or []:
            if slots.get(chave):
                falhas.append(f"o dado {chave} deveria estar vazio, veio {slots[chave]!r}")
        if esperado.get("ultima_com_pergunta") and "?" not in ultima:
            falhas.append("a ultima mensagem deveria pedir o proximo dado, e ficou sem pergunta")
        if esperado.get("ultima_sem_pergunta") and "?" in ultima:
            falhas.append("a ultima mensagem nao deveria ter pergunta")
        for termo in esperado.get("proibido") or []:
            # Palavra inteira (com plural italiano): "marca" nao pode casar com "marcata", e
            # "disponibili" nao pode casar com "disponibilità" - que e' texto aprovado.
            padrao = re.compile(rf"\b{re.escape(termo)}[ieo]?\b", re.IGNORECASE)
            if any(padrao.search(r) for r in respostas):
                falhas.append(f"citou {termo!r}, que o Matteo nunca pode dizer")
        self.score = 0.0 if falhas else 1.0
        self.reason = "; ".join(falhas) or "todas as expectativas do caso foram cumpridas"
        self.success = not falhas
        return self.score

    async def a_measure(self, test_case: ConversationalTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self):
        return "Regras do caso"


def _atendimento(juiz: GeminiModel) -> ConversationalGEval:
    return ConversationalGEval(
        name="Atendimento",
        criteria=(
            "Valuta le risposte dell'assistente (Matteo, assistente di Gomme Fenice su WhatsApp) nella "
            "conversazione. Devono rispettare il risultato atteso del caso e suonare come una "
            "persona che risponde: messaggi brevi, AL MASSIMO una domanda per messaggio (un "
            "messaggio senza domande, come \"Mi manca solo la quantità e ci siamo.\", va benissimo "
            "e non va penalizzato), senza "
            "ripetere un dato che il cliente ha già dato, senza gergo interno, senza chiedere "
            "il permesso di passare la richiesta a un operatore. Non deve MAI dare un prezzo, "
            "dichiarare che ci sono pneumatici in magazzino o citare una marca; dire che la "
            "Gomme Fenice verifica disponibilità e prezzo è invece permesso e non va penalizzato. "
            "Penalizza le risposte che ignorano quello che il cliente ha appena detto - per "
            "esempio una correzione - o che non sono in italiano."
        ),
        evaluation_params=[MultiTurnParams.ROLE, MultiTurnParams.CONTENT, MultiTurnParams.EXPECTED_OUTCOME],
        model=juiz,
        threshold=0.6,
    )


@pytest.mark.parametrize("caso", CASOS, ids=[c["nome"] for c in CASOS])
def test_conversa(caso):
    llm_settings = _llm_settings()
    rodada = asyncio.run(_conversar(caso, llm_settings))
    test_case = ConversationalTestCase(
        name=caso["nome"],
        turns=rodada["turnos"],
        scenario=" / ".join(caso["mensagens"]),
        expected_outcome=caso["resultado_esperado"],
        chatbot_role=(
            "Matteo, assistente di Gomme Fenice su WhatsApp: raccoglie i cinque dati del pneumatico e "
            "passa la richiesta a un esperto di Gomme Fenice."
        ),
        metadata={
            "motivo": rodada["motivo"],
            "slots": agent_tools.slots_visiveis(rodada["slots"]),
            "modelo": f"{llm_settings.provider}:{llm_settings.model}",
        },
    )
    metricas: list = [RegrasDoCaso(caso["esperado"])]
    juiz = _juiz()
    if juiz is not None:
        metricas.append(_atendimento(juiz))
    assert_test(test_case, metricas)
