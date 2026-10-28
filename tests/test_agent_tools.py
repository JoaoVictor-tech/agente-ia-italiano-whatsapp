"""Leitura deterministica do que o cliente escreveu e conferencia do que o LLM extraiu
(app/agent_tools.py). Sem LLM: entra texto, sai dado - ou nao sai.

E' a trava que impede o agente de gravar dado que o cliente nao disse. No motor antigo o
modelo chamava `criarResumo` com o que ele achava, e o riepilogo chegava ao operador com
medida inventada.
"""

from __future__ import annotations

import pytest

from app import agent_tools
from app.agent_tools import (
    LABEL_CONDIZIONE,
    LABEL_MISURA,
    LABEL_QUANTITA,
    LABEL_STAGIONE,
    LABEL_UTILIZZO,
)


@pytest.mark.parametrize(
    "texto, esperado",
    [
        ("225/55R18", "225/55 R18"),
        ("225/55 R18", "225/55 R18"),
        ("225/55/18", "225/55 R18"),
        ("225 55 18", "225/55 R18"),
        ("2255518", "225/55 R18"),
        ("ho bisogno di 205/55 r16 per favore", "205/55 R16"),
        ("misura 225/55ZR17", "225/55 R17"),
        # Ultima informacao vale (o cliente se corrige).
        ("205/55 R16, anzi no, 225/45 R17", "225/45 R17"),
        # Fora dos limites fisicos de um pneu de passeio.
        ("999/99 R99", None),
        ("il mio numero e 3451234567", None),
        ("non lo so", None),
    ],
)
def test_normalizar_misura(texto, esperado):
    assert agent_tools.normalizar_misura(texto) == esperado


@pytest.mark.parametrize(
    "texto, esperado",
    [
        ("4 gomme", 4),
        ("quattro pneumatici", 4),
        ("un paio di gomme", 2),
        ("mi servono 2 ruote", 2),
        # "4 stagioni" e' estacao, nao quantidade - erro classico do extrator ingenuo.
        ("vorrei 4 stagioni", None),
        ("quattro stagioni", None),
        # Numero solto sem unidade nao vira quantidade (pode ser parte da medida).
        ("205/55 R16", None),
        ("ciao", None),
    ],
)
def test_extrair_quantita(texto, esperado):
    assert agent_tools.extrair_quantita(texto) == esperado


@pytest.mark.parametrize(
    "campo, texto, esperado",
    [
        (LABEL_STAGIONE, "le voglio invernali", {"Invernale"}),
        (LABEL_STAGIONE, "estive grazie", {"Estivo"}),
        (LABEL_STAGIONE, "4 stagioni", {"4 stagioni"}),
        (LABEL_STAGIONE, "all season", {"4 stagioni"}),
        (LABEL_CONDIZIONE, "nuove", {"Nuovo"}),
        (LABEL_CONDIZIONE, "anche usate va bene", {"Usato"}),
        (LABEL_UTILIZZO, "giro solo in citta", {"Città"}),
        (LABEL_UTILIZZO, "faccio tanti km in autostrada", {"Tanti km"}),
        # "4 stagioni" nao pode virar utilizzo por causa do "4".
        (LABEL_UTILIZZO, "4 stagioni", set()),
    ],
)
def test_valores_citados(campo, texto, esperado):
    assert agent_tools.valores_citados(campo, texto) == esperado


def test_pre_registrar_le_varios_dados_de_uma_vez():
    slots: dict = {}
    registrado = agent_tools.pre_registrar_do_texto(slots, "vorrei 4 gomme 205/55 R16 invernali nuove")
    assert registrado == {
        LABEL_MISURA: "205/55 R16",
        LABEL_QUANTITA: "4",
        LABEL_STAGIONE: "Invernale",
        LABEL_CONDIZIONE: "Nuovo",
    }
    assert slots[LABEL_MISURA] == "205/55 R16"
    assert agent_tools.campos_faltando(slots) == [LABEL_UTILIZZO]


def test_registrar_recusa_o_que_o_cliente_nao_disse():
    """O coracao da trava: o LLM "extraiu" dados que nao estao na mensagem."""
    slots: dict = {}
    resultado = agent_tools.registrar(
        slots,
        {LABEL_MISURA: "225/45 R17", LABEL_QUANTITA: 4, LABEL_STAGIONE: "Invernale"},
        "ciao, mi servono delle gomme",
    )
    assert resultado["registrado"] == {}
    assert set(resultado["nao_registrado"]) == {LABEL_MISURA, LABEL_QUANTITA, LABEL_STAGIONE}
    assert slots == {}


def test_registrar_aceita_o_que_o_cliente_disse_e_normaliza():
    slots: dict = {}
    resultado = agent_tools.registrar(
        slots,
        {LABEL_MISURA: "205 55 16", LABEL_QUANTITA: 4, LABEL_STAGIONE: "invernali"},
        "mi servono 4 gomme 205 55 16 invernali",
    )
    assert resultado["registrado"] == {
        LABEL_MISURA: "205/55 R16", LABEL_QUANTITA: "4", LABEL_STAGIONE: "Invernale",
    }
    assert resultado["nao_registrado"] == {}


def test_registrar_recusa_quantidade_que_o_cliente_nao_escreveu():
    """Caso real do motor antigo: o cliente diz "205/55 R16" e o modelo registra 4 pneus
    porque e' o pedido mais comum."""
    slots: dict = {}
    resultado = agent_tools.registrar(slots, {LABEL_QUANTITA: 4}, "la misura e 205/55 R16")
    assert resultado["registrado"] == {}
    assert LABEL_QUANTITA in resultado["nao_registrado"]


def test_slots_visiveis_esconde_contadores_internos():
    slots = {LABEL_MISURA: "205/55 R16", "_chiarimenti": 1, "_ultimo_pedido": LABEL_QUANTITA}
    assert agent_tools.slots_visiveis(slots) == {LABEL_MISURA: "205/55 R16"}
