"""A maquina de estados do turno (app/politica.py): estado + entendimento -> acao.

Funcao pura, sem LLM e sem banco - cada regra de fluxo do prompt original vira um assert
aqui. No motor antigo estas regras eram texto no prompt ("REGOLA PRIORITARIA ASSOLUTA",
"LOGICA DI RISPOSTA") e so dava para conferi-las rodando o modelo de verdade.
"""

from __future__ import annotations

import pytest

from app import entendimento as nlu
from app import politica
from app.agent_tools import (
    LABEL_CONDIZIONE,
    LABEL_MISURA,
    LABEL_QUANTITA,
    LABEL_STAGIONE,
    LABEL_UTILIZZO,
)

COMPLETO = {
    LABEL_MISURA: "205/55 R16", LABEL_QUANTITA: "4", LABEL_STAGIONE: "Invernale",
    LABEL_CONDIZIONE: "Nuovo", LABEL_UTILIZZO: "Città",
}


def ent(**kwargs) -> nlu.Entendimento:
    return nlu.Entendimento(**kwargs)


def test_primeira_mensagem_sem_dados_e_o_benvenuto():
    acao = politica.decidir({}, ent(), primeira_resposta=True)
    assert acao.tipo == politica.BENVENUTO
    assert acao.slot == LABEL_MISURA


def test_pede_um_dado_por_vez_na_ordem_do_prompt():
    acao = politica.decidir({LABEL_MISURA: "205/55 R16"}, ent(), registrou_algo=True)
    assert (acao.tipo, acao.slot) == (politica.PERGUNTAR, LABEL_QUANTITA)


def test_dados_completos_transfere():
    acao = politica.decidir(dict(COMPLETO), ent())
    assert (acao.tipo, acao.motivo) == (politica.TRANSFERIR, politica.DATI_COMPLETI)


def test_foto_vai_sempre_para_o_operador():
    """O Matteo nao le imagens: foto = FASE FINALE, mesmo sem nenhum dado."""
    acao = politica.decidir({}, ent(foto_ricevuta=True), primeira_resposta=True)
    assert (acao.tipo, acao.motivo) == (politica.TRANSFERIR, politica.FOTO_MISURA)


@pytest.mark.parametrize(
    "intencao, motivo",
    [
        (nlu.RICHIESTA_OPERATORE, politica.RICHIESTA_OPERATORE),
        (nlu.DUBBIO_TECNICO, politica.DUBBIO_TECNICO),
        (nlu.FUORI_FLUSSO, politica.DUBBIO_TECNICO),
    ],
)
def test_gatilhos_que_transferem_na_hora(intencao, motivo):
    acao = politica.decidir({}, ent(intencao=intencao), primeira_resposta=True)
    assert (acao.tipo, acao.motivo) == (politica.TRANSFERIR, motivo)


def test_nao_sa_la_misura_pede_a_foto():
    acao = politica.decidir({}, ent(non_sa_la_misura=True))
    assert (acao.tipo, acao.slot) == (politica.CHIEDI_FOTO, LABEL_MISURA)


def test_misura_que_nao_passou_na_validacao_pede_a_foto():
    acao = politica.decidir({}, ent(misura_non_riconosciuta=True))
    assert acao.tipo == politica.CHIEDI_FOTO


def test_pergunta_preco_sem_misura_pede_a_misura():
    acao = politica.decidir({}, ent(chiede_prezzo=True))
    assert (acao.tipo, acao.slot) == (politica.RICHIESTA_PREZZO, LABEL_MISURA)


def test_pergunta_preco_com_dados_suficientes_transfere_mesmo_faltando_utilizzo():
    """Decisao registrada em app/politica.py: o uso so orienta a recomendacao - quem pergunta
    preco com medida, quantidade, estacao e condicao ja pode ir para o operador. No motor
    antigo o agente seguia coletando e o cliente repetia o pedido de preco (bateria E9)."""
    slots = {k: v for k, v in COMPLETO.items() if k != LABEL_UTILIZZO}
    acao = politica.decidir(slots, ent(chiede_prezzo=True))
    assert (acao.tipo, acao.motivo) == (politica.TRANSFERIR, politica.RICHIESTA_PREZZO_MOTIVO)


def test_mensagem_incompreensivel_pede_um_esclarecimento_e_depois_transfere():
    slots: dict = {}
    acao = politica.decidir(slots, ent(intencao=nlu.INCOMPRENSIBILE))
    assert acao.tipo == politica.CHIARIRE
    slots.update(politica.atualizar_contadores(slots, acao, registrou_algo=False))
    assert slots[politica.CHAVE_CHIARIMENTI] == 1

    acao = politica.decidir(slots, ent(intencao=nlu.INCOMPRENSIBILE))
    assert (acao.tipo, acao.motivo) == (politica.TRANSFERIR, politica.POCO_CHIARO)


def test_mesmo_dado_pedido_tres_vezes_seguidas_transfere():
    """Caso real do replay de 02/10/2026: o agente pediu o "uso" tres vezes a quem so queria
    preco. Contado em codigo, nao lembrado pelo modelo."""
    slots = {k: v for k, v in COMPLETO.items() if k != LABEL_UTILIZZO}
    for esperado in (politica.PERGUNTAR, politica.PERGUNTAR):
        acao = politica.decidir(slots, ent(), registrou_algo=False)
        assert (acao.tipo, acao.slot) == (esperado, LABEL_UTILIZZO)
        slots.update(politica.atualizar_contadores(slots, acao, registrou_algo=False))

    acao = politica.decidir(slots, ent(), registrou_algo=False)
    assert (acao.tipo, acao.motivo) == (politica.TRANSFERIR, politica.POCO_CHIARO)


def test_contador_zera_quando_o_cliente_traz_dado_novo():
    slots = {LABEL_MISURA: "205/55 R16"}
    acao = politica.decidir(slots, ent(), registrou_algo=False)
    slots.update(politica.atualizar_contadores(slots, acao, registrou_algo=False))
    assert slots[politica.CHAVE_PEDIDOS_SEGUIDOS] == 1

    slots[LABEL_QUANTITA] = "4"
    acao = politica.decidir(slots, ent(), registrou_algo=True)
    slots.update(politica.atualizar_contadores(slots, acao, registrou_algo=True))
    assert (acao.tipo, acao.slot) == (politica.PERGUNTAR, LABEL_STAGIONE)
    assert slots[politica.CHAVE_PEDIDOS_SEGUIDOS] == 1
