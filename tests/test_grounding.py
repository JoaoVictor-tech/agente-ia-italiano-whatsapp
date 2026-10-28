"""Verificacao do texto antes de ele chegar ao cliente (app/response_grounding.py).

As "REGOLE ASSOLUTE" do prompt viram assert aqui: preco, disponibilidade, marca, agendamento,
identidade, vazamento de instrucao, emoji fora da lista, idioma - e se o texto cumpriu a acao
que a politica decidiu.
"""

from __future__ import annotations

import pytest

from app import politica, response_grounding
from app.agent_tools import LABEL_MISURA, LABEL_QUANTITA, LABEL_STAGIONE

PEDIR_QUANTITA = politica.Acao(politica.PERGUNTAR, slot=LABEL_QUANTITA)
TRANSFERIR = politica.Acao(politica.TRANSFERIR, motivo=politica.DATI_COMPLETI)


@pytest.mark.parametrize(
    "texto, trecho_do_motivo",
    [
        ("Costano 89,90 € a gomma. Quante te ne servono?", "prezzi"),
        ("Sono subito disponibili. Quante te ne servono?", "disponibilità"),
        ("Ti consiglio le Michelin. Quante te ne servono?", "marche"),
        ("Ti prenoto il montaggio per domani. Quante te ne servono?", "appuntamenti"),
        ("Sono un assistente virtuale. Quante gomme ti servono?", "IA"),
        ("FASE FINALE: quante gomme ti servono?", "istruzioni interne"),
        ("Perfetto 🎉 quante gomme ti servono?", "emoji"),
        ('{"testo": "quante gomme ti servono?"}', "testo semplice"),
        ("Let's check the business hours. Quante gomme ti servono?", "italiano"),
    ],
)
def test_regras_absolutas_barram(texto, trecho_do_motivo):
    resultado = response_grounding.validar(texto, PEDIR_QUANTITA, {LABEL_MISURA: "205/55 R16"})
    assert not resultado.valido
    assert trecho_do_motivo in resultado.motivo


def test_texto_correto_passa():
    resultado = response_grounding.validar(
        "Perfetto 👍 Mi manca solo la quantità (2 o 4) e ci siamo.",
        PEDIR_QUANTITA,
        {LABEL_MISURA: "205/55 R16"},
    )
    assert resultado.valido, resultado.motivo


def test_mensagem_vazia_barra():
    assert not response_grounding.validar("", PEDIR_QUANTITA, {}).valido


def test_transferencia_precisa_avisar_o_cliente():
    """Caso real: a despedida saiu so "Perfetto, 4 gomme invernali." e o cliente ficou
    esperando o bot, sem saber que uma pessoa ia assumir."""
    resultado = response_grounding.validar("Perfetto 👍 quattro gomme invernali.", TRANSFERIR, {})
    assert not resultado.valido
    assert "esperto" in resultado.motivo or "operatore" in resultado.motivo


def test_transferencia_com_aviso_passa():
    resultado = response_grounding.validar(
        "Perfetto 👍 ho tutto!\nOra verifichiamo la soluzione Gomme Fenice più adatta alle tue esigenze.",
        TRANSFERIR, {},
    )
    assert resultado.valido, resultado.motivo


def test_anunciar_passagem_sem_transferir_barra():
    """A trava inversa: o motor nao decidiu transferir, entao o texto nao pode prometer isso -
    sem ela o cliente ficava esperando um operador que nunca vinha."""
    resultado = response_grounding.validar(
        "Passo la tua richiesta a un operatore. Quante gomme ti servono?", PEDIR_QUANTITA, {},
    )
    assert not resultado.valido
    assert "operatore" in resultado.motivo


def test_nao_pode_repedir_dado_que_o_cliente_ja_deu():
    resultado = response_grounding.validar(
        "Qual è la misura? E quante gomme ti servono?",
        PEDIR_QUANTITA,
        {LABEL_MISURA: "205/55 R16"},
    )
    assert not resultado.valido
    assert LABEL_MISURA in resultado.motivo


def test_pedir_um_dado_so():
    resultado = response_grounding.validar(
        "Quante gomme ti servono? E le vuoi estive o invernali?", PEDIR_QUANTITA, {},
    )
    assert not resultado.valido
    assert "un solo dato" in resultado.motivo


def test_pedir_outro_dado_que_nao_o_decidido_barra():
    resultado = response_grounding.validar(
        "Mi manca solo la stagione e ci siamo.", PEDIR_QUANTITA, {},
    )
    assert not resultado.valido
    assert LABEL_QUANTITA in resultado.motivo


@pytest.mark.parametrize(
    "acao, texto, valido",
    [
        (politica.Acao(politica.CHIEDI_FOTO, slot=LABEL_MISURA), "Mandami una foto del fianco 👍", True),
        (politica.Acao(politica.CHIEDI_FOTO, slot=LABEL_MISURA), "Qual è la misura?", False),
        (politica.Acao(politica.RICHIESTA_PREZZO, slot=LABEL_MISURA), "Mi serve prima la misura. Es. 205/55 R16", True),
        (politica.Acao(politica.CHIARIRE), "Scusami, non ho capito bene. Me lo puoi ripetere?", True),
        (politica.Acao(politica.CHIARIRE), "Scusami, non ho capito.", False),
    ],
)
def test_cada_acao_tem_sua_exigencia(acao, texto, valido):
    assert response_grounding.validar(texto, acao, {}).valido is valido


def test_limpar_resposta_tira_citacao_e_nome_de_ferramenta():
    sujo = '【4:0†source】\nregistrar_entendimento({"misura": "205/55 R16"})\nPerfetto 👍'
    assert response_grounding.limpar_resposta(sujo) == "Perfetto 👍"


def test_limpar_resposta_tira_as_aspas_do_testo_approvato():
    assert response_grounding.limpar_resposta('"Perfetto 👍 ho tutto!"') == "Perfetto 👍 ho tutto!"


@pytest.mark.parametrize(
    "mensagem, esperado",
    [
        ("voglio parlare con un operatore", True),
        ("posso parlare con una persona?", True),
        ("non voglio parlare con un bot", True),
        ("vorrei 4 gomme invernali", False),
    ],
)
def test_pediu_operador(mensagem, esperado):
    assert response_grounding.pediu_operador(mensagem) is esperado


def test_stagione_ja_dada_nao_e_repedida():
    resultado = response_grounding.validar(
        "Le vuoi estive o invernali?", PEDIR_QUANTITA, {LABEL_STAGIONE: "Invernale"},
    )
    assert not resultado.valido
