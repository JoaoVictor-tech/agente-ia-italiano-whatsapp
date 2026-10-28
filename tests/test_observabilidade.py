"""Observabilidade (app/observabilidade.py) com o Langfuse DESLIGADO - que e' o padrao.

Duas coisas precisam valer sempre:
  1. sem as chaves, nada sai do processo e nada quebra (todo metodo e' no-op);
  2. o DESFECHO continua sendo classificado mesmo sem trace - e' ele que o painel grava.
E a mascara de telefone, que roda antes de qualquer coisa sair daqui.
"""

from __future__ import annotations

import pytest

from app import observabilidade
from tests.conftest import novo_contexto


@pytest.fixture(autouse=True)
def _sem_langfuse(monkeypatch):
    """Garante o estado "desligado" mesmo se o .env da maquina tiver as chaves."""
    observabilidade._cliente.cache_clear()
    monkeypatch.setattr(observabilidade, "_cliente", lambda: None)
    yield


def test_desligado_sem_chaves():
    assert observabilidade.ativo() is False


def test_turno_e_no_op_e_nao_quebra():
    ctx = novo_contexto()
    with observabilidade.turno(ctx=ctx, user_message="ciao") as trace:
        assert trace.callbacks == []
        trace.concluir(_resultado(), ctx)   # nao pode levantar
        trace.falhou(RuntimeError("x"))


def test_desfecho_e_classificado_mesmo_sem_trace():
    with observabilidade.atendimento(
        session_id="s1", contact_id="c1", environment="teste", entrada="ciao", tipo_mensagem="TEXT",
    ) as trace:
        trace.marcar(observabilidade.RESPONDIDO)
    assert trace.desfecho == observabilidade.RESPONDIDO
    assert trace.falhou is False


def test_falha_nao_e_apagada_por_sucesso_posterior():
    """Se o envio falhou, um "respondido" depois nao pode esconder isso - foi exatamente o
    que fazia uma falha de envio aparecer como atendimento bem-sucedido no painel."""
    with observabilidade.atendimento(
        session_id="s1", contact_id="c1", environment="teste", entrada="ciao", tipo_mensagem="TEXT",
    ) as trace:
        trace.marcar(observabilidade.FALHA_ENVIO, "a Helena recusou")
        trace.marcar(observabilidade.RESPONDIDO)
    assert trace.desfecho == observabilidade.FALHA_ENVIO
    assert trace.detalhe == "a Helena recusou"
    assert trace.falhou


def test_excecao_dentro_do_atendimento_vira_falha_interna():
    with pytest.raises(ValueError):
        with observabilidade.atendimento(
            session_id="s1", contact_id="c1", environment="teste", entrada="ciao", tipo_mensagem="TEXT",
        ) as trace:
            raise ValueError("boom")
    assert trace.desfecho == observabilidade.FALHA_INTERNA
    assert "ValueError" in trace.detalhe


def test_etapa_deixa_a_excecao_subir():
    with observabilidade.atendimento(
        session_id="s1", contact_id="c1", environment="teste", entrada="ciao", tipo_mensagem="TEXT",
    ) as trace:
        with pytest.raises(RuntimeError):
            with trace.etapa("envio"):
                raise RuntimeError("helena fora do ar")
        trace.marcar(observabilidade.FALHA_ENVIO)
    assert trace.desfecho == observabilidade.FALHA_ENVIO


@pytest.mark.parametrize(
    "texto, esperado",
    [
        ("chiamami al +39 345 123 4567", "chiamami al [telefono]"),
        ("il mio numero e 345 123 4567", "il mio numero e [telefono]"),
        ("whatsapp +55 11 99999-9999", "whatsapp [telefono]"),
        # O que NAO pode ser mascarado: medida de pneu e quantidade.
        ("misura 205/55 R16", "misura 205/55 R16"),
        ("2254517", "2254517"),
        ("4 gomme", "4 gomme"),
    ],
)
def test_mascara_de_telefone(texto, esperado):
    assert observabilidade.mascarar(data=texto) == esperado


def test_mascara_entra_em_estruturas_aninhadas():
    dados = {"msg": ["chiama 345 123 4567", {"nota": "+39 02 12345678"}]}
    mascarado = observabilidade.mascarar(data=dados)
    assert mascarado == {"msg": ["chiama [telefono]", {"nota": "[telefono]"}]}


def _resultado():
    class _R:
        final_text = "ciao"
        usage_cost = None
        grounding_valido = True
        grounding_motivo = None
        modelo_usado = "gemini:gemini-2.5-flash"
        fallback_usado = False

    return _R()
