"""As duas redes de seguranca do pipeline depois do turno (app/pipeline.py:_texto_da_passagem).

No motor v2 quem decide a passagem ao operador e' a politica, entao aqui sobrou pouco - e e'
exatamente esse "pouco" que nao pode falhar: o cliente nunca pode receber silencio, e uma foto
nunca pode ficar parada na fila da IA (o Matteo nao le imagens).
"""

from __future__ import annotations

from app import pipeline, politica
from app.agent_tools import LABEL_CONDIZIONE, LABEL_MISURA, LABEL_QUANTITA, LABEL_STAGIONE, LABEL_UTILIZZO
from tests.conftest import novo_contexto

COMPLETO = {
    LABEL_MISURA: "205/55 R16", LABEL_QUANTITA: "4", LABEL_STAGIONE: "Invernale",
    LABEL_CONDIZIONE: "Nuovo", LABEL_UTILIZZO: "Città",
}


class ResultadoFalso:
    def __init__(self, final_text: str, transferido: bool = False):
        self.final_text = final_text
        self.transferido = transferido


def test_texto_normal_passa_intacto():
    ctx = novo_contexto()
    texto, transferir = pipeline._texto_da_passagem(
        ResultadoFalso("Perfetto 👍 Mi manca solo la quantità e ci siamo."), ctx, "205/55 R16",
    )
    assert texto == "Perfetto 👍 Mi manca solo la quantità e ci siamo."
    assert transferir is False


def test_texto_vazio_sem_transferencia_vira_fallback():
    ctx = novo_contexto()
    texto, transferir = pipeline._texto_da_passagem(ResultadoFalso(""), ctx, "ciao")
    assert texto == pipeline.FALLBACK_REPLY
    assert transferir is False


def test_texto_vazio_com_dados_completos_usa_o_testo_approvato():
    ctx = novo_contexto()
    ctx.slots.update(COMPLETO)
    ctx.transferido = True
    ctx.motivo_transferencia = politica.DATI_COMPLETI
    texto, transferir = pipeline._texto_da_passagem(ResultadoFalso("", transferido=True), ctx, "in citta")
    assert texto == pipeline.TESTO_DATI_COMPLETI
    assert transferir is True


def test_texto_vazio_transferindo_sem_dados_usa_a_frase_de_passagem():
    ctx = novo_contexto()
    ctx.transferido = True
    ctx.motivo_transferencia = politica.RICHIESTA_OPERATORE
    texto, transferir = pipeline._texto_da_passagem(ResultadoFalso("", transferido=True), ctx, "voglio un operatore")
    assert texto == pipeline.TESTO_PASSAGGIO
    assert transferir is True


def test_foto_sem_transferencia_do_motor_e_transferida_pelo_servidor():
    """Rede de seguranca: se a politica, por qualquer motivo, nao transferir uma foto, o
    servidor transfere - senao o lead fica parado na fila da IA."""
    ctx = novo_contexto()
    texto, transferir = pipeline._texto_da_passagem(
        ResultadoFalso("Perfetto 👍"), ctx, f"{pipeline.MARCADOR_FOTO}",
    )
    assert transferir is True
    assert ctx.transferido is True
    assert ctx.motivo_transferencia == politica.FOTO_MISURA
    assert texto == "Perfetto 👍"


def test_foto_sem_transferencia_e_sem_texto_usa_o_testo_foto_ricevuta():
    ctx = novo_contexto()
    texto, transferir = pipeline._texto_da_passagem(ResultadoFalso(""), ctx, pipeline.MARCADOR_FOTO)
    assert texto == pipeline.TESTO_FOTO_RICEVUTA
    assert transferir is True
