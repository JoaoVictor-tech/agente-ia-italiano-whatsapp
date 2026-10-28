"""Etapa 2 do motor v2: DECISAO. Funcao pura que recebe o estado da conversa e o que o cliente
disse, e devolve a acao do turno. Sem LLM, sem banco, sem I/O - testavel em tests/test_politica.py.

E' a maquina de estados que no motor antigo existia so como texto no prompt ("REGOLA
PRIORITARIA ASSOLUTA: QUANDO FARE IL TRASFERIMENTO", "LOGICA DI RISPOSTA", "GESTIONE MESSAGGI
POCO CHIARI") e era conferida depois por travas e retries (app/anti_stall.py, removido). Toda
regra de fluxo mora aqui; o prompt de redacao nao sabe nenhuma delas, so recebe a acao pronta.
"""

from __future__ import annotations

from dataclasses import dataclass

from app import entendimento as nlu
from app.agent_tools import LABEL_CONDIZIONE, LABEL_MISURA, LABEL_QUANTITA, LABEL_STAGIONE, campos_faltando, slots_visiveis

BENVENUTO = "BENVENUTO"                  # primeira mensagem: o texto de boas-vindas aprovado
PERGUNTAR = "PERGUNTAR"                  # "MANCA UN DATO": pedir o proximo dado que falta
RICHIESTA_PREZZO = "RICHIESTA_PREZZO"    # perguntou preco sem a medida: pedir a medida
CHIEDI_FOTO = "CHIEDI_FOTO"              # "MISURA NON CONOSCIUTA": pedir foto do fianco
CHIARIRE = "CHIARIRE"                    # mensagem incompreensivel: UM pedido de esclarecimento
TRANSFERIR = "TRANSFERIR"                # FASE FINALE (motivo em `motivo`)

# Motivos de transferencia (rotulo do riepilogo em app/resumo_atendimento.py).
DATI_COMPLETI = "dati_completi"
RICHIESTA_PREZZO_MOTIVO = "richiesta_prezzo"
FOTO_MISURA = "foto_misura"
DUBBIO_TECNICO = "dubbio_tecnico"
POCO_CHIARO = "messaggio_poco_chiaro"
RICHIESTA_OPERATORE = "richiesta_operatore"

# Condicao 2 do handoff no prompt: "il cliente chiede prezzo o disponibilità E i dati raccolti
# finora sono sufficienti per una richiesta di verifica". O prompt nao diz quais sao; o motor
# antigo, na pratica, seguia coletando quando faltava nuovo/usato (bateria E9, 02/10/2026).
# Decisao registrada aqui: bastam os 4 dados que definem o produto - o uso (citta/misto/tanti
# km) so orienta a recomendacao, e o operador pergunta se precisar.
SUFICIENTES_PARA_PRECO = (LABEL_MISURA, LABEL_QUANTITA, LABEL_STAGIONE, LABEL_CONDIZIONE)

# "GESTIONE MESSAGGI POCO CHIARI O SENZA RISPOSTA": um unico pedido de esclarecimento; sem
# resposta de novo, FASE FINALE. Contado, nao lembrado.
MAX_CHIARIMENTI = 1
# O mesmo dado pedido em mensagens seguidas sem o cliente trazer nada novo: na 3a vez
# transfere (pergunta + um esclarecimento). Caso real do replay de 02/10/2026: o agente pediu o
# "uso" tres vezes a quem queria o preco de duas medidas.
MAX_PEDIDOS_DO_MESMO_DADO = 2

CHAVE_CHIARIMENTI = "_chiarimenti"
CHAVE_ULTIMO_PEDIDO = "_ultimo_pedido"
CHAVE_PEDIDOS_SEGUIDOS = "_pedidos_seguidos"

_TRANSFERE_JA = {
    nlu.RICHIESTA_OPERATORE: RICHIESTA_OPERATORE,
    nlu.DUBBIO_TECNICO: DUBBIO_TECNICO,
    nlu.FUORI_FLUSSO: DUBBIO_TECNICO,  # "Richiesta fuori flusso o dubbio tecnico" no prompt
}

# Acoes que pedem um dado (e qual).
_PEDE_DADO = {PERGUNTAR, RICHIESTA_PREZZO, CHIEDI_FOTO}


@dataclass(frozen=True)
class Acao:
    tipo: str
    slot: str | None = None      # PERGUNTAR / RICHIESTA_PREZZO / CHIEDI_FOTO: qual dado pedir
    motivo: str | None = None    # TRANSFERIR: motivo da transferencia

    def para_trace(self) -> dict:
        return {"tipo": self.tipo, "slot": self.slot, "motivo": self.motivo}


def _pedidos_seguidos(slots: dict, slot: str, registrou_algo: bool) -> int:
    """Quantas vezes seguidas o dado tera sido pedido, contando esta."""
    if registrou_algo or slots.get(CHAVE_ULTIMO_PEDIDO) != slot:
        return 1
    return int(slots.get(CHAVE_PEDIDOS_SEGUIDOS) or 0) + 1


def decidir(
    slots: dict,
    entendido: nlu.Entendimento,
    *,
    primeira_resposta: bool = False,
    registrou_algo: bool = False,
) -> Acao:
    """`slots` ja inclui o que foi conferido neste turno; `registrou_algo` diz se este turno
    trouxe algum dado novo."""
    if entendido.foto_ricevuta:
        # O Matteo nao le imagens: a foto vai para o operador ("FOTO RICEVUTA").
        return Acao(TRANSFERIR, motivo=FOTO_MISURA)

    motivo = _TRANSFERE_JA.get(entendido.intencao)
    if motivo:
        return Acao(TRANSFERIR, motivo=motivo)

    faltando = campos_faltando(slots)
    if not faltando:
        return Acao(TRANSFERIR, motivo=DATI_COMPLETI)

    if entendido.chiede_prezzo and all(slots.get(c) for c in SUFICIENTES_PARA_PRECO):
        return Acao(TRANSFERIR, motivo=RICHIESTA_PREZZO_MOTIVO)

    if entendido.intencao == nlu.INCOMPRENSIBILE:
        if int(slots.get(CHAVE_CHIARIMENTI) or 0) >= MAX_CHIARIMENTI:
            return Acao(TRANSFERIR, motivo=POCO_CHIARO)
        return Acao(CHIARIRE)

    if LABEL_MISURA in faltando and (entendido.non_sa_la_misura or entendido.misura_non_riconosciuta):
        acao = Acao(CHIEDI_FOTO, slot=LABEL_MISURA)
    elif LABEL_MISURA in faltando and entendido.chiede_prezzo:
        acao = Acao(RICHIESTA_PREZZO, slot=LABEL_MISURA)
    elif primeira_resposta and not slots_visiveis(slots):
        # O texto de boas-vindas ja pede os 5 dados - conta como o primeiro pedido da medida.
        return Acao(BENVENUTO, slot=LABEL_MISURA)
    else:
        acao = Acao(PERGUNTAR, slot=faltando[0])

    if _pedidos_seguidos(slots, acao.slot, registrou_algo) > MAX_PEDIDOS_DO_MESMO_DADO:
        return Acao(TRANSFERIR, motivo=POCO_CHIARO)
    return acao


def atualizar_contadores(slots: dict, acao: Acao, *, registrou_algo: bool) -> dict:
    """Contadores que a proxima decisao le (gravados junto com os dados da conversa)."""
    novos = {CHAVE_CHIARIMENTI: int(slots.get(CHAVE_CHIARIMENTI) or 0) + 1 if acao.tipo == CHIARIRE else 0}
    if acao.tipo in _PEDE_DADO or acao.tipo == BENVENUTO:
        novos[CHAVE_PEDIDOS_SEGUIDOS] = _pedidos_seguidos(slots, acao.slot, registrou_algo)
        novos[CHAVE_ULTIMO_PEDIDO] = acao.slot
    elif acao.tipo != CHIARIRE:
        novos[CHAVE_PEDIDOS_SEGUIDOS] = 0
        novos[CHAVE_ULTIMO_PEDIDO] = None
    return novos
