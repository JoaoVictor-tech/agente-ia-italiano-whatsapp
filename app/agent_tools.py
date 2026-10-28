"""Dados da qualificacao do Matteo (Gomme Fenice): os 5 campos, a leitura deterministica do que o
cliente escreveu, a conferencia do que o entendimento (LLM) extraiu e a execucao da
transferencia na Helena.

No motor antigo (loop de tool-calling, removido em 10/2026) o modelo chamava `criarResumo` e
`transferirAtendimento` quando achava que era a hora. No motor v2 (app/motor_v2.py) quem
decide e' o codigo (app/politica.py); o riepilogo e' montado pelo codigo a partir dos dados
conferidos aqui, no mesmo formato "DATI DA PASSARE ALL'OPERATORE" do prompt original.

Os nomes dos campos sao os rotulos do riepilogo ("Misura", "Quantità"...) - e' assim que eles
ja estao gravados em `conversations.slots` desde a migracao do n8n, entao o banco nao muda.
Chaves que comecam com "_" sao contadores internos da politica, nunca mostrados ao modelo.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field

from app import helena_client, resumo_atendimento
from app.config import EnvironmentCredentials

logger = logging.getLogger(__name__)

# Rotulos dos dados (batem com o formato do riepilogo do prompt e de app/resumo_atendimento.py).
LABEL_MISURA = "Misura"
LABEL_QUANTITA = "Quantità"
LABEL_STAGIONE = "Stagione"
LABEL_CONDIZIONE = "Nuovo/Usato"
LABEL_UTILIZZO = "Utilizzo"
# Ordem de coleta (a mesma do "MANCA UN DATO" que o n8n seguia): um dado por mensagem.
CAMPOS = (LABEL_MISURA, LABEL_QUANTITA, LABEL_STAGIONE, LABEL_CONDIZIONE, LABEL_UTILIZZO)

# Valores normalizados dos campos categoricos (os do prompt: "Estivo / Invernale / 4 stagioni",
# "Nuovo / Usato", "Città / Misto / Tanti km").
STAGIONI = ("Estivo", "Invernale", "4 stagioni")
CONDIZIONI = ("Nuovo", "Usato")
UTILIZZI = ("Città", "Misto", "Tanti km")
VALORES = {LABEL_STAGIONE: STAGIONI, LABEL_CONDIZIONE: CONDIZIONI, LABEL_UTILIZZO: UTILIZZI}


def campos_faltando(slots: dict) -> list[str]:
    return [c for c in CAMPOS if not slots.get(c)]


def slots_visiveis(slots: dict) -> dict:
    return {k: v for k, v in (slots or {}).items() if k in CAMPOS and v}


def sem_acento(texto: str | None) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", texto or "") if unicodedata.category(ch) != "Mn"
    ).lower()


# --- Misura -------------------------------------------------------------------------------
# Formatos aceitos pelo prompt ("RICONOSCIMENTO DELLA MISURA"): 225/55R18, 225/55 R18,
# 225/55/18, 225 55 18 e a sequencia colada de 6-7 digitos (1756514, 2254517).
_REGEX_MISURA = re.compile(
    r"(?<!\d)(\d{3})\s*[/\s.\-]\s*(\d{2})\s*(?:[/\s.\-]|\s*Z?R\s*)\s*(\d{2})(?!\d)", re.IGNORECASE
)
_REGEX_MISURA_COLADA = re.compile(r"(?<!\d)(\d{3})(\d{2})(\d{2})(?!\d)")


def _misure(texto: str | None) -> list[tuple[int, str]]:
    achadas = []
    for padrao in (_REGEX_MISURA, _REGEX_MISURA_COLADA):
        for m in padrao.finditer(texto or ""):
            largura, perfil, aro = m.groups()
            if 125 <= int(largura) <= 355 and 25 <= int(perfil) <= 85 and 12 <= int(aro) <= 24:
                achadas.append((m.start(), f"{largura}/{perfil} R{aro}"))
    return sorted(achadas)


def normalizar_misura(texto: str | None) -> str | None:
    """Ultima medida de pneu escrita no texto, no formato '205/55 R16'. None se nao houver."""
    achadas = _misure(texto)
    return achadas[-1][1] if achadas else None


def _sem_misure(texto: str) -> str:
    for padrao in (_REGEX_MISURA, _REGEX_MISURA_COLADA):
        texto = padrao.sub(" ", texto)
    return texto


# --- Quantita -----------------------------------------------------------------------------
_NUMEROS = {"una": 1, "uno": 1, "due": 2, "paio": 2, "coppia": 2, "tre": 3, "quattro": 4, "cinque": 5, "sei": 6, "otto": 8}
# "4 stagioni" e "quattro stagioni" sao estacao, nao quantidade.
_QUATRO_STAGIONI = re.compile(r"\b(?:4|quattro)\s*stagion\w*", re.IGNORECASE)
_QTD_COM_UNIDADE = re.compile(
    r"\b(\d{1,2}|una|uno|due|tre|quattro|cinque|sei|otto)\s+(?:gomm|pneumatic|ruot|cerchi)", re.IGNORECASE
)
_PAIO = re.compile(r"\b(?:un\s+paio|una\s+coppia)\b", re.IGNORECASE)


def _texto_para_quantita(texto: str) -> str:
    return _QUATRO_STAGIONI.sub(" ", _sem_misure(sem_acento(texto)))


def extrair_quantita(texto: str | None) -> int | None:
    """Quantidade escrita com unidade ("4 gomme", "quattro pneumatici", "un paio")."""
    limpo = _texto_para_quantita(texto or "")
    achadas = [m.group(1) for m in _QTD_COM_UNIDADE.finditer(limpo)]
    if achadas:
        valor = achadas[-1]
        n = int(valor) if valor.isdigit() else _NUMEROS[valor]
        return n if 1 <= n <= 20 else None
    if _PAIO.search(limpo):
        return 2
    return None


def quantita_citata(texto: str | None, n: int) -> bool:
    """O numero `n` aparece no texto do cliente (fora de medida e de "4 stagioni")?"""
    limpo = _texto_para_quantita(texto or "")
    if re.search(rf"(?<![\d/]){n}(?![\d/])", limpo):
        return True
    palavras = {w for w, v in _NUMEROS.items() if v == n}
    return any(re.search(rf"\b{w}\b", limpo) for w in palavras) or (n == 4 and "tutte e quattro" in limpo)


# --- Campos categoricos -------------------------------------------------------------------
# Palavras do cliente que sustentam cada valor. Servem para DUAS coisas: extracao direta
# (quando a mensagem cita um unico valor do campo) e conferencia do que o LLM extraiu (o valor
# so e' aceito se o cliente escreveu algo que o sustente).
_PALAVRAS = {
    LABEL_STAGIONE: {
        "Estivo": r"\bestiv[oiea]\b|\bestate\b|\bsummer\b",
        "Invernale": r"\binvernal[ie]\b|\binverno\b|\bneve\b|\btermic[ihe]\b|\bwinter\b",
        "4 stagioni": r"\b(?:4|quattro)\s*stagion\w*|all[\s-]?season|tutto l'?anno|tutte le stagioni|quattrostagioni",
    },
    LABEL_CONDIZIONE: {
        "Nuovo": r"\bnuov[oiea]\b",
        "Usato": r"\busat[oiea]\b|seconda mano",
    },
    LABEL_UTILIZZO: {
        "Città": r"\bcitta\b|\burban[oiea]\b|in paese|tragitti brevi|casa[\s-]lavoro",
        "Misto": r"\bmist[oa]\b|extraurban|un po' di tutto|un po' tutto",
        "Tanti km": r"tanti\s*km|molti\s*km|tanti\s*chilometri|molti\s*chilometri|autostrad|"
        r"\bviaggi|lunghe percorrenze|\d+\s*(?:\.?\d{3})?\s*(?:mila\s*)?km\b",
    },
}


def valores_citados(campo: str, texto: str | None) -> set[str]:
    limpo = sem_acento(texto)
    if campo == LABEL_UTILIZZO:
        limpo = _QUATRO_STAGIONI.sub(" ", limpo)
    return {valor for valor, padrao in _PALAVRAS[campo].items() if re.search(padrao, limpo)}


def normalizar_valor(campo: str, valor) -> str | None:
    """Texto do LLM -> valor canonico do campo ("invernali" -> "Invernale"). None se nao bate."""
    if not isinstance(valor, str) or not valor.strip():
        return None
    limpo = sem_acento(valor).strip(" .")
    for canonico in VALORES[campo]:
        if limpo == sem_acento(canonico):
            return canonico
    citados = valores_citados(campo, valor)
    return next(iter(citados)) if len(citados) == 1 else None


# --- Extracao deterministica e conferencia -----------------------------------------------


def pre_registrar_do_texto(slots: dict, texto: str) -> dict:
    """Extracao barata, testada e que nao alucina: medida, quantidade com unidade e os campos
    categoricos que a mensagem cita sem ambiguidade. O LLM complementa o resto. Correcao do
    cliente ("anzi no, invernali") sobrescreve - vale a ultima informacao (regra do prompt)."""
    registrado = {}
    misura = normalizar_misura(texto)
    if misura:
        registrado[LABEL_MISURA] = misura
    quantita = extrair_quantita(texto)
    if quantita:
        registrado[LABEL_QUANTITA] = str(quantita)
    for campo in VALORES:
        citados = valores_citados(campo, texto)
        if len(citados) == 1:
            registrado[campo] = citados.pop()
    slots.update(registrado)
    return registrado


def registrar(slots: dict, dados: dict, texto_cliente: str) -> dict:
    """Confere cada valor que o entendimento (LLM) extraiu contra a fala real do cliente
    nesta mensagem e grava so o que se sustenta. Devolve {"registrado": {...},
    "nao_registrado": {campo: motivo}}."""
    registrado, recusado = {}, {}
    for campo, valor in dados.items():
        if campo == LABEL_MISURA:
            normalizada = normalizar_misura(str(valor))
            if not normalizada:
                recusado[campo] = f"medida fora dos formatos aceitos: {valor!r}"
            elif normalizada not in {m for _, m in _misure(texto_cliente)}:
                recusado[campo] = f"o cliente nao escreveu {normalizada} nesta mensagem"
            else:
                registrado[campo] = normalizada
        elif campo == LABEL_QUANTITA:
            n = valor if isinstance(valor, int) and not isinstance(valor, bool) else None
            if not n or not 1 <= n <= 20:
                recusado[campo] = f"quantidade invalida: {valor!r}"
            elif not quantita_citata(texto_cliente, n):
                recusado[campo] = f"o cliente nao escreveu {n} nesta mensagem"
            else:
                registrado[campo] = str(n)
        elif campo in VALORES:
            canonico = normalizar_valor(campo, valor)
            if canonico is None:
                recusado[campo] = f"valor fora da lista: {valor!r}"
            elif canonico not in valores_citados(campo, texto_cliente):
                recusado[campo] = f"o cliente nao disse nada que indique {canonico!r}"
            else:
                registrado[campo] = canonico
    slots.update(registrado)
    return {"registrado": registrado, "nao_registrado": recusado}


# --- Contexto do turno --------------------------------------------------------------------


@dataclass
class ToolContext:
    """Estado de um turno que nao vem do LLM: de quem e' a sessao, os dados ja conferidos e o
    que a politica decidiu (transferir, motivo). O pipeline le daqui depois do turno."""

    session_id: str
    credentials: EnvironmentCredentials
    contact_id: str | None = None
    # Dados ja conferidos nesta conversa (conversations.slots) + contadores "_..." da politica.
    slots: dict = field(default_factory=dict)
    # A politica decidiu transferir neste turno (executado pelo pipeline DEPOIS do envio).
    transferido: bool = field(default=False, init=False)
    motivo_transferencia: str | None = field(default=None, init=False)
    # Observacao para o operador (campo "Note cliente" do riepilogo), vinda do entendimento.
    note_operatore: str | None = field(default=None, init=False)
    # O PUT /transfer de fato aconteceu.
    transferencia_executada: bool = field(default=False, init=False)
    ultima_mensagem_do_cliente: str | None = field(default=None, init=False)
    # O que o motor entendeu e decidiu (vai para o painel e para o Langfuse).
    trace: dict | None = field(default=None, init=False)


async def executar_transferencia(context: ToolContext, *, motivo_sem_nota: str | None = None) -> bool:
    """Nota interna (riepilogo) e depois PUT /transfer - a ordem obrigatoria do prompt
    (criarResumo -> transferirAtendimento). A nota nunca impede a transferencia. Devolve True
    se a Helena confirmou a transferencia."""
    if context.transferencia_executada:
        return True
    if context.motivo_transferencia:
        texto = resumo_atendimento.montar(
            motivo=resumo_atendimento.rotulo_do_motivo(context.motivo_transferencia),
            dados_confirmados=context.slots,
            note=context.note_operatore,
        )
    else:
        texto = resumo_atendimento.montar(
            motivo=motivo_sem_nota or resumo_atendimento.MOTIVO_SISTEMA,
            dados_confirmados=context.slots,
            ultima_mensagem_do_cliente=context.ultima_mensagem_do_cliente,
            automatico=True,
        )
    try:
        await helena_client.add_session_note(credentials=context.credentials, session_id=context.session_id, text=texto)
    except Exception:
        logger.exception("Falha ao criar a nota interna da sessao %s", context.session_id)
    try:
        await helena_client.transferir_atendimento(credentials=context.credentials, session_id=context.session_id)
    except Exception:
        logger.exception("Falha ao transferir a sessao %s", context.session_id)
        return False
    context.transferencia_executada = True
    return True
