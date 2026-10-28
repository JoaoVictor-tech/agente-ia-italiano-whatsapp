"""Riepilogo interno (nota na sessao da Helena) que o operador Gomme Fenice le ao receber o cliente.

Mesmo formato do prompt original ("DATI DA PASSARE ALL'OPERATORE"), para o operador ler sempre a
mesma estrutura. No motor v2 ele e' montado pelo CODIGO a partir dos dados conferidos (no motor
antigo o modelo escrevia o texto na ferramenta `criarResumo`). Quando a transferencia acontece
fora do motor (audio que nao transcreve, conteudo nao suportado, falha do sistema), a nota sai
marcada como automatica.
"""

from __future__ import annotations

# Motivos do passaggio do prompt, por motivo da politica (app/politica.py).
_ROTULOS = {
    "dati_completi": "Dati completi",
    "richiesta_prezzo": "Richiesta prezzo o disponibilità",
    "foto_misura": "Foto misura da leggere",
    "dubbio_tecnico": "Dubbio tecnico o messaggio poco chiaro",
    "messaggio_poco_chiaro": "Dubbio tecnico o messaggio poco chiaro",
    "richiesta_operatore": "Richiesta esplicita di operatore",
}

# Motivos de quando o SERVIDOR transfere sem passar pelo motor (app/pipeline.py).
MOTIVO_AUDIO_NON_TRASCRITTO = "Audio non trascrivibile"
MOTIVO_MESSAGGIO_NON_SUPPORTATO = "Contenuto non supportato"
MOTIVO_SISTEMA = "Assistente non disponibile (errore tecnico)"

_CAMPOS = ("Misura", "Quantità", "Stagione", "Nuovo/Usato", "Utilizzo")


def rotulo_do_motivo(motivo: str) -> str:
    return _ROTULOS.get(motivo, motivo)


def montar(
    *,
    motivo: str,
    dados_confirmados: dict | None = None,
    note: str | None = None,
    ultima_mensagem_do_cliente: str | None = None,
    automatico: bool = False,
) -> str:
    dados = dados_confirmados or {}
    linhas = [f"{campo}: {dados.get(campo) or 'Non raccolto'}" for campo in _CAMPOS]
    if note and note.strip():
        nota = note.strip()[:500]
    elif ultima_mensagem_do_cliente:
        nota = f'ultimo messaggio del cliente: "{ultima_mensagem_do_cliente.strip()[:300]}"'
    else:
        nota = "Nessuna"
    linhas += [
        f"Note cliente: {nota}",
        f"Motivo del passaggio: {motivo}",
        "Richiesta: verifica disponibilità + prezzo.",
    ]
    if automatico:
        linhas.append("(riepilogo automatico del sistema)")
    return "\n".join(linhas)
