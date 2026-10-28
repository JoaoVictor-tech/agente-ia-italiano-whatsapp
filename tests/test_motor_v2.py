"""O grafo do turno (app/motor_v2.py) de ponta a ponta, com um modelo FALSO.

Sem rede e sem custo: o modelo falso devolve o JSON do entendimento e o texto da redacao que
cada teste quiser. O que se verifica aqui e' a costura - a extracao deterministica roda antes
do LLM, o que o modelo "extraiu" passa pela conferencia, a politica decide, a verificacao
barra e a redacao refaz UMA vez, e o reserva entra quando o provedor cai.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from langchain_core.messages import AIMessage

from app import agent_tools, motor_v2, politica
from app.agent_tools import LABEL_MISURA, LABEL_QUANTITA, LABEL_STAGIONE
from app.settings_service import ActiveLLMSettings
from tests.conftest import novo_contexto

PERSONA = "Sei Matteo, l'assistente di Gomme Fenice."


class ErroDeProvedor(Exception):
    """Finge o 429/503 do provedor (app/llm_providers/chat_models.erro_de_provedor)."""


class ModeloFalso:
    """Devolve, em ordem, as respostas que o teste programou.

    `extracoes`: dicts que viram o argumento da funcao `registrar_entendimento`.
    `redacoes`:  strings que viram o texto da resposta ao cliente.
    """

    def __init__(self, extracoes: list, redacoes: list, nome: str = "principal"):
        self.extracoes, self.redacoes, self.nome = list(extracoes), list(redacoes), nome
        self.chamadas: list[str] = []

    def bind_tools(self, tools, tool_choice=None):  # noqa: ARG002
        pai = self

        class Extrator:
            async def ainvoke(self, _mensagens):
                pai.chamadas.append("entender")
                valor = pai.extracoes.pop(0)
                if isinstance(valor, Exception):
                    raise valor
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "registrar_entendimento", "args": valor, "id": "1", "type": "tool_call",
                    }],
                )

        return Extrator()

    async def ainvoke(self, _mensagens):
        self.chamadas.append("redigir")
        valor = self.redacoes.pop(0)
        if isinstance(valor, Exception):
            raise valor
        if isinstance(valor, AIMessage):
            return valor
        return AIMessage(content=valor)


def _entendimento(**campos) -> dict:
    base = {
        "misura": "", "quantita": 0, "stagione": "", "condizione": "", "utilizzo": "",
        "intencao": "conversa", "chiede_prezzo": False, "non_sa_la_misura": False,
        "domanda_del_cliente": "", "note_per_operatore": "",
    }
    base.update(campos)
    return base


def _settings(provider: str = "gemini", model: str = "gemini-2.5-flash") -> ActiveLLMSettings:
    return ActiveLLMSettings(provider=provider, model=model, api_key="chave", system_prompt=PERSONA)


def rodar(monkeypatch, modelo, *, user_message: str, history=None, slots=None, ctx=None, reserva=None):
    """Roda um turno com o modelo falso no lugar do chat model do LangChain."""
    modelos = [modelo] + ([reserva] if reserva is not None else [])
    monkeypatch.setattr(motor_v2, "criar_modelo", lambda s, **kw: modelos.pop(0))  # noqa: ARG005
    ctx = ctx or novo_contexto()
    resultado = asyncio.run(motor_v2.run_turn(
        history=history or [],
        user_message=user_message,
        llm_settings=_settings(),
        tool_context=ctx,
        dados_confirmados=slots,
        llm_settings_fallback=_settings("gemini", "gemini-3.5-flash") if reserva is not None else None,
    ))
    return resultado, ctx


def test_turno_simples_pede_o_proximo_dado(monkeypatch):
    modelo = ModeloFalso(
        [_entendimento(misura="205/55 R16")],
        ["Perfetto 👍 Mi manca solo la quantità (2 o 4) e ci siamo."],
    )
    resultado, ctx = rodar(monkeypatch, modelo, user_message="ho bisogno di 205/55 R16")

    assert ctx.slots[LABEL_MISURA] == "205/55 R16"
    assert ctx.trace["acao"] == {"tipo": politica.PERGUNTAR, "slot": LABEL_QUANTITA, "motivo": None}
    assert not ctx.transferido
    assert resultado.grounding_valido
    assert modelo.chamadas == ["entender", "redigir"]


def test_dado_inventado_pelo_modelo_nao_vira_slot(monkeypatch):
    """O modelo "extraiu" 4 gomme que o cliente nunca escreveu."""
    modelo = ModeloFalso(
        [_entendimento(misura="205/55 R16", quantita=4)],
        ["Perfetto 👍 Mi manca solo la quantità (2 o 4) e ci siamo."],
    )
    _, ctx = rodar(monkeypatch, modelo, user_message="ho bisogno di 205/55 R16")

    assert LABEL_QUANTITA not in ctx.slots
    assert LABEL_QUANTITA in ctx.trace["descartado_na_validacao"]


def test_dados_completos_transferem_e_guardam_o_motivo(monkeypatch):
    modelo = ModeloFalso(
        [_entendimento(utilizzo="Città", note_per_operatore="Ha fretta, parte venerdì.")],
        ["Perfetto 👍 ho tutto!\nOra verifichiamo la soluzione Gomme Fenice più adatta alle tue esigenze."],
    )
    slots = {LABEL_MISURA: "205/55 R16", LABEL_QUANTITA: "4", LABEL_STAGIONE: "Invernale", "Nuovo/Usato": "Nuovo"}
    _, ctx = rodar(monkeypatch, modelo, user_message="giro solo in citta", slots=slots)

    assert ctx.transferido
    assert ctx.motivo_transferencia == politica.DATI_COMPLETI
    assert ctx.note_operatore == "Ha fretta, parte venerdì."


def test_foto_transfere_sem_o_modelo_decidir(monkeypatch):
    """Quem marca `foto_ricevuta` e' o CODIGO, pelo marcador do pipeline - nao o modelo."""
    modelo = ModeloFalso(
        [_entendimento()],
        ["Perfetto 👍 ho ricevuto la foto.\nPasso la tua richiesta a un operatore che ti aiuta."],
    )
    _, ctx = rodar(monkeypatch, modelo, user_message=f"{motor_v2.PREFIXO_FOTO} Il cliente ha inviato una foto.")

    assert ctx.transferido
    assert ctx.motivo_transferencia == politica.FOTO_MISURA


def test_pedido_de_operador_transfere_mesmo_se_o_modelo_nao_marcar(monkeypatch):
    """Rede de seguranca: o modelo classificou como "conversa", mas o cliente escreveu com
    todas as letras que quer uma pessoa."""
    modelo = ModeloFalso(
        [_entendimento(intencao="conversa")],
        ["Perfetto 👍 passo la tua richiesta a un esperto di Gomme Fenice, che continua con te."],
    )
    _, ctx = rodar(monkeypatch, modelo, user_message="voglio parlare con un operatore")

    assert ctx.transferido
    assert ctx.motivo_transferencia == politica.RICHIESTA_OPERATORE


def test_verificacao_barra_e_a_redacao_refaz_uma_vez(monkeypatch):
    """Primeira redacao cita preco (proibido); a segunda passa - e o cliente recebe a boa."""
    modelo = ModeloFalso(
        [_entendimento(misura="205/55 R16")],
        [
            "Costano 89,90 € l'una. Quante te ne servono?",
            "Perfetto 👍 Mi manca solo la quantità (2 o 4) e ci siamo.",
        ],
    )
    resultado, ctx = rodar(monkeypatch, modelo, user_message="ho bisogno di 205/55 R16")

    assert modelo.chamadas == ["entender", "redigir", "redigir"]
    assert "€" not in resultado.final_text
    # O evento continua registrado, mesmo com a reescrita tendo dado certo.
    assert not resultado.grounding_valido
    assert "prezzi" in resultado.grounding_motivo
    assert ctx.trace["verificacao_final"] is None


def test_barrou_duas_vezes_usa_o_testo_approvato(monkeypatch):
    """Nunca sai um texto proibido para o cliente, e nunca sai o fallback generico: sai o
    testo approvato da acao que a politica decidiu."""
    modelo = ModeloFalso(
        [_entendimento(misura="205/55 R16")],
        ["Costano 89,90 €.", "Te le do a 85 euro."],
    )
    resultado, _ = rodar(monkeypatch, modelo, user_message="ho bisogno di 205/55 R16")

    assert resultado.final_text == "Perfetto 👍 Mi manca solo la quantità (2 o 4) e ci siamo."
    assert "€" not in resultado.final_text


def test_resposta_cortada_pelo_limite_de_tokens_e_refeita(monkeypatch):
    cortada = AIMessage(content="Perfetto 👍 Mi manca solo la quantit", response_metadata={"finish_reason": "MAX_TOKENS"})
    modelo = ModeloFalso(
        [_entendimento(misura="205/55 R16")],
        [cortada, "Perfetto 👍 Mi manca solo la quantità (2 o 4) e ci siamo."],
    )
    resultado, _ = rodar(monkeypatch, modelo, user_message="ho bisogno di 205/55 R16")

    assert modelo.chamadas == ["entender", "redigir", "redigir"]
    assert resultado.final_text.endswith("e ci siamo.")


def test_provedor_indisponivel_cai_para_o_reserva(monkeypatch):
    import httpx
    import openai

    # O 5xx/sobrecarga chega embrulhado pelo SDK; e' o que `erro_de_provedor` reconhece.
    erro = openai.APIConnectionError(request=httpx.Request("POST", "https://api.exemplo"))
    principal = ModeloFalso([erro], [], nome="principal")
    reserva = ModeloFalso(
        [_entendimento(misura="205/55 R16")],
        ["Perfetto 👍 Mi manca solo la quantità (2 o 4) e ci siamo."],
        nome="reserva",
    )
    resultado, _ = rodar(monkeypatch, principal, user_message="205/55 R16", reserva=reserva)

    assert resultado.fallback_usado
    assert resultado.modelo_usado == "gemini:gemini-3.5-flash"
    assert reserva.chamadas == ["entender", "redigir"]


def test_erro_que_nao_e_do_provedor_nao_aciona_o_reserva(monkeypatch):
    """Um 400 de payload invalido aconteceria igual no outro modelo - trocar so gastaria
    dinheiro e latencia."""
    principal = ModeloFalso([ValueError("schema invalido")], [])
    reserva = ModeloFalso([_entendimento()], ["Ciao"])
    with pytest.raises(ValueError):
        rodar(monkeypatch, principal, user_message="205/55 R16", reserva=reserva)
    assert reserva.chamadas == []


def test_historico_decide_se_e_a_primeira_resposta(monkeypatch):
    """Com atendente ja tendo falado, nao e' mais o BENVENUTO."""
    modelo = ModeloFalso([_entendimento()], ["Perfetto 👍 Mi manca solo la misura e ci siamo."])
    historico = [
        {"role": "user", "content": "ciao"},
        {"role": "assistant", "content": "Ciao 👋 sono Matteo."},
    ]
    _, ctx = rodar(monkeypatch, modelo, user_message="mi servono gomme", history=historico)
    assert ctx.trace["acao"]["tipo"] == politica.PERGUNTAR


def test_trace_do_turno_tem_tudo_que_o_painel_mostra(monkeypatch):
    modelo = ModeloFalso([_entendimento(misura="205/55 R16")], ["Perfetto 👍 Mi manca solo la quantità e ci siamo."])
    _, ctx = rodar(monkeypatch, modelo, user_message="205/55 R16")

    assert ctx.trace["motor"] == "v2"
    assert ctx.trace["entendimento"]["dados"] == {LABEL_MISURA: "205/55 R16"}
    # O trace vai para o banco (evento decisao_do_turno) e para o Langfuse: tem de ser JSON.
    json.dumps(ctx.trace, ensure_ascii=False)


def test_contadores_ficam_nos_slots_e_nao_vazam_para_o_operador(monkeypatch):
    modelo = ModeloFalso([_entendimento()], ["Perfetto 👍 Mi manca solo la misura e ci siamo."])
    _, ctx = rodar(monkeypatch, modelo, user_message="ciao")

    assert politica.CHAVE_ULTIMO_PEDIDO in ctx.slots
    assert agent_tools.slots_visiveis(ctx.slots) == {}


def test_correcao_do_cliente_e_sinalizada_para_a_redacao(monkeypatch):
    """O cliente troca "estive" por "invernali": o dado novo vale, e a redacao PRECISA saber
    que houve troca - senao ela repete a mesma frase do turno anterior e o cliente acha que
    nao foi lido (pego pela suite de conversas em 20/10/2026)."""
    blocos: list[str] = []
    original = motor_v2.redacao.montar_bloco

    def espiar(**kwargs):
        bloco = original(**kwargs)
        blocos.append(bloco)
        return bloco

    monkeypatch.setattr(motor_v2.redacao, "montar_bloco", espiar)
    modelo = ModeloFalso(
        [_entendimento(stagione="Invernale")],
        ["Perfetto 👍 invernali allora. Mi manca solo sapere se le preferisci nuove o usate."],
    )
    _, ctx = rodar(
        monkeypatch, modelo,
        user_message="scusa, anzi no: invernali",
        history=[{"role": "user", "content": "2 gomme estive 225/45 R17"},
                 {"role": "assistant", "content": "Perfetto 👍 Mi manca solo..."}],
        slots={LABEL_MISURA: "225/45 R17", LABEL_QUANTITA: "2", LABEL_STAGIONE: "Estivo"},
    )
    assert ctx.slots[LABEL_STAGIONE] == "Invernale"
    assert "CORRETTO" in blocos[0]
    assert "prima aveva detto Estivo" in blocos[0]


def test_dado_novo_nao_e_tratado_como_correcao(monkeypatch):
    blocos: list[str] = []
    original = motor_v2.redacao.montar_bloco
    monkeypatch.setattr(
        motor_v2.redacao, "montar_bloco",
        lambda **kw: (lambda b: (blocos.append(b), b)[1])(original(**kw)),
    )
    modelo = ModeloFalso([_entendimento(stagione="Invernale")], ["Perfetto 👍 Mi manca solo la quantità."])
    rodar(monkeypatch, modelo, user_message="invernali", slots={LABEL_MISURA: "225/45 R17"})
    assert "CORRETTO" not in blocos[0]
