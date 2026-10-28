"""Conversa completa com o Matteo no terminal, SEM WhatsApp, SEM Helena e SEM banco.

Usa o motor de verdade (app/motor_v2.py: entender -> decidir -> redigir -> verificar), o
modelo e a persona do .env (GEMINI_API_KEY e app/prompts/persona.md). A transferencia e a nota
interna sao so IMPRESSAS - nada sai para a Helena.

Uso:
    docker compose exec api python scripts/simular_conversa.py                      # interativo
    docker compose exec api python scripts/simular_conversa.py "Ciao" "205/55 R16, 4, estive, nuove, città"
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import agent_tools, llm, pipeline, resumo_atendimento  # noqa: E402
from app.config import EnvironmentCredentials, get_settings  # noqa: E402
from app.settings_service import ActiveLLMSettings, default_system_prompt  # noqa: E402


def _llm_settings(model: str) -> ActiveLLMSettings:
    cfg = get_settings()
    return ActiveLLMSettings(provider="gemini", model=model, api_key=cfg.gemini_api_key, system_prompt=default_system_prompt())


async def turno(history: list[dict], dados: dict, mensagem: str) -> bool:
    cfg = get_settings()
    credenciais = EnvironmentCredentials("teste", "simulacao", "simulacao", "simulacao", None, None, None, None)
    contexto = agent_tools.ToolContext(session_id="simulacao", credentials=credenciais)
    resultado = await llm.run_turn(
        history=history,
        user_message=mensagem,
        llm_settings=_llm_settings(cfg.default_llm_model),
        tool_context=contexto,
        dados_confirmados=dados,
        llm_settings_fallback=_llm_settings(cfg.fallback_llm_model) if cfg.fallback_llm_model else None,
    )
    texto, transferir = pipeline._texto_da_passagem(resultado, contexto, mensagem)
    acao = (contexto.trace or {}).get("acao") or {}
    print(f"\n👤 {mensagem}\n🤖 {texto}")
    print(f"   [{resultado.modelo_usado} · ação: {acao.get('tipo')}{'/' + acao['slot'] if acao.get('slot') else ''}"
          f" · US$ {resultado.usage_cost.cost_usd:.5f}"
          f"{' · reescrita: ' + ','.join(resultado.motivos_retry) if resultado.motivos_retry else ''}]")
    descartado = (contexto.trace or {}).get("descartado_na_validacao")
    if descartado:
        print(f"   ⚠ descartado na validação: {descartado}")
    history += [{"role": "user", "content": mensagem}, {"role": "assistant", "content": texto}]
    dados.clear()
    dados.update(contexto.slots)
    if transferir:
        print(f"   ↪ PASSAGGIO ALL'OPERATORE (motivo: {contexto.motivo_transferencia})")
        nota = resumo_atendimento.montar(
            motivo=resumo_atendimento.rotulo_do_motivo(contexto.motivo_transferencia or ""),
            dados_confirmados=contexto.slots, note=contexto.note_operatore,
        )
        print("   📋 riepilogo:\n      " + nota.replace("\n", "\n      "))
        return True
    return False


async def main(mensagens: list[str]) -> None:
    history: list[dict] = []
    dados: dict = {}
    if mensagens:
        for mensagem in mensagens:
            if await turno(history, dados, mensagem):
                return
        return
    while True:
        try:
            mensagem = input("\nvocê> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if mensagem and await turno(history, dados, mensagem):
            return


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
