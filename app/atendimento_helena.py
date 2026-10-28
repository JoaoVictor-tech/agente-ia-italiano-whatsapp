"""Quem está com a conversa AGORA, segundo a Helena - e não segundo o nosso banco.

Herdado do agente anterior: `conversations.stage` só vira "transferido" quando a transferência
passa pela NOSSA ferramenta. Quando um operador puxa a conversa direto na Helena, nada nos avisa
- a plataforma só chama nosso webhook em mensagem de cliente, nunca em ação de atendente. Sem
esta checagem o agente voltava a responder numa conversa que já era humana (num agente anterior, 53% das
conversas "ativas" no banco já estavam com um vendedor).

A autoridade sobre "de quem é a conversa" é a Helena, e a resposta vem do departamento da
sessão (`GET /chat/v2/session/{id}`): o departamento do bot (HELENA_*_BOT_DEPARTMENT_ID) contra
o departamento para onde transferimos (HELENA_*_DEPARTMENT_ID).
"""

from __future__ import annotations

import logging

from app.config import EnvironmentCredentials, get_settings
from app.helena_client import estado_da_sessao

logger = logging.getLogger(__name__)

BOT = "bot"
HUMANO = "humano"
DESCONHECIDO = "desconhecido"


def _departamento_do_bot(environment: str) -> str | None:
    """Só do .env (nunca editável pelo painel), igual aos demais ids de integração."""
    cfg = get_settings()
    if environment == "producao":
        return cfg.helena_prod_bot_department_id or None
    return cfg.helena_test_bot_department_id or None


async def quem_atende(
    *, credentials: EnvironmentCredentials, session_id: str, environment: str
) -> str:
    """BOT / HUMANO / DESCONHECIDO (quando a Helena não respondeu).

    Quem chama decide o que fazer com DESCONHECIDO: o pipeline RESPONDE (uma falha de rede não
    pode emudecer o agente com um cliente esperando)."""
    dto = await estado_da_sessao(credentials=credentials, session_id=session_id)
    if not dto:
        return DESCONHECIDO

    departamento = dto.get("departmentId") or ""
    do_bot = _departamento_do_bot(environment)

    if do_bot:
        return BOT if departamento == do_bot else HUMANO

    # Sem o id do departamento da IA configurado, ainda dá para reconhecer o caso que motivou
    # este módulo: a sessão está no departamento para onde NÓS transferimos, logo é humana.
    if credentials.department_id and departamento == credentials.department_id:
        return HUMANO
    return DESCONHECIDO
