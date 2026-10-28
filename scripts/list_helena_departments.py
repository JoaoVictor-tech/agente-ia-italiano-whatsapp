"""Lista os departamentos (equipes) da conta Helena, para descobrir os ids que vao no `.env`:

    HELENA_<AMBIENTE>_DEPARTMENT_ID       <- a equipe que recebe o cliente na transferencia
    HELENA_<AMBIENTE>_BOT_DEPARTMENT_ID   <- a fila onde o proprio Matteo trabalha

Uso (precisa do token de mensagens do ambiente ja preenchido no .env):

    docker compose exec api python scripts/list_helena_departments.py teste
    docker compose exec api python scripts/list_helena_departments.py producao
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db import async_session  # noqa: E402
from app.settings_service import get_environment_credentials  # noqa: E402


async def main(environment: str) -> None:
    settings = get_settings()
    async with async_session() as db:
        credentials = await get_environment_credentials(db, environment)

    # /v2 devolve um ARRAY puro (sem o envelope {items, ...} do resto da API).
    url = f"{settings.helena_base_url}/core/v2/department"
    # Este endpoint usa `Authorization: Bearer {token}` - diferente da convencao sem prefixo
    # usada pelos endpoints que operam sobre uma sessao ja existente (ver app/helena_client.py).
    headers = {"Authorization": f"Bearer {credentials.message_token}", "Accept": "application/json"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        resposta = await client.get(url, headers=headers)
        resposta.raise_for_status()
        dados = resposta.json()

    itens = dados.get("items") if isinstance(dados, dict) else dados
    if not itens:
        print("Nenhum departamento encontrado (ou a resposta veio num formato inesperado).")
        print(f"Resposta crua: {dados!r}")
        return

    print(f"\nDepartamentos no ambiente '{environment}':\n")
    for item in itens:
        nome = item.get("name") or item.get("title") or "(sem nome)"
        # `agents[]` lista usuarios que NAO sao membros - membro e so quem tem isAgent = true.
        membros = sum(1 for a in (item.get("agents") or []) if a.get("isAgent"))
        print(f"  {item.get('id')}  {nome}  ({membros} membros, distribuicao automatica: {item.get('distribuitionEnabled')})")
    print(
        "\nCole cada id no campo correspondente do .env. A fila em que o BOT trabalha vai em "
        "BOT_DEPARTMENT_ID - e ela que permite ao agente saber se um operador ja assumiu a "
        "conversa (app/atendimento_helena.py)."
    )


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("teste", "producao"):
        print("Uso: python scripts/list_helena_departments.py teste|producao")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
