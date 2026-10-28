"""Base comum dos testes. Tudo roda com o DeepEval como executor:

    deepeval test run tests/                 # tudo
    deepeval test run tests/ -m "not llm"    # so o deterministico (sem banco, sem LLM, sem custo)
    deepeval test run tests/ -m llm          # so as conversas contra o LLM real

Os testes deterministicos sao pytest comum (assert): codigo sem LLM sempre da o mesmo
resultado, e uma metrica de juiz LLM ali so traria custo e oscilacao. Os de conversa
(tests/test_conversas.py) usam os casos e as metricas do DeepEval.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

# O modulo de config exige estas variaveis. Valores de brinquedo bastam para o deterministico -
# nada aqui abre conexao com banco, Redis ou Helena. Chave de LLM NAO entra aqui: a das
# conversas vem do .env (o pydantic-settings le o .env da raiz).
for _nome, _valor in {
    "SETTINGS_ENCRYPTION_KEY": "x" * 43 + "=",
    "HELENA_PROD_MESSAGE_TOKEN": "teste",
    "HELENA_PROD_TAG_TOKEN": "teste",
    "HELENA_PROD_FILE_TOKEN": "teste",
    "DATABASE_URL": "postgresql+asyncpg://x:x@localhost/x",
}.items():
    os.environ.setdefault(_nome, _valor)

# Langfuse DESLIGADO nos testes, a forca. Nao da para usar setdefault: o plugin do DeepEval e'
# importado pelo pytest ANTES deste arquivo e, ao importar, joga o .env inteiro em os.environ -
# com as chaves la, toda rodada de teste mandaria trace para producao.
# Para rastrear uma rodada de proposito: TESTES_COM_LANGFUSE=1.
if os.environ.get("TESTES_COM_LANGFUSE") != "1":
    os.environ["LANGFUSE_PUBLIC_KEY"] = ""
    os.environ["LANGFUSE_SECRET_KEY"] = ""
os.chdir(RAIZ)  # o .env e os prompts sao lidos relativos a raiz
# As conversas de teste carregam texto de atendimento: nada de telemetria para fora.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

import pytest  # noqa: E402

from app import agent_tools  # noqa: E402
from app.config import EnvironmentCredentials  # noqa: E402


def novo_contexto() -> agent_tools.ToolContext:
    return agent_tools.ToolContext(
        session_id="s1",
        credentials=EnvironmentCredentials(
            environment="teste", message_token="t", tag_token="t", file_token="t",
            panel_id=None, step_id_novo_lead=None, step_id_em_atendimento=None,
            department_id="dep",
        ),
        contact_id="c1",
    )


@pytest.fixture
def ctx() -> agent_tools.ToolContext:
    return novo_contexto()
