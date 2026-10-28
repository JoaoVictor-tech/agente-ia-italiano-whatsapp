import asyncio
import contextlib
import logging

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request

from app import observabilidade
from app.config import ENVIRONMENTS
from app.dashboard.routes import router as dashboard_router
from app.orphan_recovery import loop_de_varredura
from app.pipeline import handle_incoming_webhook
from app.webhook_schemas import HelenaWebhookPayload

logging.basicConfig(level=logging.INFO)


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    """Sobe a varredura de mensagens orfas junto com a API (app/orphan_recovery.py) - um turno
    vive dentro de uma BackgroundTask, entao deploy/crash descarta o atendimento em andamento
    sem deixar rastro. A varredura fecha esse buraco, inclusive para o que a propria
    reinicializacao acabou de derrubar."""
    tarefas = [asyncio.create_task(loop_de_varredura())]
    try:
        yield
    finally:
        for tarefa in tarefas:
            tarefa.cancel()
        for tarefa in tarefas:
            with contextlib.suppress(asyncio.CancelledError):
                await tarefa
        # Esvazia o buffer do Langfuse antes de o processo morrer (no-op sem as chaves):
        # sem isto, os traces do ultimo minuto antes de um deploy se perdem.
        observabilidade.encerrar()


app = FastAPI(title="Agente Gomme Fenice", lifespan=lifespan)
app.include_router(dashboard_router, prefix="/dashboard")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/webhook/helena/{environment}")
async def webhook_helena(environment: str, request: Request, background_tasks: BackgroundTasks) -> dict:
    """Mesmo papel do node "Webhook" (POST agente-ia-testes) do fluxo n8n. Responde 200 na
    hora e processa o turno em background - a Helena nao espera a resposta do agente aqui; o
    texto/audio de volta e enviado depois via chamada HTTP propria (app/helena_client.py)."""
    if environment not in ENVIRONMENTS:
        raise HTTPException(status_code=404, detail=f"Ambiente desconhecido: {environment!r}")

    raw = await request.json()
    payload = HelenaWebhookPayload.model_validate(raw)

    if payload.lastMessage is None:
        # Evento de sistema sem mensagem de cliente (ex: troca de bot ativo via #teste-XXX
        # gera um webhook proprio da Helena sem lastMessage) - nada a processar.
        logging.getLogger(__name__).info(
            "Webhook Helena sem lastMessage (evento de sistema) - ignorando. sessionId=%s",
            payload.sessionId,
        )
        return {"received": True, "ignored": "no_last_message"}

    background_tasks.add_task(handle_incoming_webhook, payload, environment)

    return {"received": True}
