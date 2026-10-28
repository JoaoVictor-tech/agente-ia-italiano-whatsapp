"""Wrappers para os endpoints da plataforma Helena (api.helena.run / api.wts.chat).
Toda funcao recebe um `EnvironmentCredentials` (ver app/config.py) para poder operar
tanto no ambiente de teste quanto no de producao com as credenciais corretas de cada um.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from app.config import EnvironmentCredentials, get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Intervalo entre bolhas, so pra parecer digitacao humana no WhatsApp. Ate 2026-08-13 este
# delay tambem era a UNICA defesa contra bolha fora de ordem (com 2.0s ja tinha sido relatado
# em 2026-08-07, subiu pra 3.5s como mitigacao) - mas era so heuristica: o endpoint assincrono
# responde assim que a Helena ACEITA a mensagem, nao quando ela sai no WhatsApp, entao a ordem
# real de entrega nunca foi garantida. Agora o envio usa o endpoint /sync (ver _postar_bolha
# abaixo), que so responde depois da tentativa de envio no canal - a ordem passa a ser
# garantida pelo proprio protocolo e este delay volta a ser so cosmetico, por isso menor.
BUBBLE_DELAY_SECONDS = 1.2

# Timeout por bolha: o /sync bloqueia ate a tentativa de envio no canal (a doc da Helena cita
# ~25s de teto), entao o timeout precisa ficar confortavelmente acima disso.
BUBBLE_SEND_TIMEOUT_SECONDS = 45.0


async def _postar_bolha(client: httpx.AsyncClient, *, session_id: str, headers: dict, body: dict) -> None:
    """Posta UMA bolha usando o endpoint sincrono (`/message/sync`), que so retorna depois que a
    Helena tentou entregar a mensagem no canal - e o que garante que a proxima bolha so sai
    depois desta ter saido de fato (no agente anterior, bolhas chegavam fora de ordem no WhatsApp
    mesmo saindo em ordem daqui).

    Se por algum motivo o /sync nao existir/for rejeitado neste ambiente (404/405), cai de volta
    no endpoint assincrono original - assim uma eventual diferenca de versao da API nunca deixa
    o cliente sem resposta nenhuma, so sem a garantia de ordem."""
    base = f"{settings.helena_base_url}/chat/v1/session/{session_id}/message"
    # a API exige JSON aqui - "data=" (form-urlencoded) retorna 415.
    response = await client.post(f"{base}/sync", headers=headers, json=body)
    if response.status_code in (404, 405):
        logger.warning(
            "Endpoint /message/sync indisponivel (HTTP %s) - usando envio assincrono como fallback "
            "(sem garantia de ordem entre bolhas).",
            response.status_code,
        )
        response = await client.post(base, headers=headers, json=body)
    response.raise_for_status()


async def send_text_bubbles(*, credentials: EnvironmentCredentials, session_id: str, bubbles: list[str]) -> None:
    """Envia cada bolha como uma mensagem separada, com um pequeno intervalo entre elas."""
    headers = {"Authorization": credentials.message_token}

    async with httpx.AsyncClient(timeout=BUBBLE_SEND_TIMEOUT_SECONDS) as client:
        for index, bubble in enumerate(bubbles):
            if index > 0:
                await asyncio.sleep(BUBBLE_DELAY_SECONDS)
            await _postar_bolha(client, session_id=session_id, headers=headers, body={"text": bubble})


async def send_canned_text(*, credentials: EnvironmentCredentials, session_id: str, text: str) -> None:
    await send_text_bubbles(credentials=credentials, session_id=session_id, bubbles=[text])


async def send_audio_reply(*, credentials: EnvironmentCredentials, session_id: str, audio_bytes: bytes) -> None:
    """Replica o fluxo de resposta em audio: upload do mp3 -> registra o arquivo -> envia como
    mensagem de audio. Fluxo confirmado contra a documentacao oficial (helena.readme.io,
    2026-07-30) apos a 1a tentativa (POST no passo de init) ter retornado 415/500 - o passo de
    obter a URL de upload e GET, nao POST (POST nesse mesmo path e o passo seguinte, de
    confirmar o arquivo ja enviado - por isso vinha "ENTITY_NOT_FOUND" ao chamar como POST
    direto, sem nunca ter feito upload nenhum antes). O envio final usa o endpoint dedicado
    de audio (POST /chat/v1/send/audio), nao o endpoint generico de texto."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        upload_init = await client.get(
            f"{settings.helena_base_url}/core/v2/file",
            # "MimeType" e opcional na doc, mas sem ele o upload seguinte (PUT no S3) falha com
            # SignatureDoesNotMatch: a URL pre-assinada exige um Content-Type EXATO igual ao que
            # foi usado pra gerar a assinatura, e sem MimeType aqui o backend assina para um
            # content-type que o cliente nao tem como adivinhar. Confirmado ao vivo (2026-07-30).
            params={"Type": "AUDIO", "Name": "data.mp3", "MimeType": "audio/mpeg"},
            headers={"accept": "application/json", "Authorization": credentials.file_token},
        )
        upload_init.raise_for_status()
        init_data = upload_init.json()
        upload_url = init_data["urlUpload"]
        temp_file_id = init_data["tempFileId"]

        put_response = await client.put(upload_url, content=audio_bytes, headers={"Content-Type": "audio/mpeg"})
        put_response.raise_for_status()

        register_response = await client.post(
            f"{settings.helena_base_url}/core/v2/file",
            headers={"Authorization": credentials.file_token, "Content-Type": "application/json"},
            json={"tempFileId": temp_file_id},
        )
        register_response.raise_for_status()
        file_id = register_response.json()["id"]

        message_response = await client.post(
            f"{settings.helena_base_url}/chat/v1/send/audio",
            headers={"Authorization": credentials.message_token, "Content-Type": "application/json"},
            json={"fileIdOrUrl": file_id, "sessionId": session_id},
        )
        message_response.raise_for_status()


async def list_channels(*, credentials: EnvironmentCredentials) -> list[dict]:
    """Lista os canais/numeros cadastrados na conta Helena (usado na tela de Teste Rapido do
    dashboard, para escolher canal remetente/receptor numa lista real em vez de digitar a mao).

    ATENCAO: este endpoint usa `Authorization: Bearer {token}` - diferente da convencao sem
    prefixo usada pelas demais funcoes deste arquivo (que operam sobre uma sessao ja existente).
    Confirmado ao vivo (2026-07-29) contra a conta de teste compartilhada."""
    url = f"{settings.helena_base_url}/chat/v1/channel"
    headers = {"Authorization": f"Bearer {credentials.message_token}", "Accept": "application/json"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
    return data if isinstance(data, list) else []


async def send_as_customer(
    *, credentials: EnvironmentCredentials, from_phone: str, to_phone: str, text: str
) -> dict:
    """Envia uma mensagem de `from_phone` (canal simulando o cliente) para `to_phone` (canal do
    agente sendo testado), via `POST /chat/v1/message/send-sync` - usado pela tela de Teste
    Rapido para simular uma conversa real inteiramente dentro da Helena, sem precisar do
    WhatsApp. Numeros no formato aceito pela Helena (ex: "+55|11900000001").

    CONFIRMADO AO VIVO (2026-07-29): from=canal "cliente simulado", to=canal do agente-alvo -
    a entrega tem lag perceptivel (pode levar ate alguns minutos ate o webhook do agente-alvo
    ser chamado), entao quem chama esta funcao nao deve esperar resposta sincrona aqui."""
    url = f"{settings.helena_base_url}/chat/v1/message/send-sync"
    headers = {
        "Authorization": f"Bearer {credentials.message_token}",
        "Content-Type": "application/json",
    }
    body = {"from": from_phone, "to": to_phone, "body": {"text": text}}
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, headers=headers, json=body)
        response.raise_for_status()
        return response.json()


async def add_session_note(*, credentials: EnvironmentCredentials, session_id: str, text: str) -> None:
    """Cria uma nota interna dentro da conversa (visivel para quem assume o atendimento),
    endpoint 'Save Internal Note' (helena.readme.io/reference/post_v1-session-id-note)."""
    url = f"{settings.helena_base_url}/chat/v1/session/{session_id}/note"
    headers = {"Authorization": credentials.message_token}
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, headers=headers, json={"text": text})
        response.raise_for_status()


async def estado_da_sessao(
    *, credentials: EnvironmentCredentials, session_id: str
) -> dict | None:
    """DTO da sessao na Helena (`GET /chat/v2/session/{id}`) - traz `departmentId`, `status` e
    `userId`. Devolve None em qualquer falha: quem chama decide o que fazer sem informacao
    (ver app/atendimento_helena.py).

    `userId` NAO serve para saber se um humano assumiu - vem sempre o usuario da propria
    integracao, inclusive nas sessoes do bot (conferido na conta de outro agente). Quem separa e o
    `departmentId`."""
    url = f"{settings.helena_base_url}/chat/v2/session/{session_id}"
    headers = {"Authorization": credentials.message_token}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url, headers=headers)
            if response.status_code != 200:
                logger.warning(
                    "Helena devolveu HTTP %s ao consultar a sessao %s",
                    response.status_code, session_id,
                )
                return None
            return response.json()
    except Exception:
        logger.exception("Falha ao consultar o estado da sessao %s na Helena", session_id)
        return None


async def transferir_atendimento(*, credentials: EnvironmentCredentials, session_id: str) -> None:
    """Transfere a sessao para a equipe humana da Gomme Fenice (tool `transferirAtendimento`) - mesmo
    endpoint/corpo do node do fluxo n8n (`type: DEPARTMENT` + `stopBotInExecution: true`, para
    o bot parar de responder assim que transferir)."""
    if not credentials.department_id:
        logger.warning(
            "Ambiente '%s' sem department_id configurado - transferirAtendimento nao "
            "encaminhou a sessao %s para nenhuma equipe.",
            credentials.environment,
            session_id,
        )
        return
    url = f"{settings.helena_base_url}/chat/v1/session/{session_id}/transfer"
    headers = {"Authorization": credentials.message_token}
    body = {
        "type": "DEPARTMENT",
        "newDepartmentId": credentials.department_id,
        "options": {"stopBotInExecution": True},
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.put(url, headers=headers, json=body)
        response.raise_for_status()
