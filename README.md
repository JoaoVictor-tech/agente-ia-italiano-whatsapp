# Agente de IA no WhatsApp que atende em italiano: qualificação de leads de pneus

O **Matteo** é o atendente virtual de WhatsApp de uma loja de pneus na Itália. Ele conversa
**somente em italiano**, coleta os cinco dados de que o vendedor precisa para fazer a proposta e
passa o cliente para a equipe humana com um resumo pronto. Ele **nunca** informa preço,
disponibilidade ou marca: isso é trabalho do operador.

| Dado | Exemplo |
| --- | --- |
| **misura** (medida) | `205/55 R16` (aceita "205 55 16", "2055516", "225/55ZR17") |
| **quantità** | "4 gomme", "un paio" |
| **stagione** | Estivo, Invernale ou 4 stagioni |
| **condizione** | Nuovo ou Usato |
| **utilizzo** | Città, Misto ou Tanti km |

Foi desenvolvido durante meu estágio para um cliente real. Primeiro migrei o agente de um fluxo
do n8n para um serviço em Python e depois reestruturei o motor com LangGraph. Esta é uma versão
de portfólio: a loja (**Gomme Fenice**), a persona e os contatos são fictícios, e o repositório
não tem nenhuma credencial nem dado de cliente.

```
Cliente:  ciao
Matteo:   Ciao 👋 sono Matteo, l'assistente di Gomme Fenice.
          Ti aiuto a trovare il pneumatico giusto per la tua auto. Mi servono solo:
          🛞 Misura · 🔢 Quantità · ☀️ Stagione · ♻️ Nuovo o usato · 🚗 Uso
Cliente:  4 gomme invernali nuove, 205 55 16
Matteo:   Perfetto 👍 Mi manca solo l'uso (città, misto o tanti km) e ci siamo.
Cliente:  faccio tanti km in autostrada
Matteo:   Perfetto 👍 ho tutto!
          Ora verifichiamo la soluzione Gomme Fenice più adatta alle tue esigenze per disponibilità e prezzo.
          → nota interna com o riepilogo + transferência para a equipe na Helena
```

(As falas seguem os textos aprovados de `app/redacao.py`; a redação pelo LLM pode variar a
forma, nunca o conteúdo.)

## O ponto central: o código decide, a IA entende e escreve

A primeira versão deixava um LLM com ferramentas extrair os dados, decidir e escrever na mesma
chamada. Cada desvio virava uma trava nova no prompt:
- pedia de novo uma medida que o cliente já tinha dado;
- anunciava a transferência sem transferir;
- registrava "4 pneus" quando o cliente só tinha dito a medida.

O motor atual é um grafo [LangGraph](https://www.langchain.com/langgraph) de quatro etapas:

```
START → entender → decidir → redigir → verificar → END
                               ▲            │
                               └────────────┘  barrou: 1 reescrita com o motivo;
                                               barrou de novo: sai o texto aprovado da ação
```

| Etapa | Quem faz | O que faz |
| --- | --- | --- |
| **Entender** (`app/entendimento.py`) | LLM com saída estruturada | Lê o que o cliente disse: os 5 dados, a intenção (conversa, pedido de operador, dúvida técnica, fora do assunto, incompreensível), se perguntou o preço. |
| **Conferir** (`app/agent_tools.py`) | Código | Extrai a medida e a quantidade de forma determinística e **recusa** o que o LLM "extraiu" mas o cliente não escreveu. O que foi recusado fica registrado no trace. |
| **Decidir** (`app/politica.py`) | Código (função pura) | Máquina de estados: qual dado pedir agora, quando passar ao operador e por qual dos 6 motivos (dados completos, pedido de operador, foto, dúvida técnica, preço com dados suficientes, mensagem incompreensível). |
| **Redigir** (`app/redacao.py`) | LLM sem ferramentas | Escreve a mensagem da ação já decidida, no tom da persona, usando os textos aprovados quando existem. |
| **Verificar** (`app/response_grounding.py`) | Código | Barra preço, marca, disponibilidade, agendamento, idioma errado, pedido de dado já fornecido e mensagem que não cumpre a ação, como transferir sem avisar. |

O riepilogo que o operador lê é montado pelo **código** a partir dos dados conferidos
(`app/resumo_atendimento.py`). A transferência na Helena só acontece **depois** de o cliente
receber a resposta.

O porquê de cada decisão, os frameworks e o que cada um custa estão em
[`README-ARQUITETURA.md`](README-ARQUITETURA.md).

## Outros comportamentos

- **Áudio:** a mensagem de voz é transcrita pelo Gemini. Quem mandou áudio recebe a resposta em áudio (Gemini TTS + ffmpeg).
- **Foto:** o agente não lê imagens. A foto vira um marcador e vai direto ao operador.
- **Debounce (Redis):** junta as mensagens que o cliente manda em sequência num turno só.
- **Silêncio quando o humano assume:** se um operador pegou a conversa na Helena, o agente não responde mais (`app/atendimento_helena.py`).
- **Mensagens órfãs:** depois de um deploy ou queda, uma varredura responde as mensagens que ficaram sem resposta, sem responder duas vezes (`app/orphan_recovery.py`).
- **Provedor trocável e modelo reserva:** Gemini é o principal; OpenAI e DeepSeek são opcionais e trocáveis no painel. Se o modelo principal falhar no meio do turno, o reserva assume.
- **Painel web** (`/dashboard`, HTTP Basic): conversas com a decisão de cada turno, custos por conversa, "teste rápido", base de conhecimento (avisos temporários como feriados e promoções), prompt da persona, chaves e modelo.

## Observabilidade

Cada atendimento vira um trace no [Langfuse](https://langfuse.com) (self-hosted). O trace
cobre transcrição, motor, envio e transferência, e termina com um **desfecho**:

- `respondido`
- `respondido_com_fallback`
- `silencio_intencional`
- `falha_envio`
- `falha_transferencia`
- `falha_interna`

Assim, quando um cliente reclama que "ha dato errore in chat", dá para saber se o problema foi
no LLM, no envio ou na transferência. Telefones são mascarados antes de sair do processo. Sem as
chaves, nada sai. Detalhes em [`README-OBSERVABILIDADE.md`](README-OBSERVABILIDADE.md).

## Testes

```bash
pip install -r requirements-dev.txt
pytest -m "not llm"                    # 110 testes determinísticos: sem banco, sem LLM, sem custo
deepeval test run tests/ -m llm        # 10 conversas contra o Gemini real (precisa de GEMINI_API_KEY)
```

- Os **determinísticos** cobrem:
  - a normalização de medida e quantidade;
  - a recusa de dados inventados;
  - todas as transições da política;
  - a verificação das respostas;
  - o grafo de ponta a ponta com um modelo falso;
  - o mascaramento de telefones.
- As **conversas** (`tests/conversas.json`) são avaliadas pelo [DeepEval](https://deepeval.com) com duas métricas: regras do caso, que são determinísticas, e `ConversationalGEval`, com o Gemini como juiz.
- **Regra da casa:** cada incidente de produção vira um caso novo em `conversas.json`, e não uma regex nova no código.

Também há `scripts/simular_conversa.py`, para conversar com o Matteo no terminal sem
WhatsApp, Helena ou banco. Já o `scripts/bateria_testes.py` roda o pipeline completo contra
o banco com a Helena simulada.

## Como rodar

```bash
cp .env.example .env          # GEMINI_API_KEY, tokens da Helena, SETTINGS_ENCRYPTION_KEY
docker compose up -d --build
```

Sobe a API em `127.0.0.1:8124`, com Postgres e Redis. As migrações (Alembic) rodam antes da
API.

- Saúde: http://127.0.0.1:8124/health
- Painel: http://127.0.0.1:8124/dashboard
- Webhook da Helena: `POST /webhook/helena/{producao|teste}`

## Estrutura

```
app/
  main.py                 webhook (responde 200 na hora e processa em background), /health
  pipeline.py             orquestra o atendimento: áudio, foto, debounce, motor, envio, transferência
  llm.py, motor_v2.py     entrada do turno e o grafo LangGraph
  entendimento.py         etapa "entender"
  agent_tools.py          extração determinística, conferência dos dados, transferência
  politica.py             etapa "decidir" (máquina de estados)
  redacao.py              etapa "redigir" e os textos aprovados
  response_grounding.py   etapa "verificar"
  resumo_atendimento.py   riepilogo para o operador
  observabilidade.py      Langfuse: trace, desfecho, mascaramento
  helena_client.py        mensagens, áudio, notas, kanban e transferência na Helena
  stt.py, tts.py          transcrição e voz com Gemini
  dashboard/              painel web (FastAPI + Jinja2)
  prompts/                persona.md (em italiano) e entendimento.md
migrations/               Alembic (PostgreSQL)
tests/                    determinísticos + conversas DeepEval
scripts/                  simulador de conversa e bateria de integração
```

## Tecnologias

Python 3.12 · FastAPI · LangGraph · LangChain (Gemini, OpenAI, DeepSeek) · Gemini STT/TTS ·
Langfuse · DeepEval · PostgreSQL + SQLAlchemy + Alembic · Redis · API da Helena · Docker Compose

## Autor

João Victor Lopez de Arruda · [GitHub](https://github.com/JoaoVictor-tech) · [LinkedIn](https://linkedin.com/in/joao-victor-arruda)
