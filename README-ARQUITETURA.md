# Arquitetura e decisões — Agente Gomme Fenice (Matteo)

Registro da reestruturação do agente (outubro/2026): o que mudou, por quê, quais frameworks
entraram e o que cada um custa. A referência de arquitetura foi outro agente de pneus da empresa,
que passou pela mesma reestruturação antes; aqui ela foi aplicada ao domínio do Matteo (5 dados,
italiano, sem leitura de imagem).

Documentos relacionados: `README.md` (visão geral e como rodar) e
`README-OBSERVABILIDADE.md` (como ler os traces).

**Fora do escopo desta rodada: multi-tenant.** O agente continua sendo de uma loja só
(`AppSettings` com `id=1`, um container e um banco). A ideia de um agente de segmento
atendendo N lojas foi deliberadamente deixada de fora — ver seção 6.

---

## 1. Resumo

| Antes | Depois |
|---|---|
| Um LLM num loop de ferramentas decidia **e** escrevia; as regras viviam no prompt e eram remendadas com travas e retries (`app/anti_stall.py`) | Motor v2 em etapas: o LLM **entende** e **redige**; o **código decide** (máquina de estados em `app/politica.py`) |
| SDK de cada provedor chamado à mão (`app/llm_providers/gemini_provider.py`, `openai_compatible_provider.py`) | LangGraph orquestra o turno; os chat models do LangChain falam com cada provedor, configurados num lugar só |
| Teste determinístico em script próprio; nenhuma avaliação de conversa | DeepEval: 110 testes determinísticos + 10 conversas avaliadas por regras e por juiz LLM |
| "Ha dato errore in chat" sem como descobrir onde | Langfuse: um trace por atendimento, com desfecho explícito e a etapa que falhou |

O motor antigo (v1) foi **removido** nesta rodada, junto com `app/anti_stall.py`, os dois
provedores próprios e o prompt `system_static.md`.

---

## 2. Decisões tomadas

### 2.1 Separar entender, decidir e redigir

- **Problema**: o motor v1 punha o LLM para extrair os dados, decidir a passagem ao operador e
  escrever a resposta na mesma chamada. Cada desvio virava uma trava nova: o agente pedia de
  novo a medida que estava na mesma frase (3 de 3 vezes com o gemini-2.5-flash, 30/09/2026),
  anunciava a passagem sem chamar a ferramenta, transferia junto com uma pergunta, juntava 2-3
  dados num só "Mi manca solo ..." contra a regra do prompt. A mesma regra vivia no prompt, no
  bloco dinâmico e numa regex — e elas se contradiziam.
- **Decisão**: padrão de agente de atendimento orientado a tarefa — *entendimento estruturado →
  validação em código → decisão em código → redação → verificação*. Toda regra de fluxo mora
  num lugar só (`app/politica.py`, função pura e testável).
- **Por que é o padrão aqui**: são 5 dados para coletar e 6 motivos de passagem — fluxo fechado,
  com regras de negócio que não podem depender de o modelo lembrar. Agente "livre" com
  ferramentas faz sentido em atendimento aberto; não neste.

### 2.2 O código grava os dados, nunca o modelo

Duas camadas, nesta ordem (`app/agent_tools.py`):

1. **Extração determinística** antes do LLM: medida (`205 55 16`, `2255517`, `225/55ZR17` →
   `205/55 R16`), quantidade com unidade (`4 gomme`, `un paio`) e os campos categóricos que a
   mensagem cita sem ambiguidade. Barata, testada e não alucina.
2. **Conferência do que o LLM extraiu**: cada valor só vira dado se o cliente escreveu algo que
   o sustente *naquela mensagem*. O que não passa vai para `descartado_na_validacao` no trace.

Isso fecha o buraco do motor antigo, em que o modelo chamava `criarResumo` com o que achava e o
riepilogo chegava ao operador com medida ou quantidade inventada (o caso mais comum: o cliente
diz só a medida e o modelo registra 4 pneus, porque é o pedido mais frequente).

### 2.3 LangGraph + LangChain

- **Decisão**: o turno é um grafo LangGraph (`entender → decidir → redigir → verificar`, com
  volta para `redigir` uma vez quando a verificação barra). As chamadas ao LLM passam pelos chat
  models do LangChain, configurados num lugar só (`app/llm_providers/chat_models.py`).
- **Expectativa corrigida no caminho**: o LangGraph sozinho **não reduz** código (a orquestração
  tem ~200 linhas, parecida com o loop que havia antes). O que reduz é a camada de provedores
  via LangChain, que eliminou os dois arquivos de provedor próprio. O ganho do LangGraph é
  organização, trace por etapa e padronização entre os agentes da empresa.
- **Detalhe**: a extração usa `bind_tools` com a função forçada, e não `with_structured_output`
  — mesmo resultado, sem as etapas internas do LangChain poluindo o trace.
- **Ajustes de produção preservados como parâmetro**: raciocínio do Gemini no mínimo (sem isso a
  resposta chega cortada no meio, porque os tokens de raciocínio contam no mesmo teto de saída),
  `reasoning_effort` desligado no gpt-5.6, `ChatDeepSeek` em vez de `ChatOpenAI` com `base_url`.

### 2.4 DeepEval como executor de todos os testes

- **Decisão**: `deepeval test run tests/` roda tudo. Os testes **determinísticos** (normalização
  de medida, conferência de dado, política, verificação, grafo com modelo falso, observabilidade)
  são pytest comum — código sem LLM sempre dá o mesmo resultado, e um juiz LLM ali só traria
  custo e oscilação. As **conversas** (`tests/test_conversas.py` + `tests/conversas.json`) viram
  `ConversationalTestCase` com duas métricas: regras do caso (determinística) e
  `ConversationalGEval` (juiz Gemini).
- **Regra de casa**: cada incidente de produção vira um caso novo em `conversas.json`
  (anonimizado), em vez de uma regex nova no código.
- **Custo**: DeepEval e pytest ficam em `requirements-dev.txt`, fora da imagem de produção.
- **Achado**: o plugin do DeepEval carrega o `.env` **antes** do `conftest` — sem cuidado, os
  testes mandariam traces ao Langfuse de produção sem ninguém pedir. O `conftest` desliga o
  Langfuse à força (`TESTES_COM_LANGFUSE=1` para rastrear de propósito).

### 2.5 Langfuse: um trace por atendimento, com desfecho

- **Problema real**: o cliente diz "ha dato errore in chat" e ninguém sabe onde. Com o trace só
  no LLM, uma falha de envio pela Helena aparecia como turno bem-sucedido — o cliente ficava sem
  resposta e o painel dizia que estava tudo certo.
- **Decisão**: o trace cobre o atendimento inteiro (transcrição de áudio, travas, motor, envio,
  transferência) e termina com um **desfecho** — `respondido`, `respondido_com_fallback`,
  `silencio_intencional`, `falha_envio`, `falha_transferencia`, `falha_interna` — gravado como
  score categórico no Langfuse e, quando é falha, como evento `desfecho_com_falha` no painel,
  com o link do trace.
- **Salvaguardas**: desligado sem as chaves (nada sai do processo); nunca derruba um turno;
  telefone mascarado antes do envio; o servidor precisa ser self-hosted (LGPD/GDPR).

### 2.6 A verificação não cai no texto genérico

Quando a verificação barra a redação, o motor redige de novo **uma vez** com o motivo. Se barrar
de novo, o cliente recebe o **testo approvato daquela ação** (`app/redacao.py:texto_pronto`) —
nunca o "ho avuto un problema tecnico". Refazer é seguro aqui porque a redação não decide nada:
a ação já está fixada, então o retry corrige o TEXTO, e não troca uma decisão por outra (que era
o risco dos retries do motor antigo).

### 2.7 Decisões de produto registradas

- **Quem pergunta preço com medida, quantidade, estação e condição já coletadas vai direto para
  o operador**, mesmo faltando o uso. O uso só orienta a recomendação. No motor antigo o agente
  seguia coletando e o cliente repetia o pedido de preço (caso E9 da bateria de 02/10/2026).
- **O mesmo dado pedido três vezes seguidas transfere.** Caso real do replay de 02/10/2026: o
  agente pediu o "uso" três vezes a quem só queria preço. Contado em código, não lembrado pelo
  modelo.
- **O Matteo continua sem ler imagens**: foto é sempre passagem ao operador ("FOTO RICEVUTA"),
  decidida pelo código a partir do marcador do pipeline, não pelo modelo.
- **A persona passa a ser o prompt do painel.** O prompt padrão de `/dashboard/settings` agora é
  `app/prompts/persona.md`. Se o painel ainda tiver salvo o prompt do motor antigo (ele cita
  `criarResumo`/`transferirAtendimento`), a redação usa a persona do repositório e registra erro
  no log — proteção para o deploy.

---

## 3. Arquitetura atual

```
WhatsApp ─► Helena ─► POST /webhook/helena/{teste|producao}
                         │  áudio → transcrição Gemini (app/stt.py) · foto → marcador [FOTO]
                         ▼
                 debounce (Redis) junta mensagens rápidas
                         ▼
   ┌──────────── atendimento (trace Langfuse, desfecho) ─────────── app/pipeline.py ───┐
   │ travas: operador assumiu (painel/Helena) · conversa já transferida                 │
   │ turno ── app/llm.py:run_turn ── app/motor_v2.py (grafo LangGraph)                   │
   │                                                                                     │
   │                         entender ─► decidir ─► redigir ─► verificar ─┐             │
   │                         (LLM,JSON)  (código)   (LLM)     (código)    │ barrou: 1×  │
   │                                        ▲                    └────────┘ reescreve   │
   │ envio ── Helena (texto ou áudio Gemini TTS)                                         │
   │ transferência ── riepilogo (nota interna) · PUT /transfer conferido                 │
   └─────────────────────────────────────────────────────────────────────────────────────┘
   varredura de órfãs (app/orphan_recovery.py) · painel /dashboard
```

**Divisão de responsabilidade** (o princípio que tudo segue): o **código** decide o que é
verdade e o que fazer — quais dos 5 dados o cliente deu, qual é o próximo a pedir, se já dá para
passar ao operador, se a resposta cumpriu a tarefa. O **LLM** entende o que o cliente disse e
escreve a mensagem.

| Arquivo | Papel |
|---|---|
| `app/entendimento.py` | etapa 1: LLM com saída estruturada — o que o cliente disse |
| `app/agent_tools.py` | extração determinística + conferência dos dados + execução da transferência |
| `app/politica.py` | etapa 2: máquina de estados pura — estado + entendimento → ação |
| `app/redacao.py` | etapa 3: briefing do turno e os testi approvati; LLM só escreve |
| `app/response_grounding.py` | etapa 4: proibições absolutas + a ação foi cumprida? |
| `app/motor_v2.py` | o grafo que costura as quatro etapas, com reserva de modelo |
| `app/llm.py` | porta de entrada: abre o trace e chama o motor |
| `app/observabilidade.py` | Langfuse: trace de atendimento e de turno, desfecho |

---

## 4. Frameworks e impacto

| Framework | Onde | Impacto | Custo / risco |
|---|---|---|---|
| **LangGraph** 1.2 | `app/motor_v2.py` | turno como grafo explícito; cada etapa é um nó visível no trace; a reescrita após a verificação é uma aresta, não um `if` escondido | dependência nova; o ganho é organização, não tamanho |
| **LangChain** (core 1.6, chat models) | `app/llm_providers/chat_models.py`, `app/entendimento.py` | um ponto de configuração por provedor (Gemini, OpenAI, DeepSeek); removeu os dois provedores próprios | abstrai detalhes que antes eram explícitos — cada ajuste fino foi reconferido |
| **Langfuse** SDK 4.17 | `app/observabilidade.py`, `app/pipeline.py` | diagnóstico do "deu erro": desfecho por atendimento, etapa que falhou, custo e latência por chamada | servidor self-hosted precisa estar saudável; desligado por padrão |
| **DeepEval** 4.2 | `tests/` | 110 testes determinísticos + 10 conversas avaliadas a cada mudança; o juiz pega o que regra não pega (mensagem cortada, tom) | o juiz também varia: rodar mais de uma vez antes de concluir |

---

## 5. Estado da entrega

- **110 testes determinísticos** passam sem chave nenhuma (`pytest -m "not llm"`).
- As **10 conversas contra o Gemini real** (`deepeval test run tests/ -m llm`) passaram 10/10
  na versão original (~US$ 0,06 a rodada).
- A **bateria de integração** (`scripts/bateria_testes.py`: pipeline completo contra o banco,
  com a Helena simulada) passou 28/28 depois da correção da transferência na 2ª falha seguida
  do modelo.
- **Persona no painel**: se o banco não tem prompt salvo, o código usa `app/prompts/persona.md`
  (ver `redacao.persona_ativa`) — não é preciso republicar o prompt a cada mudança.

---

## 6. Multi-tenant: fora do escopo (e por quê)

A reestruturação de referência propunha um passo seguinte: transformar o agente de uma
loja num agente de **segmento** que atende N lojas com configuração por tenant. **Isso não foi
feito aqui, a pedido** — a Gomme Fenice continua sendo agente de uma loja só.

O que vale registrar, para quando o assunto voltar: a maior parte do que está em
`app/politica.py`, `app/agent_tools.py` e `app/response_grounding.py` é do *segmento*
(qualificação de pneu), e não da Gomme Fenice. O que é da loja é configuração: persona e tom, os 5 dados
obrigatórios, credenciais e equipe da Helena, avisos. Nada do que foi feito nesta rodada
atrapalha essa separação depois — mas também nada dela foi antecipado.
