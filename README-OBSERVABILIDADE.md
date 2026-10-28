# Observabilidade — o que o agente manda ao Langfuse

Guia de leitura dos traces do Matteo. Implementação em `app/observabilidade.py`; quem chama
é `app/pipeline.py` (atendimento) e `app/llm.py` (turno). Configuração: `LANGFUSE_PUBLIC_KEY`,
`LANGFUSE_SECRET_KEY` e `LANGFUSE_HOST` no `.env` — **sem as duas chaves nada sai do agente**.

---

## 1. Dois níveis

```
atendimento                     ← o caminho todo, do webhook ao envio. Termina com DESFECHO.
├── entrada:audio               ← transcrição (modelo, custo, erro se falhou)
├── turno                       ← o grafo do motor v2
│   ├── entender                ← chamada ao LLM (tokens, custo, o JSON extraído)
│   ├── decidir                 ← código: a ação do turno
│   ├── redigir                 ← chamada ao LLM (tokens, custo, o texto)
│   └── verificar               ← código: barrou ou não
├── envio                       ← a Helena aceitou a mensagem?
├── voz:gemini-tts              ← custo da resposta em áudio (não passa pelo LLM)
└── transferencia               ← riepilogo + PUT /transfer
```

O "atendimento" existe porque o "ha dato errore in chat" quase nunca acontece no LLM: ele
acontece no envio, na transferência ou numa trava. Com o trace só no turno, uma falha de envio
aparecia como sucesso.

## 2. Atributos do trace (servem de filtro)

| Atributo | De onde vem |
|---|---|
| `session_id` | a sessão da Helena — agrupa a conversa inteira |
| `user_id` | o id do contato na Helena |
| `environment` | `teste` ou `producao` (o filtro do painel do Langfuse usa este campo) |
| `tags` | `cliente`, `entrada-audio` / `entrada-image` / `entrada-text` |

## 3. Desfechos

Cada atendimento termina com um, gravado como **score categórico** `desfecho` (vira filtro e
gráfico: "% de falha_envio por dia"):

| Desfecho | Significa | Nível |
|---|---|---|
| `respondido` | o cliente recebeu a resposta | DEFAULT |
| `respondido_com_fallback` | o motor falhou duas vezes e saiu "ho avuto un problema tecnico" | WARNING |
| `silencio_intencional` | o agente calou de propósito (operador assumiu, conversa já transferida) | DEFAULT |
| `falha_envio` | havia resposta e ela **não chegou** ao cliente | ERROR |
| `falha_transferencia` | o lead **não** chegou na fila humana | ERROR |
| `falha_interna` | exceção não tratada no pipeline | ERROR |

Regra: **falha nunca é apagada por sucesso.** Se o envio falhou, um `respondido` posterior não
sobrescreve.

Todo desfecho de falha também vira o evento `desfecho_com_falha` no painel do agente, com o
link do trace — quem abre a conversa vê o que deu errado sem procurar no Langfuse.

## 4. Avisos (o atendimento deu certo, mas degradado)

Ficam em `metadata.avisos` e sobem o nível do trace para WARNING:

- `verificacao barrou a 1a redacao: <motivo>` — a reescrita salvou, mas vale olhar o motivo;
- `modelo reserva usado: <provider:model>` — o provedor principal caiu no meio do turno;
- `a resposta em audio falhou; o cliente recebeu o texto`;
- `transferencia sem passar pelo motor: <motivo>` — áudio não transcrito, conteúdo não suportado.

## 5. O que o turno registra

Em `metadata` do nó `turno`: os dados já coletados, se transferiu e por quê, o motivo da
verificação (se barrou), o modelo usado, se o reserva entrou, o custo em dólares e o objeto
`decisao` — que é o mesmo `ctx.trace` gravado no painel como evento `decisao_do_turno`:

```json
{
  "motor": "v2",
  "entendimento": {"dados": {"Misura": "205/55 R16"}, "intencao": "conversa", ...},
  "descartado_na_validacao": {"Quantità": "o cliente nao escreveu 4 nesta mensagem"},
  "acao": {"tipo": "PERGUNTAR", "slot": "Quantità", "motivo": null},
  "verificacao": null,
  "verificacao_final": null
}
```

É por aqui que se acha a causa de uma resposta ruim: o entendimento leu errado, a decisão foi
errada, ou a redação não cumpriu a decisão.

## 6. Privacidade

O trace leva a fala do cliente e o briefing da redação. Antes de qualquer coisa sair do
processo, a função `mascarar` troca telefones (italianos e brasileiros) por `[telefono]` — e
não toca em medida de pneu (`205/55 R16`, `2255517`), que de outro modo pareceria telefone.

O Langfuse **precisa ser self-hosted**, sob o mesmo controle do banco do agente.

## 7. Nos testes

O `conftest` desliga o Langfuse à força: o plugin do DeepEval carrega o `.env` antes dele, e
sem isso toda rodada de teste mandaria trace para o projeto de produção. Para rastrear uma
rodada de propósito:

```bash
TESTES_COM_LANGFUSE=1 deepeval test run tests/ -m llm
```
