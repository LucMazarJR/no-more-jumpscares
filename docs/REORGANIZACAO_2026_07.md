# Reorganização do repositório — julho/2026

> O que mudou na estrutura de arquivos, nos logs e na documentação, e por quê. Os **comandos de
> execução e os conceitos da estrutura NÃO mudaram** — o foco foi simplificar e separar o vivo do
> acumulado.

## 1. O problema

Três tipos de acúmulo estavam atrapalhando a leitura do projeto:

1. **docs/ sem hierarquia**: 10 arquivos lado a lado, misturando docs vivos com retratos de época —
   e 3 deles ("defasados-enganosos") davam instruções vivas com números velhos (11 estados,
   n_steps=8192/2048, gate de entropia que já não existe).
2. **Log de execução poluído**: a linha do `treino.log` tinha ganhado campos de telemetria
   (`Energia fim`, `Causa`) — útil para análise, ruim para ler durante o treino.
3. **Arquivos mortos**: scripts órfãos em `src/utils/`, funções sem nenhuma chamada, 61 PNGs em
   `debug/`, ferramenta descontinuada na raiz.

## 2. docs/ — separação por atualidade

```
docs/
├── README.md                ← NOVO: índice (ordem de leitura + status de cada doc)
├── <docs vivos>             ← descrevem o sistema COMO ELE É HOJE
└── historico/               ← NOVO: retratos de época, cada um com banner 📦 no topo
    ├── AUDITORIA_RECOMPENSA_E_RL.md      (jun/2026 — as Decisões 1-7)
    ├── ALTERACOES_COMPLETAS.md           (jun/2026 — correções de bugs do pipeline)
    ├── MELHORIAS_PROJETO_ATUAL.md        (mai/2026 — ideias; curriculum/BC viraram realidade)
    └── ALEM_DO_RL.md                     (mai/2026 — alternativas ao PPO)
```

**Os 3 docs defasados-enganosos foram CORRIGIDOS para o estado atual** (decisão do usuário —
corrigir, não arquivar):
- `GUIA_CONCEITOS_E_FUNCIONAMENTO.md` — 11→**12 estados**, n_steps 2048/8192→**4096** (com a
  história do porquê), seção 6.4 reescrita do gate extinto para o **ControladorEntropia**
  (termostato), nota da fase atual da LSTM, currículo automático.
- `REFERENCIA_HIPERPARAMETROS.md` — cabeçalho de valores atuais refeito (n_steps=4096,
  clip_reward=100, knobs do termostato `FNAF_H_*`/`FNAF_ENT_*`), seção do ent_coef reescrita
  ("mexa no ALVO, não no coeficiente"), curriculum marcado como implementado, mapa
  sintoma→botão atualizado.
- `VALIDACAO_E_TESTES.md` — apêndice de parâmetros corrigido (eram 2048/64/10!), nota de fase no
  topo (LSTM só após o gatilho), testes novos no checklist (testar_noite,
  inspecionar_vecnormalize), comandos com `--bc`.

E o `PACOTE_BC_ENTROPIA.md` ganhou um **Glossário** (§7) explicando os termos técnicos que usava
sem apresentar: Markov/quase-Markov, POMDP, nats, perplexidade, malha fechada/termostato/ganho,
dithering, shaping potential-based/telescopar, warmstart/BC, sampler balanceado/estratificado,
recall/macro-recall, wall-clock, rollout/minibatch/épocas, clip de reward.

## 3. logs/ — execução limpa, análise separada

O pedido: "o log de execução deve ficar o mais limpo possível; dados de análise em outro lugar".

```
logs/
├── treino.log           ← VOLTOU ao formato enxuto: PC | Ep | Noite | RESULTADO | Passos |
│                          Tempo | Recompensa | Taxa vitória  (é o que se lê durante o treino)
├── desyncs.log          (como era)
├── analise/             ← NOVO: dados p/ análise, fora da leitura de execução
│   ├── treino_detalhado.log   mesma linha + "| Energia fim: X% | Causa: <rotulo>"
│   │                          + linhas OCORRIDO (diagnóstico de interrupções)
│   └── treino_steps.log       log por step (flag --steps; antes ficava na raiz)
└── tensorboard/         ← NOVO: eventos TB saem da raiz de logs/
                           (`tensorboard --logdir logs` continua funcionando — o TB varre subpastas)
```

- `scripts/metricas_treino.py` e `scripts/enviar_logs_mongodb.py` **preferem o detalhado** e caem
  no `treino.log` enxuto se ele não existir (logs antigos continuam parseáveis).
- O tensorboard continua com a telemetria completa (`custom/causas/*`, `custom/entropia`...).

## 4. Código morto removido (auditoria de uso)

Heurística: nome definido com **zero usos** fora da própria definição, em código + docs + configs;
cada candidato conferido manualmente (hooks de framework como `forward`/`_on_training_end` são
falsos positivos e ficaram).

| Removido | Por quê |
|---|---|
| `src/utils/testar_energia.py` | rascunho avulso (media uma "barra verde" com região hardcoded — nem era do FNAF) |
| `src/utils/testar_deteccao.py` | superado por `testar_deteccao_menu.py` (mais novo, cobre morte+vitória+menu) |
| `FNAFEnv._energia_esperada()` | sobrou do reward antigo (penalidade por déficit de energia, removida no redesenho) |
| `GameCapture.pressionar_tecla()` | nenhuma chamada (atalhos usam `atalho()`) |
| `detectar_fonte_log()` (mongodb) | nenhuma chamada (a versão plural `detectar_fontes_log` é a usada) |

**Mantidos de propósito** (órfãos por referência, mas vivos por função): `simular_energia`
(calibração do modelo de dreno), `exportar_logs_xlsx`/`limpar_banco` (pipeline MongoDB/Excel dos
2 PCs), `sonda_memoria`/`testar_masking` (pré-voo do A/B da LSTM).

## 5. Demais mudanças de estrutura

- `merge_modelos.py` (raiz) → **`scripts/merge_modelos.py`** — ferramenta descontinuada (média de
  pesos entre linhagens quebra a política); referências no README/main.py atualizadas.
- `debug/` — os **fixtures** `quadro_*.png` (lidos por `ablacao_offline` e `testar_deteccao_*`)
  ficam na raiz de `debug/`; os outros 29 PNGs de depurações antigas foram para **`debug/arquivo/`**
  (local, git-ignorado). Política: `debug/` é transiente — pode esvaziar, MENOS os `quadro_*.png`.
- **`CLAUDE.md`** (novo, raiz) — mapa do projeto otimizado p/ IA e humanos: comandos canônicos,
  pastas com caminhos hardcoded (⚠ não mover), convenções (logs, observação, GAMMA, medição).
- **`docs/README.md`** (novo) — índice por atualidade + mapa das ferramentas por categoria.
- Comentários defasados no código varridos (ex.: "10 estados" na obs zerada do
  `_interromper_episodio`).
- `README.md` — visão geral corrigida (dizia **7 estados**; são 12), fase atual + link pro índice.

## 6. O que NÃO mudou (de propósito)

- **Todos os comandos**: `python main.py teste|treino|jogar|bc`, `python -m src.utils.<X>`,
  `python scripts/<Y>.py`, `tensorboard --logdir logs`.
- **Estrutura de `src/`** e os caminhos hardcoded: `src/utils/referencias/`, `modelos/`, `logs/`,
  `dados/`, `debug/` (como destino de escrita).
- Os conceitos da arquitetura (env ↔ agente ↔ ferramentas ↔ análise) — só a organização ao redor.

## 7. Verificação executada

`py_compile` em todos os tocados · `smoke_test.py` 6/6 · `testar_recompensa` 13/13 ·
`testar_noite` 11/11 · teste unitário do novo LogCallback (linha enxuta no `treino.log`,
completa no detalhado, parseável pelo `metricas_treino`) · varredura de links `.md` sem órfãos ·
moves como renames no git.
