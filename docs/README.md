# Documentação — índice por atualidade

Dos docs **vivos** (descrevem o sistema como ele é hoje) aos **históricos** (retratos de época,
guardados em [historico/](historico/)). Última reorganização: setembro/2026 (runs com nome,
logs separados por run, [ESTADO_ATUAL.md](ESTADO_ATUAL.md) como fonte única do estado).

## Leia primeiro (quem está chegando)

1. [ESTADO_ATUAL.md](ESTADO_ATUAL.md) — **onde estamos**: o que as runs mediram, afirmações
   antigas corrigidas, lições acumuladas, plano com regras de decisão e hipóteses abertas.
2. [ESTRATEGIA_DEMOS_E_PERCEPCAO.md](ESTRATEGIA_DEMOS_E_PERCEPCAO.md) — **a nova estratégia**:
   por que o RL não passou do BC e o que muda (medições, percepção, off-policy com demos).
3. [../README.md](../README.md) — setup, comandos e configuração do `.env`.
4. [PACOTE_BC_ENTROPIA.md](PACOTE_BC_ENTROPIA.md) — os mecanismos em uso (BC, termostato de
   entropia, currículo, telemetria), com runbook e glossário.
5. [GUIA_CONCEITOS_E_FUNCIONAMENTO.md](GUIA_CONCEITOS_E_FUNCIONAMENTO.md) — a teoria do zero
   (RL, PPO, recompensa, entropia) aplicada a este projeto.
6. [MONITORAMENTO_TREINO.md](MONITORAMENTO_TREINO.md) + [GUIA_TENSORBOARD.md](GUIA_TENSORBOARD.md)
   — como acompanhar um treino rodando.

## Todos os docs

| Doc | Status | O que é |
|---|---|---|
| [ESTADO_ATUAL.md](ESTADO_ATUAL.md) | 🟢 vivo (set/2026) | **Fonte única do estado do projeto**: runs medidas, correções, lições, plano (Fases 1–3), pendências |
| [ESTRATEGIA_DEMOS_E_PERCEPCAO.md](ESTRATEGIA_DEMOS_E_PERCEPCAO.md) | 🟢 vivo (set/2026) | **A nova estratégia**: medir antes (Fase 1), percepção robusta contra estática/piscar/movimentos súbitos (Fase 2), off-policy com demos + correções ao vivo (Fase 3) — porquês, runbook, riscos, glossário |
| [HANDOFF_FASE1.md](HANDOFF_FASE1.md) | 🟡 transitório (set/2026) | **Prompt para o outro PC**: rodar as avaliações M1/M2 em blocos, pré-voo (hash do BC, .env, testes), regras e relatório |
| [PACOTE_BC_ENTROPIA.md](PACOTE_BC_ENTROPIA.md) | 🟢 vivo (jul/2026) | Os mecanismos do pacote BC + entropia: estratégias, runbook gravação→BC→treino, glossário |
| [REFERENCIA_HIPERPARAMETROS.md](REFERENCIA_HIPERPARAMETROS.md) | 🟢 vivo (jul/2026) | Consulta rápida: cada hiperparâmetro, valor atual, quando e como mexer |
| [GUIA_CONCEITOS_E_FUNCIONAMENTO.md](GUIA_CONCEITOS_E_FUNCIONAMENTO.md) | 🟢 vivo (jul/2026) | Guia didático completo (~950 linhas): do "o que é RL" ao funcionamento de cada peça |
| [VALIDACAO_E_TESTES.md](VALIDACAO_E_TESTES.md) | 🟢 vivo (jul/2026) | Checklist de execução: testes offline → percepção ao vivo → treino → A/B da LSTM (futuro) |
| [MONITORAMENTO_TREINO.md](MONITORAMENTO_TREINO.md) | 🟢 vivo (jul/2026) | Métricas por noite, causas de desfecho (skill vs sorte), currículo e rollback de checkpoints |
| [GUIA_TENSORBOARD.md](GUIA_TENSORBOARD.md) | 🟢 vivo (jun/2026) | Como ler os gráficos do tensorboard (smoothing, eixos, cada métrica) |
| [REORGANIZACAO_2026_07.md](REORGANIZACAO_2026_07.md) | 🟢 vivo (jul/2026) | O que mudou na estrutura do repositório e nos logs em julho/2026 |
| [historico/HANDOFF_PROD.md](historico/HANDOFF_PROD.md) | 📦 histórico (ago/2026) | Prompt de passagem da run 5 (que não rodou). A leitura da run 4 nele está errada; ver ESTADO_ATUAL §3 |
| [historico/AUDITORIA_RECOMPENSA_E_RL.md](historico/AUDITORIA_RECOMPENSA_E_RL.md) | 📦 histórico (jun/2026) | As Decisões 1–7 da auditoria de recompensa/RL — a origem do redesenho atual |
| [historico/ALTERACOES_COMPLETAS.md](historico/ALTERACOES_COMPLETAS.md) | 📦 histórico (jun/2026) | Registro técnico das correções de bugs do pipeline (dupla normalização, captura, energia...) |
| [historico/MELHORIAS_PROJETO_ATUAL.md](historico/MELHORIAS_PROJETO_ATUAL.md) | 📦 histórico (mai/2026) | Ideias de melhoria da época — curriculum e BC já viraram realidade |
| [historico/ALEM_DO_RL.md](historico/ALEM_DO_RL.md) | 📦 histórico (mai/2026) | Panorama de alternativas ao PPO (DAgger, GAIL, DreamerV3, Decision Transformer...) |

**Regra dos históricos:** são retratos de época — valores citados neles (recompensas, n_steps,
gates) podem não valer mais. Cada um tem um banner no topo apontando a fonte da verdade atual.

## Ferramentas do projeto (mapa rápido)

Os comandos completos estão no [README](../README.md) e no [VALIDACAO_E_TESTES.md](VALIDACAO_E_TESTES.md).

**Calibração (jogo aberto)** — `python -m src.utils.<nome>`
- `calibrar_por_passos` — calibração guiada de todas as coordenadas de clique
- `calibrar` — captura de referências individuais (morte, vitória, câmera, menu)
- `inspecionar` — grade com coordenadas p/ escolher regiões de detecção (salva em `debug/`)

**Percepção ao vivo (jogo aberto)**
- `monitor` — detecção de ameaça/energia em tempo real
- `testar_deteccao_menu` — score do template de menu/morte/vitória ao vivo
- `sonda_captura` — rajada de frames: latência do mss, frames distintos/s, ruído cru × mediana,
  piscar por região (medição M4 do ESTADO_ATUAL)

**Testes offline (jogo fechado, segundos)**
- `testar_recompensa` — sanidade da função de recompensa e do shaping Φ (17 invariantes)
- `testar_noite` — lógica de reset/currículo (`decidir_reset`, 11 casos)
- `testar_deteccao_ameaca` / `testar_deteccao_energia` — detectores contra fixtures de `debug/`
- `testar_masking` / `sonda_memoria` — LSTM (reset de memória / uso de recorrência) — p/ o A/B futuro
- `simular_energia` — modelo de dreno de energia vs o jogo real
- `ablacao_offline` — a CNN contribui? (exige modelo treinado)
- `scripts/smoke_test.py` — env + extractor + predict sem o jogo
- `scripts/inspecionar_vecnormalize.py` — escala/clip da recompensa normalizada num `.pkl`

**Análise de logs** — `python scripts/<nome>.py`
- `metricas_treino` — win rate/sobrevivência/causas por noite (lê `logs/analise/treino_detalhado.log`;
  runs passadas: `--log logs/analise/historico/<run>_detalhado.log`)
- `taxa_acao` — ações/min das demos humanas × do agente avaliado (`main.py jogar`) — medição M3
- `enviar_logs_mongodb` / `exportar_logs_xlsx` / `limpar_banco` — pipeline MongoDB/Excel (2 PCs)
- `bump_version` — versionamento (`VERSION` + `src/version.py`)
