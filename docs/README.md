# Documentação — índice por atualidade

Dos docs **vivos** (descrevem o sistema como ele é hoje) aos **históricos** (retratos de época,
guardados em [historico/](historico/)). Última reorganização: julho/2026
([REORGANIZACAO_2026_07.md](REORGANIZACAO_2026_07.md)).

## Leia primeiro (quem está chegando)

1. [../README.md](../README.md) — setup, comandos e configuração do `.env`.
2. [PACOTE_BC_ENTROPIA.md](PACOTE_BC_ENTROPIA.md) — **a fase atual**: BC warmstart + termostato de
   entropia + currículo automático, com runbook e glossário dos termos técnicos.
3. [GUIA_CONCEITOS_E_FUNCIONAMENTO.md](GUIA_CONCEITOS_E_FUNCIONAMENTO.md) — a teoria do zero
   (RL, PPO, recompensa, entropia) aplicada a este projeto.
4. [MONITORAMENTO_TREINO.md](MONITORAMENTO_TREINO.md) + [GUIA_TENSORBOARD.md](GUIA_TENSORBOARD.md)
   — como acompanhar um treino rodando.

## Todos os docs

| Doc | Status | O que é |
|---|---|---|
| [PACOTE_BC_ENTROPIA.md](PACOTE_BC_ENTROPIA.md) | 🟢 vivo (jul/2026) | A fase atual: estratégias, runbook gravação→BC→treino, métricas de sucesso/aborto, glossário |
| [REFERENCIA_HIPERPARAMETROS.md](REFERENCIA_HIPERPARAMETROS.md) | 🟢 vivo (jul/2026) | Consulta rápida: cada hiperparâmetro, valor atual, quando e como mexer |
| [GUIA_CONCEITOS_E_FUNCIONAMENTO.md](GUIA_CONCEITOS_E_FUNCIONAMENTO.md) | 🟢 vivo (jul/2026) | Guia didático completo (~950 linhas): do "o que é RL" ao funcionamento de cada peça |
| [VALIDACAO_E_TESTES.md](VALIDACAO_E_TESTES.md) | 🟢 vivo (jul/2026) | Checklist de execução: testes offline → percepção ao vivo → treino → A/B da LSTM (futuro) |
| [MONITORAMENTO_TREINO.md](MONITORAMENTO_TREINO.md) | 🟢 vivo (jul/2026) | Métricas por noite, causas de desfecho (skill vs sorte), currículo e rollback de checkpoints |
| [GUIA_TENSORBOARD.md](GUIA_TENSORBOARD.md) | 🟢 vivo (jun/2026) | Como ler os gráficos do tensorboard (smoothing, eixos, cada métrica) |
| [REORGANIZACAO_2026_07.md](REORGANIZACAO_2026_07.md) | 🟢 vivo (jul/2026) | O que mudou na estrutura do repositório e nos logs em julho/2026 |
| [AVALIACAO_TREINO_PC_CEGO.md](AVALIACAO_TREINO_PC_CEGO.md) | 🟢 vivo (jul/2026) | Análise do build "cego" do PC 2 (sem percepção): teto observacional, baseline p/ comparação cego vs observador |
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

**Testes offline (jogo fechado, segundos)**
- `testar_recompensa` — sanidade da função de recompensa e do shaping Φ (13 invariantes)
- `testar_noite` — lógica de reset/currículo (`decidir_reset`, 11 casos)
- `testar_deteccao_ameaca` / `testar_deteccao_energia` — detectores contra fixtures de `debug/`
- `testar_masking` / `sonda_memoria` — LSTM (reset de memória / uso de recorrência) — p/ o A/B futuro
- `simular_energia` — modelo de dreno de energia vs o jogo real
- `ablacao_offline` — a CNN contribui? (exige modelo treinado)
- `scripts/smoke_test.py` — env + extractor + predict sem o jogo
- `scripts/inspecionar_vecnormalize.py` — escala/clip da recompensa normalizada num `.pkl`

**Análise de logs** — `python scripts/<nome>.py`
- `metricas_treino` — win rate/sobrevivência/causas por noite (lê `logs/analise/treino_detalhado.log`)
- `enviar_logs_mongodb` / `exportar_logs_xlsx` / `limpar_banco` — pipeline MongoDB/Excel (2 PCs)
- `merge_modelos` — ⚠ descontinuado (média de pesos entre linhagens quebra a política); exige `--force`
- `bump_version` — versionamento (`VERSION` + `src/version.py`)
