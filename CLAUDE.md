# no-more-jumpscares — mapa do projeto

Agente RL (PPO/stable-baselines3) que joga o **FNAF 1 real** por captura de tela + mouse, em
**tempo real** (~0,7s por step — amostra é O recurso escasso). Objetivo: vencer as noites de forma
estável. **Fase atual:** BC warmstart + termostato de entropia + currículo automático
(`docs/PACOTE_BC_ENTROPIA.md`).

## Comandos canônicos (NÃO renomear/mover os alvos)

| Comando | O que faz |
|---|---|
| `python main.py teste` | valida reset/captura/observação (jogo aberto) |
| `python main.py treino [--novo] [--bc modelos/fnaf_bc.zip] [--steps]` | treino (retoma o maior checkpoint sem `--novo`) |
| `python main.py jogar [--ablacao imagem\|estados]` | avaliação determinística |
| `python main.py bc <dataset.json> ...` | treina o Behavioral Cloning |
| `python -m src.utils.<ferramenta>` | calibração/diagnóstico (lista em `docs/README.md`) |
| `python -m src.utils.gravar_gameplay --noite N` | grava demos humanas p/ BC (F9 inicia / F10 para) |
| `python scripts/<script>.py` | smoke test, métricas, MongoDB/xlsx, vecnormalize |
| `tensorboard --logdir logs` | métricas (eventos ficam em `logs/tensorboard/`) |

Verificação offline (sem o jogo): `scripts/smoke_test.py` (6/6) ·
`-m src.utils.testar_recompensa` (13/13) · `-m src.utils.testar_noite` (11/11) ·
`scripts/inspecionar_vecnormalize.py`.

## Mapa de pastas (⚠ = caminho hardcoded em código, não mover)

```
main.py                      ponto de entrada (teste|treino|jogar|bc)
src/environment/fnaf_env.py  ambiente Gymnasium (~1500 linhas): captura, detecção, reward, reset
src/agent/train.py           treino + callbacks (ControladorEntropia, CurriculumCallback, logs)
src/agent/behavioral_cloning.py  BC (dataset, treino, transferir_pesos)
src/agent/multimodal_policy.py   extractor CNN(84x84) + MLP(12 estados) → 256
src/utils/                   ferramentas standalone (python -m src.utils.X) ⚠
src/utils/referencias/       ⚠ templates de detecção COMMITADOS (morte, menu, ameaças, dígitos)
scripts/                     smoke test, análise de logs, MongoDB/xlsx, merge (descontinuado)
docs/                        vivos na raiz; retratos de época em docs/historico/ (índice: docs/README.md)
modelos/                     ⚠ checkpoints + vecnormalize pareado + curriculo.json (git-ignorado)
logs/                        ⚠ treino.log (ENXUTO) · desyncs.log · analise/ (detalhado) · tensorboard/
dados/                       ⚠ datasets de gameplay p/ BC (git-ignorado)
debug/                       transiente: fixtures quadro_*.png (usadas por testes) + saídas; resto em debug/arquivo/
```

## Convenções que importam

- **Logs**: `logs/treino.log` e o console ficam ENXUTOS (leitura de execução). Telemetria
  (Energia fim, Causa, OCORRIDO) vai para `logs/analise/treino_detalhado.log` — é o que os
  parsers preferem. Não adicionar campos na linha de episódio fora dos previstos pelo
  `LOG_PATTERN` (`scripts/enviar_logs_mongodb.py`).
- **Observação**: Dict com `imagem` (84,84,1) uint8 + `estados` (12,) float32 em [0,1]. Mudar o
  shape exige treino do zero; o extractor deriva a dimensão do espaço (nunca hardcodar).
- **GAMMA=0.997** tem fonte única em `fnaf_env.py` (PPO + VecNormalize + shaping Φ precisam casar).
- **Reward**: vitória +500, morte −100, denso ~60/noite por TEMPO REAL, shaping potential-based;
  `clip_reward=100` no VecNormalize (o default 10 achatava vitória e morte no mesmo teto).
- **Hiperparâmetros** por env var `FNAF_*` no `.env` (valores atuais no cabeçalho de
  `docs/REFERENCIA_HIPERPARAMETROS.md`). `n_steps`/`batch` só valem em treino fresco.
- **LSTM desligada nesta fase** (`FNAF_USAR_LSTM=0`): o BC transfere 100% dos pesos só no
  feedforward; gatilho p/ reabrir o A/B em `docs/PACOTE_BC_ENTROPIA.md` §2.7.
- **Medir por taxa de vitória/sobrevivência por noite**, nunca por recompensa (muda entre versões).
- Setup de **2 PCs** (prefixo `PC=` nos logs); merge de pesos entre linhagens é DESCONTINUADO.
- Idioma do projeto: português (código, comentários, docs, logs).
