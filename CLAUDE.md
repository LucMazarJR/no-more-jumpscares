# no-more-jumpscares — mapa do projeto

Agente de RL (stable-baselines3) que joga o **FNAF 1 real** por captura de tela + mouse, em
**tempo real** (~0,9 s por step; 100k steps ≈ 25 h de máquina). **Amostra é O recurso escasso**:
um teste longo custa dias. Objetivo: vencer as noites de forma estável.

**Estado atual, hipóteses abertas e próximos passos:** [docs/ESTADO_ATUAL.md](docs/ESTADO_ATUAL.md).
Este arquivo guarda só o que é permanente. Nada de "fase atual" aqui.

## Comandos canônicos (NÃO renomear/mover os alvos)

| Comando | O que faz |
|---|---|
| `python main.py teste` | valida reset/captura/observação (jogo aberto) |
| `python main.py treino [--novo] [--bc <zip>] [--nome <tag>]` | treino (sem `--novo`, retoma o maior checkpoint) |
| `python main.py jogar [--modelo <zip>] [--estocastico] [--noite N] [--episodios K] [--ablacao imagem\|estados]` | avaliação sem aprender (log em `logs/analise/avaliacoes.log`) |
| `python main.py bc [dataset.json ...]` | treina o Behavioral Cloning (sem args: todos os `dados/*/dataset.json`) |
| `python -m src.utils.gravar_gameplay --noite N` | grava demos humanas para o BC (F9 inicia / F10 para) |
| `python -m src.utils.<ferramenta>` | calibração/diagnóstico (lista em `docs/README.md`) |
| `python scripts/<script>.py` | métricas, taxa de ação, MongoDB/xlsx, vecnormalize |
| `tensorboard --logdir logs/tensorboard` | métricas (uma pasta por run) |

Verificação offline (sem o jogo), todos devem passar: `scripts/smoke_test.py` ·
`-m src.utils.testar_recompensa` · `-m src.utils.testar_noite` · `-m src.utils.testar_deteccao_ameaca`
· `-m src.utils.testar_masking` (LSTM).

## Mapa de pastas (⚠ = caminho hardcoded em código, não mover)

```
main.py                      ponto de entrada (teste|treino|jogar|bc)
src/environment/fnaf_env.py  ambiente Gymnasium: captura, detecção, recompensa, reset
src/agent/train.py           treino + callbacks + CONSTANTES de algoritmo/tuning
src/agent/behavioral_cloning.py  BC (dataset, treino feedforward/recorrente, transferir_pesos)
src/agent/multimodal_policy.py   extractor CNN (imagem) + MLP (estados)
src/utils/                   ferramentas standalone (python -m src.utils.X) ⚠
src/utils/referencias/       ⚠ templates de detecção COMMITADOS (morte, menu, ameaças, dígitos)
scripts/                     smoke test, análise de logs, MongoDB/xlsx
docs/                        docs vivos na raiz; retratos de época em docs/historico/ (índice: docs/README.md)
modelos/                     ⚠ checkpoints + vecnormalize pareado + curriculo.json + run.json (git-ignorado)
logs/                        ⚠ git-ignorado (estrutura abaixo)
dados/                       ⚠ datasets de gameplay para o BC (git-ignorado)
debug/                       transiente: fixtures quadro_*.png (usadas por testes) + saídas de ferramentas
```

```
logs/treino.log                       run CORRENTE, enxuto (leitura durante a execução)
logs/desyncs.log                      run CORRENTE, dessincronias de estado por episódio
logs/analise/treino_detalhado.log     run CORRENTE com telemetria (fonte preferida dos parsers)
logs/analise/historico/<run>_*.log    runs passadas (arquivadas automaticamente no --novo)
logs/analise/avaliacoes*.log          saídas do `main.py jogar` (nunca misturadas ao treino)
logs/tensorboard/<run>/               uma pasta por run; _pre_pacote/ = runs antigas sem nome
```

## Convenções que importam

- **Runs têm nome.** Formato `AAAA-MM-DD_runN_<algo>[_tag]`, gravado em `modelos/run.json`.
  - O nome é a pasta do tensorboard e o prefixo dos logs arquivados.
  - `--novo` cria uma run nova e arquiva os logs da anterior. Retomar continua a mesma run.
  - Cada sessão no log começa com `run: <nome> | NOVO/RETOMADA | commit | data`.
  - **Nunca misture runs numa análise**: já produziu conclusões erradas mais de uma vez.
- **Logs:** `logs/treino.log` e o console ficam ENXUTOS. A telemetria (energia final, causa, OCORRIDO) vai para `logs/analise/treino_detalhado.log`. Não adicione campos à linha de episódio fora dos previstos pelo `LOG_PATTERN` (`scripts/enviar_logs_mongodb.py`). A linha `Treino iniciado` fica sozinha, porque o parser casa a linha exata.
- **Observação:** um Dict com `imagem` (uint8) e `estados` (float32 em [0,1]). Mudar o shape exige treino do zero. O extractor deriva as dimensões do espaço, então nunca hardcode tamanhos.
- **GAMMA** tem fonte única em `fnaf_env.py`: PPO, VecNormalize e o shaping Φ precisam usar o mesmo valor para o shaping telescopar.
- **Recompensa:** terminais grandes (vitória ≫ morte ≫ denso), sinal denso por TEMPO REAL e dicas só por shaping potential-based (telescopa, não move o ótimo). `clip_reward` do VecNormalize precisa deixar a vitória passar inteira; o default 10 achatava vitória e morte no mesmo teto.
- **Algoritmo e tuning são constantes de código** (`src/agent/train.py`), não `.env`. O `.env` já divergiu entre máquinas e fez produção rodar a configuração errada. No `.env` fica só o que varia por máquina (janela, caminho, coordenadas, timings, `PC`), mais `FNAF_RESET_METODO` e `FNAF_NOITE_DESEJADA`.
- **Medição:**
  - Use taxa de vitória e sobrevivência **por noite**, nunca recompensa (a escala muda entre versões).
  - O treino amostra ações (estocástico). Para comparar avaliação com treino, use `jogar --estocastico`.
  - Sobrevivência e causa da morte respondem mais rápido que a taxa de vitória.
- **Método de experimento:**
  - Uma variável por vez.
  - Critério de aborto registrado ANTES de uma run longa.
  - Refutar com dados uma afirmação de doc ou comentário é resultado, não problema.
- **Setup de 2 PCs** (prefixo `PC=` nos logs). O merge de pesos entre linhagens foi DESCONTINUADO porque quebra a política.
- **Idioma do projeto:** português (código, comentários, docs, logs).
