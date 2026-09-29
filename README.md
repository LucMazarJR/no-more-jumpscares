# no-more-jumpscares

Projeto de agente por reforço (PPO) que interage com o jogo
Five Nights at Freddy's (FNAF1) via captura de tela e controle
de mouse/teclado. O objetivo é treinar uma IA para sobreviver
às noites do jogo automaticamente.

**Visão Geral**

- **Entrada:** screenshots do jogo (captura de janela, 84×84) + vetor de estados internos
- **Ações:** cliques nas posições mapeadas (portas, luzes, câmeras) — 17 ações discretas
- **Algoritmo:** PPO/RecurrentPPO via `stable_baselines3`, com warmstart por Behavioral Cloning

> **Onde o projeto está e o que vem a seguir:** [docs/ESTADO_ATUAL.md](docs/ESTADO_ATUAL.md).
> Índice de toda a documentação: [docs/README.md](docs/README.md).

**Pré-requisitos**

- **Sistema operacional:** Windows (testado)
- **Jogo:** Five Nights at Freddy's instalado (obrigatório)
- **Python:** 3.10+ (recomenda-se 3.10–3.13)
- **Dependências:** instale via `pip install -r requirements.txt`
- **Recomendado:** escala de exibição do Windows em 100% para
  garantir consistência nas coordenadas de clique

Instalação rápida

```bash
# criar e ativar virtualenv (Windows)
python -m venv venv
venv\Scripts\activate

# instalar dependências
pip install -r requirements.txt
```

Configuração — modo janela e coordenadas

- O jogo DEVE estar em modo janela (não em fullscreen). No FNAF
  isso costuma ser alternado com **ALT+ENTER**. Antes de executar
  qualquer script de calibração ou treino, pressione **ALT+ENTER**
  para colocar o jogo em modo janela.
- As coordenadas de clique ficam no `.env` (copie `.env.example` como ponto
  de partida). Sempre recalibre com o jogo em modo janela e na posição
  que usará durante o treino.
- Para fallback automático quando o jogo fechar inesperadamente,
  configure no `.env`:
  - `FNAF_EXECUTABLE_PATH` (caminho completo do `.exe` ou atalho `.lnk`)
  - `FNAF_REABRIR_ESPERA_SEGUNDOS` (espera antes do ALT+ENTER)
  - `FNAF_POS_ALT_ENTER_ESPERA_SEGUNDOS` (espera após ALT+ENTER)

Como calibrar

1. Abra o jogo e pressione **ALT+ENTER** (modo janela).
2. Posicione a janela do jogo e confirme escala 100%.
3. Use o script guiado para capturar todas as coordenadas de uma vez:

```bash
python -m src.utils.calibrar_por_passos
```

   Ou capture individualmente com `src.utils.calibrar`:

```bash
# coordenadas em tempo real (mova o mouse sobre cada botão)
python -m src.utils.calibrar

# captura imagem de referência de morte (deixe a tela de Game Over aparecer)
python -m src.utils.calibrar morte

# captura imagem de referência de vitória (quando aparecer o 6 AM)
python -m src.utils.calibrar vitoria

# captura o template do indicador 'YOU' no mapa de câmeras
python -m src.utils.calibrar camera_aberta
```

4. Cole os valores gerados no `.env`.

Referências visuais

- As imagens de referência ficam em `src/utils/referencias/`.
  São necessárias: `morte.png`, `vitoria.png` e `camera_aberta.png`.
  As duas primeiras são geradas pelos comandos acima. O template
  `camera_aberta.png` é gerado pelo comando `camera_aberta` e já está
  incluso no repositório como ponto de partida — recalibre se necessário.

Comandos principais

- Validar o projeto sem o jogo aberto (ambiente, rede e modelo):

```bash
python scripts/smoke_test.py
```

- Testar reset/observação (verifica se a captura funciona):

```bash
python main.py teste
```

- Rodar treino. Continua automaticamente do checkpoint mais avançado em `modelos/`; use
  `--novo` para começar do zero (cria uma run nova com nome, ver abaixo):

```bash
python main.py treino
python main.py treino --novo --bc modelos/fnaf_bc.zip --nome bc_recorrente
```

- Avaliar um modelo (sem aprender). Resultado por episódio e resumo por noite em
  `logs/analise/avaliacoes.log`:

```bash
python main.py jogar                                        # último checkpoint, determinístico
python main.py jogar --modelo modelos/fnaf_bc.zip --estocastico --noite 1 --episodios 20
python main.py jogar --ablacao imagem                       # a CNN contribui?
```

- **Runs têm nome.** `--novo` cria `AAAA-MM-DD_runN_<algo>[_<tag do --nome>]`, gravado em
  `modelos/run.json`. O nome é a pasta em `logs/tensorboard/` e o prefixo dos logs da run,
  que vão para `logs/analise/historico/` quando a próxima run começa. Retomar continua a
  mesma run. Não há média de pesos entre modelos: escolha o melhor e continue o treino dele.

Exportar logs para MongoDB

- Instale o driver do MongoDB para Python:

```bash
pip install pymongo
```

- Configure no `.env` (copie de `.env.example`):
  - `PC` (identificador da maquina)
  - `MONGO_URI`
  - `MONGO_DATABASE`
  - `MONGO_COLLECTION`
- Execute o treino e depois envie:

```bash
python main.py treino
python scripts/enviar_logs_mongodb.py
```

Formato padrão do log textual (`logs/treino.log` e `logs/analise/treino_detalhado.log`):

```text
============================================================
Treino iniciado
run: 2026-09-28_run5_rppo_bc | NOVO (BC: modelos/fnaf_bc.zip) | commit a1b2c3d | 2026-09-28 22:10
============================================================
pc0 | Ep    1 | Noite 1 | MORTE    | Passos:    455 | Tempo:    5.20 min | Recompensa:    114.8 | Taxa vitória: 0.0% | Energia fim:  0.0% | Causa: morte_energia
pc0 | Ep    2 | Noite 1 | VITORIA  | Passos:    600 | Tempo:    9.00 min | Recompensa:    540.0 | Taxa vitória: 50.0% | Energia fim:  0.0% | Causa: vitoria_apagao
```

O campo `Causa:` separa **skill de sorte** no desfecho:
- `vitoria_gerida` — venceu com energia de pé (gestão real); `vitoria_apagao` — venceu **após** a
  energia zerar (luzes apagadas; o 6 AM chegou antes do Freddy — vitória de alta variância);
- `morte_energia` — apagou e o Freddy pegou (gargalo de energia); `morte_animatronico` — morreu
  **com** energia (um animatrônico passou a porta — defesa/timing).

O `scripts/metricas_treino.py` quebra esses tipos por noite (quanto da taxa de vitória é sorte do
apagão, qual o gargalo de morte). Em caso de fechamento inesperado do jogo, o episódio é marcado
como `INTERROMPIDO` e não conta para a taxa de vitória.

Exemplos úteis para envio ao MongoDB:

```bash
# Validar payload sem enviar para o banco
python scripts/enviar_logs_mongodb.py --dry-run --print-json

# Forçar uma fonte de log específica
python scripts/enviar_logs_mongodb.py --source logs_analysis/episodes.csv
```

Log de dessincronizações

A cada episódio, o ambiente registra em `logs/desyncs.log`:

```
Ep    1 | steps   584 | desfecho: morte     | SYNC camera:   2 | SYNC porta:   0 | porta falha:   1
```

- **SYNC camera**: correções de estado da câmera via template matching
- **SYNC porta**: correções de estado de porta via leitura de pixel
- **porta falha**: cliques de porta que o jogo não registrou (confirmados por 3 tentativas)

Observações e dicas

- Durante o treino, segure **F12** para pausar a execução.
- Use `tensorboard --logdir logs/tensorboard` para visualizar métricas do treino (uma pasta
  por run; `_pre_pacote/` guarda as runs antigas de jun/jul-2026, sem BC).
- Se muitos episódios terminam com morte imediata (passos = 1),
  verifique as coordenadas no `.env` e confirme o modo janela.
- Se a captura não encontrar a janela pelo título, ajuste
  `FNAF_WINDOW_TITLE` no `.env`.

Arquivos úteis

- **Índice da documentação: [docs/README.md](docs/README.md)** (dos docs vivos aos históricos)
- Código do ambiente: [src/environment/fnaf_env.py](src/environment/fnaf_env.py)
- Scripts de calibração: [src/utils/calibrar.py](src/utils/calibrar.py), [src/utils/calibrar_por_passos.py](src/utils/calibrar_por_passos.py)
- Captura de tela: [src/utils/capture.py](src/utils/capture.py)
- Histórico de alterações: [docs/historico/ALTERACOES_COMPLETAS.md](docs/historico/ALTERACOES_COMPLETAS.md)
- Logs da run corrente: `logs/treino.log` (enxuto, leitura de execução) ·
  `logs/analise/treino_detalhado.log` (causa/energia por episódio) · `logs/desyncs.log`.
  Runs passadas: `logs/analise/historico/<run>_*.log`. Tensorboard: `logs/tensorboard/<run>/`.
