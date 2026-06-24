# Validação e testes — checklist de execução (ordem segura p/ ligar o RecurrentPPO)

Este é o **passo a passo, de cima para baixo**, para colocar o RecurrentPPO (Decisão 7) e as demais
mudanças em uso **sem queimar amostra de jogo** (o recurso escasso). Cada passo só começa quando o
anterior está ✅. Não pule etapas — a LSTM entra **por último, isolada, contra um controle e com
critério de desistência**.

> **Comandos:** assumem o venv ativo. Se não ativou, troque `python` por `venv/Scripts/python.exe`.
> Ative com `venv\Scripts\activate` (PowerShell).
>
> **Regras de ouro (valem o doc inteiro):**
> 1. **Uma mudança por vez.** Se fizer bundle, saiba que confunde o gate (isole pela tabela do
>    `REFERENCIA_HIPERPARAMETROS.md`).
> 2. **Meça por vitória + tempo de sobrevivência**, NUNCA pela recompensa (muda entre versões).
> 3. **Salve o controle** (`modelos/*.zip` + `modelos/vecnormalize.pkl`) antes de cada mudança.
> 4. **Gate curto offline/ao vivo** antes de comprometer um treino longo.

---

## Passo 0 — Ambiente pronto (uma vez por máquina)
- [ ] Instalar a pilha de ML no venv:
      `venv/Scripts/python.exe -m pip install -r requirements.txt`
- [ ] Conferir os imports:
      `venv/Scripts/python.exe -c "import numpy,torch,stable_baselines3,sb3_contrib,gymnasium,cv2; print('OK', torch.__version__, torch.cuda.is_available())"`
- [ ] `.env` criado e ajustado para este PC:
      `Copy-Item .env.example .env` → editar `FNAF_WINDOW_TITLE`, `PC`, coordenadas e os botões do menu (`FNAF_NEW_GAME_CLICK`, `FNAF_CONTINUE_CLICK`).

> **GPU:** o `torch` do PyPI vem **CPU-only** (`+cpu`) — o treino roda, porém mais devagar. Para CUDA,
> reinstale o torch pelo índice da sua versão de CUDA (`--index-url https://download.pytorch.org/whl/cuXXX`).
> `FNAF_USAR_LSTM` controla o algoritmo (`0`=PPO controle / `1`=RecurrentPPO) — ver Passos 3 e 4.

---

## Passo 1 — Verde OFFLINE (segundos, jogo fechado) — TUDO verde antes de treinar
| ✅ | comando | valida (decisão) | passa se |
|---|---|---|---|
| [ ] | `python -m src.utils.testar_recompensa` | incentivo (D1/D2) + shaping Φ (D4) | Estáveis 6/6 · Alvos D1 2/2 · Shaping D4 3/3 |
| [ ] | `python scripts/smoke_test.py` | env, obs (84,84,1)+(11), CNN 1×, predict, BC | 6/6 checks |
| [ ] | `python -m src.utils.testar_masking` | **reset da LSTM no episódio (D7)** | `episode_start=True` → ação independe do histórico (`OK - masking zera o estado`) |
| [ ] | `python -m src.utils.testar_deteccao_ameaca` | Bonnie/Chica (D4) | lógica de vão (std) 2/2 |
| [ ] | `python -m src.utils.testar_deteccao_energia` | energia (D4B) | Filtro OK |

> **Atenção — reader de imagem precisa de frames de referência.** As partes "rosto 13/13" (ameaça) e
> "reader 13/13" (energia) só rodam se houver os frames capturados em `debug/`. Numa cópia limpa eles
> ficam **ausentes** (`frames ausentes`) e os testes passam só na **lógica** (filtro/std). A acurácia
> real do reader você valida **ao vivo no Passo 2** (`monitor`). Não trate "0/0 ausentes" como falha.
>
> `ablacao_offline` (D5) e `sonda_memoria` (D7) **exigem modelo treinado** → ficam nos Passos 4–6.

---

## Passo 2 — Percepção AO VIVO (minutos, jogo aberto)
- [ ] `python -m src.utils.monitor`, flicando luz/porta:
  - **ESQ:** `rosto` sobe com o Bonnie (porta aberta); `sombra`(std) cai com porta fechada; `mantem` no escuro.
  - **DIR:** `Chica` sobe com a luz direita.
  - **energia:** acompanha o número do jogo e segura (`--`) em câmera/flicker.
- [ ] `python main.py teste` → confere a observação inicial **e a Noite lida** (deve dizer Noite 1).
- [ ] **Manual:** confirmar que `FNAF_NEW_GAME_CLICK` e `FNAF_CONTINUE_CLICK` caem **cada um no seu
      botão do menu** (que só aparece após a morte). Vencer emenda direto na próxima noite (sem menu).
      Se errar a coordenada, após a morte o reset clica no lugar errado e a Noite desincroniza.

---

## Passo 3 — CONTROLE primeiro (feedforward PPO) — a baseline de tudo
O controle é o feedforward com noite + schedules (D6); ele **também é o gate adiado** das D4/D6.
O treino roda sozinho: morre → clica o reset → começa de novo; se a janela some, reabre o jogo
(`FNAF_EXECUTABLE_PATH`). F12 pausa; Ctrl+C encerra salvando.

- [ ] `FNAF_USAR_LSTM=0` no `.env` (padrão).
- [ ] Treino fresco do controle: `python main.py treino --novo`. Acompanhar `logs/treino.log`.
- [ ] Avaliar determinístico: `python main.py jogar` por **~30 episódios** (50+ p/ confiança).
      Anotar **Vitórias/ep + Sobrevivência média** (saída do `jogar`).
- [ ] **Guardar o controle (CRÍTICO):** mover `modelos/*.zip` + `modelos/vecnormalize.pkl` para um
      backup (ex.: `modelos/controle/`). Sem isso, o `jogar`/`treino` da LSTM pega o checkpoint errado
      (ver Passo 4).

> **Quantos episódios até a 1ª vitória?** Por experiência, **~200 eps no mínimo** (não 30). Os 30–50
> servem só para um gate grosseiro de "não regrediu"; para ver a IA *vencer* uma noite, deixe rodar
> várias centenas de episódios. Para *comparar* dois modelos, a taxa de vitória é proporção:
> <30 eps o ruído domina (±20%); 50+ dá confiança; ±10% pediria ~80 eps.

> **A/B em duas máquinas (em paralelo):** cada PC tem seu próprio `modelos/`, então não há risco de
> misturar checkpoints. Rode o **controle** (`FNAF_USAR_LSTM=0`) numa máquina e a **LSTM**
> (`FNAF_USAR_LSTM=1`) na outra, com o **mesmo orçamento**, e compare vitória + sobrevivência no fim.

---

## Passo 4 — RecurrentPPO (LSTM): A/B isolado contra o controle
A maior aposta. Muda **só o algoritmo** — recompensa, `VecNormalize`, `gamma`, noite e schedules
**idênticos** ao controle.
- [ ] Pré-voo: `testar_masking` PASSOU no Passo 1 (obrigatório antes de qualquer run longo).
- [ ] `modelos/` **sem os checkpoints do controle** (já movidos no Passo 3). Se sobrar um `.zip` de
      PPO, `RecurrentPPO.load()` tenta carregá-lo e **quebra** (arquiteturas diferentes).
- [ ] `FNAF_USAR_LSTM=1` no `.env`.
- [ ] Treino **fresco obrigatório**: `python main.py treino --novo`, **mesmo orçamento do controle**
      (não dá para "continuar" um PPO como LSTM).
- [ ] Durante/depois: `python -m src.utils.sonda_memoria` em **~6–10 checkpoints** do modelo LSTM —
      a recorrência tem que **usar memória** (ação muda com o histórico); se vier **INERTE**, virou
      feedforward caro (rever masking/amostra).
- [ ] Avaliar com `FNAF_USAR_LSTM=1` + `python main.py jogar` — o flag faz carregar o RecurrentPPO e
      **propagar o estado da LSTM**. Sem o flag, a avaliação mente (ou o load falha).
- [ ] **Critério de desistência (definido ANTES):** se em ~100k steps a LSTM **não empatar** a
      sobrevivência do controle → reverter pro feedforward.

---

## Passo 5 — Decisão do A/B
- [ ] Comparar **Vitórias/ep + Sobrevivência média** (mesmo orçamento, ~30–50 eps cada): controle vs LSTM.
- [ ] LSTM **≥** controle → manter (`FNAF_USAR_LSTM=1`). Senão → reverter: `FNAF_USAR_LSTM=0` e
      restaurar o controle do backup do Passo 3.

> A noite (D7) não tem teste offline próprio — é rastreada **internamente** pelo desfecho
> (vitória→+1; morte→Noite 1 com `new_game`). Valida no `logs/treino.log` (`Noite X`) e no `main.py teste`.

---

## Passo 6 (opcional) — D5: a CNN está contribuindo?
- [ ] `python -m src.utils.ablacao_offline` em ~6–10 checkpoints (tendência da sensibilidade à imagem).
- [ ] `python main.py jogar --ablacao imagem|estados` em 1–2 modelos bons. Se zerar a **imagem** quase
      não derrubar a sobrevivência, a CNN não contribui; se zerar os **estados** derrubar muito, a
      política depende deles. Comparar com o run cheio (sem flag).

---

## Apêndice — parâmetros de referência (atuais)
- `gamma=0.997` · `learning_rate=linear(3e-4, 3e-5)` · `ent_coef` 0.02→0.005 (gate vitória ≥20%/50 eps)
  · `n_steps=2048` · `batch_size=64` · `n_epochs=10`.
- **LSTM:** `lstm_hidden_size=128` · `n_lstm_layers=1` · `enable_critic_lstm=True`. Ligar com `FNAF_USAR_LSTM=1`.
- **Noite:** `FNAF_RESET_METODO=new_game` (morte→Noite 1) ou `continue` (repete). `MAX_NOITE=7`.
- Episódio ~500–700 steps (~6–8 min real) · treino-alvo 500k steps (dias, multi-sessão, retomar via
  `python main.py treino` **sem** `--novo`).

---

## Resumo (cola rápida — a ordem)
```bash
# Passo 0 — ambiente (uma vez)
venv/Scripts/python.exe -m pip install -r requirements.txt

# Passo 1 — offline (segundos, jogo fechado): tudo verde antes de treinar
python -m src.utils.testar_recompensa
python scripts/smoke_test.py
python -m src.utils.testar_masking
python -m src.utils.testar_deteccao_ameaca
python -m src.utils.testar_deteccao_energia

# Passo 2 — ao vivo (jogo aberto): percepção + noite
python -m src.utils.monitor
python main.py teste

# Passo 3 — CONTROLE (FNAF_USAR_LSTM=0): treino fresco, avaliar, depois backup de modelos/
python main.py treino --novo
python main.py jogar

# Passo 4 — LSTM (FNAF_USAR_LSTM=1), modelos/ limpo, treino fresco, mesmo orçamento
python main.py treino --novo
python -m src.utils.sonda_memoria      # ~6-10 checkpoints
python main.py jogar                   # avaliar com FNAF_USAR_LSTM=1

# Passo 6 — opcional (D5): a CNN contribui?
python -m src.utils.ablacao_offline
python main.py jogar --ablacao imagem
```
