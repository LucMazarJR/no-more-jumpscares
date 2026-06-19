# Validação e testes — playbook completo (Decisões 1 → 7)

Tudo que validar antes/depois de cada mudança, com comandos, parâmetros e critérios. O fio
condutor: **amostra de jogo real é o recurso escasso** — valide offline (segundos) o máximo
possível antes de gastar horas de treino.

---

## 1. Princípios (regras de ouro)
1. **Uma mudança por vez** — ou *bundle pragmático*, ciente de que confunde o gate (use a tabela de
   isolamento do `REFERENCIA_HIPERPARAMETROS.md` se piorar).
2. **Meça pelo que NÃO depende da recompensa:** **taxa de vitória** e **tempo de sobrevivência**.
   Nunca pela recompensa (muda entre versões).
3. **Salve o controle** (melhor `modelos/*.zip` + `modelos/vecnormalize.pkl`) antes de cada mudança.
4. **Gate curto** antes de comprometer um treino longo.

---

## 2. Testes OFFLINE (segundos — rode SEMPRE antes de treinar)
| comando | valida (decisão) | passa se |
|---|---|---|
| `python -m src.utils.testar_recompensa` | incentivo (D1/D2) + shaping Φ (D4) | estáveis 6/6 · alvos D1 2/2 · shaping 3/3 |
| `python -m src.utils.testar_deteccao_ameaca` | Bonnie/Chica (D4) | rosto 13/13 · sombra/vazio 2/2 |
| `python -m src.utils.testar_deteccao_energia` | energia (D4B) | reader 13/13 · filtro OK |
| `python scripts/smoke_test.py` | env, obs (84,84,1)+(11), CNN 1×, predict, BC | todos OK |
| `python -m src.utils.testar_masking` | reset da LSTM no episódio (D7) | episode_start=True → ação independe do histórico |
| `python -m src.utils.ablacao_offline` | a CNN contribui? (D5) | **precisa de modelo**: sensib. imagem > 0 |
| `python -m src.utils.sonda_memoria` | a LSTM usa memória? (D7) | **precisa de modelo LSTM**: ação muda com o histórico |

> **Noite (D7)** não tem teste offline próprio — é rastreada **internamente** pelo desfecho
> (vitória→+1; morte→Noite 1 com `new_game`). Validada por: `smoke_test` (obs com 11 estados) +
> ao vivo (o `treino.log` mostra `Noite X`, e `python main.py teste` imprime a noite lida).

---

## 3. Validação AO VIVO da percepção (minutos — jogo aberto)
`python -m src.utils.monitor`, flicando luz/porta:
- **ESQ:** `rosto` sobe com o Bonnie (porta aberta); `sombra`(std) cai com porta fechada; `mantem` no escuro.
- **DIR:** `Chica` sobe com a luz direita.
- **energia** acompanha o número do jogo e segura (`--`) em câmera/flicker.

`python main.py teste` → confere a observação inicial, inclusive a **Noite** lida.

---

## 4. GATE de treino (metodologia, com números)
1. **Salvar controle:** copiar o melhor `modelos/*.zip` + `modelos/vecnormalize.pkl` p/ um backup.
2. **Treinar a janela:** `python main.py treino --novo` (fresco) por **~30–50 episódios**
   (≈15k–35k steps; **~3–6 h** em tempo real; F12 pausa). Acompanhar `logs/treino.log`.
3. **Avaliar os dois** determinístico, **~30 episódios cada** (≈3–4 h cada):
   `python main.py jogar` no controle e no novo → comparar **Vitórias/ep + Sobrevivência média**.
4. **Critério:** novo **≥** controle → manter; senão reverter (ou isolar, no bundle).
- *Por que ~30 eps:* taxa de vitória é proporção; com <20–30 eps o ruído do jogo domina (±20%).
  30 dá leitura grosseira; **50+** para confiança. Margem ±10% pediria ~80 eps (caro).

---

## 5. Por decisão — o que rodar e olhar
- **D1 recompensa / D2 teste:** `testar_recompensa` + gate (a passiva não pontua mais que o bom jogo).
- **D3 VecNormalize:** confirmar que `jogar` carrega sem as stats (norm_obs=False) e o treino retoma com `vecnormalize.pkl`; curva mais suave.
- **D4 ameaça + shaping:** `testar_deteccao_ameaca` + `monitor` + shaping no `testar_recompensa`. Gate: sobrevivência sobe; **vigiar reward-hacking** (shaping alto + sobrevivência baixa).
- **D4B energia:** `testar_deteccao_energia` + `monitor`.
- **D5 ablação da CNN:** `ablacao_offline` em **~6–10 checkpoints** (tendência) + `python main.py jogar --ablacao imagem|estados` em **1–2** modelos bons.
- **D6 schedules:** offline (linear/EntropiaSchedule) + gate fresco; se piorar, **isolar** pela tabela do `REFERENCIA_HIPERPARAMETROS.md` (sintoma→botão).
- **D7 LSTM + noite:**
  - `smoke_test` (obs 11 + extractor) · `testar_masking` (estado zera no episódio) · `python main.py teste` (noite lida).
  - **A/B:** controle = **feedforward** (`FNAF_USAR_LSTM=0`) + noite + D6; novo = **LSTM** (`FNAF_USAR_LSTM=1`), mesmo orçamento. Esse controle **também é o gate adiado** das D4/D6.
  - `sonda_memoria` em **~6–10 checkpoints** do modelo LSTM (a recorrência está usando memória?).
  - **Critério de desistência:** se em ~100k steps a LSTM não empatar a sobrevivência do controle → reverter.

---

## 6. Parâmetros de referência (atuais)
- `gamma=0.997` · `learning_rate=linear(3e-4, 3e-5)` · `ent_coef` 0.02→0.005 (gate vitória ≥20%/50 eps)
  · `n_steps=2048` · `batch_size=64` · `n_epochs=10`.
- **LSTM:** `lstm_hidden_size=128` · `n_lstm_layers=1` · `enable_critic_lstm=True`. Ligar com `FNAF_USAR_LSTM=1`.
- **Noite:** `FNAF_RESET_METODO=new_game` (morte→Noite 1) ou `continue` (repete). `MAX_NOITE=7`.
- Episódio ~500–700 steps (~6–8 min real) · treino-alvo 500k steps (dias, multi-sessão, retomar via `python main.py treino`).

---

## 7. Ordem recomendada de um ciclo de validação
```
# offline (segundos) — tem que estar tudo verde antes de treinar
python -m src.utils.testar_recompensa
python -m src.utils.testar_deteccao_ameaca
python -m src.utils.testar_deteccao_energia
python -m src.utils.testar_masking
python scripts/smoke_test.py
# ao vivo (jogo aberto) — percepção
python -m src.utils.monitor
python main.py teste
# treino + gate (horas) — salvar controle ANTES
python main.py treino --novo
python main.py jogar            # controle e novo, comparar vitória/sobrevivência
# diagnóstico (precisa de modelo treinado)
python -m src.utils.ablacao_offline
python -m src.utils.sonda_memoria
```
