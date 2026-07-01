# Pacote BC + Termostato de Entropia — estratégias, porquês e runbook

> **Data:** 01/07/2026 · **Status:** implementado, aguardando gravação de gameplay + treino fresco
> **Problema que ataca:** a entropia da política caía rápido, o agente chegava a ~30% de vitória
> na Noite 1 e estagnava nesse ótimo local — "tem potencial pra aprender, mas para no bom lugar
> antes de aprender de verdade".

---

## 1. O diagnóstico em uma página

O bundle anterior (n_steps 8192, n_epochs 4, target_kl, gate de entropia em 40%) atacava o
mecanismo certo — lotes maiores e mais diversos, menos reuso do mesmo lote — mas tinha três
buracos que nenhum ajuste de n_steps/n_epochs fecha:

1. **Nada mirava a entropia diretamente.** O ent_coef ficava FIXO em 0.02 até o gate
   (win_rate ≥ 40%) abrir — e estagnado em 30%, o gate nunca abria. Já se sabia que 0.03 fixo
   deixava a política aleatória demais (morria por apagão ~7min) e que 0.02 colapsava. Quando
   nenhum valor fixo funciona, o problema não é o valor: é ser fixo.
2. **A vitória chegava castrada no gradiente.** O `VecNormalize(norm_reward=True)` usa
   `clip_reward=10` por default. Medido no `vecnormalize.pkl` real deste PC
   (`scripts/inspecionar_vecnormalize.py`):

   ```
   ret_rms.var: 0.0358  (std = 0.19)
   vitoria (+500):  normalizada +2641.16 -> apos clip +10.00  [CLIPADA]
   morte   (-100):  normalizada  -528.23 -> apos clip -10.00  [CLIPADA]
   razao vitoria:morte no gradiente: 1.00:1  (design: 5:1)
   ```

   Para o gradiente, VENCER valia exatamente o mesmo que MORRER com sinal trocado — a
   assimetria 5:1 desenhada na recompensa nunca chegava na rede.
3. **Exploração aleatória é punida pelo próprio jogo.** Entropia alta = toggles aleatórios de
   porta/luz = apagão antes das 6AM = morte garantida. O dithering uniforme não consegue nem
   ALCANÇAR a estratégia reativa (fechar porta QUANDO a ameaça aparece) para depois refiná-la
   — por isso "mais entropia" sozinho nunca resolveria.

E dois agravantes estruturais: o risco do Foxy punia o agente por uma variável que **não estava
na observação** (tempo sem checar câmera), e o currículo era manual (`.env` fixo).

---

## 2. As estratégias (problema → solução → onde)

### 2.1 Termostato de entropia — `ControladorEntropia` (`src/agent/train.py`)

**Substitui:** `EntropiaSchedule` (gate binário por win_rate + decaimento linear).

**Ideia:** controle em malha fechada. A cada rollout o callback LÊ a entropia real da política
(H, em nats) e ajusta o ent_coef para ela rastrear um ALVO que decai devagar ao longo do treino:

```
H_alvo: 1.5 nats (início) → 0.75 nats (fim)
        e^1.5 ≈ 4.5 ações efetivas       e^0.75 ≈ 2.1 ações efetivas
        (uniforme sobre 17 ações seria ln(17) ≈ 2.83 nats)

ajuste por rollout:  ent_coef *= exp(clamp(0.7 · (H_alvo − H), ±0.20))
limites:             ent_coef ∈ [0.003, 0.03]
```

- Política colapsando (H < alvo) → coeficiente sobe sozinho (até +20%/rollout — dobra em ~4
  rollouts ≈ 3h, rápido o bastante para reagir, lento o bastante para não chocar).
- Aleatória demais (H > alvo) → desce. É o que faltava nos dois modos de falha conhecidos.
- **Medição:** `train/entropy_loss` do SB3 (= −média(H), igual no PPO e no RecurrentPPO), lida
  em `_on_rollout_end` — carrega o train() do rollout anterior (defasagem de 1 rollout,
  irrelevante: o colapso se desenrola por dezenas de rollouts).
- **Teto 0.03 de propósito:** 0.03 FIXO já causou morte por apagão; como teto transitório de um
  controlador só é atingido sob pressão de colapso.

**Knobs (.env):** `FNAF_H_INICIO`, `FNAF_H_FIM`, `FNAF_ENT_MIN/MAX`, `FNAF_ENT_GANHO` (oscilou?
0.4), `FNAF_ENT_PASSO_MAX`. **Tensorboard:** `custom/entropia` (H crua), `custom/h_alvo`,
`custom/ent_coef`.

### 2.2 `clip_reward=100` no VecNormalize (`src/agent/train.py`)

**Problema (medido, seção 1):** clip default 10 achatava vitória e morte no mesmo ±10 → razão
1:1. **Solução:** `clip_reward=100` nas duas construções E forçado após `VecNormalize.load`
(o pickle carrega o clip da época do save). Num run maduro o std dos retornos é dominado pelos
terminais (≈ 20), então a vitória normalizada (≈ +25) passa INTEIRA e a razão 5:1 se restaura.
Nos primeiros episódios (std ainda pequeno) ambos os terminais saturam em ±100 — transitório
curto: o std sobe assim que os primeiros terminais entram na estatística.

**Contingência:** `train/value_loss` explodindo → `FNAF_CLIP_REWARD=50` e reiniciar fresco
(decidir nos primeiros 30k steps; o vecnormalize pareado de checkpoint carrega o clip antigo).

### 2.3 12º estado: `tempo_sem_camera` (`src/environment/fnaf_env.py`)

**Problema:** o Φ do shaping punia "câmera negligenciada" (proxy do Foxy) usando um contador
interno (`passos_sem_camera`) que o agente NÃO via — punição por variável fora da observação
torna o ambiente não-Markov à toa (e foi o motivo original de ligar a LSTM).

**Solução:** a grandeza virou tempo REAL (`_tempo_sem_camera()`, wall-clock como energia/tempo
— steps têm duração variável) e entrou na observação como 12º estado, normalizado por
`FOXY_SATURACAO_S = 28s` (satura em 1.0). O Φ usa a MESMA grandeza com paciência
`FOXY_PACIENCIA_S = 14s` — o agente agora VÊ exatamente o que o shaping cobra. Potential-based
continua telescopando (não move o ótimo); treino é do zero, então não há crítico preso à
observação antiga.

**Efeito colateral desejado:** o espaço de estados agora é quase-Markov (ameaças são HELD, o
Foxy é observável) → o PPO feedforward volta a ser suficiente (ver 2.7).

### 2.4 BC warmstart — a maior alavanca (`gravar_gameplay.py`, `behavioral_cloning.py`)

**Problema:** cada step custa ~0.7s de relógio (jogo real, sem aceleração). Descobrir o básico
por tentativa aleatória custa semanas — e é ativamente punido pela energia (seção 1, item 3).

**Solução:** você grava ~2h jogando (noites 1–5), o BC pré-treina a rede inteira por clonagem,
e o RL começa de PERTO de uma estratégia vencedora em vez do zero.

**"Mas o BC não faria só chegar mais rápido no mesmo lugar?"** Meio certo — e é por isso que
ele não vem sozinho:

- *Certo:* BC não conserta a dinâmica da entropia. A política clonada nasce "afiada" e o RL
  por cima pode colapsar em volta dela. Sozinho, o BC muda ONDE estagna, não SE estagna.
- *Equivocado:* "o mesmo lugar" não. O platô de 30% é o ótimo local alcançável POR EXPLORAÇÃO
  ALEATÓRIA sob a pressão da energia. A estratégia reativa humana é uma bacia DIFERENTE, que o
  dithering não alcança nunca. Estagnar lá = vencer noites 1–3+, não 30% da Noite 1.
- *O pacote fecha o buraco:* o termostato (2.1) é exatamente o anti-colapso que falta ao BC
  puro — se a política clonada nascer com H abaixo do alvo, o ent_coef sobe gradualmente
  (máx +20%/rollout, sem destruir o que foi clonado).

**BC = ponto de partida · termostato = busca viva · currículo = noites certas na hora certa.**

**O gravador tinha 4 defeitos que sabotariam o BC** (corrigidos antes de qualquer gravação):

| Defeito | Consequência se gravasse antes | Correção |
|---|---|---|
| Gravava só 8 estados (sem `noite`, ameaças, Foxy) | `dado.get("noite", 1)` rotularia TUDO como Noite 1 → a política aprenderia a IGNORAR a dificuldade | grava os 12 estados; `--noite N` obrigatório |
| Capturava a janela INTEIRA (barra de título/bordas) | frames de BC ≠ frames que o RL vê (env captura a área cliente e escala p/ 1280x720) | mesma `regiao_cliente(melhor_janela(...))` + resize do env |
| Rotulagem com shift invertido: ação associada ao frame capturado DEPOIS do efeito dela | BC clonaria (consequência → ação) em vez de (observação → ação) | registro pendente: par `(frame_t, ação decidida vendo frame_t)` |
| Sem rótulo de ameaça | portas fechando "do nada" no dataset → BC aprende portas aleatórias | `FNAFEnv` fantasma reusa `_atualizar_ameaca` + templates do treino |

**E o treino do BC ganhou defesas:** `WeightedRandomSampler` (1/freq da classe, capado em 10×
a mediana — o oceano de "nada" não afoga as portas), split 90/10 ESTRATIFICADO por classe,
recall por classe na validação, melhor checkpoint por macro-recall, early stop (paciência 15).

**Aceite do BC:** recall val de portas ≥ 60%, câmera ≥ 50%, luzes ≥ 40%, top-1 ≥ 70%.

### 2.5 Currículo automático — `CurriculumCallback` (`src/agent/train.py`)

**Problema:** promover noite = editar `FNAF_NOITE_DESEJADA` no .env e reiniciar; e o antigo
gate misturava TODAS as noites numa janela só.

**Solução:** callback promove `noite_desejada` (via `set_attr`, chega no `decidir_reset`)
quando a janela CHEIA da noite ALVO (≥30 eps) cruza `FNAF_CURRICULO_LIMIAR` (50%). Nunca
rebaixa (morte em noite ≤ alvo → Continue retoma; as anteriores continuam no caminho — mistura
natural contra esquecimento). Persiste em `modelos/curriculo.json`; retomada aplica
`max(.env, json)`. Para regredir de propósito: apague/edite o json.
**Tensorboard:** `custom/curriculo/noite_alvo`.

### 2.6 Telemetria de causas (`fnaf_env.py`, `train.py`)

**Problema:** "morre e não se sabe DO QUÊ" — sem separar apagão de animatrônico, cada ajuste é
chute (ex.: `morte_energia` dominante = política gastadora/aleatória → baixar H_alvo;
`morte_animatronico` = defesa/timing → BC fraco em portas).

**Solução:** toda linha de episódio do `treino.log` ganha `| Energia fim: X% | Causa: <rotulo>`
(formato que `enviar_logs_mongodb.py` e `metricas_treino.py` JÁ parseiam); o caminho do Golden
Freddy (`_interromper_episodio`) agora rotula `morte_golden`; e o tensorboard ganha
`custom/causas/{rotulo}` (fração na janela de 50) para os 6 rótulos:
`vitoria_gerida, vitoria_apagao, morte_energia, morte_animatronico, menu_crash, morte_golden`.
Também: `custom/win_rate_50` (mudou do antigo schedule para `MetricasPorNoite`) e
`ameaca_esq/dir` no info de cada step.

### 2.7 Feedforward nesta fase (`FNAF_USAR_LSTM=0` no .env)

**Problema:** com RecurrentPPO, o `transferir_pesos` do BC só aproveita o `MultimodalExtractor`
(percepção) — as cabeças de decisão clonadas são jogadas fora (shapes não casam com a LSTM).
Com amostras a 0.7s cada, desperdiçar o artefato mais caro do pacote não faz sentido.

**Solução:** PPO feedforward nesta fase → BC transfere 100% dos tensores (extractor +
mlp_extractor + action_net + value_net; confira o print `[BC warmstart] transferidos N/N`).
O 12º estado (2.3) + ameaças HELD removeram o motivo original da LSTM (Foxy).

**Gatilho OBJETIVO para reabrir o A/B da LSTM:** dominada a Noite 2, se o agente estagnar na
3+ com `morte_animatronico` dominante SEM ameaça registrada no info terminal (= morreu do que
não dá para ver sem memória, ex.: Freddy), teste `FNAF_USAR_LSTM=1` — o extractor do BC ainda
transfere.

### 2.8 `n_steps` 8192 → 4096 (`FNAF_N_STEPS`)

8192 protegia bem uma política inicial RUIM (~11 noites de diversidade por lote), mas custava
1 update de política a cada ~95min de jogo real (~61 updates num run de 500k). Com o BC
warmstart a política já parte decente e a FREQUÊNCIA de updates volta a importar: 4096 ≈ 1
update/~48min, ~122 updates/500k, ~5.4 noites de diversidade. `batch=256`, `n_epochs=4` e
`target_kl=0.03` ficam. Contingência para aprendizado lento: `FNAF_N_EPOCHS=6` (o target_kl
corta épocas excedentes com segurança). Lembre: n_steps/batch definem o rollout buffer — só
valem em treino FRESCO (`--novo`).

---

## 3. Runbook — do zero ao treino

### Etapa A — sem o jogo (já executada na implementação)

```
venv\Scripts\python scripts\inspecionar_vecnormalize.py     # evidência do clip (seção 1)
venv\Scripts\python scripts\smoke_test.py                   # 6/6 OK (espaço (12,), extractor, BC)
venv\Scripts\python -m src.utils.testar_recompensa          # 13/13 OK (Φ em tempo real)
venv\Scripts\python -m src.utils.testar_noite               # 11/11 OK (decidir_reset)
```

### Etapa B — gravação (~2h de jogo SEU)

Para cada noite N de 1 a 5 (navegue no jogo até a noite antes de rodar):

```
venv\Scripts\python -m src.utils.gravar_gameplay --noite N
# F9 quando a noite carregar → jogue ATÉ O 6AM pelas teclas → F10
```

- **Meta:** ≥2 vitórias por noite 1–5 (10 sessões) + 1 sessão extra nas noites 4–5 (mais
  eventos de porta). ~21k frames no total.
- Jogue pelas TECLAS do gravador (a/d portas, q/e luzes, tab câmera, 1-9/0/- câmeras), não
  pelo mouse — senão a ação não é registrada.
- Golden Freddy fechou o jogo? F10 — a sessão parcial é válida.
- **Validação do dataset** (o print do `GameplayDataset` ao carregar): `nada` ≤ ~92%;
  portas ≥ ~150 exemplos somados; noites 1–5 todas presentes; abra 3–4 PNGs e confira que são
  a área do jogo (sem barra de título).

### Etapa C — BC (sem o jogo)

```
venv\Scripts\python main.py bc dados\gameplay_*_noite1\dataset.json dados\gameplay_*_noite2\dataset.json ...
```

Aceite: recall de portas ≥ 60%, câmera ≥ 50%, luzes ≥ 40%, top-1 ≥ 70% (validação).
Sanity ao vivo: com `modelos/fnaf_bc.zip` como zip mais recente, `python main.py jogar` por 3
episódios de Noite 1 → sobrevivência média ≥ 200s (não precisa vencer; precisa não morrer cedo).

### Etapa D — RL do zero

Pré-requisitos já garantidos nesta implementação: modelos antigos movidos para
`modelos/backup_pre_bc/`, `.env` com `FNAF_USAR_LSTM=0` e `FNAF_NOITE_DESEJADA=1`.
**Pare qualquer treino antigo em andamento** (o teste do bundle 8192 perdeu o sentido).

```
venv\Scripts\python main.py treino --novo --bc modelos\fnaf_bc.zip
```

Confira no console: `[BC warmstart] transferidos N/N tensores` (**100%** — menos que isso,
aborte e investigue) e `[hparams] n_steps=4096 ... clip_reward=100.0`.

---

## 4. O que observar (tensorboard) e quando agir

**Marcos esperados:**

| Steps | Esperado |
|---|---|
| ~10k (~2h) | Noite 1 ≥ 20%, sobrevivência ≥ 300s, `custom/entropia` ∈ [0.6, 1.8] |
| ~50k | **Noite 1 ≥ 60%** e promoção para alvo 2 (esperada entre 30–80k) |
| ~150k | Noite 2 ≥ 50%, alvo 3 |
| ~300k | alvo 4–5 em prática |
| 500k | Noite 5 em prática/dominada (6–7: ver seção 5) |

**Saúde contínua:** `custom/entropia` rastreando `custom/h_alvo` (|erro| < 0.3 na maior parte);
`custom/ent_coef` sem saturar no teto por >10 rollouts seguidos; `morte_energia` < 60% em
`custom/causas`; `train/value_loss` estável; `train/approx_kl` ~0.02–0.03.

**Correções (todas por env var, sem código):**

- **BC com recall de portas ~0** mesmo com sampler → grave +3 sessões focadas em situações de
  porta antes de seguir.
- **RL@20k: H < 0.3 com ent_coef saturado no teto** → `FNAF_ENT_MAX=0.04` e/ou `FNAF_H_INICIO=1.6`.
- **RL@50k: Noite 1 < 40%** → diagnostique por `custom/causas`: `morte_energia` dominante →
  `FNAF_H_INICIO=1.2` (menos aleatoriedade); `morte_animatronico` dominante → BC fraco em
  portas (volte à Etapa B/C).
- **`value_loss` explodindo** → `FNAF_CLIP_REWARD=50` + reiniciar fresco (decidir até ~30k).
- **`custom/ent_coef` serrilhado (período 2)** → `FNAF_ENT_GANHO=0.4`.

---

## 5. Pendência declarada: Noites 6 e 7

No FNAF 1 real, vencer a Noite 5 volta ao MENU — a **6th Night** e a **Custom Night (7)** são
botões PRÓPRIOS, não alcançáveis pelo `Continue`. A mecânica de reset atual cobre as noites
1–5. Quando a Noite 5 estiver dominada, a próxima fase é calibrar os cliques extras de menu
(ex.: `FNAF_NIGHT6_CLICK_*`, fluxo da Custom Night 20/20/20/20) — mudança pequena e isolada em
`decidir_reset`/`_preparar_reset` + .env. Este pacote leva o agente até dominar a Noite 5.

## 6. Riscos aceitos e suas redes de segurança

1. **RL erode o BC / BC vicia** → BC é só init (nada fixa os pesos); termostato injeta entropia
   gradualmente (+20%/rollout máx).
2. **Controlador oscilar** → medição suave (média do train inteiro), passo clampado, ganho 0.7;
   knob `FNAF_ENT_GANHO`.
3. **Feedforward sem memória (Freddy)** → gatilho objetivo de reversão à LSTM (seção 2.7).
4. **Currículo promover cedo** → janela cheia de 30 eps da noite ALVO; nunca rebaixa; auditável
   em `custom/curriculo/noite_alvo` e `modelos/curriculo.json`.
5. **Parsers de log** → a linha de episódio só ganhou campos que o `LOG_PATTERN` já previa
   (`Energia fim`, `Causa`); telemetria nova é só tensorboard.
