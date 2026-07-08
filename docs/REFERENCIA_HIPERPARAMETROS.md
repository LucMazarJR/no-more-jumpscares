# Referência de hiperparâmetros e conceitos de treinamento

Este documento descreve os principais parâmetros e conceitos que afetam o
comportamento do treinamento. O objetivo é servir de guia para interpretar
o que os logs estão mostrando e prever o efeito de qualquer ajuste.

> **Para aprender os conceitos do zero** (o que é entropia, epoch, política, recompensa, etc.) veja
> o guia didático **[GUIA_CONCEITOS_E_FUNCIONAMENTO.md](GUIA_CONCEITOS_E_FUNCIONAMENTO.md)**. Este
> aqui é de **consulta rápida** de hiperparâmetros.
>
> **Valores atuais (fonte: código — pacote BC+entropia de julho/2026):** `gamma=0.997`,
> `n_steps=4096`, `batch_size=256`, `n_epochs=4`, `target_kl=0.03`, `clip_reward=100`,
> `learning_rate` linear 3e-4→3e-5, e `ent_coef` **adaptativo** via `ControladorEntropia`
> (termostato: alvo de entropia 1.1→0.7 nats — calibrado p/ warmstart BC, run 2 jul/2026 —,
> coeficiente em [0.003, 0.012], inicial 0.02).
> Env vars: `FNAF_N_STEPS`, `FNAF_BATCH_SIZE`, `FNAF_N_EPOCHS`, `FNAF_TARGET_KL`,
> `FNAF_CLIP_REWARD`, `FNAF_ENT_INICIO`, `FNAF_H_INICIO/H_FIM`, `FNAF_ENT_MIN/MAX/GANHO/PASSO_MAX`,
> `FNAF_CURRICULO_LIMIAR`, `FNAF_PESO_ENERGIA` (potencial da reserva de energia no Φ, default
> 4.0 — adicionado após a 1ª run do pacote diagnosticar morte por apagão sistemático),
> `FNAF_WARMUP_FRAC/WARMUP_CLIP` (warmup do crítico p/ treino fresco
> com `--bc`: o BC clona só o ator; o crítico chega aleatório e aprende V com o ator quase
> parado — clip_range 0.03 e ent_coef em ENT_MIN — nos primeiros ~8% do treino).
> Detalhes e porquês: [PACOTE_BC_ENTROPIA.md](PACOTE_BC_ENTROPIA.md).
> Onde uma seção abaixo citar um valor antigo no exemplo, o **valor atual prevalece**.

---

## Coeficiente de entropia (`ent_coef`)

**Valor atual:** ADAPTATIVO. O `ControladorEntropia` (termostato) mede a entropia H real da
política a cada rollout e ajusta o `ent_coef` para ela seguir um alvo que decai devagar
(1.1 → 0.7 nats ao longo do treino — ver "regra do warmstart" abaixo), com coeficiente limitado
a `[0.003, 0.012]` e valor inicial `0.02`. Substituiu o antigo `EntropiaSchedule` (gate por win_rate): não existia valor FIXO certo —
0.03 deixava a política aleatória demais p/ conservar energia e 0.02 colapsava; e o gate nunca
abria quando o agente estagnava abaixo dele. (Os exemplos abaixo citam valores fixos antigos; a
explicação do conceito segue válida.)

### O que é

Em RL com política probabilística, a cada step o agente sorteia uma ação a partir
de uma distribuição. A **entropia** dessa distribuição mede o quão "espalhada" ela
está: se o agente escolhe a ação A com 99% de probabilidade, a entropia é quase
zero — comportamento quase determinístico. Se escolhe entre 17 ações de forma
aproximadamente uniforme, a entropia é máxima.

O `ent_coef` é um multiplicador que adiciona a entropia da política diretamente
na função de perda que o PPO minimiza:

```
perda_total = perda_política − ent_coef × H(π)
```

Como a perda é *minimizada*, subtrair a entropia significa que o otimizador é
incentivado a *maximizá-la* — ou seja, a manter a política mais dispersa.

### Por que isso importa

Sem regularização de entropia (`ent_coef=0.0`), o PPO converge rapidamente para
uma política determinística. Nas primeiras atualizações, assim que encontra ações
que reduzem a perda, concentra a probabilidade nelas e para de explorar as demais.
Em ambientes com sinal de recompensa fraco ou reward function em evolução (como
ocorreu neste projeto), isso resulta em fixar uma estratégia ruim e nunca sair dela.

Com `ent_coef=0.01`, o otimizador ainda aprende, mas é penalizado quando a política
fica muito determinística. O agente continua explorando ações menos óbvias, o que
aumenta a chance de encontrar comportamentos novos — como usar câmeras regularmente
ou descansar para conservar energia.

### Efeito observável nos logs

| ent_coef muito baixo (≈0.0) | ent_coef adequado (0.005–0.02) | ent_coef muito alto (≥0.1) |
|---|---|---|
| Recompensa plana por centenas de eps | Recompensa com variância moderada e tendência de melhora | Recompensa muito ruidosa, sem tendência clara |
| Agente sempre faz as mesmas ações | Agente experimenta ações diferentes | Agente parece aleatório mesmo após muito treino |
| Mesma sequência de ações por episódio | Estratégia muda gradualmente | Sem convergência visível |
| SYNC camera próximo de zero (evita câmeras) | SYNC camera variável, uso de câmera crescente | Comportamento caótico |

### Como ajustar (no controlador, mexa no ALVO, não no coeficiente)

Com o termostato, o `ent_coef` se ajusta sozinho — os botões são os **alvos e limites**.

**Regra do warmstart (jul/2026, run 2):** com `--bc`, o alvo inicial deve CASAR com a entropia
que o clone entrega (H≈1.1), nunca forçá-la p/ cima — o alvo antigo (1.5, pensado p/ política
aleatória de treino do zero) fez o controlador passar 80k steps inflando o `ent_coef`
(0.003→0.020) e derreteu o clone até H 1.44 (morte por animatrônico com 42% de bateria
sobrando = defesa ruidosa). Defaults atuais: `H_INICIO=1.1`, `H_FIM=0.7`, `ENT_MAX=0.012`.
Treino do zero SEM `--bc` é o único caso p/ subir de volta (H_INICIO~1.5, ENT_MAX~0.03).

**Política aleatória demais** (morre por apagão, `morte_energia` dominante em `custom/causas`):
- baixar `FNAF_H_INICIO` (ex.: 1.1 → 0.9) — menos "ações efetivas" no começo.

**Política colapsando mesmo com o controlador** (`custom/entropia` < 0.3 com `custom/ent_coef`
saturado no teto por >10 rollouts):
- subir `FNAF_ENT_MAX` (0.012 → 0.03) e/ou `FNAF_H_INICIO` (+0.2).

**`custom/ent_coef` serrilhando (oscilação com período 2):**
- reduzir `FNAF_ENT_GANHO` (0.7 → 0.4).

**Nunca deixar o alvo chegar a 0** (`FNAF_H_FIM` ≥ ~0.5): política totalmente determinística
congela em padrão subótimo. O que observar no tensorboard: `custom/entropia` deve rastrear
`custom/h_alvo` com |erro| < 0.3 na maior parte do tempo.

---

## Fator de desconto (`gamma`)

**Valor atual:** `0.997` (era `0.995` — atualizado na Decisão 6; horizonte efetivo ~333 steps).

### O que é

Quando o PPO calcula o valor de estar num estado, soma as recompensas futuras com
desconto exponencial:

```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
```

O **horizonte efetivo** — número de steps futuros que o agente "enxerga" de forma
significativa — é aproximadamente `1 / (1 − γ)`:

| gamma | Horizonte efetivo |
|-------|------------------|
| 0.90  | ~10 steps        |
| 0.99  | ~100 steps       |
| 0.995 | ~200 steps       |
| 0.999 | ~1000 steps      |

### Por que 0.997 neste projeto

Uma noite do FNAF tem ~700 steps (com step real de ~0.7s). A recompensa de vitória
(+500) está no step ~700. Com gamma=0.99, essa recompensa vale `0.99^700 ≈ 0.0005`
no step inicial — praticamente zero, o agente não "sente" que sobreviver importa.
Com gamma=0.997, vale `0.997^700 ≈ 0.12` — pequeno, mas suficiente para criar
gradiente em direção à sobrevivência (reforçado pelos bônus de marco de hora,
que chegam bem antes).

### Efeito observável

**Gamma muito baixo (≤0.99):** O agente age como se só importassem os próximos
~100 steps. Aprende a evitar mortes imediatas, mas não desenvolve planejamento de
energia para a segunda metade da noite. Os checkpoints de hora têm pouco peso nas
decisões iniciais.

**Gamma muito alto (≥0.999):** O horizonte se estende além do episódio. O otimizador
fica numericamente instável porque pequenas variações na política causam grandes
mudanças na estimativa de valor. Treino fica mais lento e ruidoso.

**Regra prática:** `gamma` deve ser ajustado de forma que o horizonte efetivo seja
compatível com o comprimento do episódio. Para episódios de ~700 steps, valores
entre 0.993 e 0.997 são razoáveis.

---

## Passos por atualização (`n_steps`)

**Valor atual:** `4096` (histórico: 2048 → 8192 no bundle anti-colapso → **4096** no pacote BC)

### O que é

O PPO coleta exatamente `n_steps` de experiência (pares estado-ação-recompensa)
antes de fazer uma atualização da política. Com episódios de ~560–700 steps,
`n_steps=4096` (atual) equivale a ver aproximadamente **~5-6 episódios completos** antes
de cada update — 1 update a cada ~48 min de jogo real.

**Por que desceu de 8192:** o rollout gigante protegia bem uma política inicial RUIM, mas
custava 1 update a cada ~95 min. Com o BC warmstart a política já parte decente e a
FREQUÊNCIA de updates volta a importar — 4096 dobra os updates mantendo diversidade razoável.

### Efeito nos gradientes

**n_steps baixo (512–1024):** Atualizações frequentes, mas cada uma usa poucos
dados. Alta variância nos gradientes — a política oscila mais. Pode ser útil se
o ambiente muda rapidamente ou se os episódios são curtos.

**n_steps alto (4096–8192):** Cada atualização usa mais experiência diversa,
gradiente mais estável. Mais lento por update, mas potencialmente mais eficiente
em termos de timesteps totais para convergir. Pode ajudar quando episódios têm
muita variância natural (como FNAF, onde os animatrônicos se movem de forma
semi-aleatória).

**Ajuste recomendado se a recompensa estiver muito ruidosa:** subir para 8192
(`FNAF_N_STEPS`) para estabilizar os gradientes — pagando metade da frequência de updates.
Lembre: n_steps define o rollout buffer, só vale em treino FRESCO (`--novo`).

---

## Épocas por atualização (`n_epochs`)

**Valor atual:** `4` (era `10` — bundle anti-colapso)

### O que é

Após coletar `n_steps` experiências, o PPO reutiliza esses dados para fazer
`n_epochs` passes pelo batch antes de descartar e coletar novos dados. O clip
do PPO (`clip_range=0.2` por padrão) limita o quanto a política pode mudar em
cada pass, prevenindo atualizações destrutivas.

**n_epochs baixo (3–5):** Usa menos o batch coletado. Seguro mas menos eficiente
em termos de dados.

**n_epochs alto (15–20):** Extrai mais informação de cada batch, mas aumenta
o risco de over-fitting ao batch atual — a política muda demais em relação ao que
foi coletado, e o clip começa a rejeitar uma fração grande dos updates (visível
no log de `approx_kl` ou `clip_fraction` do tensorboard).

Para este projeto, o valor foi **reduzido de 10 para 4** (bundle anti-colapso): com lotes pequenos
e pouco diversos, reusar 10× super-otimizava a política e colapsava a entropia cedo. Combinado com
`n_steps` maior e `target_kl`, o decaimento da entropia fica mais lento e saudável. Monitore
`clip_fraction` e `approx_kl` no tensorboard ao ajustar.

---

## Learning rate (`learning_rate`)

**Valor atual:** schedule linear `3e-4 → 3e-5` (decai ao longo do treino; piso ≠ 0 para
não congelar retomadas)

### O que é

Controla o tamanho de cada passo do otimizador Adam ao atualizar os pesos da rede.
É um dos parâmetros mais sensíveis: muito alto causa instabilidade, muito baixo
causa aprendizado lento.

**Sintomas de learning rate muito alto:**
- Recompensa melhora rapidamente no início e depois colapsa
- Loss diverge (valores de loss muito altos no tensorboard)
- Taxa de vitória aparece e desaparece de forma brusca

**Sintomas de learning rate muito baixo:**
- Recompensa melhora muito lentamente mesmo com muitas atualizações
- O modelo não está "esquecendo" comportamentos ruins

O `3e-4` inicial é o padrão recomendado pelo SB3 para PPO. O projeto JÁ usa o schedule
linear com piso (`linear(3e-4, 3e-5)` em `train.py`): começa no padrão e decai conforme o
progresso do treino — refina no fim sem congelar retomadas.

---

## Reward shaping — princípios e riscos

### O que é

O ambiente real tem recompensa esparsa: −100 por morte, +500 por vitória. Com
centenas de steps por episódio, o sinal chega tarde demais para o agente associar
ações específicas ao resultado. **Reward shaping** é adicionar recompensas
intermediárias densas que guiam o aprendizado.

> **Nota (junho/2026):** os valores terminais eram −500/+1000 e foram reduzidos
> para −100/+500. Magnitudes terminais muito maiores que a recompensa por step
> (hoje ~0.08 — orçamento denso de ~60 por noite + marcos de 3.0/hora) dominam a
> função de valor: o crítico precisa prever alvos com variância enorme concentrada
> em um único step. O retorno do episódio continua monotônico no tempo de
> sobrevivência (morrer mais tarde sempre rende mais que morrer cedo).
> **Complemento (julho/2026):** o `clip_reward=100` do VecNormalize garante que os
> terminais normalizados passem inteiros (o default 10 clipava vitória e morte no
> MESMO teto — razão 5:1 virava 1:1 no gradiente; medido em
> `scripts/inspecionar_vecnormalize.py`).

### Riscos

**Reward hacking:** O agente encontra formas de maximizar a recompensa formatada
que não correspondem ao comportamento desejado. Exemplo neste projeto: a penalidade
por "nada" repetido foi adicionada para evitar inação, mas um modelo anterior
aprendeu a fazer sempre a mesma ação (repetição de porta) para fugir da penalidade
de "nada" — comportamento igualmente inútil.

**Incentivos contraditórios:** Uma penalidade mal calibrada pode criar um dilema
sem saída. Exemplo neste projeto: com câmeras penalizadas em todo uso (pelo bug do
outer `if`), o agente recebia −1.0 por abrir câmera mas também acumulava penalidade
de `passos_sem_camera` por não abrir. Nenhuma estratégia era boa, a política ficou
num equilíbrio ruim.

### Como identificar problemas

- Se o agente tiver recompensa média estável mas nunca vencer: está maximizando
  os bônus intermediários sem alcançar o objetivo real
- Se a recompensa for muito negativa mas o agente sobreviver razoavelmente: alguma
  penalidade está sendo aplicada com frequência incorreta — verificar logs por step
  (`python main.py treino --steps`) para ver qual componente domina

---

## Convergência prematura e ótimos locais

### O que é

Em RL, o agente pode encontrar um comportamento que não é o melhor possível, mas
que é localmente estável — qualquer desvio pequeno da política atual parece pior.
Isso é chamado de **ótimo local**. A política para de melhorar não porque aprendeu
o comportamento correto, mas porque não tem incentivo para explorar saídas do
padrão atual.

### Como reconhecer nos logs

- Recompensa média se estabiliza por 100+ episódios sem melhora
- Desvio padrão da recompensa diminui (menos variância = política mais determinística)
- Steps por episódio param de crescer (o agente sempre morre no mesmo ponto)
- SYNC camera = 0 por vários episódios (câmera nunca usada)

### Como sair

1. **Aumentar `ent_coef`** temporariamente (ex: 0.05) força exploração
2. **Reiniciar do zero** com reward function corrigida elimina o viés nos pesos
3. **Curriculum learning** (ver abaixo) oferece sub-objetivos mais fáceis para
   guiar a política para fora do ótimo local

---

## Conceitos adicionais

### Horizonte efetivo vs. comprimento do episódio

Se o horizonte efetivo (`1/(1-γ)`) for muito menor que o comprimento do episódio,
o agente age de forma míope — ignora o que acontece na segunda metade da noite.
Se for muito maior, o agente tenta otimizar além do episódio, o que é matematicamente
inconsistente e desestabiliza o treinamento.

Para este projeto: episódio ~700 steps, horizonte efetivo com γ=0.997 é ~333 steps.
O agente "enxerga" os próximos ~4 minutos de jogo por vez. Isso é suficiente
para aprender a gerenciar energia e câmeras, mas significa que ações tomadas nos
primeiros steps da noite têm menos peso na estimativa de valor do que idealmente
teriam.

### Curriculum learning

Técnica de treinar primeiro em versões mais fáceis do problema e gradualmente
aumentar a dificuldade. **Implementado (julho/2026)** via `CurriculumCallback`: a
noite-alvo do modo `continue` é promovida automaticamente quando a janela de 30
episódios da noite alvo atinge `FNAF_CURRICULO_LIMIAR` (50%), persistindo em
`modelos/curriculo.json` (retomadas mantêm o alvo). As noites anteriores continuam
aparecendo no caminho (mortes acima do alvo reescalam do 1) — mistura natural contra
esquecimento. Ver `docs/MONITORAMENTO_TREINO.md` §4.

### Observação multimodal (Dict space)

O SB3 com `MultiInputPolicy` processa espaços de observação `Dict` passando cada
chave por seu próprio extrator. Neste projeto, a chave `"imagem"` passa pela CNN e
`"estados"` passa pelo MLP. A concatenação dos dois é o que o ator e o crítico
recebem como entrada.

Isso é relevante porque **mudanças na dimensão de qualquer campo invalidam o modelo
salvo**. Hoje são **12 estados** (o 12º, `tempo_sem_camera`, entrou no pacote de julho/2026);
se virarem 13, o `Linear` de entrada do MLP não carrega os pesos antigos (o extractor já
deriva a dimensão do espaço, mas os PESOS não migram). Qualquer modificação no
`observation_space` exige reinício do treinamento do zero.

---

## Schedules em uso (gamma, learning rate, entropia): o que mudou junto e como isolar

**Histórico:** a Decisão 6 (junho/2026) introduziu o bundle gamma 0.997 + LR linear +
`EntropiaSchedule` gateado por win_rate. Em julho/2026 o pacote BC substituiu o schedule de
entropia pelo `ControladorEntropia` (termostato — ver seção `ent_coef` acima e
[PACOTE_BC_ENTROPIA.md](PACOTE_BC_ENTROPIA.md)). O que vale HOJE:

- `gamma`: **0.997** (horizonte ~333 steps; a vitória propaga melhor pro início).
  Em `fnaf_env.GAMMA` (fonte única: PPO + VecNormalize + shaping Φ). **Só em treino fresco.**
- `learning_rate`: **`linear(3e-4, 3e-5)`** — decai ao longo do treino; **piso 3e-5**
  (não 0) para não congelar a retomada (o `progress_remaining` reinicia a cada `learn()`).
- `ent_coef`: **adaptativo** via `ControladorEntropia` (alvo H 1.1→0.7 nats — calibrado p/
  warmstart BC desde jul/2026, run 2 —, coef em [0.003, 0.012], inicial 0.02).

**Mapa sintoma → botão culpado** (reusa os sintomas das seções acima):

| sintoma nos logs (tensorboard / `logs/analise/`) | suspeito | o que fazer |
|---|---|---|
| crítico instável, loss diverge, vitória **aparece e some** bruscamente | gamma alto **ou** LR alto cedo **ou** clip_reward alto | gamma 0.995 / LR fixo 3e-4 / `FNAF_CLIP_REWARD=50` |
| caótico, nunca converge, ação ~aleatória mesmo após muito treino; **morte com energia sobrando** | alvo de H alto demais (derretendo o clone BC) | baixar `FNAF_H_INICIO` (default já é 1.1 p/ warmstart) |
| **congela cedo**, repete a mesma ação, exploração some | teto do ent_coef baixo p/ a pressão de colapso **ou** LR caiu rápido | `FNAF_ENT_MAX=0.03` + `FNAF_H_INICIO`+0.2 / subir piso do LR |
| aprende lento, não "esquece" comportamento ruim | piso de LR baixo demais | subir o piso (ex.: 1e-4) |

**Lembretes:**
- gamma ≥ 0.999 → horizonte > episódio → crítico instável; 0.997 está na faixa segura (0.993–0.997).
- O termostato NUNCA leva a entropia a zero (`FNAF_H_FIM=0.7` e piso `FNAF_ENT_MIN=0.003`) —
  consolida sem congelar. Isso é proposital.
- **Medir sempre por taxa de vitória / tempo de sobrevivência**, nunca pela recompensa (muda entre
  versões). Salvar o controle (`modelos/*.zip` + `vecnormalize.pkl`) **antes** de treinar.

---

## RecurrentPPO (LSTM): por que, QUANDO, e como diagnosticar

> **Fase atual: DESLIGADA (`FNAF_USAR_LSTM=0`).** O 12º estado (`tempo_sem_camera`) tornou o
> risco do Foxy observável e as ameaças já são HELD (o detector guarda memória) — o motivo
> original da LSTM sumiu, e o warmstart de BC só transfere 100% dos pesos no PPO feedforward.
> **Gatilho para reabrir o A/B:** dominada a Noite 2, estagnar na 3+ com `morte_animatronico`
> dominante SEM ameaça registrada no info terminal (= morrer do que não se vê sem memória,
> ex.: Freddy). Ver [PACOTE_BC_ENTROPIA.md](PACOTE_BC_ENTROPIA.md) §2.7.

**Por que LSTM (não frame-stacking):** Freddy **não é detectável por frame** (só aparece nas
câmeras; seu avanço é um processo de minutos). Lidar com ele exigiria memória de **longo
alcance** — frame-stacking (poucos frames) não cobre. A noite entra no estado p/ condicionar a
agressividade (a LSTM aprenderia "noite 4 = mais rápido"). Ligar com `FNAF_USAR_LSTM=1`.

**Começa pequena:** `lstm_hidden_size=128`, `n_lstm_layers=1`. Memória maior = mais parâmetros =
mais amostra; só cresça se ajudar. **A/B contra o controle** (feedforward, `FNAF_USAR_LSTM=0`),
mudando SÓ o algoritmo, com **critério de desistência** (se em ~100k steps não empatar a
sobrevivência do controle → reverter).

**Mapa sintoma → o que fazer:**

| sintoma | causa provável | o que fazer |
|---|---|---|
| `approx_kl` explode / `clip_fraction` alto / loss diverge (tensorboard) | sequência muito correlacionada / LR alto p/ recorrência | baixar LR, reduzir `n_epochs`, `lstm_hidden_size` menor |
| `value_loss` diverge | crítico recorrente instável | `enable_critic_lstm` menor impacto: testar crítico não-recorrente; reduzir LSTM |
| `sonda_memoria` diz **INERTE** (ação não muda com o histórico) | a recorrência não está sendo usada (vira feedforward caro) | conferir o masking (`testar_masking`); mais amostra; rever se a tarefa exige memória |
| `testar_masking` FALHA (estado vaza) | `episode_starts` não propagado | bug no caminho de treino/avaliação — corrigir antes de qualquer run longo |
| LSTM **não empata** o controle no orçamento | recorrência custando mais amostra do que rende | desistir (reverter pro feedforward) — frame-stacking não resolveria Foxy/Freddy de qualquer forma |

**Avaliação (`jogar`) precisa propagar o estado:** `FNAF_USAR_LSTM=1` faz o `modo_jogar` carregar
RecurrentPPO e propagar `lstm_states`/`episode_starts` (resetando no início de cada episódio). Sem
isso a avaliação mente.
