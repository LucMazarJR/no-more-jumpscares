# Referência de hiperparâmetros e conceitos de treinamento

Este documento descreve os principais parâmetros e conceitos que afetam o
comportamento do treinamento. O objetivo é servir de guia para interpretar
o que os logs estão mostrando e prever o efeito de qualquer ajuste.

> **Para aprender os conceitos do zero** (o que é entropia, epoch, política, recompensa, etc.) veja
> o guia didático **[GUIA_CONCEITOS_E_FUNCIONAMENTO.md](GUIA_CONCEITOS_E_FUNCIONAMENTO.md)**. Este
> aqui é de **consulta rápida** de hiperparâmetros.
>
> **Valores atuais (fonte: código):** `gamma=0.997`, `ent_coef` 0.02→0.005 (via `EntropiaSchedule`,
> gateado em 20% de vitória), `learning_rate` linear 3e-4→3e-5, `n_steps=2048`, `batch_size=64`,
> `n_epochs=10`. Onde uma seção abaixo citar um valor antigo no exemplo, o **valor atual prevalece**.

---

## Coeficiente de entropia (`ent_coef`)

**Valor atual:** `0.02` inicial, decaindo até `0.005` via `EntropiaSchedule` — o decaimento só
começa **depois** que a taxa de vitória (janela de 50 episódios) cruza 20% (o "gate"). Nunca vai a
zero. (O `0.01` fixo abaixo é o valor antigo; a explicação do conceito segue válida.)

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

### Como ajustar

O valor 0.01 é um ponto de partida razoável para este ambiente. As situações em
que faz sentido mexer:

**Aumentar para 0.02–0.05** se:
- Após 200k steps, o agente ainda repete a mesma sequência de ações
- A distribuição de ações nos logs mostra uma ação dominando >70% do tempo
- Os logs mostram SYNC camera ≈ 0 (câmera sendo evitada sistematicamente)

**Reduzir para 0.005** se:
- Após 300k steps, a recompensa melhorou mas está muito ruidosa sem estabilizar
- Vitórias apareceram mas a taxa não está crescendo (agente não consolida)
- O agente alterna entre estratégias boas e ruins de forma aleatória

**Nunca reduzir para 0.0 antes de atingir taxa de vitória estável**, pois a política
pode congelar num padrão subótimo. Após atingir, por exemplo, 30% de vitória de forma
consistente, reduzir gradualmente ajuda a refinar a estratégia sem perder o que foi
aprendido.

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

### Por que 0.995 neste projeto

Uma noite do FNAF tem ~700 steps (com step real de ~0.7s). A recompensa de vitória
(+500) está no step ~700. Com gamma=0.99, essa recompensa vale `0.99^700 ≈ 0.0005`
no step inicial — praticamente zero, o agente não "sente" que sobreviver importa.
Com gamma=0.995, vale `0.995^700 ≈ 0.015` — ainda pequeno, mas suficiente para
criar gradiente em direção à sobrevivência (reforçado pelos bônus de checkpoint
de hora, que chegam bem antes).

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

**Valor atual:** `2048`

### O que é

O PPO coleta exatamente `n_steps` de experiência (pares estado-ação-recompensa)
antes de fazer uma atualização da política. Com episódios de ~560–700 steps,
`n_steps=2048` equivale a ver aproximadamente **3–4 episódios completos** antes
de cada update.

### Efeito nos gradientes

**n_steps baixo (512–1024):** Atualizações frequentes, mas cada uma usa poucos
dados. Alta variância nos gradientes — a política oscila mais. Pode ser útil se
o ambiente muda rapidamente ou se os episódios são curtos.

**n_steps alto (4096–8192):** Cada atualização usa mais experiência diversa,
gradiente mais estável. Mais lento por update, mas potencialmente mais eficiente
em termos de timesteps totais para convergir. Pode ajudar quando episódios têm
muita variância natural (como FNAF, onde os animatrônicos se movem de forma
semi-aleatória).

**Ajuste recomendado se a recompensa estiver muito ruidosa:** aumentar para 4096
para estabilizar os gradientes sem alterar outros parâmetros.

---

## Épocas por atualização (`n_epochs`)

**Valor atual:** `10`

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

Para este projeto, 10 é padrão e não precisa ser alterado a menos que o tensorboard
mostre `clip_fraction > 0.3` sistematicamente.

---

## Learning rate (`learning_rate`)

**Valor atual:** `3e-4` (0.0003)

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

O valor `3e-4` é o padrão recomendado pelo SB3 para PPO e funciona bem na maioria
dos casos. Uma alternativa caso o aprendizado esteja instável é usar um **learning
rate schedule** que reduz gradualmente (`linear` ou `cosine_decay`), disponível
via `learning_rate=lambda progress: 3e-4 * (1 - progress)` no SB3.

---

## Reward shaping — princípios e riscos

### O que é

O ambiente real tem recompensa esparsa: −100 por morte, +500 por vitória. Com
centenas de steps por episódio, o sinal chega tarde demais para o agente associar
ações específicas ao resultado. **Reward shaping** é adicionar recompensas
intermediárias densas que guiam o aprendizado.

> **Nota (junho/2026):** os valores terminais eram −500/+1000 e foram reduzidos
> para −100/+500. Magnitudes terminais muito maiores que a recompensa por step
> (~0.5) dominam a função de valor: o crítico precisa prever alvos com variância
> enorme concentrada em um único step, o que torna o aprendizado do valor lento e
> ruidoso. O retorno do episódio continua monotônico no tempo de sobrevivência
> (morrer mais tarde sempre rende mais que morrer cedo).

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

Para este projeto: episódio ~700 steps, horizonte efetivo com γ=0.995 é ~200 steps.
O agente "enxerga" apenas os próximos ~2–3 minutos de jogo por vez. Isso é suficiente
para aprender a gerenciar energia e câmeras, mas significa que ações tomadas nos
primeiros steps da noite têm menos peso na estimativa de valor do que idealmente
teriam.

### Curriculum learning

Técnica de treinar primeiro em versões mais fáceis do problema e gradualmente
aumentar a dificuldade. Para este projeto, uma forma de aplicar seria limitar o
episódio a sobreviver até 2AM inicialmente (truncar após ~178s de jogo), depois
expandir para 4AM, depois para 6AM. O agente aprende sub-tarefas menores antes
de lidar com a noite completa.

Ainda não implementado, mas seria útil se o treinamento continuar estagnado após
atingir 200k timesteps sem vitórias com a configuração atual.

### Observação multimodal (Dict space)

O SB3 com `MultiInputPolicy` processa espaços de observação `Dict` passando cada
chave por seu próprio extrator. Neste projeto, a chave `"imagem"` passa pela CNN e
`"estados"` passa pelo MLP. A concatenação dos dois é o que o ator e o crítico
recebem como entrada.

Isso é relevante porque **mudanças na dimensão de qualquer campo invalidam o modelo
salvo**. Se o número de estados mudar de 8 para 9, o `Linear(8, 32)` do MLP não
carrega os pesos antigos. Qualquer modificação no `observation_space` exige reinício
do treinamento do zero.

---

## Decisão 6 — bundle de schedules: o que mudou junto e como isolar

**Aplicado (junho/2026).** Os 3 botões mudaram **de uma vez** (bundle pragmático — amostra é o
recurso escasso, 3-4 runs isolados sai caro). Por isso este guia: se o bundle **piorar**, reverter
**um botão por vez** ao valor de controle e re-treinar uma janela curta para achar o culpado.

**O que está no bundle:**
- `gamma`: 0.995 → **0.997** (horizonte ~200 → ~333 steps; a vitória propaga melhor pro início).
  Em `fnaf_env.GAMMA` (fonte única: PPO + VecNormalize + shaping Φ). **Só em treino fresco.**
- `learning_rate`: 3e-4 fixo → **`linear(3e-4, 3e-5)`** — decai ao longo do treino; **piso 3e-5**
  (não 0) para não congelar a retomada (o `progress_remaining` reinicia a cada `learn()`).
- `ent_coef`: 0.01 → **0.02 inicial, decai até 0.005** via callback `EntropiaSchedule`, **gateado**
  pela taxa de vitória (≥20% numa janela de 50 episódios) — só consolida (decai) **depois de
  começar a vencer**; nunca vai a 0.

**Mapa sintoma → botão culpado** (reusa os sintomas das seções acima):

| sintoma nos logs (`treino.log` / tensorboard) | suspeito | reverter para |
|---|---|---|
| crítico instável, loss diverge, vitória **aparece e some** bruscamente | gamma alto **ou** LR alto cedo | gamma 0.995 / LR fixo 3e-4 |
| caótico, nunca converge, ação ~aleatória mesmo após muito treino | ent_coef alto | ent_coef 0.01 fixo |
| **congela cedo**, repete a mesma ação, exploração some | ent_coef decaiu cedo (gate baixo) **ou** LR caiu rápido | subir o `gate` do `EntropiaSchedule` / piso do LR |
| aprende lento, não "esquece" comportamento ruim | piso de LR baixo demais | subir o piso (ex.: 1e-4) |

**Lembretes:**
- gamma ≥ 0.999 → horizonte > episódio → crítico instável; 0.997 está na faixa segura (0.993–0.997).
- O `EntropiaSchedule` **não decai antes do gate** — se o agente nunca vence, `ent_coef` fica em 0.02
  o treino todo (explora, não congela). Isso é proposital.
- **Medir sempre por taxa de vitória / tempo de sobrevivência**, nunca pela recompensa (muda entre
  versões). Salvar o controle (`modelos/*.zip` + `vecnormalize.pkl`) **antes** de treinar.

---

## Decisão 7 — RecurrentPPO (LSTM): por que e como diagnosticar

**Por que LSTM (não frame-stacking):** Foxy/Freddy **não são detectáveis por frame** (Foxy é um
*buildup* de minutos; Freddy só aparece nas câmeras). Lidar com eles exige memória de **longo
alcance** — frame-stacking (poucos frames) não cobre. A noite entra no estado p/ condicionar a
agressividade (a LSTM aprende "noite 4 = Foxy mais rápido"). Ligar com `FNAF_USAR_LSTM=1`.

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
