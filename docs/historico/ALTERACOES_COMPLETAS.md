# Alterações do ambiente — histórico técnico

> 📦 **HISTÓRICO** — registro das correções até junho/2026 (pré-pacote BC). Valores citados
> (recompensas −500/+1000, n_steps=2048/8192 etc.) são da época e foram alterados depois.
> Estado atual: [../PACOTE_BC_ENTROPIA.md](../PACOTE_BC_ENTROPIA.md) · índice: [../README.md](../README.md).

Documento de referência para as mudanças feitas no ambiente de RL do FNAF 1.

---

## Correções de aprendizado — junho/2026

Auditoria completa do pipeline encontrou bugs que impediam aprendizado real.
Como vários deles mudam o que o agente "vê", **recomenda-se começar o treino
do zero** (`python main.py treino --novo`) — modelos antigos aprenderam sobre
entradas diferentes das atuais.

### 1. Dupla normalização da imagem (o agente estava cego)

O SB3 (`extract_features → preprocess_obs`) já divide observações de imagem
uint8 por 255 **antes** de chamar o features extractor. A `MultimodalExtractor`
dividia por 255 de novo, entregando à CNN pixels no intervalo [0, 0.004] —
entrada quase nula. Na prática o agente decidia só pelos 8 estados internos e
ignorava a tela do jogo. Corrigido removendo a divisão no extractor
(validado por `scripts/smoke_test.py`, que verifica o valor que chega na CNN).

### 2. Observação capturava a tela inteira, não a janela do jogo

`_capturar_observacao` usava captura do monitor inteiro (desktop, barra de
tarefas etc.) reduzida a 84×84 — o jogo virava uma fração do frame. A detecção
de morte/vitória e a gravação de gameplay (BC) já usavam a janela. Agora a
observação também captura apenas a janela do jogo, consistente com o dataset
de behavioral cloning.

### 3. Episódio interrompido devolvia observação com 7 estados

`_interromper_episodio` retornava `estados` com shape (7,) num espaço de (8,) —
crash do SB3 no meio do treino sempre que o jogo fechava inesperadamente
(perdendo até 2h de progresso desde o último checkpoint). Corrigido para (8,).

### 4. Energia e relógio interno passaram a usar tempo real

Ver nota na seção "Sistema de energia" — o dreno por `STEP_DELAY` subestimava
o consumo em ~50% em relação ao jogo real.

### 5. Recompensas terminais reescalonadas (−500/+1000 → −100/+500)

Magnitudes terminais extremas dominavam a função de valor. Ver nota em
`docs/REFERENCIA_HIPERPARAMETROS.md` (um nível acima desta pasta).

### 6. Gates de detecção por tempo real + trava de segurança

Os bloqueios de detecção de morte (120 passos) e vitória (30 passos) passaram
a usar tempo real (~40s e ~10s), já que a duração de cada step varia. Episódios
agora também truncam após 700s reais — se o template de vitória falhar, o
episódio não fica preso por horas.

### 7. Média de pesos entre modelos independentes descontinuada

`merge_modelos.py` fazia média de pesos de modelos treinados em PCs diferentes
(inicializações aleatórias diferentes). Os neurônios das redes não se
correspondem entre si, então a média **não combina aprendizado — produz uma
política quebrada, próxima do aleatório**. Cada merge efetivamente zerava o
progresso. O script agora exige `--force` e documenta quando a média é válida
(checkpoints da mesma linhagem). `main.py` não prioriza mais modelos
`*merged*`, e `combinar_bc_com_ppo` foi descontinuado — o modelo BC deve ser
usado diretamente como ponto de partida do PPO.

### 8. Novos utilitários

- `python main.py jogar` — roda o último modelo em modo avaliação
  (determinístico, sem aprender) e imprime resultado por episódio.
- `python main.py treino --novo` — força começar do zero.
- `scripts/smoke_test.py` — valida ambiente/policy/modelo sem o jogo aberto.
- `requirements.txt` regenerado (agora inclui `keyboard`, que faltava).

---

## Limpeza do repositório — junho/2026

Os modelos antigos (treinados com o pipeline quebrado) foram apagados e o
treino recomeçou do zero. `modelos/` e `dados/` agora estão no `.gitignore` —
checkpoints não são código e não devem ser commitados.

### Pendente: reescrever o histórico do git (~2 GB)

O histórico guarda 100+ checkpoints de ~19 MB que já foram removidos do
diretório, mas continuam nos objetos do git. Quando todos os PCs estiverem
com o trabalho commitado/pushado, rodar (em um clone fresco, por segurança):

```bash
pip install git-filter-repo
git filter-repo --path modelos --invert-paths
git remote add origin https://github.com/DiegoHenriqueMelo/no-more-jumpscares.git
git push --force --all origin
git push --force --tags origin
```

Depois disso, **todos os outros clones precisam ser reclonados** (não dá para
fazer pull por cima de histórico reescrito). O repo cai de ~2 GB para poucos MB.

---

## Espaço de observação

O espaço original era um único `Box(84, 84, 1)` — só a imagem. O agente não tinha
acesso a nenhum estado interno do jogo: não sabia se a porta estava fechada, quanto
de energia tinha, se a câmera estava aberta. Ele precisaria aprender tudo isso só
pelos pixels, o que é possível mas lento e frágil.

A mudança foi trocar para `Dict` com dois campos:

- `imagem`: o frame capturado em escala de cinza, redimensionado para 84×84
- `estados`: vetor float32 de **8 dimensões** com os estados internos normalizados

Os 8 estados são, nessa ordem: `porta_esq`, `porta_dir`, `luz_esq`, `luz_dir`,
`camera_aberta`, `camera_ativa / 11.0`, `energia / 100.0`,
`tempo_real_ep / 535.0`.

A 8ª dimensão usa `time.perf_counter() - episode_start_time` (tempo real desde o
início do episódio), não tempo simulado. Isso é importante porque o tempo real por
step (~0.7s com animações) é maior que `STEP_DELAY` (~0.35s), então o tempo
simulado acumularia apenas ~40–50% do progresso real da noite. Com tempo real, o
agente sabe em qual hora do jogo está de fato, o que é informação relevante para
calibrar o comportamento (Freddy quase não se move antes das 3AM; a pressão dos
animatrônicos cresce progressivamente).

`episode_start_time` é definido **após** o último sleep do `reset()`, imediatamente
antes do primeiro step. Isso garante que o timer reflita somente o tempo de
gameplay, sem incluir o overhead de reset (~35s).

A rede processa a imagem por uma CNN de 3 camadas conv e os estados por um MLP
(`Linear(8→32)`), concatena os dois e passa por uma camada final de 256 unidades.

---

## Sistema de energia

O consumo de energia é separado em parcela passiva (fixo) e ativa (proporcional ao
número de itens ligados), com base nas taxas medidas do FNAF1 Night 1:

```
consumo_por_segundo = 0.104 + itens_ativos * 0.100
itens_ativos = min(porta_esq + porta_dir + luz_esq + luz_dir + camera_aberta, 3)
```

O dreno é aplicado por step usando o **tempo real decorrido** desde o step
anterior (`time.perf_counter()`):

```python
delta = agora - self._t_ultima_energia
self.energia -= consumo_por_segundo * delta
```

> **Nota (correção de junho/2026):** a versão anterior usava `STEP_DELAY`
> (0.35s) como proxy de tempo, mas o step real leva ~0.7s com animações.
> O jogo drena energia por segundo de wall-clock, então a simulação
> subestimava o dreno em ~50% — a observação de energia "mentia" para o
> agente (mostrava 60% quando o jogo estava perto de 20%) e a morte por
> blackout era imprevisível para a função de valor. Agora todo o relógio
> interno (`tempo_jogo`, energia, checkpoints de hora, progresso) usa tempo
> real, consistente com a 8ª dimensão da observação. A taxa de
> 0.104 + 0.100 por item foi calibrada contra o jogo real usando o script
> `src/utils/simular_energia.py` e é expressa por segundo real.

Quando `energia <= 0`, o ambiente desliga tudo e aguarda a morte via template
matching — o Freddy demora alguns segundos para aparecer após a energia zerar, e
encerrar o episódio imediatamente corromperia o estado do reset.

---

## Validação de ações por contexto

`_executar_acao` retorna `bool` indicando se a ação teve efeito.

Portas e luzes só funcionam com câmera **fechada** — no jogo real o painel de
controle fica oculto quando a câmera está aberta. Trocar de câmera só funciona
com câmera **aberta**. Ações inválidas retornam `False` e recebem recompensa −0.5.

---

## Luzes — comportamento toggle persistente

As luzes funcionam como toggle: a primeira pressão liga, a segunda desliga.
Uma vez ativas, permanecem ligadas até:

1. O agente pressionar o mesmo botão novamente (desliga)
2. A câmera ser aberta (força off ambas)
3. A energia zerar (força off ambas)

Apenas uma luz pode estar ativa por vez: ligar a esquerda desliga a direita
automaticamente e vice-versa.

Fisicamente, o botão da luz é pressionado e solto a cada step imediatamente
antes da captura de observação. Isso garante que a imagem observe o corredor
iluminado enquanto a luz está ativa, sem manter o mouse clicado entre steps
(o que causaria conflitos com outras ações).

---

## Arrasto do mouse para a câmera

O botão de abrir/fechar a prancheta de câmeras no FNAF só aparece quando o mouse
**se move** em direção à parte inferior da tela — um teleporte direto para as
coordenadas não aciona a animação.

A solução: antes de interagir com o botão, o mouse teleporta para `CAMERA_DRAG_PIXELS`
acima da coordenada do botão e então desliza suavemente até ele usando
`pyautogui.moveTo(x, y, duration=CAMERA_DRAG_DURATION, tween=easeInOutQuad)`.
Dois parâmetros configuráveis no `.env`:

```
FNAF_CAMERA_DRAG_PIXELS=80      # distância de partida acima do botão
FNAF_CAMERA_DRAG_DURATION=0.15  # duração do arrasto em segundos
```

### Bug: mouseDown causava captura pelo handler de pan da câmera

A implementação anterior usava `arrastar_clicando` ao fechar a câmera — que executa
`mouseDown` na posição de partida, move o mouse até o botão e depois `mouseUp`. O
problema: quando a câmera está **aberta**, a UI do jogo registra um `mouseDown`
dentro da área de visualização como início de pan (arrasto do mapa), capturando o
mouse. O hover sobre o botão de toggle nunca chegava a disparar, pois o handler de
pan consumia o evento. O estado interno do agente já havia sido atualizado para
"câmera fechada" antes do gesto, então quando o template matching detectava a câmera
ainda aberta, corrigia de volta — uma sequência de correções a cada episódio.

O comportamento era assimétrico entre máquinas: no notebook (pc2) as coordenadas
fazem o ponto de partida cair ligeiramente fora da área de pan, então o gesto
funcionava. No PC de produção (pc4) as coordenadas caem dentro da área, causando a
captura.

**Correção**: o gesto foi simplificado para hover puro (`arrastar_para` =
`pyautogui.moveTo`), sem nenhum `mouseDown`. O toggle do FNAF dispara em
`mouseenter` (hover), não em clique, portanto pressionar o botão não é necessário.
Sem `mouseDown`, o handler de pan nunca é ativado. O fluxo completo ficou:

1. Mouse teleporta para `(x, y - CAMERA_DRAG_PIXELS)` — acima do botão
2. `time.sleep(CAMERA_EXIT_DELAY)` — aguarda animação de saída da câmera
3. `arrastar_para(x, y)` — hover suave até o botão, dispara toggle
4. `time.sleep(0.08)` — pausa mínima
5. `arrastar_para(x, y - CAMERA_DRAG_PIXELS)` — retorna para posição neutra

O mesmo fluxo se aplica tanto para abrir quanto para fechar.

---

## Sincronização de estado (câmera e portas)

O ambiente mantém estado interno (`camera_aberta`, `porta_esq`, `porta_dir`) que
pode desincronizar do jogo real por atrasos de animação, perda de clique ou
qualquer ruído externo. Três mecanismos passivos corrigem isso sem injetar cliques
fora da política do agente:

### Template matching — câmera

A cada 3 steps (fora do cooldown de abertura), `_camera_aberta_por_template()`
roda `cv2.matchTemplate` contra o indicador "YOU" no mapa de câmeras. Uma única
leitura discordante é descartada; duas leituras consecutivas discordantes corrigem
`self.camera_aberta`. Isso evita falsos positivos durante a animação de transição.

O template de referência (`src/utils/referencias/camera_aberta.png`) é gerado uma
vez pelo script de calibração:

```bash
python -m src.utils.calibrar camera_aberta
```

### Pre-click — portas

Imediatamente antes de togglear uma porta, o ambiente lê o pixel do botão e compara
com o estado interno. Se a cor dominante (verde = fechada, vermelho = aberta)
discordar, o estado interno é corrigido **antes** do toggle. Isso garante que a
ação aplicada (abrir ou fechar) corresponde ao que o agente pretendia.

### Pós-clique — portas

Após clicar em uma porta, `_verificar_botao_porta()` verifica se a cor dominante
do botão **inverteu** (verde→vermelho ou vermelho→verde). Se não tiver invertido
em até 3 tentativas, sincroniza o estado interno pela cor lida — sem reverter às
cegas. A verificação exige inversão de cor dominante (não apenas variação de delta),
o que elimina falsos positivos por ruído de iluminação.

### Sync passivo — luz → porta

Quando o agente pressiona uma luz, o ambiente aproveita a captura de tela já feita
para ler o pixel do botão de **porta do mesmo lado**. Se a cor dominante indicar
estado diferente do interno, corrige silenciosamente. Nenhum clique extra é
emitido.

### Log de desyncs por episódio

Ao fim de cada episódio, `_escrever_log_desyncs()` appenda uma linha em
`logs/desyncs.log`:

```
Ep    1 | steps   584 | desfecho: morte     | SYNC camera:   2 | SYNC porta:   0 | porta falha:   1
```

- **SYNC camera**: quantas vezes o template corrigiu `camera_aberta`
- **SYNC porta**: quantas vezes pre-click ou luz-sync corrigiram uma porta
- **porta falha**: quantas vezes `_verificar_botao_porta` não confirmou o toggle
  após 3 tentativas (indica clique perdido pelo jogo)

---

## Correções de timing

### Energia por STEP_DELAY em vez de wall-clock

A versão anterior calculava `dt = time.perf_counter() - ultimo_update_energia`,
o que causava dois problemas: (1) o primeiro step do episódio calculava um `dt`
de ~35s (todo o sleep do reset), consumindo energia indevidamente; (2) variações
de latência entre steps (animações de porta, drag da câmera) distorciam o modelo
de energia, fazendo-o drenar 2–3× mais rápido que o esperado.

Correção: `_atualizar_energia` usa `STEP_DELAY` como dt fixo. A energia drena
de forma determinística e proporcional ao número de steps, independente do
wall-clock. `ultimo_update_energia` foi removido.

### Timer do episódio

`episode_start_time` era definido antes do último sleep do `reset()` (~20s de
antecedência). Isso inflava `tempo_real` no log e deslocava os checkpoints de hora
em ~20s. Corrigido: o timer é definido **depois** do sleep, imediatamente antes
do primeiro step.

### Guard de detecção de morte e vitória

O guard que ignora detecção nos primeiros N passos (para o jogo terminar de
transicionar da tela de Game Over) estava em 10 passos (~2.5s). Insuficiente —
o FNAF pode demorar mais para limpar a tela. Aumentado para 120 passos (~30s).
Mesmo ajuste aplicado em `_detectar_vitoria`.

---

## Checkpoint de hora

Milestones de sobrevivência são recompensados ao atingir cada hora do jogo (1AM a
6AM). O checkpoint usa **tempo real do episódio** (`time.perf_counter() -
episode_start_time`) em vez de tempo simulado, pois o step real dura ~0.7s enquanto
`STEP_DELAY` é ~0.35s — tempo simulado refletiria apenas metade do progresso real.

O bônus é proporcional à energia disponível vs. a esperada para aquele horário:

```python
ratio = min(self.energia / e_cp, 1.5)
bonus = max(ratio * 50.0, 5.0)  # floor 5, máx 75
```

| Energia no checkpoint | Ratio | Bônus |
|-----------------------|-------|-------|
| 0% | 0.0 | +5 (floor) |
| Metade do esperado | 0.5 | +25 |
| Exatamente o esperado | 1.0 | +50 |
| 1.5× o esperado | 1.5 | +75 (cap) |

O cap em 1.5× garante que mesmo os 6 checkpoints no máximo (6 × 75 = 450) não
superem a penalidade de morte (−500), evitando que o agente aprenda a conservar
energia passivamente ignorando ameaças.

---

## Função de recompensa

### O que mudou e por quê

**Recompensa base de "nada"** era +0.5 por passo. Com um gamma baixo e recompensas
pequenas por passo, a estratégia mais segura era não fazer nada — afinal, qualquer
ação poderia ter penalidade. Zerado para 0.0.

**Bônus fixo por câmera** (+0.4 por qualquer ação de câmera) gerava spam de troca
de câmera sem contexto. Substituído por uma penalidade de **inatividade**:

```python
if self.passos_sem_camera > 20:          # ~5s sem abrir câmera
    excesso = self.passos_sem_camera - 20
    recompensa -= min(excesso * 0.05, 1.0)
```

**Survival bonus por step** garante que sobreviver mais é sempre melhor do que
morrer cedo, mesmo com penalidades acumulando. A recompensa base por passo é +0.5,
crescendo até +1.0 linearmente conforme o progresso da noite:

```python
recompensa = 0.5 + (self.tempo_jogo / 535.0) * 0.5
```

**Penalidades de energia por threshold fixo** (<20%, <40%) foram substituídas por
uma penalidade graduada baseada nos thresholds recomendados do jogo por horário:

| Horário | Energia esperada |
|---------|-----------------|
| 12 AM   | 100%            |
| 1 AM    | 85%             |
| 2 AM    | 60%             |
| 3 AM    | 40%             |
| 4 AM    | 25%             |
| 5 AM    | 15%             |
| 6 AM    | 5%              |

A penalidade é proporcional ao déficit em relação à curva esperada: `deficit * 0.02`
por passo. Isso significa que ficar abaixo do esperado *para o horário atual* custa
mais do que simplesmente ter energia baixa — o agente é recompensado por manter
energia alinhada com a demanda progressiva da noite.

### Tabela final de recompensas

| Evento | Valor |
|--------|-------|
| Morte | −500 |
| Sobreviver (6 AM) | +1000 |
| Ação inválida | −0.5 |
| Sobrevivência por step | +0.5 a +1.0 (cresce com progresso) |
| Checkpoint de hora (1AM–6AM) | +5 a +75 (proporcional à margem de energia) |
| Porta ou luz repetida (2ª vez seguida) | −1.5 |
| Câmera ou toggle repetido (2ª vez seguida) | −1.0 |
| "nada" > 8 steps consecutivos | −0.15 × (contador − 8), máx −2.0 |
| Ambas as portas fechadas | −1.0/passo |
| Uso de luz | −0.2 |
| Câmera inativa >20 passos | −0.05 × excesso, máx −1.0 |
| Energia abaixo do esperado | −0.02 × déficit |
| Limite mínimo por passo | −2.0 |

---

## Penalidade por ação repetida — bugs e correções

### Bug: estrutura de verificação sempre-verdadeira

A penalidade de repetição usava a seguinte lógica:

```python
if nome_acao == self.ultima_acao:
    if nome_acao in ["porta_esquerda", ...]:
        if nome_acao == self.penultima_acao:
            recompensa -= 1.5
    else:
        recompensa -= 1.0
```

O problema: `_executar_acao` define `self.ultima_acao = nome_acao` **antes** de
`_calcular_recompensa` ser chamado. Portanto, `nome_acao == self.ultima_acao` é
**sempre True** — o bloco `else: recompensa -= 1.0` executava em todo step que
usasse câmera ou "nada", independente de repetição.

Na prática, o agente era penalizado −1.0 em toda ação de câmera (`camera_1a`,
`camera_2b`, etc.), tornando o uso de câmera sistematicamente desvantajoso e
forçando o agente a evitá-las.

**Correção**: o outer `if` foi removido e a lógica reestruturada para verificar
`self.penultima_acao` (ação do step anterior) diretamente:

```python
if nome_acao in ["porta_esquerda", "porta_direita", "luz_esquerda", "luz_direita"]:
    if nome_acao == self.penultima_acao:
        recompensa -= 1.5
elif nome_acao == "nada":
    if self.contador_nada > 8:
        recompensa -= min((self.contador_nada - 8) * 0.15, 2.0)
elif nome_acao == self.penultima_acao:
    recompensa -= 1.0
```

### Penalidade de "nada" — threshold em vez de flat

O comportamento anterior penalizava "nada" com −1.0 em qualquer repetição, o que
impedia o agente de descansar para conservar energia. Mas remover a penalidade
completamente permitia um modelo anterior aprender a não fazer nada a noite
inteira e vencer por omissão (a Noite 1 é fácil o suficiente que inação passiva
ainda resulta em vitórias).

A solução foi separar os dois casos:

- **Descanso curto** (até 8 steps consecutivos de "nada" ≈ ~2s): sem penalidade
- **Inação prolongada** (>8 steps): penalidade crescente de 0.15 por step extra,
  cap em −2.0

`contador_nada` incrementa a cada step com "nada" e reseta quando qualquer outra
ação é executada. A penalidade combinada com a inatividade de câmera (`passos_sem_camera`)
torna inviável ignorar o jogo por mais de 40–50 steps.

---

## Parâmetros de treino

### Fator de desconto (gamma)

`gamma` foi aumentado de 0.99 para **0.995**. Com `STEP_DELAY=0.35` e step real
~0.7s, uma noite tem ~700 steps. Com gamma=0.99, o +1000 de sobreviver vale
`0.99^700 ≈ 0.0009` no passo inicial — virtualmente zero. Com 0.995, vale ~0.03,
dando ao agente um sinal real para "enxergar" que sobreviver a noite vale a pena.

### Coeficiente de entropia (ent_coef)

`ent_coef=0.01` foi adicionado. Por padrão o SB3 usa 0.0 (sem regularização de
entropia), o que faz a política convergir para comportamento quase determinístico
rapidamente. Com 568 episódios todos terminando em MORTE e recompensa média plana,
o modelo tinha convergido para um padrão fixo que não explorava ações novas mesmo
após mudanças na função de recompensa.

Com `ent_coef=0.01`, a política mantém distribuição de ações mais diversa durante
o treinamento, o que é necessário para que o agente encontre primeiras vitórias por
exploração e receba o sinal de +1000.

---

## Diagnóstico do treinamento — estado atual

Após ~568 episódios (todos MORTE), o modelo mostrava curva de recompensa plana sem
convergência. Dois problemas foram identificados como causas:

### 1. Reward function com câmeras sistematicamente penalizadas

O bug do outer `if` (descrito acima) fazia cada ação de câmera custar −1.0. Com
recompensa base de +0.5/step, usar câmera resultava em recompensa líquida −0.5 por
step. O agente aprendeu a evitar câmeras quase completamente, o que deixava a
penalidade de `passos_sem_camera` acumular — contradição irresolvível que mantinha
a política num equilíbrio ruim.

### 2. Convergência prematura sem exploração

Sem `ent_coef`, a política converge para um padrão determinístico em poucas
atualizações. Combinado com pesos pré-treinados no reward quebrado, o modelo
reforçava a estratégia errada a cada update sem explorar alternativas.

### Implicações para o treinamento futuro

Dado que os pesos do modelo foram formados com reward inválido, reiniciar o
treinamento do zero (em vez de continuar o modelo existente) oferece convergência
mais rápida. Com a função de recompensa corrigida e `ent_coef=0.01`, espera-se
que primeiras vitórias ocorram dentro de 100–200 episódios por exploração aleatória
— e essas vitórias fornecem o gradiente positivo necessário para o PPO aprender a
direção correta.

---

## Previsão de aprendizado

Com `n_steps=2048` e episódios de ~560–700 steps, cada atualização de política
cobre aproximadamente 3–4 episódios. Os seguintes marcos são esperados com
treinamento a partir do zero, função de recompensa corrigida e `ent_coef=0.01`:

| Timesteps | Atualizações | Comportamento esperado |
|-----------|-------------|------------------------|
| 0–50k | 0–25 | Política aleatória. Recompensa −800 a −600. Primeiras vitórias isoladas possíveis. |
| 50k–200k | 25–100 | Câmera começa a ser priorizada (penalidade de inatividade). Energia dura mais com descanso. Recompensa −600 a −400. |
| 200k–500k | 100–244 | Economia de energia emergindo. Checkpoint 3AM possível. Recompensa −400 a −100. |
| 500k–1M | 244–500 | Estratégia mais consistente. Fechamento de portas aprendido. Vitórias regulares. |
| 1M–3M | 500–1500 | Taxa de vitória crescendo para 10–30%. |
| 3M+ | 1500+ | Taxa de vitória >50% possível se política convergida. |

Esses marcos assumem treinamento em um único PC. Com múltiplos PCs rodando em
paralelo e modelos combinados periodicamente, os timesteps efetivos escalam
linearmente com o número de máquinas.

As estimativas podem variar dependendo da qualidade da captura de imagem
(iluminação, resolução da janela do jogo) e da variância introduzida pelo
comportamento dos animatrônicos.

---

## Arquivos modificados

- `src/environment/fnaf_env.py` — todas as mudanças do ambiente
- `src/agent/multimodal_policy.py` — extrator multimodal: CNN para imagem + MLP para 8 estados
- `src/agent/train.py` — gamma=0.995, ent_coef=0.01, MultiInputPolicy, log de episódio
- `src/utils/capture.py` — método `arrastar_para` (hover puro, sem mouseDown)
- `src/utils/calibrar.py` — comando `camera_aberta` para gerar o template de referência
- `src/utils/simular_energia.py` — utilitário para calibrar e verificar taxas de energia
- `.env` / `.env.example` — variáveis de drag da câmera, coordenadas das ações, identificador de PC
