# Além do aprendizado por reforço

> 📦 **HISTÓRICO** — panorama especulativo de maio/2026 (DAgger, GAIL, DreamerV3, etc.).
> Do que está aqui, o Behavioral Cloning foi implementado como warmstart (pacote de julho/2026).
> Estado atual: [../PACOTE_BC_ENTROPIA.md](../PACOTE_BC_ENTROPIA.md) · índice: [../README.md](../README.md).

Este documento descreve abordagens alternativas ao PPO puro para treinar uma IA a
jogar FNAF1. O objetivo não é substituir o RL, mas entender quais outros paradigmas
existem, o que cada um resolve melhor, e como poderiam ser combinados com o projeto
atual.

---

## O problema fundamental do RL puro em FNAF

Antes de apresentar alternativas, vale entender por que o RL puro é lento aqui:

1. **Recompensa esparsa e atrasada**: a grande recompensa (+1000) só chega depois
   de ~535 segundos de jogo. O agente precisa conectar centenas de decisões
   intermediárias com um resultado que chegou muito depois.

2. **Observabilidade parcial**: os animatronicos se movem fora da câmera. Um agente
   sem memória temporal trata cada frame como se fosse o início da partida.

3. **Velocidade de coleta**: cada step dura ~0.7s em tempo real. Não é possível
   acelerar além do tempo real do jogo sem um simulador.

4. **Exploração ineficiente**: o espaço de estratégias possíveis é enorme. O agente
   descobre por acidente que fechar a porta salva a vida — não por inferência.

Cada abordagem abaixo ataca um ou mais desses problemas de forma diferente.

---

## Imitation Learning — Aprendizado por Imitação

### Motivação

Um humano experiente em FNAF1 já internalizou uma estratégia: checar câmeras
regularmente, fechar a porta quando ouvir barulho, economizar energia. Em vez de o
agente descobrir isso por tentativa e erro, a ideia é **ensinar primeiro**.

Imitation Learning é a classe de métodos que usa demonstrações humanas (pares
observação → ação) para inicializar ou guiar o aprendizado.

---

### Behavioral Cloning (BC)

O caso mais simples: gravamos gameplay humano e treinamos a rede para reproduzir
as ações humanas via supervised learning.

```
Dataset: [(obs_1, ação_1), (obs_2, ação_2), ..., (obs_N, ação_N)]
Objetivo: minimizar a diferença entre ação prevista e ação humana
```

**Vantagem:** rápido de implementar, não precisa rodar o jogo durante o treino.
O módulo `src/agent/behavioral_cloning.py` já existe no projeto.

**Limitação central — distribuição shift:** o agente aprende a imitar o humano
nos estados que o *humano* visita. Quando ele começa a errar e entra em estados
nunca vistos pelo humano, não sabe o que fazer — a política entra em colapso.

---

### DAgger (Dataset Aggregation)

DAgger resolve o problema do BC iterativamente. O processo é:

```
1. Coleta demonstração humana inicial (2-3h de gameplay)
2. Treina BC no dataset
3. Deixa o agente jogar: onde ele desviou do humano?
4. Pede ao humano para rotular esses novos estados
5. Adiciona ao dataset e re-treina
6. Repete até convergir
```

A cada iteração, o dataset cresce com estados que o próprio agente (não o humano)
gera. Isso cobre progressivamente os estados "fora da distribuição" que o BC puro
nunca vê.

**Para FNAF:** o humano rota de demonstração seria jogar com o gameplay gravado
automaticamente. Na fase 2, o agente joga sozinho e situações onde ele toma
decisões ruins (ex: porta aberta com animatronico na entrada) são marcadas para
re-anotação humana.

**Complexidade:** Baixa-Média. A biblioteca `imitation` para Python implementa
DAgger e integra com SB3 e Gymnasium.

---

### GAIL (Generative Adversarial Imitation Learning)

GAIL usa uma abordagem adversarial: treina um **discriminador** para distinguir
trajetórias humanas de trajetórias do agente, e usa o sinal do discriminador como
recompensa para o RL.

```
Discriminador: "isso parece jogada humana ou do agente?"
Agente: aprende a agir de forma indistinguível do humano
```

O resultado é que o agente aprende a recompensa *implicitamente* a partir das
demonstrações, sem precisar de engenharia manual de reward. Se o humano gerencia
energia bem, o agente aprende que gerenciar energia é uma característica de "jogar
bem" — mesmo sem receber uma recompensa explícita por isso.

**Vantagem sobre BC:** não sofre distribuição shift; combina imitação com
exploração RL.

**Complexidade:** Média. A biblioteca `imitation` implementa GAIL com suporte a SB3.

---

## World Models — O Agente Aprende o Jogo por Dentro

### Motivação

O gargalo de velocidade do projeto é que cada step demora ~0.7s em tempo real.
Não tem como acelerar isso enquanto o agente interagir com o jogo real.

E se o agente pudesse aprender um *modelo interno do jogo* e treinar nesse modelo,
sem precisar rodar o jogo? Isso é o que os World Models fazem.

### Como funciona (ideia geral)

Um World Model tem três componentes:

```
Encoder: frame 84×84 → representação latente compacta z_t
Modelo de dinâmica: (z_t, ação) → z_t+1 (prevê o próximo estado)
Decoder de recompensa: z_t → recompensa esperada
```

Com esses três treinados, o agente pode "sonhar": simular trajetórias inteiramente
no espaço latente, sem capturar nenhuma tela. O treinamento da política acontece
nesses sonhos — milhares de vezes mais rápido que o jogo real.

```
Fase 1 (coleta real): jogar ~50k steps para aprender o mundo
Fase 2 (imaginação):  treinar a política em 10M+ steps "sonhados"
Fase 3 (fine-tuning): voltar ao jogo real para corrigir erros do modelo
```

---

### DreamerV3

DreamerV3 (Hafner et al., Nature 2025) é o estado da arte em model-based RL visual.
Ele unifica encoder, modelo de dinâmica e política em um único loop de treinamento.

**Resultados conhecidos:**
- Aprendeu a jogar Minecraft do zero (encontrar e usar diamante) sem reward shaping
- Convergiu em domínios que PPO puro nunca aprendeu em tempo razoável
- Generaliza para 150+ domínios sem mudança de hiperparâmetros

**Por que seria relevante para FNAF:**
- Observabilidade parcial é tratada nativamente: o estado latente recorrente
  (similar ao LSTM) resume o histórico de observações
- Não requer reward shaping denso — a recompensa esparsa de sobrevivência é
  suficiente no modelo latente
- A fase de "imaginação" permite milhões de steps sem rodar o jogo

**Limitação:** A implementação é significativamente mais complexa que SB3.
A implementação de referência (`danijar/dreamerv3` no GitHub) está em JAX.
Portar para PyTorch existe (`RSSMv3` e forks), mas requer integração cuidadosa.

**Para o TCC:** implementar DreamerV3 e comparar com PPO seria um resultado
de pesquisa sólido — especialmente se o DreamerV3 convergir em 10× menos
interações com o jogo real.

---

## Decision Transformer — RL como Predição de Sequência

### Motivação

O Transformer (a arquitetura base do GPT) é muito bom em predizer a próxima
palavra dado o contexto anterior. Decision Transformer (DT) aplica essa mesma
ideia para ações: dado o histórico de observações e ações, e o **retorno desejado**,
qual ação tomar agora?

### Como funciona

O input do modelo é uma sequência de triplas:

```
(retorno_desejado_t, observação_t, ação_t),
(retorno_desejado_t+1, observação_t+1, ação_t+1),
...
```

Na inferência, você passa `retorno_desejado = 1000` (quero vencer) e o modelo
prevê as ações que levariam a esse resultado. O Transformer usa atenção sobre
o histórico completo, o que resolve a observabilidade parcial naturalmente —
ele "lembra" todas as câmeras que viu.

**Vantagem para FNAF:**
- Treinamento offline: usa os 700k steps de PPO já coletados como dataset
  (sem precisar rodar o jogo novamente)
- O attention sobre histórico substitui o LSTM sem precisar de `RecurrentPPO`
- Interpretável: os attention weights mostram para quais frames passados o
  modelo está "olhando" ao tomar uma decisão

**Limitação:** sem melhora on-policy — o teto é a qualidade do dataset.
Com 15% de win rate nos dados de PPO, o DT dificilmente supera muito isso
sem dados humanos adicionais.

**Combinação ideal:** DT pré-treinado em dados humanos (BC como base) +
fine-tuning online com PPO.

**Complexidade:** Média. Implementações de referência em PyTorch existem
para Atari (mesmo espaço de observação 84×84). A adaptação para observação
multimodal (imagem + vetor de estado) requer algumas modificações.

---

## Offline RL — Extrair Mais dos Dados Existentes

### Motivação

O projeto já coletou ~700k steps de interação com o jogo. Offline RL é a classe
de métodos que aprende *apenas* de um dataset fixo, sem novas interações.

O PPO descarta a maior parte da experiência coletada (on-policy: só usa o buffer
atual). Offline RL usa *todo* o histórico acumulado.

### CQL (Conservative Q-Learning)

CQL aprende uma Q-function a partir do dataset, mas adiciona uma penalidade que
conserva o valor de ações **fora** do dataset. Isso evita que o agente "alucine"
ações ótimas que nunca apareceram nos dados.

```
Objetivo CQL = Q-learning padrão − α × E[Q(s, ações fora do dataset)]
```

**Vantagem imediata:** os 700k steps de PPO são o dataset. Dá para rodar CQL
hoje, sem coletar mais dados.

**Limitação:** o teto de qualidade é a distribuição do dataset. Se o PPO nunca
explorou a estratégia ótima, CQL não vai descobri-la.

### IQL (Implicit Q-Learning)

IQL usa regressão de expectil para estimar o valor das melhores ações no dataset,
sem precisar extrair a política ótima explicitamente. É mais estável que CQL e
funciona bem com datasets mistos (bons e maus episódios juntos).

**Biblioteca:** `imitation` (Python) implementa ambos, com suporte a SB3 e
espaços de observação `Dict`.

---

## Observabilidade: Parcial vs. Total

### Contexto

O projeto atual mantém observabilidade parcial intencionalmente — o agente vê
apenas o que um humano veria (a tela do jogo). Isso é consistente com o objetivo
de estudar como uma IA aprende a partir de percepção visual similar à humana.

No entanto, é possível expandir o espaço de estado lendo dados diretamente da
memória do processo do jogo.

### O que seria possível com leitura de memória

O FNAF1 é um jogo de 2014. A comunidade modding documentou os offsets de memória
do processo `fnaf.exe`:

- Posição atual de Freddy, Bonnie, Chica e Foxy (inteiros 0-6 mapeando locais)
- Nível de IA de cada animatronico (0-20, aumenta conforme a noite avança)
- Estado interno das portas e câmeras (verificação independente do pixel)

Com `pymem` (biblioteca Python), esses valores podem ser lidos a cada step e
adicionados ao vetor `estados`. O espaço de observação passaria de 8 para ~16-18
dimensões, com informação *completa* sobre o estado do jogo.

### Por que isso é academicamente valioso (como experimento comparativo)

Implementar as duas versões — parcial (atual) e total (com memória) — e comparar
os resultados cria um experimento científico legítimo:

```
Hipótese: o acesso ao estado completo do jogo acelera o aprendizado em X×
Experimento: mesmos hiperparâmetros, mesma arquitetura, diferença apenas no espaço de estado
Métricas: steps até primeiro win, win rate por noite, sample efficiency
```

Isso não invalida o objetivo de estudar aprendizado por percepção visual — pelo
contrário, quantifica exatamente *quanto* a observabilidade parcial custa.

Para o TCC, essa comparação seria um resultado concreto e mensurável.

---

## VLMs como Guia ou Avaliador

### Ideia

Modelos de visão-linguagem (VLMs) como Claude, GPT-4V ou LLaVA locais recebem
uma imagem e texto e retornam uma resposta em linguagem natural. Isso pode ser
usado de três formas diferentes no projeto:

**1. Descrição de estado:**
Enviar o frame atual para o VLM e receber uma descrição: "câmera mostra o corredor
esquerdo vazio, energia em 73%, porta direita fechada". Essa descrição pode ser
codificada como embedding e adicionada à observação — um canal de informação mais
rico que os 84×84 pixels brutos.

**2. Sinal de recompensa auxiliar:**
Em vez de recompensa manual, perguntar ao VLM: "dado esse estado, o agente está
jogando bem?" e usar a resposta como sinal auxiliar. Isso é especialmente útil
para comportamentos difíceis de especificar (ex: "verificar câmeras de forma
ordenada").

**3. Planner de alto nível:**
A cada N steps (não a cada step, por causa da latência), consultar o VLM para
uma decisão estratégica de alto nível: "o que você faria agora?" O RL cuida da
execução tática, o VLM cuida da estratégia.

**Limitação prática:** a latência de inferência de um VLM (0.5-2s por chamada)
é incompatível com controle frame-a-frame. O uso mais realista é a cada 10-20
steps como sinal de alto nível, não como controlador direto.

---

## Resumo comparativo

| Método | Resolve qual problema | Requer dados humanos? | Complexidade | Interesse acadêmico |
|--------|----------------------|----------------------|--------------|---------------------|
| DAgger / GAIL | Exploração ineficiente | Sim (2-5h de gameplay) | Baixa-Média | Alto |
| DreamerV3 | Velocidade de coleta, obs. parcial | Não | Alta | Muito alto |
| Decision Transformer | Obs. parcial, uso do histórico | Ideal (mas funciona sem) | Média | Alto |
| Offline RL (CQL/IQL) | Aproveitar dados existentes | Não (usa dados do PPO) | Baixa | Médio |
| VLM como guia | Reward engineering | Não | Média-Alta | Alto (novidade) |
| Memória do processo | Observabilidade (como experimento) | Não | Média | Alto (comparativo) |

A combinação de maior potencial para o TCC:
**DAgger para bootstrapping → PPO/RecurrentPPO refinando → DreamerV3 como
experimento de eficiência de samples.**
