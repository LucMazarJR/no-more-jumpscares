# Guia de Conceitos e Funcionamento — *No More Jumpscares*

> Um guia didático, do zero ao avançado, sobre **como o projeto funciona** e **a teoria por trás
> de cada peça**. Escrito para quem nunca viu termos como *entropia*, *epoch* ou *política*, mas
> também útil como referência depois.

---

## Sumário

1. [Como ler este guia](#0-como-ler-este-guia)
2. [O problema em uma frase](#1-o-problema-em-uma-frase)
3. [Fundamentos de RL (do zero)](#2-fundamentos-de-rl-do-zero)
4. [O "cérebro": política e valor](#3-o-cérebro-política-e-valor)
5. [Recompensa: o que o agente realmente persegue](#4-recompensa-o-que-o-agente-realmente-persegue)
6. [Como o agente aprende — PPO](#5-como-o-agente-aprende--ppo)
7. [Exploração × Explotação — o coração do problema atual](#6-exploração--explotação--o-coração-do-problema-atual)
8. [Normalização e estabilidade](#7-normalização-e-estabilidade)
9. [Como o ambiente "enxerga" o jogo (visão)](#8-como-o-ambiente-enxerga-o-jogo-visão)
10. [Engenharia de execução (RL em tempo real)](#9-engenharia-de-execução-rl-em-tempo-real)
11. [Técnicas avançadas (BC, LSTM, currículo)](#10-técnicas-avançadas-bc-lstm-currículo)
12. [As Decisões de design 4–7](#11-as-decisões-de-design-47)
13. [Como ler o treino na prática](#12-como-ler-o-treino-na-prática)
14. [Glossário](#13-glossário)
15. [Mapa conceito → código](#14-mapa-conceito--código)

<div style="page-break-after: always;"></div>

## 0. Como ler este guia

Você não precisa saber nada de IA para começar. Cada conceito segue sempre a mesma receita:

1. **Intuição / analogia** — a ideia em linguagem do dia a dia.
2. **A matemática** — a fórmula, num bloco de código, só para fixar.
3. **No projeto** — como isso foi aplicado aqui, com o arquivo onde mora.
4. **Nos logs** — como ver isso acontecendo durante o treino.
5. **Armadilhas** — onde costuma dar errado.

Caixas que vão aparecer:

> **Analogia:** uma comparação do mundo real.

> **No projeto:** onde/como o conceito vive no código.

> **Atenção:** um erro comum ou um detalhe que morde.

> **Onde no código:** o arquivo e a linha para você ir conferir.

Os símbolos das fórmulas: `γ` (gama, desconto), `π` (pi, a política), `Φ` (fi, potencial de
segurança), `Σ` (soma), `H` (entropia), `≈` (aproximadamente).

<div style="page-break-after: always;"></div>

## 1. O problema em uma frase

> **Ensinar um programa a sobreviver à noite do *Five Nights at Freddy's 1* sozinho — olhando
> para a tela e apertando teclas/mouse — sem nunca ler a memória interna do jogo.**

Isso é importante e define todas as escolhas técnicas:

- A IA **não** sabe onde os animatrônicos estão. Ela só tem **a imagem da tela** (igual a um
  humano) e alguns números que o projeto **extrai dessa imagem** (energia, portas, etc.).
- A IA **não** recebe um manual de "como jogar". Ela precisa **descobrir** a estratégia tentando,
  errando, morrendo e sendo recompensada.

Por que não escrever um *script* fixo ("se Bonnie aparece, fecha a porta")? Porque:

1. Os animatrônicos se movem de forma **semi-aleatória** — não há regra fixa.
2. A energia é limitada: fechar porta à toa **mata por esgotamento**. É preciso **gerenciar um
   recurso ao longo do tempo**, equilibrando risco e custo.

Esse tipo de problema — **decidir repetidamente sob incerteza para maximizar um resultado de longo
prazo** — é exatamente o que **Aprendizado por Reforço** (RL, *Reinforcement Learning*) resolve.

<div style="page-break-after: always;"></div>

## 2. Fundamentos de RL (do zero)

### 2.1 O loop agente ↔ ambiente

> **Analogia:** imagine ensinar alguém a jogar um videogame **sem explicar as regras**, só dizendo
> "boa!" ou "ruim!" depois de cada partida. Com o tempo, a pessoa associa o que fez ao resultado.
> O "alguém" é o **agente**; o videogame é o **ambiente**; o "boa/ruim" é a **recompensa**.

O coração do RL é um laço que se repete milhares de vezes:

```text
        ┌─────────────────────────────────────────────┐
        │                                             │
        ▼                                             │
   ┌─────────┐   ação (a)        ┌──────────────┐     │
   │ AGENTE  │ ───────────────►  │  AMBIENTE    │     │
   │ (a IA)  │                   │ (o jogo FNAF)│     │
   └─────────┘                   └──────────────┘     │
        ▲                              │              │
        │   observação (s) + recompensa (r)           │
        └─────────────────────────────────────────────┘
```

Cada volta desse laço é um **step** (passo). Uma sequência de steps do início ao fim (do começo da
noite até vencer/morrer) é um **episódio**.

### 2.2 Os cinco ingredientes

| Termo | O que é | No projeto |
|---|---|---|
| **Ambiente** | O mundo onde o agente age | `FNAFEnv`, em [fnaf_env.py](../src/environment/fnaf_env.py) |
| **Observação (estado `s`)** | O que o agente "vê" a cada step | imagem 84×84 + 11 números |
| **Ação (`a`)** | O que o agente pode fazer | 17 ações (não fazer nada, portas, luzes, câmeras) |
| **Recompensa (`r`)** | O sinal de "bom/ruim" | +500 vitória, −100 morte, + bônus densos |
| **Política (`π`)** | A estratégia do agente | a rede neural treinada |

### 2.3 A observação no projeto

A cada step, o agente recebe um **dicionário** com duas partes (um *espaço de observação*
multimodal — duas modalidades de dado diferentes):

```text
observação = {
    "imagem":  matriz 84 × 84 × 1  (a tela do jogo, reduzida e em tons de cinza)
    "estados": vetor de 11 números (todos normalizados entre 0 e 1)
}
```

Os 11 estados, na ordem exata do código:

```text
[0] porta_esquerda   (0/1)         [6]  energia / 100         (0.0–1.0)
[1] porta_direita    (0/1)         [7]  tempo_jogo / 535      (0.0–1.0)
[2] luz_esquerda     (0/1)         [8]  ameaça_esquerda (0/1)
[3] luz_direita      (0/1)         [9]  ameaça_direita  (0/1)
[4] camera_aberta    (0/1)         [10] noite / 7            (dificuldade)
[5] camera_ativa / 11
```

> **Onde no código:** o vetor é montado em `_capturar_observacao`
> ([fnaf_env.py:1086](../src/environment/fnaf_env.py#L1086)).

> **Atenção:** por que normalizar tudo entre 0 e 1? Redes neurais aprendem muito melhor quando as
> entradas têm escalas parecidas. Se "energia" fosse 0–100 e "porta" fosse 0/1, a energia
> dominaria os cálculos só por ser numericamente maior.

### 2.4 As 17 ações

```text
 0  nada                 6  camera_1a     12  camera_4a
 1  porta_esquerda       7  camera_1b     13  camera_4b
 2  porta_direita        8  camera_1c     14  camera_5
 3  luz_esquerda         9  camera_2a     15  camera_6
 4  luz_direita         10  camera_2b     16  camera_7
 5  abrir_fechar_camera 11  camera_3
```

É um **espaço de ação discreto** (`Discrete(17)`): a cada step o agente escolhe **exatamente uma**
das 17. "Não fazer nada" (ação 0) também é uma escolha legítima — às vezes a melhor jogada é
economizar energia.

> **Onde no código:** dicionário `ACOES` em [fnaf_env.py:33](../src/environment/fnaf_env.py#L33).

<div style="page-break-after: always;"></div>

## 3. O "cérebro": política e valor

### 3.1 O que é uma política (`π`)

> **Analogia:** a política é o "instinto" do jogador. Dada a situação atual, ela diz **com que
> probabilidade** tomar cada ação possível.

Formalmente, a política é uma **distribuição de probabilidade sobre as ações**, dado o estado:

```text
π(a | s) = probabilidade de escolher a ação 'a' quando se está no estado 's'

Exemplo num dado momento:
   nada              → 60%
   fechar porta dir  → 25%
   abrir câmera      → 10%
   (as outras 14)    →  5% somadas
```

Repare que é **probabilístico**, não fixo. Isso é proposital e será o tema central da Parte 6: uma
política que sempre escolhe a mesma ação não consegue **explorar** alternativas melhores.

### 3.2 O que é a função de valor (`V`)

> **Analogia:** se a política é o instinto ("o que fazer agora"), o valor é o **placar mental**
> ("quão boa é a minha situação?"). Um bom jogador sente que "2h da manhã com 80% de energia e
> portas livres" é uma situação melhor que "5h com 10% e Bonnie batendo".

```text
V(s) = quanto de recompensa eu espero acumular do estado 's' até o fim do episódio
```

### 3.3 Ator-Crítico

Os algoritmos modernos (como o **PPO**, que este projeto usa) treinam **duas coisas ao mesmo
tempo**:

- **Ator** = a política `π` (escolhe as ações).
- **Crítico** = a função de valor `V` (julga quão boa foi a situação).

O crítico serve para o ator aprender mais rápido: em vez de esperar o fim do episódio para saber se
uma jogada foi boa, o crítico dá um palpite imediato. (Detalhes na Parte 5, "vantagem".)

### 3.4 Como isso vira uma rede neural — o `MultimodalExtractor`

A imagem e os 11 estados são tipos de dado muito diferentes, então cada um passa por um caminho
próprio antes de se juntarem:

```text
   imagem 84×84×1                      estados [11]
        │                                  │
        ▼                                  ▼
 ┌──────────────┐                   ┌──────────────┐
 │     CNN      │                   │  MLP pequeno │
 │ (3 camadas   │                   │ Linear(11→32)│
 │  convoluc.)  │                   │   + ReLU     │
 └──────────────┘                   └──────────────┘
        │ 3136 números                     │ 32 números
        └────────────┬─────────────────────┘
                     ▼
            concatena → 3168 números
                     ▼
            ┌──────────────────┐
            │ Linear(3168→256) │  ← "features" finais (256 números)
            │     + ReLU       │
            └──────────────────┘
                     ▼
        ┌────────────┴────────────┐
        ▼                         ▼
   Ator (π)                  Crítico (V)
   17 probabilidades         1 número (valor)
```

- A **CNN** (*Convolutional Neural Network*, rede convolucional) é especializada em imagens:
  detecta bordas, formas e padrões na tela. As 3 camadas são o padrão clássico de RL com pixels
  (o mesmo da DQN da Atari).
- O **MLP** (rede densa simples) processa os 11 números de estado.
- Os dois resultados são **concatenados** e comprimidos em **256 "features"** que o ator e o
  crítico usam.

> **Onde no código:** [multimodal_policy.py](../src/agent/multimodal_policy.py). A CNN tem
> `Conv2d(1→32, 8, stride 4)` → `Conv2d(32→64, 4, stride 2)` → `Conv2d(64→64, 3, stride 1)`, o que
> transforma 84×84 em 7×7×64 = 3136 números.

> **Atenção:** mudar o número de estados (de 11 para 12, por exemplo) muda a camada `Linear(11→32)`
> e **invalida todos os modelos salvos** — eles teriam pesos do tamanho errado. Por isso mexer na
> observação obriga a treinar do zero.

<div style="page-break-after: always;"></div>

## 4. Recompensa: o que o agente realmente persegue

A recompensa é a **única** forma de dizer ao agente o que você quer. Ele não tem objetivos
próprios — ele literalmente maximiza a soma de recompensas. Se a recompensa estiver mal desenhada,
ele vai fazer **exatamente o que você pediu, não o que você queria**.

### 4.1 Recompensa esparsa vs densa

```text
Esparsa:  só no fim →  ... 0  0  0  0  0  0  +500   (venceu)
Densa:    ao longo  →  +0.1 +0.1 +3 +0.1 ... +500
```

> **Analogia:** ensinar alguém a cozinhar dando nota **só no final do prato** (esparsa) é lento —
> a pessoa não sabe qual passo estragou. Dar microfeedback durante o preparo (densa) acelera, mas
> tem um risco: se você elogiar demais "mexer a panela", a pessoa fica mexendo a panela e esquece
> de cozinhar. Isso é **reward hacking**.

No projeto, o sinal terminal (o objetivo verdadeiro) é:

```text
RECOMPENSA_VITORIA =  +500   (sobreviveu à noite)
RECOMPENSA_MORTE   =  −100   (qualquer animatrônico, incluindo fechar o jogo = Golden Freddy)
```

> **Onde no código:** [fnaf_env.py:257](../src/environment/fnaf_env.py#L257).

> **Nota histórica:** esses valores já foram **+1000 / −500** e foram **reduzidos**. Terminais
> gigantes em relação ao sinal denso (~0.5 por step) fazem o **crítico** ter que prever alvos com
> variância enorme concentrada num único step — o aprendizado do valor fica lento e ruidoso.

### 4.2 O sinal denso: sobreviver no relógio, não por step

O sinal denso **não** paga por step (número de ações), e sim por **tempo real sobrevivido**:

```text
recompensa_densa_do_step = (Δt / 535) × 60       # 535s = duração da noite; 60 = orçamento total
+ 3.0 a cada "hora" do jogo alcançada (checkpoint)
```

> **Atenção (reward hacking real que já aconteceu aqui):** quando se pagava por *step*, o agente
> aprendia a **spammar ações rápidas e baratas** para acumular mais steps — durar "acampado" na
> câmera pagava quase tanto quanto vencer. Atrelar ao **relógio** conserta isso: 10 segundos
> sobrevividos valem o mesmo, faça o que fizer.

O orçamento denso total da noite (~60) é **muito menor** que a vitória (500). Isso é deliberado:
**vencer domina "só durar"**. O denso só serve para o agente não ficar totalmente no escuro até a
recompensa final chegar.

### 4.3 *Reward shaping* baseado em potencial (a peça mais sofisticada)

Como dar dicas de **como jogar** (ex.: "fechar a porta com a ameaça presente é bom") **sem** mudar
qual é a estratégia ótima e **sem** criar vícios? A resposta é uma técnica clássica
(Ng, Harada & Russell, 1999): **potential-based shaping**.

Define-se uma função `Φ(s)` ("phi") que mede **quão segura é a situação agora**. A recompensa extra
dada num step é a **variação descontada** desse potencial:

```text
shaping = γ · Φ(estado_depois) − Φ(estado_antes)
```

A mágica: ao somar isso ao longo de um episódio inteiro, os termos **se cancelam em cascata**
(telescopam) e o total tende a zero. Ou seja, **não há ganho líquido em ficar manipulando o Φ** —
ele só faz o crédito de uma boa ação **chegar mais cedo**, sem mover o ótimo.

O `Φ` do projeto:

```text
Φ = + 0.5  para cada lado com AMEAÇA presente E porta FECHADA   (lidou com o perigo)
    − 0.5 × risco_foxy                                          (câmera negligenciada demais)

risco_foxy sobe só DEPOIS de 20 steps sem olhar câmera (paciência), até no máx 1.0
```

> **Onde no código:** `_potencial_seguranca` em
> [fnaf_env.py:1011](../src/environment/fnaf_env.py#L1011); a soma `γ·Φ' − Φ` entra no `step` em
> [fnaf_env.py:780](../src/environment/fnaf_env.py#L780).

> **Atenção:** uma versão **antiga** penalizava unilateralmente "não checar câmera", e o agente
> aprendeu a **acampar na câmera** para fugir da penalidade. A versão potential-based conserta: como
> checar a câmera **sobe** o Φ e negligenciar **baixa**, e isso telescopa, **não existe vantagem em
> acampar**. Esse é o poder do método — guia sem viciar.

### 4.4 Composição final da recompensa de um step

```text
recompensa_do_step =
        terminal (+500 vitória / −100 morte)            ← se o episódio acabou
      ou
        denso (tempo sobrevivido + bônus de hora)        ← se continua
      − 0.1 se a ação foi inválida (sem efeito)
      + (γ · Φ_depois − Φ_antes)                          ← shaping potential-based
```

<div style="page-break-after: always;"></div>

## 5. Como o agente aprende — PPO

**PPO** (*Proximal Policy Optimization*) é o algoritmo de treino. A ideia geral: jogar um pouco,
olhar o que deu certo, ajustar a política **com cuidado** (sem mudar demais de uma vez), repetir.

### 5.1 Retorno e desconto (`γ`)

O agente não quer maximizar a recompensa **agora**, e sim a **soma futura**. Mas o futuro distante
é incerto, então cada recompensa futura é **descontada**:

```text
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + γ³·r_{t+3} + ...

G_t = "retorno": a recompensa total descontada a partir do step t
γ   = "gamma", fator de desconto entre 0 e 1
```

`γ` controla **quão longe** o agente enxerga. O **horizonte efetivo** (quantos steps à frente
importam) é aproximadamente `1 / (1 − γ)`:

| γ | Horizonte efetivo |
|---|---|
| 0.90 | ~10 steps |
| 0.99 | ~100 steps |
| 0.995 | ~200 steps |
| **0.997** (atual) | **~333 steps** |
| 0.999 | ~1000 steps |

> **No projeto:** `γ = 0.997` ([fnaf_env.py:196](../src/environment/fnaf_env.py#L196)). Uma noite
> tem ~700 steps. Com γ baixo (0.99), a vitória no fim valeria `0.99^700 ≈ 0.0005` no início —
> praticamente zero, o agente "não sente" que sobreviver importa. Com 0.997, a vitória propaga
> melhor para os primeiros minutos.

> **Atenção:** γ **alto demais** (≥0.999) estende o horizonte **além do episódio** e desestabiliza
> o crítico (pequenas mudanças na política viram grandes mudanças na estimativa de valor). A faixa
> segura aqui é 0.993–0.997.

### 5.2 Vantagem (*advantage*) — o crítico em ação

Como saber se uma ação foi **melhor do que o esperado**? Compara-se o que de fato aconteceu com o
palpite do crítico:

```text
vantagem ≈ (retorno observado)  −  V(s)
              (o que veio)         (o que o crítico esperava)

vantagem > 0  →  a ação foi melhor que o esperado  → reforça
vantagem < 0  →  a ação foi pior  que o esperado   → enfraquece
```

Na prática o PPO usa uma versão suavizada disso chamada **GAE** (*Generalized Advantage
Estimation*), que equilibra "confiar no crítico" vs "esperar o retorno real". A intuição acima
basta para este guia.

### 5.3 O ciclo do PPO: rollout → update

```text
┌── 1. ROLLOUT: joga e coleta n_steps=2048 experiências (≈ 3 noites) ──┐
│                                                                       │
│   (s, a, r) (s, a, r) (s, a, r) ... × 2048                            │
└───────────────────────────────────────────────────────────────────┬─┘
                                                                      ▼
┌── 2. UPDATE: reusa esse lote n_epochs=10 vezes para ajustar a rede ──┐
│   - calcula vantagens                                                 │
│   - em mini-lotes de batch_size=64                                    │
│   - com o "freio" do clipping (5.5)                                   │
└───────────────────────────────────────────────────────────────────┬─┘
                                                                      ▼
                            descarta o lote, volta ao passo 1
```

> **PPO é *on-policy*:** ele só aprende com dados gerados pela **política atual**. Por isso joga,
> aprende e **descarta** — não guarda um histórico gigante como alguns outros métodos. Isso é parte
> do porquê de "amostra ser o gargalo": cada update precisa de dados novos, e gerar dados aqui
> custa **tempo real de jogo**.

### 5.4 `n_steps`, `batch_size` e `n_epochs`

| Parâmetro | Valor | O que é | Efeito |
|---|---|---|---|
| `n_steps` | 2048 | experiências coletadas antes de cada update (≈3 noites) | maior = gradiente mais estável, porém mais lento |
| `batch_size` | 64 | tamanho do mini-lote dentro do update | menor = updates mais ruidosos |
| `n_epochs` | 10 | quantas vezes o **mesmo** lote é reusado | maior = extrai mais do lote, **mas arrisca over-otimizar** |

> **Atenção — isto conecta direto com o problema atual (Parte 6):** com `n_steps=2048` ≈ apenas
> **3 episódios** e `n_epochs=10`, cada lote pequeno e pouco diverso é "espremido" 10 vezes. Isso
> **super-otimiza** a política para aquelas 3 noites específicas e é um dos motivos de a entropia
> colapsar cedo. O experimento **E1** (reduzir `n_epochs` para 4) ataca exatamente isso.

### 5.5 O "freio": clipping, `approx_kl` e `clip_fraction`

O "Proximal" do PPO vem daqui: ele **impede que a política mude demais** num único update. Se uma
ação parece ótima, o PPO limita o quanto sua probabilidade pode subir de uma vez (o `clip_range`,
padrão 0.2 = ±20%). Sem esse freio, um lote sortudo poderia jogar a política para um extremo e
destruir o que já foi aprendido.

Dois medidores no tensorboard mostram o freio trabalhando:

```text
approx_kl      = o quanto a política mudou neste update (KL alto = mudou muito)
clip_fraction  = fração dos updates que bateram no limite do freio
```

> **Nos logs:** `clip_fraction > 0.3` de forma sistemática indica `n_epochs` alto demais (a política
> está tentando andar mais do que o freio permite).

### 5.6 Learning rate e schedule

A **taxa de aprendizado** (`learning_rate`) é o tamanho do passo ao ajustar os pesos. Alta demais =
instável; baixa demais = lento. O projeto usa um **schedule linear** que decai ao longo do treino:

```text
learning_rate: 3e-4 ────────────────► 3e-5   (decai conforme o treino avança)
                (início)               (piso, não chega a zero)
```

> **Onde no código:** `linear(3e-4, 3e-5)` em [train.py:317](../src/agent/train.py#L317). O piso
> (3e-5, não 0) evita "congelar" quando se **retoma** um treino — ver Parte 11, Decisão 6.

<div style="page-break-after: always;"></div>

## 6. Exploração × Explotação — o coração do problema atual

Este é **o** tema que motivou todo o resto desta sessão. Leia com calma.

### 6.1 O dilema

> **Analogia:** você se muda para uma cidade nova. **Explorar** = testar restaurantes diferentes
> (pode achar um ótimo, pode comer mal). **Explotar** (*exploit*) = sempre voltar ao que você já
> sabe que é bom. Se você nunca explora, **nunca descobre o restaurante excelente** da esquina —
> fica preso no "razoável" para sempre.

O agente vive esse dilema a cada step: repetir o que parece bom (explotar) ou testar algo novo
(explorar). RL precisa de **exploração suficiente** para encontrar boas estratégias — senão, fixa
cedo numa medíocre.

### 6.2 Entropia `H(π)` — medindo a exploração

A **entropia** mede **quão espalhada** está a distribuição de ações da política:

```text
H(π) = − Σ  π(a)·log π(a)        (soma sobre todas as ações)

política quase uniforme (explora muito)  → entropia ALTA
política quase determinística (1 ação)   → entropia BAIXA (≈ 0)
```

Exemplo concreto com as 17 ações do projeto:

```text
Uniforme (cada ação 1/17):   H = log(17) ≈ 2.83   ← entropia máxima
Quase decidida (1 ação 99%): H ≈ 0.08              ← quase sem exploração
```

> **Guarde esse 2.83.** Ele vai reaparecer no estudo de caso (6.5).

### 6.3 `ent_coef` — empurrando o agente a explorar

O PPO **minimiza** uma função de perda. Para **incentivar** entropia (exploração), subtrai-se a
entropia da perda, multiplicada por um coeficiente:

```text
perda_total = perda_da_política  −  ent_coef · H(π)

Como a perda é minimizada, subtrair H faz o otimizador querer AUMENTAR H.
ent_coef = "quão forte" é esse empurrão para explorar.
```

| `ent_coef` | Efeito |
|---|---|
| ≈ 0.0 | converge rápido para uma política fixa — **e às vezes fixa numa ruim** |
| 0.005–0.02 (faixa do projeto) | explora o suficiente, ainda aprende |
| ≥ 0.1 | quase aleatório mesmo depois de muito treino |

> **No projeto:** o `ent_coef` **não é fixo**. Começa em **0.02** e decai até **0.005**, mas só
> **depois** que o agente começa a vencer (ver 6.4).

### 6.4 O `EntropiaSchedule` e o conceito de "gate"

A ideia do projeto: manter a exploração **alta enquanto o agente ainda não vence** e só
**consolidar** (reduzir entropia) **depois** que ele aprende a ganhar. Isso é controlado por um
**gate** (portão):

```text
SE a taxa de vitória (janela de 50 episódios) cruzar 0.20 (20%):
    abre o gate → começa a decair ent_coef de 0.02 → 0.005
SENÃO:
    mantém ent_coef = 0.02 (continua explorando)

O gate abre uma vez e não fecha. Nunca vai a zero (zero = congela a política).
```

> **Onde no código:** classe `EntropiaSchedule` em [train.py:210](../src/agent/train.py#L210).

> **Atenção (o problema que diagnosticamos):** o gate abre em **20%**. Mas 20% é exatamente o valor
> do **ótimo local** em que o agente fica preso (vence a Noite 1, morre na Noite 2). Ou seja, o
> schedule começa a **matar a exploração no exato momento em que o agente alcança a armadilha** —
> ajudando a cravá-lo nela. O experimento **E2** propõe subir esse gate para uma competência acima
> do ótimo local (ex.: 40%).

### 6.5 Colapso de entropia, convergência prematura e ótimo local

```text
Ótimo local = uma estratégia que não é a melhor possível, mas da qual qualquer
              desvio pequeno parece pior. O agente para de melhorar não porque
              aprendeu o ideal, mas porque perdeu o incentivo de explorar saídas.
```

O que acontece mecanicamente: o PPO encontra cedo um sinal positivo **fácil** (sobreviver à Noite 1
= +500). As vantagens daquela estratégia ficam altas, a política concentra probabilidade nela, a
**entropia colapsa**, e os estados difíceis (Noite 2: Foxy, ritmo mais rápido) **nunca chegam a ser
explorados**. O agente trava.

#### Estudo de caso real (este projeto, ~400k steps)

Reconstruímos a entropia da política a partir do tensorboard. O resultado:

```text
entropia H da política:  ~141  ────────► ~80  e estabilizou (por volta de 234k steps)
taxa de vitória:         travada em ~20% (vence Noite 1, morre Noite 2), sem subir
```

> **A dúvida que isso resolve:** "se amostra é o gargalo e a rede precisaria de **milhões** de
> steps, por que a entropia já cai em 400k?" Porque **colapsar a entropia ≠ resolver a tarefa**:
>
> - **Step ≠ episódio ≠ trajetória diversa.** Uma noite ≈ 700 steps / ~8 min. 400k steps ≈ apenas
>   **500–880 noites**. O PPO atualiza a cada ~3 noites, reusadas 10×. Em número de *trajetórias*,
>   isso é **pouquíssimo** — colapsar entropia aí é rápido **por construção**.
> - Comprometer-se com **uma** política é barato (dezenas de milhares de steps). Aprender a política
>   **ótima** sobre todo o espaço é caro (milhões). O agente fez o barato e parou: agarrou o ótimo
>   local antes de a parte difícil ser amostrada. Os "milhões de steps" nunca são usados
>   produtivamente porque a exploração morreu antes.

Por isso "sempre acontece em todo teste": a causa é **estrutural** (otimizador + schedule de
exploração + terminal esparso), não o detalhe de recompensa que se mexe. As mitigações (E1, E2, E3)
atacam a estrutura.

<div style="page-break-after: always;"></div>

## 7. Normalização e estabilidade

### 7.1 `VecNormalize`

Redes neurais treinam melhor quando os números têm escala controlada. O `VecNormalize` do
Stable-Baselines3 padroniza valores em tempo real. No projeto ele é usado assim:

```text
VecNormalize(norm_obs=False, norm_reward=True, gamma=0.997)
```

- `norm_reward=True` → **normaliza a recompensa** (divide por um desvio-padrão móvel do retorno).
  Isso evita que os terminais grandes (+500/−100) desestabilizem o crítico.
- `norm_obs=False` → **NÃO normaliza a observação**. A observação já vem pronta: a imagem é
  normalizada pelo próprio SB3 (pixels → 0–1) e os 11 estados já são montados entre 0 e 1 no
  ambiente. Normalizar de novo seria redundante e poderia distorcer.

> **Onde no código:** [train.py:296](../src/agent/train.py#L296).

> **Atenção:** o `gamma` passado ao `VecNormalize` **tem que ser o mesmo** do PPO e do shaping Φ.
> São três lugares usando γ; se divergirem, o shaping deixa de "telescopar" e passa a mover o ótimo.
> Por isso o γ é definido **num único lugar** (`fnaf_env.GAMMA`) e importado por todos.

### 7.2 Por que medir por vitória, não por recompensa

A recompensa **muda de significado** entre versões (toda vez que você ajusta o shaping ou os
terminais, o número vira outra coisa). Então comparar "recompensa do run A vs run B" é enganoso. As
métricas honestas e estáveis são **taxa de vitória** e **tempo de sobrevivência** — elas significam
a mesma coisa em qualquer versão.

<div style="page-break-after: always;"></div>

## 8. Como o ambiente "enxerga" o jogo (visão)

Aqui está uma diferença enorme em relação a um ambiente de RL "de laboratório": o jogo **não expõe**
seu estado interno. O projeto precisa **extrair tudo da imagem** com visão computacional (OpenCV).
Isso é metade do trabalho do `FNAFEnv`.

### 8.1 Captura de tela

```text
mss (captura rápida da janela)  →  recorta SÓ a janela do jogo  →
converte para tons de cinza  →  redimensiona para 84×84
```

> **Onde no código:** `GameCapture` em [capture.py](../src/utils/capture.py). A função
> `melhor_janela` escolhe a janela real do jogo (filtra janelas-fantasma 1×1 e prefere as que
> parecem ser o FNAF), garantindo que a IA e as ferramentas leiam **a mesma** janela.

### 8.2 Detecção de ameaça (template matching) — Decisão 4A

Para saber se há um animatrônico no vão da porta, o projeto usa **template matching**: compara um
pedaço da tela com uma imagem de referência (o rosto do animatrônico) e mede a semelhança (0 a 1).

```text
semelhança = match(região_do_vão, template_do_rosto)

semelhança > 0.70 (LIMIAR_AMEACA)  →  ameaça PRESENTE
```

Há também a detecção complementar de **corredor vazio iluminado**: um corredor sem ninguém, com a
luz acesa, tem **textura** (tijolos) e portanto **desvio-padrão alto** dos pixels. Acima de
`LIMIAR_VAZIO = 11.0`, confirma-se "vão vazio".

> **Onde no código:** `_match_ameaca`, `_sombra_no_vao` e `_atualizar_ameaca` em
> [fnaf_env.py](../src/environment/fnaf_env.py) (a partir da linha ~1220).

### 8.3 Leitura de energia (OCR) — Decisão 4B

A energia real é lida diretamente do texto "Power left: XX%" na tela, reconhecendo os **dígitos**
por template. Como a leitura visual pode falhar num frame, o projeto combina duas fontes:

```text
energia simulada (estimada por consumo)  ←── corrigida por ──→  leitura visual (foto)

regra "photo-primary": a foto manda; se a foto falhar (None), mantém a simulação;
uma subida súbita é rejeitada (energia não sobe), uma queda re-ancora na foto.
```

> **Onde no código:** `_ler_energia` + `validar_leitura_energia`, usados em
> [fnaf_env.py:1082](../src/environment/fnaf_env.py#L1082).

### 8.4 Estado real da porta pela cor do botão — Decisão 4B

O projeto **não confia cegamente** no que a IA *acha* que fez. Ele confere o estado real lendo a
**cor do botão** de porta no painel (HUD):

```text
verde (G > R)     → porta FECHADA
vermelho (R > G)  → porta ABERTA
```

Isso é usado para **sincronizar** o estado interno com a realidade (ver 9.2).

<div style="page-break-after: always;"></div>

## 9. Engenharia de execução (RL em tempo real)

Treinar contra um jogo **real, rodando em tempo real**, cria problemas que ambientes simulados não
têm. Esta parte é o que faz o projeto funcionar na prática.

### 9.1 Step de duração variável

Cada step leva tempo real (há um `time.sleep` e a captura/processamento variam). Por isso o "Δt"
real é medido a cada step (~0.7s) e a recompensa densa usa **esse Δt** (tempo), não a contagem de
steps — como vimos em 4.2.

### 9.2 Dessincronização (*desync*) e auto-correção

> **Analogia:** é como dirigir por GPS com sinal ruim. O GPS (estado interno) acha que você está
> numa rua; de vez em quando ele se reposiciona com o sinal real (a tela). Sem isso, o erro se
> acumula e você "vira numa rua que não existe".

A IA mantém um estado interno (porta aberta/fechada, câmera aberta), mas a animação do jogo pode
divergir. O projeto **re-ancora** periodicamente lendo a tela:

- **Câmera:** a cada 3 steps, confere o template "YOU" do mapa de câmeras. Se discordar 2 vezes
  seguidas, corrige o estado interno.
- **Porta:** ao acender a luz de um lado, lê a cor do botão daquele lado e corrige.

Cada correção é registrada em `logs/desyncs.log` para auditoria.

> **Onde no código:** bloco de sincronia no `step` ([fnaf_env.py:668](../src/environment/fnaf_env.py#L668)).

### 9.3 Cooldowns (modelando a animação)

Abrir/fechar uma porta ou a câmera tem uma **animação** que leva tempo; durante ela, clicar de novo
não faz nada. O projeto modela isso com **cooldowns** (porta ~0.6s, câmera ~1.0s). Uma ação durante
o cooldown é marcada como **inválida** (`−0.1` de recompensa, "isso não fez nada").

### 9.4 Reset pelo menu (Decisão 7)

Reiniciar um episódio no FNAF não é trivial — depende do que aconteceu:

```text
decidir_reset(...):
   vitória  → "nenhum"   (o jogo já emendou na próxima noite sozinho, sem menu)
   morte    → "new_game" (clica New Game → Noite 1)
              ou "continue" (retoma a noite onde morreu, no modo currículo)
```

> **Onde no código:** função pura `decidir_reset` em
> [fnaf_env.py:142](../src/environment/fnaf_env.py#L142) — separada e testável offline.

### 9.5 Interrupção: quando a janela some

O Golden Freddy (um animatrônico raro) **fecha o jogo** com um *crash-jumpscare* em vez de mostrar
"Game Over". O projeto trata "a janela sumiu no meio da noite" como **morte** (mesma penalidade
−100) e tenta reabrir o jogo.

> **Onde no código:** `_interromper_episodio`, chamado quando `_verificar_e_focar_janela` falha
> ([fnaf_env.py:664](../src/environment/fnaf_env.py#L664)).

### 9.6 Pausa por F12

Durante o treino, segurar **F12** pausa a IA (útil para mexer no PC sem atrapalhar). Implementado no
`LogCallback` ([train.py:99](../src/agent/train.py#L99)).

<div style="page-break-after: always;"></div>

## 10. Técnicas avançadas (BC, LSTM, currículo)

### 10.1 Behavioral Cloning (BC) e warm-start

> **Analogia:** antes de deixar o aprendiz tentar sozinho (RL), você o deixa **assistir a partidas
> de um humano** e imitar. Ele já começa sabendo "onde olhar".

**Behavioral Cloning** treina a rede para **imitar** jogadas humanas gravadas (aprendizado
supervisionado: "neste estado, o humano fez esta ação"). O resultado pode **inicializar** o RL.

A peça-chave é `transferir_pesos`: ela copia para o modelo de RL **apenas a percepção** (o
`MultimodalExtractor`) — aquece a parte cara (enxergar) e deixa a **estratégia** para o RL aprender
livremente.

```text
BC (imita humano) → pesos da percepção → transferir_pesos → RL parte com percepção pronta
```

> **Onde no código:** [behavioral_cloning.py:149](../src/agent/behavioral_cloning.py#L149). É
> **init, não recompensa**: o RL fica livre para divergir (não fixa o ótimo).

### 10.2 Memória com LSTM (Decisão 7)

Alguns perigos do FNAF **não cabem num único frame**:

- **Foxy** é um *buildup* de minutos (quanto menos você olha a câmera dele, mais ele avança).
- **Freddy** só aparece nas câmeras.

Uma rede comum ("feedforward") só vê o **agora**. Uma **LSTM** (*Long Short-Term Memory*) é uma rede
com **memória**: carrega um resumo do passado de step a step. Isso permite aprender padrões de
**longo alcance** ("faz tempo que não olho a câmera → risco subindo").

```text
FNAF_USAR_LSTM=1  → usa RecurrentPPO (LSTM) em vez de PPO (feedforward)
LSTM pequena: lstm_hidden_size=128, n_lstm_layers=1
```

> **Atenção:** a LSTM exige **masking** correto (`episode_starts`) — ela precisa **zerar a memória**
> no início de cada episódio, senão "lembra" da noite anterior e a avaliação mente. Há utilitários
> (`testar_masking`, `sonda_memoria`) para validar isso.

### 10.3 Curriculum learning

> **Analogia:** você não ensina natação jogando a pessoa no mar. Começa na parte rasa.

**Currículo** = treinar primeiro em versões mais fáceis e aumentar a dificuldade aos poucos. No
problema atual, o agente nunca aprende a Noite 2 porque **morre antes de amostrá-la**. O modo
`continue` (mirando `FNAF_NOITE_DESEJADA=2`) **força** a exposição à Noite 2 reusando a máquina de
reset que já existe — atacando a fome de amostra direto.

<div style="page-break-after: always;"></div>

## 11. As Decisões de design 4–7

O código referencia "Decisões" numeradas — o registro vivo das escolhas de projeto. As que aparecem
diretamente no código:

| Decisão | Conceito que usa | O que é | Onde |
|---|---|---|---|
| **D4 / 4A / 4B** | Visão computacional + potential shaping | Extrair estado da imagem (ameaça por template, energia por OCR, porta por cor) e guiar com Φ | `_atualizar_ameaca`, `_ler_energia`, `_potencial_seguranca` |
| **D5** | Ablação | Testar se a CNN realmente contribui, zerando um ramo da observação na avaliação | `main.py jogar --ablacao imagem\|estados` |
| **D6** | Schedules (γ, LR, ent_coef) | Bundle: γ 0.995→0.997, LR linear 3e-4→3e-5, ent_coef 0.02→0.005 com gate | `EntropiaSchedule`, `linear()` em train.py |
| **D7** | Memória + currículo | LSTM (RecurrentPPO) para Foxy/Freddy; noite no estado; reset new_game/continue | `FNAF_USAR_LSTM`, `decidir_reset` |

> **Sobre D6 (importante para os experimentos):** os três botões (γ, LR, ent_coef) mudaram **juntos**
> porque amostra é cara (rodar 3 runs isolados sairia caro). A regra do projeto: se o bundle
> **piorar**, reverter **um botão por vez**. É a mesma disciplina A/B que os experimentos E1/E2/E3
> seguem — **uma variável por vez**.

> **Atenção:** a `docs/REFERENCIA_HIPERPARAMETROS.md` ainda cita valores **antigos** em alguns
> trechos (γ=0.995, ent_coef=0.01). Os valores **atuais** corretos estão neste guia e no código.

<div style="page-break-after: always;"></div>

## 12. Como ler o treino na prática

### 12.1 `logs/treino.log`

Uma linha por episódio:

```text
pc4 | Ep 109 | Noite 1 | VITORIA | Passos: 737 | Tempo: 8.93 min | Recompensa: 561.1 | Taxa vitória: 21.1%
```

> **Atenção:** a "Taxa vitória" aqui é **cumulativa** (vitórias ÷ episódios do run inteiro). Ela
> **achata por construção** conforme os episódios se acumulam — "parou de subir" no log **não** é
> prova de estagnação. A métrica honesta é a **janela móvel** (rolling-50), que é o que o gate usa.

### 12.2 Tensorboard

Rodar: `venv\Scripts\tensorboard.exe --logdir logs` e abrir http://localhost:6006.

| Tag | O que diz |
|---|---|
| `rollout/ep_rew_mean` | recompensa média por episódio (lembre: muda de escala entre versões) |
| `train/entropy_loss` | `−ent_coef · H`. A entropia crua sai de `H = −entropy_loss / ent_coef` |
| `train/approx_kl` | quanto a política mudou por update (alto = instável) |
| `train/clip_fraction` | fração de updates freados (>0.3 sistemático = `n_epochs` alto demais) |
| `train/explained_variance` | quão bem o crítico prevê o retorno (perto de 1 = bom) |
| `custom/ent_coef` | **(novo)** o coeficiente de entropia atual do schedule |
| `custom/win_rate_50` | **(novo)** taxa de vitória na janela de 50 (a que o gate usa) |

> **As duas métricas `custom/` foram adicionadas nesta sessão** (em `EntropiaSchedule._on_rollout_end`,
> [train.py](../src/agent/train.py)) justamente para parar de reconstruir a entropia "de cabeça" e
> ver o colapso direto.

### 12.3 Regra de ouro

**Meça por taxa de vitória e tempo de sobrevivência, nunca pela recompensa crua.** E sempre salve o
modelo de controle (`modelos/*.zip` + `vecnormalize.pkl`) **antes** de um experimento.

<div style="page-break-after: always;"></div>

## 13. Glossário

| Termo | Em uma linha |
|---|---|
| **Agente** | o programa que decide as ações (a IA) |
| **Ambiente** | o mundo onde o agente age (aqui, o jogo via `FNAFEnv`) |
| **Episódio** | uma partida completa, do reset até vencer/morrer |
| **Step** | uma volta do laço: observar → agir → receber recompensa |
| **Política (π)** | a estratégia: distribuição de probabilidade sobre as ações |
| **Valor (V)** | quanto de recompensa futura se espera de um estado |
| **Ator-Crítico** | treinar a política (ator) e o valor (crítico) juntos |
| **Recompensa** | o sinal de bom/ruim; o agente maximiza a soma dela |
| **Reward shaping** | recompensas densas extras para guiar o aprendizado |
| **Potential-based shaping** | shaping `γ·Φ'−Φ` que guia **sem** mover o ótimo |
| **Reward hacking** | maximizar a recompensa de um jeito que não é o desejado |
| **Retorno (G)** | soma das recompensas futuras descontadas |
| **γ (gamma)** | fator de desconto; define o horizonte efetivo `1/(1−γ)` |
| **Vantagem** | quanto uma ação foi melhor/pior que o esperado pelo crítico |
| **GAE** | estimador suavizado da vantagem |
| **PPO** | o algoritmo de treino (atualiza a política com um "freio") |
| **On-policy** | aprende só com dados da política atual (joga, aprende, descarta) |
| **Rollout** | o lote de experiências coletado antes de um update |
| **n_steps / batch_size / n_epochs** | tamanho do rollout / mini-lote / reusos do lote |
| **Clipping** | o freio do PPO contra mudanças bruscas na política |
| **approx_kl / clip_fraction** | medidores de quanto a política mudou / quanto foi freada |
| **Learning rate** | tamanho do passo ao ajustar os pesos |
| **Entropia (H)** | quão espalhada está a política (alta = explora) |
| **ent_coef** | força do incentivo à exploração na perda |
| **Gate** | condição (vitória ≥20%) que libera o decaimento da entropia |
| **Colapso de entropia** | a política fica determinística cedo demais |
| **Ótimo local** | estratégia estável porém subótima da qual o agente não sai |
| **Convergência prematura** | travar num ótimo local antes de aprender o bom |
| **VecNormalize** | normaliza recompensa (e/ou observação) para estabilizar |
| **CNN** | rede convolucional, especializada em imagens |
| **LSTM** | rede com memória, para padrões de longo alcance |
| **Behavioral Cloning (BC)** | treinar imitando jogadas humanas gravadas |
| **Warm-start** | inicializar o RL com pesos prontos (ex.: a percepção do BC) |
| **Curriculum learning** | treinar do fácil ao difícil |
| **Desync** | divergência entre o estado interno e a tela real; é re-ancorado |
| **Template matching** | achar algo na tela comparando com uma imagem de referência |

<div style="page-break-after: always;"></div>

## 14. Mapa conceito → código

| Conceito | Arquivo : ponto |
|---|---|
| Ambiente, `step`, `reset` | [fnaf_env.py](../src/environment/fnaf_env.py) — `step` (L660), `reset` (L585) |
| Ações (17) | [fnaf_env.py:33](../src/environment/fnaf_env.py#L33) (`ACOES`) |
| Observação (imagem + 11 estados) | [fnaf_env.py:1086](../src/environment/fnaf_env.py#L1086) (`_capturar_observacao`) |
| Recompensa terminal/densa | [fnaf_env.py:1034](../src/environment/fnaf_env.py#L1034) (`_calcular_recompensa`) |
| Potencial Φ (shaping) | [fnaf_env.py:1011](../src/environment/fnaf_env.py#L1011) (`_potencial_seguranca`) |
| γ (fonte única) | [fnaf_env.py:196](../src/environment/fnaf_env.py#L196) (`GAMMA = 0.997`) |
| Rede (CNN + MLP) | [multimodal_policy.py](../src/agent/multimodal_policy.py) (`MultimodalExtractor`) |
| Hiperparâmetros do PPO | [train.py:313](../src/agent/train.py#L313) (`Modelo(...)`) |
| Schedule de entropia (gate) | [train.py:210](../src/agent/train.py#L210) (`EntropiaSchedule`) |
| Logging novo (`custom/...`) | [train.py](../src/agent/train.py) (`_on_rollout_end`) |
| VecNormalize | [train.py:296](../src/agent/train.py#L296) |
| Captura de tela | [capture.py](../src/utils/capture.py) (`GameCapture`, `melhor_janela`) |
| Reset por menu / currículo | [fnaf_env.py:142](../src/environment/fnaf_env.py#L142) (`decidir_reset`) |
| Behavioral Cloning / warm-start | [behavioral_cloning.py:149](../src/agent/behavioral_cloning.py#L149) (`transferir_pesos`) |

---

### Documentos relacionados

- `docs/REFERENCIA_HIPERPARAMETROS.md` — consulta rápida de hiperparâmetros (atenção: alguns valores
  defasados; este guia tem os atuais).
- `docs/GUIA_TENSORBOARD.md` e `docs/MONITORAMENTO_TREINO.md` — leitura dos logs e do tensorboard.
- `docs/AUDITORIA_RECOMPENSA_E_RL.md` e `docs/ALEM_DO_RL.md` — auditoria da recompensa e ideias além
  do RL puro.

*Fim do guia.*
