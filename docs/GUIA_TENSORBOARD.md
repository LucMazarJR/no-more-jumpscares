# Guia do TensorBoard — como ler os gráficos do treino (para quase leigos)

Este guia explica **o que cada gráfico do TensorBoard significa**, em linguagem
simples, e **como olhar para eles** para saber se a IA está aprendendo ou não.

Você não precisa entender matemática para usar este guia. Sempre que aparecer um
termo técnico, ele vem com uma analogia do dia a dia.

> Se quiser se aprofundar nos *hiperparâmetros* (os "botões" que ajustam o
> treino), veja [REFERENCIA_HIPERPARAMETROS.md](REFERENCIA_HIPERPARAMETROS.md).
> Aqui o foco é **ler os resultados**, não ajustá-los.

---

## Índice

1. [O que é o TensorBoard](#1-o-que-é-o-tensorboard)
2. [Como abrir](#2-como-abrir)
3. [A tela do TensorBoard, parte por parte](#3-a-tela-do-tensorboard-parte-por-parte)
4. [Glossário mínimo (leia antes dos gráficos)](#4-glossário-mínimo-leia-antes-dos-gráficos)
5. [Os três grupos de gráficos](#5-os-três-grupos-de-gráficos)
6. [Gráfico por gráfico, explicado](#6-gráfico-por-gráfico-explicado)
7. [Quais olhar primeiro (ordem de importância)](#7-quais-olhar-primeiro-ordem-de-importância)
8. [Estudo de caso real: "o loss explodiu, mas tá tudo bem?"](#8-estudo-de-caso-real-o-loss-explodiu-mas-tá-tudo-bem)
9. [Tabela de sintomas → o que provavelmente é](#9-tabela-de-sintomas--o-que-provavelmente-é)
10. [Glossário rápido](#10-glossário-rápido)

---

## 1. O que é o TensorBoard

É um painel de gráficos que mostra, **enquanto a IA treina**, vários números que
revelam se ela está melhorando. Pense nele como o "painel do carro": velocímetro,
temperatura, combustível. Cada mostrador conta uma parte da história; nenhum
sozinho conta tudo.

Neste projeto a IA aprende por **Reforço (RL)** com um algoritmo chamado **PPO**.
Em uma frase: a IA tenta jogar, recebe uma "pontuação" (recompensa) conforme se
sai bem ou mal, e vai ajustando seu comportamento para conseguir mais pontos.

---

## 2. Como abrir

Com o treino rodando (ou já tendo rodado), no terminal do projeto:

```bash
tensorboard --logdir logs
```

Ele vai imprimir um endereço como `http://localhost:6006/`. Abra no navegador.

> Os dados ficam na pasta `logs/` (dentro dela, uma subpasta por execução, tipo
> `PPO_1`, `PPO_2`...). Essa pasta **não** vai para o git — é histórico local.

---

## 3. A tela do TensorBoard, parte por parte

Quando você abre, na aba **SCALARS** (a principal), cada gráfico tem o mesmo
formato. Entender estes elementos resolve 90% das dúvidas:

### O eixo horizontal (X) = "Step"

É o número de **passos de treino** já vividos pela IA (cada passo = uma ação no
jogo). Quando você lê "120k" no eixo X, significa 120.000 passos. É o "tempo" do
gráfico, mas medido em experiência acumulada, não em horas.

### O eixo vertical (Y) = o valor da métrica

O que está sendo medido (recompensa, erro, etc.). O significado muda conforme o
gráfico — é o que este guia explica.

### Linha clara x linha escura (o "Smoothing")

Você quase sempre verá **duas linhas da mesma cor**:

- **Linha clara (apagada):** os valores **reais**, crus. Costumam ser bem
  "tremidos" (muita variação de um ponto pro outro). Normal em RL.
- **Linha escura (forte):** a versão **suavizada** — uma média que "alisa" o
  tremor para você enxergar a **tendência**. É nela que você deve focar.

Há um controle **"Smoothing"** (geralmente um slider, 0 a 1) na barra lateral.
Quanto maior, mais alisada a linha escura. Para ver tendência, deixe alto (0.9).
Para ver os picos reais, abaixe para 0.

> **Cuidado:** com smoothing alto, picos curtos somem da linha escura. Se quiser
> investigar uma explosão pontual, **baixe o smoothing** para vê-la na linha clara.

### As colunas embaixo (Smoothed / Value / Step / Relative)

Embaixo do gráfico, numa tabelinha, aparece:

| Coluna | O que é |
|---|---|
| **Run** | Qual execução do treino (ex.: `PPO_1`). Útil para comparar runs. |
| **Smoothed** | O valor **suavizado** no ponto onde seu mouse está (ou no fim). |
| **Value** | O valor **real (cru)** naquele ponto. |
| **Step** | Em qual passo está esse ponto. |
| **Relative** | Há quanto tempo (relógio real) aquele ponto foi gravado. Ex.: `8.36 day` = o treino já dura 8 dias e meio. |

### Outras dicas de navegação

- **Passe o mouse** sobre a linha para ver o valor exato em cada step.
- **Caixa de busca** no topo: filtra os gráficos por nome (ex.: digite `loss`).
- **Ícone de tela cheia / expandir** em cada gráfico para olhar de perto.
- **Escala logarítmica (log scale):** um botãozinho no gráfico que comprime
  valores grandes. Útil quando algo explode (ex.: o `loss`) e os picos gigantes
  esmagam o resto do gráfico. Ligue para ver os detalhes da parte "baixa".
- **Comparar runs:** se houver `PPO_1`, `PPO_2`... cada um vira uma cor. Dá para
  ligar/desligar na lista lateral — ótimo para comparar "antes e depois".

---

## 4. Glossário mínimo (leia antes dos gráficos)

Cinco palavras que aparecem o tempo todo:

- **Step (passo):** uma ação tomada no jogo. A unidade básica de tempo do treino.
- **Episódio:** uma partida inteira — do início da noite até morrer, vencer (6 AM)
  ou ser interrompida. Um episódio tem centenas de steps.
- **Rollout (coleta):** o PPO joga por um tanto de steps (aqui, **2048**),
  guarda essa experiência e só então aprende com ela. Cada bloco desses é um
  rollout.
- **Update (atualização):** depois de coletar um rollout, a IA reaproveita
  aqueles dados algumas vezes (aqui, **10 épocas**) para ajustar a rede. Cada
  ajuste é um update.
- **Política (policy) e Valor (value):** a IA tem duas "cabeças":
  - A **política** é quem **decide a ação** ("fecho a porta ou abro a câmera?").
  - O **valor** (também chamado **crítico**) é um "palpiteiro": olha a situação e
    **chuta quantos pontos ainda dá para fazer dali até o fim**. Ele não joga;
    ele avalia. Serve para a política saber se uma ação foi melhor ou pior do que
    o esperado.

---

## 5. Os três grupos de gráficos

Os nomes dos gráficos vêm com um prefixo que indica o "departamento":

| Prefixo | Departamento | Pergunta que responde |
|---|---|---|
| **`rollout/`** | Resultado no jogo | "A IA está jogando melhor?" ⭐ o que mais importa |
| **`train/`** | Saúde do aprendizado | "O ajuste interno da rede está saudável?" |
| **`time/`** | Velocidade/tempo | "Quão rápido o treino anda?" |

A regra de ouro: **`rollout/` diz se está dando certo; `train/` diz por quê.**

---

## 6. Gráfico por gráfico, explicado

Para cada um: **o que é**, **como ler** e **o que é bom/ruim**.

---

### ⭐ `rollout/ep_rew_mean` — recompensa média por episódio

**A métrica mais importante de todas.** É a "pontuação média" que a IA está
tirando por partida (média dos últimos ~100 episódios).

- **Analogia:** sua média de pontos no jogo nas últimas 100 tentativas.
- **Como ler:** olhe a **tendência da linha escura**. Deve **subir** ao longo dos
  steps. Subir = a IA está aprendendo a fazer o que dá pontos (sobreviver mais).
- **Bom:** linha que cresce, mesmo que devagar e tremida.
- **Ruim:** linha plana por muito tempo (não aprende) ou que cresce e depois
  **desaba** (instabilidade — desaprendeu).

> Neste projeto a recompensa vem de **sobreviver** (bônus a cada hora da noite) e
> de **vencer** (+500), menos **penalidades** (ações inúteis, gastar energia à
> toa) e a **morte** (−100). Então `ep_rew_mean` subindo = noites mais longas.

---

### ⭐ `rollout/ep_len_mean` — duração média do episódio (em steps)

Quantos passos, em média, a IA dura por partida.

- **Analogia:** quanto tempo você sobrevive por rodada.
- **Como ler:** neste jogo, **durar mais é melhor** (sobreviveu mais da noite).
  Deve subir junto com a recompensa.
- **Bom:** sobe e se aproxima do tamanho de uma noite completa (~700 steps).
- **Ruim:** fica baixo e travado (morre sempre cedo, no mesmo ponto) — sinal de
  que empacou numa estratégia ruim.

> Olhar `ep_rew_mean` e `ep_len_mean` **juntos** conta quase toda a história.

---

### ⭐ `train/explained_variance` — o quão bom é o "palpiteiro" (crítico)

Mede se o **valor/crítico** acerta seus palpites sobre os pontos futuros.

- **Escala:** vai até **1**.
  - **Perto de 1:** o crítico explica quase tudo — ótimo.
  - **Perto de 0:** o crítico não explica nada (chuta tão bem quanto dar a média).
  - **Negativo:** o crítico está pior que um chute burro — sinal de divergência.
- **Como ler:** queremos a linha **alta e estável** (acima de ~0.8 é muito bom).
- **Detalhe importante:** essa métrica é **"invariante de escala"** — ela mede o
  acerto *relativo*, não importa se as recompensas são grandes ou pequenas. Por
  isso ela é o melhor **termômetro de saúde** do crítico. Se ela está alta, o
  crítico está bem **mesmo que o `loss` pareça gigante** (veja o estudo de caso).

---

### `train/loss` — a "nota de erro" total que o treino tenta reduzir

É a soma combinada de três custos (política + valor − exploração) que o
otimizador tenta **minimizar**.

- **Analogia:** uma "nota de erro geral". O treino empurra esse número para baixo.
- **Como ler — atenção, esta engana:** um `loss` **menor não é sempre melhor**, e
  um `loss` **subindo não significa necessariamente que piorou**. Por quê? Porque
  ele é dominado pelo **erro do crítico**, que cresce com o **tamanho** das
  recompensas. Recompensas grandes (ex.: ±500) fazem o erro, ao ser elevado ao
  quadrado, virar números enormes — mesmo quando o crítico está acertando bem
  (veja `explained_variance`).
- **Use o `loss` para:** ver **estabilidade**. Uma subida suave no começo e queda
  é normal. **Picos enormes e repetidos** indicam instabilidade que vale
  investigar — mas **sempre cruze com `explained_variance`** antes de concluir.

---

### `train/value_loss` — erro do crítico (sozinho)

A parte do `loss` que mede só o erro do palpiteiro: a diferença (ao quadrado)
entre o que ele previu e o que realmente aconteceu.

- **Como ler:** se é **ele** que explode no `train/loss`, o problema é de
  **escala de recompensa** (números terminais grandes demais), não
  necessariamente de aprendizado. A solução típica é reduzir/normalizar as
  recompensas.
- **Bom:** estável ou caindo. **Ruim:** picos gigantes recorrentes.

---

### `train/policy_gradient_loss` — o "empurrão" na política

A parte do `loss` ligada a ajustar **quem decide as ações**.

- **Como ler:** costuma ser um número **pequeno e negativo**, bem tremido. Não
  tente lê-lo sozinho — ele só faz sentido junto de `approx_kl` e `clip_fraction`
  (abaixo). Por si só, não diz "bom" ou "ruim".

---

### `train/entropy_loss` — o quanto a IA ainda "experimenta"

**Entropia** = o quão **aleatória/exploradora** a política está. Alta entropia =
a IA ainda testa ações variadas; baixa = ela já "decidiu" e quase sempre faz a
mesma coisa.

- **Detalhe:** no TensorBoard esse valor aparece **negativo** (é o negativo da
  entropia). Não se assuste com o sinal; olhe o **movimento**.
- **Como ler:** é **normal** a exploração diminuir devagar conforme a IA aprende
  (a linha sobe em direção a zero). O que preocupa é ela **despencar muito cedo**
  — a IA "fechou" numa estratégia antes de explorar o suficiente e pode ter
  empacado num hábito ruim.
- **Ligado ao botão `ent_coef`** (ver [REFERENCIA_HIPERPARAMETROS.md](REFERENCIA_HIPERPARAMETROS.md)),
  que controla o quanto incentivamos exploração.

---

### `train/approx_kl` — o tamanho do "salto" da política a cada update

Mede **o quanto a política mudou** de um update para o outro.

- **Analogia:** o tamanho do passo que você dá. Passinhos = estável; saltões =
  arriscado, pode cair.
- **Como ler:** queremos valores **pequenos** (tipicamente abaixo de ~0.02).
- **Bom:** baixo e estável. **Ruim:** picos altos — a IA está mudando bruscamente,
  o que costuma andar junto de instabilidade no `loss`.

---

### `train/clip_fraction` — quanto o PPO precisou "segurar" a política

O PPO tem um freio embutido (o *clipping*) que impede a política de mudar demais
de uma vez. Esta métrica é a **fração das amostras que bateram nesse freio**.

- **Como ler:** algo em torno de **0.1 a 0.2** é normal e saudável.
- **Ruim:** **acima de ~0.3 de forma persistente** = a IA está "querendo" mudar
  mais do que o freio permite. Normalmente pede um **learning rate menor** ou
  menos épocas por update.

---

### `train/clip_range` — o valor do freio do PPO

Apenas mostra o tamanho do freio de clipping (padrão **0.2** neste projeto).

- **Como ler:** é uma **linha reta** (constante), a menos que se use um *schedule*
  que o reduza ao longo do treino. É informativo — confirma a configuração.

---

### `train/learning_rate` — o tamanho do passo de ajuste da rede

O quão "agressivamente" a rede é ajustada a cada update.

- **Como ler:** aqui é **constante** (`0.0003`), então será uma **linha reta**. Se
  algum dia for usado um *schedule* decrescente, este gráfico mostra a descida —
  é a forma de **confirmar** que o schedule está funcionando.

---

### `train/n_updates` — total de ajustes já feitos

Contagem acumulada de updates de gradiente.

- **Como ler:** sobe em linha reta. Puramente informativo (mede progresso
  interno, não qualidade).

---

### `time/fps` — velocidade do treino (passos por segundo)

Quantos steps a IA consegue processar por segundo.

- **Como ler:** neste projeto será **baixíssimo** (na casa de 1–3), e isso é
  **esperado**: cada passo envolve capturar a tela, mover o mouse, esperar
  animações do jogo real. Não é bug — é o custo de jogar um jogo de verdade em
  vez de uma simulação rápida.
- Serve para estimar quanto tempo faltará: `passos_que_faltam / fps`.

### `time/iterations`, `time/time_elapsed`, `time/total_timesteps`

Contadores de apoio:
- **iterations:** quantos rollouts já foram coletados.
- **time_elapsed:** segundos de relógio desde o início.
- **total_timesteps:** total de steps coletados (bate com o eixo X).

Todos informativos — úteis para saber "onde estamos", não para julgar qualidade.

---

## 7. Quais olhar primeiro (ordem de importância)

Se você só tem 30 segundos, olhe nesta ordem:

1. **`rollout/ep_rew_mean`** — está subindo? Então está aprendendo. (o veredito)
2. **`rollout/ep_len_mean`** — está durando mais? Confirma o item 1.
3. **`train/explained_variance`** — o crítico está saudável (alto)?
4. **`train/loss` / `train/value_loss`** — só para investigar **estabilidade**, e
   sempre cruzando com o item 3.
5. **`train/approx_kl` e `train/clip_fraction`** — se houver instabilidade, dizem
   se a causa é a política saltando demais.

---

## 8. Estudo de caso real: "o loss explodiu, mas tá tudo bem?"

Aconteceu neste projeto, e é o melhor exemplo de por que **nunca se lê um gráfico
sozinho**.

**O que víamos:**
- `train/loss`: caía bonito até ~60k steps e depois **explodia** em picos de 300+,
  repetidamente. Assustador.
- `train/explained_variance`: subia até ~0.9 e **continuava alto (0.6–0.9)** mesmo
  durante as explosões do loss.

**Como interpretar:** se o crítico estivesse realmente quebrando, o
`explained_variance` **despencaria** para perto de 0. Como ele se manteve **alto**,
o crítico estava de fato **acertando bem**. Logo, a explosão do `loss` **não era o
modelo quebrando** — era só o **erro ao quadrado** ficando gigante por causa das
recompensas terminais enormes daquela versão (−500 / +1000). O número do `loss`
estava **inflado pela escala**, não pela qualidade.

**Lição:** o `loss` sozinho enganaria você a jogar fora um modelo que estava
aprendendo. O `explained_variance` contou a verdade. **Cruze sempre.**

---

## 9. Tabela de sintomas → o que provavelmente é

| O que você vê | Provável significado | Para onde olhar a seguir |
|---|---|---|
| `ep_rew_mean` subindo | Está aprendendo 🎉 | Continue treinando |
| `ep_rew_mean` plano por 100+ eps | Empacou (ótimo local) | `entropy_loss` (explorou pouco?) |
| `ep_rew_mean` sobe e depois desaba | Instabilidade / desaprendeu | `approx_kl`, `learning_rate` |
| `ep_len_mean` baixo e travado | Morre sempre no mesmo ponto | Estratégia ruim; rever recompensa |
| `loss` explode **mas** `explained_variance` alto | Escala de recompensa, não quebra | Normalizar/reduzir recompensa |
| `explained_variance` cai para ~0 ou negativo | Crítico divergindo de verdade | `learning_rate`, `value_loss` |
| `approx_kl` com picos altos | Política saltando demais | Reduzir `learning_rate`/épocas |
| `clip_fraction` > 0.3 sempre | Updates grandes demais | Reduzir `learning_rate` |
| `entropy_loss` despenca cedo | Parou de explorar cedo demais | Aumentar `ent_coef` |
| `fps` muito baixo | Esperado neste projeto | Nada — é o custo do jogo real |

---

## 10. Glossário rápido

- **Agente / IA:** o programa que aprende a jogar.
- **Recompensa (reward):** a pontuação que guia o aprendizado.
- **Política (policy):** a parte que decide as ações.
- **Valor / Crítico (value):** a parte que estima os pontos futuros (o palpiteiro).
- **Step:** uma ação no jogo (unidade do eixo X).
- **Episódio:** uma partida inteira.
- **Rollout:** um bloco de experiência coletado antes de aprender (2048 steps).
- **Update / época:** cada ajuste da rede sobre os dados coletados.
- **Loss:** a "nota de erro" que o treino minimiza (pode enganar — ver §8).
- **Entropia:** o quanto a política ainda explora ações variadas.
- **KL:** o tamanho da mudança da política entre updates.
- **Clipping:** o freio do PPO que evita mudanças bruscas.
- **Smoothing:** o alisamento visual das linhas no TensorBoard.
- **Explained variance:** o termômetro de acerto do crítico (quanto mais perto de
  1, melhor).

---

### Veja também

- [REFERENCIA_HIPERPARAMETROS.md](REFERENCIA_HIPERPARAMETROS.md) — os "botões" do
  treino (learning rate, gamma, ent_coef...) e quando mexer em cada um.
- [ALTERACOES_COMPLETAS.md](ALTERACOES_COMPLETAS.md) — histórico técnico das
  mudanças no ambiente.
- [MELHORIAS_PROJETO_ATUAL.md](MELHORIAS_PROJETO_ATUAL.md) — próximos passos.
