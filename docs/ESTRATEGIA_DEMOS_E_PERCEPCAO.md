# Nova estratégia: medir, enxergar melhor e aprender com cada amostra — porquês e runbook

> **Data:** 28/09/2026 · **Status:**
> - Fase 0 (limpeza e runs com nome): feita.
> - Fase 1 (medições): ferramentas prontas, falta rodar com o jogo.
> - Fases 2 e 3: propostas; os números da Fase 1 decidem.
>
> **Problema que ataca:** em 4 runs (~750k steps, ~8 dias de máquina) o RL nunca levou a Noite 1
> acima do nível que o BC entregou, e nunca aprendeu a Noite 2. O que ensinou o agente foram
> 8 demos humanas.
>
> **Relação com os outros docs:**
> - O [ESTADO_ATUAL.md](ESTADO_ATUAL.md) tem os números e as regras de decisão. Este doc explica
>   as estratégias e como executá-las.
> - O [PACOTE_BC_ENTROPIA.md](PACOTE_BC_ENTROPIA.md) continua descrevendo os mecanismos em uso
>   (BC, termostato, currículo, telemetria).
>
> 📖 Termos como *on-policy*, *replay*, *DQfD*, *covariate shift*, *HG-DAgger*, *mediana temporal* e
> *phase correlation* estão no **[Glossário](#6-glossário--os-termos-técnicos-usados-neste-doc)**.

---

## 1. O diagnóstico em uma página

Três fatos medidos nos logs (detalhes e tabelas no [ESTADO_ATUAL.md](ESTADO_ATUAL.md) §2):

1. **O BC ensina e o RL não.** A vitória na N1 de cada run ficou no nível em que o BC a deixou:
   57% na run 1, 60% na run 2 (que depois esqueceu tudo), 51,6% na run 4, **plana** do começo ao fim.
   O PPO fez ~46 atualizações de política em 47 h. Cada lote de ~7 noites foi usado em 4 épocas e
   **jogado fora**. Com amostra a ~0,9 s, um algoritmo on-policy é o mais caro possível.
2. **A Noite 2 é problema de informação, não de quantidade de treino.**
   - A run 2 teve 653 episódios de N2 e venceu 3. Ela oscilou entre dois modos de falha: morrer
     cedo com 40% de bateria (defendeu pouco) ou ter apagão (defendeu demais). Nunca achou o meio.
   - Isso é o que acontece quando o agente só consegue ajustar o QUANTO defende, e não QUANDO.
   - Você vence a N2 com 1–2% de bateria: a margem exige defesa **seletiva**, e defesa seletiva
     exige **ver** a ameaça.
   - Hoje Foxy e Freddy não estão na observação. A imagem é a janela inteira reduzida a 84×84 em
     cinza, com estática e luzes piscando.
3. **As demos são poucas justamente onde importa.**
   - Há 2 demos de N2.
   - Portas aparecem 2 a 8 vezes por noite (~36 no total), e são a ação mais crítica.
   - As demos foram gravadas a ~0,33 s por registro, e o agente age a ~0,9 s por step (hipótese de
     descasamento, medida no M3).

**Conclusão:** a próxima fase muda três coisas, **nesta ordem**: (a) medir antes de mexer,
(b) enxergar melhor, (c) aprender com cada amostra e com o humano de forma contínua.

---

## 2. As estratégias (problema → solução → onde)

### 2.1 Medir antes de mudar (Fase 1)

**Problema:** três suposições centrais nunca foram testadas. Mudar a arquitetura sem saber a
resposta delas é apostar dias de máquina no escuro:
- o RL agregou algo ao BC?
- a CNN contribui?
- o clone age tanto quanto o humano?

**Solução:** quatro medições baratas, com regras de decisão escritas ANTES de medir.

| # | Mede | Ferramenta |
|---|---|---|
| M1 | BC puro, sem RL, na mesma dinâmica de reset da run 4 | `main.py jogar --modelo modelos/fnaf_bc.zip --estocastico --noite 1 --episodios 20` |
| M2 | Se a imagem contribui (ablação) | M1 + `--ablacao imagem` |
| M3 | Ações/min do clone contra as do humano | `scripts/taxa_acao.py` |
| M4 | Latência, estática e piscar na captura | `python -m src.utils.sonda_captura` |

**Onde:** `main.py` (`modo_jogar`, que grava em `logs/analise/avaliacoes.log`),
`scripts/taxa_acao.py` e `src/utils/sonda_captura.py`.

### 2.2 Rajada de frames: estática e luzes piscando

**Problema:** o env tira **1 foto por step**. Se ela cai num frame de estática ou no instante em
que a luz piscou, a leitura sai errada. Hoje isso é absorvido por debounce ENTRE steps: são
precisas 2 a 4 leituras seguidas, o que custa 2 a 4 segundos de reação num jogo em que segundos
matam.

**Solução:** capturar uma **rajada** de k=3–5 frames em ~60–120 ms e combinar por região:
- **Estática** é ruído aleatório por frame, enquanto o sprite do animatrônico é igual em todos.
  A **mediana temporal** dos k frames apaga o ruído e mantém o sprite.
- **Luz piscando** alterna frames acesos e apagados. Para as regiões de porta e corredor, usar o
  **frame mais claro** da rajada (ou o maior score do detector). Se a luz esteve acesa em algum
  dos k frames, a leitura enxerga.
- A decisão fica por step, sem esperar o próximo. O debounce passa a ser rede de segurança, e não
  o filtro principal.

A sonda (M4) mede o que isso compra antes de qualquer mudança no env:
- latência do `mss` por frame (a rajada custa k × latência em todo step);
- quantos frames **distintos** o jogo entrega por segundo (se duas capturas seguidas saem iguais,
  k maior não adianta);
- o ruído cru contra a mediana de 3 e de 5;
- a fração de frames escuros por região.

Num teste sintético, a mediana de 5 cortou ~43% do ruído.

**Onde:** `capturar_rajada()` em `src/utils/capture.py`, usada por `_capturar_janela` em
`src/environment/fnaf_env.py`, atrás de uma constante desligada por padrão. Assim M1 e M2 medem o
sistema atual sem contaminação. Se a latência do `mss` não couber, a alternativa é `dxcam` ou
`bettercam` (Desktop Duplication API, bem mais rápidos).

### 2.3 Movimentos súbitos: câmera perturbada e memória por câmera

**Problema:** no FNAF os animatrônicos **teleportam** entre câmeras entre um frame e outro. Quando
um deles se move na câmera que você está olhando, o feed dá um surto de estática (a confirmar com a
sonda). Hoje isso é só ruído para a CNN, e a observação não diz "algo mudou aqui desde a última vez
que olhei".

**Solução em duas peças, ambas autocalibradas (sem templates manuais, como você prefere):**
- **Estado "câmera perturbada":** a energia de alta frequência no viewport da câmera sobe no surto
  de estática. Vira um estado em [0,1] e transforma o incômodo em **sinal**: um surto indica que algo
  se moveu nesta câmera agora.
- **Memória por câmera:** guardar a última vista limpa (pós-mediana) de cada câmera. Ao voltar
  nela, calcular um escore de mudança entre a vista atual e a guardada.
  - As câmeras do FNAF 1 fazem **pan** horizontal, então antes é preciso alinhar as duas imagens
    com `cv2.phaseCorrelate`, que estima o deslocamento.
  - O escore responde "o Bonnie saiu do palco?" ou "a cortina da Pirate Cove abriu?" sem que
    ninguém desenhe um template.

**Onde:** módulo novo de percepção em `src/environment/` (funções puras, testáveis offline com
fixtures em `debug/`), consumido por `_capturar_observacao`.

### 2.4 Observação redesenhada

**Problema:** a janela inteira em 84×84 mistura o que importa (câmera, vãos das portas) com
painel, bordas e HUD. Um animatrônico numa câmera vira meia dúzia de pixels.

**Solução:**
- **Recortes em resolução maior** no lugar da janela inteira: o viewport da câmera (quando aberta)
  e as faixas das duas portas.
- **Canais:** o frame limpo (2.2) mais o canal de diferença contra a última vista da mesma câmera (2.3).
- **2 frames empilhados** (frame stacking), para dar noção de movimento curto.
- **Estados novos:** câmera perturbada e escore de mudança das câmeras que decidem o jogo
  (1C Foxy, 2A/2B oeste, 4A/4B leste).

**Onde:** `_capturar_observacao` e `observation_space` em `fnaf_env.py`. O extractor em
`multimodal_policy.py` já deriva as dimensões do espaço. **Muda o shape**, então exige treino do zero
e entra **junto** com a 2.5: uma única rodada paga as duas mudanças.

### 2.5 Aprender com cada amostra: off-policy com demos permanentes (estilo DQfD)

**Problema:**
- O PPO é **on-policy**: só aprende com dados da política atual, usa cada lote 4 vezes e descarta.
  ~750k transições reais já foram para o lixo.
- As demos entram só como ponto de partida (init), e o RL as corrói. O esquecimento da run 2 é o
  exemplo, e a âncora BC foi um remendo para isso.

**Solução — Deep Q-learning from Demonstrations (DQfD):**
- **Tudo vai para um replay buffer em disco** (`dados/transicoes/`): cada transição do agente e cada
  registro humano. Nada é descartado, e o que foi coletado numa run serve para a próxima.
- **As demos ficam no buffer para sempre.** Nos registros humanos, uma **perda de margem** empurra o
  Q da ação humana acima das outras. À medida que o buffer do agente cresce, o peso relativo das
  demos cai naturalmente, e o agente pode **superar** um demonstrador imperfeito.
- **Peças padrão:**
  - Double DQN + Dueling (estabilidade);
  - retornos **n-step** (n≈10–20; a vitória é esparsa e está a ~600 steps);
  - replay **priorizado**, com bônus para as transições humanas.
- **O treino roda em paralelo ao jogo**, numa thread. A GPU fica ociosa nos ~0,9 s de cada step
  esperando o jogo, então dá para fazer ≥4 atualizações por step coletado (razão updates/dados ≥ 4),
  com regularização. Isso é ordens de grandeza mais que as ~46 atualizações do PPO em 47 h.
- **Ações discretas (17)** casam naturalmente com Q-learning.
- **Memória:** os relógios (tempo sem câmera, idade da informação) já estão no estado e o frame
  stacking cobre o movimento curto, então **começa feedforward**. A versão recorrente (R2D2/R2D3) só
  entra se a sonda de memória mostrar necessidade.

**Onde:** `src/agent/dqfd.py` (novo) e `main.py treino --algo dqfd`. O caminho do PPO continua
intacto. Reaproveita `FNAFEnv`, `MultimodalExtractor` e a recompensa (o shaping potential-based
funciona com Q-learning). A recompensa do buffer usa **escala fixa** (ex.: ÷100) em vez do
VecNormalize: normalização móvel muda o significado das recompensas já guardadas.

### 2.6 Correções ao vivo no lugar de noites inteiras (HG-DAgger)

**Problema:**
- O BC aprende só os estados que **o humano** visita. Quando o agente erra e cai num estado que o
  humano nunca viu, o clone não sabe o que fazer (*covariate shift*).
- Jogar noites inteiras é caro e exige habilidade.
- Faltam justamente demos de N2 e de portas.

**Solução — HG-DAgger (human-gated DAgger):**
- O **agente joga** e você assiste. Quando vir que ele vai errar, aperta as teclas do gravador e
  assume o controle até a situação ficar segura.
- **Só os trechos em que você assumiu viram rótulo**, e entram no mesmo buffer com a marca
  "intervenção".
- Exige bem menos habilidade que jogar a noite toda: basta reconhecer o perigo. E a sua habilidade
  basta: as 8 demos atuais são todas vitórias, inclusive N2 e N3.
- Mira exatamente onde o agente falha. Cada intervenção vale mais que uma noite inteira de demo.

**Onde:** um modo "jogar assistido" que junta o `modo_jogar` do `main.py` com a fila de teclas do
`gravar_gameplay.py` (o executor já é o próprio env, então a coreografia é idêntica). Tudo gravado
**no tick do agente**, o que resolve o descasamento da M3. Prioridade: N2 e fechamentos de porta.

### 2.7 O que NÃO muda, e o que foi descartado

**Continua igual:**
- o ambiente (captura, atuação, reset, reabertura do jogo), que está confiável (0,2 desync/ep na run 4);
- a recompensa e o shaping;
- a telemetria de causas;
- o currículo por noite;
- a regra de uma variável por vez.

**Descartado ou adiado:**
- **A run 5 como caminho principal:** testa "mais amostra de N2", premissa que as runs 1 e 2 já
  enfraquecem. Pode rodar no PC-LUCIANO como baseline enquanto o resto é construído.
- **O simulador headless** continua descartado. Só reabre se a M2 mostrar que a imagem não
  contribui, porque aí a política vive dos estados e o simulador transfere.
- **Templates manuais para Foxy/Freddy:** último recurso. As peças 2.2–2.4 tentam o mesmo
  aprendendo.

---

## 3. Runbook

### Etapa A — medições (Fase 1, jogo aberto)

```
venv\Scripts\python main.py jogar --modelo modelos\fnaf_bc.zip --estocastico --noite 1 --episodios 20
venv\Scripts\python main.py jogar --modelo modelos\fnaf_bc.zip --estocastico --episodios 20 --ablacao imagem
venv\Scripts\python scripts\taxa_acao.py
venv\Scripts\python -m src.utils.sonda_captura --rotulo escritorio
venv\Scripts\python -m src.utils.sonda_captura --rotulo cam_1c        (com a CAM 1C aberta)
venv\Scripts\python -m src.utils.sonda_captura --rotulo cam_2a        (com a CAM 2A aberta)
venv\Scripts\python -m src.utils.sonda_captura --rotulo luz_esq --segurar-luz esq
```

- M1 leva ~3,5 h e M2 ~3 h; rode em noites separadas.
- O resumo por noite sai no terminal e em `logs/analise/avaliacoes.log`.
- A sonda grava `debug/sonda_<rotulo>.json` e imagens para comparar a olho: frame cru, mediana e
  máximo.

### Etapa B — decidir (regras registradas antes de medir)

| Resultado | Decisão |
|---|---|
| M1: N1 ≥ 40% | O RL não agregou nada. Seguir para a 2.5 (DQfD). |
| M1: N1 < 30% | O RL da run 4 agregou algo. Reavaliar antes de trocar o algoritmo. |
| M2: queda < 10 p.p. | A imagem não contribui. As 2.2–2.4 viram prioridade; o simulador pode ser reaberto. |
| M3: agente/humano < 0,6 | Descasamento confirmado. Toda gravação nova usa o tick do agente (2.6). |
| M4: rajada de 5 ≤ ~120 ms e mediana corta ≥ 30% do ruído | A rajada vale. Implementar a 2.2. |

### Etapa C — percepção (2.2, depois 2.3 e 2.4)

1. Implementar a rajada atrás da constante, com testes offline em fixtures de `debug/`.
2. A/B curto com o `jogar` sobre o **mesmo** BC, com a rajada desligada e ligada (~20 eps cada).
   Muda só a percepção, sem treino.
3. Implementar 2.3 e 2.4 e validar os estados novos ao vivo com o `monitor`.

### Etapa D — dados humanos no formato novo

A observação da 2.4 muda o shape, então as 8 demos antigas (frames 84×84) não servem para ela.
- Regravar ~8–10 noites no tick do agente, priorizando N2.
- Depois, sessões curtas de correção ao vivo (2.6).

### Etapa E — DQfD

1. **Offline primeiro:** treinar só com as demos (pré-treino do DQfD). O critério é igualar o BC
   em acurácia na validação. Se não igualar, há bug; não vá para o jogo.
2. **Online:** `main.py treino --novo --algo dqfd --nome dqfd_percepcao`, com critério de aborto
   registrado no ESTADO_ATUAL antes de começar.

---

## 4. O que observar e quando agir

**Marcos esperados da linha nova** (hipóteses a confirmar e registrar no ESTADO_ATUAL antes da run):

| Momento | Esperado |
|---|---|
| Fim do pré-treino offline | Acurácia na validação ≈ BC (top-1 e macro-recall no nível do clone de referência) |
| ~20k steps online | N1 ≥ nível do BC puro medido na M1 (não pode começar pior que o BC) |
| ~100k | N1 **subindo** acima do BC (o que o PPO nunca fez) e sobrevivência mediana na N2 > 290 s |
| ~200k | Primeiras vitórias **geridas** na N2 (não por apagão) |

**Saúde:**
- a TD loss estável;
- os Q-valores dentro da faixa dos retornos possíveis (superestimação sistemática é o modo de falha
  clássico);
- a fração de demos no lote caindo de forma suave;
- o `desyncs.log` limpo (a percepção nova não pode piorar a atuação).

**Aborto:** N1 abaixo de 30% por uma janela cheia, ou Q-valores divergindo. Parar, reportar e
voltar à etapa offline.

---

## 5. Riscos aceitos e suas redes de segurança

1. **DQfD é implementação própria (fora do SB3).** Rede de segurança: validação offline com as
   demos antes do jogo, e o caminho do PPO intacto para comparar.
2. **Q-learning é menos estável que o PPO.** Double DQN, rede-alvo, n-step, recompensa em escala
   fixa, regularização e aborto por divergência de Q.
3. **Demos imperfeitas ensinam erros.** A margem vale só nas transições humanas, e o peso delas
   cai conforme o agente acumula dados próprios. O DQfD pode superar o demonstrador, o BC puro não.
4. **A rajada custa latência em todo step.** Medida pela M4 antes, com k escolhido para caber em
   ~120 ms e uma constante para desligar.
5. **O pan das câmeras atrapalha a comparação com a última vista.** Alinhamento por
   `phaseCorrelate`. Se falhar, fica só o estado de câmera perturbada, que não depende do pan.
6. **Trocar o shape joga fora as demos antigas.** É custo aceito: ~1,5 h de regravação, que já
   seria necessário para ter mais N2 e portas.
7. **Duas mudanças grandes no mesmo treino (percepção + algoritmo).** Isso fere o "uma variável por
   vez", mas os dois obrigam treino do zero. A mitigação é validar cada peça isoladamente antes
   (A/B de percepção com o `jogar` sobre o mesmo BC; DQfD offline contra o BC).

---

## 6. Glossário — os termos técnicos usados neste doc

Em linguagem direta. Para a teoria do zero, veja o
[GUIA_CONCEITOS_E_FUNCIONAMENTO.md](GUIA_CONCEITOS_E_FUNCIONAMENTO.md).

**On-policy / off-policy** — um algoritmo *on-policy* (como o PPO) só aprende com dados gerados pela
política atual: mudou a política, os dados antigos não servem mais. Um *off-policy* (como o DQN)
aprende com dados de qualquer origem, inclusive de políticas antigas e de humanos. Quando cada
amostra custa quase 1 segundo de jogo real, poder reusar tudo é a diferença central.

**Replay buffer** — o "arquivo" de transições (observação, ação, recompensa, próxima observação)
de onde o algoritmo sorteia lotes para treinar. Cada transição é reusada muitas vezes. Aqui ele fica
em disco, para sobreviver entre runs.

**Q-valor / DQN** — Q(s, a) é a estimativa de "quanto de recompensa futura eu ganho se fizer a ação
*a* agora e jogar bem depois". O DQN aprende essa função com uma rede neural e age escolhendo a ação
de maior Q.

**Double DQN / Dueling** — duas correções padrão do DQN. A *Double* separa quem escolhe a ação de
quem avalia, o que reduz a superestimação dos Q-valores. A *Dueling* separa "quão bom é este estado"
de "quanto cada ação acrescenta", o que ajuda quando muitas ações dão no mesmo, como o "nada" em
~80% do tempo.

**Retorno n-step** — em vez de olhar 1 passo à frente e confiar na estimativa dali, soma as
recompensas reais dos próximos *n* passos. Propaga mais rápido o efeito de eventos raros e distantes,
como vencer a noite.

**Replay priorizado** — sorteia com mais frequência as transições em que a rede mais erra, e dá um
bônus às humanas. O treino gasta tempo onde há o que aprender.

**DQfD (Deep Q-learning from Demonstrations)** — DQN que (1) pré-treina só com as demos e (2) mantém
as demos no replay para sempre, com uma **perda de margem**: nas transições humanas, o Q da ação que
o humano escolheu tem que ficar acima do Q das outras por uma margem. Resultado: começa imitando e
continua podendo melhorar. Referência: Hester et al., 2018. A versão recorrente para observação
parcial é o R2D3 (Paine et al., 2019).

**Razão updates/dados (UTD)** — quantas atualizações de rede se faz por transição nova coletada. O
PPO aqui fazia ~0,001 por step; um off-policy em thread paralela faz ≥4. Razões altas exigem
regularização para não decorar o buffer.

**Covariate shift** — o BC aprende nos estados que o humano visita. O agente, ao errar, cai em
estados que o humano nunca viu, e ali o clone não tem o que imitar. Os erros se acumulam.

**DAgger / HG-DAgger** — o DAgger corrige o covariate shift pedindo ao humano o rótulo certo nos
estados que o **agente** visita. No *human-gated* (HG-DAgger, Kelly et al., 2019), o humano só
intervém quando julga necessário, assumindo o controle. Só esses trechos viram rótulo. É mais
natural e mais barato que rotular tudo.

**Mediana temporal / rajada** — capturar vários frames seguidos (a rajada) e, pixel a pixel, ficar
com o valor do meio. Ruído que muda a cada frame, como a estática, some. O que é constante, como o
sprite, fica.

**Phase correlation (`cv2.phaseCorrelate`)** — técnica que estima, pela transformada de Fourier, o
deslocamento entre duas imagens quase iguais. Serve para "desfazer" o pan da câmera antes de comparar
a vista atual com a anterior.

**Frame stacking** — empilhar os últimos k frames como canais da imagem, para a rede ver movimento
sem precisar de memória recorrente.

**Ablação** — desligar uma parte do sistema (zerar a imagem, por exemplo) e medir quanto o
desempenho cai. Se não cai, aquela parte não estava sendo usada.
