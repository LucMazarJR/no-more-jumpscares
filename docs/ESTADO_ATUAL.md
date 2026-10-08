# Estado atual do projeto (out/2026)

> **Único doc de "onde estamos".** O [CLAUDE.md](../CLAUDE.md) guarda só o que é permanente;
> os docs em [historico/](historico/) são retratos de época. Atualize ESTE arquivo quando uma
> medição mudar o quadro, com o número que justifica a mudança.
>
> **Regra de ouro:** não confie em número de doc sem conferir no log. Refutar uma afirmação daqui
> com dados é resultado, não problema. Já aconteceu de novo nesta fase (ver §3).

## 1. Resumo

1. **O que funciona:** o ambiente (captura, atuação, detecção, reset) e o **Behavioral Cloning**.
   Oito demos humanas (~72 min) levam a Noite 1 a ~43% de vitória (medido na Fase 1).
2. **O que não funciona:** o **RL (PPO) por cima do BC**.
   - Até 190k steps ele levou a N1 de 43% para 55% (p = 0,29), sem ganho em vitórias geridas.
   - **Continuar treinando piorou o agente:** aos 340k as vitórias geridas caíram de 31/60 para 3/33
     (p = 0,00003). Ele aprendeu a gastar energia para durar mais na N2 e perdeu a habilidade da N1.
3. **A parede é a Noite 2:** 1 vitória em ~56 episódios de avaliação, somando todos os modelos.
   Tirar a imagem da CNN não muda a vitória, então a imagem atual não carrega o que importa.
4. **Decisão (out/2026):** parar de investir no PPO. Seguir para percepção robusta (Fase 2) e
   aprendizado off-policy com demos permanentes (Fase 3). Detalhes em
   [ESTRATEGIA_DEMOS_E_PERCEPCAO.md](ESTRATEGIA_DEMOS_E_PERCEPCAO.md).

## 2. As runs de treino, medidas nos logs

Fontes: `logs/analise/historico/<run>_detalhado.log` e `logs/tensorboard/<run>/`.
Steps a ~0,9 s cada.

| Run (pasta do TB) | Datas | Algoritmo + init | Steps | Noite 1 (no treino) | Noite 2 (no treino) |
|---|---|---|---|---|---|
| `_pre_pacote/*` | 20/jun–2/jul | PPO/RPPO sem BC | até 400k | — | — |
| `2026-07-03_run1_ppo_bc` | 3–5/jul | PPO + BC feedforward | 160k | 57% logo no início (BC); promovida à N2 com ~28k | **0%** por ~120k steps |
| `2026-07-06_run2_ppo_bc_curriculo` | 6–14/jul | PPO + BC ff + currículo | 231k | 60% no início → **0–8%** depois da promoção (esquecimento) | **3/653 (0,5%)** |
| `2026-07-16_run3_rppo_bc_parcial` | 16–22/jul | RecurrentPPO + BC ff (só o extractor transfere) | 175k | 0–13% (cold-start, H 2,65→2,07) | 15 eps |
| `2026-08-05_run4_rppo_bc_recorrente` | 5–7/ago | RecurrentPPO + BC recorrente (100% transferido) | 188k | **51,6%** (128/248), plana | **3/127, todas por apagão (sorte)** |
| continuação da run 4 | 10–11/ago | a mesma, retomada, com currículo | 190k → 340k (154k) | 25/38 (66%); promovida à N2 após ~60 eps | **6/369 (1,6%), todas por apagão** |

**A continuação de 10–11/ago** (log recuperado em
`logs/analise/historico/2026-08-10_run4_continuacao_detalhado.log`): o currículo promoveu o agente
para a N2 logo no início, e ele passou 369 episódios lá. Ao longo deles, as mortes migraram de
animatrônico para **energia** (de 12% para 71% por bloco de 75 eps) e a sobrevivência mediana subiu
de 6,0 para 7,0 min, sem nenhuma vitória gerida. É o mecanismo por trás da degradação do checkpoint
de 340k medida na §2.2: treinar a N2 sem informação ensinou a defender mais, não na hora certa.

### 2.1 Detalhes da run 4 que importam

- Na N1, 119 das 128 vitórias foram `vitoria_gerida`, e 115 das 120 mortes foram por animatrônico.
  Mesmo na N1 o gargalo já é **defesa**, não energia.
- Na N2, a sobrevivência mediana ficou parada em 4,2–4,6 min (a noite tem 8,9), com 40–50% de
  bateria sobrando na morte.
- O termostato empurrou o `ent_coef` ao teto (0,012) aos ~98k steps: o BC entregou H=0,83, mas
  `H_INICIO`=1,1. É a mesma assinatura da run 2, que o PACOTE §2.1 já mandava evitar.

**Assinatura da N2 (runs 2 e 4, e agora o 340k):** o agente oscila entre dois modos de falha. Ou
morre cedo para animatrônico com bateria sobrando, ou dura mais e morre de apagão. É o que acontece
com quem só consegue ajustar o QUANTO defende, não QUANDO. O humano vence a N2 com 1–2% de bateria:
a margem exige defesa seletiva, ou seja, **informação**.

### 2.2 Fase 1: avaliações sem treino (30/set–6/out, PC pc4)

Todas com `main.py jogar --estocastico --noite 1`, em blocos de 5 eps de N1, modelos **intercalados
na mesma noite** (`scripts/rodar_fase1.py`). Resumo: `python scripts/resumo_avaliacoes.py`.
Percepção limpa em todos os blocos (~0,3 resync de câmera e ~0,1 falha de porta por episódio).

| Modelo | Vitórias N1 | Vitórias **geridas** | Mortes N1 | Ações/min | s/step | N2 |
|---|---|---|---|---|---|---|
| BC (`fnaf_bc.zip`) | 15/35 (43%) | 15 | 20 animatrônico | 41 | 0,81 | 0/14 · sobrev. 232 s |
| BC sem imagem (M2) | 8/20 (40%) | 4 | 7 animatrônico, 5 energia | 34 | 0,66 | 0/5 · 235 s |
| RL 190k (fim da run 4) | 33/60 (55%) | 31 | 27 animatrônico | 45 | 0,93 | 1/27 · 273 s |
| RL 340k (continuação) | 12/33 (36%) | **3** (9 por apagão) | 10 energia, 10 animatrônico | 39 | 1,14 | 0/10 · **436 s**, 8 de 10 mortes por energia |

**Leitura:**
- **BC × RL 190k:** +12 p.p. (p = 0,29) e vitórias geridas no mesmo nível. O PPO não melhorou o BC
  de forma mensurável. (Com 20 eps a diferença parecia +20 p.p.; ela encolheu com mais dados.)
- **RL 190k × 340k:** vitórias geridas de 31/60 para 3/33 (p = 0,00003). Mais PPO **degradou** a
  habilidade: o agente passou a gastar energia (step mais lento = mais ações de porta e luz) e
  durar mais na N2, mas morre de apagão nas duas noites. É o esquecimento da run 2 de novo.
- **Ablação da imagem (M2):** a vitória não caiu (40% × 43%), mas o comportamento mudou (menos
  ações, step mais rápido, mortes por energia). A CNN hoje influencia o uso de câmera, e esse uso
  não vira vitória.
- **Taxa de ação (M3):** o agente faz ~41 ações/min contra ~29 nas demos humanas. Ele age **mais**,
  não menos: a hipótese do descasamento de tick caiu.
- **Variação entre dias:** o BC deu 68% no início da run 4 (ago) e 43% agora, com o mesmo arquivo
  (hash conferido) e o mesmo código de ambiente. Não explicado. Lição: só comparar modelos
  **intercalados na mesma noite**.

## 3. Afirmações antigas corrigidas

| Afirmação (onde) | Correção |
|---|---|
| "Run 4 resolveu a N1: de 2% para 55%" (HANDOFF §2) | A tabela somava a run 3 com a run 4. A run 4 já começou alta, a partir do BC. |
| "`morte_energia` caiu de 55/60 para 1/100" | Compara run 3 com run 4, ou seja, é o efeito do BC. |
| "Divergência: 392k steps no log × 188k no TB" | 392k = run 3 (~202k) + run 4 (~191k). Resolvida. |
| "~0,75 s por step, 100k ≈ 21 h" | Medido ~0,9 s por step: 100k ≈ 25 h. |
| "O PPO está estável" (HANDOFF §7) | Verdade (approx_kl 0,02–0,05), mas não quer dizer que aprende. |
| "Mais amostra de N2 basta" (premissa da run 5) | Runs 1 e 2: centenas de eps de N2 com ~0 vitórias. O 340k: mais treino em N2 virou gasto de energia, não vitória. |
| "Ficar às cegas custa −0,6 contínuo no Φ" (PACOTE §2.3) | Custa −0,6 uma vez: o shaping telescopa. |
| "A run 5 não rodou" (set/2026) | A run 4 foi retomada de 190k a 340k em 10–11/ago, com currículo. Log recuperado; ver §2. |
| "O clone age menos que o humano" (hipótese de set/2026) | Ele age mais (~41 × ~29 ações/min). |
| "O BC dá ~50–68% na N1" | Medido intercalado: 43% (15/35). |

## 4. Lições acumuladas (cada item custou dias; não repita)

**Behavioral Cloning**
- O BC precisa **casar com o algoritmo**. BC feedforward numa RecurrentPPO só transfere o
  extractor: cabeças e LSTM nascem aleatórias (cold-start, run 3). `main.py bc` roteia pela
  constante `USAR_LSTM`.
- Escolher o checkpoint por macro-recall puro seleciona clones degenerados. Critério atual: média
  harmônica de top-1 e macro (`_score_clone`).
- **Top-1 alto não é sinal de qualidade**: "nada" é ~82% dos registros.
- Balanceamento de classes: expoente 1,0 deixa o clone indeciso, 0,5 colapsa em "nada". Default 0,75.
- **Portas são raríssimas nas demos:** 2 a 8 fechamentos por noite, ~36 no total.

**RL, currículo e esquecimento**
- A promoção para a N2 derrubou a N1 da run 2 de ~53% para 0–3%.
- Continuar o PPO de 190k para 340k derrubou as vitórias geridas de 31/60 para 3/33.
- Treinar a N2 sem informação sobre as ameaças ensina a **defender mais**, não a defender na hora certa.

**Entropia**
- O alvo inicial do termostato precisa casar com a H que o BC entrega. Forçar H para cima derrete
  o clone (runs 2 e 4).

**Percepção**
- A detecção do Bonnie pela sombra exige a luz acesa.
- `morte_anim_com_flag ≈ 0` não prova cegueira: Foxy e Freddy não têm flag.
- Os frames das demos são 84×84 em cinza e não servem para template.

**Medição**
- A mesma política pode variar ~25 p.p. entre dias. Compare modelos **intercalados na mesma noite**.
- 20 eps por modelo dão margem de ~±20 p.p. Para diferenças de ~15 p.p., são precisos ~60+ eps.
- Olhe as **vitórias geridas** e as causas de morte, não só a taxa de vitória: vitória por apagão é sorte.

**Configuração e robustez**
- Tuning e algoritmo são **constantes de código**, não `.env` (o `.env` divergiu entre máquinas).
- Se o jogo fechar, o env reabre e o treino continua.

## 5. Plano

### Fase 1 — medições: CONCLUÍDA

M1, M2, M3 e a comparação BC × RL 190k × RL 340k estão na §2.2.

**M4 — sonda de captura (6/out, `debug/sonda_*.json`):**

| Situação | Ruído cru → mediana de 5 | Frames distintos/s |
|---|---|---|
| CAM 1C | 13,4 → 8,4 (−38%) | 59 |
| CAM 2A | 13,5 → 8,8 (−35%) | 58 |
| Escritório, luz esquerda acesa | 0,4 → 0,1 (sem estática) | 2 |

- Captura de ~14 ms por frame: uma rajada de 5 custa ~70 ms por step. Cabe.
- **A estática é um problema só das câmeras.** No escritório a imagem é estável; a única variação
  é a luz da porta, que é o sinal que os detectores usam.
- O piscar ocasional da luz não apareceu nos ~0,9 s de captura. Pelo que o usuário observa no jogo,
  a luz do escritório só varia com a luz da porta acesa. Não é bloqueio.

Conclusão: a rajada vale para ler as **câmeras**, e não compra nada para os detectores do escritório.

### Próximos passos, em ordem

0. **(7/out, passa na frente) A decisão de reabrir a porta esquerda.** O Bonnie causa 16 de 18 mortes
   na N2, quase sempre depois de o agente fechar a porta certa e reabri-la. Separar detector ×
   política com o `monitor` (§6) e, se for o detector, trocar a decisão "corredor vazio" de textura
   em cinza para cor. Isso **não muda a observação**: dá para medir com o `jogar` sobre os modelos
   atuais, sem treinar.
1. **Percepção das câmeras** (adiada: o Foxy causou 1 de 18 mortes), começando pelo estágio do Foxy na CAM 1C, com a rajada de frames.
   Ataca a parede da N2 (falta de informação). Roteiro passo a passo:
   [ROTEIRO_FOXY_CAM1C.md](ROTEIRO_FOXY_CAM1C.md).
2. **DQfD validado offline** só com as demos: tem que igualar o BC em acurácia na validação.
   Ataca o esquecimento visto no 340k.
3. **Treino do zero** com a observação nova (Fase 2) e o DQfD (Fase 3), mais correções ao vivo na
   N2 (HG-DAgger).

O antigo passo "rajada só nos detectores" saiu: o M4 mostrou que o escritório não tem estática.

Os porquês, o runbook, os riscos e o glossário estão em
[ESTRATEGIA_DEMOS_E_PERCEPCAO.md](ESTRATEGIA_DEMOS_E_PERCEPCAO.md).

### Descartado

- **Retomar a run 4 (a antiga "run 5")**: o 340k mostrou que mais PPO piora o agente.
- **Simulador headless**: descartado pelo usuário em jul/2026 porque a CNN não transferiria. O M2
  trouxe o argumento novo que a regra previa (sem a imagem a vitória não cai), então a questão
  **pode ser reaberta**. A decisão é do usuário. Contra: a imagem ainda muda o comportamento, e a
  Fase 2 pretende torná-la útil.

## 6. Hipóteses abertas (não provadas)

- ~~**Quem mata na N2.**~~ **Respondido em 7/out** (vídeo de 20 eps de N2 com o RL 190k, `noite alvo 2`):
  **16 de 18 mortes foram do Bonnie**, 1 do Foxy, 1 da Chica. Padrão dominante: o agente **fecha a
  porta esquerda certa, acende a luz e depois reabre** com o Bonnie ainda lá. O humano vê o Bonnie
  normalmente com a porta fechada. A hipótese do Foxy caiu; a prioridade passa a ser o Bonnie.
- **Por que o agente reabre a porta com o Bonnie lá.**
  - Hipótese A (principal): o detector limpa a flag por engano. Com a porta fechada, ele decide
    pela textura do vão **em cinza** (`LIMIAR_VAZIO = 11.0`, calibrado com só 2 imagens:
    11,65 vazio × 9,24 Bonnie). O Bonnie é roxo, e em cinza pode se confundir com o corredor.
  - Hipótese B: a flag está certa e a política abre para economizar energia (o shaping de energia
    foi calibrado para "abrir com ameaça velha compensa").
  - Teste que separa as duas: `python -m src.utils.monitor` ao vivo, porta fechada e luz acesa com o
    Bonnie presente. Se o detector disser "vazio", é a A.
  - **Resultado (7/out, prints em `debug/bonnie_porta/`): é a A, mas o defeito está na lógica, não
    no sensor.** A `SOMBRA_REGIAO` mede a **sombra do Bonnie** ao lado da porta (visível com a porta
    fechada e a luz acesa): 9,1–9,25 com a sombra e 11,66 sem ela, batendo com a calibração original.
    O defeito: no `_atualizar_ameaca`, leitura de sombra **não zera** a contagem de "vazio", e essa
    contagem é **acumulada** (4 leituras > 11 em qualquer momento limpam a flag). Picos isolados
    (luz acendendo/apagando, animação da porta) somam ao longo do tempo. Bug reproduzido: no print 04 a
    textura dizia sombra presente (9,25) e a flag já estava limpa; ao abrir (print 05) o Bonnie estava lá.
    O detector de rosto (0,96 × 0,39) e a leitura da porta pela cor do botão funcionam.
  - Correção proposta (não muda a observação; mede-se com o `jogar` nos modelos atuais):
    luz acesa + textura baixa = sombra → **zera** a contagem de vazio e mantém a ameaça; luz acesa +
    textura alta → limpar só com leituras **seguidas**; luz apagada → mantém. Limiar 11,0 está colado
    no "vazio" (11,66): o meio é ~10,4; recalibrar com mais prints.
- **O Freddy está inativo nas noites 1–2.** Conhecimento do jogo, não verificado neste ambiente.
- **A rajada resolve a estática e o piscar.** Raciocínio sólido, não medido (M4).
- **Por que o BC deu 68% em agosto e 43% agora.** Mesmo arquivo e mesmo código de ambiente.

## 7. Pendências

- **Noites 6/7** têm botões de menu próprios (o Continue não chega nelas). Calibrar quando a N5
  estiver dominada.
- **`morte_animatronico` é genérico** (não diz quem matou). Hoje só se sabe assistindo o vídeo.
  Proposta (out/2026): guardar os últimos frames coloridos antes de cada morte (o jumpscare costuma
  estar neles) e, com eles rotulados, classificar o jumpscare pela cor para virar
  `morte_bonnie`/`morte_chica`/`morte_foxy`/`morte_freddy` no log.
- **Fixtures de teste faltando no PC pc4:** as imagens `debug/quadro_*.png` (git-ignoradas) não
  estão aqui, então o `testar_deteccao_ameaca` só roda 2 dos 15 casos ("Rosto: 0/0"). Copiar do
  outro PC.
