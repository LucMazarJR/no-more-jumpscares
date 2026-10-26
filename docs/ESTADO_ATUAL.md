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
4. **Ressalva (7/out):** todas as runs e avaliações acima usaram o detector do Bonnie com o bug que
   limpava a flag com ele ainda na porta (§6). Os números valem como retrato do que aconteceu, mas a
   **causa** das conclusões sobre o PPO fica em aberto: a degradação do 340k (defender demais e
   gastar energia) é o que um agente aprenderia se a flag mentisse. E a coluna de ameaça das demos
   humanas também foi gravada com a lógica antiga (os frames 84×84 não permitem recalcular).
5. **Percepção do Bonnie corrigida (7–9/out)** e diagnóstico da política: o RL 190k reabre a porta
   com a flag acesa em 85–89% das vezes. Barrar essa reabertura leva a N2 de 302 a 351 s (§6).
6. **Decisão (9/out): treino do zero com a percepção certa**, conforme
   [PROPOSTA_TREINO_DO_ZERO.md](PROPOSTA_TREINO_DO_ZERO.md): demos novas → BC novo → RL em dois
   braços (A: RecurrentPPO com âncora; B: DQfD). O 190k e as 8 demos antigas saem: aprenderam com a
   flag quebrada. A decisão anterior ("parar o PPO") foi tirada com o sensor quebrado; o PPO volta
   como braço A, e o DQfD como B para comparar.

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
- **O jogo abaixa a prancheta sozinho quando o Foxy bate na porta** (relato do usuário, confirmado
  na demo de N2 de 21/out: 7 s com o estado "câmera aberta" e a tela no escritório). Com o estado
  preso em "aberta", porta e luz são recusadas. O env (a todo step com a câmera aberta) e o
  gravador (a cada iteração) corrigem pelo "YOU" do mapa; antes o gravador não corrigia nunca.

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

Treino do zero com a percepção corrigida, etapas e critérios em
[PROPOSTA_TREINO_DO_ZERO.md](PROPOSTA_TREINO_DO_ZERO.md):

0. ~~**GPU**~~ (9/out): PyTorch com CUDA 12.8 instalado no venv do pc4 (RTX 5060). O outro PC
   precisa do mesmo comando ([GUIA_COMANDOS.md](GUIA_COMANDOS.md)).
1. **Demos novas** (usuário): 3× N1, 6–8× N2, 2× N3, com a disciplina do diagnóstico (segurar a porta
   enquanto o Bonnie estiver lá, checar a luz esquerda a cada poucos segundos, pouca câmera).
   Andamento (21/out, PC 1): 2× N1 (vitórias) e 1× N2 (vitória com 1,5%). Na N2, 3 das 4
   reaberturas da porta esquerda ficaram gravadas com a flag acesa: o corredor estava vazio, mas o
   detector só limpa a flag com 4 leituras de vazio (`DEBOUNCE_VAZIO`), e cada olhada rende 1–2.
   Em 23/out, 1× N3 parcial (393 s). **25/out: o usuário encerrou as gravações**, com ~33 min de
   demos novas (2 N1, 1 N2, 1 N3 parcial), contra as 11–13 noites planejadas.
2. **BC novo** (`main.py bc`) e avaliação com `jogar --estocastico --trace`: N1 ≥ 40% e reaberturas
   com a flag acesa < 50%.
   25/out, no pc4 (GPU), decisões do usuário para compensar o pouco dado:
   - 9 demos: as 4 novas + as 5 N1 antigas (~45 min), de volta a `dados/`. Na N1 a flag antiga
     quase não acendia (0–5% do tempo), então a contaminação é pequena. As N2/N3 antigas seguem em
     `dados/_antigas_flag_errada/`.
   - Reabrir a porta com a flag acesa vira "nada" na carga (`_neutralizar_reabertura_com_flag`):
     6 registros de 13.796 (3 da N2 nova, pelo atraso do detector; 3 das N1 antigas).
   - Avaliação de 20 eps de N1 (`rodar_fase1.py --blocos BC --trace`) encadeada com o treino A: se
     o BC reprovar, o treino é parado ainda no warmup (ator quase parado), sem perda.
3. **Braço A, RecurrentPPO**, a partir do BC novo. Código pronto (9/out): âncora BC ligada, alvo de
   entropia calibrado pela H que o BC entrega no warmup, currículo promovendo só por vitórias
   **geridas**.
4. **Braço B, DQfD**: implementar e validar offline com as demos novas (tem de igualar o BC) enquanto
   o A roda. Não dá para validar antes de as demos existirem.
5. Depois: correções ao vivo na N2 (HG-DAgger) e a percepção das câmeras, começando pelo Foxy na
   CAM 1C ([ROTEIRO_FOXY_CAM1C.md](ROTEIRO_FOXY_CAM1C.md); adiada porque o Foxy causou 1 de 18 mortes).

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
  - **Corrigido (7/out, commit `116d117`).** A diferença entre frames reais do F8 mostrou que a sombra
    é a **silhueta do Bonnie numa janela de vidro** ao lado da porta, que só a luz ilumina. O detector
    passou a usar duas máscaras nessa janela (em `referencias/sombra/`):
    borda do vidro, "a luz chegou ao corredor?" (6,5–9,5 escuro × 38–42 aceso); e silhueta, "o Bonnie
    está lá?" (7,1–7,6 com ele × 47 vazio). A luz é lida da cena, então o frame em que o botão
    acende antes do corredor não vira falso alarme, e a sombra **acende** a ameaça com a porta aberta
    ou fechada. Lógica única em `decidir_bonnie` (env e monitor); teste com 13 casos, incluindo o
    print 04 do bug. Recalibração: `python -m src.utils.calibrar sombra vazio|sombra|apagada`.
  - **Efeito no jogo (7–8/out, RL 190k na N2, sem vídeo): nenhum.** 1/20 vitórias e sobrevivência
    mediana de 281 s, contra 1/20 e 273 s antes da correção; segue morrendo cedo com bateria
    sobrando (88% aos 55 s). Duas explicações em aberto: (A) dentro do loop do agente a captura sai
    antes de o corredor iluminar e o detector fica mudo (no monitor a luz ficava segurada); (B) o
    detector vê, mas a política, treinada com a flag errada, não usa. Para separar, o `jogar` agora
    imprime por episódio `luzE n (acesa x, silhueta y) | flagE z%` (commit `b65a167`).
  - **8/out (mesmo teste, com telemetria): a hipótese A caiu.** Com a luz segurada, a cena sai
    iluminada em 56% das capturas (662/1189 na N2; o resto é o frame de transição, ignorado) e a
    silhueta foi vista 140 vezes. N2 segue em 1/20, sobrevivência mediana 321 s. Novo sinal
    suspeito: a flag esquerda fica acesa em 61–92% dos steps, até na N1. Pode "grudar" (só apaga com
    4 checagens de luz vendo o corredor vazio) ou a política pode ignorá-la. Próximo passo:
    `jogar --trace` + `scripts/analisar_trace.py` para ver, step a step, o que acontece antes de
    cada morte.
  - **8/out (trace de 70 noites de N2; relatório completo em
    [analises/2026-10-08_trace_rl190_n2.md](analises/2026-10-08_trace_rl190_n2.md)): a flag era
    ruído.** 337 dos 344 acendimentos foram com a luz esquerda solta: com o agente olhando para o
    centro/direita, a região da janela mostra outra parte da sala (girando ~200 px lê "silhueta",
    240–300 px lê "vazio"). **Corrigido** com uma trava de alinhamento (botão LIGHT aceso no lugar,
    commit `ec98fc2`). Reconstruída com a trava, a flag fica acesa 15% do tempo (não 73%) e estaria
    acesa com a porta aberta em 42 das 69 mortes. O agente reage ao Bonnie (fecha em ~2 s), mas
    **segura a porta só 3 s** (mediana) e reabre com ele lá; deixa a câmera aberta 47% do tempo.
    A continuação 190k→340k (currículo promoveu aos 218,7k, **sem âncora**) foi de 50% de vitórias
    geridas a 0 e de 0 a 74% de mortes por energia.
  - **9/out (60 noites de N2 com a trava; relatório em
    [analises/2026-10-09_trace_rl190_n2_trava.md](analises/2026-10-09_trace_rl190_n2_trava.md)):
    a percepção está resolvida, o problema agora é a política.** Flag acesa 12% do tempo e só com
    avistamento real; sobrevivência mediana 262 → 314 s (p = 0,08); 0/60 vitórias. **85% das
    reaberturas da porta esquerda são com a flag acesa**, e 34 das 60 mortes foram com a flag acesa
    e a porta aberta: o agente sabe que o Bonnie está lá e reabre. As outras 25 são de checagem
    tardia (flag apagada, última luz 22 s antes). Próximo: diagnóstico `jogar --segurar-porta`.
  - **9/out — diagnóstico "segurar a porta"** ([relatório](analises/2026-10-09_diagnostico_segurar_porta.md)):
    barrando a reabertura com a flag acesa (806 tentativas em 50 noites), a sobrevivência na N2 vai a
    **351 s** (+49 s, p = 0,055) e as mortes "sabendo" caem de 55 para 2, mas as vitórias ficam em
    2/50. Sobram a **checagem tardia** (46/48 mortes com a flag apagada, ~21 s depois da última luz) e
    a **energia** (8 mortes por apagão; câmera ~50% do tempo). Proposta de treino do zero em
    [PROPOSTA_TREINO_DO_ZERO.md](PROPOSTA_TREINO_DO_ZERO.md), para revisão.
- **O Freddy está inativo nas noites 1–2.** Conhecimento do jogo, não verificado neste ambiente.
- **A rajada resolve a estática e o piscar.** Raciocínio sólido, não medido (M4).
- **Por que o BC deu 68% em agosto e 43% agora.** Mesmo arquivo e mesmo código de ambiente.

## 7. Pendências

- **Arquivos fora de uso (9/out, git-ignorados):** as 8 demos antigas estão em
  `dados/_antigas_flag_errada/` (fora do glob `dados/*/dataset.json` do BC e da âncora; para reusar,
  mova de volta). O RL 190k/340k e o BC antigo (cópia `fnaf_bc_flag_errada.zip`) estão em
  `modelos/pc_luciano_run4/`, fora da retomada automática.
- **O `.git` tem ~1,9 GB de checkpoints** commitados em branches antigos (pc1–pc4, union, record,
  3.0). Apagar esses branches reduz o repo, mas é irreversível: só com decisão do usuário.
- **Noites 6/7** têm botões de menu próprios (o Continue não chega nelas). Calibrar quando a N5
  estiver dominada.
- **`morte_animatronico` é genérico** (não diz quem matou). Hoje só se sabe assistindo o vídeo.
  Proposta (out/2026): guardar os últimos frames coloridos antes de cada morte (o jumpscare costuma
  estar neles) e, com eles rotulados, classificar o jumpscare pela cor para virar
  `morte_bonnie`/`morte_chica`/`morte_foxy`/`morte_freddy` no log.
- **Fixtures de teste faltando no PC pc4:** as imagens `debug/quadro_*.png` (git-ignoradas) não
  estão aqui, então o `testar_deteccao_ameaca` só roda 2 dos 15 casos ("Rosto: 0/0"). Copiar do
  outro PC.
