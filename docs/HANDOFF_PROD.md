# Handoff para a máquina de produção — run 5 (ago/2026)

> **Este documento é um PROMPT.** Ele passa o contexto acumulado de várias sessões para o agente
> que roda na máquina de treino. Leia inteiro antes de agir. Idioma do projeto: **português**
> (código, comentários, docs, logs).

---

## 0. Seu objetivo IMEDIATO

A run 5 está (ou estará) rodando nesta máquina. Sua primeira tarefa é **ler os resultados,
interpretá-los e recomendar os próximos passos** — não implementar nada antes de medir.

Ordem de trabalho:

1. **Meça** o estado atual (§6 diz onde e como).
2. **Compare** com as predições da run 5 (§5) e com os critérios de aborto.
3. **Decida** entre os caminhos do §7, com o número que justifica a decisão.
4. Só então implemente — e **uma variável por vez** (§8).

**Regra de ouro:** não confie nos números deste documento nem nos comentários do código —
eles registram o que *acreditávamos* quando foram escritos. Meça dos logs e do tensorboard.
**Refutar uma afirmação daqui é resultado valioso, não problema.** Já aconteceu várias vezes
(§4) e foi o que mais destravou o projeto.

---

## 1. O projeto em um parágrafo

Agente de RL (stable-baselines3) que joga o **FNAF 1 real** por captura de tela + mouse, em
**tempo real**: ~0,75 s por step. Isso torna a **amostra o recurso escasso** — 100k steps ≈ 21 h
de máquina. Um teste mal desenhado custa dias. Objetivo: vencer as noites de forma estável.
Algoritmo atual: **RecurrentPPO (LSTM)** com warmstart de **BC recorrente** sobre demos humanas.

Mapa do repositório e comandos canônicos: `CLAUDE.md` na raiz. Docs vivos: `docs/README.md`.

---

## 2. Onde estamos: a run 4 foi um sucesso parcial importante

**A Noite 1 foi resolvida.** Progressão medida em `logs/analise/historico/` (a run 4 foi
arquivada quando a 5 começou; se não houver arquivo, está em `logs/analise/treino_detalhado.log`):

| Bloco de 60 eps (Noite 1) | Vitórias | Energia ao morrer | Causa dominante |
|---|---|---|---|
| 0-60 | 1 (2%) | 1,2% | `morte_energia` (54/60) |
| 120-180 | 1 (2%) | 0,2% | `morte_energia` (56/60) |
| **300-360** | **33 (55%)** | 22,2% | **`vitoria_gerida` (28)** |
| 480-540 | 30 (50%) | 21,2% | `vitoria_gerida` / animatrônico |

Totais da run 4: 698 episódios, ~392k steps somados, 553 eps de Noite 1 / 142 de Noite 2 / 3 de
Noite 3. Das 143 vitórias na Noite 1, **120 foram `vitoria_gerida`** (gestão real) contra 23
`vitoria_apagao` (sorte) — inversão histórica: antes quase toda vitória era apagão sortudo.
`morte_energia` caiu de ~55 em cada 60 episódios para **1 em cada 100**.

**Saúde do treino:** entropia começou em **0,83** (o warmstart do BC recorrente funcionou — na
run 3, sem ele, começava em 2,6) e ficou entre 0,83 e 0,97 a run toda. Sem colapso, sem
cold-start. `ent_coef` subiu até o teto 0,012.

**O novo gargalo é a Noite 2, e é um problema DIFERENTE:**

| Noite 2 (run 4) | Valor |
|---|---|
| Vitórias | 3 em 142 |
| Morre em | ~4,5 min (a noite tem 8,9 min) |
| Bateria ao morrer | **43-50% sobrando** |
| Causa | `morte_animatronico` ~100% |
| `custom/noite_2/morte_anim_com_flag` | **0, sempre** |

Morre **cedo, com bateria de sobra, para um animatrônico sem flag**. Não é mais energia — é
percepção/defesa.

---

## 3. O que mudou para a run 5 (e por quê)

Duas constantes em `src/agent/train.py` foram viradas, mais um conserto:

- **`CURRICULO_ATIVO = True`** (estava False). Motivo: o agente só chega na Noite 2 *vencendo* a
  Noite 1, então coletou 553 eps de N1 contra 142 de N2 — está faminto justamente na noite que
  precisa aprender. Destravar multiplica a amostra de N2 sem tocar na observação.
- **`ANCORA_BC = True`** (estava False). É a defesa contra o esquecimento catastrófico que matou
  a run 2 (§4). Faz passos de BC recorrente sobre as demos a cada rollout, com peso decaindo,
  num Adam dedicado. As demos são 5 de 8 da Noite 1 → puxa a política de volta pro
  comportamento frugal enquanto o RL aprende a Noite 2.
- **Conserto do decaimento da âncora:** ela decaía pelo `_current_progress_remaining` do SB3, que
  em **retomada** já vem adiantado (a âncora nasceria meia-força). Agora decai sobre a própria
  duração (`ANCORA_BC_STEPS = 150_000`), contada do step em que liga.

A run 5 é uma **RETOMADA** (sem `--novo`, sem `--bc`) — a política de ~50% na Noite 1 é valiosa
demais para jogar fora.

---

## 4. Experiência acumulada — erros já cometidos, NÃO repita

Esta seção é o ativo mais valioso deste documento. Cada item custou dias.

**Sobre o Behavioral Cloning**
- BC **feedforward** transfere só o `MultimodalExtractor` para RecurrentPPO (heads e LSTM nascem
  aleatórias) → **cold-start**. Foi o que matou a run 3: 175k steps, entropia ~2,07, Noite 1 em
  ~5%. O **BC recorrente** (`treinar_bc_recorrente`) transfere **50/50 tensores (100%)`.
  `main.py bc` roteia pelo `USAR_LSTM` — o BC precisa CASAR com o algoritmo da run.
- **Selecionar o checkpoint por macro-recall puro escolhe modelos DEGENERADOS** (spamma uma
  classe rara: macro 12,8% com top-1 2,8%). Hoje o critério é a **média harmônica** de top-1 e
  macro (`_score_clone`).
- **`top-1` alto NÃO é sinal de qualidade**: "nada" é 82% dos frames, então "sempre nada" já
  marca ~86,5%. Exigir top-1 alto *exige o colapso* na classe majoritária. Referência real: o
  clone que comprovadamente funcionou tinha **top-1 46,6% / macro 36,9% / score 41,2**.
- **Balanceamento de classes é um dial com dois modos de falha**, ambos observados:
  expoente 1.0 → empurra pra uniforme (H≈2,4-2,8, clone indeciso, inútil como warmstart porque
  o RL **sorteia** a ação); expoente 0.5 → "nada" domina 38% do gradiente e o argmax **colapsa**.
  **0.75 é o meio-termo** e é o default.
- **Early stop não pode contar paciência enquanto a loss ainda cai** — matava a run no platô
  inicial de "sempre nada".
- Com poucas sequências (~7 episódios de treino) o resultado depende muito da **inicialização**:
  por isso `restarts=4` e escolhe o melhor.

**Sobre o currículo**
- A promoção automática **matou a run 2**: promoveu para a Noite 2 em ~52k e a Noite 1 DESABOU
  de 53,3% para 0-3,3% (janelas cheias de 30 eps), com `morte_energia` indo a 99,3%. A Noite 2
  ensinou gasto de energia e sobrescreveu a política frugal. **É o risco nº 1 da run 5** — e a
  razão de a âncora existir.

**Sobre percepção e detecção**
- A detecção do Bonnie pela **sombra funciona de porta FECHADA** (a `SOMBRA_REGIAO` continua
  visível; calibração: vazio 11,65 vs Bonnie 9,24), **mas exige a luz esquerda ACESA**. Acampar
  de luz apagada nunca limpa a flag.
- **`morte_anim_com_flag ≈ 0` NÃO prova que o agente está cego para Bonnie/Chica.** Só
  Bonnie/Chica têm flag (estados 9-10). **Foxy e Freddy não estão na observação**, então morte
  causada por eles registra flag=0 *por construção*. Já interpretamos isso errado uma vez.
- Frames gravados pelas demos são **84×84 em cinza** — inúteis para template matching. Templates
  exigem captura em resolução cheia (1280×720) via as ferramentas de calibração.

**Sobre configuração**
- O `.env` é git-ignored e **divergiu entre máquinas**, fazendo produção rodar a configuração
  errada (LSTM desligada) por uma run inteira. Por isso **todo o tuning e o algoritmo agora são
  constantes de código**. No `.env` ficam só coisas que variam por máquina (janela, caminho,
  coordenadas, timings, `PC`) **mais** `FNAF_RESET_METODO` e `FNAF_NOITE_DESEJADA`, que o
  usuário usa para mirar noites específicas em testes.

**Sobre robustez**
- O `reset()` derrubava o processo inteiro quando o jogo fechava de madrugada. Hoje há
  `_garantir_janela()` e `_capturar_observacao_resiliente()`. Se o jogo fechar, deve aparecer
  `[RESET] ... reabrir` **e o treino continuar**. O `finally` do `treinar()` salva o modelo mesmo
  em crash.

**Correções que uma auditoria cega dos dados nos impôs** (exemplos de que este doc pode errar):
- Achávamos que a entropia da run 3 tinha "travado" — na verdade **descia monotonicamente**.
- Achávamos que o feedforward mantinha ~50% na Noite 1 — eram os **primeiros 20%** da run; ela
  fechou em 2,4%.
- Os docs dizem que ficar às cegas "custa −0,6 **contínuo**" no Φ. **Está errado**: shaping
  potential-based **telescopa** (`phi_depois = 0.0` no terminal), então custa −0,6 **uma vez**.
  Isto ainda não foi corrigido em `docs/PACOTE_BC_ENTROPIA.md` — ver §9.

---

## 5. Predições da run 5 e critérios de aborto

**A métrica mais importante NÃO é o progresso da Noite 2. É a Noite 1 não desabar.**

| # | Predição | Onde medir |
|---|---|---|
| P1 | `custom/noite_1/win_rate` **se mantém** em ~50% (a âncora segurou) | tensorboard |
| P2 | `custom/noite_2/sobrevivencia_s` **sobe** de ~290s (noite = 535s) | tensorboard |
| P3 | `custom/ancora_bc/peso` decai linear; `custom/ancora_bc/loss` estável, sem explodir | tensorboard |
| P4 | `train/approx_kl` **não** estoura — a âncora dá passos fora da região de confiança do PPO | tensorboard |
| P5 | Promoção acontece: `[curriculo] Noite 1 dominada -> alvo Noite 2` no log | `logs/treino.log` |

**ABORTO:** se `custom/noite_1/win_rate` cair abaixo de **30%** e permanecer por uma janela
cheia, a âncora não segurou. Pare, e reporte — as alavancas são `ANCORA_BC_PESO` (1.0 → 2.0) ou
`ANCORA_BC_PASSOS` (1 → 2), ou re-travar o currículo.

**Dica de eficiência medida:** taxa de vitória é a métrica mais *cara* de todas. Detectar uma
melhora via **tempo até o apagão / tempo de sobrevivência** custou ~9 episódios contra ~67 pela
taxa de vitória — 1,5 h contra 11 h. Prefira sobrevivência e causas de morte para decisões
rápidas; use win rate só para a conclusão final.

---

## 6. Como medir (tudo offline, jogo fechado)

```powershell
venv\Scripts\python scripts\metricas_treino.py     # win rate / sobrevivência / causas por noite
tensorboard --logdir logs                          # séries custom/* e train/*
```

Fontes:
- `logs/analise/treino_detalhado.log` — 1 linha por episódio com telemetria (energia final,
  causa). **É a fonte preferida dos parsers.**
- `logs/treino.log` — versão enxuta (leitura durante a execução).
- `logs/analise/historico/` — runs anteriores, arquivadas automaticamente quando um treino
  **fresco** (`--novo`) começa. **Não misture runs na mesma análise** — isso já poluiu
  diagnósticos nossos.
- `logs/tensorboard/<Algo>_<N>/` — a run mais recente é a de maior N e mtime.

Séries próprias úteis: `custom/noite_N/{win_rate,sobrevivencia_s,n_eps,morte_anim_com_flag}`,
`custom/causas/*`, `custom/entropia`, `custom/h_alvo`, `custom/ent_coef`,
`custom/curriculo/noite_alvo`, `custom/ancora_bc/{peso,loss}`.

Verificação offline do código (sem o jogo):
```powershell
venv\Scripts\python scripts\smoke_test.py              # 6/6
venv\Scripts\python -m src.utils.testar_recompensa     # 17/17
venv\Scripts\python -m src.utils.testar_noite          # 11/11
venv\Scripts\python -m src.utils.testar_deteccao_ameaca
venv\Scripts\python -m src.utils.testar_masking        # pré-voo da LSTM
```

**Uma inconsistência a resolver:** na run 4, a soma de `passos` no log deu ~392k, mas o
tensorboard registrou ~188k steps. Descubra a causa (provavelmente retomadas e a contagem do
SB3) antes de citar "steps" em qualquer conclusão.

---

## 7. Planos: o que vem depois, em ordem de custo

**Se a run 5 destravar a Noite 2** → deixe rodar, deixe o currículo promover para a Noite 3 e
reavalie. Provavelmente o Freddy entra em cena (ele ativa tarde) e a percepção volta à pauta.

**Se a Noite 2 empacar mesmo com amostra farta** → aí a Fase 2 (percepção) se justifica. Ordem
por retorno/esforço:

1. **Frame stacking (2-4 frames).** Dá movimento e cancela a estática do FNAF (que é aleatória
   por frame, enquanto o sprite é consistente). **Não precisa de referência nenhuma** — é
   arquitetural. As demos existentes servem: os frames estão gravados **em sequência**, então as
   pilhas podem ser montadas offline, sem regravar.
2. **Detector de estágio do Foxy (CAM 1C).** A Pirate Cove tem 4 estágios visualmente
   distintos, e "cortina VAZIA" = "o Foxy está correndo AGORA" — provavelmente o sinal mais
   valioso do jogo. Bem mais viável que o caso do Bonnie (cuja sombra é indistinguível do
   escuro).
3. **Freddy no East Hall Corner (CAM 4B).** Só faz sentido da Noite 3 em diante.

**Economia importante:** (1) e (2) mudam o shape da observação e ambos obrigam **treino do
zero** — se forem feitos, faça **no mesmo treino**, poupando uma rodada de dias.

**Não faça agora:** voltar para feedforward; mexer em `n_steps`/LR/clip (o PPO está estável);
subir `PESO_ENERGIA` (o gargalo de energia foi resolvido); templates antes de tentar aprendizado
por experiência (preferência explícita do usuário).

---

## 8. Convenções de trabalho (o usuário pediu isso explicitamente)

- **Uma variável por vez.** Cada teste custa horas ou dias; empilhar mudanças torna impossível
  saber o que funcionou.
- **Critério de aborto pré-registrado** antes de começar um run longo.
- **Medir por taxa de vitória / sobrevivência**, nunca por recompensa (a escala muda entre
  versões).
- **Tuning no CÓDIGO**, não no `.env` (§4).
- **Logs de execução enxutos**; telemetria vai para `logs/analise/`.
- Não renomear nem mover os alvos dos comandos canônicos (`CLAUDE.md`).

---

## 9. Pendências conhecidas

1. **Estado do código:** rotação automática de logs, `CURRICULO_ATIVO=True`, `ANCORA_BC=True` e
   o conserto do decaimento da âncora estão **commitados** (`5d0d96d`). Confirme com `git log`
   que esta máquina os tem antes de tirar conclusões sobre o comportamento da run.
2. **Doc com erro conceitual não corrigido:** `docs/PACOTE_BC_ENTROPIA.md` afirma que o termo de
   informação do Φ "custa −0,6 contínuo". Shaping potential-based telescopa → custa **uma vez**.
   Corrigir o texto (o código está certo).
3. **Checkpoint da run 3 (175k)** pode existir em outra máquina: os logs dizem `pc4` e o `.env`
   local diz `PC=1`. Vale localizar antes de considerá-lo perdido (~33 h de máquina).
4. **Divergência steps log × tensorboard** na run 4 (§6).
5. **`morte_animatronico` é genérico** — não identifica qual animatrônico. `menu_crash` é
   assinatura do Bonnie e `morte_golden` do Golden Freddy, mas Foxy e Freddy não são separáveis.
   Se isso virar bloqueio para decidir a Fase 2, é uma melhoria de telemetria a considerar.

---

## 10. Suposições ainda NÃO comprovadas

Trate como hipóteses, não como fatos. Se puder testar alguma com os dados, isso vale mais que
qualquer implementação nova.

- **"O Foxy é quem mata na Noite 2."** Inferido de: morte precoce (~4,5 min), bateria sobrando,
  flag=0, e do conhecimento de que o Freddy ativa tarde. **Não foi medido** — a causa registrada
  é genérica.
- **"O Freddy está inativo nas noites 1-2."** Conhecimento do jogo, não verificado neste ambiente.
- **"A âncora BC impede o esquecimento."** É exatamente o que a run 5 testa. Sem evidência ainda.
- **"Frame stacking ajuda a enxergar através da estática."** Raciocínio sólido, não testado aqui.
- **"Mais amostra de Noite 2 basta para destravá-la."** É a aposta da run 5. A alternativa é que
  o gargalo seja observacional (Foxy/Freddy invisíveis), e aí só a Fase 2 resolve.
