# Estado atual do projeto (set/2026)

> **Único doc de "onde estamos".** O [CLAUDE.md](../CLAUDE.md) guarda só o que é permanente;
> os docs em [historico/](historico/) são retratos de época. Atualize ESTE arquivo quando uma
> medição mudar o quadro, com o número que justifica a mudança.
>
> **Regra de ouro (herdada, e confirmada de novo nesta auditoria):** não confie em número de doc
> sem conferir no log. Refutar uma afirmação daqui com dados é resultado, não problema.

## 1. Resumo

1. **O que funciona:** o ambiente (captura, atuação, detecção, reset) e o **Behavioral Cloning**.
   Oito demos humanas (~72 min) levaram a Noite 1 a ~50% de vitória.
2. **O que não funciona:** o **RL por cima do BC**. Em 4 runs (~750k steps, ~8 dias de máquina),
   o PPO nunca levou a vitória na N1 acima do nível em que o BC a deixou e nunca aprendeu a Noite 2.
3. **Por quê (hipótese principal):** o PPO on-policy a ~1 step/s dá ~1 update de política por
   hora (~46 em 47 h na run 4). Na N2 o gargalo parece ser **informação** (o agente não sabe
   *quando* defender), não quantidade de amostra.
4. **Próximo passo:** as medições baratas da §5 (Fase 1). Os números delas decidem entre
   percepção nova (Fase 2) e algoritmo novo (Fase 3), que provavelmente vêm juntos num único
   treino do zero.

## 2. As runs, medidas nos logs

Fontes: `logs/analise/historico/<run>_detalhado.log` e `logs/tensorboard/<run>/`.
Steps a ~0,9 s cada.

| Run (pasta do TB) | Datas | Algoritmo + init | Steps | Noite 1 | Noite 2 |
|---|---|---|---|---|---|
| `_pre_pacote/*` | 20/jun–2/jul | PPO/RPPO sem BC | até 400k | — | — |
| `2026-07-03_run1_ppo_bc` | 3–5/jul | PPO + BC feedforward | 160k | 57% logo no início (BC); promovida à N2 com ~28k | **0%** por ~120k steps |
| `2026-07-06_run2_ppo_bc_curriculo` | 6–14/jul | PPO + BC ff + currículo | 231k | 60% no início → **0–8%** depois da promoção (esquecimento) | **3/653 (0,5%)** |
| `2026-07-16_run3_rppo_bc_parcial` | 16–22/jul | RecurrentPPO + BC ff (só o extractor transfere) | 175k | 0–13% (cold-start, H 2,65→2,07) | 15 eps |
| `2026-08-05_run4_rppo_bc_recorrente` | 5–7/ago | RecurrentPPO + BC recorrente (100% transferido) | 188k | **51,6%** (128/248), **plana**: 68% nos 25 primeiros eps, 54% na 1ª metade, 49% na 2ª | **3/127, todas por apagão (sorte)** |

A **run 5** (retomada da 4 com currículo + âncora BC, descrita em
[historico/HANDOFF_PROD.md](historico/HANDOFF_PROD.md)) **não rodou**.

**Detalhes da run 4 que importam:**
- Na N1, 119 das 128 vitórias foram `vitoria_gerida`, e 115 das 120 mortes foram por animatrônico.
  Mesmo na N1 o gargalo já é **defesa**, não energia.
- Na N2, a sobrevivência mediana ficou parada em 4,2–4,6 min (a noite tem 8,9), com 40–50% de
  bateria sobrando na morte.
- O termostato empurrou o `ent_coef` ao teto (0,012) aos ~98k steps: o BC entregou H=0,83, mas
  `H_INICIO`=1,1. Com isso H subiu para 0,97, e a vitória na N1 no TB caiu de 0,67 para 0,53
  (dentro do ruído). É a mesma assinatura da run 2, que o PACOTE §2.1 já mandava evitar.
- A percepção está limpa: 0,2 falha de porta e 0,2 resync de câmera por episódio, contra 25–70
  nas primeiras runs.

**Assinatura da N2 na run 2:** ela oscilou entre dois modos de falha. Ou morria cedo para
animatrônico com ~40% de bateria, ou durava mais e morria de apagão. Nunca achou o meio-termo.
É o comportamento esperado de quem só consegue ajustar o QUANTO defende, não QUANDO. O humano
vence a N2 com 1–2% de bateria: a margem exige defesa seletiva, ou seja, informação.

## 3. Afirmações antigas corrigidas por esta auditoria

| Afirmação (onde) | Correção |
|---|---|
| "Run 4 resolveu a N1: de 2% para 55%" (HANDOFF §2) | A tabela somava a **run 3** (sessões 1–5 do log) com a **run 4** (sessão 6, treino novo). A run 4 já começou em ~68% e ficou plana. O ganho foi do **BC**, não do RL. |
| "`morte_energia` caiu de 55/60 para 1/100" | Compara run 3 com run 4, ou seja, é o efeito do BC. |
| "Divergência: 392k steps no log × 188k no TB" | 392k = run 3 (~202k) + run 4 (~191k). Resolvida. |
| "~0,75 s por step, 100k ≈ 21 h" | Medido ~0,9 s por step (190.693 steps em ~47 h): 100k ≈ 25 h. |
| "O PPO está estável" (HANDOFF §7) | Verdade (approx_kl 0,02–0,05, EV até 0,78), mas não quer dizer que aprende. |
| "Mais amostra de N2 basta" (premissa da run 5) | As runs 1 e 2 tiveram centenas de eps de N2 com ~0 vitórias. |
| "Ficar às cegas custa −0,6 **contínuo** no Φ" (PACOTE §2.3) | Custa −0,6 **uma vez**: o shaping potential-based telescopa. Texto corrigido. |

Causa-raiz da leitura errada: os logs acumulavam runs no mesmo arquivo, e o cabeçalho de sessão
não dizia qual run era. **Corrigido:** as runs agora têm nome (`modelos/run.json`), pasta
própria no TB, histórico arquivado com o nome da run e a linha
`run: <nome> | NOVO/RETOMADA | commit` em toda sessão.

## 4. Lições acumuladas (cada item custou dias; não repita)

**Behavioral Cloning**
- O BC precisa **casar com o algoritmo**. BC feedforward numa RecurrentPPO só transfere o
  extractor: cabeças e LSTM nascem aleatórias e o treino entra em cold-start (foi o que matou a run 3).
  `main.py bc` roteia pela constante `USAR_LSTM`.
- Escolher o checkpoint por macro-recall puro seleciona clones **degenerados**. O critério hoje é
  a média harmônica de top-1 e macro (`_score_clone`).
- **Top-1 alto não é sinal de qualidade**: "nada" é ~82% dos registros. Referência de um clone que
  funcionou: top-1 46,6%, macro 36,9%.
- O balanceamento de classes tem dois modos de falha. Expoente 1,0 deixa o clone indeciso;
  expoente 0,5 faz o argmax colapsar em "nada". **0,75** é o default.
- Com ~7 episódios de treino o resultado depende da inicialização, por isso `restarts=4`.
- **Portas são raríssimas nas demos:** 2 a 8 fechamentos por noite, ~36 em todas as demos juntas.
  É a ação mais crítica e a de menos exemplos.

**Currículo e esquecimento**
- A promoção automática derrubou a N1 da run 2 de ~53% para 0–3% (esquecimento catastrófico).
- Com a entrada no currículo **só por vitória**, o agente passa mais tempo onde já é bom.

**Entropia**
- O alvo inicial do termostato precisa **casar com a H que o BC entrega**. Forçar H para cima
  derrete o clone (runs 2 e 4).

**Percepção**
- A detecção do Bonnie pela sombra funciona de porta fechada, mas **exige a luz acesa**.
- `morte_anim_com_flag ≈ 0` **não prova cegueira**: só Bonnie e Chica têm flag. Foxy e Freddy não
  estão na observação, então a morte causada por eles registra flag=0 por construção.
- Os frames das demos são 84×84 em cinza e não servem para template. Templates exigem captura em
  1280×720.

**Configuração e robustez**
- O `.env` divergiu entre máquinas e fez produção rodar a configuração errada por uma run inteira.
  Tuning e algoritmo hoje são **constantes de código**.
- Se o jogo fechar, o env reabre (`_garantir_janela`) e o treino continua. O `finally` do
  `treinar()` salva o modelo mesmo em crash.

## 5. Plano

### Fase 1 — medições baratas (~1 noite de máquina, com o jogo)

Todas saem com `jogar`, que não aprende e grava em `logs/analise/avaliacoes.log`.

| # | Comando | Custo | Decide |
|---|---|---|---|
| M1 | `python main.py jogar --modelo modelos/fnaf_bc.zip --estocastico --noite 1 --episodios 20` | ~3,5 h | Mede o **BC puro**, sem RL, na mesma dinâmica de reset da run 4 (alvo N1; cada vitória rende 1 ep de N2, ~10 no total). Se a N1 ficar em ≈50%, está provado que o RL não agregou nada e a Fase 3 se justifica. 20 eps separam 50% de ≤25%, mas não de 40%; use também a sobrevivência mediana. |
| M2 | `python main.py jogar --modelo modelos/fnaf_bc.zip --estocastico --episodios 20 --ablacao imagem` | ~3 h | Se nada muda, a CNN (84×84 da janela inteira) não carrega informação. Isso prioriza a Fase 2 e reabre o simulador em nível de estados com um argumento novo: se a política vive dos estados, o simulador transfere. |
| M3 | `python scripts/taxa_acao.py` depois do M1 | offline | Compara ações/min do clone com as do humano (N1 ≈29, N2 ≈33, N3 ≈38). Se o clone agir muito menos, as próximas gravações usam o tick do agente. |
| M4 | `python -m src.utils.sonda_captura --rotulo <situação> [--segurar-luz esq\|dir]` no escritório, na CAM 1C, na 2A e com luz acesa | ~15 min | Latência do `mss`, frames distintos/s, quanto a mediana de 3/5 reduz o ruído, fração de frames escuros (piscar). Define a rajada da Fase 2. |

Para M1 e M2 no mesmo PC, rode em noites diferentes. Os checkpoints da run 4 estão no
PC-LUCIANO; se forem trazidos, repita M1 com o checkpoint final dela para medir BC contra BC+RL
diretamente.

**Regras de decisão (registradas antes de medir):**
- M1 com N1 ≥ 40% → o RL não agregou nada. Seguir para a Fase 3.
- M1 com N1 < 30% → o RL da run 4 agregou algo; reavaliar antes de trocar o algoritmo.
- M2 com queda < 10 p.p. → a imagem não contribui. A Fase 2 vira prioridade.
- M3 com razão agente/humano < 0,6 → gravar as próximas demos no tick do agente.

### Fase 2 — percepção robusta (depois do M4)

Tudo atrás de flag, desligado por padrão, para não contaminar M1/M2.

- **Rajada por step** (k=3–5 frames em ~60–120 ms):
  - Mediana temporal contra a **estática** (ruído aleatório por frame, enquanto o sprite é consistente).
  - Frame mais claro (ou score máximo) contra o **piscar das luzes**.
  - Entra em `capture.py` e `_capturar_janela`.
- **Movimentos súbitos:**
  - Transformar o surto de estática da câmera observada (quando um animatrônico se move) em um
    **estado** "câmera perturbada".
  - Memória por câmera: comparar a vista limpa atual com a última vez que a mesma câmera foi
    vista. Alinhar antes com `cv2.phaseCorrelate`, porque as câmeras fazem pan. Isso se
    autocalibra, sem templates manuais.
- **Observação redesenhada** (muda o shape, então entra junto com a Fase 3):
  - Recortes em resolução maior (viewport da câmera + faixas das portas) no lugar da janela
    inteira em 84×84.
  - Canais: frame limpo e diferença contra a última vista.
  - Mais 2 frames empilhados.
- **Depois:** áudio por loopback WASAPI. A corrida do Foxy, as batidas na porta e a cozinha são
  pistas sonoras imunes à estática.

### Fase 3 — off-policy com demos + correções humanas (depois do M1)

- **Guardar TODA transição em disco** (`dados/transicoes/`). Até hoje ~750k transições reais
  foram descartadas depois de 4 épocas do PPO.
- **Algoritmo no estilo DQfD:** Double/Dueling DQN, n-step (n≈10–20), replay priorizado e as
  demos **permanentes** no buffer, com uma perda de margem que decai.
  - O treino roda numa thread em segundo plano durante o jogo (a GPU fica ociosa nos ~0,9 s de
    cada step), com razão updates/dados ≥ 4.
  - Isso elimina por construção o "RL corrói o BC".
  - Fica em `src/agent/dqfd.py`, chamado por `main.py treino --algo dqfd`. O PPO continua intacto.
  - Validar **offline, só com as demos**, antes de tocar no jogo.
- **HG-DAgger:** o agente joga e o humano assume o controle, pelas mesmas teclas do gravador,
  quando ele vai errar. Só esses trechos viram rótulo.
  - Exige menos habilidade que jogar a noite inteira. As 8 demos atuais são todas vitórias,
    inclusive N2 e N3.
  - Mira exatamente os estados em que o agente falha.
  - Prioridade: N2 (hoje há só 2 demos) e fechamentos de porta.

### Opcional — run 5 como baseline

É barata em atenção, mas ocupa a máquina do jogo. Retomar a run 4 no PC-LUCIANO com o código
atual (currículo + âncora) enquanto as Fases 2 e 3 são desenvolvidas. Antes disso, ajustar
`H_INICIO` para ≈0,85 (o H real do clone). Critério de aborto herdado: N1 < 30% por uma janela
cheia significa que a âncora não segurou.

## 6. Hipóteses abertas (não provadas)

- **Descasamento de tick no BC.** As demos têm ~0,33 s por registro e o agente ~0,9 s por step.
  A LSTM aprende a dinâmica a ~3 Hz e roda a ~1,1 Hz; com ~82% de "nada" por tick, o clone agiria
  ~1/3 das vezes por minuto em relação ao humano. Medir com o M3.
- **Quem mata na N2.** "É o Foxy" é inferência (morte cedo, bateria sobrando, flag=0);
  `morte_animatronico` não identifica o animatrônico.
- **O Freddy está inativo nas noites 1–2.** Conhecimento do jogo, não verificado neste ambiente.
- **A CNN contribui.** Nunca medido (M2).
- **A rajada resolve a estática e o piscar.** Raciocínio sólido, não medido (M4).

## 7. Pendências

- **Checkpoints da run 4** estão só no PC-LUCIANO (neste PC, `modelos/` tem apenas `fnaf_bc.zip`).
  O checkpoint da run 3 (175k, prefixo `pc4`) pode estar lá também.
- **Organizar os logs e o TB do PC-LUCIANO** do mesmo jeito que foi feito aqui. Sem isso, a
  primeira run nova lá recebe o número `run1`, porque o número vem do maior `runN` em
  `logs/tensorboard/`.
- **Noites 6/7** têm botões de menu próprios (o Continue não chega nelas). Calibrar quando a N5
  estiver dominada.
- **`morte_animatronico` é genérico** (não separa Foxy de Freddy). É melhoria de telemetria a
  considerar se isso bloquear uma decisão da Fase 2.
