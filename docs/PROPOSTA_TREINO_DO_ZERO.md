# Proposta — treino do zero com a percepção corrigida (out/2026)

> **Aprovada em 9/out** (decisões na §5). Baseada em
> [analises/2026-10-09_diagnostico_segurar_porta.md](analises/2026-10-09_diagnostico_segurar_porta.md)
> e nas análises anteriores. Preferência registrada do usuário: **maior chance de sucesso, mesmo que
> leve mais tempo**; não precisa reaproveitar o 190k.

## 1. Por que do zero

- Todo modelo até hoje (BC e PPO) aprendeu com a flag do Bonnie quebrada: primeiro a contagem
  acumulada, depois o desalinhamento da visão. Em 08/10 a flag recebida era ruído (73% acesa).
- A coluna de ameaça das 8 demos humanas foi gravada com o mesmo detector, e os frames 84×84 não
  permitem recalcular. Um BC refeito com elas herdaria o erro.
- O RL 190k, mesmo com a flag correta, reabre a porta com a flag acesa em 85–89% das vezes: o hábito
  errado está nos pesos.
- O diagnóstico de 09/10 mostrou o que a política nova precisa aprender:
  1. **segurar a porta enquanto a flag está acesa** (sozinho: +49 s de sobrevivência na N2, mortes
     "sabendo" de 55 para 2);
  2. **checar a luz esquerda com regularidade** (as mortes restantes vêm ~21 s depois da última olhada);
  3. **gastar menos câmera** (aberta ~50% do tempo; segurar a porta já leva a 8 mortes por apagão).

## 2. O que fica e o que muda

| Fica | Muda |
|---|---|
| Ambiente, atuação, reset, reabertura do jogo | **Demos novas**, gravadas com a percepção certa |
| Percepção corrigida (máscaras + trava de alinhamento) | **BC novo** a partir delas |
| Observação no mesmo formato (14 estados + 84×84) | RL a partir do BC novo (ver §3, Etapa 3) |
| Recompensa e shaping (incl. bloqueio + idade da informação) | PyTorch com GPU (hoje roda tudo na CPU) |

A observação **não** é redesenhada nesta rodada (Fase 2, câmeras/Foxy, fica para depois): uma rodada
já muda muita coisa, e o diagnóstico mostrou que a informação necessária já está na observação.

## 3. Etapas

**Etapa 0 — GPU (minutos).** Instalar o PyTorch com CUDA 12.8 (RTX 5060):
`python -m pip install "torch==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128`
(~2,8 GB). Conferir com o smoke test e medir o BC na GPU. Sem isso o DQfD (Etapa 3B) disputaria a
CPU com o jogo.

**Etapa 1 — demos novas (precisa de você; ~2–3 h somando sessões).** `gravar_gameplay` já usa a
percepção nova, então a flag sai certa. Sugestão: 3× N1, **6–8× N2**, 2× N3. Na N2, jogar com a
disciplina que o diagnóstico pediu: checar a luz esquerda a cada poucos segundos, **segurar a porta
fechada enquanto o Bonnie estiver lá** e não ficar muito tempo na câmera. Não precisa vencer todas.

**Etapa 2 — BC novo (offline, horas).** `python main.py bc` com as demos novas (as antigas podem
entrar para N1 se a análise mostrar que a flag errada não atrapalha — ❓). Avaliar com
`jogar --estocastico --trace`, 20 eps de N1 e 20 de N2, olhando além da vitória: taxa de reabertura
com a flag acesa e duração da porta fechada.

**Etapa 3 — RL a partir do BC novo, em dois braços (❓ decisão sua):**

- **A) RecurrentPPO** (código pronto): âncora BC **ligada** (na continuação de agosto ela estava
  desligada), `H_INICIO` igual à entropia que o BC entregar, currículo mais conservador (promover
  só com N1 ≥ 50% de vitórias **geridas**). Começa assim que o BC estiver pronto.
- **B) DQfD** (a implementar): off-policy, replay em disco com as demos permanentes, treino na GPU
  em paralelo ao jogo. Implementação e **validação offline** (tem de igualar o BC nas demos) enquanto
  o braço A roda; depois roda online com o mesmo orçamento de steps, para comparar A × B nas mesmas
  demos e na mesma percepção.

Recomendação: **os dois**, nessa ordem. O A dá resultado rápido com código conhecido; o B é a aposta
de maior teto (aproveita cada amostra e não esquece as demos). A comparação é limpa: muda só o
algoritmo.

**Etapa 4 (depois) — correções ao vivo (HG-DAgger)** nas situações em que o agente ainda erra.

## 4. Critérios registrados antes de rodar (braço A; o B usa os mesmos)

Medidos com `jogar --trace` em checkpoints (sem misturar com o log de treino).

| Marco | Sinal de que está funcionando |
|---|---|
| BC novo | N1 ≥ 40% de vitórias; reaberturas com a flag acesa < 50% (o 190k: 85–89%) |
| RL ~50k steps | N1 com vitórias geridas ≥ 40%; porta esquerda fechada com mediana ≥ 10 s |
| RL ~150k steps | N2 com sobrevivência mediana ≥ 350 s (o teto do diagnóstico); reaberturas com flag < 30% |
| Meta | primeiras vitórias **geridas** na N2 |

**Aborto:** N1 com vitórias geridas < 25% numa janela cheia (30 eps) depois de 50k steps; ou
entropia < 0,4 com o `ent_coef` no teto; ou mortes por energia > 50% na N1.

## 5. Decisões e andamento (9/out)

1. **Etapa 0 aprovada e feita no pc4**: `torch 2.11.0+cu128` no venv. O outro PC precisa do mesmo
   comando ([GUIA_COMANDOS.md](GUIA_COMANDOS.md)).
2. **Demos da Etapa 1**: o usuário grava, nas sessões que der.
3. **Os dois braços, A primeiro.** O código do A está pronto (abaixo). O B (DQfD) se implementa
   enquanto o A roda: a validação offline dele depende das demos novas.
4. **Demos antigas fora**: movidas para `dados/_antigas_flag_errada/`, onde o BC e a âncora não as
   pegam. Para reincluir na N1 depois, é só mover a pasta de volta para `dados/`.
5. **Limpeza feita**: ~3,2 GB liberados (backups, checkpoints intermediários, `git gc`). O `.git`
   segue com ~1,9 GB de checkpoints em branches antigos; apagar é decisão do usuário.

**Ajustes do braço A já no código** (`src/agent/train.py`, teste offline
`python -m src.utils.testar_callbacks_treino`):
- **Alvo de entropia calibrado pelo BC**: numa run que parte de um BC, o termostato mede a H que o BC
  entrega no próprio jogo durante o warmup do crítico (ator quase parado, `ent_coef` no piso) e
  usa essa H como alvo inicial, no lugar da constante 1,1. O alvo nunca sobe acima dela. O valor
  fica salvo no checkpoint e volta na retomada.
- **Currículo por vitórias geridas**: só promove a N1 → N2 com ≥ 50% de vitórias **geridas** na
  janela de 30 eps. A vitória por apagão não conta. O tensorboard ganha `noite_N/win_rate_gerida`.
- **Âncora BC ligada** (já estava em `ANCORA_BC = True`): usa as demos de `dados/*/dataset.json`, que
  agora são só as novas.

## 6. Riscos

- Muitas coisas mudam juntas (demos, BC, percepção). Aceitável porque o objetivo é sucesso, e a
  comparação controlada fica entre os braços A e B.
- Demos de N2 são difíceis de jogar bem. Mitigação: não precisam ser perfeitas; o HG-DAgger corrige
  depois, e o DQfD tolera demonstrador imperfeito.
- O DQfD é código novo. Mitigação: validação offline antes do jogo, e o braço A segue de referência.
