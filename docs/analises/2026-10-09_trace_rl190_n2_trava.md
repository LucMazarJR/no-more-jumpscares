# Análise — RL 190k na N2 com a trava de alinhamento (09/10/2026)

> Registro dos números; dados brutos em `logs/analise/trace_20261008_2313.csv` (podem ser apagados).
> Mesmo modelo e configuração do [relatório de 08/10](2026-10-08_trace_rl190_n2.md); a única
> mudança é a trava de alinhamento na janela do Bonnie (commit `ec98fc2`).

## Configuração

- `pc4_fnaf_ppo_190000_steps.zip`, `--estocastico --noite 2 --episodios 60 --trace`, commit
  `0dacbbd`, 08/10 23:13 → 09/10 04:59, PC pc4.

## Antes × depois da trava (N2)

| | 08/10 (sem trava) | 09/10 (com trava) |
|---|---|---|
| Episódios | 70 | 60 |
| Vitórias | 1 (apagão) | 0 |
| Sobrevivência mediana | 262 s | **314 s** (+52 s; permutação p = 0,08) |
| Sobrevivência média | 276 s | 299 s |
| Flag esquerda acesa | 73% dos steps | **12%** (a reconstrução de 08/10 previa 15%) |
| Steps de flag acesa sem silhueta vista no episódio | 9.479 | **5** |
| Avistamentos contraditos logo em seguida por "vazio" | 20 de 326 | 4 de 231 |
| Fechamentos da porta esquerda por noite | 6,3 | 1,9 |
| Duração fechada (mediana) | 3 s | 5 s |
| **Reaberturas com a flag acesa** | 95% (com a flag ruidosa) | **85%** (99 de 116, com a flag certa) |
| Mortes com a flag acesa e a porta aberta | 6/69 | **34/60** |
| Mortes com a flag apagada e a porta aberta | 62/69 | 25/60 |
| Câmera aberta | 47% dos steps | 50% |
| Gasto de energia | ~12%/min | ~11%/min |

Mortes por hora (09/10): 1AM 4 · 2AM 19 · 3AM 20 · 4AM 13 · 5AM 4 (antes o pico era às 2AM).
Na morte: última luz esquerda há 22 s (mediana); porta esquerda aberta há 41 s.

## Leitura

1. **A percepção do Bonnie está resolvida.** A flag agora só acende com um avistamento real, e o
   agente fecha menos vezes à toa (6,3 → 1,9 por noite).
2. **O que sobra é a política.** Em 85% das reaberturas o agente **sabia** que o Bonnie estava na
   porta. Ele foi treinado com uma flag que acendia e apagava ao acaso, então "flag acesa" nunca
   significou "mantenha fechado" para ele.
3. **Duas classes de morte:** 34/60 com a informação certa e a porta aberta (política); 25/60 com a
   flag apagada (o agente não checou a luz a tempo: última checagem 22 s antes da morte).

## Réplica (09/10, 08:14–12:42, `trace_20261009_0814.csv`)

Mesmo comando e configuração (o `--segurar-porta` ficou de fora por engano), 42 episódios de N2:
1 vitória (apagão), sobrevivência mediana 283 s, flag acesa 16%, **reaberturas com a flag acesa
93/105 (89%)**, mortes com a flag acesa e a porta aberta 21/41. Juntando as duas rodadas com a
trava (102 eps): mediana 302 s contra 262 s sem a trava, **+40 s, p = 0,20** — a melhora de
sobrevivência não se sustenta; o padrão "reabre sabendo" se repete.

## Próximo passo proposto

Diagnóstico `jogar --segurar-porta`: ignora "abrir a porta esquerda" enquanto a flag está acesa.
Mede o teto do que o retreino pode ganhar ao aprender a segurar a porta, e se a energia aguenta.
