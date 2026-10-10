# Diagnóstico — "segurar a porta" com o RL 190k na N2 (09/10/2026)

> Registro dos números; dados brutos em `logs/analise/trace_20261009_1251.csv` (podem ser apagados).
> `jogar --estocastico --noite 2 --episodios 50 --trace --segurar-porta`, commit `29121e7`,
> 09/10 12:51–18:54, PC pc4. A trava `--segurar-porta` (só avaliação) ignora "abrir a porta
> esquerda" enquanto a flag do Bonnie está acesa. Pergunta: qual o teto se a política aprender a
> segurar a porta?

## Resultado (N2)

| | Sem trava de alinhamento (08/10) | Com a trava (09/10, 2 rodadas) | **Segurar porta** |
|---|---|---|---|
| Noites | 70 | 102 | 50 |
| Vitórias | 1 | 1 | **2** (ambas por apagão) |
| Sobrevivência mediana (média) | 262 s (276) | 302 s (298) | **351 s (353)** |
| Porta esquerda fechada | 12,1% dos steps | 4,9% | 12,7% |
| Duração fechada (mediana) | 3 s | 5 s | **21 s** |
| Mortes: flag acesa + porta aberta | 6 | 55 | **2** |
| Mortes: flag apagada + porta aberta | 62 | 44 | **46** |
| Energia na morte (mediana) | 47% | 42% | 30% |
| Causas | 68 animatrônico | — | 40 animatrônico, **8 energia** |

- Segurar porta × só a trava: **+49 s** na mediana (permutação p = 0,055). × sem trava: +89 s (p = 0,006).
- A política tentou reabrir com a flag acesa **806 vezes** em 50 noites (~16 por noite).
- Na morte, a última luz esquerda foi há ~21 s (mediana), igual às outras rodadas.

## Leitura

1. **Segurar a porta enquanto a flag está acesa elimina as mortes "sabendo"** (55 → 2) e dá o maior
   ganho de sobrevivência medido até agora.
2. **Não basta para vencer a N2.** Sobram:
   - **checagem tardia**: quase todas as mortes restantes são com a flag apagada e a porta aberta,
     ~21 s depois da última olhada na luz esquerda (Bonnie ou Foxy chegando entre checagens);
   - **energia**: segurar a porta custa bateria; com a câmera aberta ~50% do tempo, o orçamento não
     fecha (8 mortes por apagão, energia na morte 42% → 30%).
3. Para o treino do zero: a política precisa aprender **três** comportamentos juntos — segurar a
   porta com a flag acesa, checar a luz esquerda com regularidade e gastar menos câmera. A
   observação já carrega a informação das três (flag correta, idade da informação, energia).
