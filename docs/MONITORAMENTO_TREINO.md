# Monitoramento do treino — métricas por noite, currículo e checkpoints

Guia prático para acompanhar o treino do agente e decidir **(a)** quando alternar o método de
reset (`new_game` ↔ `continue`) e **(b)** até qual checkpoint é seguro voltar.

---

## 1. Por que a "Taxa vitória" do log engana

A linha do `LogCallback` em `logs/treino.log` mostra uma **taxa de vitória agregada**:

```
taxa_vitoria = vitorias_de_todas_as_noites / episodios_validos
```

Com `FNAF_RESET_METODO=new_game`, uma morte volta o agente para a **noite 1**, então a
esmagadora maioria dos episódios é noite 1. Esse número agregado é, na prática, "win rate da
noite 1 levemente diluído". Pior: conforme o agente **domina** a noite 1, ele chega mais na
noite 2 e **morre mais lá** → a taxa agregada **estagna ou cai** mesmo havendo progresso real.
Como sinal de progresso num currículo, ela é enganosa por construção.

**A solução:** olhar as métricas **por noite**. O campo `Noite N` já existe em cada linha do
log, então dá pra reconstruir tudo agrupando por noite — é o que o script abaixo faz.

---

## 2. As métricas (generalizadas para qualquer noite N)

Tudo medido numa **janela móvel** (ex.: últimos 150 episódios), por noite:

| símbolo | definição | para quê serve |
|--------|-----------|----------------|
| **pₙ** | win rate da noite N = `vitórias_N / (vitórias_N + mortes_N)` | desempenho real **naquela** noite |
| **nₙ** | nº de episódios **válidos** na noite N na janela | **fome de dados** — é isto que `continue` conserta |
| **tₙ** | tempo médio de sobrevivência **entre as mortes** da noite N (min) | progresso contínuo quando pₙ ainda é ~0 |
| **tendência de tₙ** | média da 2ª metade das mortes − da 1ª metade | distingue *aprendendo devagar* (sobe) de *preso* (plateau) |
| **frente** | a noite mais avançada já alcançada na janela | onde está a fronteira do currículo |

Por que **nₙ** é central: com `new_game`, em regime, `episódios_noite_{N+1} ≈ pₙ ×
episódios_noite_N`. Se `p₁ = 0.3`, a noite 2 recebe só ~30% das amostras da noite 1 → já está
faminta. `continue` existe justamente para **inflar nₙ na frente** (o agente "mora" na noite
onde morreu em vez de voltar para a 1).

### 2.1. Tipo de desfecho — separa *skill* de *sorte* (campo `Causa:`)

A win rate sozinha mistura vitórias por **gestão** com vitórias por **sorte do apagão**. A causa
de cada desfecho terminal fica registrada no log **detalhado** (`logs/analise/treino_detalhado.log`
— o `treino.log` de execução fica enxuto de propósito), e o script quebra por noite:

| rótulo | o que é | o que diz |
|--------|---------|-----------|
| `vitoria_gerida` | venceu com energia de pé | vitória por **skill** (gestão do recurso) |
| `vitoria_apagao` | venceu **após** a energia zerar | venceu **no fio** — o 6 AM chegou antes do Freddy pós-blackout. Alta variância; infla a win rate sem refletir competência |
| `morte_energia` | apagou e o Freddy pegou | gargalo = **energia** (atacar a parede de energia rende mais que detecção de ameaça) |
| `morte_animatronico` | morreu **com** energia | gargalo = **defesa/timing** (animatrônico passou a porta) |

Leitura prática: **% de vitórias por apagão alto** → sua win rate superestima a skill, espere
variância (pode "regredir" só por azar do RNG). **% de mortes por energia alto** → o agente já
sobrevive a noite quase inteira e o que falta é *economizar energia*, não defender melhor. O
script imprime a tabela e um diagnóstico automático da noite com mais amostra.

---

## 3. Como rodar

```bash
python scripts/metricas_treino.py                 # janela padrão = 150 eps
python scripts/metricas_treino.py --janela 200
python scripts/metricas_treino.py --log logs/analise/treino_detalhado.log --janela 100
# (sem --log, o script já prefere o detalhado e cai no treino.log enxuto se não existir)
```

Saída (exemplo):

```
Noite |    n |   vit% |  vit morte  int | t_morte(min) | tendencia
------------------------------------------------------------------
    1 |  120 |  55.0% |   66    54    0 |          5.1 |      +0.4
    2 |   22 |   4.5% |    1    21    0 |          5.8 |      +0.1

Frente = noite 2 | noite 1 (abastece): 55.0% | frente: 4.5% em 22 eps | tendencia: +0.1 min
>> SUGESTAO: trocar TEMPORARIAMENTE para FNAF_RESET_METODO=continue ...
```

Dá pra deixar rodando ao lado do treino (ele lê o log em disco, não interfere).

---

## 4. Regra de decisão — `new_game` ↔ `continue`

Seja **F** a noite-frente e **F−1** a anterior. Troque para `continue` **temporariamente**
quando, na janela, **todos** valerem:

- **p₍F−1₎ ≥ ~50%** — a noite anterior *abastece* a frente (o agente chega lá com frequência);
- **n₍F₎ < ~25** — a frente está *faminta* de amostras;
- **p₍F₎ < ~10%** com **tendência ≈ 0 (plateau)** — está *preso*, não só lento.

E **volte para `new_game`** assim que **p₍F₎** encostar (~20–30%).

Casos em que **NÃO** se troca:
- **Tendência de t₍F₎ subindo** → está aprendendo, só devagar. Trocar seria precipitado.
- **p₍F−1₎ baixo** → o gargalo ainda é a noite F−1; resolver ela primeiro.

Os números (50%, 25, 10%) são **pontos de partida** — calibre olhando seus próprios dados. O
script já aplica essa heurística e imprime uma sugestão; trate-a como dica, não veredito.

> **Como alternar:** edite `FNAF_RESET_METODO` no `.env` (`new_game` ou `continue`) e
> reinicie o treino. A troca só vale em processo novo (o `.env` é lido na importação).
> A **noite-alvo** do `continue`, por outro lado, é promovida AUTOMATICAMENTE pelo
> `CurriculumCallback` (50% na janela de 30 eps da noite alvo) e persiste em
> `modelos/curriculo.json` — não precisa editar `FNAF_NOITE_DESEJADA` a cada avanço.

### Cuidado com `continue` por tempo demais
`continue` faz o agente "morar" numa noite só. Em excesso, arrisca:
- **esquecimento catastrófico** da noite 1 (a política deriva para a noite presa);
- a feature `noite` virar refém de uma *streak* longa, atrapalhando a generalização.

Por isso a regra diz **temporariamente**: é um *booster* pontual para destravar a frente, não
um regime permanente.

---

## 5. O que observar ao longo do tempo (saúde)

- **pₙ por noite subindo** na janela = currículo avançando. O número agregado do log pode
  enganar; confie no por-noite.
- **frente aumentando** (noite máx alcançada sobe) = progresso de verdade.
- **Mortes marcando `MORTE` (−100)** e não `INTERROMPIDO` quando o Golden fecha o jogo —
  confirma a correção do desfecho funcionando (janela fechada = morte por Golden Freddy).
- **`INTERROMPIDO` raro** — interrupção neutra (0.0) só deve aparecer em falha transitória de
  captura. Se aparecer muito, investigue captura/foco de janela, não o agente.
- **Recompensa por episódio**: vitória ~+500, morte ~−100 + denso. Um episódio de morte com
  recompensa **positiva** é sinal de bug no desfecho (era o caso antigo do Golden = +12).

---

## 6. Checkpoints — quando é seguro voltar

O treino salva um checkpoint a cada **10.000 steps** em `modelos/`. Agora cada save **anuncia
no terminal** com contexto de episódio (via `CheckpointComLog`):

```
[CHECKPOINT] salvo: pc1_fnaf_ppo_120000_steps.zip
   step 120,000 | tirado DURANTE o ep 318 (ultimo ep CONCLUIDO: 317)
   ate aqui: 290 eps validos | 96 vitorias (33.1%) | noite max alcancada: 2
```

Como ler para decidir um rollback seguro:
- **arquivo** — o `.zip` exato para o qual voltar;
- **"durante o ep X / último concluído"** — o checkpoint é tirado **entre steps**, então pega o
  episódio em andamento no meio. O último episódio íntegro nele é o "último CONCLUÍDO";
- **eps válidos / vitórias (%) / noite máx** — a *saúde* naquele instante. Um bom ponto de
  retorno é um checkpoint **antes** de uma degradação (taxa caindo, frente regredindo, erros
  pipocando) e **com** a métrica saudável.

**Estratégia prática:** se o treino azedar (muitos erros, métricas piorando por várias janelas),
volte ao **último checkpoint cujo print mostrava saúde boa** — tipicamente o anterior ao começo
da queda.

### Como voltar na prática
`python main.py treino` carrega automaticamente o checkpoint de **maior step** em `modelos/`
(via `encontrar_ultimo_modelo`). Para voltar a um anterior:

1. **Mova** os `.zip` mais novos que o desejado para fora de `modelos/` (ex.: uma pasta
   `modelos/descartados/`). O desejado passa a ser o "mais avançado".
2. Rode `python main.py treino` normalmente.

**Normalização:** cada checkpoint salva JUNTO o seu `*_vecnormalize_N_steps.pkl` pareado
(`save_vecnormalize=True`), e ao retomar o treino prefere o `.pkl` do próprio checkpoint —
o rollback volta pesos **e** stats de normalização casados. Só checkpoints muito antigos
(anteriores a essa mudança) caem no `modelos/vecnormalize.pkl` global do fim do run.

---

## Referências no código
- Métricas/logs: `src/agent/train.py` (`LogCallback`, `CheckpointComLog`).
- Desfecho de morte/Golden e método de reset: `src/environment/fnaf_env.py`
  (`_interromper_episodio`, `_preparar_reset`/`decidir_reset`, `RESET_METODO`, `NOITE_DESEJADA`).
- Script: `scripts/metricas_treino.py`.
