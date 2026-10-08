# Guia de comandos — o que rodar para cada tarefa

> Guia prático para o dia a dia. Todos os comandos rodam da **raiz do projeto**, num terminal com
> o venv já ativo (por isso é só `python`). O "onde estamos" fica no
> [ESTADO_ATUAL.md](ESTADO_ATUAL.md); este guia diz **como** fazer cada coisa e por quê.

---

## 0. Antes de qualquer coisa com o jogo aberto

- Jogo em **modo janela** (ALT+ENTER), na posição em que o `.env` deste PC foi calibrado.
- Notebook **na tomada** (na bateria ele suspende) e **"Não incomodar"** ligado (uma notificação
  rouba o foco do jogo).
- Enquanto o agente joga, **o mouse é dele**. Não mexa.

Os scripts que encadeiam blocos (`rodar_fase1.py`) reabrem o jogo sozinhos; os outros esperam o
jogo já aberto, no menu.

## 1. Conferir que está tudo funcionando

**Sem o jogo (segundos):** confirma que o código não quebrou depois de um `git pull` ou de uma
mudança. Todos devem passar.

```
python scripts/smoke_test.py
python -m src.utils.testar_recompensa
python -m src.utils.testar_noite
python -m src.utils.testar_deteccao_ameaca
python -m src.utils.testar_masking
```

**Com o jogo aberto:** confirma que a captura e o reset funcionam neste PC.

```
python main.py teste
```

## 2. Ver resultados

| Quero saber… | Comando |
|---|---|
| Como foram as avaliações (`jogar`) | `python scripts/resumo_avaliacoes.py` |
| …juntando logs de 2 PCs | `python scripts/resumo_avaliacoes.py logs/analise/avaliacoes.log logs/analise/avaliacoes_pc1.log` |
| Vitória por noite de um treino passado | `python scripts/metricas_treino.py --log logs/analise/historico/<run>_detalhado.log` |
| Se o agente age tanto quanto eu | `python scripts/taxa_acao.py` |
| Os gráficos dos treinos | `tensorboard --logdir logs/tensorboard` e abrir o link no navegador |

**Por que olhar as vitórias geridas e as causas, não só a taxa de vitória:** vitória por apagão é
sorte, e morte por energia e morte por animatrônico pedem correções opostas.

## 3. Avaliar um modelo (sem treinar)

Um bloco, com você olhando:

```
python main.py jogar --modelo modelos/fnaf_bc.zip --estocastico --noite 1 --episodios 5
```

- `--modelo`: qual zip avaliar (o BC ou qualquer checkpoint).
- `--estocastico`: sorteia a ação como no treino. Sem ele, o resultado não é comparável aos logs
  de treino.
- `--noite 1 --episodios 5`: mira a Noite 1 e para depois de 5 episódios dela. Cada vitória rende
  um episódio de Noite 2 de brinde.
- `--ablacao imagem`: zera a imagem para medir se a CNN ajuda.
- `--trace`: grava cada step num CSV (`logs/analise/trace_<data>.csv`). Depois,
  `python scripts/analisar_trace.py --detalhe 12` mostra o que o agente fazia antes de cada morte
  (porta, luz, leitura da janela do Bonnie, flag) e se ele abriu a porta com a flag acesa.

**A noite inteira, sem você** (intercalando modelos, que é o único jeito justo de comparar):

```
python scripts/rodar_fase1.py --blocos M1,RL,RL340 --rodadas 4 --ate 08:00
```

- `--blocos`: a ordem de uma rodada. Opções: `M1` (BC), `M2` (BC sem imagem), `RL` (run 4, 190k),
  `RL340` (run 4, 340k). Para outro modelo, adicione-o em `CONFIGS` no script.
- `--ate`: não começa um bloco que terminaria depois dessa hora. Cada bloco leva ~50–60 min.
- Teste rápido sem abrir o jogo: `--rodadas 0` (só faz as checagens).

## 4. Diagnóstico da percepção

| Para quê | Comando |
|---|---|
| Ver ao vivo o que o agente "enxerga" (ameaça, energia) | `python -m src.utils.monitor` |
| Achar coordenadas de uma região da tela | `python -m src.utils.inspecionar` (salva uma grade em `debug/`) |
| Medir estática, piscar e latência de captura | `python -m src.utils.sonda_captura --rotulo <nome> [--segurar-luz esq\|dir] [--frames 600]` |
| Recalibrar os cliques (mudou tela ou posição da janela) | `python -m src.utils.calibrar_por_passos` |

A sonda grava em `debug/sonda_<rotulo>.json` e **sobrescreve** a execução anterior com o mesmo
rótulo. Use rótulos diferentes se quiser guardar várias.

## 5. Dados humanos e Behavioral Cloning

```
python -m src.utils.gravar_gameplay --noite 2
```

Leve o jogo até a noite, aperte **F9** para começar, jogue **pelas teclas** (a lista aparece no
terminal) e aperte **F10** no 6 AM. Uma noite por gravação. Vai para `dados/gameplay_<data>_noiteN/`.

```
python main.py bc
```

Treina o BC com todos os `dados/*/dataset.json` e gera `modelos/fnaf_bc.zip`. **Atenção:** isso
substitui o BC atual. Faça uma cópia antes se quiser comparar.

## 6. Treinar

O PPO não é o caminho atual (ver ESTADO_ATUAL §1). Fica aqui para referência.

```
python main.py treino --novo --bc modelos/fnaf_bc.zip --nome <tag>
python main.py treino
```

- O primeiro começa uma run nova a partir do BC. Ela ganha um nome
  (`AAAA-MM-DD_runN_<algo>_<tag>`), e os logs da run anterior vão para o histórico.
- O segundo retoma a run atual do maior checkpoint.
- Segure **F12** para pausar.

## 7. Git e os 2 PCs

Rotina:

```
git status
git add <arquivos>
git commit -m "tipo: descrição"
git push
```

O formato da mensagem é Conventional Commits: `feat`, `fix`, `docs`, `refactor`, `chore`…

Ao trocar de PC:
1. `git pull` no PC de destino.
2. Copie por pendrive ou nuvem o que **não vai pelo git**: `modelos/`, `dados/`, os logs que
   interessam em `logs/analise/` e as fixtures `debug/quadro_*.png`.
3. Confira que o modelo é o mesmo arquivo:
   `(Get-FileHash modelos\fnaf_bc.zip -Algorithm SHA256).Hash`.
4. O `.env` é de cada máquina (coordenadas, `PC`). Não copie de um PC para o outro.

## 8. Onde fica cada coisa

O mapa completo está no [CLAUDE.md](../CLAUDE.md). O essencial:

- `logs/analise/avaliacoes.log`: resultados do `jogar`.
- `logs/analise/historico/`: logs das runs passadas, uma por arquivo.
- `logs/tensorboard/<run>/`: gráficos de cada run.
- `modelos/`: o BC e os checkpoints. `modelos/pc_luciano_run4/` guarda os da run 4.
- `debug/`: saídas de ferramentas (transientes) e as fixtures dos testes.
