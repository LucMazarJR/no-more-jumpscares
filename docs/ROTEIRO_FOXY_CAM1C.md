# Roteiro — percepção do Foxy na CAM 1C (passo 1 da Fase 2)

> **Para quem:** você, executando por conta própria. É um roteiro de mentor: diz o que fazer, por
> quê, onde mexer no código e o que pesquisar. O código fica com você. Ao fim de cada etapa, mostre
> o resultado para revisão antes de seguir.
>
> É **transitório**: quando a percepção do Foxy estiver integrada, vai para `docs/historico/`.

## 0. Por que isso, e por que agora (recapitulação)

- **A parede é a Noite 2:** 1 vitória em ~56 episódios, com qualquer modelo. O agente morre cedo,
  com bateria sobrando, para um animatrônico que não aparece nas flags. Só Bonnie e Chica têm flag.
- **O suspeito principal é o Foxy.** Ele mora na Pirate Cove (CAM 1C) e sai em estágios: cortina
  fechada → espiando → fora da cortina → **cortina vazia** (está correndo até a porta esquerda
  AGORA). Saber o estágio é provavelmente a informação mais valiosa do jogo, e hoje o agente não tem.
- **A estática das câmeras atrapalha a leitura.** A sonda (M4) mediu: a mediana de uma rajada de
  5 frames corta ~38% da estática na CAM 1C e custa ~70 ms. É aqui que a rajada vale.
- **Por que não template:** você prefere aprender a desenhar regras. Este roteiro começa com um
  detector simples aprendido dos seus próprios dados, e só complica se os números pedirem.

Fato do jogo a confirmar com os dados: na Noite 1 o Foxy quase não se move (a agressividade dele
começa em 0 e só sobe de madrugada). As imagens dos estágios virão sobretudo das Noites 2 e 3.

## Etapa 1 — Coletar imagens rotuladas da CAM 1C

**Objetivo:** algumas dezenas de imagens de cada estágio, em resolução cheia e já sem estática.

**O que construir:** um coletor que roda **em paralelo** enquanto você joga normalmente a N2 ou a
N3. Quando você estiver olhando a CAM 1C e apertar uma tecla, ele captura uma rajada, tira a
mediana, recorta a área da cortina e salva com o rótulo daquela tecla. Por exemplo:
`1` = fechada, `2` = espiando, `3` = fora, `4` = vazia.

**Onde se apoiar:**
- `src/utils/sonda_captura.py` já faz a rajada (`capturar_rajada`) e a mediana. Reaproveite.
- `src/utils/gravar_gameplay.py` já mostra como escutar teclas com a biblioteca `keyboard` sem
  travar o jogo (`keyboard.on_press_key` + fila). Reaproveite o padrão.
- Para achar o retângulo da cortina em coordenadas 1280×720, use
  `python -m src.utils.inspecionar`: ele salva em `debug/` uma grade com coordenadas. O recorte
  deve **excluir** o mapa das câmeras (canto inferior direito) e o texto de hora/noite.
- Salve em `dados/cam1c/<rótulo>/<data_hora>.png`. A pasta `dados/` é git-ignorada.

**Para pensar antes de codar:**
- Por que salvar em resolução cheia e não em 84×84, como as demos fazem hoje?
- As câmeras fazem pan. O recorte da cortina continua cobrindo a cortina nos dois extremos do pan?
  Teste nos dois lados antes de coletar muito.

**Pronto quando:** ≥ 30 imagens por estágio. O estágio "vazia" é o mais raro e o mais importante.

## Etapa 2 — Olhar os dados

Antes de qualquer modelo, abra as imagens e responda:
- Quantas imagens você tem por estágio?
- O que muda visualmente de um estágio para o outro? Em que parte do recorte?
- Dois estágios vizinhos (ex.: fechada e espiando) se parecem muito?

Isso decide o quão simples o detector pode ser.

## Etapa 3 — O detector mais simples que funcione

**Comece com "centróide mais próximo":**
1. Reduza cada recorte a um tamanho pequeno (ex.: 64×64) em cinza.
2. Separe ~20% das imagens de cada estágio para **validação**. Não use essas no passo seguinte.
3. Calcule a imagem **média** de cada estágio com o resto.
4. Para classificar uma imagem nova, veja de qual média ela está mais perto.

**Meça na validação:**
- a acurácia geral;
- a **matriz de confusão** (quem é confundido com quem);
- o **recall do estágio "vazia"** (das imagens que são "vazia", quantas o detector acerta).
  Errar esse é o que mata.

**O que pesquisar:** "nearest centroid classifier", "train/validation split", "matriz de
confusão", "recall", e no OpenCV `cv2.resize` com `INTER_AREA` (o jeito certo de reduzir imagem).

**Se não bastar:** o próximo degrau é uma regressão logística ou uma CNN pequena sobre os mesmos
recortes. Só suba se o centróide errar demais, e explique para si mesmo **onde** ele erra antes.

**Pronto quando:** recall de "vazia" ≥ 90% e acurácia geral ≥ 85% na validação. Esses números
são uma proposta: ajuste-os conforme o que a Etapa 2 mostrar, mas **antes** de medir.

## Etapa 4 — Integrar no ambiente (depois da revisão da Etapa 3)

A ideia, no mesmo padrão dos estados de idade da informação que já existem:
- um estado **estágio do Foxy** em [0,1], atualizado só quando a CAM 1C está aberta
  (`camera_ativa == 3` em `fnaf_env.py`; ela é `acao - 5`, e `camera_1c` é a ação 8);
- um estado **idade da última vista da 1C**, porque "vazia há 2 s" e "vazia há 60 s" são situações
  muito diferentes;
- a rajada entra **só** na captura da câmera, atrás de uma constante desligada por padrão.

Isso **muda o shape da observação**, então exige treino do zero. Por isso a integração só
acontece junto com o resto da Fase 2 e o DQfD (ver [ESTADO_ATUAL.md](ESTADO_ATUAL.md) §5).

## Perguntas para levar junto

- O agente atual abre pouco a CAM 1C? Como você mediria isso nos logs que já existem?
- Se o detector disser "vazia", qual é a ação certa e quanto tempo o agente tem?
  (Pense na distância entre a Pirate Cove e a porta esquerda.)
- O que deve acontecer com o estado do estágio quando a câmera está fechada: mantém o último
  valor, zera ou outra coisa? Por quê?
