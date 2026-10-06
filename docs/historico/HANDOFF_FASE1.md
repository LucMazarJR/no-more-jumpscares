# Handoff — Fase 1 no outro PC (avaliações em blocos)

> **Este documento é um PROMPT** para o agente que roda no outro PC (o de treino). Leia inteiro
> antes de agir. Idioma do projeto: **português**. É **transitório**: quando a Fase 1 acabar, ele
> vai para `docs/historico/`.

---

## 0. Seu objetivo

**Rodar blocos de avaliação do BC puro (sem treinar nada) e devolver os resultados.** Você não
decide arquitetura nem muda código ou configuração. A análise junta os dados dos dois PCs e aplica
as regras de decisão já registradas em [ESTADO_ATUAL.md](ESTADO_ATUAL.md) §5.

**Por que isso importa**, em uma frase: nas 4 runs, a vitória na Noite 1 ficou sempre no nível
que o BC entregou. Medir o BC sozinho prova (ou refuta) que o RL não agregou nada. Contexto
completo: [ESTADO_ATUAL.md](ESTADO_ATUAL.md) §1–2 e §5.

## 1. As medições

| Medição | Comando (1 bloco ≈ 1 h) | Total, somando os 2 PCs |
|---|---|---|
| **M1** — BC puro | `venv\Scripts\python main.py jogar --modelo modelos\fnaf_bc.zip --estocastico --noite 1 --episodios 5` | 4 blocos (20 eps de N1) |
| **M2** — BC sem a imagem | o mesmo comando + `--ablacao imagem` | 4 blocos (20 eps de N1) |

- **Pergunte ao usuário** quantos blocos de cada medição já foram feitos no PC 1. Ele começou
  pelo M1 lá.
- **A comparação M1 × M2 é feita dentro do mesmo PC.** Se este PC rodar blocos de M2, rode aqui
  também pelo menos 2 blocos de M1. O ideal é alternar: M1, M2, M1, M2. Máquinas diferentes têm
  tempo de step diferente, e isso não pode se confundir com o efeito da ablação.
- Cada episódio de N1 leva ~11 min, contando o episódio de N2 que vem de brinde quando ele vence.

## 2. Pré-voo (antes do primeiro bloco neste PC)

Faça tudo e **pare, reportando ao usuário, se qualquer item falhar**.

1. **Código:** `git pull` no `develop`. Confirme com `git log --oneline -5` que o commit
   `5cb336b` (`feat(jogar): registra o PC e o s/step...`) ou um posterior está presente. Sem ele,
   o log não registra o PC e os blocos dos dois PCs não se separam.
2. **Modelo — o MESMO arquivo do PC 1.** O `modelos/` é git-ignorado, então confira o hash:
   ```
   (Get-FileHash modelos\fnaf_bc.zip -Algorithm SHA256).Hash
   ```
   Tem que dar `74436089BB1AA395FB4D28FBA683FCF9558F3C152EFE8E1FD3F3B0E176DCFF70`. Se o arquivo
   faltar ou o hash for outro, **pare** e peça ao usuário para copiar o arquivo do PC 1. Não
   retreine o BC: seria outro modelo.
3. **`.env` deste PC:**
   - `FNAF_RESET_METODO=continue` é obrigatório (é o que faz o `--noite 1` funcionar).
   - `PC` tem que estar preenchido e **diferente de `1`**, que é o PC 1. Os logs antigos deste PC
     usavam `pc4`. Se estiver vazio ou igual a `1`, peça ao usuário para definir.
   - Compare os timings com os do PC 1: `FNAF_STEP_DELAY=0.35`, `FNAF_SIDE_SWITCH_DELAY=0.85`,
     `FNAF_CAMERA_EXIT_DELAY=0.65`, `FNAF_CAMERA_DRAG_PIXELS=80`, `FNAF_CAMERA_DRAG_DURATION=0.15`,
     `FNAF_VITORIA_ESPERA_SEGUNDOS=20`. Se algum for diferente, **não mude** (é a calibração desta
     máquina). Só registre no relatório: a análise separa por PC.
4. **Testes offline** (todos devem passar):
   ```
   venv\Scripts\python scripts\smoke_test.py
   venv\Scripts\python -m src.utils.testar_recompensa
   venv\Scripts\python -m src.utils.testar_noite
   venv\Scripts\python -m src.utils.testar_deteccao_ameaca
   ```
5. **Jogo:** em modo janela (ALT+ENTER), na posição em que este PC foi calibrado e na tela do
   menu. Nada pesado rodando em paralelo. Ninguém mexe no mouse durante um bloco: ele é do agente.

## 3. Regras durante a execução

- **Tamanho do bloco fixo** (`--episodios 5`). Nunca pare ou estenda um bloco por causa do
  resultado: isso enviesa a medição.
- **Evite Ctrl+C no meio de um episódio.** O agente segura o botão do mouse enquanto a luz está
  acesa, e interromper nesse instante pode deixar o botão "preso" (um clique esquerdo solta). Se
  precisar parar, o episódio incompleto é descartado, o que é correto.
- **Não rode `main.py treino`.** Um `--novo` arquivaria os logs e criaria uma run. Não mude
  código, constantes ou `.env`.
- Se o `jogar` quebrar (é a primeira vez que a versão nova roda com o jogo), **pare e reporte o
  traceback**. A correção é feita no PC 1 e chega por `git pull`.
- Se o jogo fechar sozinho (Golden Freddy), o env reabre. Se travar, anote e comece outro bloco.

## 4. Relatório (ao fim de cada sessão de blocos)

1. Rode `venv\Scripts\python scripts\resumo_avaliacoes.py` e cole a saída. Ele separa por
   configuração e PC e mostra vitórias, sobrevivência mediana, ações/min e s/step por noite.
2. Informe:
   - o hash do BC;
   - o commit;
   - o `PC`;
   - os timings do `.env` (e se diferem dos do PC 1);
   - quantos blocos de cada medição foram feitos;
   - incidentes.
3. Olhe `logs/analise/avaliacoes_desyncs.log` e reporte a média de `porta falha` e `SYNC camera`
   por episódio. Na run 4 foi ~0,2. Muito acima disso indica percepção ruim nesta máquina, e o
   resultado dela fica suspeito.
4. **Devolva os logs ao PC 1.** Eles são git-ignorados, então **não commite**: copie por
   pendrive ou nuvem.
   - `logs/analise/avaliacoes.log` → `logs/analise/avaliacoes_pc<N>.log` no PC 1;
   - `logs/analise/avaliacoes_desyncs.log` → `logs/analise/avaliacoes_desyncs_pc<N>.log`.
   - No PC 1, o resumo conjunto sai com
     `python scripts/resumo_avaliacoes.py logs/analise/avaliacoes.log logs/analise/avaliacoes_pc<N>.log`.
5. **Não tire conclusões de arquitetura.** Relate os números. Se algo parecer contradizer o
   ESTADO_ATUAL, diga, com o dado.

## 5. Opcional (só se o usuário pedir)

- **BC contra BC+RL na mesma máquina.** Liste `Get-ChildItem modelos\*.zip | Sort-Object
  LastWriteTime` e procure o checkpoint final da run 4 (5–7/ago/2026, prefixo `pc4_fnaf_ppo_*`
  ou `fnaf_ppo_final.zip`). Se existir, rode blocos de M1 com `--modelo <esse checkpoint>` no
  lugar do BC. É a comparação mais limpa de todas, porque é o mesmo PC.
- **Inventário para as pendências** (ESTADO_ATUAL §7). Liste o que existe em `logs/`,
  `logs/tensorboard/` e `modelos/` e reporte. **Não mova nem apague nada**: a organização dos
  logs deste PC é decidida com o usuário.
