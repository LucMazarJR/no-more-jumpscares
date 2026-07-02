# Melhorias para o projeto atual

> 📦 **HISTÓRICO** — escrito em maio/2026, antes das implementações. Do que está aqui:
> curriculum learning e BC warmstart FORAM implementados (pacote de julho/2026); RecurrentPPO
> está pronto mas desligado nesta fase; simplificação de ações e paralelização não foram feitas.
> Estado atual: [../PACOTE_BC_ENTROPIA.md](../PACOTE_BC_ENTROPIA.md) · índice: [../README.md](../README.md).

Melhorias que podem ser aplicadas sem trocar o algoritmo base (PPO) nem a
arquitetura fundamental do projeto. Cada seção descreve o problema que a melhoria
resolve, como funciona e o que precisaria mudar no código.

---

## Curriculum Learning

### O problema

O episódio atual dura ~535 segundos de tempo real. Isso significa que cada vez
que o agente morre nos primeiros 30 segundos, ele recebe um sinal de punição
enorme (−500) mas aprendeu *quase nada* sobre o que fazer nos últimos 500 segundos.
A relação entre ação e consequência fica muito diluída para um horizonte tão longo.

Além disso, a Night 1 é uma noite de tutorial: os animatronicos têm IA no nível 0
na maior parte da noite e só começam a se mover de fato nas últimas horas. Treinar
para sobreviver a noite inteira desde o início é como aprender a escalar o Everest
sem nunca ter subido uma colina.

### Como funciona

A ideia é truncar artificialmente os episódios no início e só expandir o horizonte
conforme o agente demonstra competência. O treinamento é dividido em fases:

```
Fase 1 — Sobreviver até 2AM (~178s):
  Treinar até atingir >80% de win rate nessa janela.
  O agente aprende: "não ficar sem energia cedo, fechar portas quando necessário".

Fase 2 — Expandir até 4AM (~356s):
  Adicionar a pressão progressiva dos animatronicos.
  Aprende: reconhecer padrões de movimento, gerenciar energia no médio prazo.

Fase 3 — Noite completa (535s):
  Apenas agora o agente enfrenta o desafio completo.
```

Um episódio truncado por limite de tempo ainda recebe uma recompensa parcial
proporcional ao progresso (ex: `+500 × (tempo_sobrevivido / 535)`), para não
punir o agente por "não terminar" algo que ele nem podia terminar ainda.

### O que mudar no código

**`src/environment/fnaf_env.py`:**

Adicionar parâmetro `max_episode_time` ao `__init__`. Na lógica de step, checar:

```python
if self.tempo_real_ep >= self.max_episode_time:
    recompensa += 500 * (self.max_episode_time / 535.0)
    truncated = True
```

**`src/agent/train.py`:**

Adicionar um callback que calcula o win rate dos últimos N episódios. Quando
ultrapassar o threshold configurado, aumenta o `max_episode_time` do ambiente:

```python
class CurriculumCallback(BaseCallback):
    def _on_step(self):
        if win_rate_recente > 0.80:
            self.training_env.env_method("set_max_time", nova_fase)
```

---

## RecurrentPPO — Memória Temporal com LSTM

### O problema

No estado atual, o agente toma cada decisão olhando apenas para **um único frame**
(84×84px) mais o vetor de estado instantâneo. Isso significa que ele não tem como
raciocinar sobre o que aconteceu nos últimos segundos.

No FNAF, isso é um problema estrutural. Os animatronicos se movem quando *não estão
sendo observados*. Se o agente vê Bonnie na câmera 2A e depois fecha o tablet, ele
perde o rastro. Sem memória, cada step é tratado como se fosse o primeiro da partida.
Um humano lembra: "eu vi Bonnie lá há 5 segundos, se ele sumiu preciso checar a
porta esquerda". O agente atual não tem essa capacidade.

### Como funciona

O `RecurrentPPO` da biblioteca `sb3-contrib` é idêntico ao PPO padrão, mas insere
uma camada LSTM entre o extrator de features e as cabeças de ator/crítico. O LSTM
mantém um estado oculto `(h, c)` que persiste entre os steps do episódio.

```
Observação (frame + estado) → Extrator de features (CNN + MLP)
    → LSTM (memória temporal)
        → Ator: distribui probabilidade sobre 17 ações
        → Crítico: estima V(estado)
```

O estado oculto é zerado no início de cada episódio e propagado entre steps. O
agente efetivamente tem acesso a um "resumo comprimido" de tudo que viu no episódio.

### O que mudar no código

**`requirements.txt`:**

```
sb3-contrib>=2.8.0
```

**`src/agent/train.py`:**

```python
from sb3_contrib import RecurrentPPO

model = RecurrentPPO(
    "MultiInputLstmPolicy",
    env,
    lstm_hidden_size=256,
    n_lstm_layers=1,
    ...  # demais hiperparâmetros permanecem iguais
)
```

A política `MultiInputLstmPolicy` é o equivalente recorrente da `MultiInputPolicy`
atual — compatível com o espaço de observação `Dict` que já existe.

**Atenção:** o `RecurrentPPO` exige que os episódios não sejam quebrados no meio do
buffer de experiências. O parâmetro `n_steps` precisa ser múltiplo do tamanho médio
de episódio ou usar `episode_start` masks corretamente. O `sb3-contrib` lida com
isso automaticamente, mas vale verificar os logs de treinamento.

---

## Behavioral Cloning como Warmup

### O problema

PPO começa com uma política completamente aleatória: o agente não faz ideia do que
os botões fazem, o que é uma câmera, ou que fechar a porta pode salvá-lo. Ele
precisa descobrir tudo isso por tentativa e erro. Isso é o que explica os ~200 mil
steps para a primeira vitória.

### Como funciona

Behavioral Cloning (BC) é o caso mais simples de aprendizado por imitação: dado
um dataset de pares `(observação, ação)` de um especialista humano, treinar a rede
para reproduzir essas ações via supervised learning. É equivalente a mostrar ao
agente como jogar antes de deixá-lo jogar sozinho.

O fluxo é:

```
1. Humano joga FNAF1 e o gameplay é gravado (observação + ação a cada step)
2. BC treina a rede por 50-100k steps no dataset gravado (sem rodar o jogo)
3. O modelo resultante é usado como ponto de partida para o PPO
4. PPO refina a política a partir de um estado que já sabe jogar razoavelmente
```

O resultado esperado é que o primeiro win passe de ~200k steps para menos de 20k,
porque o agente já sabe as mecânicas básicas e o PPO só precisa otimizar a
estratégia, não descobrir o que cada ação faz.

O módulo `src/agent/behavioral_cloning.py` já existe no projeto. O gargalo é
**gravar o dataset de gameplay humano**. O formato JSON esperado pelo módulo está
definido no próprio arquivo.

### DAgger — BC iterativo (extensão)

O BC puro tem um problema: o agente aprende a imitar o humano nos estados que o
*humano* visita, mas quando ele começa a errar e entra em estados nunca vistos pelo
humano, não sabe o que fazer.

DAgger resolve isso iterativamente:

```
1. Treina com BC inicial (poucas horas de gameplay)
2. Deixa o agente jogar; identifica states onde ele agiu diferente do humano
3. Pede ao humano para rotular esses novos estados (o que você faria aqui?)
4. Adiciona ao dataset e re-treina
5. Repete
```

Isso é mais trabalhoso, mas produz um agente mais robusto — especialmente para
lidar com situações de pressão alta (energia crítica, dois animatronicos ativos).

---

## Simplificação do Espaço de Ações

### O problema

O espaço atual tem 17 ações, incluindo 9 câmeras individuais (1A, 1B, 1C, 2A, 2B,
3, 4A, 4B, 7). Isso significa que o agente precisa aprender, por tentativa e erro,
que câmera 4A mostra o corredor esquerdo, que 1C mostra o palco, etc. Essas
associações levam muitos steps para emergir.

Além disso, ter muitas ações piora a exploração: com 17 opções por step, a
probabilidade de escolher aleatoriamente a ação correta numa situação crítica é
1/17 ≈ 6%.

### Alternativa

Reduzir para ~9 ações agrupando as câmeras em navegação relativa:

| ID | Ação | Descrição |
|----|------|-----------|
| 0 | nada | |
| 1 | porta_esquerda | toggle |
| 2 | porta_direita | toggle |
| 3 | luz_esquerda | toggle |
| 4 | luz_direita | toggle |
| 5 | abrir_fechar_camera | toggle tablet |
| 6 | camera_anterior | navega para câmera anterior |
| 7 | camera_proxima | navega para câmera seguinte |

O agente ainda visita todas as câmeras, mas aprende "navegar" ao invés de "ir para
câmera específica". Para o comportamento humano isso é mais natural: você vai
passando as câmeras até encontrar o animatrônico.

**Trade-off:** navegar câmera a câmera é mais lento que pular direto para a câmera
certa. Para noites avançadas onde a velocidade de verificação importa, pode ser
um limitante. Vale experimentar e comparar métricas.

---

## Paralelização com VecEnv

### O problema

O PPO coleta experiências de forma sequencial: um episódio por vez, em tempo real.
Com ~535s por episódio e overhead de captura/controle, um ciclo de 2048 steps leva
~25 minutos. São menos de 60 atualizações de política por hora.

### Como funciona

O SB3 suporta `SubprocVecEnv` — múltiplos ambientes rodando em processos paralelos
que coletam dados simultaneamente. Com 4 instâncias paralelas:

```
4 instâncias × 2048 steps / 4 = mesmo ciclo de update, 4× mais rápido
```

Para FNAF, isso requer rodar 4 janelas do jogo simultaneamente. Cada instância
precisa de coordenadas de janela independentes (sem sobreposição), e o
`FNAFEnv` precisa receber um parâmetro identificando qual janela capturar e onde
clicar.

**Requisitos de hardware:**
- ~500MB RAM por instância → ~2GB para 4 instâncias
- Tela com resolução suficiente para 4 janelas lado a lado (ou múltiplos monitores)
- GPU não é gargalo aqui; o gargalo é a captura de tela e interação

É a melhoria de throughput mais direta, mas a mais trabalhosa de implementar por
exigir refatoração do gerenciamento de janelas.
