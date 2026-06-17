# Decisões de recompensa e RL — em ordem de urgência

Verificação feita direto no código. Só entram pontos que **mudam o desempenho do agente** e
exigem decisão. Cada decisão tem três partes: **o problema** (por que atrapalha, com os
números reais), **o que mudar** (com exemplo de código antes/depois) e **o que esperar**.

Contexto que define a ordem:

- O ambiente é o **jogo real** (captura de tela 84×84 + mouse). Cada step custa centenas de ms.
- **Uma instância, em tempo real.** Sem paralelizar dezenas de ambientes.
- **Episódio longo** (~535s, centenas de steps). Poucos episódios por hora → **amostra de jogo
  real é o recurso escasso**, não computação.
- **Observabilidade parcial:** o frame mostra só o corredor/câmera atual.
- **Objetivo:** sobreviver a noite jogando de verdade — não maximizar uma métrica *proxy* (um
  número que serve de atalho para o objetivo, mas não é o objetivo em si).

A ordem prioriza **parar de desperdiçar amostra** (não treinar a coisa errada) antes de
mudanças estruturais caras.

> Termos técnicos (gamma, entropia, schedule, shaping…) são explicados **na primeira vez que
> aparecem**, ali no texto, entre parênteses ou em nota — não precisa de conhecimento prévio.

---

## DECISÃO 1 — Rebalancear a recompensa para não premiar a passividade (nem o "travar")  ⟵ mais urgente

### O problema

Hoje o agente é punido por **agir** e também por **ficar parado** — ele fica espremido.

Olhe as magnitudes reais. A recompensa-base de um step vivo é pequena:

```
base = bonus_hora + 0.5  +  progresso*0.5     →  ~0.5 a 1.0 por step (fora dos checkpoints)
```

E as penalidades por step são **maiores que essa base**:

| Penalidade (linha) | Valor | vs base (~0.5–1.0) |
|---|---|---|
| spam de porta/luz — L834 | −1.5 | apaga 2–3 steps de sobrevivência |
| inação prolongada — L837 | até −2.0 | idem |
| câmera/toggle repetido — L841 | −1.0 | idem |
| ambas as portas fechadas — L850 | −1.0 | idem |
| câmera ociosa — L853 | até −1.0 | idem |
| luz — L846 | −0.2 | menor, ok |

Três consequências concretas, ruins para *este* jogo:

1. **O sinal imediato afoga o objetivo.** Sobreviver rende +0.5/step; uma ação "feia" tira
   −1.5. Como a recompensa de vencer (+500) só chega lá no fim e é rara no começo do treino, o
   agente aprende primeiro o que é imediato: "agir machuca". Mas a política passiva, no FNAF,
   **perde** (a energia drena, o Foxy corre, a porta fica aberta).
2. **As penalidades se contradizem.** Repetir ação pune (−1.0/−1.5) e ficar parado também pune
   (−2.0). O caminho de menor punição vira *alternar ações inúteis só para não repetir nem
   parar* — comportamento errático que não tem nada a ver com sobreviver.
3. **A penalidade por repetição trata o sintoma, não a causa — e pode estar piorando.** Este é
   exatamente o problema que já apareceu no treino: a IA "desistindo" e repetindo a mesma ação
   sem parar. Mecanicamente isso é **colapso da política** — a distribuição de ações desaba para
   ~100% numa ação só, porque a *entropia* — o quanto a política ainda experimenta ações variadas
   em vez de cravar numa só — caiu a zero. A repetição é o *sintoma*; a causa é a exploração
   colapsando. E aqui vai o ponto contraintuitivo: **quando quase toda ação é
   punida, nenhuma ação parece melhor que a outra** — o valor dos estados fica achatado e
   negativo, o agente "se sente condenado" e trava numa ação qualquer. Ou seja, a penalidade
   criada para impedir a desistência ajuda a *causá-la*. E quando ela funciona, o colapso só se
   disfarça: o agente dribla o termo ciclando 2-3 ações inúteis (luz→câmera→luz→câmera) —
   continua desistido, só que de um jeito que evita a penalidade.

E o `bonus_hora` é grande demais: `max(ratio*50, 5)` por checkpoint, com `ratio` até 1.5 → **até
75 por checkpoint × 6 = ~450** (L814-819), quase igual à vitória inteira (+500, L809). Resultado:
chegar perto do fim e morrer rende quase tanto quanto vencer, então "vencer de verdade" pesa
pouco no que o agente acaba perseguindo.

### O que mudar

Penalidades de proxy viram **fração** da base por step, e o `bonus_hora` cai para bem abaixo do
terminal. Exemplo concreto em `_calcular_recompensa` (fnaf_env.py:800-864):

```python
# ── ANTES ──────────────────────────────────────────────
if nome_acao == self.penultima_acao:
    recompensa -= 1.5            # spam
...
elif nome_acao == self.penultima_acao:
    recompensa -= 1.0            # câmera/toggle repetido
...
if nome_acao in ["luz_esquerda", "luz_direita"]:
    recompensa -= 0.2
if self.porta_esq and self.porta_dir:
    recompensa -= 1.0
...
bonus_hora += max(ratio * 50.0, 5.0)   # 5 a 75 por checkpoint

# ── DEPOIS ─────────────────────────────────────────────
if nome_acao == self.penultima_acao:
    recompensa -= 0.15           # ~1/3 da base; desencoraja sem dominar
...
elif nome_acao == self.penultima_acao:
    recompensa -= 0.1
...
if nome_acao in ["luz_esquerda", "luz_direita"]:
    recompensa -= 0.05
if self.porta_esq and self.porta_dir:
    recompensa -= 0.1
...
bonus_hora += max(ratio * 8.0, 1.0)    # 1 a 12 por checkpoint → total <~70 << 500
```

Com penalidades menores, o piso pode ser afrouxado também (a trava existe para o caso de muitas
penalidades somarem; com elas pequenas, `−1.0` basta):

```python
recompensa = max(recompensa, -2.0)   # ANTES
recompensa = max(recompensa, -1.0)   # DEPOIS
```

**Reduzir a penalidade sozinha pode deixar a repetição voltar** — ela tratava o sintoma. Para
atacar a causa (o colapso da Decisão 1.3), faça junto:

- **Mantenha um anti-spam pequeno e mire no que é inútil.** Repare que ação sem efeito já cai em
  `if not acao_valida: return -0.5 + bonus_hora` (L821) — spam de câmera fechada, porta em
  cooldown etc. já são punidos ali. A penalidade extra por repetição incide justamente sobre as
  ações *válidas* (que tiveram efeito) — ou seja, pune o jogo bem jogado (segurar a porta na
  ameaça). Reduza-a bastante e, se quiser, condicione a "repetiu **e** não mudou o estado".
- **Sustente a entropia (Decisão 6).** É o lever direto contra colapso de política: se a IA
  trava, é exploração morrendo. `ent_coef` (o parâmetro que controla quanto o treino recompensa
  explorar) mais alto / sem decair cedo demais combate isso melhor que qualquer penalidade.
- **Adicione o sinal causal (Decisão 4) assim que der.** É o conserto de fundo: a IA repete
  porque nada diz que a ação repetida está *falhando* até tarde demais. Um sinal que reage **na
  hora** a bloquear/expor a ameaça (Decisão 4) dá motivo real para variar a ação com propósito, e
  a ação inútil repetida deixa de ser "de graça".
- **Cubra o anti-colapso no teste (Decisão 2).** Além de "boa > passiva", asserte que um *rollout*
  (uma partida simulada do começo ao fim) que faz spam de 1-2 ações pontua mal — assim, se um
  rebalanceamento reabrir a porta para o colapso, o teste pega antes do treino.

Regra de validação (vira o teste da Decisão 2): no **melhor episódio sem vitória**, a soma de
tudo deve ficar **claramente abaixo de +500**.

### O que esperar

Encolher os proxies + sustentar a entropia + (idealmente) o sinal causal, **juntos**, atacam
tanto a passividade quanto o "travar repetindo": o agente passa a ter estados com valor positivo
e diferenciado — um motivo para agir com propósito em vez de spammar ou congelar. Fazer só o
rebalanceamento, sem entropia/sinal causal, é tirar o curativo sem tratar a ferida — aí a
repetição pode voltar. A penalidade nunca foi a cura; era um band-aid que, no seu caso, também
sangrava.

---

## DECISÃO 2 — Teste de sanidade antes do próximo treino longo

### O problema

A Decisão 1 mexe em 7 termos que competem entre si. É fácil rebalancear e, sem querer, deixar a
política passiva pontuando mais que uma boa. Hoje **não há nada que pegue isso** antes de treinar
— você só descobriria pela curva, horas e amostras de jogo real depois. Não existe teste de
recompensa (só `testar_deteccao`/`testar_energia`, que cobrem captura).

### O que mudar

Tornar `_calcular_recompensa` chamável sem subir o jogo (esse é o único motivo prático de
"isolar a recompensa" — viabilizar o teste) e escrever um teste que compara dois roteiros:

```python
# tests/test_reward.py (esboço)
def test_politica_boa_pontua_mais_que_passiva():
    # roteiro "passivo": só ação 0 (nada) por N steps até morrer
    retorno_passivo = simular(acoes=[0]*200, morre_no_fim=True)

    # roteiro "bom": fecha a porta certa nas ameaças, gerencia energia, sobrevive
    retorno_bom = simular(acoes=roteiro_humano(), sobrevive=True)

    assert retorno_bom > retorno_passivo          # o básico
    assert retorno_bom > retorno_quase_venceu()   # vencer > chegar perto e morrer
```

`simular(...)` apenas alimenta estados/eventos sintéticos na função de recompensa e soma o
retorno — não precisa do jogo.

### O que este teste NÃO valida (e onde a imagem entra)

A imagem importa para a *decisão*, mas não para *este* teste — e a separação é proposital. São
**três validações diferentes** que não se misturam:

1. **Incentivo (este teste).** `_calcular_recompensa` (L800-864) é função do **estado abstrato**
   do jogo — energia, portas, tempo, e os eventos `morreu`/`sobreviveu` como booleanos. Ela
   **não olha pixels**. Por isso dá para validar offline injetando estados/eventos sintéticos:
   aqui você confere se o *incentivo* está certo, não se a percepção está.
2. **Percepção (validação separada, precisa de prints).** Se morte/vitória/ameaça são detectadas
   corretamente na imagem é outra questão — essa sim exige **screenshots reais rotulados** rodando
   contra `_detectar_morte`, `_detectar_vitoria` e o detector novo da Decisão 4. É o que
   `testar_deteccao.py` já faz em parte; o jeito de "testar sem o jogo" é um *fixture* de imagens
   salvas em disco.
3. **A política usa a imagem? (precisa do agente no jogo).** Se a CNN de fato pesa na decisão só
   dá para medir com o modelo treinado rodando — é a ablação da Decisão 5.

Ou seja: o teste da Decisão 2 não tenta (nem deve) validar a captura de tela. Ele isola "a
recompensa premia o comportamento certo?" de "a percepção enxerga certo?" e "a política aproveita
a imagem?". Misturar os três é o que tornaria o teste impossível de rodar offline — separá-los é
o que o deixa barato e confiável.

### O que esperar

Vira um *guard-rail* (uma trava de segurança automática): qualquer rebalanceamento futuro que
inverta o incentivo falha localmente, em segundos, em vez de queimar um treino noturno inteiro.

---

## DECISÃO 3 — Ligar `VecNormalize(norm_reward)`

### O problema

O PPO treina **duas** redes: a *política* (escolhe a ação) e o *crítico* (estima "quanto retorno
total eu ainda vou somar a partir deste estado"). O crítico aprende por tentativa e erro: chuta um
valor, compara com o retorno real, e ajusta os pesos **na proporção do erro**.

O problema é a **escala dos números** que ele tem que prever. Hoje as recompensas convivem em
ordens de grandeza muito diferentes:

```
step vivo comum:   ~ +0.5
bônus de hora:     + até 75  (de uma vez)
morte:               −100
vitória:             +500
```

Quando o crítico erra um evento de +500 por 20%, o erro é 100 — um "puxão" enorme nos pesos.
Quando erra um step de +0.5, o ajuste é minúsculo. Três efeitos ruins saem disso:

- O treino fica **dominado pelos eventos raros e gigantes** (vitória/morte/bônus) e anda aos
  solavancos, em vez de aprender de forma estável com os milhares de steps comuns.
- **Nenhum `learning_rate` serve para os dois ao mesmo tempo** — `learning_rate` é o tamanho do
  passo que a rede dá ao se corrigir: o que é seguro para os +500 é pequeno demais para aprender
  os +0.5; o que aprende os +0.5 é grande demais para os +500 e faz a rede "passar do ponto"
  (overshoot, "passar do alvo e oscilar").
- Como o PPO usa o crítico para calcular a **vantagem** de cada ação (o quanto ela foi melhor que
  a média), um crítico instável gera vantagens ruidosas → a política aprende erraticamente, o que
  **realimenta o colapso da Decisão 1**.

Analogia: é como ensinar alguém a estimar preços, mas você às vezes cota em centavos e às vezes em
milhões, sem avisar. Toda vez que aparece um "milhões", a pessoa surta e reescreve o modelo mental
inteiro, esquecendo o que aprendeu com os "centavos". Hoje o env vai cru para o PPO
(train.py:173,183), sem nenhuma normalização — exatamente esse cenário.

### O que mudar

Envolver o env com `VecNormalize` normalizando **só a recompensa** (as observações já estão em
[0,1], L874-883). O que ele faz: mantém uma estimativa contínua da **escala** da recompensa e
divide tudo por ela, então o crítico sempre vê alvos numa faixa parecida (~unitária). As
diferenças relativas continuam (vitória ainda >> step), mas os tamanhos absolutos ficam domados, e
uma só `learning_rate` passa a funcionar para tudo. O detalhe que **não pode esquecer**:
salvar/carregar as estatísticas junto do modelo, senão a avaliação roda com normalização errada.

```python
# train.py — ANTES
env = FNAFEnv()
...
modelo = PPO(policy="MultiInputPolicy", env=env, ...)

# train.py — DEPOIS
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
env = DummyVecEnv([lambda: FNAFEnv()])
env = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=0.995)
# gamma = "fator de desconto" (0 a 1): quanto o agente valoriza o futuro vs. o agora — ver Decisão 6
...
modelo = PPO(policy="MultiInputPolicy", env=env, ...)

# ao salvar o modelo, salve também as stats:
modelo.save(caminho_final)
env.save(f"{PASTA_MODELOS}/vecnormalize.pkl")

# ao carregar para jogar/continuar:
env = VecNormalize.load("modelos/vecnormalize.pkl", DummyVecEnv([lambda: FNAFEnv()]))
env.training = False      # não atualiza stats na avaliação
env.norm_reward = False   # na avaliação você quer a recompensa real
```

### O que esperar

Treino bem mais estável apesar das magnitudes díspares — provavelmente o melhor custo-benefício
da lista. Se esquecer de salvar as stats, a avaliação fica inconsistente; por isso o save/load é
parte obrigatória da mudança.

---

## DECISÃO 4 — Sinal causal de bloqueio: guiar a IA sem "ensinar a jogar do nosso jeito"

### O problema

Esta é a maior fonte de ineficiência de aprendizado. No FNAF a ação certa (fechar a porta do lado
certo **no instante** em que o animatrônico está no vão) e a consequência (não morrer) podem
estar separadas por **centenas de steps** — ou a morte nem acontece naquela noite. Esse é o
problema de *credit assignment*: descobrir **qual** das centenas de ações causou o resultado lá no
fim. Com o env atual, que **não rastreia animatrônicos** e não dá nenhuma recompensa por bloquear,
o agente tem que adivinhar toda a causalidade do jogo a partir de um único sinal terminal. É o
cenário mais lento e ruidoso possível para RL — péssimo com amostra cara.

### A dúvida certa: isso não vicia a IA a jogar do nosso jeito?

Sim — e é uma preocupação legítima. Vale separar dois tipos de recompensa:

- **O objetivo verdadeiro** já está lá: vencer (+500) / morrer (−100). Esse sinal não diz *como*
  jogar, só define ganhar e perder — é o que a gente realmente quer que ela maximize.
- **Reward shaping** é dar *dicas intermediárias* além do "ganhou/perdeu", para o agente não
  depender só do sinal lá do fim. Um "+5 por fechar a porta na ameaça" é shaping — e, do jeito
  ingênuo, ele **induz** a IA a jogar do jeito que a gente conhece. Você apontou isso corretamente.

Por que então não deixar **só** o objetivo verdadeiro e ela aprender sozinha? Isso se chama
*recompensa esparsa* (sinal só no fim) e, em teoria, é o menos enviesado. Mas no nosso caso —
episódio de ~535s em tempo real, uma instância, poucas partidas por hora — cada episódio dá **um
bit** de feedback separado de centenas de decisões. Na prática, isso provavelmente **não converge**
num orçamento de treino realista. O trade-off: puro = sem viés, mas pode nunca aprender; guiado =
aprende rápido, mas com o nosso viés.

A boa notícia: dá para guiar **sem mover o alvo**.

### O que mudar — duas formas de shaping (e a que não enviesa)

O ingrediente comum às duas é **detectar a ameaça por lado** (template/visão, como já se faz para
morte/vitória) e **expor isso nos estados** — hoje os 8 estados trazem porta/luz/câmera/energia/
tempo, mas **não onde estão os animatrônicos**, a informação mais decisiva do jogo:

```python
# expor a ameaça nos 8→10 estados (fnaf_env.py:874-883 e observation_space)
ameaca_esq = self._animatronico_no_vao("esquerdo")   # detecção visual
ameaca_dir = self._animatronico_no_vao("direito")
estados = np.array([
    ...,
    float(ameaca_esq),
    float(ameaca_dir),
], dtype=np.float32)
```

**Opção A — bônus direto (simples, mas enviesa).** Recompensa fixa por bloquear certo:

```python
if ameaca_esq:
    recompensa += 5.0 if self.porta_esq else -8.0   # bloqueou certo / deixou passar
if ameaca_dir:
    recompensa += 5.0 if self.porta_dir else -8.0
```

Aprende rápido, mas paga o agente por *uma ação específica* — é exatamente o que você teme: empurra
para a estratégia que a gente já conhece.

**Opção B — potential-based shaping (matematicamente não move o alvo).** Em vez de pagar pela ação,
você define um "potencial" Φ(estado) que mede *quão segura é a situação agora*, e recompensa só a
**variação** desse potencial de um step para o seguinte:

```python
def _potencial_seguranca(self) -> float:
    # 0 = exposto a uma ameaça presente; mais alto = ameaças presentes estão bloqueadas
    phi = 0.0
    if self.ameaca_esq:
        phi += 0.5 if self.porta_esq else 0.0
    if self.ameaca_dir:
        phi += 0.5 if self.porta_dir else 0.0
    return phi

phi_antes  = self._potencial_seguranca()   # antes de aplicar a ação
# ... executa a ação, atualiza o estado ...
phi_depois = self._potencial_seguranca()   # depois

# recompensa de shaping = variação de segurança, ponderada pelo gamma do treino
recompensa += GAMMA * phi_depois - phi_antes
```

Por que isso **não enviesa**? Há um resultado clássico (Ng, Harada & Russell, 1999): se a dica tem
exatamente essa forma `γ·Φ(depois) − Φ(antes)`, ela **não muda qual estratégia é a melhor** — só
muda a velocidade de aprender. A intuição: somando esses termos ao longo de um episódio inteiro,
eles se cancelam (telescopam) e sobra só uma constante, igual para qualquer estratégia. Ou seja,
**nenhum caminho ganha pontos líquidos** por causa da dica; ela só faz o sinal "chegar mais cedo",
perto da ação que de fato importou, em vez de só no fim. (É aqui que o `gamma` entra: ele precisa
ser o mesmo do treino para esse cancelamento funcionar.)

**Anelar a dica (annealing).** Se for pela Opção A (que enviesa), dá para multiplicar o bônus por
um fator que vai de 1 a 0 ao longo do treino: guia forte no começo (para destravar o aprendizado) e
some no fim, deixando a fase final otimizar só o objetivo verdadeiro. Junta o melhor dos dois.

**Curriculum como acelerador sem viés (complementar).** Independente do shaping, dá para encurtar o
problema: começar com noites curtas e ir alongando (já consta no `MELHORIAS_PROJETO_ATUAL.md`). Isso
**não diz nada sobre como jogar** — só aproxima o sinal terminal das ações, tornando a recompensa
esparsa de fato aprendível. É a rota mais "ela aprende sozinha".

### O que esperar

Com detecção confiável: salto grande de eficiência — a IA aprende a relação "ameaça → bloquear" em
poucos episódios, não em centenas. **Recomendação:** prefira a **Opção B (potential-based)** +
curriculum + entropia (Decisão 6), porque guiam sem prender a estratégia num molde humano; use a
Opção A só se precisar de um empurrão inicial e, nesse caso, **anele** para tirar o viés no fim.

Dois cuidados honestos:

- **Detecção frágil = *reward hacking*** — o agente ganha pontos explorando o falso positivo do
  detector em vez de jogar. Por isso **valide a detecção isolada** (o item "percepção" da Decisão 2)
  antes de ligá-la na recompensa.
- **"Sem nenhuma chance de convergir errado" não é 100% garantível.** O potential-based garante que
  a *dica* não desloca o ótimo, não que a IA alcance o ótimo global (aproximação por rede, ótimos
  locais e exploração limitada existem). No FNAF, porém, o risco de viés é baixo: bloquear a ameaça
  não é uma "mania humana", é **a** mecânica de sobreviver; o espaço para ela descobrir algo melhor
  está nos detalhes (timing, quando vale gastar energia) — e é justo isso que a Opção B + annealing
  preservam.

Esta decisão vem depois da #1 (que é só ajuste de constantes, sem construir nada) porque exige
construir e validar a detecção visual.

---

## DECISÃO 4B — Percepção fiel dos estados: ler o real em vez de inferir (energia, portas, luz, câmera)

### O problema

Vários dos 8 estados que a IA recebe não são **medidos** do jogo — são **inferidos** pelo env, e a
inferência erra.

- **Energia é simulada.** `_atualizar_energia` (fnaf_env.py:767) drena `0.104 + itens*0.100` por
  segundo real. É boa aproximação, mas **nunca bate 100%**: eventos discretos não entram na conta —
  o **Foxy batendo na porta tira um pedaço acumulativo de energia**, além de variação de framerate
  e pausa. A IA recebe `energia/100` errado como estado, e o termo de déficit da recompensa
  (`deficit*0.02`, L859) é calculado sobre o valor errado.
- **Portas/luz/câmera podem divergir por miss-click.** Já existe validação: o env confere o clique
  de porta pela cor do botão e tenta de novo (`_verificar_botao_porta`, L221) e lê a câmera por
  template (`_camera_aberta_por_template`, L210). Mas não é perfeito — fica em ~90% de acerto. Os
  ~10% restantes são estado interno mentindo para a IA: ela "acha" que a porta está fechada quando
  não está.

Estado que mente é pior que estado ausente — a IA aprende uma relação que não existe.

### O que mudar

Para cada vetor, **trocar o inferido pelo percebido**, reusando o `matchTemplate` que o projeto já
usa (morte/vitória/câmera) e a ferramenta `calibrar`.

- **Energia — template por algarismo (não OCR).** A fonte e a posição do "Power left: XX%" são
  fixas → calibrar 10 glifos (`0`–`9`), recortar a região do número, fatiar em dígitos e dar
  `matchTemplate` em cada um. (OCR/tesseract seria frágil com fonte de jogo em baixa resolução, e
  nem está no projeto.) Dois cuidados deixam o leitor robusto quase de graça:
  - **Prior monotônico:** energia no FNAF1 só **cai**. Rejeite qualquer leitura que *suba*
    (oclusão/erro) e caia para a simulação; uma queda brusca (Foxy bateu) é aceita — ainda é
    decréscimo.
  - **Fallback + re-âncora:** leitura confiável corrige a `energia`; leitura incerta (animação de
    câmera tapando o número, apagão) cai para a simulação, que preenche o intervalo. Nunca injeta
    lixo.
- **Portas/luz/câmera — endurecer a verificação que já existe.** Generalizar o padrão do
  `_verificar_botao_porta`/`_camera_aberta_por_template`: depois de cada ação, **ler o estado real**
  (cor do botão / template) e adotar isso como verdade, em vez de confiar no toggle interno. Os
  ~10% que escapam hoje são onde vale fechar para chegar perto de 100%.
- **Validar offline antes de ligar.** É o item "percepção" da Decisão 2: um *fixture* de
  screenshots rotulados (energia 100/99/50/9/1/0; portas e câmera abertas/fechadas) rodando contra
  o leitor, medindo a precisão. Só sobe para a observação depois de passar.

### Validar que não atrapalha o aprendizado

Trocar "estado que mente" por "estado fiel" **deveria** ajudar — mas confirme, não confie:

- **Meça pelo independente.** Taxa de vitória e tempo de sobrevivência, não a recompensa. Ligue o
  leitor de energia e a validação de ação **separadamente**, cada um contra o controle (regras de
  ouro #1 e #3 do plano).
- **Cuidado com latência.** Energia e tempo correm por wall-clock; um leitor lento muda o **ritmo
  real** do jogo a cada step e distorce justamente a energia que você quer medir. Mantenha o custo
  baixo (`matchTemplate` em recortes pequenos).
- **Fallback obrigatório.** Um detector frágil ligado direto na observação é o mesmo risco de
  *reward hacking* / estado mentindo que a Decisão 4 alerta — por isso o prior monotônico e o
  fallback.

### O que esperar

Estado mais fiel → a IA decide sobre a verdade, e some uma fonte de ruído que hoje a faz "ver"
energia/portas erradas justo nos momentos críticos (eventos do Foxy, miss-clicks na hora da
ameaça). É a primeira peça da fase de percepção — mais simples que a detecção de ameaça da Decisão
4 e reusa a mesma calibração/validação, então é o aquecimento natural antes dela.

---

## DECISÃO 5 — Confirmar por ablação se o ramo de imagem (CNN) contribui

### O problema

A imagem (ramo CNN — a parte da rede que processa os pixels — multimodal_policy.py:11-19,51) é o
que carrega as pistas visuais de ameaça que os 8 estados não têm. Mas o próprio código registra um bug **já corrigido** que deixava a
CNN cega — dupla normalização punha os pixels em [0, 0.004] (multimodal_policy.py:36-39). Isso é
o aviso: é fácil o ramo estar *presente mas inerte*. Se a CNN não pesa na decisão, o agente está
decidindo só pelos 8 estados — cego para as ameaças visuais — e você paga o custo da CNN à toa.

### O que mudar

Rodar uma *ablação* simples (ablação = desligar uma parte do sistema para medir o quanto ela
importava): zerar um ramo de cada vez na observação e medir a queda de desempenho.

```python
# zera a imagem: se o desempenho quase não cair, a CNN não está contribuindo
obs["imagem"] = np.zeros_like(obs["imagem"])

# zera os estados: mede o quanto a política depende deles
obs["estados"] = np.zeros_like(obs["estados"])
```

### O que esperar

Se zerar a imagem derrubar o desempenho: o design híbrido está valendo. Se **não** derrubar: há
bug silencioso de novo ou a CNN não aprendeu — investigar isso rende mais que qualquer tuning,
porque a imagem é a fonte das pistas de ameaça.

---

## DECISÃO 6 — Schedules de exploração e horizonte (`ent_coef`, `learning_rate`, `gamma`)

### O problema

Hoje são constantes: `learning_rate=3e-4`, `ent_coef=0.01`, `gamma=0.995` (train.py:187,192,191).
Dois efeitos:

- **Exploração que não se adapta.** No começo o agente precisa explorar muito para *achar* a
  estratégia de bloqueio (rara de acertar por acaso); no fim, precisa parar de explorar para
  estabilizar. Entropia/LR constantes não fazem essa transição.
- **Horizonte curto demais para a noite.** `gamma=0.995` dá horizonte efetivo de ~200 steps, mas
  a noite tem centenas. A vitória final é descontada a ponto de influenciar pouco as decisões do
  início — justo quando a boa partida é montada.

### O que mudar

Usar *schedules* (schedule = deixar um parâmetro **variar ao longo do treino** em vez de fixo; o
SB3 aceita uma função que recebe o progresso restante, indo de 1 a 0) e subir `gamma` — **uma
mudança de cada vez**, medindo:

```python
# learning_rate e ent_coef decaindo linearmente
def linear(inicio, fim=0.0):
    return lambda progresso_restante: fim + (inicio - fim) * progresso_restante

modelo = PPO(
    ...
    learning_rate=linear(3e-4),     # 3e-4 → 0
    ent_coef=0.02,                  # começa mais alto p/ explorar; ver nota
    gamma=0.997,                    # horizonte mais longo (era 0.995)
)
```

(`ent_coef` em si não aceita schedule nativo no PPO do SB3; para decair entropia, ou começa um
pouco mais alto e aceita o valor fixo, ou implementa via *callback* — um gancho de código que roda
durante o treino. LR e `clip_range` aceitam função direto.)

### O que esperar

Bem aplicado: explora mais cedo (acha a estratégia mais rápido), estabiliza no fim, e propaga
melhor o crédito do desfecho. Mal aplicado (decair rápido demais, `gamma` alto sobre recompensa
ainda ruim): o agente "congela" numa política medíocre ou o crítico desestabiliza. Por isso vem
**depois** das Decisões 1 e 3.

---

## DECISÃO 7 — RecurrentPPO + LSTM sobre o extractor híbrido  ⟵ maior aposta, menos urgente

### O problema

O FNAF é fortemente *parcialmente observável* (o agente nunca vê o estado completo do jogo de uma
vez): o frame mostra só o corredor/câmera atual, e a estratégia humana é *rastrear quem está vindo
ao longo do tempo*. O agente atual (`MultiInputPolicy`, train.py:184) decide olhando **um frame
isolado** — não consegue lembrar "vi o Bonnie se mover há 3 steps, ele deve estar chegando". A
fusão CNN+MLP existe (multimodal_policy.py:11-55) — CNN processando a imagem e MLP (uma rede comum
totalmente conectada) processando os números dos estados — mas alimenta uma cabeça **sem
memória**.

### O que mudar

Adicionar `sb3_contrib` (não está no `requirements.txt`) e trocar para `RecurrentPPO`,
reaproveitando o `MultimodalExtractor`. A arquitetura passa a ser: CNN+MLP → fusão(256) → **LSTM**
→ ator/crítico. (LSTM = uma camada com *memória*, que carrega informação dos passos anteriores
para o atual — é o que dá o "lembrar" que falta hoje.)

```python
# train.py — DEPOIS
from sb3_contrib import RecurrentPPO

modelo = RecurrentPPO(
    policy="MultiInputLstmPolicy",
    env=env,
    policy_kwargs=dict(
        features_extractor_class=MultimodalExtractor,
        lstm_hidden_size=256,
        n_lstm_layers=1,
        enable_critic_lstm=True,
    ),
    n_steps=2048, batch_size=64, gamma=0.997, ...
)
```

**Passo intermediário mais barato (recomendado antes do LSTM):** empilhar os últimos N frames
(frame-stacking) dá pseudo-memória de movimento sem o custo do treino recorrente.

### Por que é caro e exige cuidado (o que essas mudanças implicam)

Não é trocar uma linha — dar **memória** ao agente muda como o treino inteiro funciona:

1. **Passa a treinar por sequências, não por passos soltos.** O PPO normal embaralha os passos e
   aprende deles quase independentes. A LSTM tem memória: a decisão no passo *t* depende do que ela
   viu em *t−1, t−2…* Então os passos **não podem mais ser embaralhados** — têm que entrar em ordem,
   em blocos contíguos, carregando a "memória" (o estado interno da LSTM) de um passo para o outro.
   Dados mais correlacionados = gradientes mais ruidosos = costuma precisar de **mais amostra** para
   convergir. E amostra é justamente o nosso gargalo: a técnica que mais precisa de dados sendo
   aplicada onde os dados são mais caros.
2. **A memória precisa ser zerada na troca de episódio.** Quando uma noite acaba e outra começa, o
   estado interno tem que ser resetado — senão a IA "lembra" de uma partida que não tem nada a ver
   com a atual e aprende lixo. Esse reset de fim de episódio (*masking*) é um **bug clássico e
   silencioso**: não quebra nada visível, só piora o aprendizado sem avisar.
3. **Mais botões para ajustar, e eles interagem.** Surgem parâmetros novos — tamanho da memória
   (`lstm_hidden_size`), número de camadas, se o crítico compartilha a memória ou tem a sua,
   comprimento da sequência. Cada um é mais uma dimensão para calibrar, e o treino recorrente é mais
   sensível (a memória pode "apagar" ou "explodir" o sinal ao longo do tempo). Com cada experimento
   custando horas de jogo real, cada botão extra é caro.
4. **Mais lento e mais difícil de depurar.** Cada atualização processa a sequência inteira, então o
   treino fica mais pesado. E quando ela erra, a causa pode estar em algo que ela "lembrou errado"
   10 passos atrás — bem mais difícil de rastrear do que numa política que decide só pelo frame
   atual.
5. **A avaliação também muda.** Rodar o modelo (`jogar`) passa a exigir carregar e propagar o estado
   interno entre os passos e sinalizar início de episódio — não é mais um `predict(obs)` simples.

### O que esperar

Capacidade de rastrear ameaças no tempo → **provável teto de desempenho mais alto** (é o caminho
para ela jogar "de verdade", como um humano que acompanha quem está se movendo). Mas, somando os 5
pontos acima: mais amostra necessária, mais instabilidade, mais cuidado. Migrar **antes** de a
recompensa estar sã (Decisões 1-3) é empilhar um eixo instável sobre uma base ruim — e aí, se
piorar, você não sabe se a culpa é da LSTM ou da recompensa, e gasta muitas amostras caras só para
descobrir. **Por isso é a última** — e por isso o frame-stacking (acima) é o teste barato antes de
comprar toda essa complexidade: dá um pouco de memória sem mudar o algoritmo nem o risco.

---

## Já está correto — não exige decisão

- **Relógio único (IMPLEMENTADO).** `_atualizar_tempo`/`_atualizar_energia` usam
  `time.perf_counter()` (fnaf_env.py:767-788); o bug de tempo simulado × real que fazia a energia
  mentir está corrigido. *Melhoria menor:* congelar o relógio durante a pausa F12.
- **Desenho da observação (IMPLEMENTADO).** Dar os 8 estados estruturados de graça é boa decisão;
  compensa parte da observabilidade parcial sem custo de LSTM. A *fidelidade* desses estados
  (energia inferida; porta/câmera ~90% por miss-click) está na Decisão 4B, e a lacuna de ameaça por
  lado na Decisão 4.
- **Uma instância / `DummyVecEnv` implícito (CORRETO).** Paralelizar dezenas de ambientes **não
  se aplica** — o jogo é instância real controlada por mouse/janela única; tentar corromperia as
  partidas. Manter 1 env é o certo, e é o que faz "eficiência de amostra" ser o critério de tudo
  acima.

---

## Resumo da ordem

| # | Decisão | Custo | Impacto provável | Por que nesta posição |
|---|---------|-------|------------------|------------------------|
| 1 | Rebalancear recompensa (anti-passividade) | baixo | **Alto** — para de premiar a passividade/"travar"; é o que mais muda o comportamento | pode estar ensinando a coisa errada **agora** |
| 2 | Teste de sanidade antes de treinar | baixo | Indireto — não melhora a política, mas evita queimar treinos com incentivo errado | guard-rail que protege a #1 |
| 3 | `VecNormalize(norm_reward)` | baixo | Médio-alto — treino mais estável e converge mais rápido | estabiliza o treino com magnitudes díspares |
| 4 | Detecção + sinal causal de bloqueio | alto | **Alto** (se a detecção for boa) — maior salto de eficiência de amostra | maior lever de aprendizado, mas exige construir/validar detecção |
| 4B | Percepção fiel dos estados (energia/ação real) | médio | Médio-alto — remove estado que mente (energia do Foxy, miss-clicks) | mesma fase/máquina da #4; fazer antes dela como aquecimento |
| 5 | Ablação do ramo de imagem | baixo | Indireto — diagnóstico; pode revelar um bug que vale muito corrigir | garante que a CNN paga o próprio custo |
| 6 | Schedules de entropia/LR e `gamma` | médio | Médio — melhora a exploração inicial e a convergência final | só compensa sobre recompensa já sã |
| 7 | RecurrentPPO + LSTM | alto | Alto potencial / **alto risco** — eleva o teto, mas pode piorar se a base não estiver sã | maior teto, maior risco; por último |

> Fio condutor: o recurso escasso é **amostra de jogo real**. A ordem prioriza não desperdiçar
> amostra (parar de treinar a coisa errada → proteger com teste → estabilizar → atacar o credit
> assignment) antes de mudanças estruturais caras como recorrência.

---

## Plano de aplicação (passo a passo, com travas de segurança)

A ordem das decisões já está definida; falta **como** aplicá-las sem desperdiçar amostra. Quatro
regras de ouro valem para *todas* as fases:

1. **Uma mudança por vez.** Se mexer em duas coisas e o resultado mudar, você não sabe qual causou.
2. **Meça pelo que NÃO depende da recompensa.** Como vamos mexer na própria recompensa, "a
   recompensa subiu" não significa nada entre versões diferentes. Use **taxa de vitória** e **tempo
   de sobrevivência** (minutos até morrer/vencer) — esses números têm o mesmo significado em
   qualquer versão. O `treino.log` já registra resultado e tempo por episódio.
3. **Sempre guarde o melhor modelo anterior como "controle".** Antes de cada mudança, salve o
   modelo atual + suas métricas. Se a mudança piorar, você reverte para ele.
4. **Rode uma janela curta antes de comprometer um treino longo.** Um "gate" (trava) de poucos
   episódios diz se vale seguir; só então deixe rodar a noite toda.

### Fase 0 — Instrumentação (antes de qualquer mudança)
- **Faça:** rode N episódios com o modelo atual e anote taxa de vitória e tempo médio de
  sobrevivência. Esse é o **baseline** — o número que todas as fases vão tentar superar.
- **Por quê:** sem um ponto de referência fixo, "melhorou" vira achismo.

### Fase 1 — Recompensa sã (Decisão 2, depois 1)
- **Faça primeiro o teste (Decisão 2)**, depois rebalanceie a recompensa (Decisão 1): encolher as
  penalidades de proxy, mirar o anti-spam no que é inútil, limitar o `bonus_hora`. Já suba um pouco
  o `ent_coef` (parte da Decisão 6) para sustentar a exploração e evitar o "travar".
- **Trava:** o teste passa (boa > passiva, vitória > quase-venceu, spam pontua mal) **e** um gate
  curto de treino mostra que o "travar/repetir" diminuiu e a sobrevivência não piorou vs. baseline.
- **Se falhar:** ajuste os pesos e rode o teste de novo — não treine até o teste passar.

### Fase 2 — Estabilidade (Decisão 3)
- **Faça:** ligar `VecNormalize(norm_reward)` salvando as estatísticas junto do modelo.
- **Trava:** curva de treino mais suave e sobrevivência ≥ Fase 1; confirme que a avaliação carrega
  as stats (`training=False`, `norm_reward=False`) — senão o `jogar` mente.

### Fase 3 — Diagnóstico da imagem (Decisão 5)
- **Faça:** a ablação (zerar imagem vs. zerar estados).
- **Por quê aqui:** decide se vale investir na detecção visual da Fase 4. Se a CNN estiver inerte,
  conserte isso antes — não adianta construir detecção sobre uma imagem que a rede ignora.

### Fase 4 — Percepção (Decisão 4B, depois 4)
- **Comece pela Decisão 4B:** ler energia real (template por algarismo + prior monotônico +
  fallback) e endurecer a validação de porta/câmera para ~100%. É mais simples, reusa a calibração
  e já corrige um estado que a IA usa hoje. Valide isolada (fixture de prints) e confirme métrica
  independente antes de seguir.
- **Depois, a Decisão 4:** construir a detecção de ameaça e **validá-la isolada** (fixture de prints
  rotulados) **antes** de ligá-la na recompensa. Depois, adicionar o shaping potential-based (Opção
  B) e expor a ameaça nos estados.
- **Trava:** sobrevivência melhora **e** sem sinal de *reward hacking* — verifique episódios em que
  a recompensa de shaping foi alta mas a sobrevivência foi baixa (denuncia o agente explorando o
  falso positivo do detector).

### Fase 5 — Afinação (Decisão 6)
- **Faça:** um botão por vez — annealing de LR, depois subir `gamma`, depois entropia — cada um
  comparado contra o melhor anterior.

### Fase 6 — Memória (Decisão 7): como evitar que dê errado

É a fase de maior risco. Este roteiro existe para não queimar semanas de jogo:

1. **Tente frame-stacking primeiro, não a LSTM.** É a memória barata (empilhar os últimos N frames,
   sem trocar o algoritmo). **Trava:** se já melhorar a sobrevivência o suficiente, **pare aqui** —
   você ganhou memória sem o risco da LSTM.
2. **Só então a LSTM, e como teste A/B.** "A/B" = rodar a versão nova (LSTM) e a antiga (o melhor
   sem recorrência — o *controle*) com o **mesmo orçamento** e comparar na taxa de vitória/
   sobrevivência. Só mantenha a LSTM se ela **bater o controle**.
3. **Verifique o reset de memória ANTES do treino longo.** O bug silencioso é a memória não zerar na
   troca de episódio. Cheque com um teste curto: a decisão no primeiro passo de um episódio não pode
   depender do episódio anterior. (No SB3 isso depende de passar `episode_starts` corretamente.)
4. **Mude só o algoritmo, segure todo o resto.** Mesma recompensa, mesmo `VecNormalize`, mesmo
   `gamma` que você já fixou. Não calibre LSTM e recompensa ao mesmo tempo — senão volta a não saber
   de quem é a culpa.
5. **Comece pequeno.** `lstm_hidden_size` modesto (ex.: 128), `n_lstm_layers=1`. Memória maior =
   mais parâmetros = mais amostra necessária. Só cresça se ajudar.
6. **Um botão de LSTM por vez**, cada um comparado contra o controle.
7. **Defina um critério de desistência antes de começar.** Ex.: "se em X timesteps a LSTM não
   empatar com o controle na sobrevivência, reverto." Evita perseguir a LSTM por semanas.
8. **Conserte o caminho de avaliação.** O `jogar` precisa propagar o estado interno entre os passos
   e marcar início de episódio — senão a avaliação mente e você mantém (ou descarta) a LSTM pelo
   motivo errado.

> Em uma frase: a Decisão 7 só "dá certo" se entrar **por último, isolada, contra um controle, com
> critério de desistência** — e, idealmente, depois de o frame-stacking já ter mostrado se memória
> ajuda.
