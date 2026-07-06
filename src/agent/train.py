import json
import math
import os
import sys
import time
from collections import Counter, deque, defaultdict
from datetime import datetime
import keyboard
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.environment.fnaf_env import FNAFEnv, GAMMA, MAX_NOITE, RESET_METODO, PESO_ENERGIA
from src.agent.multimodal_policy import MultimodalExtractor

PASTA_MODELOS = "modelos"
PASTA_LOGS    = "logs"
# GAMMA vem do env (fonte única) — usado no PPO, no VecNormalize e no shaping
# potential-based da Decisão 4; precisam casar p/ o shaping telescopar.
CAMINHO_STATS = f"{PASTA_MODELOS}/vecnormalize.pkl"
os.makedirs(PASTA_MODELOS, exist_ok=True)
os.makedirs(PASTA_LOGS,    exist_ok=True)


def linear(inicio: float, fim: float = 0.0):
    """Schedule linear do SB3 (Decisão 6): recebe o progresso RESTANTE (1.0 → 0.0) e vai de
    `inicio` (começo do treino) a `fim` (fim). Usado p/ o learning_rate decair."""
    return lambda progresso_restante: fim + (inicio - fim) * progresso_restante


def clip_com_warmup(frac: float):
    """clip_range do PPO com warmup do crítico (BC→PPO): fica em WARMUP_CLIP nos primeiros
    `frac` do treino (ator quase parado enquanto o crítico aprende V da política clonada) e
    volta ao 0.2 padrão depois. O SB3 chama com o progresso RESTANTE (1.0 → 0.0) a cada
    rollout — serve de relógio; retomadas continuam o progresso, então não re-dispara."""
    return lambda progresso_restante: (
        WARMUP_CLIP if (frac > 0 and progresso_restante > 1.0 - frac) else 0.2
    )


def _vecnormalize_do_checkpoint(carregar_modelo: str) -> str | None:
    """Deriva o vecnormalize salvo JUNTO de um checkpoint (save_vecnormalize=True). O
    CheckpointCallback nomeia:
        modelo  = {prefix}_{N}_steps.zip
        vecnorm = {prefix}_vecnormalize_{N}_steps.pkl
    Retorna o caminho do .pkl se existir (par modelo+normalizacao casado), senão None — aí o
    chamador cai no vecnormalize global. Sem isso, voltar um checkpoint antigo usava a
    normalizacao do FIM do treino (desalinhada com aquele modelo)."""
    import re
    base = os.path.basename(carregar_modelo)
    m = re.match(r"^(.+)_(\d+)_steps(?:\.zip)?$", base)
    if not m:
        return None
    prefixo, n = m.group(1), m.group(2)
    pasta = os.path.dirname(carregar_modelo) or "."
    candidato = os.path.join(pasta, f"{prefixo}_vecnormalize_{n}_steps.pkl")
    return candidato if os.path.exists(candidato) else None

def _carregar_env(caminho: str = ".env") -> None:
    if not os.path.exists(caminho):
        return

    with open(caminho, "r", encoding="utf-8") as arquivo:
        for linha in arquivo:
            conteudo = linha.strip()
            if not conteudo or conteudo.startswith("#") or "=" not in conteudo:
                continue

            chave, valor = conteudo.split("=", 1)
            chave = chave.strip()
            valor = valor.strip().strip('"').strip("'")
            os.environ.setdefault(chave, valor)

_carregar_env()

# Decisão 7 — ligar a LSTM (RecurrentPPO) via FNAF_USAR_LSTM=1 no .env. Padrão 0 = feedforward
# (PPO), que é o CONTROLE do A/B. Trocar SÓ isso entre o controle e a LSTM.
USAR_LSTM = os.getenv("FNAF_USAR_LSTM", "0").strip() == "1"

def _env_int(nome: str, padrao: int) -> int:
    try:
        return int(os.getenv(nome, str(padrao)).strip())
    except (TypeError, ValueError):
        return padrao


def _env_float(nome: str, padrao: float) -> float:
    try:
        return float(os.getenv(nome, str(padrao)).strip())
    except (TypeError, ValueError):
        return padrao


# ── Pacote anti-colapso de entropia (BC warmstart + termostato + currículo) ───────────────────
# Objetivo: SUSTENTAR a exploração até o agente vencer noites avançadas, sem os dois modos de
# falha já observados: ent_coef fixo alto (0.03 = aleatório demais, morria por apagão) e fixo
# baixo (0.02 = colapsa); e sem o gate por win_rate, que nunca abria quando o agente estagnava
# abaixo dele. Tudo sobrescrevível por env var no .env.
#
#   n_steps 4096      : meio-termo — 8192 protegia bem a política INICIAL ruim, mas custava 1
#                       update a cada ~95min de jogo real; com BC warmstart a política já parte
#                       decente e a FREQUÊNCIA de updates volta a importar (~48min/update, ~5.4
#                       noites de diversidade por lote). Contingência p/ aprendizado lento:
#                       FNAF_N_EPOCHS=6 (o target_kl corta épocas excedentes com segurança).
#   n_epochs↓ + target_kl: menos reuso/super-otimização do MESMO lote (10× afiava cedo demais);
#                       o KL corta as épocas se a política andar longe demais.
#   clip_reward=100   : o default do VecNormalize (10) clipava a vitória (+500 normalizada) no
#                       MESMO teto de qualquer recompensa — a razão vitória:morte 5:1 do design
#                       chegava até 1:1 no gradiente. 100 deixa os terminais passarem inteiros.
#                       (Contingência se train/value_loss explodir: FNAF_CLIP_REWARD=50.)
#   H alvo (nats)     : o ControladorEntropia rastreia um ALVO de entropia que decai devagar
#                       (1.5 → 0.75 ≈ 4.5 → 2.1 ações efetivas; uniforme seria ln17 ≈ 2.83),
#                       ajustando o ent_coef em malha fechada — ver docstring da classe.
N_STEPS    = max(64, _env_int("FNAF_N_STEPS",   4096))
BATCH_SIZE = max(8,  _env_int("FNAF_BATCH_SIZE", 256))
N_EPOCHS   = max(1,  _env_int("FNAF_N_EPOCHS",     4))
TARGET_KL  = _env_float("FNAF_TARGET_KL",  0.03)
CLIP_REWARD = _env_float("FNAF_CLIP_REWARD", 100.0)
ENT_INICIO = _env_float("FNAF_ENT_INICIO", 0.02)   # valor INICIAL do ent_coef (o controlador assume dali)
H_INICIO   = _env_float("FNAF_H_INICIO", 1.5)      # alvo de entropia (nats) no começo do treino
H_FIM      = _env_float("FNAF_H_FIM",    0.75)     # alvo no fim (nunca 0 — congela em política subótima)
ENT_MIN    = _env_float("FNAF_ENT_MIN",  0.003)    # piso do ent_coef
ENT_MAX    = _env_float("FNAF_ENT_MAX",  0.03)     # teto: 0.03 FIXO já causou apagão; como teto
                                                   # transitório de controlador é seguro
ENT_GANHO  = _env_float("FNAF_ENT_GANHO", 0.7)     # ganho do controlador (subamortecido; oscilou? 0.4)
ENT_PASSO_MAX = _env_float("FNAF_ENT_PASSO_MAX", 0.20)  # passo máx por rollout (±20% em log-espaço)

# Warmup do crítico (usar junto com --bc): o BC clona só o ATOR — o value_net não recebe
# gradiente no BC e chega ALEATÓRIO no RL. Sem warmup, as primeiras vantagens (GAE) são ruído
# e o PPO pode erodir a política clonada logo nos primeiros updates. Durante os primeiros
# WARMUP_FRAC do treino: clip_range minúsculo (ator quase parado; o value_loss NÃO passa pelo
# clip e treina normal) + ent_coef preso em ENT_MIN (o bônus de entropia também não passa pelo
# clip — seria o único termo capaz de randomizar o BC). 0 = desligado.
WARMUP_FRAC = _env_float("FNAF_WARMUP_FRAC", 0.0)
WARMUP_CLIP = _env_float("FNAF_WARMUP_CLIP", 0.03)

# Currículo automático: promove a noite-alvo quando a janela CHEIA da noite alvo cruza o limiar.
CURRICULO_LIMIAR = _env_float("FNAF_CURRICULO_LIMIAR", 0.50)

# Métrica informativa: a partir de qual win_rate (janela móvel) na Noite 1 considerá-la "dominada".
# Só um alerta no resumo [POR NOITE] — quem muda o alvo é o CurriculumCallback.
NOITE1_DOMINIO = _env_float("FNAF_NOITE1_DOMINIO", 0.60)


def _env_str_obrigatorio(nome: str) -> str:
    valor = os.getenv(nome)
    if valor is None or valor.strip() == "":
        raise ValueError(f"Variavel obrigatoria ausente no .env: {nome}")
    return valor.strip()


class LogCallback(BaseCallback):
    def __init__(self, log_steps: bool = False):
        super().__init__()
        self.episodio          = 0
        self.episodios_validos = 0
        self.mortes            = 0
        self.vitorias          = 0
        self.interrompidos     = 0
        self.noite_max         = 1     # noite mais avançada já alcançada (contexto p/ checkpoints)
        self.recompensa_total  = 0.0
        self._pausa_disponivel = True
        self._log_steps        = log_steps

        os.makedirs("logs/analise", exist_ok=True)
        cabecalho = f"\n{'='*60}\nTreino iniciado\n{'='*60}\n"

        # Convenção de logs: treino.log (e o console) ficam ENXUTOS — é o que se lê durante a
        # execução. Telemetria p/ análise (Energia fim, Causa, linhas OCORRIDO) vai para
        # logs/analise/treino_detalhado.log, que é o que os parsers (metricas_treino,
        # enviar_logs_mongodb) preferem ler.
        self.arquivo_log = open("logs/treino.log", "a", encoding="utf-8")
        self.arquivo_log.write(cabecalho)

        self.arquivo_log_detalhado = open("logs/analise/treino_detalhado.log", "a", encoding="utf-8")
        self.arquivo_log_detalhado.write(cabecalho)

        self.arquivo_log_steps = None
        if log_steps:
            self.arquivo_log_steps = open("logs/analise/treino_steps.log", "a", encoding="utf-8")
            self.arquivo_log_steps.write(cabecalho)

    def _on_step(self) -> bool:
        # F12 pausa a IA — segura para pausar, larga para continuar
        if self._pausa_disponivel:
            try:
                while keyboard.is_pressed("F12"):
                    print("PAUSADO — solte F12 para continuar...", end="\r")
                    time.sleep(0.5)
            except Exception as erro:
                print(
                    "Aviso: pausa por F12 desativada nesta execucao. "
                    f"Motivo: {erro}"
                )
                self._pausa_disponivel = False

        info = self.locals.get("infos", [{}])[0]
        # Com VecNormalize, locals["rewards"] vem normalizado; loga a recompensa real.
        try:
            recompensa_step = self.training_env.get_original_reward()[0]
        except Exception:
            recompensa_step = self.locals.get("rewards", [0])[0]
        self.recompensa_total += recompensa_step

        energia = info.get("energia")
        if energia is not None:
            pe    = int(info.get("porta_esq",     False))
            pd    = int(info.get("porta_dir",     False))
            le    = int(info.get("luz_esq",       False))
            ld    = int(info.get("luz_dir",       False))
            ca    = int(info.get("camera_aberta", False))
            cv    = int(info.get("camera_ativa",  0))
            acao  = info.get("acao_nome", "?")
            valida = "OK" if info.get("acao_valida", True) else "X "
            linha_step = (
                f"{_env_str_obrigatorio('PC')} | "
                f"Ep {self.episodio:4d} | "
                f"E:{energia:5.1f}% | "
                f"PE:{pe} PD:{pd} LE:{le} LD:{ld} | "
                f"CAM:{ca}/{cv:2d} | "
                f"#{info.get('passos', 0):5d} | "
                f"{acao:<20} [{valida}]"
            )
            if self._log_steps:
                print(linha_step)
            if self.arquivo_log_steps:
                self.arquivo_log_steps.write(linha_step + "\n")
                self.arquivo_log_steps.flush()

        done = self.locals.get("dones", [False])[0]
        if done:
            # tempo_real vem do ambiente — medido antes do reset do próximo episódio
            tempo_ep_minutos = info.get("tempo_real", 0.0) / 60.0

            self.episodio += 1
            self.noite_max = max(self.noite_max, info.get("noite", 1))
            interrompido = info.get("interrompido", False)

            if interrompido:
                self.interrompidos += 1
                resultado = "INTERROMPIDO"
            elif info.get("vitoria", False):     # 6AM REAL — truncamento não é vitória
                self.episodios_validos += 1
                self.vitorias += 1
                resultado = "VITORIA"
            elif info.get("morreu", False):
                self.episodios_validos += 1
                self.mortes += 1
                resultado = "MORTE"
            else:
                # Truncado (700s/max_passos) sem morte nem 6AM: o rótulo antigo virava
                # VITORIA e inflava a taxa do log com episódios-fantasma presos no menu.
                # Conta como válido SEM vitória — igual ao gate/MetricasPorNoite.
                self.episodios_validos += 1
                resultado = "TRUNCADO"

            taxa_vitoria = (
                (self.vitorias / self.episodios_validos) * 100
                if self.episodios_validos > 0
                else 0.0
            )

            # Linha ENXUTA no console e no treino.log (leitura de execução). A telemetria de
            # desfecho — "sem saber DO QUE morre, cada ajuste é chute" — vai na MESMA linha,
            # mas só no detalhado: campos "Energia fim" e "Causa" no fim, na ordem que o
            # LOG_PATTERN do mongodb e o CAUSA do metricas_treino já preveem.
            linha = (
                f"{_env_str_obrigatorio('PC')} | "
                f"Ep {self.episodio:4d} | "
                f"Noite {info.get('noite', 1)} | "
                f"{resultado:8s} | "
                f"Passos: {info.get('passos', 0):6d} | "
                f"Tempo: {tempo_ep_minutos:7.2f} min | "
                f"Recompensa: {self.recompensa_total:8.1f} | "
                f"Taxa vitória: {taxa_vitoria:.1f}%"
            )
            causa = info.get("causa")
            detalhe = (
                f" | Energia fim: {info.get('energia', 0.0):5.1f}% | Causa: {causa}"
                if causa else ""
            )

            print(linha)
            self.arquivo_log.write(linha + "\n")
            self.arquivo_log_detalhado.write(linha + detalhe + "\n")

            ocorrido = info.get("ocorrido")
            if interrompido and ocorrido:
                # Diagnóstico de interrupção: só no log de análise (execução fica limpa).
                self.arquivo_log_detalhado.write(
                    f"{_env_str_obrigatorio('PC')} | "
                    f"Ep {self.episodio:4d} | "
                    f"OCORRIDO | {ocorrido}\n"
                )

            self.arquivo_log.flush()
            self.arquivo_log_detalhado.flush()
            self.recompensa_total = 0.0

        return True

    def _on_training_end(self):
        for arquivo in (self.arquivo_log, self.arquivo_log_detalhado, self.arquivo_log_steps):
            if arquivo and not arquivo.closed:
                arquivo.write("Treino finalizado\n")
                arquivo.close()


class ControladorEntropia(BaseCallback):
    """Termostato de entropia — substitui o EntropiaSchedule (gate binário por win_rate).

    Por que malha fechada: não existe ent_coef FIXO certo. 0.03 deixava a política aleatória
    demais p/ conservar energia (morria por apagão ~7min); 0.02 colapsava; e o gate por
    win_rate nunca abria quando o agente estagnava abaixo dele (ent_coef travado p/ sempre).
    Aqui o sensor é a PRÓPRIA entropia H da política: começando a colapsar → o coeficiente
    sobe sozinho; aleatória demais → desce. O alvo decai devagar ao longo do treino
    (H_INICIO → H_FIM nats; perplexidade e^1.5 ≈ 4.5 → e^0.75 ≈ 2.1 ações efetivas, de um
    máximo uniforme ln(17) ≈ 2.83) — consolidação gradual SEM depender de cruzar win_rate.

    Medição: em _on_rollout_end, logger.name_to_value["train/entropy_loss"] ainda guarda o
    train() do rollout ANTERIOR (ordem do SB3: collect_rollouts → _on_rollout_end →
    _dump_logs → train), e entropy_loss = -mean(H) tanto no PPO quanto no RecurrentPPO.
    Defasagem de 1 rollout — irrelevante: o colapso se desenrola por dezenas de rollouts.

    Ajuste multiplicativo em log-espaço com passo clampado (±ENT_PASSO_MAX por rollout):
    com 1 update a cada ~48min precisa reagir em poucas horas, sem serrilhar (ganho < 1,
    subamortecido; medição já é a média de todas as épocas/minibatches do train). O PPO lê
    model.ent_coef (float) a cada train(), então basta setá-lo entre rollouts."""
    def __init__(self, h_inicio: float = H_INICIO, h_fim: float = H_FIM,
                 ent_inicial: float = ENT_INICIO, ent_min: float = ENT_MIN,
                 ent_max: float = ENT_MAX, ganho: float = ENT_GANHO,
                 passo_max: float = ENT_PASSO_MAX):
        super().__init__()
        self.h_inicio, self.h_fim = h_inicio, h_fim
        self.ent_inicial          = ent_inicial
        self.ent_min, self.ent_max = ent_min, ent_max
        self.ganho, self.passo_max = ganho, passo_max

    def _em_warmup(self) -> bool:
        # Janela de warmup do crítico (BC→PPO): ent_coef preso em ENT_MIN — o bônus de
        # entropia não passa pelo clip_range e randomizaria a política clonada.
        return WARMUP_FRAC > 0 and self.model._current_progress_remaining > 1.0 - WARMUP_FRAC

    def _on_training_start(self) -> None:
        self.model.ent_coef = self.ent_min if self._em_warmup() else self.ent_inicial

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        prog   = self.model._current_progress_remaining        # 1.0 → 0.0
        h_alvo = self.h_fim + (self.h_inicio - self.h_fim) * prog
        entropy_loss = self.model.logger.name_to_value.get("train/entropy_loss")
        em_warmup = self._em_warmup()
        if entropy_loss is not None:                           # ausente só antes do 1º train()
            h = -float(entropy_loss)
            self.logger.record("custom/entropia", h)           # H crua, direta (sem conta de cabeça)
            if not em_warmup:                                  # warmup: controlador em espera
                erro  = self.ganho * (h_alvo - h)
                fator = math.exp(min(max(erro, -self.passo_max), self.passo_max))
                novo  = float(self.model.ent_coef) * fator
                self.model.ent_coef = float(min(max(novo, self.ent_min), self.ent_max))
        if em_warmup:
            self.model.ent_coef = self.ent_min
        self.logger.record("custom/warmup", 1.0 if em_warmup else 0.0)
        self.logger.record("custom/h_alvo", h_alvo)
        self.logger.record("custom/ent_coef", float(self.model.ent_coef))


class MetricasPorNoite(BaseCallback):
    """Quebra as métricas POR NOITE para informar a decisão de currículo (new_game → continue).

    O win_rate agregado MENTE pra essa decisão: junta muitas vitórias de Noite 1 com poucas mortes
    de Noite 2 num número só. Aqui cada noite tem sua própria janela móvel, então dá pra ver:
      • Noite 1 já está DOMINADA? (win_rate alto e estável) → vale forçar a Noite 2 (continue).
      • Noite 2 está APRENDENDO? (tempo de sobrevivência subindo ao longo do tempo) — esta é a
        métrica certa depois de trocar, já que a taxa de vitória da Noite 2 fica ~0 por um tempo.

    Loga em custom/noite_{n}/... e imprime um resumo legível a cada `resumo_a_cada` episódios.
    É SÓ medição/alerta — não muda o treino nem decide nada sozinho."""
    # Rótulos de _classificar_desfecho + morte_golden (janela fechada, _interromper_episodio).
    # Gravar 0.0 nos ausentes mantém as séries do tensorboard contínuas (fração cai a zero
    # visivelmente em vez de congelar no último valor).
    ROTULOS_CAUSA = ("vitoria_gerida", "vitoria_apagao", "morte_energia",
                     "morte_animatronico", "menu_crash", "morte_golden")

    def __init__(self, janela: int = 30, resumo_a_cada: int = 20, dominio: float = NOITE1_DOMINIO):
        super().__init__()
        self.janela, self.resumo_a_cada, self.dominio = janela, resumo_a_cada, dominio
        self.win  = defaultdict(lambda: deque(maxlen=janela))   # noite -> deque(0/1)
        self.surv = defaultdict(lambda: deque(maxlen=janela))   # noite -> deque(tempo_jogo s)
        self.win_geral = deque(maxlen=50)   # todas as noites — era do EntropiaSchedule (dashboards)
        self.causas    = deque(maxlen=50)   # rótulo de desfecho por episódio (skill vs sorte)
        self.n_eps = 0

    def _on_step(self) -> bool:
        if self.locals.get("dones", [False])[0]:
            info = self.locals.get("infos", [{}])[0]
            if info.get("interrompido", False):     # interrompido não conta (fora da métrica de skill)
                return True
            noite = int(info.get("noite", 1))
            self.win[noite].append(1 if info.get("vitoria", False) else 0)   # 6AM real, não truncamento
            self.surv[noite].append(float(info.get("tempo", 0.0)))
            self.win_geral.append(1 if info.get("vitoria", False) else 0)
            if info.get("causa"):
                self.causas.append(info["causa"])
            self.n_eps += 1
            if self.n_eps % self.resumo_a_cada == 0:
                self._imprimir_resumo()
        return True

    def _noite1_dominada(self) -> bool:
        d = self.win[1]
        return len(d) >= self.janela and (sum(d) / len(d)) >= self.dominio

    def _on_rollout_end(self) -> None:
        for noite, d in self.win.items():
            if d:
                self.logger.record(f"custom/noite_{noite}/win_rate", sum(d) / len(d))
                self.logger.record(f"custom/noite_{noite}/n_eps", len(d))
        for noite, d in self.surv.items():
            if d:
                self.logger.record(f"custom/noite_{noite}/sobrevivencia_s", sum(d) / len(d))
        # Sinal de currículo: 1 = Noite 1 dominada (hora de considerar o continue).
        self.logger.record("custom/curriculo/noite1_dominada", 1.0 if self._noite1_dominada() else 0.0)
        # win_rate_50: mesma janela do antigo gate (rolling-50, não a cumulativa do treino.log).
        if self.win_geral:
            self.logger.record("custom/win_rate_50", sum(self.win_geral) / len(self.win_geral))
        # Fração de cada desfecho na janela: responde "morre de QUÊ" (apagão = política
        # gastadora/aleatória; animatrônico = defesa/timing) direto no tensorboard.
        if self.causas:
            contagem = Counter(self.causas)
            for rotulo in self.ROTULOS_CAUSA:
                self.logger.record(f"custom/causas/{rotulo}", contagem.get(rotulo, 0) / len(self.causas))

    def _imprimir_resumo(self) -> None:
        partes = []
        for noite in sorted(self.win):
            d, s = self.win[noite], self.surv[noite]
            wr = (sum(d) / len(d) * 100) if d else 0.0
            sv = (sum(s) / len(s)) if s else 0.0
            partes.append(f"N{noite}: {wr:3.0f}% vit | {sv:4.0f}s | {len(d):2d}ep")
        print(f"[POR NOITE | janela {self.janela}]  " + "   ".join(partes))
        if self._noite1_dominada():
            print(f"   -> Noite 1 DOMINADA (>= {self.dominio*100:.0f}%). Considere "
                  f"FNAF_RESET_METODO=continue p/ forcar a Noite 2 (a metrica vai PIORAR de inicio — normal).")
        else:
            wr1 = (sum(self.win[1]) / len(self.win[1]) * 100) if self.win[1] else 0.0
            print(f"   -> Noite 1 ainda nao dominada ({wr1:.0f}% < {self.dominio*100:.0f}%) — siga em new_game.")


class CurriculumCallback(BaseCallback):
    """Currículo AUTOMÁTICO: promove a noite-alvo do env (noite_desejada, lida por
    decidir_reset no modo continue) quando a janela CHEIA da noite alvo em MetricasPorNoite
    cruza o limiar. Substitui a rotina manual de editar FNAF_NOITE_DESEJADA e reiniciar.

      • Janela cheia (>= janela_min eps da noite ALVO) — não mistura noites, ao contrário
        do antigo gate de entropia, e não promove por meia dúzia de episódios sortudos.
      • NUNCA rebaixa: regressão de skill se recupera praticando (morte na noite k <= alvo
        → Continue retoma a noite k — as anteriores continuam no caminho).
      • Persiste em modelos/curriculo.json a cada promoção; ao retomar aplica
        max(.env, json) — p/ regredir de propósito, apague/edite o json.

    Deve ser registrado DEPOIS de MetricasPorNoite na lista de callbacks (lê as janelas
    dele no mesmo _on_step, já atualizadas). noite_desejada só é lida em _preparar_reset
    (próximo reset), então mudar a qualquer momento é seguro."""
    CAMINHO = f"{PASTA_MODELOS}/curriculo.json"

    def __init__(self, metricas: MetricasPorNoite, limiar: float = CURRICULO_LIMIAR,
                 janela_min: int = 30, noite_max: int = MAX_NOITE):
        super().__init__()
        self.metricas   = metricas
        self.limiar     = limiar
        self.janela_min = janela_min
        self.noite_max  = noite_max
        self.alvo       = 1

    def _on_training_start(self) -> None:
        alvo = int(self.training_env.get_attr("noite_desejada")[0])   # valor inicial do .env
        if os.path.exists(self.CAMINHO):
            try:
                with open(self.CAMINHO, "r", encoding="utf-8") as f:
                    alvo = max(alvo, int(json.load(f).get("noite_desejada", alvo)))
            except (ValueError, OSError, TypeError) as erro:
                print(f"[curriculo] aviso: nao li {self.CAMINHO} ({erro}) — seguindo com alvo {alvo}")
        if RESET_METODO != "continue":
            print("[curriculo] AVISO: FNAF_RESET_METODO != continue — a promocao nao muda os resets "
                  "(new_game sempre volta pra Noite 1).")
        self._aplicar(alvo)
        print(f"[curriculo] noite-alvo inicial: {self.alvo} "
              f"(promove a {self.limiar*100:.0f}% na janela de {self.janela_min} eps)")

    def _aplicar(self, alvo: int) -> None:
        self.alvo = alvo
        # set_attr atravessa VecNormalize → DummyVecEnv → FNAFEnv (atributo de instância).
        self.training_env.set_attr("noite_desejada", alvo)
        try:
            with open(self.CAMINHO, "w", encoding="utf-8") as f:
                json.dump({"noite_desejada": alvo, "step": int(self.num_timesteps),
                           "quando": datetime.now().isoformat(timespec="seconds")}, f)
        except OSError as erro:
            print(f"[curriculo] aviso: nao salvei {self.CAMINHO}: {erro}")

    def _on_step(self) -> bool:
        if self.locals.get("dones", [False])[0]:
            d = self.metricas.win[self.alvo]      # janela SÓ da noite alvo
            if (self.alvo < self.noite_max and len(d) >= self.janela_min
                    and sum(d) / len(d) >= self.limiar):
                print(f"[curriculo] Noite {self.alvo} dominada "
                      f"({sum(d) / len(d) * 100:.0f}% na janela de {len(d)}) -> alvo Noite {self.alvo + 1}")
                self._aplicar(self.alvo + 1)
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record("custom/curriculo/noite_alvo", float(self.alvo))


class CheckpointComLog(CheckpointCallback):
    """CheckpointCallback que ANUNCIA cada checkpoint no terminal com o contexto do treino
    (lido do LogCallback). Serve pra decidir até onde é seguro voltar: mostra o arquivo salvo,
    o step, o episódio em andamento e a saúde naquele instante (eps válidos, vitórias, noite
    máx). O checkpoint é tirado ENTRE steps, então pega o episódio atual no meio — por isso o
    print diz 'durante o ep X' e qual foi o último concluído."""

    def __init__(self, *args, log_callback: "LogCallback | None" = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._log = log_callback

    def _on_step(self) -> bool:
        resultado = super()._on_step()
        if self.save_freq > 0 and self.n_calls % self.save_freq == 0:
            nome = f"{self.name_prefix}_{self.num_timesteps}_steps.zip"
            if self._log is not None:
                concluidos = self._log.episodio
                validos    = self._log.episodios_validos
                taxa = (self._log.vitorias / validos * 100) if validos else 0.0
                contexto = (
                    f"   step {self.num_timesteps:,} | tirado DURANTE o ep {concluidos + 1} "
                    f"(ultimo ep CONCLUIDO: {concluidos})\n"
                    f"   ate aqui: {validos} eps validos | {self._log.vitorias} vitorias "
                    f"({taxa:.1f}%) | noite max alcancada: {self._log.noite_max}"
                )
            else:
                contexto = f"   step {self.num_timesteps:,}"
            print(f"\n[CHECKPOINT] salvo: {nome}\n{contexto}\n")
        return resultado


def treinar(timesteps: int = 500_000, carregar_modelo: str = None, log_steps: bool = False,
            bc_path: str = None):
    print("Iniciando ambiente FNAF1...")
    print("ATENÇÃO: Deixe o jogo aberto e na tela inicial!")
    print("Dica: segure F12 a qualquer momento para pausar.\n")
    # Guarda anti-incidente: o .env é LOCAL por máquina (git-ignorado) — um `git pull` NÃO
    # atualiza a fase. Uma run de 27h já foi queimada por FNAF_USAR_LSTM=1 esquecido.
    if USAR_LSTM:
        print("=" * 70)
        print("AVISO: FNAF_USAR_LSTM=1 -> este treino usara RecurrentPPO (LSTM).")
        print("A fase atual do projeto e FEEDFORWARD + BC warmstart (FNAF_USAR_LSTM=0);")
        print("a LSTM so volta no A/B apos o gatilho de docs/PACOTE_BC_ENTROPIA.md secao 2.7.")
        print("Confirme que e intencional — o .env desta maquina NAO vem com o git pull.")
        print("=" * 70)
    time.sleep(3)

    env_base = DummyVecEnv([lambda: FNAFEnv()])
    if carregar_modelo:
        # Prefere o vecnormalize salvo JUNTO ao checkpoint (save_vecnormalize=True) p/ o par
        # modelo+normalizacao casar; cai no vecnormalize global se o do checkpoint nao existir
        # (ex.: retomar o fnaf_ppo_final, ou checkpoints antigos sem o .pkl correspondente).
        caminho_stats = _vecnormalize_do_checkpoint(carregar_modelo) or CAMINHO_STATS
        if os.path.exists(caminho_stats):
            print(f"Carregando normalizacao: {caminho_stats}")
            env = VecNormalize.load(caminho_stats, env_base)
            # O pickle carrega o clip da ÉPOCA do save (ex.: 10, o default antigo, que clipava
            # a vitória +500 no mesmo teto da morte). Força o atual em qualquer retomada.
            env.clip_reward = CLIP_REWARD
        else:
            env = VecNormalize(env_base, norm_obs=False, norm_reward=True, gamma=GAMMA,
                               clip_reward=CLIP_REWARD)
    else:
        env = VecNormalize(env_base, norm_obs=False, norm_reward=True, gamma=GAMMA,
                           clip_reward=CLIP_REWARD)

    # Decisão 7 — memória: USAR_LSTM troca PPO (feedforward, controle do A/B) por RecurrentPPO
    # (LSTM, reusando o MultimodalExtractor). Só o ALGORITMO muda — recompensa/VecNormalize/gamma/
    # noite/schedules iguais ao controle. Ligar via FNAF_USAR_LSTM=1 no .env.
    Modelo   = RecurrentPPO if USAR_LSTM else PPO
    politica = "MultiInputLstmPolicy" if USAR_LSTM else "MultiInputPolicy"
    if carregar_modelo and os.path.exists(carregar_modelo):
        print(f"Carregando modelo ({'LSTM' if USAR_LSTM else 'PPO'}): {carregar_modelo}")
        modelo = Modelo.load(carregar_modelo, env=env)
    else:
        print(f"Criando novo modelo {'RecurrentPPO (LSTM)' if USAR_LSTM else 'PPO'}...")
        policy_kwargs = dict(features_extractor_class=MultimodalExtractor)
        if USAR_LSTM:                          # LSTM pequena (Decisão 7): menos amostra/instabilidade
            policy_kwargs.update(lstm_hidden_size=128, n_lstm_layers=1, enable_critic_lstm=True)
        modelo = Modelo(
            policy=politica,
            env=env,
            policy_kwargs=policy_kwargs,
            learning_rate=linear(3e-4, 3e-5),  # Decisão 6: decai 3e-4 → 3e-5 (piso p/ retomada)
            clip_range=clip_com_warmup(WARMUP_FRAC),  # warmup do crítico (BC→PPO); 0.2 fora dele
            n_steps=N_STEPS,                   # bundle: rollout maior (~11 noites) → mais diversidade
            batch_size=BATCH_SIZE,             # bundle: escala com n_steps
            n_epochs=N_EPOCHS,                 # bundle: menos reuso do lote (default 4)
            gamma=GAMMA,
            ent_coef=ENT_INICIO,               # valor inicial — o ControladorEntropia assume dali
            target_kl=TARGET_KL,               # bundle: freio extra contra colapso de entropia
            verbose=0,
            # Subpasta própria: a raiz de logs/ fica só com treino.log + desyncs.log.
            # `tensorboard --logdir logs` continua funcionando (o TB varre subpastas).
            tensorboard_log=f"{PASTA_LOGS}/tensorboard",
            device="auto",
        )
        # BC warmstart (opcional): inicializa a percepção a partir de um checkpoint (modelo de
        # BC ou outro). É INIT, não recompensa — o RL fica livre p/ divergir (não fixa o ótimo).
        # Compatível com a LSTM (transfere só o MultimodalExtractor; ver transferir_pesos).
        if bc_path:
            if os.path.exists(bc_path):
                from src.agent.behavioral_cloning import transferir_pesos
                transferir_pesos(modelo, bc_path)
            else:
                print(f"[BC warmstart] caminho não encontrado, ignorando: {bc_path}")

    # Os knobs que NÃO mexem na forma do rollout buffer (n_epochs, target_kl) podem ser trocados
    # mesmo num modelo carregado, então valem também ao RETOMAR. Já n_steps/batch_size definem o
    # buffer e NÃO são reconfiguráveis no load — só valem em treino fresco (--novo); por isso o print
    # lê modelo.n_steps/batch_size (mostra o valor REAL em uso, herdado do checkpoint se for retomada).
    modelo.n_epochs  = N_EPOCHS
    modelo.target_kl = TARGET_KL
    print(f"[hparams] n_steps={modelo.n_steps} batch={modelo.batch_size} n_epochs={N_EPOCHS} "
          f"target_kl={TARGET_KL} clip_reward={CLIP_REWARD} peso_energia={PESO_ENERGIA} | "
          f"ent_coef inicial {ENT_INICIO} | "
          f"controlador: H {H_INICIO}->{H_FIM} nats, ent [{ENT_MIN}, {ENT_MAX}], ganho {ENT_GANHO}"
          + (f" | warmup do crítico: {WARMUP_FRAC*100:.0f}% do treino (clip {WARMUP_CLIP})"
             if WARMUP_FRAC > 0 else ""))
    if WARMUP_FRAC > 0 and not (bc_path and carregar_modelo is None):
        print("[warmup] AVISO: FNAF_WARMUP_FRAC > 0 faz sentido em treino FRESCO com --bc "
              "(ator clonado + crítico frio). Sem BC, só atrasa o início; em retomada, a janela "
              "já passou (o progresso continua).")

    # log_callback ANTES do checkpoint na lista: assim, no step do save, o contador de episódio
    # já está atualizado quando CheckpointComLog imprime o contexto.
    log_callback = LogCallback(log_steps=log_steps)
    checkpoint = CheckpointComLog(
        save_freq=10_000,
        save_path=PASTA_MODELOS,
        name_prefix=f"{_env_str_obrigatorio('PC')}_fnaf_ppo",
        save_vecnormalize=True,   # salva o vecnormalize JUNTO de cada checkpoint (par casado p/ retomar)
        log_callback=log_callback,
    )
    entropia = ControladorEntropia()     # termostato: ent_coef em malha fechada sobre a H medida
    metricas_noite = MetricasPorNoite()  # quebra win_rate/sobrevivência/causas por noite
    curriculum = CurriculumCallback(metricas_noite)  # DEPOIS de metricas_noite: lê as janelas dele

    print(f"Treinando por {timesteps:,} timesteps...\n")
    try:
        modelo.learn(
            total_timesteps=timesteps,
            callback=[log_callback, checkpoint, entropia, metricas_noite, curriculum],
            reset_num_timesteps=carregar_modelo is None,
        )
    except KeyboardInterrupt:
        print("\nTreino interrompido pelo usuario. Salvando estado atual...")
    finally:
        for arquivo in (log_callback.arquivo_log, log_callback.arquivo_log_detalhado,
                        log_callback.arquivo_log_steps):
            if arquivo and not arquivo.closed:
                arquivo.write("Treino finalizado\n")
                arquivo.close()

        caminho_final = f"{PASTA_MODELOS}/fnaf_ppo_final"
        modelo.save(caminho_final)
        env.save(CAMINHO_STATS)
        print(f"\nModelo final salvo em: {caminho_final}")
        print(f"Normalizacao salva em: {CAMINHO_STATS}")

        env.close()


if __name__ == "__main__":
    treinar(
        timesteps=500_000,
        carregar_modelo=None,
        log_steps="--steps" in sys.argv,
    )