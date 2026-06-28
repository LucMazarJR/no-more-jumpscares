import os
import sys
import time
from collections import deque
import keyboard
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.environment.fnaf_env import FNAFEnv, GAMMA
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

# E1 (experimento de exploração, disciplina A/B): nº de épocas do PPO por rollout. Reusar muito
# (10) um lote de poucas trajetórias (~3 noites) super-otimiza e colapsa a entropia cedo. Default
# 10 = baseline (comportamento atual, no-op). Setar FNAF_N_EPOCHS=4 no .env pra rodar o E1.
N_EPOCHS = max(1, int(os.getenv("FNAF_N_EPOCHS", "10").strip()))


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

        os.makedirs("logs", exist_ok=True)
        cabecalho = f"\n{'='*60}\nTreino iniciado\n{'='*60}\n"

        self.arquivo_log = open("logs/treino.log", "a", encoding="utf-8")
        self.arquivo_log.write(cabecalho)

        self.arquivo_log_steps = None
        if log_steps:
            self.arquivo_log_steps = open("logs/treino_steps.log", "a", encoding="utf-8")
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
            elif info.get("morreu", False):
                self.episodios_validos += 1
                self.mortes += 1
                resultado = "MORTE"
            else:
                self.episodios_validos += 1
                self.vitorias += 1
                resultado = "VITORIA"

            taxa_vitoria = (
                (self.vitorias / self.episodios_validos) * 100
                if self.episodios_validos > 0
                else 0.0
            )

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

            print(linha)
            self.arquivo_log.write(linha + "\n")

            ocorrido = info.get("ocorrido")
            if interrompido and ocorrido:
                linha_ocorrido = (
                    f"{_env_str_obrigatorio('PC')} | "
                    f"Ep {self.episodio:4d} | "
                    f"OCORRIDO | {ocorrido}"
                )
                print(linha_ocorrido)
                self.arquivo_log.write(linha_ocorrido + "\n")

            self.arquivo_log.flush()
            self.recompensa_total = 0.0

        return True

    def _on_training_end(self):
        self.arquivo_log.write("Treino finalizado\n")
        self.arquivo_log.close()
        if self.arquivo_log_steps:
            self.arquivo_log_steps.write("Treino finalizado\n")
            self.arquivo_log_steps.close()


class EntropiaSchedule(BaseCallback):
    """Decai ent_coef de `inicio` p/ `fim` (Decisão 6), mas SÓ depois da taxa de vitória numa
    janela recente cruzar `gate` — antes disso mantém alto (explorar / não congelar antes de
    vencer). NUNCA vai a 0 (o md avisa: congela em política subótima). O PPO lê model.ent_coef
    (float) a cada train(), então basta setá-lo entre rollouts.

    O gate abre uma vez e não fecha: uma vez que o agente passa a vencer com consistência, começa
    a consolidar (decair). Se nunca cruzar o gate, ent_coef fica em `inicio` o treino todo."""
    def __init__(self, inicio: float = 0.02, fim: float = 0.005, gate: float = 0.20,
                 janela: int = 50):
        super().__init__()
        self.inicio, self.fim, self.gate, self.janela = inicio, fim, gate, janela
        self.resultados = deque(maxlen=janela)   # 1 = vitória, 0 = morte (ignora interrompidos)
        self._prog_gate = None                   # progress_remaining quando o gate abriu

    def _on_training_start(self) -> None:
        self.model.ent_coef = self.inicio

    def _on_step(self) -> bool:
        if self.locals.get("dones", [False])[0]:
            info = self.locals.get("infos", [{}])[0]
            if not info.get("interrompido", False):
                self.resultados.append(0 if info.get("morreu", False) else 1)

        prog = self.model._current_progress_remaining          # 1.0 → 0.0
        if self._prog_gate is None:
            taxa = (sum(self.resultados) / len(self.resultados)
                    if len(self.resultados) >= self.janela else 0.0)
            if taxa >= self.gate:
                self._prog_gate = prog                          # abre o decaimento
        if self._prog_gate is not None:
            # frac: 0 no gate → 1 no fim do treino
            frac = 1.0 - (prog / self._prog_gate) if self._prog_gate > 1e-9 else 1.0
            frac = min(max(frac, 0.0), 1.0)
            self.model.ent_coef = self.inicio + (self.fim - self.inicio) * frac
        return True

    def _on_rollout_end(self) -> None:
        # Instrumentação (Parte 0): expõe no tensorboard o que antes só dava pra reconstruir
        # de cabeça. win_rate_50 é a MESMA janela que o gate usa (rolling-50, não a cumulativa
        # do treino.log, que achata por construção). A entropia CRUA da política sai derivada
        # no tensorboard: H = -train/entropy_loss / custom/ent_coef.
        taxa = (sum(self.resultados) / len(self.resultados)) if self.resultados else 0.0
        self.logger.record("custom/ent_coef", float(self.model.ent_coef))
        self.logger.record("custom/win_rate_50", taxa)


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
        else:
            env = VecNormalize(env_base, norm_obs=False, norm_reward=True, gamma=GAMMA)
    else:
        env = VecNormalize(env_base, norm_obs=False, norm_reward=True, gamma=GAMMA)

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
            n_steps=2048,
            batch_size=64,
            n_epochs=N_EPOCHS,                 # E1: FNAF_N_EPOCHS (default 10 = baseline)
            gamma=GAMMA,
            ent_coef=0.02,                     # Decisão 6: começa alto; EntropiaSchedule decai depois
            verbose=0,
            tensorboard_log=PASTA_LOGS,
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

    # E1/A-B: o env var sempre vence — vale também ao RETOMAR (load restaura o n_epochs salvo no
    # checkpoint). Default 10 = no-op. Assim FNAF_N_EPOCHS controla o experimento em qualquer caso.
    modelo.n_epochs = N_EPOCHS
    print(f"n_epochs = {N_EPOCHS}" + ("  [E1 ativo]" if N_EPOCHS != 10 else "  (baseline)"))

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
    entropia = EntropiaSchedule()  # Decisão 6: decai ent_coef após a taxa de vitória estabilizar

    print(f"Treinando por {timesteps:,} timesteps...\n")
    try:
        modelo.learn(
            total_timesteps=timesteps,
            callback=[log_callback, checkpoint, entropia],
            reset_num_timesteps=carregar_modelo is None,
        )
    except KeyboardInterrupt:
        print("\nTreino interrompido pelo usuario. Salvando estado atual...")
    finally:
        if not log_callback.arquivo_log.closed:
            log_callback.arquivo_log.write("Treino finalizado\n")
            log_callback.arquivo_log.close()
        if log_callback.arquivo_log_steps and not log_callback.arquivo_log_steps.closed:
            log_callback.arquivo_log_steps.write("Treino finalizado\n")
            log_callback.arquivo_log_steps.close()

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