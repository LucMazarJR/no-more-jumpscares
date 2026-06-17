import os
import sys
import time
import keyboard
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.environment.fnaf_env import FNAFEnv
from src.agent.multimodal_policy import MultimodalExtractor

PASTA_MODELOS = "modelos"
PASTA_LOGS    = "logs"
GAMMA         = 0.995  # usado no PPO e no VecNormalize — precisam casar
CAMINHO_STATS = f"{PASTA_MODELOS}/vecnormalize.pkl"
os.makedirs(PASTA_MODELOS, exist_ok=True)
os.makedirs(PASTA_LOGS,    exist_ok=True)

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


def treinar(timesteps: int = 500_000, carregar_modelo: str = None, log_steps: bool = False):
    print("Iniciando ambiente FNAF1...")
    print("ATENÇÃO: Deixe o jogo aberto e na tela inicial!")
    print("Dica: segure F12 a qualquer momento para pausar.\n")
    time.sleep(3)

    env_base = DummyVecEnv([lambda: FNAFEnv()])
    if carregar_modelo and os.path.exists(CAMINHO_STATS):
        print(f"Carregando normalizacao: {CAMINHO_STATS}")
        env = VecNormalize.load(CAMINHO_STATS, env_base)
    else:
        env = VecNormalize(env_base, norm_obs=False, norm_reward=True, gamma=GAMMA)

    if carregar_modelo and os.path.exists(carregar_modelo):
        print(f"Carregando modelo: {carregar_modelo}")
        modelo = PPO.load(carregar_modelo, env=env)
    else:
        print("Criando novo modelo PPO...")
        policy_kwargs = dict(
            features_extractor_class=MultimodalExtractor,
        )
        modelo = PPO(
            policy="MultiInputPolicy",
            env=env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=GAMMA,
            ent_coef=0.01,
            verbose=0,
            tensorboard_log=PASTA_LOGS,
            device="auto",
        )

    checkpoint = CheckpointCallback(
        save_freq=10_000,
        save_path=PASTA_MODELOS,
        name_prefix=f"{_env_str_obrigatorio('PC')}_fnaf_ppo",
    )
    log_callback = LogCallback(log_steps=log_steps)

    print(f"Treinando por {timesteps:,} timesteps...\n")
    try:
        modelo.learn(
            total_timesteps=timesteps,
            callback=[checkpoint, log_callback],
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