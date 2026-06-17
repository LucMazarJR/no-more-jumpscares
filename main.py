import glob
import os
import re
import sys

from src.environment.fnaf_env import FNAFEnv


def encontrar_ultimo_modelo() -> str | None:
    """Retorna o checkpoint mais avançado em modelos/.

    Preferência: maior número de steps no nome; senão, o .zip mais recente.
    Modelos '*merged*' não têm mais prioridade — fazer média de pesos entre
    modelos treinados de inicializações diferentes produz uma política
    quebrada (ver aviso em merge_modelos.py).
    """
    def extrair_steps(path):
        m = re.search(r"_(\d+)_steps\.zip$", path)
        return int(m.group(1)) if m else -1

    numerados = [p for p in glob.glob("modelos/*.zip") if extrair_steps(p) >= 0]
    if numerados:
        return max(numerados, key=extrair_steps)

    outros = glob.glob("modelos/*.zip")
    return max(outros, key=os.path.getctime) if outros else None


def modo_teste():
    print("Testando reset...")
    env = FNAFEnv()
    obs, _ = env.reset()
    print(f"Reset OK! Imagem: {obs['imagem'].shape}, Estados: {obs['estados'].shape}")
    print(f"Estados iniciais:")
    print(f"  - Porta esquerda: {obs['estados'][0]}")
    print(f"  - Porta direita: {obs['estados'][1]}")
    print(f"  - Luz esquerda: {obs['estados'][2]}")
    print(f"  - Luz direita: {obs['estados'][3]}")
    print(f"  - Câmera aberta: {obs['estados'][4]}")
    print(f"  - Câmera ativa: {obs['estados'][5]:.2f}")
    print(f"  - Energia: {obs['estados'][6]*100:.1f}%")
    print(f"  - Ameaça esquerda: {obs['estados'][8]}")
    print(f"  - Ameaça direita: {obs['estados'][9]}")
    input("O jogo iniciou a noite 1? (aperta Enter para confirmar)")
    env.close()


def modo_treino():
    from src.agent.train import treinar

    if "--novo" in sys.argv:
        print("Flag --novo: começando treino do zero (modelos antigos ignorados)")
        ultimo_modelo = None
    else:
        ultimo_modelo = encontrar_ultimo_modelo()
        if ultimo_modelo:
            print(f"Continuando treino: {ultimo_modelo}")
            print("(use 'python main.py treino --novo' para começar do zero)")
        else:
            print("Nenhum modelo encontrado — começando do zero")

    treinar(timesteps=500_000, carregar_modelo=ultimo_modelo)


def modo_jogar():
    """Roda o modelo treinado em modo avaliação (sem aprender, determinístico)."""
    from stable_baselines3 import PPO

    caminho = encontrar_ultimo_modelo()
    if not caminho:
        print("Nenhum modelo encontrado em modelos/. Treine primeiro: python main.py treino")
        return

    print(f"Carregando modelo: {caminho}")
    # Sem VecNormalize aqui de propósito: o treino normaliza só a recompensa
    # (norm_obs=False), então a política vê a observação crua igual no treino.
    # As stats de normalização só importam para retomar o treino, não para jogar.
    env = FNAFEnv()
    modelo = PPO.load(caminho, env=env)

    episodio = 0
    vitorias = 0
    try:
        while True:
            episodio += 1
            obs, _ = env.reset()
            terminado = truncado = False
            recompensa_total = 0.0
            info = {}

            while not (terminado or truncado):
                acao, _ = modelo.predict(obs, deterministic=True)
                obs, recompensa, terminado, truncado, info = env.step(int(acao))
                recompensa_total += recompensa

            if info.get("interrompido"):
                resultado = "INTERROMPIDO"
            elif info.get("morreu"):
                resultado = "MORTE"
            elif terminado:
                resultado = "VITORIA"
                vitorias += 1
            else:
                resultado = "TRUNCADO"

            print(
                f"Ep {episodio:3d} | {resultado:12s} | "
                f"Passos: {info.get('passos', 0):5d} | "
                f"Recompensa: {recompensa_total:8.1f} | "
                f"Vitórias: {vitorias}/{episodio}"
            )
    except KeyboardInterrupt:
        print(f"\nAvaliação encerrada. Vitórias: {vitorias}/{episodio}")
    finally:
        env.close()


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "teste"

    if modo == "teste":
        modo_teste()
    elif modo == "treino":
        modo_treino()
    elif modo == "jogar":
        modo_jogar()
    else:
        print(f"Modo desconhecido: {modo}")
        print("Use: python main.py teste | python main.py treino [--novo] | python main.py jogar")
