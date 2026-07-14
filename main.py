import glob
import os
import re
import sys

from src.environment.fnaf_env import FNAFEnv, MAX_NOITE, FOXY_SATURACAO_S, INFO_SATURACAO_S


def encontrar_ultimo_modelo() -> str | None:
    """Retorna o checkpoint mais avançado em modelos/.

    Preferência: maior número de steps no nome; senão, o .zip mais recente.
    Modelos '*merged*' não têm mais prioridade — fazer média de pesos entre
    modelos treinados de inicializações diferentes produz uma política
    quebrada (ver aviso em scripts/merge_modelos.py).
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
    print(f"  - Noite: {obs['estados'][10] * MAX_NOITE:.0f}")
    print(f"  - Tempo sem câmera: {obs['estados'][11] * FOXY_SATURACAO_S:.1f}s")
    print(f"  - Idade da info esq: {obs['estados'][12] * INFO_SATURACAO_S:.1f}s")
    print(f"  - Idade da info dir: {obs['estados'][13] * INFO_SATURACAO_S:.1f}s")
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

    # BC warmstart (opcional): --bc <caminho.zip> inicializa a percepção a partir de um modelo
    # de BC. Combina bem com --novo (treino fresco). Compatível com a LSTM (FNAF_USAR_LSTM=1).
    bc_path = None
    if "--bc" in sys.argv:
        i = sys.argv.index("--bc")
        bc_path = sys.argv[i + 1] if i + 1 < len(sys.argv) else None
        if bc_path:
            print(f"BC warmstart: inicializando a partir de {bc_path}")

    treinar(timesteps=500_000, carregar_modelo=ultimo_modelo, bc_path=bc_path)


def modo_bc():
    """Treina Behavioral Cloning a partir de datasets de gameplay humano gravados
    (src/utils/gravar_gameplay.py). Gera modelos/fnaf_bc.zip para warmstart do treino.
    NÃO precisa do jogo aberto (lê frames já gravados).
    Uso: python main.py bc [dataset.json ...]
         Sem argumentos, pega automaticamente TODOS os dados/*/dataset.json."""
    from src.agent.behavioral_cloning import treinar_bc

    caminhos = [a for a in sys.argv[2:] if not a.startswith("--")]
    if not caminhos:
        caminhos = sorted(glob.glob("dados/*/dataset.json"))
        if caminhos:
            print(f"Nenhum caminho informado — usando os {len(caminhos)} datasets encontrados em dados/:")
            for caminho in caminhos:
                print(f"  {caminho}")
            print()

    if not caminhos:
        print("Uso: python main.py bc [dataset.json ...]")
        print("Sem argumentos, pega automaticamente todos os dados/*/dataset.json.")
        print("Grave demos antes com: python -m src.utils.gravar_gameplay --noite N")
        return
    treinar_bc(caminhos)


def modo_jogar():
    """Roda o modelo treinado em modo avaliação (sem aprender, determinístico).

    Decisão 5 (ablação da CNN): com `--ablacao imagem` ou `--ablacao estados`, zera aquele
    ramo da observação antes do predict. Se zerar a IMAGEM quase não derrubar a sobrevivência,
    a CNN não está contribuindo; se zerar os ESTADOS derrubar muito, a política depende deles.
    Compare a taxa de vitória + sobrevivência média vs. o run cheio (sem flag)."""
    import numpy as np

    ablacao = None
    if "--ablacao" in sys.argv:
        i = sys.argv.index("--ablacao")
        ablacao = sys.argv[i + 1] if i + 1 < len(sys.argv) else None
        if ablacao not in ("imagem", "estados"):
            print("Uso: python main.py jogar [--ablacao imagem|estados]")
            return

    caminho = encontrar_ultimo_modelo()
    if not caminho:
        print("Nenhum modelo encontrado em modelos/. Treine primeiro: python main.py treino")
        return

    # Decisão 7: FNAF_USAR_LSTM=1 carrega RecurrentPPO e propaga o estado da LSTM (igual ao treino).
    usar_lstm = os.getenv("FNAF_USAR_LSTM", "0").strip() == "1"
    print(f"Carregando modelo ({'LSTM' if usar_lstm else 'PPO'}): {caminho}")
    if ablacao:
        print(f">>> ABLAÇÃO (Decisão 5): ramo '{ablacao}' ZERADO na observação <<<")
    # Sem VecNormalize aqui de propósito: o treino normaliza só a recompensa
    # (norm_obs=False), então a política vê a observação crua igual no treino.
    # As stats de normalização só importam para retomar o treino, não para jogar.
    env = FNAFEnv()
    if usar_lstm:
        from sb3_contrib import RecurrentPPO
        modelo = RecurrentPPO.load(caminho, env=env)
    else:
        from stable_baselines3 import PPO
        modelo = PPO.load(caminho, env=env)

    episodio = 0
    vitorias = 0
    tempo_total = 0.0
    try:
        while True:
            episodio += 1
            obs, _ = env.reset()
            terminado = truncado = False
            recompensa_total = 0.0
            info = {}
            lstm_states = None                          # estado da LSTM (Decisão 7)
            ep_start = np.ones((1,), dtype=bool)        # sinaliza início → a LSTM zera o estado

            while not (terminado or truncado):
                obs_pred = obs
                if ablacao:                       # zera o ramo só para o predict (Decisão 5)
                    obs_pred = dict(obs)
                    obs_pred[ablacao] = np.zeros_like(obs[ablacao])
                if usar_lstm:
                    acao, lstm_states = modelo.predict(
                        obs_pred, state=lstm_states, episode_start=ep_start, deterministic=True)
                    ep_start = np.zeros((1,), dtype=bool)
                else:
                    acao, _ = modelo.predict(obs_pred, deterministic=True)
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

            tempo_total += info.get("tempo", 0.0)
            print(
                f"Ep {episodio:3d} | {resultado:12s} | "
                f"sobrev {info.get('tempo', 0.0):5.0f}s ({info.get('passos', 0):4d}p) | "
                f"rec {recompensa_total:7.1f} | "
                f"vitórias {vitorias}/{episodio} méd {tempo_total/episodio:.0f}s"
            )
    except KeyboardInterrupt:
        tag = f" [ablação: {ablacao}]" if ablacao else ""
        media = tempo_total / episodio if episodio else 0.0
        print(f"\nAvaliação encerrada{tag}. "
              f"Vitórias: {vitorias}/{episodio} | Sobrevivência média: {media:.0f}s")
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
    elif modo == "bc":
        modo_bc()
    else:
        print(f"Modo desconhecido: {modo}")
        print("Use: python main.py teste | python main.py treino [--novo] [--bc <zip>] | "
              "python main.py jogar [--ablacao imagem|estados] | "
              "python main.py bc [dataset.json ...]  (sem args: pega dados/*/dataset.json)")
