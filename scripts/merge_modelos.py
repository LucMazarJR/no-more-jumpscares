import sys
from stable_baselines3 import PPO
from src.environment.fnaf_env import FNAFEnv

AVISO = """
╔══════════════════════════════════════════════════════════════════════════╗
║  AVISO IMPORTANTE — leia antes de usar este script                        ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Fazer a média de pesos de redes neurais SÓ faz sentido quando os         ║
║  modelos vêm da MESMA linhagem de treino (mesma inicialização, ex.:       ║
║  checkpoints diferentes do mesmo run, ou runs que partiram do mesmo       ║
║  modelo base e treinaram pouco depois disso).                             ║
║                                                                            ║
║  Modelos treinados do zero em PCs diferentes têm inicializações           ║
║  aleatórias diferentes: os neurônios não se correspondem entre as redes.  ║
║  A média dos pesos NÃO combina o aprendizado — produz uma política        ║
║  quebrada, próxima do aleatório, e o treino "recomeça do zero" sem        ║
║  você perceber. (Federated averaging real exige inicialização comum e     ║
║  sincronização frequente.)                                                ║
║                                                                            ║
║  Se o objetivo é aproveitar o treino de vários PCs, escolha o MELHOR      ║
║  modelo (maior taxa de vitória / recompensa) e continue o treino dele.    ║
╚══════════════════════════════════════════════════════════════════════════╝
"""


def merge_modelos(caminhos: list[str], saida: str = "modelos/fnaf_merged.zip"):
    """Faz a média dos pesos de múltiplos modelos PPO.

    Só use com checkpoints da MESMA linhagem de treino (mesma inicialização).
    Para modelos independentes, o resultado é uma política quebrada.
    """
    print(f"Carregando {len(caminhos)} modelos...")

    env = FNAFEnv()

    # Carrega o primeiro modelo como base
    modelo_base = PPO.load(caminhos[0], env=env)
    params_base = modelo_base.policy.state_dict()

    print(f"  [{1}/{len(caminhos)}] {caminhos[0]} carregado")

    # Acumula os pesos de todos os outros modelos
    for i, caminho in enumerate(caminhos[1:], start=2):
        modelo = PPO.load(caminho, env=env)
        params = modelo.policy.state_dict()

        for chave in params_base:
            params_base[chave] = params_base[chave] + params[chave]

        print(f"  [{i}/{len(caminhos)}] {caminho} carregado")

    # Divide pelo número de modelos para fazer a média
    for chave in params_base:
        params_base[chave] = params_base[chave] / len(caminhos)

    # Aplica os pesos médios no modelo base
    modelo_base.policy.load_state_dict(params_base)

    # Salva o modelo merged
    modelo_base.save(saida)
    print(f"\nModelo merged salvo em: {saida}")

    env.close()


if __name__ == "__main__":
    print(AVISO)

    args = [a for a in sys.argv[1:] if a != "--force"]

    if "--force" not in sys.argv:
        print("Para confirmar que os modelos vêm da mesma linhagem de treino,")
        print("rode novamente com a flag --force:")
        print("  python scripts/merge_modelos.py --force modelo1.zip modelo2.zip [...]")
        sys.exit(1)

    if len(args) < 2:
        print("Uso: python scripts/merge_modelos.py --force modelo1.zip modelo2.zip [modelo3.zip ...]")
        sys.exit(1)

    merge_modelos(args)
