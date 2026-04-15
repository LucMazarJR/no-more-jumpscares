import sys
import numpy as np
from stable_baselines3 import PPO
from src.environment.fnaf_env import FNAFEnv

def merge_modelos(caminhos: list[str], saida: str = "modelos/fnaf_merged.zip"):
    """
    Faz a média dos pesos de múltiplos modelos PPO.
    Equivale ao Federated Learning — combina o aprendizado de vários PCs.
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
    print(f"Equivalente a {len(caminhos)}x mais experiência combinada!")

    env.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python merge_modelos.py modelo1.zip modelo2.zip [modelo3.zip ...]")
        print("Exemplo: python merge_modelos.py modelos/pc1_20k.zip modelos/pc2_20k.zip modelos/pc3_20k.zip")
        sys.exit(1)

    caminhos = sys.argv[1:]
    merge_modelos(caminhos)