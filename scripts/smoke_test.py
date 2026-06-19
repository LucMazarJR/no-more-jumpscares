"""Smoke test do projeto — valida ambiente, policy e modelo SEM o jogo aberto.

Roda na raiz do projeto:
    venv\\Scripts\\python scripts\\smoke_test.py

Verifica:
  1. FNAFEnv constrói (templates + .env ok) e o espaço de observação tem 8 estados
  2. Episódio interrompido devolve observação compatível com o espaço (shape 8)
  3. Modelo PPO constrói com a MultimodalExtractor
  4. A CNN recebe pixels em [0, 1] — sem dupla normalização (bug antigo: ~0.004)
  5. model.predict funciona com observação do ambiente (channels-last)
  6. get_distribution funciona com batch channels-last (caminho do BC)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch as th
from stable_baselines3 import PPO

from src.environment.fnaf_env import FNAFEnv
from src.agent.multimodal_policy import MultimodalExtractor


def main():
    falhas = []

    # 1. Ambiente constrói e espaço correto
    env = FNAFEnv()
    assert env.observation_space["estados"].shape == (11,), \
        f"estados shape: {env.observation_space['estados'].shape}"
    assert env.observation_space["imagem"].shape == (84, 84, 1)
    print("[OK] 1. FNAFEnv construido — espaco de observacao: imagem(84,84,1) + estados(11)")

    # 2. Episódio interrompido devolve obs válida (sem tentar reabrir o jogo)
    env._abrir_jogo_fallback = lambda: False
    obs_int, _, terminado, _, info = env._interromper_episodio("smoke test")
    assert env.observation_space.contains(obs_int), "obs interrompida fora do espaco!"
    assert terminado and info.get("interrompido")
    print("[OK] 2. Episodio interrompido devolve observacao compativel (estados shape 11)")

    # 3. Modelo PPO com a extractor multimodal (mesma config do train.py)
    modelo = PPO(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs=dict(features_extractor_class=MultimodalExtractor),
        n_steps=64,
        verbose=0,
        device="cpu",
    )
    print("[OK] 3. Modelo PPO construido com MultimodalExtractor")

    # 4. Hook na primeira conv: imagem branca (255) deve chegar como 1.0 na CNN
    entradas = []
    hook = modelo.policy.features_extractor.cnn[0].register_forward_pre_hook(
        lambda mod, inp: entradas.append(inp[0].detach())
    )
    obs_branca = {
        "imagem": np.full((84, 84, 1), 255, dtype=np.uint8),
        "estados": np.zeros(11, dtype=np.float32),
    }
    acao, _ = modelo.predict(obs_branca, deterministic=True)
    hook.remove()
    pico = max(t.max().item() for t in entradas)
    assert 0.99 <= pico <= 1.01, (
        f"Pixel maximo na entrada da CNN: {pico:.6f} — esperado ~1.0. "
        "Valor ~0.0039 indica dupla normalizacao (/255 duas vezes)."
    )
    print(f"[OK] 4. CNN recebe pixels normalizados 1x (max={pico:.4f}) — agente enxerga a imagem")

    # 5. predict com observação amostrada do espaço
    obs_sample = env.observation_space.sample()
    acao, _ = modelo.predict(obs_sample, deterministic=True)
    assert env.action_space.contains(int(acao))
    print(f"[OK] 5. model.predict funciona (acao={int(acao)})")

    # 6. Caminho do behavioral cloning: batch channels-last direto na policy
    obs_bc = {
        "imagem": th.zeros(4, 84, 84, 1, dtype=th.uint8),
        "estados": th.zeros(4, 11, dtype=th.float32),
    }
    dist = modelo.policy.get_distribution(obs_bc)
    log_probs = dist.log_prob(th.zeros(4, dtype=th.long))
    assert log_probs.shape == (4,)
    print("[OK] 6. get_distribution aceita batch channels-last (caminho do BC)")

    env.close()
    print("\nTodos os checks passaram. Projeto pronto para treinar.")


if __name__ == "__main__":
    main()
