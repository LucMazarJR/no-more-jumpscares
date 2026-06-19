"""Sonda de MEMORIA da LSTM (Decisao 7) - a recorrencia usa o historico ou esta inerte?

Analogo a ablacao da CNN (Decisao 5), mas para a memoria: alimenta a LSTM com HISTORIAS
diferentes e, na MESMA observacao final, mede com que frequencia a ACAO muda.
  - muda bastante -> a LSTM usa a memoria (rastreia o passado: essencial pro Foxy/Freddy).
  - quase nao muda -> recorrencia INERTE (decide so pelo frame atual, como um feedforward).
Compara com episode_start=True (estado mascarado -> nao deve mudar). Roda offline; precisa de um
modelo LSTM em modelos/ (senao usa um NAO treinado so pra validar a mecanica -> da ~0).

Rodar: python -m src.utils.sonda_memoria
"""

import glob
import os

import numpy as np

from src.environment.fnaf_env import FNAFEnv
from src.agent.multimodal_policy import MultimodalExtractor
from sb3_contrib import RecurrentPPO


def _modelo(env):
    zips = glob.glob("modelos/*.zip")
    if zips:
        caminho = max(zips, key=os.path.getctime)
        print(f"modelo: {caminho}\n")
        return RecurrentPPO.load(caminho, env=env), True
    print("AVISO: nenhum modelo em modelos/ - usando LSTM NAO TREINADA (so valida a mecanica).\n")
    modelo = RecurrentPPO(
        "MultiInputLstmPolicy", env,
        policy_kwargs=dict(features_extractor_class=MultimodalExtractor,
                           lstm_hidden_size=128, n_lstm_layers=1, enable_critic_lstm=True),
        n_steps=64, device="cpu", verbose=0,
    )
    return modelo, False


def _estado(modelo, env, n):
    """Acumula um estado interno alimentando n observacoes (uma 'historia')."""
    st, eps = None, np.ones((1,), dtype=bool)
    for _ in range(n):
        _, st = modelo.predict(env.observation_space.sample(), state=st,
                               episode_start=eps, deterministic=True)
        eps = np.zeros((1,), dtype=bool)
    return st


def main() -> int:
    env = FNAFEnv()
    modelo, treinado = _modelo(env)
    N = 40
    n0, n1 = np.zeros((1,), dtype=bool), np.ones((1,), dtype=bool)
    muda_hist = muda_reset = 0
    for _ in range(N):
        stA = _estado(modelo, env, int(np.random.randint(3, 8)))
        stB = _estado(modelo, env, int(np.random.randint(8, 15)))   # historia mais longa/diferente
        obs = env.observation_space.sample()
        aF, _ = modelo.predict(obs, state=stA, episode_start=n0, deterministic=True)  # historia conta
        bF, _ = modelo.predict(obs, state=stB, episode_start=n0, deterministic=True)
        aT, _ = modelo.predict(obs, state=stA, episode_start=n1, deterministic=True)  # estado mascarado
        bT, _ = modelo.predict(obs, state=stB, episode_start=n1, deterministic=True)
        muda_hist += int(int(aF) != int(bF))
        muda_reset += int(int(aT) != int(bT))
    env.close()

    print(f"acao muda com HISTORICO diferente (episode_start=False): {muda_hist/N:.2f}  ({muda_hist}/{N})")
    print(f"acao muda com estado MASCARADO (episode_start=True):     {muda_reset/N:.2f}  (deve ser ~0)")
    if not treinado:
        print("\nVeredito: (LSTM nao treinada - numeros sem significado; mecanica OK)")
        return 0
    if muda_hist / N < 0.05:
        print("\nVeredito: recorrencia INERTE - a acao quase nao depende do historico (memoria nao usada).")
    else:
        print("\nVeredito: a LSTM USA a memoria - o historico muda a decisao (rastreia o passado).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
