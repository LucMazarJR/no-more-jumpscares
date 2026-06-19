"""Teste do MASKING da LSTM (Decisão 7): o estado interno zera na troca de episódio.

O bug silencioso clássico da recorrência: se o estado da LSTM NÃO for zerado no início de um
episódio, a memória de uma partida vaza pra outra e o agente aprende lixo. Aqui confirmamos que,
com `episode_start=True`, a ação NÃO depende do estado que vinha antes — o RecurrentPPO mascara
(zera) o estado. Funciona com modelo NÃO treinado (é uma propriedade estrutural, não aprendida).

Rodar: python -m src.utils.testar_masking
"""

import numpy as np

from src.environment.fnaf_env import FNAFEnv
from src.agent.multimodal_policy import MultimodalExtractor
from sb3_contrib import RecurrentPPO


def main() -> int:
    env = FNAFEnv()
    modelo = RecurrentPPO(
        "MultiInputLstmPolicy", env,
        policy_kwargs=dict(features_extractor_class=MultimodalExtractor,
                           lstm_hidden_size=128, n_lstm_layers=1, enable_critic_lstm=True),
        n_steps=64, device="cpu", verbose=0,
    )

    def construir_estado(n):
        """Roda n steps alimentando observações p/ acumular um estado interno na LSTM."""
        st, eps = None, np.ones((1,), dtype=bool)
        for _ in range(n):
            _, st = modelo.predict(env.observation_space.sample(), state=st,
                                   episode_start=eps, deterministic=True)
            eps = np.zeros((1,), dtype=bool)
        return st

    st1 = construir_estado(5)
    st2 = construir_estado(9)        # história diferente → estado interno diferente

    obs = env.observation_space.sample()
    inicio = np.ones((1,), dtype=bool)
    a1, _ = modelo.predict(obs, state=st1, episode_start=inicio, deterministic=True)
    a2, _ = modelo.predict(obs, state=st2, episode_start=inicio, deterministic=True)

    # Sanidade: SEM o reset (episode_start=False), o estado pode mudar a ação.
    n0 = np.zeros((1,), dtype=bool)
    b1, _ = modelo.predict(obs, state=st1, episode_start=n0, deterministic=True)
    b2, _ = modelo.predict(obs, state=st2, episode_start=n0, deterministic=True)
    env.close()

    masking_ok = int(a1) == int(a2)
    print(f"episode_start=True  com estados diferentes -> acoes {int(a1)} e {int(a2)}  "
          f"[{'OK - masking zera o estado' if masking_ok else 'FALHA - estado VAZOU'}]")
    print(f"episode_start=False com estados diferentes -> acoes {int(b1)} e {int(b2)}  "
          f"[{'a memoria afeta a acao' if int(b1) != int(b2) else 'iguais (modelo nao treinado)'}]")
    return 0 if masking_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
