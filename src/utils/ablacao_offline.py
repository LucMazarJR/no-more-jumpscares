"""Diagnóstico OFFLINE da ablação (Decisão 5): a CNN (ramo de imagem) influencia a decisão?

PRA QUE SERVE
-------------
O agente decide a partir de DUAS fontes: a imagem 84x84 (processada pela CNN) e os 12 estados
(porta, luz, energia, ameaça...). Há o risco — já visto no projeto (bug da dupla normalização que
deixava a CNN cega) — da CNN estar "presente mas inerte": a rede ignora a imagem e decide só pelos
estados. Aí você paga o custo da CNN à toa e perde as pistas visuais. Este script detecta isso.

COMO FUNCIONA
-------------
Mede a SENSIBILIDADE da política, sem subir o jogo. Pega frames reais de debug/ + vetores de estado
plausíveis e mede o quanto a DISTRIBUIÇÃO DE AÇÕES muda quando:
  - varia a IMAGEM com os estados fixos  -> sensibilidade à imagem;
  - varia os ESTADOS com a imagem fixa   -> sensibilidade aos estados.
A "mudança" é a distância de variação total (TV ∈ [0,1]) entre os vetores de probabilidade de ação.
  - sensibilidade à imagem ≈ 0            -> a CNN está sendo IGNORADA (inerte).
  - sensibilidade à imagem comparável à dos estados -> a imagem contribui.
Usa imagens REAIS (sem o "preto OOD" da ablação por zerar a entrada) e roda em segundos. É o check
rápido; o teste comportamental definitivo (impacto no jogo) é `python main.py jogar --ablacao ...`.

COMO USAR
---------
  python -m src.utils.ablacao_offline
Precisa de um modelo treinado em modelos/. Sem modelo, roda com um modelo NÃO treinado só para
validar a mecânica (os números não significam nada nesse caso — é só pra ver que o diagnóstico roda).
"""

import glob
import os
from pathlib import Path

import cv2
import numpy as np
import torch as th
from stable_baselines3 import PPO

from src.environment.fnaf_env import FNAFEnv, ALTURA, LARGURA
from src.agent.multimodal_policy import MultimodalExtractor

DEBUG = Path("debug")

# Frames reais e VARIADOS (imagens distintas do jogo) — só os que existirem são usados.
FRAMES = ["escritorio", "luz_esq_bonnie", "luz_dir_chica", "luz_esq_vazia", "power_10", "no_power"]

# Vetores de estado plausíveis e variados (14 estados, em [0,1]) — espelha _capturar_observacao:
# [porta_esq, porta_dir, luz_esq, luz_dir, camera, camera_ativa, energia, tempo,
#  ameaca_esq, ameaca_dir, noite/7, tempo_sem_camera/28s, idade_info_esq/30s, idade_info_dir/30s]
ESTADOS = [
    np.array([0, 0, 0, 0, 0, 0.0, 1.00, 0.00, 0, 0, 0.14, 0.0, 0.0, 0.0], dtype=np.float32),  # início calmo, cheio (N1)
    np.array([0, 0, 1, 0, 0, 0.0, 0.80, 0.20, 1, 0, 0.29, 0.3, 0.0, 0.6], dtype=np.float32),  # ameaça esq, porta aberta (N2)
    np.array([1, 0, 1, 0, 0, 0.0, 0.70, 0.30, 1, 0, 0.29, 0.5, 0.1, 0.8], dtype=np.float32),  # ameaça esq, porta fechada (N2)
    np.array([0, 1, 0, 1, 0, 0.0, 0.50, 0.60, 0, 1, 0.43, 0.8, 0.9, 0.0], dtype=np.float32),  # ameaça dir, Foxy negligenciado (N3)
    np.array([0, 0, 0, 0, 1, 0.5, 0.20, 0.85, 0, 0, 0.71, 0.0, 1.0, 1.0], dtype=np.float32),  # tarde, pouca energia, câmera (N5)
]


def _imagem(nome):
    p = DEBUG / f"quadro_{nome}.png"
    if not p.exists():
        return None
    g = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (LARGURA, ALTURA))            # mesmo formato de _capturar_observacao
    return np.expand_dims(g, axis=-1).astype(np.uint8)


def _carregar_modelo():
    zips = glob.glob("modelos/*.zip")
    if zips:
        caminho = max(zips, key=os.path.getctime)
        print(f"modelo treinado: {caminho}\n")
        return PPO.load(caminho, device="cpu"), True
    print("AVISO: nenhum modelo em modelos/ — usando modelo NÃO TREINADO (só valida a mecânica).\n")
    env = FNAFEnv()
    modelo = PPO("MultiInputPolicy", env,
                 policy_kwargs=dict(features_extractor_class=MultimodalExtractor),
                 device="cpu")
    env.close()
    return modelo, False


def _probs(modelo, imagem, estados):
    """Vetor de probabilidade de ação da política para uma observação."""
    obs = {"imagem": imagem, "estados": estados}
    obs_t, _ = modelo.policy.obs_to_tensor(obs)
    with th.no_grad():
        dist = modelo.policy.get_distribution(obs_t)
    return dist.distribution.probs.cpu().numpy().reshape(-1)


def _tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())       # distância de variação total ∈ [0,1]


def main() -> int:
    modelo, treinado = _carregar_modelo()

    imagens = [(n, _imagem(n)) for n in FRAMES]
    imagens = [(n, im) for n, im in imagens if im is not None]
    if len(imagens) < 2:
        print("Precisa de ao menos 2 frames em debug/ (quadro_*.png).")
        return 1

    # probs[i][s] = probabilidade de ação para (imagem i, estado s)
    probs = [[_probs(modelo, im, st) for st in ESTADOS] for _, im in imagens]

    # Sensibilidade à IMAGEM: estados fixos, varia a imagem
    difs_img = [_tv(probs[i][s], probs[j][s])
                for s in range(len(ESTADOS))
                for i in range(len(imagens)) for j in range(i + 1, len(imagens))]
    # Sensibilidade aos ESTADOS: imagem fixa, varia o estado
    difs_est = [_tv(probs[i][s], probs[i][t])
                for i in range(len(imagens))
                for s in range(len(ESTADOS)) for t in range(s + 1, len(ESTADOS))]

    sens_img = float(np.mean(difs_img))
    sens_est = float(np.mean(difs_est))

    print(f"frames usados: {[n for n, _ in imagens]}")
    print(f"sensibilidade à IMAGEM  (varia imagem, estados fixos): {sens_img:.4f}")
    print(f"sensibilidade aos ESTADOS (varia estados, imagem fixa): {sens_est:.4f}")
    razao = sens_img / (sens_est + 1e-9)
    print(f"razão imagem/estados: {razao:.3f}\n")

    if not treinado:
        print("Veredito: (modelo não treinado — números sem significado; mecânica OK)")
        return 0
    if sens_img < 0.02:
        print("Veredito: a CNN parece INERTE — a política quase não reage à imagem (decide pelos estados).")
    elif razao < 0.20:
        print("Veredito: a imagem influencia POUCO comparado aos estados — investigar a CNN.")
    else:
        print("Veredito: a imagem CONTRIBUI para a decisão (design híbrido valendo).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
