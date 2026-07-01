"""Inspeciona um vecnormalize.pkl SEM subir env nenhum — evidência do achado D1:
com clip_reward no default (10.0), a vitória (+500) normalizada é clipada no mesmo
teto que qualquer recompensa grande, comprimindo a razão vitória:morte (5:1 → até 1:1).

Uso (na raiz do projeto):
    venv\\Scripts\\python scripts\\inspecionar_vecnormalize.py [modelos/vecnormalize.pkl ...]

Sem argumentos, inspeciona modelos/vecnormalize.pkl. O VecNormalize.save é um pickle do
objeto (o venv interno é excluído no __getstate__), então carrega direto, sem ambiente.
"""
import math
import pickle
import sys
from pathlib import Path

RECOMPENSA_VITORIA = 500.0
RECOMPENSA_MORTE = -100.0
EPS = 1e-8


def inspecionar(caminho: str) -> None:
    p = Path(caminho)
    if not p.exists():
        print(f"[SKIP] {caminho}: arquivo nao existe")
        return

    with open(p, "rb") as f:
        vn = pickle.load(f)

    ret_var = float(getattr(vn.ret_rms, "var", float("nan")))
    ret_std = math.sqrt(ret_var + EPS)
    clip = float(vn.clip_reward)
    vit_norm = RECOMPENSA_VITORIA / ret_std
    morte_norm = RECOMPENSA_MORTE / ret_std
    vit_clipada = max(-clip, min(clip, vit_norm))
    morte_clipada = max(-clip, min(clip, morte_norm))

    print(f"\n=== {caminho} ===")
    print(f"gamma:           {vn.gamma}")
    print(f"clip_reward:     {clip}")
    print(f"ret_rms.mean:    {float(vn.ret_rms.mean):.4f}")
    print(f"ret_rms.var:     {ret_var:.4f}  (std = {ret_std:.4f})")
    print(f"vitoria (+500):  normalizada {vit_norm:+8.2f} -> apos clip {vit_clipada:+8.2f}"
          f"{'  [CLIPADA]' if abs(vit_norm) > clip else ''}")
    print(f"morte   (-100):  normalizada {morte_norm:+8.2f} -> apos clip {morte_clipada:+8.2f}"
          f"{'  [CLIPADA]' if abs(morte_norm) > clip else ''}")
    if morte_clipada != 0:
        print(f"razao vitoria:morte no gradiente: {abs(vit_clipada / morte_clipada):.2f}:1 "
              f"(design: {abs(RECOMPENSA_VITORIA / RECOMPENSA_MORTE):.0f}:1)")


if __name__ == "__main__":
    caminhos = sys.argv[1:] or ["modelos/vecnormalize.pkl"]
    for c in caminhos:
        inspecionar(c)
