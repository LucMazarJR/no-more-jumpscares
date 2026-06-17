"""Validação offline da detecção de ameaça (Decisão 4A).

Roda o matchTemplate do ambiente (_match_ameaca) contra frames rotulados em
debug/, sem subir o jogo. Confirma que Bonnie/Chica passam do limiar e que
vazio/escuro/lado-trocado ficam abaixo.

Rodar: python -m src.utils.testar_deteccao_ameaca
"""

from pathlib import Path

import cv2

from src.environment.fnaf_env import FNAFEnv, LIMIAR_AMEACA

DEBUG = Path("debug")
REFS = Path("src/utils/referencias")

# (rótulo do frame, lado, esperado_presente)
CASOS = [
    ("luz_esq_bonnie",  "esquerdo", True),
    ("luz_esq_bonnie2", "esquerdo", True),
    ("luz_esq_vazia",   "esquerdo", False),
    ("luz_esq_vazia2",  "esquerdo", False),
    ("luz_esq_apagada", "esquerdo", False),
    ("escritorio",      "esquerdo", False),
    ("luz_dir_chica",   "esquerdo", False),  # Chica não pode casar no template do Bonnie
    ("luz_dir_chica",   "direito",  True),
    ("luz_dir_vazia",   "direito",  False),
    ("luz_dir_vazia2",  "direito",  False),
    ("luz_dir_apagada", "direito",  False),
    ("escritorio",      "direito",  False),
    ("luz_esq_bonnie",  "direito",  False),  # Bonnie não pode casar no template da Chica
]


def _env():
    env = FNAFEnv.__new__(FNAFEnv)  # sem __init__ (sem GameCapture)
    env.template_ameaca_esq = cv2.imread(str(REFS / "ameaca_esquerda.png"), cv2.IMREAD_GRAYSCALE)
    env.template_ameaca_dir = cv2.imread(str(REFS / "ameaca_direita.png"), cv2.IMREAD_GRAYSCALE)
    if env.template_ameaca_esq is None or env.template_ameaca_dir is None:
        raise SystemExit("templates de ameaça ausentes em referencias/")
    return env


def _cinza(nome):
    p = DEBUG / f"quadro_{nome}.png"
    if not p.exists():
        return None
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY)


def main() -> int:
    env = _env()
    print(f"limiar = {LIMIAR_AMEACA}\n")
    ok = faltando = 0
    for nome, lado, presente in CASOS:
        g = _cinza(nome)
        if g is None:
            print(f"  [--]   {nome:16s} {lado:8s} (frame ausente em debug/)")
            faltando += 1
            continue
        score = env._match_ameaca(g, lado)
        certo = (score > LIMIAR_AMEACA) == presente
        ok += certo
        print(f"  [{'OK' if certo else 'FAIL':4s}] {nome:16s} {lado:8s} "
              f"score={score:.3f} esperado={'presente' if presente else 'vazio'}")
    total = len(CASOS) - faltando
    extra = f"  ({faltando} frames ausentes)" if faltando else ""
    print(f"\n{ok}/{total} corretos{extra}")
    return 0 if total > 0 and ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
