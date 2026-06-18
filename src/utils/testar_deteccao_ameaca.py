"""Validação offline da detecção de ameaça (Decisão 4A), sem subir o jogo.

Duas detecções, dois lados:
- ROSTO (matchTemplate): porta aberta à esquerda (Bonnie) e Chica à direita.
  Confere contra frames inteiros rotulados em debug/ (transiente).
- SOMBRA (std do vão): porta fechada à esquerda — o corpo do Bonnie bloqueia a luz e
  deixa o vão mais liso (std baixo). Confere contra os recortes do vão committados em
  referencias/sombra/ (presente/vazio), análogos a referencias/digitos/.

Rodar: python -m src.utils.testar_deteccao_ameaca
"""

from pathlib import Path

import cv2

from src.environment.fnaf_env import FNAFEnv, LIMIAR_AMEACA, LIMIAR_VAZIO

DEBUG = Path("debug")
REFS = Path("src/utils/referencias")
SOMBRA_REF = REFS / "sombra"  # recortes do vão (presente/vazio), committados como digitos/

# Detecção por ROSTO (matchTemplate): porta aberta à esquerda (Bonnie) e Chica à direita.
# Frames inteiros rotulados em debug/ (transiente, como o resto deste teste).
# (rótulo do frame, lado, esperado_presente)
CASOS_ROSTO = [
    ("luz_esq_bonnie",  "esquerdo", True),
    ("luz_esq_bonnie2", "esquerdo", True),
    ("luz_esq_vazia",   "esquerdo", False),
    ("luz_esq_vazia2",  "esquerdo", False),
    ("luz_esq_apagada", "esquerdo", False),
    ("escritorio",      "esquerdo", False),
    ("luz_dir_chica",   "esquerdo", False),  # Chica não casa no template do Bonnie
    ("luz_dir_chica",   "direito",  True),
    ("luz_dir_vazia",   "direito",  False),
    ("luz_dir_vazia2",  "direito",  False),
    ("luz_dir_apagada", "direito",  False),
    ("escritorio",      "direito",  False),
    ("luz_esq_bonnie",  "direito",  False),  # Bonnie não casa no template da Chica
]

# Confirmação de CORREDOR VAZIO pela textura (std do vão). std ALTO = corredor vazio
# iluminado confirmado → Bonnie saiu. std BAIXO = Bonnie OU escuro (não confirma vazio).
# Valida contra os RECORTES do vão committados em referencias/sombra/ — presente (Bonnie,
# std ~9.2) vs vazio (corredor, std ~11.65). O recorte já É a região SOMBRA_REGIAO.
# (nome do recorte em referencias/sombra/, vazio_confirmado_esperado)
CASOS_SOMBRA = [
    ("presente", False),  # Bonnie no vão → NÃO confirma vazio
    ("vazio",    True),   # corredor iluminado → vazio confirmado
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


def _rodar_rosto(env):
    ok = faltando = 0
    for nome, lado, presente in CASOS_ROSTO:
        g = _cinza(nome)
        if g is None:
            print(f"  [--]   {nome:20s} {lado:8s} (frame ausente em debug/)")
            faltando += 1
            continue
        score = env._match_ameaca(g, lado)
        certo = (score > LIMIAR_AMEACA) == presente
        ok += certo
        print(f"  [{'OK' if certo else 'FAIL':4s}] {nome:20s} {lado:8s} rosto  "
              f"score={score:.3f} esperado={'presente' if presente else 'vazio'}")
    return ok, len(CASOS_ROSTO) - faltando


def _rodar_sombra():
    # O recorte em referencias/sombra/ já É a região SOMBRA_REGIAO → std direto sobre ele
    # equivale a _sombra_no_vao(frame_inteiro). std ALTO = corredor vazio confirmado.
    ok = faltando = 0
    for nome, vazio in CASOS_SOMBRA:
        p = SOMBRA_REF / f"{nome}.png"
        if not p.exists():
            print(f"  [--]   {nome:20s} (recorte ausente em referencias/sombra/)")
            faltando += 1
            continue
        std = float(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY).std())
        certo = (std > LIMIAR_VAZIO) == vazio
        ok += certo
        print(f"  [{'OK' if certo else 'FAIL':4s}] {nome:20s} {'esquerdo':8s} std={std:.3f} "
              f"esperado={'corredor vazio' if vazio else 'Bonnie/escuro'}")
    return ok, len(CASOS_SOMBRA) - faltando


def main() -> int:
    env = _env()
    print(f"rosto: Bonnie/Chica presente se score > {LIMIAR_AMEACA}   "
          f"vazio: corredor vazio confirmado se std > {LIMIAR_VAZIO}")
    print("(positivos de rosto marcam 1.000 = self-match: o template foi recortado do\n"
          " próprio frame; capture um frame de Bonnie INDEPENDENTE p/ uma margem honesta)\n")

    print("Detecção por rosto (seta ameaça — porta aberta / Chica):")
    ok_rosto, total_rosto = _rodar_rosto(env)

    print("\nConfirmação de corredor vazio (limpa ameaça — std do vão):")
    ok_sombra, total_sombra = _rodar_sombra()

    ok = ok_rosto + ok_sombra
    total = total_rosto + total_sombra
    print(f"\nRosto: {ok_rosto}/{total_rosto}   Sombra: {ok_sombra}/{total_sombra}   "
          f"Total: {ok}/{total}")
    return 0 if total > 0 and ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
