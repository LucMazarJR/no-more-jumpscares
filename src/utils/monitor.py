"""Monitor ao vivo da percepção — captura o jogo e mostra o estado no terminal.

Testa a detecção de ameaça (4A) — e, quando calibrada, a leitura de energia (4B)
— contra o jogo real, sem treinar. O match alto só aparece com a luz daquele lado
acesa (sem luz o vão é escuro).

Dica: deixe o jogo num monitor e este terminal no outro, pra ver os dois.

Rodar: python -m src.utils.monitor   (Ctrl+C para sair)
"""

import time
from pathlib import Path

import cv2

from src.environment.fnaf_env import FNAFEnv, LIMIAR_AMEACA, validar_leitura_energia
from src.utils.capture import GameCapture, regiao_cliente
from src.utils.inspecionar import _melhor_janela

REFS = Path("src/utils/referencias")
REF_W, REF_H = 1280, 720
INTERVALO = 0.3  # segundos entre leituras


def _detector() -> FNAFEnv:
    env = FNAFEnv.__new__(FNAFEnv)  # sem __init__ (sem subir captura/jogo)
    env.template_ameaca_esq = cv2.imread(str(REFS / "ameaca_esquerda.png"), cv2.IMREAD_GRAYSCALE)
    env.template_ameaca_dir = cv2.imread(str(REFS / "ameaca_direita.png"), cv2.IMREAD_GRAYSCALE)
    if env.template_ameaca_esq is None or env.template_ameaca_dir is None:
        raise SystemExit("templates de ameaça ausentes em referencias/")
    env.glifos_energia = {}
    for d in "0123456789":
        g = cv2.imread(str(REFS / "digitos" / f"{d}.png"), cv2.IMREAD_GRAYSCALE)
        if g is not None:
            env.glifos_energia[d] = g
    return env


def _capturar_cinza(cap: GameCapture):
    win = _melhor_janela()
    try:
        if not win.isActive:
            win.activate()  # sem foco o jogo renderiza preto
            time.sleep(0.1)
    except Exception:
        pass
    frame = cap.capturar_tela(regiao_cliente(win))
    return cv2.cvtColor(cv2.resize(frame, (REF_W, REF_H)), cv2.COLOR_BGR2GRAY)


def main():
    env = _detector()
    cap = GameCapture()
    print("Monitor de percepção — Ctrl+C para sair")
    print("(deixe o jogo num monitor e este terminal no outro)\n")
    filtrada = None
    try:
        while True:
            try:
                g = _capturar_cinza(cap)
            except RuntimeError as erro:
                print(f"\r{erro}   ", end="", flush=True)
                time.sleep(1.0)
                continue
            esq = env._match_ameaca(g, "esquerdo")
            direita = env._match_ameaca(g, "direito")
            lido = env._ler_energia(g)
            # filtro photo-primary: segura em None, rejeita subida, aceita queda
            if filtrada is None:
                filtrada = float(lido) if lido is not None else None
            else:
                filtrada = validar_leitura_energia(lido, filtrada)
            fe = "AMEAÇA" if esq > LIMIAR_AMEACA else " ---- "
            fd = "AMEAÇA" if direita > LIMIAR_AMEACA else " ---- "
            txt_lido = f"{lido:3d}%" if lido is not None else " -- "
            txt_filt = f"{filtrada:4.1f}%" if filtrada is not None else " -- "
            print(f"\r  ESQ {esq:4.2f} [{fe}]   DIR {direita:4.2f} [{fd}]   "
                  f"energia lida={txt_lido} filtrada={txt_filt}   ", end="", flush=True)
            time.sleep(INTERVALO)
    except KeyboardInterrupt:
        print("\nencerrado.")


if __name__ == "__main__":
    main()
