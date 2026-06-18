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

from src.environment.fnaf_env import (FNAFEnv, LIMIAR_AMEACA, LIMIAR_VAZIO, DEBOUNCE_VAZIO,
                                      DEBOUNCE_PRESENCA, validar_leitura_energia)
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


def _capturar(cap: GameCapture):
    """Retorna (colorido, cinza) já no enquadramento de referência (1280x720). O colorido
    é p/ a cor do botão DOOR (porta); o cinza p/ os templates/std/luz."""
    win = _melhor_janela()
    try:
        if not win.isActive:
            win.activate()  # sem foco o jogo renderiza preto
            time.sleep(0.1)
    except Exception:
        pass
    cor = cv2.resize(cap.capturar_tela(regiao_cliente(win)), (REF_W, REF_H))
    return cor, cv2.cvtColor(cor, cv2.COLOR_BGR2GRAY)


def main():
    env = _detector()
    cap = GameCapture()
    print("Monitor de percepção — Ctrl+C para sair")
    print("(deixe o jogo num monitor e este terminal no outro)")
    print(f"Bonnie é HELD: rosto SETA, corredor vazio (s>{LIMIAR_VAZIO:.0f}) LIMPA, "
          f"sombra/escuro MANTÉM.")
    print("legenda: r=rosto(>%.2f)  s=std do vão  c=chica  porta=botão DOOR\n" % LIMIAR_AMEACA)
    filtrada = None
    ameaca_e = ameaca_d = False               # estados HELD (memória)
    n_vazio = n_rosto = n_chica_off = 0        # contadores de debounce (anti-estática)
    try:
        while True:
            try:
                cor, g = _capturar(cap)
            except RuntimeError as erro:
                print(f"\r{erro}   ", end="", flush=True)
                time.sleep(1.0)
                continue
            rosto = env._match_ameaca(g, "esquerdo")    # rosto do Bonnie (seta a ameaça)
            std = env._sombra_no_vao(g, "esquerdo")     # std do vão (alto = corredor vazio)
            chica = env._match_ameaca(g, "direito")     # Chica na janela
            porta_e = env._porta_fechada_visual(cor, "esquerdo")  # True=fechada/False=aberta/None
            lido = env._ler_energia(g)
            if filtrada is None:
                filtrada = float(lido) if lido is not None else None
            else:
                filtrada = validar_leitura_energia(lido, filtrada)
            # Bonnie HELD (mesma lógica do env): debounce ACUMULADO — 'mantém' (estática) não
            # zera os contadores, só o sinal real oposto zera.
            if std > LIMIAR_VAZIO:                       # corredor vazio confirmado → limpa
                n_rosto, n_vazio = 0, n_vazio + 1
                if n_vazio >= DEBOUNCE_VAZIO:
                    ameaca_e, n_vazio = False, 0
            elif rosto > LIMIAR_AMEACA:                  # rosto confirmado → seta
                n_vazio, n_rosto = 0, n_rosto + 1
                if n_rosto >= DEBOUNCE_PRESENCA:
                    ameaca_e, n_rosto = True, 0
            # else: sombra/escuro → mantém estado e contadores
            # Chica HELD: rosto seta; limpa após DEBOUNCE_VAZIO frames sem rosto
            if chica > LIMIAR_AMEACA:
                n_chica_off, ameaca_d = 0, True
            else:
                n_chica_off += 1
                if n_chica_off >= DEBOUNCE_VAZIO:
                    ameaca_d = False
            fb = "AMEAÇA" if ameaca_e else " ---- "
            fd = "AMEAÇA" if ameaca_d else " ---- "
            pe = "fech" if porta_e is True else ("abr " if porta_e is False else " ?  ")
            # std/rosto oscilam com a estática (normal); o estado HELD B[] fica estável.
            txt_filt = f"{filtrada:4.1f}%" if filtrada is not None else " --  "
            print(f"\rESQ[{pe}] B[{fb}] (rosto{rosto:.2f} vao{std:4.1f})  "
                  f"Chica[{fd}] c{chica:.2f}  en{txt_filt}   ", end="", flush=True)
            time.sleep(INTERVALO)
    except KeyboardInterrupt:
        print("\nencerrado.")


if __name__ == "__main__":
    main()
