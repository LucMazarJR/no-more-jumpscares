"""
Diagnóstico da detecção de menu-crash (Decisão 8).

Monitora o score do template "New Game" em tempo real para validar que:
  - No MENU PRINCIPAL o score fica > 0.60 (limiar de detecção) → "MENU DETECTADO"
  - Em telas de jogo (noite, morte, vitória) o score fica < 0.60 → nenhum falso positivo

Uso:
  python -m src.utils.testar_deteccao_menu

Deixe o jogo no estado desejado e observe a coluna "Menu score".
Ctrl+C para sair.

O que esperar:
  - Menu principal aberto      →  menu ~0.90+,  morte ~0.25,  vitória ~0.25
  - Jogo rodando (noite)       →  menu ~0.10-0.30, sem alarme
  - Tela de Game Over          →  morte sobe, menu permanece baixo
"""
import sys
import os
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.capture import GameCapture, melhor_janela, regiao_cliente

REFS     = Path(__file__).parent / "referencias"
REF_SIZE = (1280, 720)

THRESH_MENU   = 0.60
THRESH_OUTROS = 0.70
DEBOUNCE      = 2   # frames consecutivos p/ confirmar (igual ao ambiente)


def _carregar(nome_cinza: bool, *nomes):
    flag = cv2.IMREAD_GRAYSCALE if nome_cinza else cv2.IMREAD_COLOR
    for n in nomes:
        p = REFS / n
        if p.exists():
            img = cv2.imread(str(p), flag)
            if img is not None:
                return img, n
    return None, None


def carregar_templates():
    morte_img,   morte_nome   = _carregar(True, "morte.png",   "morte.jpg",   "morte.jpeg")
    vitoria_img, vitoria_nome = _carregar(True, "vitoria.png", "vitoria.jpg", "vitoria.jpeg")
    menu_img,    menu_nome    = _carregar(True, "menu.png",    "menu.jpg",    "menu.jpeg")

    if morte_img is None or vitoria_img is None:
        print("ERRO: morte/vitoria não encontrados em src/utils/referencias/")
        sys.exit(1)

    h, w = morte_img.shape
    tmpl_morte = morte_img[int(h * 0.88):, int(w * 0.82):]

    h, w = vitoria_img.shape
    tmpl_vitoria = vitoria_img[int(h * 0.38):int(h * 0.58), int(w * 0.38):int(w * 0.62)]

    tmpl_menu = None
    if menu_img is not None:
        menu_img = cv2.resize(menu_img, REF_SIZE)
        tmpl_menu = menu_img[402:439, 172:379]
        print(f"Template menu ({menu_nome}):    {tmpl_menu.shape[1]}x{tmpl_menu.shape[0]}  [x 172-379, y 402-439]")
    else:
        print("AVISO: menu.png não encontrado — rode: python -m src.utils.calibrar menu")

    print(f"Template morte ({morte_nome}):   {tmpl_morte.shape[1]}x{tmpl_morte.shape[0]}")
    print(f"Template vitória ({vitoria_nome}): {tmpl_vitoria.shape[1]}x{tmpl_vitoria.shape[0]}")
    return tmpl_morte, tmpl_vitoria, tmpl_menu


def capturar_janela(cap: GameCapture) -> np.ndarray:
    """Captura a área-cliente da janela e redimensiona p/ 1280x720 — idêntico ao ambiente."""
    try:
        from src.environment.fnaf_env import WINDOW_TITLE
        win = melhor_janela(WINDOW_TITLE)
        frame = cap.capturar_tela(regiao_cliente(win))
    except Exception:
        frame = cap.capturar_tela()
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(cinza, REF_SIZE)


def score(frame: np.ndarray, tmpl: np.ndarray) -> float:
    res = cv2.matchTemplate(frame, tmpl, cv2.TM_CCOEFF_NORMED)
    return float(res.max())


if __name__ == "__main__":
    cap = GameCapture()
    tmpl_morte, tmpl_vitoria, tmpl_menu = carregar_templates()

    print("\nMonitorando... (Ctrl+C para sair)\n")
    print(f"{'Menu score':>12}  {'Morte score':>12}  {'Vitória score':>14}  {'Status'}")
    print("-" * 68)

    cont_menu = 0
    ultimo_salvo = None

    try:
        while True:
            frame = capturar_janela(cap)

            s_menu   = score(frame, tmpl_menu)   if tmpl_menu   is not None else -1.0
            s_morte  = score(frame, tmpl_morte)
            s_vitoria = score(frame, tmpl_vitoria)

            # Debounce do menu (igual ao ambiente)
            if s_menu > THRESH_MENU:
                cont_menu += 1
            else:
                cont_menu = 0

            if cont_menu >= DEBOUNCE:
                status = "<<< MENU DETECTADO (morte) >>>"
                if ultimo_salvo != "menu":
                    os.makedirs("debug", exist_ok=True)
                    cv2.imwrite("debug/testar_menu_frame.png", frame)
                    print(f"\n  [frame salvo em debug/testar_menu_frame.png]")
                    ultimo_salvo = "menu"
            elif s_morte > THRESH_OUTROS:
                status = "<<< MORTE DETECTADA >>>"
                ultimo_salvo = None
            elif s_vitoria > THRESH_OUTROS:
                status = "<<< VITÓRIA DETECTADA >>>"
                ultimo_salvo = None
            else:
                status = "normal"
                ultimo_salvo = None

            menu_str = f"{s_menu:12.3f}" if s_menu >= 0 else "  (sem tmpl)"
            print(f"{menu_str}  {s_morte:12.3f}  {s_vitoria:14.3f}  {status}", end="\r")
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nEncerrando.")
