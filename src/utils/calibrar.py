import cv2
import time
from src.utils.capture import GameCapture

"""
Script de calibragem — roda uma vez para capturar as imagens de referência.
Execute cada função na hora certa:
  - capturar_morte()   → rode quando a tela de game over aparecer
  - capturar_vitoria() → rode quando o "6 AM" aparecer
  - capturar_coords()  → mostra as coordenadas do mouse em tempo real
"""

cap = GameCapture()

def capturar_morte():
    """
    Abra o FNAF1, morra de propósito, e rode essa função
    ENQUANTO a tela de game over estiver aparecendo.
    """
    print("Você tem 5 segundos para deixar a tela de game over aparecer...")
    time.sleep(5)

    frame = cap.capturar_tela()
    cv2.imwrite("src/utils/referencias/morte.png", frame)
    print("Imagem de morte salva!")

def capturar_vitoria():
    """
    Rode quando o '6 AM' aparecer na tela.
    """
    print("Você tem 5 segundos para deixar o 6 AM aparecer...")
    time.sleep(5)

    frame = cap.capturar_tela()
    cv2.imwrite("src/utils/referencias/vitoria.png", frame)
    print("Imagem de vitória salva!")

def _desenhar_grade(img):
    """Desenha uma grade de 100 em 100px com coordenadas rotuladas (p/ marcar
    a região das letras 'New Game' pelo número). Igual à do inspecionar."""
    out = img.copy()
    h, w = out.shape[:2]
    for x in range(0, w, 100):
        cv2.line(out, (x, 0), (x, h), (0, 255, 0), 1)
        cv2.putText(out, str(x), (x + 2, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    for y in range(0, h, 100):
        cv2.line(out, (0, y), (w, y), (0, 255, 0), 1)
        cv2.putText(out, str(y), (2, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return out

def capturar_menu():
    """
    Referência do MENU PRINCIPAL (Bonnie/crash volta pro menu SEM tela de morte).
    Deixe o menu principal aparecer com o texto 'New Game' visível e rode.
    Salva o frame inteiro em referencias/menu.png (o recorte das letras é feito
    em _carregar_templates) e uma cópia COM GRADE em debug/menu_grade.png, para
    marcar a região das letras pelas coordenadas da grade.
    """
    import os
    from src.utils.capture import melhor_janela, regiao_cliente
    from src.environment.fnaf_env import WINDOW_TITLE

    print("Você tem 5 segundos para deixar o MENU PRINCIPAL ('New Game') aparecer...")
    time.sleep(5)

    # Captura a ÁREA-CLIENTE da janela e redimensiona p/ 1280x720 — MESMO enquadramento
    # que _capturar_janela() usa na detecção. Garante que o template bata, esteja o jogo
    # em fullscreen ou em janela (capturar_tela() do monitor inteiro só bateria em fullscreen).
    win = melhor_janela(WINDOW_TITLE)
    frame = cv2.resize(cap.capturar_tela(regiao_cliente(win)), (1280, 720))
    cv2.imwrite("src/utils/referencias/menu.png", frame)
    print(f"Imagem de menu salva! ({frame.shape[1]}x{frame.shape[0]})")

    os.makedirs("debug", exist_ok=True)
    cv2.imwrite("debug/menu_grade.png", _desenhar_grade(frame))
    print("Cópia com grade salva: debug/menu_grade.png (use p/ marcar as letras 'New Game')")

def capturar_camera_aberta():
    """
    Com a câmera ABERTA no jogo (qualquer tab), clique sobre o indicador 'YOU'
    no mapa de câmeras. Salva o recorte em referencias/camera_aberta.png.
    """
    import ctypes
    import sys
    import pyautogui

    VK_LBUTTON = 0x01

    print("Abra a câmera no FNAF1 (qualquer tab serve).")
    print("Localize o quadrado 'YOU' no mapa de câmeras (indica sua posição no mapa).")
    print("Clique sobre ele para capturar o template.")
    print("Pressione Ctrl+C para cancelar.\n")

    MARG_X, MARG_Y = 40, 30  # recorte: 80×60 px ao redor do clique

    if sys.platform == "win32":
        user32 = ctypes.windll.user32

        def _pressionado():
            return bool(user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000)

        while _pressionado():
            time.sleep(0.01)

        ultimo = (-1, -1)
        try:
            while True:
                x, y = pyautogui.position()
                if (x, y) != ultimo:
                    print(f"\r  x={x:4d}, y={y:4d}   ", end="", flush=True)
                    ultimo = (x, y)

                if _pressionado():
                    cx, cy = pyautogui.position()
                    while _pressionado():
                        time.sleep(0.01)

                    frame = cap.capturar_tela()
                    h, w = frame.shape[:2]
                    x1 = max(0, cx - MARG_X)
                    y1 = max(0, cy - MARG_Y)
                    x2 = min(w, cx + MARG_X)
                    y2 = min(h, cy + MARG_Y)

                    recorte = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                    caminho = "src/utils/referencias/camera_aberta.png"
                    cv2.imwrite(caminho, recorte)
                    print(f"\nTemplate salvo: {caminho}")
                    print(f"Tamanho: {recorte.shape[1]}x{recorte.shape[0]}px | Centro capturado: ({cx}, {cy})")
                    return

                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\nCancelado.")
    else:
        input("Posicione o mouse sobre o 'YOU' e pressione Enter: ")
        cx, cy = pyautogui.position()
        frame = cap.capturar_tela()
        h, w = frame.shape[:2]
        x1 = max(0, cx - MARG_X)
        y1 = max(0, cy - MARG_Y)
        x2 = min(w, cx + MARG_X)
        y2 = min(h, cy + MARG_Y)
        recorte = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        cv2.imwrite("src/utils/referencias/camera_aberta.png", recorte)
        print("Template salvo em src/utils/referencias/camera_aberta.png!")

def capturar_sombra(estado: str):
    """
    Referência do vão da porta esquerda (Decisão 4) — recorte committado em referencias/sombra/.
    Com a LUZ ESQUERDA acesa, rode com:
      - 'presente': Bonnie ainda no vão (sombra — std baixo)
      - 'vazio':    Bonnie saiu, corredor iluminado (std alto)
    Captura a janela igual ao ambiente (redimensiona p/ 1280x720) e recorta SOMBRA_REGIAO.
    Os dois juntos calibram LIMIAR_VAZIO (Bonnie ~9.2 < vazio ~11.65 → confirma corredor vazio).
    """
    from pathlib import Path
    from src.utils.capture import melhor_janela, regiao_cliente
    from src.environment.fnaf_env import SOMBRA_REGIAO, WINDOW_TITLE

    if estado not in ("presente", "vazio"):
        print("uso: python -m src.utils.calibrar sombra [presente | vazio]")
        return

    print(f"5 segundos para deixar a luz esquerda acesa, porta fechada, estado '{estado}'...")
    time.sleep(5)

    win = melhor_janela(WINDOW_TITLE)
    frame = cv2.resize(cap.capturar_tela(regiao_cliente(win)), (1280, 720))
    left, top, w, h = SOMBRA_REGIAO
    recorte = cv2.cvtColor(frame[top:top + h, left:left + w], cv2.COLOR_BGR2GRAY)  # cinza, como as outras refs

    pasta = Path("src/utils/referencias/sombra")
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / f"{estado}.png"
    cv2.imwrite(str(caminho), recorte)
    print(f"Referência salva: {caminho}  ({recorte.shape[1]}x{recorte.shape[0]})  std={recorte.std():.2f}")

def capturar_coords():
    import pyautogui
    print("Movendo o mouse sobre os botões do jogo...")
    print("Pressione Ctrl+C para parar.\n")

    ultimo_x, ultimo_y = 0, 0
    try:
        while True:
            x, y = pyautogui.position()
            if x != ultimo_x or ultimo_y != y:
                print(f"x={x}, y={y}")
                ultimo_x, ultimo_y = x, y
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nPronto!")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        capturar_coords()
    elif sys.argv[1] == "morte":
        capturar_morte()
    elif sys.argv[1] == "vitoria":
        capturar_vitoria()
    elif sys.argv[1] == "menu":
        capturar_menu()
    elif sys.argv[1] == "camera_aberta":
        capturar_camera_aberta()
    elif sys.argv[1] == "sombra":
        capturar_sombra(sys.argv[2] if len(sys.argv) > 2 else "")
    else:
        print(f"Argumento desconhecido: {sys.argv[1]}")
        print("Uso: python -m src.utils.calibrar [morte | vitoria | menu | camera_aberta | sombra <presente|vazio>]")