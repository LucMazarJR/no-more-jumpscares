import mss
import numpy as np
import cv2
import pyautogui
import time

# Desativa a proteção de failsafe do pyautogui
# (por padrão ele trava se o mouse for pro canto da tela)
pyautogui.FAILSAFE = False

class GameCapture:
    def __init__(self):
        self.sct = mss.mss()


    def focar_janela(self, titulo: str = "FiveNightsatFreddys (1)"):
        import pygetwindow as gw
    
        janelas = gw.getWindowsWithTitle(titulo)
        if not janelas:
            print(f"Janela '{titulo}' não encontrada!")
            return False
    
        janelas[0].activate()
        time.sleep(0.3)
        return True

    def capturar_tela(self, regiao: dict = None) -> np.ndarray:
        """
        Captura a tela inteira ou uma região específica.
        regiao = {"top": 0, "left": 0, "width": 1920, "height": 1080}
        """
        if regiao is None:
            monitor = self.sct.monitors[1]  # monitor principal
        else:
            monitor = regiao

        frame = self.sct.grab(monitor)

        # Converte para numpy array (formato que o OpenCV entende)
        img = np.array(frame)

        # mss captura em BGRA, converte para BGR (padrão do OpenCV)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        return img

    def redimensionar(self, img: np.ndarray, largura: int, altura: int) -> np.ndarray:
        """
        Reduz a imagem para economizar memória e acelerar o treino.
        A IA não precisa de 1080p para aprender.
        """
        return cv2.resize(img, (largura, altura))

    def para_escala_cinza(self, img: np.ndarray) -> np.ndarray:
        """
        Converte para escala de cinza.
        Reduz 3 canais (RGB) para 1 — 3x menos dados pra processar.
        """
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def clicar(self, x: int, y: int):
        """Clica numa posição da tela."""
        pyautogui.click(x, y)

    def segurar_botao(self, x: int, y: int):
        """Pressiona e segura o botão esquerdo do mouse em (x, y)."""
        pyautogui.mouseDown(x=x, y=y)

    def soltar_botao(self, x: int, y: int):
        """Solta o botão esquerdo do mouse."""
        pyautogui.mouseUp(x=x, y=y)

    def mover_mouse(self, x: int, y: int):
        """Move o mouse sem clicar (instantâneo)."""
        pyautogui.moveTo(x, y)

    def arrastar_para(self, x: int, y: int, duration: float = 0.15):
        """Move o mouse suavemente até (x, y) sem pressionar botão."""
        pyautogui.moveTo(x, y, duration=duration, tween=pyautogui.easeInOutQuad)

    def arrastar_clicando(self, x_ini: int, y_ini: int, x_fim: int, y_fim: int, duration: float = 0.15):
        """Drag real: pressiona botão esquerdo no início, move suavemente, solta no fim."""
        pyautogui.mouseDown(x=x_ini, y=y_ini)
        pyautogui.moveTo(x_fim, y_fim, duration=duration, tween=pyautogui.easeInOutQuad)
        pyautogui.mouseUp()

    def pressionar_tecla(self, tecla: str):
        """Pressiona uma tecla do teclado."""
        pyautogui.press(tecla)

    def atalho(self, *teclas: str):
        """Pressiona um atalho combinando varias teclas (ex.: alt+enter)."""
        pyautogui.hotkey(*teclas)


PALAVRAS_FNAF = ("freddy", "fnaf", "five nights")


def melhor_janela(titulo: str, listar: bool = False):
    """Escolhe a janela real do jogo. Filtra janelas-fantasma (1x1) e, mesmo se o
    título vier vazio (sem .env), prefere as que parecem ser o FNAF; entre essas,
    pega a de maior área. Seleção ÚNICA usada pelo ambiente e pelas ferramentas de
    captura — garante que a IA e as referências saiam da MESMA janela."""
    import pygetwindow as gw
    janelas = gw.getWindowsWithTitle(titulo)
    if not janelas:
        raise RuntimeError(f"janela '{titulo}' nao encontrada — o jogo esta aberto?")
    if listar:
        print("janelas candidatas:")
        for w in janelas:
            print(f"  '{w.title}' left={w.left} top={w.top} {w.width}x{w.height}")
    reais = [w for w in janelas if w.width > 100 and w.height > 100]
    fnaf = [w for w in reais if any(p in w.title.lower() for p in PALAVRAS_FNAF)]
    return max(fnaf or reais or janelas, key=lambda w: w.width * w.height)


def regiao_cliente(win) -> dict:
    """Região da ÁREA CLIENTE da janela (sem barra de título nem bordas), em
    coordenadas de tela. Cai para o retângulo cheio se o win32 falhar."""
    try:
        import ctypes
        from ctypes import wintypes
        hwnd = win._hWnd
        rect = wintypes.RECT()
        ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect))
        ponto = wintypes.POINT(0, 0)
        ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(ponto))
        if rect.right > 0 and rect.bottom > 0:
            return {"left": ponto.x, "top": ponto.y, "width": rect.right, "height": rect.bottom}
    except Exception:
        pass
    return {"left": win.left, "top": win.top, "width": win.width, "height": win.height}


# ─── Teste rápido ────────────────────────────────────────────────
if __name__ == "__main__":
    print("Iniciando teste de captura...")
    cap = GameCapture()

    # Captura a tela
    frame = cap.capturar_tela()
    print(f"Tela capturada! Tamanho: {frame.shape}")
    # frame.shape retorna (altura, largura, canais)
    # ex: (1080, 1920, 3)

    # Redimensiona para 84x84 (tamanho clássico de RL com imagem)
    frame_pequeno = cap.redimensionar(frame, 84, 84)
    print(f"Redimensionado para: {frame_pequeno.shape}")

    # Converte para cinza
    frame_cinza = cap.para_escala_cinza(frame_pequeno)
    print(f"Escala de cinza: {frame_cinza.shape}")

    # Salva uma screenshot de teste
    cv2.imwrite("teste_captura.png", frame)
    print("Screenshot salva como teste_captura.png!")