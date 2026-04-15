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


    def focar_janela(self, titulo: str = "Five Nights at Freddy's"):
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
            monitor = self.sct.monitors[2]  # monitor principal
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

    def mover_mouse(self, x: int, y: int):
        """Move o mouse sem clicar."""
        pyautogui.moveTo(x, y)

    def pressionar_tecla(self, tecla: str):
        """Pressiona uma tecla do teclado."""
        pyautogui.press(tecla)


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