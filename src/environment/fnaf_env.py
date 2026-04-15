import gymnasium as gym
import numpy as np
import cv2
import time
from pathlib import Path
from gymnasium import spaces
from src.utils.capture import GameCapture

LARGURA = 84
ALTURA  = 84

ACOES = {
    0:  "nada",
    1:  "porta_esquerda",
    2:  "porta_direita",
    3:  "luz_esquerda",
    4:  "luz_direita",
    5:  "abrir_fechar_camera",
    6:  "camera_1a",
    7:  "camera_1b",
    8:  "camera_1c",
    9:  "camera_2a",
    10: "camera_2b",
    11: "camera_3",
    12: "camera_4a",
    13: "camera_4b",
    14: "camera_5",
    15: "camera_6",
    16: "camera_7",
}

COORDS = {
    "porta_esquerda":      (386,  535),
    "porta_direita":       (1540, 544),
    "luz_esquerda":        (371,  645),
    "luz_direita":         (1527, 653),
    "abrir_fechar_camera": (870,  869),
    "camera_1a":           (1311, 549),
    "camera_1b":           (1268, 603),
    "camera_1c":           (1253, 666),
    "camera_2a":           (1307, 794),
    "camera_2b":           (1298, 828),
    "camera_3":            (1220, 766),
    "camera_4a":           (1411, 796),
    "camera_4b":           (1404, 841),
    "camera_5":            (1169, 632),
    "camera_6":            (1497, 764),
    "camera_7":            (1523, 620),
}


class FNAFEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()

        self.capture          = GameCapture()
        self.render_mode      = render_mode
        self.contador_vitoria = 0
        self._carregar_templates()

        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(ALTURA, LARGURA, 1),
            dtype=np.uint8
        )
        self.action_space = spaces.Discrete(len(ACOES))

        self.passos    = 0
        self.max_passos = 10_000
        self.energia   = 100.0
        self.porta_esq = False
        self.porta_dir = False
        self.vivo      = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.passos           = 0
        self.energia          = 100.0
        self.porta_esq        = False
        self.porta_dir        = False
        self.vivo             = True
        self.contador_vitoria = 0

        self.capture.focar_janela("Five Nights at Freddy's")
        time.sleep(0.5)

        self.capture.clicar(676, 619)
        time.sleep(15)
        self.capture.clicar(575, 606)
        time.sleep(20)

        print("Reset completo — noite iniciada!")
        observacao = self._capturar_observacao()
        return observacao, {}

    def step(self, acao: int):
        self.passos += 1
        self._executar_acao(acao)
        time.sleep(0.25)

        observacao = self._capturar_observacao()
        morreu     = self._detectar_morte()
        sobreviveu = self._detectar_vitoria()
        recompensa = self._calcular_recompensa(morreu, sobreviveu, acao)
        terminado  = morreu or sobreviveu
        truncado   = self.passos >= self.max_passos

        info = {
            "passos":    self.passos,
            "energia":   self.energia,
            "porta_esq": self.porta_esq,
            "porta_dir": self.porta_dir,
            "morreu":    morreu,
        }

        return observacao, recompensa, terminado, truncado, info

    def _executar_acao(self, acao: int):
        nome_acao = ACOES[acao]

        if nome_acao == "nada":
            return

        if nome_acao == "porta_esquerda":
            self.porta_esq = not self.porta_esq

        if nome_acao == "porta_direita":
            self.porta_dir = not self.porta_dir

        if nome_acao in COORDS:
            x, y = COORDS[nome_acao]
            self.capture.clicar(x, y)

    def _calcular_recompensa(self, morreu: bool, sobreviveu: bool, acao: int) -> float:
        if morreu:
            return -100.0

        if sobreviveu:
            return +500.0

        recompensa = +1.0
        nome_acao  = ACOES[acao]

        if nome_acao in ["porta_esquerda", "porta_direita"]:
            recompensa -= 0.5

        if nome_acao in ["luz_esquerda", "luz_direita"]:
            recompensa -= 0.3

        if self.porta_esq and self.porta_dir:
            recompensa -= 2.0

        return recompensa

    def _capturar_observacao(self) -> np.ndarray:
        frame = self.capture.capturar_tela()
        frame = self.capture.redimensionar(frame, LARGURA, ALTURA)
        frame = self.capture.para_escala_cinza(frame)
        frame = np.expand_dims(frame, axis=-1)
        return frame

    def _carregar_templates(self):
        refs = Path(__file__).parent.parent / "utils" / "referencias"

        morte_img   = cv2.imread(str(refs / "morte.jpg"),   cv2.IMREAD_GRAYSCALE)
        vitoria_img = cv2.imread(str(refs / "vitoria.png"), cv2.IMREAD_GRAYSCALE)

        if morte_img is None or vitoria_img is None:
            raise FileNotFoundError("Imagens de referência não encontradas em utils/referencias/")

        # Resolução das referências — o frame capturado será redimensionado para isso
        self._ref_size = (morte_img.shape[1], morte_img.shape[0])  # (w, h) = (1280, 720)

        # Recorta só o texto "Game Over" (canto inferior direito de morte.jpg)
        h, w = morte_img.shape
        self.template_morte = morte_img[int(h * 0.88):, int(w * 0.82):]

        # Recorta só o texto "6 AM" (centro de vitoria.png)
        h, w = vitoria_img.shape
        self.template_vitoria = vitoria_img[int(h * 0.38):int(h * 0.58), int(w * 0.38):int(w * 0.62)]

    def _capturar_janela(self) -> np.ndarray:
        """Captura apenas a janela do jogo e redimensiona para a resolução de referência."""
        import pygetwindow as gw
        janelas = gw.getWindowsWithTitle("Five Nights at Freddy's")
        if janelas:
            win = janelas[0]
            regiao = {
                "left":   win.left,
                "top":    win.top,
                "width":  win.width,
                "height": win.height,
            }
            frame = self.capture.capturar_tela(regiao)
        else:
            frame = self.capture.capturar_tela()

        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(cinza, self._ref_size)

    def _detectar_morte(self) -> bool:
        frame = self._capturar_janela()
        resultado = cv2.matchTemplate(frame, self.template_morte, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(resultado)
        return float(max_val) > 0.70

    def _detectar_vitoria(self) -> bool:
        frame = self._capturar_janela()
        resultado = cv2.matchTemplate(frame, self.template_vitoria, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(resultado)

        if float(max_val) > 0.70:
            self.contador_vitoria += 1
        else:
            self.contador_vitoria = 0

        return self.contador_vitoria >= 3

    def render(self):
        pass

    def close(self):
        pass