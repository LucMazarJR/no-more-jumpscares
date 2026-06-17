"""Ferramenta de inspeção visual para calibrar a detecção (Decisão 4).

Captura a janela do jogo do MESMO jeito que o ambiente vê (redimensionada para
1280x720) e salva em debug/, para a gente decidir as regiões de detecção.

Uso:
  python -m src.utils.inspecionar quadro <rotulo>
      Salva o quadro inteiro + uma versão com grade de coordenadas (para marcar
      regiões pelo número). Dá 5s para você deixar o jogo no estado desejado.

  python -m src.utils.inspecionar regiao <rotulo> <left> <top> <largura> <altura>
      Recorta a região (coords na imagem 1280x720), salva e imprime média/desvio
      de cor e cinza — para medir se uma região distingue "ameaça" de "vazio".
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.utils.capture import GameCapture, regiao_cliente
from src.environment.fnaf_env import WINDOW_TITLE

REF_W, REF_H = 1280, 720
PASTA = Path("debug")


PALAVRAS_FNAF = ("freddy", "fnaf", "five nights")


def _melhor_janela(listar: bool = False):
    """Escolhe a janela real do jogo. Filtra janelas-fantasma (1x1) e, mesmo se o
    título vier vazio (sem .env), prefere as que parecem ser o FNAF; entre essas,
    pega a de maior área."""
    import pygetwindow as gw
    janelas = gw.getWindowsWithTitle(WINDOW_TITLE)
    if not janelas:
        raise RuntimeError(f"janela '{WINDOW_TITLE}' nao encontrada — o jogo esta aberto?")
    if listar:
        print("janelas candidatas:")
        for w in janelas:
            print(f"  '{w.title}' left={w.left} top={w.top} {w.width}x{w.height}")
    reais = [w for w in janelas if w.width > 100 and w.height > 100]
    fnaf = [w for w in reais if any(p in w.title.lower() for p in PALAVRAS_FNAF)]
    return max(fnaf or reais or janelas, key=lambda w: w.width * w.height)


def _capturar_janela_cor() -> np.ndarray:
    win = _melhor_janela()
    try:
        # Sem trazer o jogo para frente, o mss captura preto (igual ao ambiente faz).
        if not win.isActive:
            win.activate()
            time.sleep(0.2)
    except Exception:
        pass
    frame = GameCapture().capturar_tela(regiao_cliente(win))  # área cliente: sem barra de título
    return cv2.resize(frame, (REF_W, REF_H))                  # mesmo enquadramento do ambiente


def _avisar_se_preto(img: np.ndarray) -> float:
    brilho = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())
    print(f"brilho medio: {brilho:.1f}/255")
    if brilho < 8:
        print("AVISO: quadro quase preto. Causas provaveis:")
        print("  - FNAF em tela cheia EXCLUSIVA -> rode em modo janela (windowed/borderless);")
        print("  - jogo perdeu o foco no instante da captura.")
    return brilho


def _com_grade(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    for x in range(0, REF_W, 100):
        cv2.line(out, (x, 0), (x, REF_H), (0, 255, 0), 1)
        cv2.putText(out, str(x), (x + 2, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    for y in range(0, REF_H, 100):
        cv2.line(out, (0, y), (REF_W, y), (0, 255, 0), 1)
        cv2.putText(out, str(y), (2, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return out


def quadro(rotulo: str, espera: int = 5):
    PASTA.mkdir(exist_ok=True)
    if espera > 0:
        print(f"{espera}s para deixar o jogo no estado desejado...")
        time.sleep(espera)
    img = _capturar_janela_cor()
    p = PASTA / f"quadro_{rotulo}.png"
    pg = PASTA / f"quadro_{rotulo}_grade.png"
    cv2.imwrite(str(p), img)
    cv2.imwrite(str(pg), _com_grade(img))
    print(f"salvo: {p}  ({img.shape[1]}x{img.shape[0]})")
    print(f"salvo (grade p/ marcar regioes): {pg}")
    _avisar_se_preto(img)


def regiao(rotulo: str, left: int, top: int, w: int, h: int, espera: int = 5):
    PASTA.mkdir(exist_ok=True)
    print(f"{espera}s para deixar o jogo no estado desejado...")
    time.sleep(espera)
    img = _capturar_janela_cor()
    crop = img[top:top + h, left:left + w]
    p = PASTA / f"regiao_{rotulo}.png"
    cv2.imwrite(str(p), crop)
    cinza = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    b, g, r = (float(crop[:, :, i].mean()) for i in range(3))
    print(f"salvo: {p}  ({crop.shape[1]}x{crop.shape[0]})")
    print(f"cinza  media={cinza.mean():.1f} desvio={cinza.std():.1f} min={cinza.min()} max={cinza.max()}")
    print(f"cor    B={b:.1f} G={g:.1f} R={r:.1f}")
    _avisar_se_preto(img)


def diagnostico(espera: int = 5):
    """Captura a tela inteira (todos os monitores) marcando onde a janela do jogo
    está, para conferir se a região bate com o jogo em setup multi-monitor."""
    PASTA.mkdir(exist_ok=True)
    try:
        win = _melhor_janela(listar=True)
    except RuntimeError as erro:
        print(erro)
        return
    cap = GameCapture()
    print(f"janela: left={win.left} top={win.top} width={win.width} height={win.height} ativa={win.isActive}")
    print("monitores (mss):")
    for i, m in enumerate(cap.sct.monitors):
        print(f"  [{i}] {m}")

    print(f"{espera}s para deixar o jogo no estado desejado...")
    try:
        if not win.isActive:
            win.activate()
            time.sleep(0.2)
    except Exception:
        pass
    time.sleep(espera)

    reg = regiao_cliente(win)                     # área cliente (sem barra de título)
    virtual = cap.sct.monitors[0]                 # união de todos os monitores
    full = cap.capturar_tela(virtual)
    x0 = reg["left"] - virtual["left"]
    y0 = reg["top"] - virtual["top"]
    marcado = full.copy()
    cv2.rectangle(marcado, (x0, y0), (x0 + reg["width"], y0 + reg["height"]), (0, 0, 255), 4)
    if marcado.shape[1] > 1600:
        escala = 1600 / marcado.shape[1]
        marcado = cv2.resize(marcado, None, fx=escala, fy=escala)
    cv2.imwrite(str(PASTA / "diag_tela_cheia.png"), marcado)

    janela = cap.capturar_tela(reg)
    cv2.imwrite(str(PASTA / "diag_janela.png"), janela)
    brilho = float(cv2.cvtColor(janela, cv2.COLOR_BGR2GRAY).mean())
    print("salvo: debug/diag_tela_cheia.png (retangulo vermelho = onde o jogo deveria estar)")
    print(f"salvo: debug/diag_janela.png (brilho {brilho:.1f}/255)")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    cmd = args[0]
    if cmd == "quadro":
        rotulo = args[1] if len(args) > 1 else "sem_rotulo"
        espera = int(args[2]) if len(args) > 2 else 5  # 0 = captura instantânea
        quadro(rotulo, espera)
    elif cmd == "diag":
        diagnostico()
    elif cmd == "regiao":
        if len(args) < 6:
            print("uso: regiao <rotulo> <left> <top> <largura> <altura>")
            sys.exit(1)
        regiao(args[1], *map(int, args[2:6]))
    else:
        print(f"comando desconhecido: {cmd}")
        print(__doc__)
