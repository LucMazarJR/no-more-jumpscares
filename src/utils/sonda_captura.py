"""Sonda de captura — mede o que a rajada de frames resolveria ANTES de mexer no env (M4).

Hoje o env captura 1 frame por step (~0,9 s) e depende de debounce ENTRE steps para aguentar a
estática das câmeras e o piscar das luzes. A Fase 2 (docs/ESTADO_ATUAL.md) propõe capturar uma
RAJADA de k frames por step e combinar: mediana temporal (estática) e frame mais claro (luz
piscando). Esta sonda mede, no jogo real, os números que decidem se isso vale e com que k:

  1. latência do mss por frame (custo da rajada: k × latência entra em todo step);
  2. frames DISTINTOS por segundo (o jogo redesenha rápido o bastante p/ a rajada ver
     variação? se 2 capturas seguidas saem idênticas, k maior não compra nada);
  3. ruído temporal por região (std entre frames) cru × mediana de 3 × mediana de 5;
  4. piscar: brilho médio por frame em cada região → fração de frames "escuros".

Uso (jogo aberto, na situação que quer medir — escritório, câmera X, luz acesa...):
    python -m src.utils.sonda_captura --rotulo escritorio
    python -m src.utils.sonda_captura --rotulo cam_1c --frames 90
    python -m src.utils.sonda_captura --rotulo luz_esq --segurar-luz esq

Saídas: resumo no terminal + debug/sonda_<rotulo>.json + debug/sonda_<rotulo>_{bruto,mediana5,
maximo5}.png (um frame cru e as combinações, para comparar a olho).
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from src.environment.fnaf_env import (BOTAO_DOOR, COORDS, ROI_AMEACA, SOMBRA_REGIAO,
                                      WINDOW_TITLE)
from src.utils.capture import GameCapture, melhor_janela, regiao_cliente

REF_W, REF_H = 1280, 720
PASTA_SAIDA = Path("debug")

# Regiões (left, top, larg, alt) em 1280x720 — as MESMAS que os detectores do env leem.
REGIOES = {
    "frame_inteiro": (0, 0, REF_W, REF_H),
    "vao_esquerdo":  SOMBRA_REGIAO,
    "roi_ameaca_esq": ROI_AMEACA["esquerdo"],
    "roi_ameaca_dir": ROI_AMEACA["direito"],
    "botao_porta_esq": BOTAO_DOOR["esquerdo"],
}


def _recorte(frame: np.ndarray, regiao) -> np.ndarray:
    left, top, w, h = regiao
    return frame[top:top + h, left:left + w]


def capturar_rajada(capture: GameCapture, regiao_janela: dict, n: int):
    """Captura n frames o mais rápido possível. Devolve (frames cinza 1280x720, latências s)."""
    frames, latencias = [], []
    for _ in range(n):
        t = time.perf_counter()
        bgr = capture.capturar_tela(regiao_janela)
        cinza = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (REF_W, REF_H))
        latencias.append(time.perf_counter() - t)
        frames.append(cinza)
    return np.stack(frames), np.array(latencias)


def ruido_temporal(pilha: np.ndarray, k: int) -> float:
    """Std temporal média por pixel depois de combinar blocos de k frames pela mediana.
    k=1 = cru. Queda forte de k=1 p/ k=3/5 = o ruído é por-frame (estática) e a mediana limpa."""
    blocos = len(pilha) // k
    if blocos < 2:
        return float("nan")
    combinados = np.stack([np.median(pilha[i * k:(i + 1) * k], axis=0) for i in range(blocos)])
    return float(combinados.astype(np.float32).std(axis=0).mean())


def medir(pilha: np.ndarray, latencias: np.ndarray, duracao: float) -> dict:
    difs = [float(np.abs(pilha[i].astype(np.int16) - pilha[i - 1]).mean())
            for i in range(1, len(pilha))]
    distintos = sum(1 for d in difs if d > 0.5)   # >0,5 nível de cinza médio = frame novo
    resultado = {
        "frames": int(len(pilha)),
        "latencia_ms": {"media": round(1000 * float(latencias.mean()), 1),
                        "p95": round(1000 * float(np.percentile(latencias, 95)), 1)},
        "capturas_por_s": round(len(pilha) / duracao, 1),
        "frames_distintos_por_s": round(distintos / duracao, 1),
        "regioes": {},
    }
    for nome, regiao in REGIOES.items():
        sub = np.stack([_recorte(f, regiao) for f in pilha])
        brilho = sub.reshape(len(sub), -1).mean(axis=1)
        meio = (brilho.min() + brilho.max()) / 2
        amplitude = float(brilho.max() - brilho.min())
        resultado["regioes"][nome] = {
            "ruido_cru": round(ruido_temporal(sub, 1), 2),
            "ruido_mediana3": round(ruido_temporal(sub, 3), 2),
            "ruido_mediana5": round(ruido_temporal(sub, 5), 2),
            "brilho_min": round(float(brilho.min()), 1),
            "brilho_max": round(float(brilho.max()), 1),
            # só conta frames "escuros" quando há variação real de brilho (senão é ruído)
            "frac_frames_escuros": round(float((brilho < meio).mean()), 2) if amplitude > 5 else 0.0,
        }
    return resultado


def main():
    parser = argparse.ArgumentParser(description="Sonda de captura (M4)")
    parser.add_argument("--rotulo", default="escritorio", help="nome da situação medida")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--segurar-luz", choices=("esq", "dir"), default=None,
                        help="segura o botão de luz daquele lado durante a rajada")
    args = parser.parse_args()

    janela = melhor_janela(WINDOW_TITLE)
    if janela is None:
        print(f"Janela '{WINDOW_TITLE}' não encontrada — abra o jogo.")
        return
    regiao_janela = regiao_cliente(janela)
    capture = GameCapture()

    botao = None
    if args.segurar_luz:
        botao = COORDS["luz_esquerda" if args.segurar_luz == "esq" else "luz_direita"]
        capture.segurar_botao(*botao)
        time.sleep(0.15)   # a luz acende com 1-2 frames de atraso
    try:
        t0 = time.perf_counter()
        pilha, latencias = capturar_rajada(capture, regiao_janela, args.frames)
        duracao = time.perf_counter() - t0
    finally:
        if botao:
            capture.soltar_botao(*botao)

    resultado = medir(pilha, latencias, duracao)
    resultado["rotulo"] = args.rotulo
    resultado["luz_segurada"] = args.segurar_luz

    PASTA_SAIDA.mkdir(exist_ok=True)
    base = PASTA_SAIDA / f"sonda_{args.rotulo}"
    with open(f"{base}.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    cv2.imwrite(f"{base}_bruto.png", pilha[len(pilha) // 2])
    cv2.imwrite(f"{base}_mediana5.png", np.median(pilha[:5], axis=0).astype(np.uint8))
    cv2.imwrite(f"{base}_maximo5.png", pilha[:5].max(axis=0))

    print(f"\n[{args.rotulo}] {resultado['frames']} frames em {duracao:.2f}s")
    print(f"  latência mss: média {resultado['latencia_ms']['media']} ms, "
          f"p95 {resultado['latencia_ms']['p95']} ms  → rajada de 5 custa "
          f"~{5 * resultado['latencia_ms']['media']:.0f} ms/step")
    print(f"  capturas/s {resultado['capturas_por_s']} | frames DISTINTOS/s "
          f"{resultado['frames_distintos_por_s']}")
    print(f"  {'região':<16} {'ruído cru':>9} {'med3':>6} {'med5':>6} {'brilho':>13} {'escuros':>8}")
    for nome, r in resultado["regioes"].items():
        print(f"  {nome:<16} {r['ruido_cru']:>9} {r['ruido_mediana3']:>6} {r['ruido_mediana5']:>6} "
              f"{r['brilho_min']:>6}-{r['brilho_max']:<6} {r['frac_frames_escuros']:>8}")
    print(f"\nSalvo: {base}.json e {base}_{{bruto,mediana5,maximo5}}.png")


if __name__ == "__main__":
    main()
