"""Validação offline da detecção de ameaça (Decisão 4A), sem subir o jogo.

Três partes:
- ROSTO (matchTemplate): porta aberta à esquerda (Bonnie) e Chica à direita.
  Confere contra frames inteiros rotulados em debug/ (transiente).
- JANELA do Bonnie (máscaras borda/silhueta, out/2026): com a luz esquerda acesa, a janela de vidro
  ao lado da porta mostra a silhueta do Bonnie — porta aberta ou fechada. Confere contra os recortes
  committados em referencias/sombra/janela_*.png (rodam em qualquer máquina).
- SEQUÊNCIAS: decidir_bonnie sobre frames reais em debug/bonnie_porta/ (transiente), incluindo o
  caso do bug "fecha, reabre e morre".

Rodar: python -m src.utils.testar_deteccao_ameaca
"""

from pathlib import Path

import cv2
import numpy as np

from src.environment.fnaf_env import (BOTAO_LUZ_ESQ, FNAFEnv, JANELA_SOMBRA, LIMIAR_AMEACA,
                                      LIMIAR_BORDA_ACESA, LIMIAR_SILHUETA_VAZIA, decidir_bonnie,
                                      ler_janela_sombra)

DEBUG = Path("debug")
REFS = Path("src/utils/referencias")
SOMBRA_REF = REFS / "sombra"

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

# JANELA do Bonnie — recortes committados em referencias/sombra/janela_<nome>.png.
# vazio/sombra/apagada são as fontes das máscaras (sanidade); transicao e rosto são independentes.
# (nome, luz chegou ao corredor?, Bonnie na janela?)  — o 3º só vale com luz
CASOS_JANELA = [
    ("vazio",     True,  False),  # luz acesa, corredor vazio
    ("sombra",    True,  True),   # luz acesa, porta fechada, Bonnie atrás (sombra)
    ("rosto",     True,  True),   # luz acesa, porta aberta, Bonnie visível
    ("apagada",   False, None),   # luz apagada
    ("transicao", False, None),   # botão LIGHT já aceso, corredor ainda escuro
]

# SEQUÊNCIAS — frames reais em debug/bonnie_porta/ (1280x720, git-ignorados).
#   01/05 = porta aberta, Bonnie visível     02/06 = porta fechada, luz acesa, sombra do Bonnie
#   03    = porta fechada, luz APAGADA        04    = porta fechada, sombra (o print do bug: a flag
#   07    = porta fechada, luz acesa, vazio           antiga estava limpa com ele ali)
#   08    = porta aberta, botão aceso mas corredor escuro (transição)
#   09    = porta aberta, luz acesa, vazio
# (nome do caso, sequência de frames, estado esperado da flag no fim)
#   "NN>D" = o frame NN com a visão GIRADA D px para a direita (o conteúdo anda para a esquerda),
#            simulando o agente olhando para o centro/direita — a trava de alinhamento tem de barrar.
CASOS_SEQUENCIA = [
    ("vê o rosto",                             ["01", "01"], True),
    ("picos isolados entre sombras (bug)",     ["01", "01", "07", "02", "07", "06", "07", "02", "07", "06"], True),
    ("luz apagada não confirma nada",          ["01", "01", "03", "03", "03", "03", "03"], True),
    ("Bonnie saiu: vazio repetido",            ["01", "01", "07", "07", "07", "07"], False),
    ("transição da luz não inventa ameaça",    ["09", "08", "08", "08", "09", "08", "08"], False),
    ("sombra com a porta fechada acende",      ["07", "02", "06"], True),
    ("o print 04: Bonnie ainda lá",            ["02", "02", "04", "04"], True),
    ("Bonnie saiu pela porta aberta",          ["01", "01", "09", "09", "09", "09"], False),
    ("visão girada não acende (200 px)",       ["09", "07>200", "07>200", "03>200", "07>10"], False),
    ("visão girada não apaga (240 px)",        ["01", "01", "07>240", "09>240", "07>300", "09>300"], True),
]
BONNIE_PORTA = DEBUG / "bonnie_porta"


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


def _rodar_janela():
    """Cola cada recorte na posição da JANELA_SOMBRA de um frame preto e lê pelas máscaras.
    O botão LIGHT é pintado aceso no lugar calibrado para passar a trava de alinhamento (a trava
    em si é testada nas sequências, com frames reais girados)."""
    ok = total = 0
    left, top, w, h = JANELA_SOMBRA
    bl, bt, bw, bh = BOTAO_LUZ_ESQ
    for nome, luz_esp, bonnie_esp in CASOS_JANELA:
        recorte = cv2.imread(str(SOMBRA_REF / f"janela_{nome}.png"), cv2.IMREAD_GRAYSCALE)
        frame = np.zeros((720, 1280), np.uint8)
        frame[bt:bt + bh, bl:bl + bw] = 255
        if recorte is not None:
            frame[top:top + h, left:left + w] = recorte
        leitura = None if recorte is None else ler_janela_sombra(frame)
        if leitura is None:
            print(f"  [--]   {nome:10s} (recorte ou máscaras ausentes em {SOMBRA_REF}/)")
            continue
        borda, silh = leitura
        luz = borda > LIMIAR_BORDA_ACESA
        bonnie = luz and silh < LIMIAR_SILHUETA_VAZIA
        certo = luz == luz_esp and (bonnie_esp is None or bonnie == bonnie_esp)
        ok += certo
        total += 1
        print(f"  [{'OK' if certo else 'FAIL':4s}] {nome:10s} borda={borda:5.1f} silhueta={silh:5.1f} "
              f"→ luz={'sim' if luz else 'não'}{'' if not luz else ', Bonnie=' + ('sim' if bonnie else 'não')}")
    return ok, total


def _girar(frame, desloc: int):
    """Visão girada `desloc` px para a direita: o conteúdo anda para a esquerda."""
    g = np.zeros_like(frame)
    g[:, :-desloc] = frame[:, desloc:]
    return g


def _rodar_sequencias(env):
    frames = {p.name[:2]: cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY)
              for p in BONNIE_PORTA.glob("0*.png")}
    ok = total = 0
    for nome, seq, esperado in CASOS_SEQUENCIA:
        if any(k.split(">")[0] not in frames for k in seq):
            print(f"  [--]   {nome:38s} (frames ausentes em {BONNIE_PORTA}/)")
            continue
        ameaca, n_p, n_v = False, 0, 0
        for k in seq:
            base, _, desloc = k.partition(">")
            g = _girar(frames[base], int(desloc)) if desloc else frames[base]
            ameaca, n_p, n_v, _ = decidir_bonnie(
                ameaca, n_p, n_v, rosto=env._match_ameaca(g, "esquerdo"),
                janela=ler_janela_sombra(g))
        certo = ameaca == esperado
        ok += certo
        total += 1
        print(f"  [{'OK' if certo else 'FAIL':4s}] {nome:38s} {'-'.join(seq):32s} "
              f"flag={'AMEAÇA' if ameaca else 'limpa'} esperado={'AMEAÇA' if esperado else 'limpa'}")
    return ok, total


def main() -> int:
    env = _env()
    print(f"rosto: presente se score > {LIMIAR_AMEACA}   janela: luz se borda > {LIMIAR_BORDA_ACESA}, "
          f"Bonnie se silhueta < {LIMIAR_SILHUETA_VAZIA}")
    print("(positivos de rosto marcam 1.000 = self-match: o template foi recortado do\n"
          " próprio frame; capture um frame de Bonnie INDEPENDENTE p/ uma margem honesta)\n")

    print("Detecção por rosto (porta aberta / Chica):")
    ok_rosto, total_rosto = _rodar_rosto(env)

    print("\nJanela do Bonnie (máscaras borda/silhueta, recortes committados):")
    ok_janela, total_janela = _rodar_janela()

    print("\nSequências do Bonnie (decidir_bonnie sobre frames reais):")
    ok_seq, total_seq = _rodar_sequencias(env)

    ok = ok_rosto + ok_janela + ok_seq
    total = total_rosto + total_janela + total_seq
    print(f"\nRosto: {ok_rosto}/{total_rosto}   Janela: {ok_janela}/{total_janela}   "
          f"Sequências: {ok_seq}/{total_seq}   Total: {ok}/{total}")
    return 0 if total > 0 and ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
