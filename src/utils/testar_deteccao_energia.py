"""Validação offline da leitura de energia (Decisão 4B).

Roda o _ler_energia do ambiente contra frames rotulados em debug/, sem subir o
jogo. A leitura é por template binarizado de cada algarismo (imune ao fundo).

Rodar: python -m src.utils.testar_deteccao_energia
"""

from pathlib import Path

import cv2

from src.environment.fnaf_env import FNAFEnv, validar_leitura_energia

DEBUG = Path("debug")
REFS = Path("src/utils/referencias/digitos")

# rótulo do frame -> valor de power esperado (None = não deve ler: 1 dígito vazio/0% apagado)
ESPERADO = {
    "escritorio": 97,
    "luz_dir_apagada": 97,
    "luz_dir_chica": 32,
    "luz_dir_vazia": 88,
    "luz_dir_vazia2": 66,
    "luz_esq_apagada": 98,
    "luz_esq_bonnie": 56,
    "luz_esq_bonnie2": 54,
    "luz_esq_vazia": 92,
    "luz_esq_vazia2": 69,
    "power_10": 10,
    "power_unidigito": 8,    # 1 dígito (alinhado à direita)
    "no_power": None,        # 0% apagado: UI some → reader não pode chutar
}


def _env():
    env = FNAFEnv.__new__(FNAFEnv)  # sem __init__
    env.glifos_energia = {}
    for d in "0123456789":
        g = cv2.imread(str(REFS / f"{d}.png"), cv2.IMREAD_GRAYSCALE)
        if g is not None:
            env.glifos_energia[d] = g
    if len(env.glifos_energia) < 10:
        raise SystemExit("glifos de energia ausentes em referencias/digitos/")
    return env


def _testar_proximidade() -> bool:
    """Filtro photo-primary: a foto manda. A simulação drena um pouco por step
    (inferência) e só serve de fallback; a leitura corrige o drift e quedas grandes
    reais (Foxy) são aceitas. Só subida é rejeitada (impossível)."""
    DRENO = 0.3
    estimativa = 50.0
    # (leitura_crua, esperado_apos_validar) — None = deve manter a estimativa drenada
    sequencia = [
        (49, 49.0),    # leitura boa → re-ancora
        (None, None),  # câmera aberta: sem leitura → mantém a simulação
        (20, 20.0),    # queda GRANDE real (Foxy em câmera): decréscimo → ACEITA
        (60, None),    # subiu → impossível → rejeita
        (8, 8.0),      # 1 dígito, decréscimo → aceita
    ]
    ok = True
    for lido, esperado in sequencia:
        estimativa = round(estimativa - DRENO, 2)          # simulação drena (inferência)
        antes = estimativa
        estimativa = validar_leitura_energia(lido, estimativa)
        alvo = antes if esperado is None else esperado
        certo = abs(estimativa - alvo) < 0.01
        ok = ok and certo
        print(f"  [{'OK' if certo else 'FAIL':4s}] leitura={str(lido):4s} -> energia={estimativa:.2f} "
              f"(esperado {'mantém '+f'{antes:.2f}' if esperado is None else f'{esperado:.2f}'})")
    return ok


def main() -> int:
    env = _env()
    print("Filtro photo-primary (inferência só de fallback):")
    ok_filtro = _testar_proximidade()
    print("\nReader (leitura crua por frame):")
    ok = faltando = 0
    for nome, exp in ESPERADO.items():
        p = DEBUG / f"quadro_{nome}.png"
        if not p.exists():
            print(f"  [--]   {nome:16s} (frame ausente em debug/)")
            faltando += 1
            continue
        g = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY)
        lido = env._ler_energia(g)
        certo = lido == exp
        ok += certo
        print(f"  [{'OK' if certo else 'FAIL':4s}] {nome:16s} esperado={str(exp):4s} lido={lido}")
    total = len(ESPERADO) - faltando
    extra = f"  ({faltando} frames ausentes)" if faltando else ""
    print(f"\nReader: {ok}/{total} corretos{extra}   Filtro: {'OK' if ok_filtro else 'FAIL'}")
    return 0 if total > 0 and ok == total and ok_filtro else 1


if __name__ == "__main__":
    raise SystemExit(main())
