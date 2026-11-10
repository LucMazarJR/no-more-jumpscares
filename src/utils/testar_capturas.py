"""Teste offline das capturas HD das noites do agente (src/utils/capturas.py), com dados sintéticos.

Cobre a classe do quadro (blip, estática), as janelas de eventos, a novidade contra o banco, a
rotação de recentes/, a cota de disco e o gravador de ponta a ponta (thread, fechamento da noite).

Rodar: python -m src.utils.testar_capturas
"""
import tempfile
import time
from pathlib import Path

import numpy as np

from src.utils import capturas as C

RNG = np.random.default_rng(0)


def _cena(semente: int = 0) -> np.ndarray:
    """Quadro cinza 1280x720 'normal': gradiente escuro com textura fraca e estável."""
    r = np.random.default_rng(semente)
    base = np.tile(np.linspace(2, 16, 1280, dtype=np.float32), (720, 1))  # escuro como o escritório
    return np.clip(base + r.normal(0, 2, base.shape), 0, 255).astype(np.uint8)


# ── classe do quadro ────────────────────────────────────────────────────────────────────────
def test_blip_e_transicao():
    q = _cena()
    q[100:190, :] = 255                       # faixa branca da troca de câmera
    assert C.classificar_quadro(q, camera_aberta=True) == "transicao"


def test_borda_da_camera_nao_e_blip():
    q = _cena()
    q[0:4, :] = 255                           # a câmera tem uma borda branca de poucas linhas
    assert C.classificar_quadro(q, camera_aberta=True) == "normal"


def test_estatica():
    q = RNG.integers(0, 255, (720, 1280), dtype=np.uint8)
    assert C.classificar_quadro(q, camera_aberta=False) == "estatica"


def test_canto_claro_com_camera_fechada():
    q = _cena()
    q[400:720, 780:1280] = 120                # rosto do animatrônico na tela toda
    assert C.classificar_quadro(q, camera_aberta=False) == "claro_fechada"


# ── eventos ─────────────────────────────────────────────────────────────────────────────────
def _metas(n=200, dt=0.9, **extra):
    return [dict({"t": k * dt, "noite": 1, "energia": 100 - k * 0.1}, **extra) for k in range(n)]


def test_morte_guarda_os_ultimos_20s():
    m = _metas()
    tipo, a, b = [j for j in C.janelas_de_eventos(m, "morte_animatronico") if j[0] == "morte_animatronico"][0]
    assert b == len(m) - 1 and abs((m[b]["t"] - m[a]["t"]) - C.JANELA_MORTE) < 1.0


def test_golden_freddy_e_morte():
    tipos = {j[0] for j in C.janelas_de_eventos(_metas(), "morte_golden")}
    assert "morte_golden" in tipos


def test_queda_de_energia_e_porta_falhou():
    m = _metas()
    for k in range(100, len(m)):
        m[k]["energia"] -= 6                  # Foxy bateu: -6 de uma vez
    m[150]["porta_falhou"] = True
    tipos = [j[0] for j in C.janelas_de_eventos(m, None)]
    assert tipos.count("queda_energia") == 1 and tipos.count("porta_falhou") == 1


def test_consumo_normal_nao_e_queda():
    assert not C.janelas_de_eventos(_metas(), None)


def test_ameaca_pega_quadros_com_luz():
    m = _metas()
    for k in range(50, 60):
        m[k]["ameaca_esq"] = True
        m[k]["luz_esq"] = k % 2 == 0
    janelas = [j for j in C.janelas_de_eventos(m, None) if j[0] == "ameaca_esq"]
    assert janelas and all(m[a]["luz_esq"] for _, a, b in janelas for a in range(a, b + 1))


def test_noite_3_inteira():
    m = _metas(noite=3)
    for x in m:
        x["noite"] = 3
    assert ("noite_inteira", 0, len(m) - 1) in C.janelas_de_eventos(m, None)


# ── novidade ────────────────────────────────────────────────────────────────────────────────
def test_novidade_separa_quadro_diferente():
    banco = np.array([C.miniatura(_cena(s)) for s in range(60)])
    lim = C.limiar_banco(banco)
    normal = C.score_novidade(np.array([C.miniatura(_cena(999))]), banco)[0]
    diferente = _cena(5)
    diferente[200:500, 300:700] = 200         # algo apareceu na cena
    novo = C.score_novidade(np.array([C.miniatura(diferente)]), banco)[0]
    assert normal <= lim < novo


def test_banco_pequeno_nao_julga():
    banco = np.array([C.miniatura(_cena(s)) for s in range(C.BANCO_MIN - 1)])
    assert np.isnan(C.score_novidade(banco[:1], banco)).all() and C.limiar_banco(banco) == float("inf")


# ── disco ───────────────────────────────────────────────────────────────────────────────────
def test_rotacao_mantem_4():
    with tempfile.TemporaryDirectory() as d:
        rec = Path(d) / "recentes"
        for k in range(6):
            (rec / f"2026111{k}_000000_ep0001_n1").mkdir(parents=True)
        C.rotacionar(rec)
        assert sorted(p.name[:8] for p in rec.iterdir()) == [f"2026111{k}" for k in (2, 3, 4, 5)]


def test_cota_apaga_recentes_antes_dos_eventos():
    with tempfile.TemporaryDirectory() as d:
        raiz = Path(d)
        for k in range(3):
            p = raiz / "recentes" / f"2026111{k}_x"
            p.mkdir(parents=True)
            (p / "a.bin").write_bytes(b"0" * 1000)
        ev = raiz / "notaveis" / "20261110_x" / "morte_animatronico_00300s"
        ev.mkdir(parents=True)
        (ev / "a.bin").write_bytes(b"0" * 1000)
        protegida = raiz / "recentes" / "20261110_x"
        C.aplicar_cota(raiz, limite_bytes=2500, proteger=protegida)
        assert ev.exists() and protegida.exists()
        assert not (raiz / "recentes" / "20261111_x").exists()


def test_hardlink_conta_uma_vez():
    with tempfile.TemporaryDirectory() as d:
        a = Path(d) / "a.bin"
        a.write_bytes(b"0" * 1000)
        C._linkar(a, Path(d) / "sub" / "b.bin")
        assert C.uso_disco(Path(d)) == 1000


# ── gravador de ponta a ponta ───────────────────────────────────────────────────────────────
def test_gravador_noite_completa():
    with tempfile.TemporaryDirectory() as d:
        g = C.GravadorCapturas(Path(d), cota_gb=1.0)
        g.iniciar_noite(1, 1)
        quadros = [np.repeat(_cena(s)[..., None], 3, axis=2) for s in range(3)]
        t0 = time.perf_counter()
        for k in range(40):
            g.registrar(quadros[k % 3], {"t": k * 0.9, "noite": 1, "energia": 100, "camera_aberta": False,
                              "acao": "nada"})
        gasto = time.perf_counter() - t0
        g.marcar_desfecho("morte_animatronico")
        g.encerrar_noite(assincrono=True)
        g.iniciar_noite(2, 1)                 # espera o fechamento da anterior
        raiz = Path(d)
        assert gasto < 1.0, f"registrar segurou o step: {gasto:.2f}s"
        noites = list((raiz / "notaveis").iterdir())
        assert len(noites) == 1 and any(p.name.startswith("morte_animatronico") for p in noites[0].iterdir())
        assert (raiz / "resumo.md").exists() and list((raiz / "amostras").iterdir())
        assert list((raiz / "banco").glob("*.npz"))
        g.encerrar_noite()                    # noite vazia não quebra


TESTES = [
    test_blip_e_transicao, test_borda_da_camera_nao_e_blip, test_estatica, test_canto_claro_com_camera_fechada,
    test_morte_guarda_os_ultimos_20s, test_golden_freddy_e_morte, test_queda_de_energia_e_porta_falhou,
    test_consumo_normal_nao_e_queda, test_ameaca_pega_quadros_com_luz, test_noite_3_inteira,
    test_novidade_separa_quadro_diferente, test_banco_pequeno_nao_julga,
    test_rotacao_mantem_4, test_cota_apaga_recentes_antes_dos_eventos, test_hardlink_conta_uma_vez,
    test_gravador_noite_completa,
]


def main() -> int:
    ok = 0
    for teste in TESTES:
        try:
            teste()
            print(f"  [OK]   {teste.__name__}")
            ok += 1
        except AssertionError as erro:
            print(f"  [FAIL] {teste.__name__}  ({erro or 'assercao falhou'})")
        except Exception as erro:  # noqa: BLE001
            print(f"  [ERRO] {teste.__name__}: {type(erro).__name__}: {erro}")
    print(f"\nTOTAL: {ok}/{len(TESTES)}")
    return 0 if ok == len(TESTES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
