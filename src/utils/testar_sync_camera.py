"""Teste offline da ressincronia da câmera pelo "YOU" do mapa (decidir_sync_camera).

O jogo abaixa a prancheta sozinho quando o Foxy bate na porta. Sem a correção, o estado ficava
"aberta": porta e luz eram recusadas e o próximo Tab invertia tudo (demo de N2, 21/out, 458,9 s).
Valida a regra pura e o _sincronizar_camera do env (com a leitura do template simulada), sem
subir o jogo.

Rodar: python -m src.utils.testar_sync_camera
"""

from src.environment.fnaf_env import FNAFEnv, decidir_sync_camera, DISCORDANCIA_SYNC_CAMERA


# ── Regra pura ───────────────────────────────────────────────────────────────────────────
def test_uma_leitura_discordante_nao_corrige():
    assert decidir_sync_camera(True, False, 0) == (True, 1, False)


def test_leituras_seguidas_corrigem_foxy():
    # Acreditada aberta, o jogo abaixou a prancheta: corrige na DISCORDANCIA_SYNC_CAMERA-ésima.
    aberta, disc, corrigiu = True, 0, False
    for _ in range(DISCORDANCIA_SYNC_CAMERA):
        aberta, disc, corrigiu = decidir_sync_camera(aberta, False, disc)
    assert (aberta, disc, corrigiu) == (False, 0, True)


def test_toggle_perdido_corrige_para_aberta():
    aberta, disc, corrigiu = False, 0, False
    for _ in range(DISCORDANCIA_SYNC_CAMERA):
        aberta, disc, corrigiu = decidir_sync_camera(aberta, True, disc)
    assert (aberta, disc, corrigiu) == (True, 0, True)


def test_leitura_concordante_zera_contagem():
    # Discordâncias intercaladas com concordância são ruído: nunca corrigem.
    aberta, disc = True, 0
    for real in (False, True, False, True, False):
        aberta, disc, corrigiu = decidir_sync_camera(aberta, real, disc)
        assert not corrigiu
    assert aberta is True


def test_sem_template_mantem():
    assert decidir_sync_camera(True, None, 1) == (True, 1, False)


# ── _sincronizar_camera do env (leitura do template simulada) ───────────────────────────
def _env(camera_aberta: bool, leituras: list[bool]) -> FNAFEnv:
    env = FNAFEnv.__new__(FNAFEnv)   # sem __init__: não sobe captura nem jogo
    env.template_camera_aberta = object()
    env.camera_aberta = camera_aberta
    env.camera_ativa = 4 if camera_aberta else 0
    env.cooldown_camera = 0
    env._template_camera_discordancia = 0
    env._count_sync_camera = 0
    fila = list(leituras)
    env._camera_aberta_por_template = lambda frame=None: fila.pop(0)
    return env


def test_env_foxy_abaixa_a_prancheta():
    # Sequência da demo: na CAM 2A (ativa=4), o jogo volta ao escritório.
    env = _env(True, [True, False, False])
    assert [env._sincronizar_camera() for _ in range(3)] == [False, False, True]
    assert (env.camera_aberta, env.camera_ativa, env._count_sync_camera) == (False, 0, 1)


def test_env_nao_le_durante_o_cooldown():
    # No meio da animação o "YOU" some/aparece: nem lê o template (a fila não é consumida).
    env = _env(True, [False, False])
    env.cooldown_camera = 2
    assert not env._sincronizar_camera() and not env._sincronizar_camera()
    assert env.camera_aberta is True and env._template_camera_discordancia == 0


PURA = [
    test_uma_leitura_discordante_nao_corrige,
    test_leituras_seguidas_corrigem_foxy,
    test_toggle_perdido_corrige_para_aberta,
    test_leitura_concordante_zera_contagem,
    test_sem_template_mantem,
]
ENV = [
    test_env_foxy_abaixa_a_prancheta,
    test_env_nao_le_durante_o_cooldown,
]


def _rodar(grupo):
    ok = 0
    for teste in grupo:
        try:
            teste()
            print(f"  [PASS] {teste.__name__}")
            ok += 1
        except AssertionError as erro:
            print(f"  [FAIL] {teste.__name__}  ({erro or 'assercao falhou'})")
        except Exception as erro:  # noqa: BLE001
            print(f"  [ERRO] {teste.__name__}: {type(erro).__name__}: {erro}")
    return ok


def main() -> int:
    print("Regra pura (decidir_sync_camera):")
    ok_pura = _rodar(PURA)

    print("\nEnv (_sincronizar_camera, template simulado):")
    ok_env = _rodar(ENV)

    total_ok, total = ok_pura + ok_env, len(PURA) + len(ENV)
    print(f"\nregra: {ok_pura}/{len(PURA)}   env: {ok_env}/{len(ENV)}   TOTAL: {total_ok}/{total}")
    return 0 if total_ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
