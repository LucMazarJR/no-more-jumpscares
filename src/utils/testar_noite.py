"""Teste offline da decisao de reset (Decisao 7) — acao de reset + noite alvo.

Valida decidir_reset() sem subir o jogo: a funcao e pura (so olha metodo, desfecho,
noite atual, noite desejada e o flag de 1o reset) e devolve (acao, nova_noite), onde
acao e "new_game" (clica New Game), "continue" (clica Continue) ou "nenhum" (NAO clica:
o jogo emendou sozinho na proxima noite apos a vitoria — nao ha menu).

Mecanica do jogo guardada pelos invariantes: o MENU so aparece depois de MORRER; vencer
avanca direto pra proxima noite. Logo a escolha New Game vs Continue so ocorre na morte,
e vencer a noite alvo obriga jogar D+1 ate morrer (auto-avanco, sem menu pra voltar antes).

Rodar: python -m src.utils.testar_noite
"""

from src.environment.fnaf_env import decidir_reset, MAX_NOITE


def d(metodo, resultado, noite_atual, noite_desejada=1, primeiro_reset=False):
    return decidir_reset(metodo, resultado, noite_atual, noite_desejada, primeiro_reset)


# ── 1o reset: ancora o save na Noite 1 (New Game), seja qual for o resto ─────────────────
def test_primeiro_reset_forca_new_game():
    assert d("continue", "vitoria", 5, noite_desejada=3, primeiro_reset=True) == ("new_game", 1)
    assert d("new_game", "morte", 4, primeiro_reset=True) == ("new_game", 1)


# ── Vitoria: auto-avanco, sem menu, sem clique (vale p/ os dois metodos) ─────────────────
def test_vitoria_new_game_avanca_sem_clique():
    assert d("new_game", "vitoria", 2) == ("nenhum", 3)


def test_vitoria_continue_avanca_mesmo_na_alvo():
    # Venceu a alvo, mas NAO da pra clicar New Game (sem menu) — emenda em D+1.
    assert d("continue", "vitoria", 3, noite_desejada=3) == ("nenhum", 4)


def test_vitoria_cap_max_noite():
    assert d("new_game", "vitoria", MAX_NOITE) == ("nenhum", MAX_NOITE)


# ── Truncado/None: sem morte detectada = sem menu p/ clicar ──────────────────────────────
def test_truncado_sem_clique():
    assert d("continue", None, 4, noite_desejada=2) == ("nenhum", 4)
    assert d("new_game", None, 3) == ("nenhum", 3)


# ── Morte, modo new_game: sempre volta pra Noite 1 ───────────────────────────────────────
def test_new_game_morte_volta_noite_1():
    assert d("new_game", "morte", 5) == ("new_game", 1)


# ── Morte, modo continue: mira a noite desejada D ────────────────────────────────────────
def test_continue_morte_abaixo_da_alvo_retoma():
    # D=4, morreu na 2 -> Continue (retoma a 2 p/ vencer e seguir subindo).
    assert d("continue", "morte", 2, noite_desejada=4) == ("continue", 2)


def test_continue_morte_na_alvo_retoma():
    # D=3, morreu na alvo -> Continue (retoma a alvo; treino principal).
    assert d("continue", "morte", 3, noite_desejada=3) == ("continue", 3)


def test_continue_morte_acima_da_alvo_reescala():
    # Venceu a alvo, foi obrigado a jogar D+1+, morreu acima -> New Game (volta pro 1).
    assert d("continue", "morte", 4, noite_desejada=3) == ("new_game", 1)
    assert d("continue", "morte", 6, noite_desejada=2) == ("new_game", 1)


def test_continue_alvo_1():
    # D=1 (exemplo do usuario): vence a 1 -> emenda na 2 (nenhum) ; morre na 2 -> New Game ;
    # morre na 1 -> Continue (retoma a 1).
    assert d("continue", "vitoria", 1, noite_desejada=1) == ("nenhum", 2)
    assert d("continue", "morte", 2, noite_desejada=1) == ("new_game", 1)
    assert d("continue", "morte", 1, noite_desejada=1) == ("continue", 1)


def test_continue_alvo_max_nunca_new_game():
    # D=MAX: a morte nunca cai ACIMA da alvo -> sempre Continue; nunca New Game.
    assert d("continue", "morte", 4, noite_desejada=MAX_NOITE) == ("continue", 4)
    assert d("continue", "morte", MAX_NOITE, noite_desejada=MAX_NOITE) == ("continue", MAX_NOITE)
    assert d("continue", "vitoria", MAX_NOITE, noite_desejada=MAX_NOITE) == ("nenhum", MAX_NOITE)


PRIMEIRO_RESET = [
    test_primeiro_reset_forca_new_game,
]

VITORIA_E_TRUNCADO = [
    test_vitoria_new_game_avanca_sem_clique,
    test_vitoria_continue_avanca_mesmo_na_alvo,
    test_vitoria_cap_max_noite,
    test_truncado_sem_clique,
]

MORTE = [
    test_new_game_morte_volta_noite_1,
    test_continue_morte_abaixo_da_alvo_retoma,
    test_continue_morte_na_alvo_retoma,
    test_continue_morte_acima_da_alvo_reescala,
    test_continue_alvo_1,
    test_continue_alvo_max_nunca_new_game,
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
    print("1o reset (ancora New Game):")
    ok_primeiro = _rodar(PRIMEIRO_RESET)

    print("\nVitoria e truncado (auto-avanco / sem menu, sem clique):")
    ok_vit = _rodar(VITORIA_E_TRUNCADO)

    print("\nMorte (unica tela com menu — escolhe New Game vs Continue):")
    ok_morte = _rodar(MORTE)

    total_ok = ok_primeiro + ok_vit + ok_morte
    total = len(PRIMEIRO_RESET) + len(VITORIA_E_TRUNCADO) + len(MORTE)
    print(f"\n1o reset: {ok_primeiro}/{len(PRIMEIRO_RESET)}   vitoria/trunc: {ok_vit}/{len(VITORIA_E_TRUNCADO)}"
          f"   morte: {ok_morte}/{len(MORTE)}   TOTAL: {total_ok}/{total}")
    return 0 if total_ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
