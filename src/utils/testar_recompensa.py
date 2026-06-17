"""Teste offline da funcao de recompensa (Decisao 2).

Valida o incentivo de _calcular_recompensa sem subir o jogo: a funcao so le
estado abstrato (energia, portas, tempo, contadores), nunca pixels.

Rodar: python -m src.utils.testar_recompensa
"""

from src.environment.fnaf_env import FNAFEnv, ACOES, CHECKPOINTS_NOITE

NOME_PARA_ACAO = {nome: i for i, nome in ACOES.items()}

PASSO_DT = 0.7  # ~0.7s por step, como o step real com animacoes


def _novo_env() -> FNAFEnv:
    # Cria o env sem rodar __init__ (que subiria GameCapture + templates).
    env = FNAFEnv.__new__(FNAFEnv)
    env.energia = 100.0
    env.tempo_jogo = 0.0
    env.porta_esq = False
    env.porta_dir = False
    env.luz_esq = False
    env.luz_dir = False
    env.camera_aberta = False
    env.penultima_acao = None
    env.contador_nada = 0
    env.passos_sem_camera = 0
    env._horas_bonificadas = set()
    env._total_bonus_hora = 0.0
    return env


def recompensa_snapshot(*, energia=70.0, tempo=200.0, porta_esq=False, porta_dir=False,
                        penultima_acao=None, contador_nada=0, passos_sem_camera=0,
                        acao="nada", morreu=False, sobreviveu=False, acao_valida=True,
                        com_bonus=False) -> float:
    # Por padrao suprime o bonus_hora (pre-marca os checkpoints) para isolar o step.
    env = _novo_env()
    if not com_bonus:
        env._horas_bonificadas = {t for t, _ in CHECKPOINTS_NOITE}
    env.energia = energia
    env.tempo_jogo = tempo
    env.porta_esq = porta_esq
    env.porta_dir = porta_dir
    env.penultima_acao = penultima_acao
    env.contador_nada = contador_nada
    env.passos_sem_camera = passos_sem_camera
    return env._calcular_recompensa(morreu, sobreviveu, NOME_PARA_ACAO[acao], acao_valida)


# Lido da propria recompensa, nao cravado, para acompanhar a implementacao atual.
VITORIA = recompensa_snapshot(sobreviveu=True)


def simular(roteiro, *, morre_no_fim=False, sobrevive_no_fim=False) -> float:
    # Roda o roteiro pela recompensa real com transicao de estado aproximada.
    # Os numeros absolutos nao sao exatos; os testes so comparam rankings.
    env = _novo_env()
    cd_porta_esq = cd_porta_dir = cd_camera = 0
    prev_acao = None
    retorno = 0.0

    for i, nome in enumerate(roteiro):
        env.penultima_acao = prev_acao

        acao_valida = True
        env.contador_nada = env.contador_nada + 1 if nome == "nada" else 0

        if nome in ("porta_esquerda", "porta_direita"):
            cd = cd_porta_esq if nome == "porta_esquerda" else cd_porta_dir
            if env.camera_aberta or cd > 0:
                acao_valida = False
            elif nome == "porta_esquerda":
                env.porta_esq = not env.porta_esq
                cd_porta_esq = 3
            else:
                env.porta_dir = not env.porta_dir
                cd_porta_dir = 3
        elif nome in ("luz_esquerda", "luz_direita"):
            acao_valida = not env.camera_aberta
        elif nome == "abrir_fechar_camera":
            if cd_camera > 0:
                acao_valida = False
            else:
                env.camera_aberta = not env.camera_aberta
                cd_camera = 4
        elif nome.startswith("camera_"):
            acao_valida = env.camera_aberta

        env.passos_sem_camera = 0 if env.camera_aberta else env.passos_sem_camera + 1

        env.tempo_jogo += PASSO_DT
        itens = min(int(env.porta_esq) + int(env.porta_dir) + int(env.camera_aberta), 3)
        env.energia = max(0.0, env.energia - (0.104 + itens * 0.100) * PASSO_DT)

        morreu = morre_no_fim and i == len(roteiro) - 1
        sobreviveu = sobrevive_no_fim and i == len(roteiro) - 1
        retorno += env._calcular_recompensa(morreu, sobreviveu, NOME_PARA_ACAO[nome], acao_valida)

        cd_porta_esq = max(0, cd_porta_esq - 1)
        cd_porta_dir = max(0, cd_porta_dir - 1)
        cd_camera = max(0, cd_camera - 1)
        prev_acao = nome

    return retorno


def bonus_hora_maximo() -> float:
    # Bonus_hora maximo possivel num episodio (energia cheia em todo checkpoint).
    env = _novo_env()
    env.energia = 100.0
    for t_cp, _ in CHECKPOINTS_NOITE[1:]:
        env.tempo_jogo = t_cp
        env._calcular_recompensa(False, False, NOME_PARA_ACAO["nada"], True)
    return env._total_bonus_hora


def _roteiro_proposito(n: int):
    ciclo = ["abrir_fechar_camera", "camera_1a", "abrir_fechar_camera",
             "porta_direita", "nada", "porta_direita", "nada", "nada"]
    return [ciclo[i % len(ciclo)] for i in range(n)]


def _roteiro_spam(n: int):
    return ["luz_esquerda"] * n


# Invariantes estaveis (verdes hoje; nao podem quebrar num rebalanceamento)
def test_terminais_ordenados():
    vivo_bom = recompensa_snapshot(acao="porta_direita", porta_dir=True)
    vivo_ruim = recompensa_snapshot(acao="nada", contador_nada=40)
    vitoria = recompensa_snapshot(sobreviveu=True)
    morte = recompensa_snapshot(morreu=True)
    assert vitoria > vivo_bom > morte
    assert vitoria > vivo_ruim > morte


def test_proposito_supera_passividade():
    bom = recompensa_snapshot(acao="porta_direita", porta_dir=True, energia=70)
    passivo = recompensa_snapshot(acao="nada", contador_nada=40, energia=70)
    assert bom > passivo


def test_repetir_nao_compensa():
    assert (recompensa_snapshot(acao="camera_1a", penultima_acao="camera_1a")
            <= recompensa_snapshot(acao="camera_1a", penultima_acao=None))
    assert (recompensa_snapshot(acao="luz_direita", penultima_acao="luz_direita")
            <= recompensa_snapshot(acao="luz_direita", penultima_acao=None))


def test_ambas_portas_nao_compensa():
    duas = recompensa_snapshot(acao="nada", porta_esq=True, porta_dir=True)
    uma = recompensa_snapshot(acao="nada", porta_esq=False, porta_dir=True)
    assert duas <= uma


def test_vencer_supera_quase_vencer():
    venceu = simular(_roteiro_proposito(120), sobrevive_no_fim=True)
    quase = simular(_roteiro_proposito(120), morre_no_fim=True)
    assert venceu > quase


def test_spam_nao_supera_proposito():
    proposito = simular(_roteiro_proposito(120), morre_no_fim=True)
    spam = simular(_roteiro_spam(120), morre_no_fim=True)
    assert spam <= proposito


# Alvos da Decisao 1 (vermelhos antes do rebalanceamento, verdes depois)
def test_alvo_d1_penalidade_nao_afoga_base():
    base = recompensa_snapshot(acao="camera_1a", penultima_acao=None, energia=70)
    com_penalidade = recompensa_snapshot(acao="camera_1a", penultima_acao="camera_1a", energia=70)
    assert base > 0
    assert com_penalidade >= 0


def test_alvo_d1_bonus_nao_rivaliza_vitoria():
    assert bonus_hora_maximo() < 0.5 * VITORIA


ESTAVEIS = [
    test_terminais_ordenados,
    test_proposito_supera_passividade,
    test_repetir_nao_compensa,
    test_ambas_portas_nao_compensa,
    test_vencer_supera_quase_vencer,
    test_spam_nao_supera_proposito,
]

ALVOS_D1 = [
    test_alvo_d1_penalidade_nao_afoga_base,
    test_alvo_d1_bonus_nao_rivaliza_vitoria,
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
    print(f"bonus_hora maximo: {bonus_hora_maximo():.1f}  (vitoria = {VITORIA:.0f})\n")

    print("Invariantes estaveis:")
    ok_estaveis = _rodar(ESTAVEIS)

    print("\nAlvos da Decisao 1:")
    ok_alvos = _rodar(ALVOS_D1)

    print(f"\nEstaveis: {ok_estaveis}/{len(ESTAVEIS)}   Alvos D1: {ok_alvos}/{len(ALVOS_D1)}")
    return 0 if ok_estaveis == len(ESTAVEIS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
