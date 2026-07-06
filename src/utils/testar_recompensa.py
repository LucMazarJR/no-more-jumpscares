"""Teste offline da funcao de recompensa (Decisao 2) — redesenho NAO-PRESCRITIVO.

Valida o incentivo de _calcular_recompensa + o shaping potential-based sem subir o jogo:
a funcao so le estado abstrato (energia, portas, tempo, contadores), nunca pixels.

Os invariantes guardam o redesenho contra regressoes (especialmente o vicio de acampar
na camera): sinal denso por TEMPO REAL (nao por nro de steps), orcamento << vitoria, e
toda "dica de jogo" so via Phi (telescopa, nao move o otimo).

Rodar: python -m src.utils.testar_recompensa
"""

import time

from src.environment.fnaf_env import (
    FNAFEnv, ACOES, CHECKPOINTS_NOITE, GAMMA,
    RECOMPENSA_NOITE, BONUS_MARCO_HORA, FOXY_PACIENCIA_S, PESO_ENERGIA,
)

NOME_PARA_ACAO = {nome: i for i, nome in ACOES.items()}

PASSO_DT = 0.7  # ~0.7s por step, como o step real com animacoes


def _novo_env() -> FNAFEnv:
    # Cria o env sem rodar __init__ (que subiria GameCapture + templates).
    env = FNAFEnv.__new__(FNAFEnv)
    env.energia = 100.0
    env.tempo_jogo = 0.0
    env._dt_step = PASSO_DT
    env.porta_esq = False
    env.porta_dir = False
    env.luz_esq = False
    env.luz_dir = False
    env.camera_aberta = False
    env.penultima_acao = None
    env.contador_nada = 0
    env._t_ultima_camera = None
    env._horas_bonificadas = set()
    env._total_bonus_hora = 0.0
    env.ameaca_esq = False
    env.ameaca_dir = False
    return env


def recompensa_snapshot(*, energia=70.0, tempo=200.0, porta_esq=False, porta_dir=False,
                        tempo_sem_camera=0.0, dt=PASSO_DT, acao="nada", morreu=False,
                        sobreviveu=False, acao_valida=True, com_bonus=False) -> float:
    # Por padrao suprime o bonus_hora (pre-marca os checkpoints) para isolar o step.
    env = _novo_env()
    if not com_bonus:
        env._horas_bonificadas = {t for t, _ in CHECKPOINTS_NOITE}
    env.energia = energia
    env.tempo_jogo = tempo
    env._dt_step = dt
    env.porta_esq = porta_esq
    env.porta_dir = porta_dir
    # O risco do Foxy agora e wall-clock: "t segundos sem camera" = ancora no passado.
    env._t_ultima_camera = time.perf_counter() - tempo_sem_camera
    return env._calcular_recompensa(morreu, sobreviveu, NOME_PARA_ACAO[acao], acao_valida)


# Lido da propria recompensa, nao cravado, para acompanhar a implementacao atual.
VITORIA = recompensa_snapshot(sobreviveu=True)


def simular(roteiro, *, dt=PASSO_DT, morre_no_fim=False, sobrevive_no_fim=False,
            ameaca_esq=False, ameaca_dir=False, com_shaping=False) -> float:
    # Roda o roteiro pela recompensa real com transicao de estado aproximada. Com
    # com_shaping=True soma tambem o potential-based (gamma*Phi'-Phi), como step() faz.
    # As ameacas (se passadas) ficam fixas no episodio (o teste injeta o cenario).
    env = _novo_env()
    env._dt_step = dt
    env.ameaca_esq = ameaca_esq
    env.ameaca_dir = ameaca_dir
    cd_porta_esq = cd_porta_dir = cd_camera = 0
    prev_acao = None
    retorno = 0.0
    tempo_sem_camera = 0.0   # relogio VIRTUAL do roteiro (avanca dt por step, zera com camera)
    phi_antes = env._potencial_seguranca() if com_shaping else 0.0

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

        # Foxy por tempo real: reproduz o relogio virtual ancorando _t_ultima_camera no passado
        # (o _tempo_sem_camera() do env le perf_counter; o desvio entre set e leitura e ~us).
        tempo_sem_camera = 0.0 if env.camera_aberta else tempo_sem_camera + dt
        env._t_ultima_camera = time.perf_counter() - tempo_sem_camera

        env.tempo_jogo += dt
        itens = min(int(env.porta_esq) + int(env.porta_dir) + int(env.camera_aberta), 3)
        env.energia = max(0.0, env.energia - (0.104 + itens * 0.100) * dt)

        morreu = morre_no_fim and i == len(roteiro) - 1
        sobreviveu = sobrevive_no_fim and i == len(roteiro) - 1
        r = env._calcular_recompensa(morreu, sobreviveu, NOME_PARA_ACAO[nome], acao_valida)
        if com_shaping:
            phi_depois = 0.0 if (morreu or sobreviveu) else env._potencial_seguranca()
            r += GAMMA * phi_depois - phi_antes
            phi_antes = phi_depois
        retorno += r

        cd_porta_esq = max(0, cd_porta_esq - 1)
        cd_porta_dir = max(0, cd_porta_dir - 1)
        cd_camera = max(0, cd_camera - 1)
        prev_acao = nome

    return retorno


def bonus_hora_maximo() -> float:
    # Bonus_hora maximo possivel num episodio (todos os marcos de hora alcancados).
    env = _novo_env()
    for t_cp, _ in CHECKPOINTS_NOITE[1:]:
        env.tempo_jogo = t_cp
        env._calcular_recompensa(False, False, NOME_PARA_ACAO["nada"], True)
    return env._total_bonus_hora


def _phi(*, ameaca_esq=False, porta_esq=False, ameaca_dir=False, porta_dir=False,
         tempo_sem_camera=0.0, energia=100.0) -> float:
    env = _novo_env()
    env.ameaca_esq, env.porta_esq = ameaca_esq, porta_esq
    env.ameaca_dir, env.porta_dir = ameaca_dir, porta_dir
    env._t_ultima_camera = time.perf_counter() - tempo_sem_camera
    env.energia = energia
    return env._potencial_seguranca()


# ── Objetivo: o terminal e que separa jogar bem de mal (nao ha bonus por acao) ──────────
def test_terminais_ordenados():
    vivo = recompensa_snapshot(acao="porta_direita", porta_dir=True)
    vitoria = recompensa_snapshot(sobreviveu=True)
    morte = recompensa_snapshot(morreu=True)
    assert vitoria > vivo > morte


def test_vencer_supera_quase_vencer():
    venceu = simular(["nada"] * 120, sobrevive_no_fim=True)
    quase = simular(["nada"] * 120, morre_no_fim=True)
    assert venceu > quase


def test_vitoria_domina_sobrevivencia():
    # Todo o sinal denso de uma noite (sobrevivencia + marcos) << vitoria: vencer pesa mais
    # que so durar. E o conserto direto do "sobreviver acampando rivaliza com vencer".
    denso_max = RECOMPENSA_NOITE + BONUS_MARCO_HORA * (len(CHECKPOINTS_NOITE) - 1)
    assert denso_max < 0.5 * VITORIA


# ── Anti-vicio: denso por TEMPO REAL, nao por nro de steps ──────────────────────────────
def test_sobreviver_mais_rende_mais():
    curto = simular(["nada"] * 50)
    longo = simular(["nada"] * 150)
    assert longo > curto


def test_densa_independe_de_steps():
    # Mesma duracao de relogio (70s), nro de steps diferente: a recompensa densa NAO pode
    # pagar mais por spammar acoes rapidas (era um vetor do vicio de camera/spam).
    poucos = simular(["nada"] * 100, dt=0.70)   # 70s
    muitos = simular(["nada"] * 200, dt=0.35)   # 70s
    assert abs(muitos - poucos) < 0.01 * RECOMPENSA_NOITE


def test_acao_sem_efeito_penaliza():
    valida = recompensa_snapshot(acao="camera_1a", acao_valida=True)
    invalida = recompensa_snapshot(acao="camera_1a", acao_valida=False)
    assert invalida < valida


def test_marcos_nao_rivalizam_vitoria():
    assert bonus_hora_maximo() < 0.5 * VITORIA


# ── Shaping potential-based (Phi): ameaca + risco do Foxy, telescopa ─────────────────────
def test_phi_bloqueio_supera_exposto():
    bloqueado = _phi(ameaca_esq=True, porta_esq=True)
    exposto = _phi(ameaca_esq=True, porta_esq=False)
    assert bloqueado > exposto


def test_phi_sem_ameaca_so_reserva_de_energia():
    # Sem ameaca e com camera fresca, fechar porta nao da seguranca: Phi e SO a reserva
    # de energia (e com bateria zerada e nada mais em jogo, Phi = 0).
    assert _phi(ameaca_esq=False, porta_esq=True, ameaca_dir=False, porta_dir=True,
                energia=70.0) == _phi(energia=70.0)
    assert abs(_phi(energia=0.0)) < 1e-9


def test_phi_energia_monotonica():
    # Mais bateria = situacao mais segura (base do preco imediato do gasto).
    assert _phi(energia=80.0) > _phi(energia=20.0)
    assert abs((_phi(energia=100.0) - _phi(energia=0.0)) - PESO_ENERGIA) < 1e-9


def test_shaping_gastar_energia_custa_no_step():
    # O invariante certo do PBRS de energia: no SOMATORIO DESCONTADO ele telescopa para
    # -Phi(s0) (nao muda o otimo), mas o sinal POR STEP fica imediato — o step que drena
    # mais energia recebe shaping (gamma*Phi' - Phi) menor que o step frugal equivalente.
    e0 = 70.0
    dreno_base  = 0.104 * PASSO_DT              # so o passivo
    dreno_gasto = (0.104 + 2 * 0.100) * PASSO_DT  # 2 itens ligados
    phi0 = _phi(energia=e0)
    shap_frugal = GAMMA * _phi(energia=e0 - dreno_base) - phi0
    shap_gasto  = GAMMA * _phi(energia=e0 - dreno_gasto) - phi0
    assert shap_gasto < shap_frugal


def test_phi_foxy_reduz_seguranca():
    sem = _phi(tempo_sem_camera=0.0)
    negligenciado = _phi(tempo_sem_camera=FOXY_PACIENCIA_S * 2)
    assert negligenciado < sem


def test_phi_foxy_tolera_ate_paciencia():
    # Ate a paciencia, o risco do Foxy e 0 — sem empurrao para a camera no caso comum
    # (e o oposto da antiga penalidade unilateral, que so punia NAO-checar).
    # Margem de 50ms abaixo do limiar: a leitura por perf_counter avanca ~us entre
    # o set e o calculo, e exatamente NO limiar o teste ficaria flaky.
    assert _phi(tempo_sem_camera=FOXY_PACIENCIA_S - 0.05) == _phi(tempo_sem_camera=0.0)


def test_shaping_telescopa():
    # Soma do shaping (gamma*Phi'-Phi) num episodio que comeca e termina seguro (Phi=0)
    # fica ~0 (residuo so do desconto) — nao domina o objetivo verdadeiro.
    phis = [0.0, 0.5, 0.5, 0.0, -0.5, 0.0, 0.0]
    soma = sum(GAMMA * phis[i + 1] - phis[i] for i in range(len(phis) - 1))
    assert abs(soma) < 0.05


def test_acampar_na_ameaca_nao_supera_lidar():
    # Ameaca presente o tempo todo. "Lidar": fecha a porta e sobrevive (o jogo nao mata).
    # "Acampar": fica na camera, ameaca exposta (nao da p/ fechar porta na camera) e morre.
    # Com terminal + shaping, lidar tem de render bem mais — guarda contra o vicio de camera.
    lidar = simular(["porta_esquerda"] + ["nada"] * 59, ameaca_esq=True,
                    sobrevive_no_fim=True, com_shaping=True)
    acampar = simular(["abrir_fechar_camera"] + ["camera_1a"] * 59, ameaca_esq=True,
                      morre_no_fim=True, com_shaping=True)
    assert lidar > acampar


OBJETIVO = [
    test_terminais_ordenados,
    test_vencer_supera_quase_vencer,
    test_vitoria_domina_sobrevivencia,
    test_marcos_nao_rivalizam_vitoria,
]

ANTI_VICIO = [
    test_sobreviver_mais_rende_mais,
    test_densa_independe_de_steps,
    test_acao_sem_efeito_penaliza,
    test_acampar_na_ameaca_nao_supera_lidar,
]

SHAPING = [
    test_phi_bloqueio_supera_exposto,
    test_phi_sem_ameaca_so_reserva_de_energia,
    test_phi_energia_monotonica,
    test_shaping_gastar_energia_custa_no_step,
    test_phi_foxy_reduz_seguranca,
    test_phi_foxy_tolera_ate_paciencia,
    test_shaping_telescopa,
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

    print("Objetivo (terminal manda):")
    ok_obj = _rodar(OBJETIVO)

    print("\nAnti-vicio (denso por tempo, nao por steps):")
    ok_anti = _rodar(ANTI_VICIO)

    print("\nShaping potential-based (Phi):")
    ok_shaping = _rodar(SHAPING)

    total_ok = ok_obj + ok_anti + ok_shaping
    total = len(OBJETIVO) + len(ANTI_VICIO) + len(SHAPING)
    print(f"\nObjetivo: {ok_obj}/{len(OBJETIVO)}   Anti-vicio: {ok_anti}/{len(ANTI_VICIO)}"
          f"   Shaping: {ok_shaping}/{len(SHAPING)}   TOTAL: {total_ok}/{total}")
    return 0 if total_ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
