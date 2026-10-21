"""Teste offline dos callbacks que conduzem uma run (termostato de entropia e currículo).

Uma run custa dias de máquina; um erro nestes callbacks só apareceria horas depois (o alvo de
entropia errado derrete o BC, uma promoção cedo demais derruba a Noite 1). Aqui eles rodam
contra um modelo/env falsos, sem o jogo:
  • ControladorEntropia calibra o alvo pela H que o BC entrega no warmup (e só com BC);
  • o alvo nunca sobe acima da H do BC; a calibração volta do checkpoint na retomada;
  • o SB3 salva/carrega o atributo da calibração no .zip (o mecanismo de que a retomada depende);
  • o currículo promove por vitórias GERIDAS, não por vitória por apagão.

Rodar: python -m src.utils.testar_callbacks_treino
"""

import os
import tempfile

import src.agent.train as train
from src.agent.train import (ControladorEntropia, CurriculumCallback, MetricasPorNoite,
                             H_INICIO, H_FIM, ENT_MIN, WARMUP_FRAC)

FIM_WARMUP = 1.0 - WARMUP_FRAC   # progresso restante abaixo disto = fora do warmup


class _Logger:
    def __init__(self):
        self.name_to_value: dict = {}
        self.gravado: dict = {}

    def record(self, chave, valor, exclude=None):
        self.gravado[chave] = valor


class _Env:
    def __init__(self, noite: int = 1):
        self.noite_desejada = noite

    def get_attr(self, nome):
        return [getattr(self, nome)]

    def set_attr(self, nome, valor):
        setattr(self, nome, valor)


class _Modelo:
    def __init__(self, num_timesteps: int = 0, progresso: float = 1.0, env=None):
        self.logger = _Logger()
        self.ent_coef = 0.02
        self.num_timesteps = num_timesteps
        self._current_progress_remaining = progresso
        self._env = env or _Env()

    def get_env(self):
        return self._env


def _iniciar(cb, modelo):
    cb.init_callback(modelo)
    cb.on_training_start({}, {})
    return cb


def _rollout(cb, modelo, progresso, h=None):
    """Um fim de rollout: h = entropia do train() anterior (None = ainda não houve train)."""
    modelo._current_progress_remaining = progresso
    if h is None:
        modelo.logger.name_to_value.pop("train/entropy_loss", None)
    else:
        modelo.logger.name_to_value["train/entropy_loss"] = -h
    cb.on_rollout_end()


def _perto(a, b, tol=1e-6):
    return abs(a - b) < tol


# ── Termostato de entropia ──────────────────────────────────────────────────────────────
def test_calibra_pela_h_do_bc_no_fim_do_warmup():
    m = _Modelo()
    cb = _iniciar(ControladorEntropia(calibrar_pelo_bc=True), m)
    _rollout(cb, m, 0.99)                       # 1º rollout: ainda sem train()
    for prog, h in ((0.98, 0.95), (0.96, 0.90), (0.95, 0.88), (0.93, 0.86)):
        _rollout(cb, m, prog, h)
        assert m.ent_coef == ENT_MIN, "no warmup o ent_coef fica no piso"
    assert not hasattr(m, "h_alvo_calibrado"), "não calibra antes do warmup acabar"
    prog = FIM_WARMUP - 0.01
    _rollout(cb, m, prog, 0.85)
    h_bc = (0.90 + 0.88 + 0.86) / 3             # média das 3 últimas do warmup
    assert _perto(m.h_alvo_calibrado[0], h_bc), m.h_alvo_calibrado
    assert _perto(m.h_alvo_calibrado[1], min(H_FIM, h_bc))
    alvo = min(H_FIM, h_bc) + (h_bc - min(H_FIM, h_bc)) * prog
    assert _perto(m.logger.gravado["custom/h_alvo"], alvo), m.logger.gravado["custom/h_alvo"]


def test_alvo_nunca_sobe_acima_da_h_do_bc():
    m = _Modelo()
    cb = _iniciar(ControladorEntropia(calibrar_pelo_bc=True), m)
    for prog in (0.98, 0.96, 0.94):
        _rollout(cb, m, prog, 0.5)              # BC muito confiante: H abaixo do H_FIM
    _rollout(cb, m, 0.5, 0.5)
    assert m.h_alvo_calibrado == [0.5, 0.5]
    assert _perto(m.logger.gravado["custom/h_alvo"], 0.5)


def test_sem_bc_usa_a_constante():
    m = _Modelo()
    cb = _iniciar(ControladorEntropia(), m)
    for prog in (0.98, 0.96, 0.94):
        _rollout(cb, m, prog, 0.9)
    _rollout(cb, m, 0.5, 0.9)
    assert not hasattr(m, "h_alvo_calibrado")
    assert _perto(m.logger.gravado["custom/h_alvo"], H_FIM + (H_INICIO - H_FIM) * 0.5)


def test_warmup_sem_medida_cai_na_constante():
    m = _Modelo()
    cb = _iniciar(ControladorEntropia(calibrar_pelo_bc=True), m)
    _rollout(cb, m, 0.98)                       # nenhum train() dentro do warmup
    _rollout(cb, m, 0.5, 0.9)
    assert not hasattr(m, "h_alvo_calibrado")
    assert _perto(m.logger.gravado["custom/h_alvo"], H_FIM + (H_INICIO - H_FIM) * 0.5)


def test_retomada_reaplica_a_calibracao_e_mantem_ent_coef():
    m = _Modelo(num_timesteps=200_000, progresso=0.5)
    m.h_alvo_calibrado = [0.9, 0.7]             # como volta do .zip (lista)
    m.ent_coef = 0.007
    cb = _iniciar(ControladorEntropia(calibrar_pelo_bc=False), m)
    assert m.ent_coef == 0.007, "retomada não pode reaplicar o ENT_INICIO"
    _rollout(cb, m, 0.5, 0.8)
    assert _perto(m.logger.gravado["custom/h_alvo"], 0.7 + 0.2 * 0.5)


def test_sb3_salva_e_carrega_a_calibracao():
    import gymnasium as gym
    from stable_baselines3 import PPO
    modelo = PPO("MlpPolicy", gym.make("CartPole-v1"), n_steps=64, batch_size=32, device="cpu")
    modelo.h_alvo_calibrado = [0.93, 0.7]
    with tempfile.TemporaryDirectory() as pasta:
        caminho = os.path.join(pasta, "m.zip")
        modelo.save(caminho)
        carregado = PPO.load(caminho, device="cpu")
    assert list(getattr(carregado, "h_alvo_calibrado", [])) == [0.93, 0.7], \
        getattr(carregado, "h_alvo_calibrado", None)


# ── Currículo por vitórias geridas ──────────────────────────────────────────────────────
def _curriculo(pasta):
    env = _Env(noite=1)
    m = _Modelo(env=env)
    metricas = MetricasPorNoite(resumo_a_cada=10**9)
    cur = CurriculumCallback(metricas)
    cur.CAMINHO = os.path.join(pasta, "curriculo.json")   # não toca em modelos/
    _iniciar(metricas, m)
    _iniciar(cur, m)
    return env, m, metricas, cur


def _episodio(cbs, m, **info):
    info.setdefault("noite", 1)
    for cb in cbs:
        cb.locals = {"dones": [True], "infos": [info]}
        cb.on_step()


def test_vitoria_por_apagao_nao_promove():
    with tempfile.TemporaryDirectory() as pasta:
        env, m, metricas, cur = _curriculo(pasta)
        for i in range(30):                     # 100% de vitórias, só 1/3 geridas
            causa = "vitoria_gerida" if i % 3 == 0 else "vitoria_apagao"
            _episodio((metricas, cur), m, vitoria=True, causa=causa, tempo=535.0)
        assert sum(metricas.win[1]) == 30
        assert env.noite_desejada == 1, "promoveu contando vitória por apagão"


def test_vitorias_geridas_promovem():
    with tempfile.TemporaryDirectory() as pasta:
        env, m, metricas, cur = _curriculo(pasta)
        for i in range(30):                     # metade geridas, metade mortes
            if i % 2 == 0:
                _episodio((metricas, cur), m, vitoria=True, causa="vitoria_gerida", tempo=535.0)
            else:
                _episodio((metricas, cur), m, vitoria=False, causa="morte_animatronico", tempo=300.0)
        assert env.noite_desejada == 2, env.noite_desejada
        assert os.path.exists(cur.CAMINHO)


def test_janela_precisa_estar_cheia():
    with tempfile.TemporaryDirectory() as pasta:
        env, m, metricas, cur = _curriculo(pasta)
        for _ in range(29):
            _episodio((metricas, cur), m, vitoria=True, causa="vitoria_gerida", tempo=535.0)
        assert env.noite_desejada == 1, "promoveu com a janela incompleta"


def test_interrompido_nao_conta_e_tb_ganha_a_serie_gerida():
    with tempfile.TemporaryDirectory() as pasta:
        env, m, metricas, cur = _curriculo(pasta)
        _episodio((metricas, cur), m, vitoria=False, interrompido=True, tempo=10.0)
        _episodio((metricas, cur), m, vitoria=True, causa="vitoria_gerida", tempo=535.0)
        _episodio((metricas, cur), m, vitoria=True, causa="vitoria_apagao", tempo=535.0)
        assert len(metricas.win_gerida[1]) == 2
        metricas.on_rollout_end()
        assert _perto(m.logger.gravado["custom/noite_1/win_rate_gerida"], 0.5)
        assert _perto(m.logger.gravado["custom/noite_1/win_rate"], 1.0)


ENTROPIA = [test_calibra_pela_h_do_bc_no_fim_do_warmup, test_alvo_nunca_sobe_acima_da_h_do_bc,
            test_sem_bc_usa_a_constante, test_warmup_sem_medida_cai_na_constante,
            test_retomada_reaplica_a_calibracao_e_mantem_ent_coef,
            test_sb3_salva_e_carrega_a_calibracao]
CURRICULO = [test_vitoria_por_apagao_nao_promove, test_vitorias_geridas_promovem,
             test_janela_precisa_estar_cheia, test_interrompido_nao_conta_e_tb_ganha_a_serie_gerida]


def _rodar(testes) -> int:
    ok = 0
    for teste in testes:
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
    ativo_antes = train.CURRICULO_ATIVO
    train.CURRICULO_ATIVO = True        # testa a lógica de promoção mesmo se a trava estiver ligada
    try:
        print("Termostato de entropia (calibração pelo BC):")
        ok_ent = _rodar(ENTROPIA)
        print("\nCurrículo (promoção por vitórias geridas):")
        ok_cur = _rodar(CURRICULO)
    finally:
        train.CURRICULO_ATIVO = ativo_antes
    total_ok, total = ok_ent + ok_cur, len(ENTROPIA) + len(CURRICULO)
    print(f"\nentropia: {ok_ent}/{len(ENTROPIA)}   curriculo: {ok_cur}/{len(CURRICULO)}"
          f"   TOTAL: {total_ok}/{total}")
    return 0 if total_ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
