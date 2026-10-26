"""Calibra o RITMO de ações do BC (viés da cabeça de ação) ou só mede o ritmo previsto.

O BC treina com perda ponderada (as ações raras pesam mais), e por isso, com a ação sorteada,
porta e luz saem muito mais vezes que nas demos. O BC de 25/out apertou porta ~5 vezes por minuto
no jogo (humano: 0,2) e morreu sem energia em 11 de 18 noites. A calibração soma um delta ao viés
de cada ação para o ritmo previsto no jogo bater com o humano por minuto (ver calibrar_ritmo).

O `main.py bc` já calibra no fim do treino. Esta ferramenta serve para um BC treinado antes disso
e para medir qualquer modelo sem alterar nada.

Rodar (sem o jogo):
    python -m src.utils.calibrar_ritmo_bc [dataset.json ...]
        calibra: parte de modelos/fnaf_bc_bruto.zip (se não existe, copia o fnaf_bc.zip atual
        para lá antes) e grava modelos/fnaf_bc.zip calibrado + modelos/fnaf_bc_calibracao.json.
    python -m src.utils.calibrar_ritmo_bc --medir <zip> [--s-por-step 0.9] [dataset.json ...]
        só imprime o ritmo previsto do modelo contra o humano.
Sem datasets, usa todos os dados/*/dataset.json (igual ao `main.py bc`).
"""
import glob
import os
import shutil
import sys

from sb3_contrib import RecurrentPPO

from src.agent.behavioral_cloning import (
    CAMINHO_BC_BRUTO, S_POR_STEP_JOGO, _carregar_episodios, _imprimir_ritmos, calibrar_ritmo,
    carregar_delta_calibracao, ritmos_por_minuto,
)

CAMINHO_BC = "modelos/fnaf_bc.zip"


def _valor(flag: str) -> str | None:
    if flag in sys.argv:
        i = sys.argv.index(flag)
        return sys.argv[i + 1] if i + 1 < len(sys.argv) else None
    return None


def main() -> int:
    medir = _valor("--medir")
    s_por_step = float(_valor("--s-por-step") or S_POR_STEP_JOGO)
    valores = {medir, _valor("--s-por-step")}
    caminhos = [a for a in sys.argv[1:] if not a.startswith("--") and a not in valores]
    caminhos = caminhos or sorted(glob.glob("dados/*/dataset.json"))
    if not caminhos:
        print("Nenhum dados/*/dataset.json encontrado.")
        return 1
    episodios = _carregar_episodios(caminhos)

    if medir:
        modelo = RecurrentPPO.load(medir, device="auto")
        modelo.policy.set_training_mode(False)
        previsto, humano = ritmos_por_minuto(modelo.policy, episodios, modelo.device,
                                             s_por_step=s_por_step)
        print(f"\nRitmo previsto no jogo (step de {s_por_step} s) × humano, nos estados das demos:")
        _imprimir_ritmos("humano (demos)", humano)
        _imprimir_ritmos(os.path.basename(medir), previsto)
        return 0

    if not os.path.exists(CAMINHO_BC_BRUTO):
        if carregar_delta_calibracao(CAMINHO_BC) is not None:
            print(f"{CAMINHO_BC} já está calibrado e não há {CAMINHO_BC_BRUTO} para recalibrar.")
            return 1
        shutil.copyfile(CAMINHO_BC, CAMINHO_BC_BRUTO)
        print(f"BC atual guardado como bruto: {CAMINHO_BC_BRUTO}")
    modelo = RecurrentPPO.load(CAMINHO_BC_BRUTO, device="auto")
    calibrar_ritmo(modelo, episodios, CAMINHO_BC)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
