#!/usr/bin/env python3
"""Taxa de AÇÃO (ações != "nada" por minuto de jogo): demos humanas × agente avaliado.

Por que existe (medição M3 de docs/ESTADO_ATUAL.md): o gravador de demos roda a ~0,25-0,33 s
por registro, o agente a ~0,9 s por step. O BC recorrente aprende a dinâmica no tick das demos
e é executado no tick do agente — se o clone herda a FRAÇÃO de "nada" por tick (~82%), ele age
~1/3 das vezes por minuto que o humano. Este script põe os dois números lado a lado.

Fontes:
  - demos: dados/*/dataset.json (campos 'nome' e 'tempo_ep' de cada registro)
  - agente: logs/analise/avaliacoes.log (escrito por `python main.py jogar`, campo "ações/min")

Uso:
    python scripts/taxa_acao.py
    python scripts/taxa_acao.py --avaliacoes logs/analise/avaliacoes.log
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict

CATEGORIAS = (("porta", "portas"), ("luz", "luzes"), ("abrir_fechar_camera", "abre/fecha cam"),
              ("camera_", "troca cam"))

LINHA_AVALIACAO = re.compile(r"Ep\s+\d+ \| Noite (\d) \| .*ações/min\s+([\d.]+)")


def categoria(nome: str) -> str:
    for prefixo, rotulo in CATEGORIAS:
        if nome.startswith(prefixo):
            return rotulo
    return "outra"


def resumo_demos() -> dict[int, list[float]]:
    por_noite: dict[int, list[float]] = defaultdict(list)
    caminhos = sorted(glob.glob("dados/*/dataset.json"))
    if not caminhos:
        print("Nenhuma demo em dados/*/dataset.json.")
        return por_noite
    print("DEMOS HUMANAS")
    print(f"  {'demo':<36} {'noite':>5} {'regs':>5} {'tick':>6} {'nada%':>6} {'ações/min':>9}  por tipo")
    for caminho in caminhos:
        with open(caminho, encoding="utf-8") as f:
            regs = json.load(f)
        if not regs:
            continue
        duracao = max(r.get("tempo_ep", 0.0) for r in regs)
        acoes = [r.get("nome", "nada") for r in regs if r.get("nome", "nada") != "nada"]
        noite = int(regs[0].get("noite", 1))
        por_min = len(acoes) / (duracao / 60) if duracao > 0 else 0.0
        por_noite[noite].append(por_min)
        tipos = Counter(categoria(a) for a in acoes)
        print(f"  {os.path.basename(os.path.dirname(caminho)):<36} {noite:>5} {len(regs):>5} "
              f"{duracao / len(regs):>5.2f}s {100 * (1 - len(acoes) / len(regs)):>5.0f}% "
              f"{por_min:>9.1f}  " + ", ".join(f"{k} {v}" for k, v in tipos.most_common()))
    return por_noite


def resumo_agente(caminho: str) -> dict[int, list[float]]:
    por_noite: dict[int, list[float]] = defaultdict(list)
    if not os.path.exists(caminho):
        print(f"\n(sem {caminho} — rode `python main.py jogar` p/ medir o agente)")
        return por_noite
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            m = LINHA_AVALIACAO.search(linha)
            if m:
                por_noite[int(m.group(1))].append(float(m.group(2)))
    return por_noite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--avaliacoes", default="logs/analise/avaliacoes.log")
    args = parser.parse_args()

    demos = resumo_demos()
    agente = resumo_agente(args.avaliacoes)

    print("\nCOMPARAÇÃO (média de ações/min por noite)")
    for noite in sorted(set(demos) | set(agente)):
        h = sum(demos[noite]) / len(demos[noite]) if demos.get(noite) else None
        a = sum(agente[noite]) / len(agente[noite]) if agente.get(noite) else None
        razao = f" | agente/humano = {a / h:.2f}" if h and a is not None else ""
        print(f"  Noite {noite}: humano {f'{h:.1f}' if h is not None else '-':>5} "
              f"({len(demos.get(noite, []))} demos) | agente {f'{a:.1f}' if a is not None else '-':>5} "
              f"({len(agente.get(noite, []))} eps){razao}")
    print("\nObs.: o arquivo de avaliações junta TODAS as avaliações já feitas (modelos e flags "
          "diferentes). Para comparar um modelo só, filtre pela seção dele no log.")


if __name__ == "__main__":
    main()
