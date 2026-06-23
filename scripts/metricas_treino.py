#!/usr/bin/env python3
"""Metricas POR NOITE do treino, lidas de logs/treino.log.

A "Taxa vitoria" que aparece no log do treino e AGREGADA (todas as noites juntas) e engana
para decisoes de curriculo: conforme o agente domina a noite 1, ele chega mais na 2 e morre
mais la, entao o numero agregado estagna (ou cai) mesmo com progresso real. Este script quebra
as metricas POR NOITE numa janela movel — e o que se usa pra decidir trocar new_game <-> continue.

Uso:
    python scripts/metricas_treino.py
    python scripts/metricas_treino.py --janela 150
    python scripts/metricas_treino.py --log logs/treino.log --janela 200

Ver docs/MONITORAMENTO_TREINO.md para a interpretacao e as regras de decisao.
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

# Casa as linhas de episodio do LogCallback (train.py). Linhas OCORRIDO/cabecalho nao casam.
LINHA = re.compile(
    r"Ep\s+(\d+)\s+\|\s+Noite\s+(\d+)\s+\|\s+(VITORIA|MORTE|INTERROMPIDO)\s+\|"
    r".*?Tempo:\s+([\d.]+)\s+min"
)

# Heuristicas da regra de decisao (ver doc). Pontos de partida, calibre olhando seus dados.
P_ABASTECE   = 50.0   # win rate da noite anterior (%) que considera a frente "abastecida"
N_FAMINTA    = 25     # nº de eps na noite-frente abaixo disto = faminta de amostras
P_PRESO      = 10.0   # win rate da frente (%) abaixo disto = ainda nao destravou
PLATEAU_MIN  = 0.5    # ganho de sobrevivencia (min) entre metades abaixo disto = plateau


def parse(caminho: Path) -> list[tuple[int, int, str, float]]:
    eps: list[tuple[int, int, str, float]] = []
    for linha in caminho.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = LINHA.search(linha)
        if m:
            eps.append((int(m.group(1)), int(m.group(2)), m.group(3), float(m.group(4))))
    return eps


def media(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def tendencia(t_mortes: list[float]) -> float | None:
    """Ganho de sobrevivencia: media da 2a metade das mortes menos a 1a metade (ordem do log).
    >0 = morrendo cada vez mais tarde (aprendendo); ~0 = plateau. None se poucas mortes."""
    n = len(t_mortes)
    if n < 4:
        return None
    meio = n // 2
    return media(t_mortes[meio:]) - media(t_mortes[:meio])


def resumo_por_noite(eps: list[tuple[int, int, str, float]]) -> dict:
    por: dict[int, dict] = defaultdict(lambda: {"vit": 0, "mortes": 0, "int": 0, "t_mortes": []})
    for _, noite, res, tmin in eps:
        d = por[noite]
        if res == "VITORIA":
            d["vit"] += 1
        elif res == "MORTE":
            d["mortes"] += 1
            d["t_mortes"].append(tmin)
        else:
            d["int"] += 1
    return por


def imprimir(por: dict, total: int, janela: int) -> None:
    print(f"\nJanela: ultimos {min(total, janela)} de {total} episodios registrados\n")
    print(f"{'Noite':>5} | {'n':>4} | {'vit%':>6} | {'vit':>4} {'morte':>5} {'int':>4} | "
          f"{'t_morte(min)':>12} | {'tendencia':>9}")
    print("-" * 66)
    for noite in sorted(por):
        d = por[noite]
        validos = d["vit"] + d["mortes"]
        p = (d["vit"] / validos * 100) if validos else 0.0
        tmed = media(d["t_mortes"])
        tend = tendencia(d["t_mortes"])
        tend_str = "-" if tend is None else f"{tend:+.1f}"
        print(f"{noite:>5} | {validos:>4} | {p:>5.1f}% | {d['vit']:>4} {d['mortes']:>5} "
              f"{d['int']:>4} | {tmed:>12.1f} | {tend_str:>9}")


def recomendar(por: dict) -> None:
    nights = sorted(por)
    if not nights:
        return
    frente = nights[-1]                     # noite mais avancada alcancada = a "frente"
    if frente == 1:
        print("\n>> Ainda preso na noite 1. Metrica de currículo so vale quando a frente chega "
              "na noite 2+. Foque em destravar a noite 1.")
        return

    d_frente = por[frente]
    validos_f = d_frente["vit"] + d_frente["mortes"]
    p_f = (d_frente["vit"] / validos_f * 100) if validos_f else 0.0
    tend_f = tendencia(d_frente["t_mortes"])

    d_ant = por[frente - 1]
    validos_a = d_ant["vit"] + d_ant["mortes"]
    p_ant = (d_ant["vit"] / validos_a * 100) if validos_a else 0.0

    abastece = p_ant >= P_ABASTECE
    faminta  = validos_f < N_FAMINTA
    preso    = p_f < P_PRESO
    plateau  = tend_f is not None and tend_f < PLATEAU_MIN

    print(f"\nFrente = noite {frente} | noite {frente-1} (abastece): {p_ant:.1f}% | "
          f"frente: {p_f:.1f}% em {validos_f} eps | "
          f"tendencia frente: {'-' if tend_f is None else f'{tend_f:+.1f} min'}")

    if abastece and faminta and preso and plateau:
        print(f">> SUGESTAO: trocar TEMPORARIAMENTE para FNAF_RESET_METODO=continue. A noite "
              f"{frente-1} abastece (>= {P_ABASTECE:.0f}%), mas a frente esta faminta "
              f"(< {N_FAMINTA} eps), travada (< {P_PRESO:.0f}%) e em plateau. continue da mais "
              f"reps na frente. Volte pra new_game quando p{frente} encostar (~20-30%).")
    elif abastece and not plateau:
        print(">> MANTER new_game: a frente ainda esta APRENDENDO (sobrevivencia subindo). "
              "Trocar agora seria precipitado.")
    elif not abastece:
        print(f">> MANTER new_game: a noite {frente-1} ainda nao abastece bem a frente "
              f"({p_ant:.1f}% < {P_ABASTECE:.0f}%). O gargalo e a noite {frente-1}, nao a frente.")
    else:
        print(">> Sinais mistos — decida olhando a tabela (a heuristica nao bateu todos os "
              "criterios). Ver docs/MONITORAMENTO_TREINO.md.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Metricas por noite do treino FNAF.")
    parser.add_argument("--log", type=Path, default=Path("logs/treino.log"))
    parser.add_argument("--janela", type=int, default=150, help="ultimos N episodios (padrao 150)")
    args = parser.parse_args()

    if not args.log.exists():
        raise SystemExit(f"Log nao encontrado: {args.log}")

    eps = parse(args.log)
    if not eps:
        raise SystemExit(f"Nenhum episodio reconhecido em {args.log}.")

    janela = eps[-args.janela:]
    por = resumo_por_noite(janela)
    imprimir(por, total=len(eps), janela=args.janela)
    recomendar(por)


if __name__ == "__main__":
    main()
