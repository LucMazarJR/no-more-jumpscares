#!/usr/bin/env python3
"""Junta as avaliações do `main.py jogar` feitas em BLOCOS (várias sessões e/ou vários PCs).

Cada sessão do `jogar` grava um cabeçalho com a configuração (modelo, estocástico, noite alvo,
ablação, PC, commit, data) e uma linha por episódio. Este script agrupa as sessões pela
configuração (ignorando commit e data), soma os episódios e imprime o resumo POR NOITE:
  - por PC (para ver se as máquinas concordam: s/step e taxa de vitória);
  - e o total de todos os PCs.

Uso:
    python scripts/resumo_avaliacoes.py                                   # logs/analise/avaliacoes.log
    python scripts/resumo_avaliacoes.py logs/analise/avaliacoes.log logs/analise/avaliacoes_pc2.log
"""
from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from statistics import median

EPISODIO = re.compile(
    r"Ep\s+\d+ \| Noite (\d) \| (\w+)\s+\| sobrev\s+(\d+)s \(\s*(\d+)p\) \| energia\s+([\d.]+)% \| "
    r"ações/min\s+([\d.]+) \| causa (\S+)")
DATA = re.compile(r"^\d{4}-\d{2}-\d{2}")


def ler(caminhos: list[str]) -> dict[tuple[str, str], list[dict]]:
    """{(configuração sem PC/commit/data, PC): [episódios]}"""
    grupos: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for caminho in caminhos:
        chave = None
        esperando_config = False
        with open(caminho, encoding="utf-8") as f:
            for linha in f:
                linha = linha.rstrip("\n")
                if linha == "Avaliação iniciada":
                    esperando_config = True
                    continue
                if esperando_config:
                    campos = linha.split(" | ")
                    pc = next((c[3:] for c in campos if c.startswith("PC ")), "?")
                    config = " | ".join(c for c in campos if not (
                        c.startswith(("PC ", "commit ")) or DATA.match(c)))
                    chave = (config, pc)
                    esperando_config = False
                    continue
                m = EPISODIO.search(linha)
                if m and chave:
                    noite, resultado, tempo, passos, energia, acoes, causa = m.groups()
                    grupos[chave].append({
                        "noite": int(noite), "resultado": resultado, "tempo": float(tempo),
                        "passos": int(passos), "acoes_min": float(acoes), "causa": causa})
    return grupos


def resumir(eps: list[dict], recuo: str = "    ") -> None:
    for noite in sorted({e["noite"] for e in eps}):
        g = [e for e in eps if e["noite"] == noite]
        v = sum(1 for e in g if e["resultado"] == "VITORIA")
        causas = ", ".join(f"{c} {n}" for c, n in Counter(e["causa"] for e in g).most_common(3))
        print(f"{recuo}Noite {noite}: vitórias {v}/{len(g)} ({100 * v / len(g):.0f}%) | "
              f"sobrev mediana {median(e['tempo'] for e in g):.0f}s | "
              f"ações/min {sum(e['acoes_min'] for e in g) / len(g):.1f} | "
              f"s/step {sum(e['tempo'] for e in g) / max(1, sum(e['passos'] for e in g)):.2f} | "
              f"{causas}")


def main() -> None:
    caminhos = sys.argv[1:] or ["logs/analise/avaliacoes.log"]
    grupos = ler(caminhos)
    if not grupos:
        print("Nenhuma avaliação encontrada em:", ", ".join(caminhos))
        return
    for config in sorted({c for c, _ in grupos}):
        pcs = sorted(pc for c, pc in grupos if c == config)
        print(f"\n{config}")
        for pc in pcs:
            print(f"  PC {pc}:")
            resumir(grupos[(config, pc)])
        if len(pcs) > 1:
            print("  TODOS OS PCs:")
            resumir([e for pc in pcs for e in grupos[(config, pc)]])
    print("\nObs.: se os PCs discordarem muito (s/step ou vitória), reporte separado em vez de somar.")


if __name__ == "__main__":
    main()
