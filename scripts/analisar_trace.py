#!/usr/bin/env python3
"""Lê o trace por step do `main.py jogar --trace` e responde, para o lado ESQUERDO (Bonnie):

  1. A flag gruda? Quanto tempo ela fica acesa, e quanto disso foi confirmado por uma leitura
     da janela (silhueta vista) × herdado de uma leitura antiga.
  2. O que o agente fazia antes de cada morte: a porta esquerda estava aberta? a flag estava
     acesa? ele ABRIU a porta esquerda com a flag acesa nos últimos segundos?
  3. As mortes com a porta esquerda FECHADA não podem ser do Bonnie (ele não passa a porta):
     contam como "outro" (Chica pela direita, Foxy batendo, Freddy...).

Uso:
    python scripts/analisar_trace.py                     # o trace mais recente
    python scripts/analisar_trace.py logs/analise/trace_20261008_2300.csv --noite 2
    python scripts/analisar_trace.py --detalhe 12        # imprime os últimos 12 steps de cada morte
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict

JANELA_S = 10.0   # segundos antes da morte olhados para "abriu a porta com a flag acesa"


def ler(caminho: str) -> dict[int, list[dict]]:
    eps: dict[int, list[dict]] = defaultdict(list)
    with open(caminho, encoding="utf-8-sig") as f:   # -sig: tolera BOM (CSV salvo pelo Excel/PowerShell)
        for r in csv.DictReader(f):
            for k in ("ep", "passo", "noite", "acao_valida", "porta_esq", "porta_dir", "luz_esq",
                      "luz_dir", "camera", "ameaca_esq", "ameaca_dir"):
                r[k] = int(r[k])
            r["tempo"] = float(r["tempo"])
            r["energia"] = float(r["energia"])
            r["borda"] = float(r["borda"]) if r["borda"] else None
            r["silhueta"] = float(r["silhueta"]) if r["silhueta"] else None
            eps[r["ep"]].append(r)
    return eps


def corridas(passos: list[dict], chave: str) -> list[list[dict]]:
    """Trechos contínuos com passos[i][chave] == 1."""
    out, atual = [], []
    for p in passos:
        if p[chave]:
            atual.append(p)
        elif atual:
            out.append(atual)
            atual = []
    if atual:
        out.append(atual)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="?")
    ap.add_argument("--noite", type=int, default=None, help="só episódios dessa noite")
    ap.add_argument("--detalhe", type=int, default=0, help="últimos N steps de cada morte")
    args = ap.parse_args()
    caminho = args.trace or max(glob.glob("logs/analise/trace_*.csv"), key=os.path.getmtime, default=None)
    if not caminho:
        raise SystemExit("Nenhum trace em logs/analise/ (rode o jogar com --trace).")
    eps = ler(caminho)
    if args.noite:
        eps = {k: v for k, v in eps.items() if v and v[0]["noite"] == args.noite}
    print(f"Trace: {caminho} | {len(eps)} episódios" + (f" (noite {args.noite})" if args.noite else ""))

    # 1. A flag gruda?
    passos = [p for v in eps.values() for p in v]
    flag = sum(p["ameaca_esq"] for p in passos)
    print(f"\n1. Flag esquerda acesa em {flag}/{len(passos)} steps ({100 * flag / max(1, len(passos)):.0f}%)")
    dur = [r[-1]["tempo"] - r[0]["tempo"] for v in eps.values() for r in corridas(v, "ameaca_esq")]
    if dur:
        dur.sort()
        print(f"   {len(dur)} trechos com a flag acesa; duração mediana {dur[len(dur) // 2]:.0f}s, "
              f"máxima {dur[-1]:.0f}s")
    # Dentro dos trechos de flag acesa: quanto tempo desde a última leitura com silhueta?
    idades = []
    for v in eps.values():
        ultima_silh = None
        for p in v:
            vendo = p["luz_esq"] and p["borda"] is not None and p["borda"] > 24 and p["silhueta"] < 27
            if vendo:
                ultima_silh = p["tempo"]
            if p["ameaca_esq"]:
                idades.append(None if ultima_silh is None else p["tempo"] - ultima_silh)
    if idades:
        sem = sum(1 for i in idades if i is None)
        recentes = sum(1 for i in idades if i is not None and i <= 15)
        print(f"   steps com flag acesa: {recentes} com a silhueta vista nos últimos 15s, "
              f"{len(idades) - recentes - sem} só com leitura mais antiga, {sem} sem silhueta "
              f"nenhuma no episódio (acesa pelo rosto ou herdada)")

    # 2/3. Antes de cada morte
    mortes = {k: v for k, v in eps.items() if v and v[-1]["fim"] == "morte"}
    cont = defaultdict(int)
    print(f"\n2. {len(mortes)} mortes:")
    for k, v in sorted(mortes.items()):
        ult = v[-2] if len(v) > 1 else v[-1]          # o step anterior ao jumpscare
        t_fim = v[-1]["tempo"]
        recente = [p for p in v if t_fim - p["tempo"] <= JANELA_S]
        abriu_com_flag = any(a["porta_esq"] == 1 and b["porta_esq"] == 0 and b["ameaca_esq"]
                             for a, b in zip(recente, recente[1:]))
        # Porta esquerda fechada na morte: o Bonnie (e o Foxy) não passam por ela → foi outro.
        # Com o Bonnie dentro da sala os botões param de funcionar, então o estado da porta no
        # step anterior ao jumpscare é o que valeu.
        tipo = "porta esq FECHADA (não foi Bonnie/Foxy)" if ult["porta_esq"] else "porta esq ABERTA"
        if abriu_com_flag:
            tipo += " | abriu a porta COM a flag acesa"
        tipo += " | flag " + ("acesa" if ult["ameaca_esq"] else "apagada")
        cont[tipo] += 1
        print(f"   ep {k:3d} noite {v[0]['noite']} {t_fim:5.0f}s energia {v[-1]['energia']:5.1f}% | {tipo}")
        if args.detalhe:
            for p in v[-args.detalhe:]:
                jan = f"bor{p['borda']:5.1f} sil{p['silhueta']:5.1f}" if p["borda"] is not None else "   (sem leitura)  "
                print(f"        {p['tempo']:5.1f}s {p['acao']:20s} portaE{p['porta_esq']} luzE{p['luz_esq']} "
                      f"cam{p['camera']} {jan} flagE{p['ameaca_esq']} portaD{p['porta_dir']} flagD{p['ameaca_dir']}")
    print("\n3. Resumo das mortes:")
    for tipo, n in sorted(cont.items(), key=lambda x: -x[1]):
        print(f"   {n:3d}  {tipo}")


if __name__ == "__main__":
    main()
