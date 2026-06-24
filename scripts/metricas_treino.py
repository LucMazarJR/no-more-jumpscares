#!/usr/bin/env python3
"""Metricas POR NOITE do treino, lidas de logs/treino.log.

A "Taxa vitoria" que aparece no log do treino e AGREGADA (todas as noites juntas) e engana
para decisoes de curriculo: conforme o agente domina a noite 1, ele chega mais na 2 e morre
mais la, entao o numero agregado estagna (ou cai) mesmo com progresso real. Este script quebra
as metricas POR NOITE numa janela movel — e o que se usa pra decidir o proximo passo.

A recomendacao se adapta ao FNAF_RESET_METODO (lido do .env, ou via --metodo):
  • new_game = curriculo classico: avalia a "frente" (noite mais avancada) vs a anterior.
  • continue = mira FNAF_NOITE_DESEJADA: avalia a NOITE ALVO (ignora o overshoot D+1, que e
    forcado ao vencer a alvo) e sugere subir/baixar a alvo ou onde o climb empaca.

Uso:
    python scripts/metricas_treino.py
    python scripts/metricas_treino.py --janela 150
    python scripts/metricas_treino.py --metodo continue --noite-desejada 4
    python scripts/metricas_treino.py --log logs/treino.log --janela 200

Ver docs/MONITORAMENTO_TREINO.md para a interpretacao e as regras de decisao.
"""
from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

MAX_NOITE = 7  # FNAF1: noites 1-6 + custom (espelha src/environment/fnaf_env.py)

# Casa as linhas de episodio do LogCallback (train.py). Linhas OCORRIDO/cabecalho nao casam.
LINHA = re.compile(
    r"Ep\s+(\d+)\s+\|\s+Noite\s+(\d+)\s+\|\s+(VITORIA|MORTE|INTERROMPIDO)\s+\|"
    r".*?Tempo:\s+([\d.]+)\s+min"
)

# Heuristicas da regra de decisao (ver doc). Pontos de partida, calibre olhando seus dados.
P_ABASTECE   = 50.0   # win rate da noite anterior (%) que considera a frente "abastecida"
N_FAMINTA    = 25     # nº de eps na noite-frente/alvo abaixo disto = faminta de amostras
P_PRESO      = 10.0   # win rate da frente/alvo (%) abaixo disto = ainda nao destravou
P_DOMINA     = 70.0   # win rate da alvo (%) acima disto = dominada (pode subir a alvo)
PLATEAU_MIN  = 0.5    # ganho de sobrevivencia (min) entre metades abaixo disto = plateau


def _ler_env_arquivo(caminho: Path) -> dict[str, str]:
    """Le KEY=VALUE de um .env (ignora comentarios/linhas vazias). Sem dependencias."""
    valores: dict[str, str] = {}
    if not caminho.exists():
        return valores
    for linha in caminho.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = linha.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        chave, valor = s.split("=", 1)
        valores[chave.strip()] = valor.strip().strip('"').strip("'")
    return valores


def carregar_config(metodo_cli: str | None, noite_cli: int | None,
                    env_path: Path) -> tuple[str, int]:
    """Resolve (metodo, noite_desejada): CLI > variavel de ambiente > .env > default."""
    env = _ler_env_arquivo(env_path)

    def get(chave: str) -> str | None:
        return os.environ.get(chave) or env.get(chave)

    metodo = (metodo_cli or get("FNAF_RESET_METODO") or "new_game").strip().lower()

    if noite_cli is not None:
        noite = noite_cli
    else:
        bruto = get("FNAF_NOITE_DESEJADA")
        try:
            noite = int(bruto) if bruto not in (None, "") else 1
        except ValueError:
            noite = 1
    return metodo, max(1, min(noite, MAX_NOITE))


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


def win_rate(d: dict) -> tuple[int, float]:
    """(eps validos, win rate %) de um resumo de noite. validos = vitorias + mortes."""
    validos = d["vit"] + d["mortes"]
    return validos, (d["vit"] / validos * 100) if validos else 0.0


def imprimir(por: dict, total: int, janela: int, noite_alvo: int | None = None) -> None:
    print(f"\nJanela: ultimos {min(total, janela)} de {total} episodios registrados\n")
    print(f"{'Noite':>5} | {'n':>4} | {'vit%':>6} | {'vit':>4} {'morte':>5} {'int':>4} | "
          f"{'t_morte(min)':>12} | {'tendencia':>9}")
    print("-" * 72)
    for noite in sorted(por):
        d = por[noite]
        validos, p = win_rate(d)
        tmed = media(d["t_mortes"])
        tend = tendencia(d["t_mortes"])
        tend_str = "-" if tend is None else f"{tend:+.1f}"
        marca = ""
        if noite_alvo is not None:
            if noite == noite_alvo:
                marca = "  <- alvo"
            elif noite > noite_alvo:
                marca = "  ~ overshoot"
        print(f"{noite:>5} | {validos:>4} | {p:>5.1f}% | {d['vit']:>4} {d['mortes']:>5} "
              f"{d['int']:>4} | {tmed:>12.1f} | {tend_str:>9}{marca}")


def _gargalo_climb(por: dict, alvo: int, min_reps: int = 8) -> tuple[int, float, int] | None:
    """Noite < alvo com MENOR win rate (e reps suficientes) — onde o climb ate a alvo empaca."""
    candidatos = []
    for n in range(1, alvo):
        d = por.get(n)
        if not d:
            continue
        validos, p = win_rate(d)
        if validos >= min_reps:
            candidatos.append((p, n, validos))
    if not candidatos:
        return None
    p, n, validos = min(candidatos)            # menor win rate primeiro
    return n, p, validos


def recomendar_curriculo(por: dict) -> None:
    """Modo new_game: avalia a frente (noite mais avancada) vs a anterior (que abastece)."""
    nights = sorted(por)
    if not nights:
        return
    frente = nights[-1]                     # noite mais avancada alcancada = a "frente"
    if frente == 1:
        print("\n>> Ainda preso na noite 1. Metrica de currículo so vale quando a frente chega "
              "na noite 2+. Foque em destravar a noite 1.")
        return

    validos_f, p_f = win_rate(por[frente])
    tend_f = tendencia(por[frente]["t_mortes"])
    _, p_ant = win_rate(por[frente - 1])

    abastece = p_ant >= P_ABASTECE
    faminta  = validos_f < N_FAMINTA
    preso    = p_f < P_PRESO
    plateau  = tend_f is not None and tend_f < PLATEAU_MIN

    print(f"\nFrente = noite {frente} | noite {frente-1} (abastece): {p_ant:.1f}% | "
          f"frente: {p_f:.1f}% em {validos_f} eps | "
          f"tendencia frente: {'-' if tend_f is None else f'{tend_f:+.1f} min'}")

    if abastece and faminta and preso and plateau:
        print(f">> SUGESTAO: trocar TEMPORARIAMENTE para FNAF_RESET_METODO=continue e "
              f"FNAF_NOITE_DESEJADA={frente}. A noite {frente-1} abastece (>= {P_ABASTECE:.0f}%), "
              f"mas a frente esta faminta (< {N_FAMINTA} eps), travada (< {P_PRESO:.0f}%) e em "
              f"plateau. continue da mais reps na frente. Volte pra new_game quando p{frente} "
              f"encostar (~20-30%).")
    elif abastece and not plateau:
        print(">> MANTER new_game: a frente ainda esta APRENDENDO (sobrevivencia subindo). "
              "Trocar agora seria precipitado.")
    elif not abastece:
        print(f">> MANTER new_game: a noite {frente-1} ainda nao abastece bem a frente "
              f"({p_ant:.1f}% < {P_ABASTECE:.0f}%). O gargalo e a noite {frente-1}, nao a frente.")
    else:
        print(">> Sinais mistos — decida olhando a tabela (a heuristica nao bateu todos os "
              "criterios). Ver docs/MONITORAMENTO_TREINO.md.")


def recomendar_targeting(por: dict, alvo: int) -> None:
    """Modo continue: avalia a NOITE ALVO (FNAF_NOITE_DESEJADA), ignorando o overshoot."""
    nights = sorted(por)
    print(f"\nModo continue — mirando a noite {alvo} (FNAF_NOITE_DESEJADA).")

    overshoot = [n for n in nights if n > alvo]
    if overshoot:
        reps = sum(win_rate(por[n])[0] for n in overshoot)
        print(f"   noites {overshoot} = overshoot (vencer {alvo} obriga jogar {alvo+1}+ ate "
              f"morrer): {reps} eps esperados, fora da analise.")

    d_alvo = por.get(alvo)
    validos = win_rate(d_alvo)[0] if d_alvo else 0

    if validos == 0:
        garg = _gargalo_climb(por, alvo)
        if garg:
            n, p, v = garg
            print(f">> Nao chegou na noite {alvo} nesta janela. O climb empaca na noite {n} "
                  f"({p:.1f}% em {v} eps). Deixe rodar mais, ou baixe FNAF_NOITE_DESEJADA "
                  f"ate destravar essa noite.")
        else:
            print(f">> Nao chegou na noite {alvo} nesta janela e ha poucos eps no climb. "
                  f"Deixe rodar mais ou baixe FNAF_NOITE_DESEJADA temporariamente.")
        return

    p_alvo = win_rate(d_alvo)[1]
    tend = tendencia(d_alvo["t_mortes"])
    tmed = media(d_alvo["t_mortes"])
    print(f"Noite {alvo}: {p_alvo:.1f}% em {validos} eps | t_morte medio {tmed:.1f} min | "
          f"tendencia {'-' if tend is None else f'{tend:+.1f} min'}")

    faminta = validos < N_FAMINTA
    domina  = p_alvo >= P_DOMINA
    preso   = p_alvo < P_PRESO
    aprende = tend is not None and tend >= PLATEAU_MIN
    plateau = tend is not None and tend < PLATEAU_MIN

    if domina and not faminta:
        prox = min(alvo + 1, MAX_NOITE)
        if prox > alvo:
            print(f">> SUGESTAO: noite {alvo} dominada (>= {P_DOMINA:.0f}%). Suba "
                  f"FNAF_NOITE_DESEJADA={prox} para avancar o foco.")
        else:
            print(f">> Noite {alvo} dominada e ja e a ultima ({MAX_NOITE}). Objetivo cumprido.")
    elif faminta:
        garg = _gargalo_climb(por, alvo)
        extra = f" Gargalo do climb: noite {garg[0]} ({garg[1]:.1f}%)." if garg else ""
        print(f">> Noite {alvo} faminta (< {N_FAMINTA} eps): poucas reps chegam ate ela "
              f"(o climb e caro).{extra} Deixe rodar mais ou baixe FNAF_NOITE_DESEJADA p/ "
              f"acumular reps na alvo.")
    elif aprende:
        print(f">> MANTER: noite {alvo} ainda APRENDENDO (sobrevivencia subindo {tend:+.1f} min). "
              f"Deixe o treino seguir.")
    elif preso and plateau:
        print(f">> Noite {alvo} TRAVADA (< {P_PRESO:.0f}%) e em PLATEAU. Pode estar dificil "
              f"demais: considere baixar FNAF_NOITE_DESEJADA, treinar mais a base, ou revisar "
              f"recompensa/estrategia.")
    else:
        print(">> Sinais mistos — decida olhando a tabela. Ver docs/MONITORAMENTO_TREINO.md.")


def recomendar(por: dict, metodo: str, noite_desejada: int) -> None:
    if metodo == "continue":
        recomendar_targeting(por, noite_desejada)
    else:
        recomendar_curriculo(por)


def main() -> None:
    parser = argparse.ArgumentParser(description="Metricas por noite do treino FNAF.")
    parser.add_argument("--log", type=Path, default=Path("logs/treino.log"))
    parser.add_argument("--janela", type=int, default=150, help="ultimos N episodios (padrao 150)")
    parser.add_argument("--metodo", choices=["new_game", "continue"],
                        help="sobrescreve FNAF_RESET_METODO do .env")
    parser.add_argument("--noite-desejada", type=int,
                        help="sobrescreve FNAF_NOITE_DESEJADA do .env (modo continue)")
    parser.add_argument("--env", type=Path, default=Path(".env"),
                        help="caminho do .env p/ ler metodo/noite (padrao .env)")
    args = parser.parse_args()

    if not args.log.exists():
        raise SystemExit(f"Log nao encontrado: {args.log}")

    metodo, noite_desejada = carregar_config(args.metodo, args.noite_desejada, args.env)

    eps = parse(args.log)
    if not eps:
        raise SystemExit(f"Nenhum episodio reconhecido em {args.log}.")

    janela = eps[-args.janela:]
    por = resumo_por_noite(janela)
    alvo = noite_desejada if metodo == "continue" else None
    imprimir(por, total=len(eps), janela=args.janela, noite_alvo=alvo)

    cfg = f"FNAF_RESET_METODO={metodo}"
    if metodo == "continue":
        cfg += f" | FNAF_NOITE_DESEJADA={noite_desejada}"
    print(f"\nConfig: {cfg}")
    recomendar(por, metodo, noite_desejada)


if __name__ == "__main__":
    main()
