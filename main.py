import glob
import os
import re
import sys

from src.environment.fnaf_env import FNAFEnv, MAX_NOITE, FOXY_SATURACAO_S, INFO_SATURACAO_S


def _valor_flag(flag: str) -> str | None:
    """Valor de `--flag valor` em sys.argv (None se a flag faltar ou vier sem valor)."""
    if flag not in sys.argv:
        return None
    i = sys.argv.index(flag)
    valor = sys.argv[i + 1] if i + 1 < len(sys.argv) else None
    return None if valor is None or valor.startswith("--") else valor


def encontrar_ultimo_modelo() -> str | None:
    """Retorna o checkpoint mais avançado em modelos/.

    Preferência: maior número de steps no nome; senão, o .zip mais recente.
    (Média de pesos entre linhagens foi descontinuada: quebra a política.)
    """
    def extrair_steps(path):
        m = re.search(r"_(\d+)_steps\.zip$", path)
        return int(m.group(1)) if m else -1

    numerados = [p for p in glob.glob("modelos/*.zip") if extrair_steps(p) >= 0]
    if numerados:
        return max(numerados, key=extrair_steps)

    outros = glob.glob("modelos/*.zip")
    return max(outros, key=os.path.getctime) if outros else None


def modo_teste():
    print("Testando reset...")
    env = FNAFEnv()
    obs, _ = env.reset()
    print(f"Reset OK! Imagem: {obs['imagem'].shape}, Estados: {obs['estados'].shape}")
    print(f"Estados iniciais:")
    print(f"  - Porta esquerda: {obs['estados'][0]}")
    print(f"  - Porta direita: {obs['estados'][1]}")
    print(f"  - Luz esquerda: {obs['estados'][2]}")
    print(f"  - Luz direita: {obs['estados'][3]}")
    print(f"  - Câmera aberta: {obs['estados'][4]}")
    print(f"  - Câmera ativa: {obs['estados'][5]:.2f}")
    print(f"  - Energia: {obs['estados'][6]*100:.1f}%")
    print(f"  - Ameaça esquerda: {obs['estados'][8]}")
    print(f"  - Ameaça direita: {obs['estados'][9]}")
    print(f"  - Noite: {obs['estados'][10] * MAX_NOITE:.0f}")
    print(f"  - Tempo sem câmera: {obs['estados'][11] * FOXY_SATURACAO_S:.1f}s")
    print(f"  - Idade da info esq: {obs['estados'][12] * INFO_SATURACAO_S:.1f}s")
    print(f"  - Idade da info dir: {obs['estados'][13] * INFO_SATURACAO_S:.1f}s")
    input("O jogo iniciou a noite 1? (aperta Enter para confirmar)")
    env.close()


def modo_treino():
    from src.agent.train import treinar

    if "--novo" in sys.argv:
        print("Flag --novo: começando treino do zero (modelos antigos ignorados)")
        ultimo_modelo = None
    else:
        ultimo_modelo = encontrar_ultimo_modelo()
        if ultimo_modelo:
            print(f"Continuando treino: {ultimo_modelo}")
            print("(use 'python main.py treino --novo' para começar do zero)")
        else:
            print("Nenhum modelo encontrado — começando do zero")

    # BC warmstart (opcional): --bc <caminho.zip> inicializa a política a partir de um modelo
    # de BC. Combina com --novo (treino fresco). O BC precisa casar com USAR_LSTM (train.py).
    bc_path = _valor_flag("--bc")
    if bc_path:
        print(f"BC warmstart: inicializando a partir de {bc_path}")

    # --nome <tag>: sufixo descritivo da run nova (pasta do tensorboard e logs arquivados),
    # ex.: --nome bc_recorrente -> logs/tensorboard/2026-09-28_run5_rppo_bc_recorrente.
    treinar(timesteps=500_000, carregar_modelo=ultimo_modelo, bc_path=bc_path,
            tag_run=_valor_flag("--nome"))


def modo_bc():
    """Treina Behavioral Cloning a partir de datasets de gameplay humano gravados
    (src/utils/gravar_gameplay.py). Gera modelos/fnaf_bc.zip para warmstart do treino.
    NÃO precisa do jogo aberto (lê frames já gravados).
    Uso: python main.py bc [dataset.json ...]
         Sem argumentos, pega automaticamente TODOS os dados/*/dataset.json.

    O ALGORITMO segue a constante USAR_LSTM (de train.py, igual ao treino): True usa BC RECORRENTE
    (clona a própria LSTM+cabeças, transfere ~100%); False usa BC feedforward (só o extractor pra
    LSTM). O BC precisa CASAR com a run — senão o warmstart volta a semear só a percepção."""
    from src.agent.behavioral_cloning import treinar_bc, treinar_bc_recorrente
    from src.agent.train import USAR_LSTM as usar_lstm

    caminhos = [a for a in sys.argv[2:] if not a.startswith("--")]
    if not caminhos:
        caminhos = sorted(glob.glob("dados/*/dataset.json"))
        if caminhos:
            print(f"Nenhum caminho informado — usando os {len(caminhos)} datasets encontrados em dados/:")
            for caminho in caminhos:
                print(f"  {caminho}")
            print()

    if not caminhos:
        print("Uso: python main.py bc [dataset.json ...]")
        print("Sem argumentos, pega automaticamente todos os dados/*/dataset.json.")
        print("Grave demos antes com: python -m src.utils.gravar_gameplay --noite N")
        return

    if usar_lstm:
        print("[BC] USAR_LSTM=True -> BC RECORRENTE (clona a LSTM+cabeças).")
        treinar_bc_recorrente(caminhos)
    else:
        print("[BC] USAR_LSTM=False -> BC feedforward (transfere só o extractor pra LSTM).")
        treinar_bc(caminhos)


USO_JOGAR = ("python main.py jogar [--modelo <zip>] [--estocastico] [--noite N] "
              "[--episodios K] [--ablacao imagem|estados]")


def _resumo_avaliacao(eps: list[dict]) -> list[str]:
    """Linhas de resumo POR NOITE: vitórias, sobrevivência mediana, ações/min, s/step e causas.
    O s/step importa ao comparar PCs: o jogo é em tempo real, step mais lento = agente mais lento."""
    from collections import Counter
    from statistics import median
    linhas = []
    for noite in sorted({e["noite"] for e in eps}):
        grupo = [e for e in eps if e["noite"] == noite]
        v = sum(1 for e in grupo if e["resultado"] == "VITORIA")
        causas = ", ".join(f"{c} {n}" for c, n in Counter(e["causa"] for e in grupo).most_common(3))
        linhas.append(
            f"Noite {noite}: vitórias {v}/{len(grupo)} ({100 * v / len(grupo):.0f}%) | "
            f"sobrev mediana {median(e['tempo'] for e in grupo):.0f}s | "
            f"ações/min {sum(e['acoes_min'] for e in grupo) / len(grupo):.1f} | "
            f"s/step {sum(e['tempo'] for e in grupo) / max(1, sum(e['passos'] for e in grupo)):.2f} | "
            f"{causas}")
    return linhas


def modo_jogar():
    """Avalia um modelo SEM aprender (medições da Fase 1 em docs/ESTADO_ATUAL.md).

      --modelo <zip>   modelo a avaliar (default: o checkpoint mais avançado de modelos/).
                       Aceita o próprio modelos/fnaf_bc.zip → mede o BC PURO, sem RL.
      --estocastico    amostra a ação (como no TREINO) em vez do argmax. Use p/ comparar com
                       as taxas de vitória dos logs de treino, que são estocásticas.
      --noite N        mira a noite N (exige FNAF_RESET_METODO=continue): morrer na N → Continue.
      --episodios K    para após K episódios na noite alvo (ou K no total, sem --noite).
      --ablacao imagem|estados  zera aquele ramo da observação antes do predict (Decisão 5):
                       se zerar a IMAGEM não derruba a vitória, a CNN não está contribuindo.

    Cada episódio vai p/ logs/analise/avaliacoes.log (com a configuração no cabeçalho) e o
    desyncs da avaliação p/ logs/analise/avaliacoes_desyncs.log — nada disso se mistura aos
    logs de TREINO. Ctrl+C encerra e imprime o resumo por noite."""
    from datetime import datetime

    import numpy as np

    from src.environment.fnaf_env import RESET_METODO

    ablacao = _valor_flag("--ablacao")
    if "--ablacao" in sys.argv and ablacao not in ("imagem", "estados"):
        print("Uso:", USO_JOGAR)
        return
    estocastico = "--estocastico" in sys.argv
    noite_alvo = int(_valor_flag("--noite")) if _valor_flag("--noite") else None
    max_eps = int(_valor_flag("--episodios")) if _valor_flag("--episodios") else None

    caminho = _valor_flag("--modelo") or encontrar_ultimo_modelo()
    if not caminho or not os.path.exists(caminho):
        print(f"Modelo não encontrado: {caminho}. Treine primeiro: python main.py treino")
        return
    if noite_alvo and RESET_METODO != "continue":
        print(f"[aviso] --noite {noite_alvo} exige FNAF_RESET_METODO=continue no .env "
              f"(atual: {RESET_METODO}) — toda morte volta p/ a Noite 1.")

    # USAR_LSTM=True carrega RecurrentPPO e propaga o estado da LSTM (igual ao treino).
    from src.agent.train import USAR_LSTM as usar_lstm, _versao_codigo
    config = (f"modelo {caminho} | {'LSTM' if usar_lstm else 'PPO'} | "
              f"{'estocástico' if estocastico else 'determinístico'} | "
              f"noite alvo {noite_alvo or '-'} | ablação {ablacao or '-'} | "
              f"PC {os.getenv('PC') or '?'} | "
              f"commit {_versao_codigo()} | {datetime.now():%Y-%m-%d %H:%M}")
    print(f"Avaliação: {config}")
    # Sem VecNormalize aqui de propósito: o treino normaliza só a recompensa
    # (norm_obs=False), então a política vê a observação crua igual no treino.
    env = FNAFEnv()
    env._log_desyncs_path = "logs/analise/avaliacoes_desyncs.log"
    if noite_alvo:
        env.noite_desejada = noite_alvo
    if usar_lstm:
        from sb3_contrib import RecurrentPPO
        modelo = RecurrentPPO.load(caminho, env=env)
    else:
        from stable_baselines3 import PPO
        modelo = PPO.load(caminho, env=env)

    os.makedirs("logs/analise", exist_ok=True)
    log = open("logs/analise/avaliacoes.log", "a", encoding="utf-8")
    log.write(f"\n{'=' * 60}\nAvaliação iniciada\n{config}\n{'=' * 60}\n")

    eps: list[dict] = []
    try:
        while max_eps is None or sum(
                1 for e in eps if noite_alvo is None or e["noite"] == noite_alvo) < max_eps:
            obs, _ = env.reset()
            terminado = truncado = False
            info = {}
            n_acoes = 0
            lstm_states = None                          # estado da LSTM
            ep_start = np.ones((1,), dtype=bool)        # sinaliza início → a LSTM zera o estado

            while not (terminado or truncado):
                obs_pred = obs
                if ablacao:                       # zera o ramo só para o predict (Decisão 5)
                    obs_pred = dict(obs)
                    obs_pred[ablacao] = np.zeros_like(obs[ablacao])
                if usar_lstm:
                    acao, lstm_states = modelo.predict(
                        obs_pred, state=lstm_states, episode_start=ep_start,
                        deterministic=not estocastico)
                    ep_start = np.zeros((1,), dtype=bool)
                else:
                    acao, _ = modelo.predict(obs_pred, deterministic=not estocastico)
                obs, _, terminado, truncado, info = env.step(int(acao))
                if info.get("acao_nome", "nada") != "nada":
                    n_acoes += 1

            if info.get("interrompido"):
                resultado = "INTERROMPIDO"
            elif info.get("morreu"):
                resultado = "MORTE"
            elif info.get("vitoria"):
                resultado = "VITORIA"
            else:
                resultado = "TRUNCADO"
            tempo = info.get("tempo", 0.0)
            ep = {"noite": info.get("noite", 1), "resultado": resultado, "tempo": tempo,
                  "passos": info.get("passos", 0), "causa": info.get("causa") or "-",
                  "energia": info.get("energia", 0.0),
                  "acoes_min": n_acoes / (tempo / 60) if tempo > 0 else 0.0}
            eps.append(ep)
            linha = (f"Ep {len(eps):3d} | Noite {ep['noite']} | {resultado:12s} | "
                     f"sobrev {tempo:5.0f}s ({ep['passos']:4d}p) | energia {ep['energia']:5.1f}% | "
                     f"ações/min {ep['acoes_min']:5.1f} | causa {ep['causa']}")
            print(linha)
            log.write(linha + "\n")
            log.flush()
    except KeyboardInterrupt:
        print("\nAvaliação interrompida (Ctrl+C).")
    finally:
        env.close()
        if eps:
            resumo = _resumo_avaliacao(eps)
            print(f"\nResumo ({config}):")
            for linha in resumo:
                print("  " + linha)
            log.write("Resumo:\n" + "".join(f"  {l}\n" for l in resumo))
        log.write("Avaliação finalizada\n")
        log.close()


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "teste"

    if modo == "teste":
        modo_teste()
    elif modo == "treino":
        modo_treino()
    elif modo == "jogar":
        modo_jogar()
    elif modo == "bc":
        modo_bc()
    else:
        print(f"Modo desconhecido: {modo}")
        print("Use: python main.py teste | "
              "python main.py treino [--novo] [--bc <zip>] [--nome <tag>] | "
              f"{USO_JOGAR} | "
              "python main.py bc [dataset.json ...]  (sem args: pega dados/*/dataset.json)")
