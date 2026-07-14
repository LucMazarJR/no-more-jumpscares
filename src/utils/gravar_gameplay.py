"""Grava gameplay humano para Behavioral Cloning (dataset.json + frames 84x84).

Uso:
    python -m src.utils.gravar_gameplay --noite N
        N = noite que VOCÊ vai jogar nesta sessão (1..7). Uma noite por sessão — o
        número vai em cada registro e vira o estado `noite` que a política enxerga.

Fluxo: rode o comando, navegue no jogo até a noite carregar, aperte F9 para começar a
gravar, jogue até o 6AM (pelas TECLAS abaixo, não pelo mouse!) e aperte F10 para parar.

FIDELIDADE POR CONSTRUÇÃO: o executor das ações é o PRÓPRIO FNAFEnv (um env "fantasma",
cujo __init__ só carrega templates — não abre nem reseta o jogo). Cada tecla vira uma
chamada a `env._executar_acao`, que aplica a MESMA coreografia do treino: delay de virada
de cabeça (SIDE_SWITCH_DELAY) antes de clicar do outro lado, delay de saída de câmera,
gesto de hover da prancheta, cooldowns de porta/câmera, pré-leitura e verificação do
clique de porta pela cor do botão. O estado gravado é o estado do env — não existe mais
uma cópia paralela da mecânica para dessincronizar. Também espelhado do step() do RL:
  • luz ligada = botão SEGURADO durante a captura (corredor aceso NA IMAGEM, como o RL
    vê — e requisito da detecção da Chica) + re-sync passivo da porta do mesmo lado;
  • energia simulada re-ancorada pela FOTO (`_ler_energia` + validar_leitura_energia);
  • ameaças rotuladas por `_atualizar_ameaca` (mesmos templates/debounce do treino);
  • rotulagem SEM shift: registro = (frame_t, ação decidida VENDO frame_t).

THREADING: handlers do `keyboard` rodam na thread de hook e o mss/pyautogui não são
thread-safe — o handler só ENFILEIRA a tecla; o loop principal drena e executa no máximo
1 ação por iteração, ANTES de capturar o próximo frame (mesma semântica do step do RL).
"""
import cv2
import json
import os
import sys
import time
from collections import deque
from datetime import datetime

import keyboard
import pygetwindow as gw

from src.utils.capture import melhor_janela, regiao_cliente
from src.environment.fnaf_env import (
    FNAFEnv, ACOES, COORDS, STEP_DELAY, WINDOW_TITLE, MAX_NOITE,
    validar_leitura_energia,
)

# ─── Mapeamento tecla → ação ──────────────────────────────────────
TECLAS = {
    "a":   "porta_esquerda",
    "d":   "porta_direita",
    "q":   "luz_esquerda",
    "e":   "luz_direita",
    "tab": "abrir_fechar_camera",
    "1":   "camera_1a",
    "2":   "camera_1b",
    "3":   "camera_1c",
    "4":   "camera_2a",
    "5":   "camera_2b",
    "6":   "camera_3",
    "7":   "camera_4a",
    "8":   "camera_4b",
    "9":   "camera_5",
    "0":   "camera_6",
    "-":   "camera_7",
}

# Intervalo alvo do loop (segundos). Mais rápido que o step do RL (~0.7s) de propósito:
# rende mais rótulos das ações RARAS (portas/luzes); o desbalanceio de classes é corrigido
# no treino do BC (WeightedRandomSampler). Iterações com ação de troca de lado estouram o
# período (delays da coreografia) — inofensivo: energia/tempo correm em wall-clock.
_LOOP_DT = 0.25


def acao_para_numero(nome_acao: str) -> int:
    for numero, nome in ACOES.items():
        if nome == nome_acao:
            return numero
    return 0


def _noite_da_cli() -> int:
    """--noite N obrigatório: cada sessão grava UMA noite, e o número vai no dataset.
    Sem ele, o BC rotularia tudo como Noite 1 (dado.get("noite", 1)) e ensinaria a
    política a IGNORAR o estado de dificuldade."""
    if "--noite" in sys.argv:
        i = sys.argv.index("--noite")
        try:
            n = int(sys.argv[i + 1])
            if 1 <= n <= MAX_NOITE:
                return n
        except (IndexError, ValueError):
            pass
    print("Uso: python -m src.utils.gravar_gameplay --noite N   (N = 1..7)")
    print("Grave UMA noite por sessão — o número rotula cada frame do dataset.")
    raise SystemExit(1)


def gravar():
    noite = _noite_da_cli()
    pasta = f"gameplay_{datetime.now().strftime('%Y%m%d_%H%M%S')}_noite{noite}"
    os.makedirs(f"dados/{pasta}/frames", exist_ok=True)

    # Env fantasma = EXECUTOR das ações + fonte de estado + visão (ameaça/energia).
    # O __init__ do FNAFEnv só carrega templates e monta espaços — não abre o jogo.
    env = FNAFEnv()

    print(f"=== Gravação da NOITE {noite} ===")
    print("Mapeamento de teclas (jogue por elas, não pelo mouse):")
    for tecla, acao in TECLAS.items():
        print(f"  [{tecla}] → {acao}")
    print("\nAs ações têm os MESMOS delays do treino (virada de lado ~0.85s, saída de")
    print("câmera ~0.65s, cooldowns) — o clique acontece quando o jogo aceita, não na tecla.")
    print("\nNavegue no jogo até a noite carregar e aperte [F9] para COMEÇAR a gravar.")
    print("[F10] → PARA a gravação (se o jogo fechar sozinho — Golden Freddy — aperte F10;")
    print("        a sessão parcial continua válida para o BC).\n")

    dados     = []
    frame_idx = 0
    registro_pendente = None
    fila: deque[str] = deque()

    # Handler roda na thread de hook do keyboard: SÓ enfileira (deque.append é atômico).
    # Dedupe do topo absorve o auto-repeat de tecla segurada; os cooldowns do env seguram
    # o resto (ação em cooldown volta inválida SEM clicar, igual ao treino).
    def fazer_handler(nome_acao):
        def handler(_event):
            if not (fila and fila[-1] == nome_acao):
                fila.append(nome_acao)
        return handler

    keyboard.wait("f9")            # hooks só DEPOIS do F9: menu/carregamento não contamina
    hooks = [keyboard.on_press_key(tecla, fazer_handler(nome))
             for tecla, nome in TECLAS.items()]

    # Espelha o fim do reset() do env: relógios ancorados no início da noite.
    t0 = time.perf_counter()
    env.episode_start_time = t0
    env._t_ultima_energia  = t0
    env._t_ultima_camera   = t0
    env._t_confirmacao_esq = t0   # idade da informação (estados 13-14) parte de "fresco"
    env._t_confirmacao_dir = t0
    print(">>> Gravando! Jogue até o 6AM e aperte F10. <<<\n")

    try:
        while not keyboard.is_pressed("f10"):
            t_iter = time.perf_counter()

            # (a) Drena e executa NO MÁXIMO 1 ação — pelo executor do env (coreografia,
            # cooldowns e estado idênticos ao treino). "nada" também passa pelo executor
            # p/ manter a semântica de ultima_acao/saída de câmera igual ao RL.
            nome_acao = fila.popleft() if fila else "nada"
            acao_valida = env._executar_acao(acao_para_numero(nome_acao))
            if env._pixel_antes_porta is not None:
                # Igual ao step(): espera a animação e confirma o clique pela cor do botão.
                time.sleep(STEP_DELAY)
                if not env._verificar_botao_porta(nome_acao):
                    acao_valida = False
                env._pixel_antes_porta = None
            if nome_acao != "nada":
                print(f"  {nome_acao:<20} [{'OK' if acao_valida else 'X '}] | "
                      f"E:{env.energia:5.1f}% cam:{int(env.camera_aberta)} "
                      f"porta:{int(env.porta_esq)}/{int(env.porta_dir)}")

            # (b) Fecha o registro do frame ANTERIOR com a ação decidida VENDO aquele
            # frame — a convenção (obs_t, ação_t) do MDP que o BC clona.
            if registro_pendente is not None:
                registro_pendente["acao"] = acao_para_numero(nome_acao)
                registro_pendente["nome"] = nome_acao
                dados.append(registro_pendente)
                registro_pendente = None

            if not gw.getWindowsWithTitle(WINDOW_TITLE):
                print("Jogo não encontrado! Aguardando... (F10 encerra)")
                time.sleep(1)
                continue

            # (c) Luz ligada = botão SEGURADO durante a captura (igual ao step do RL) +
            # re-sync passivo da porta do mesmo lado pela cor do botão.
            luz_segurada = None
            if env.luz_esq:
                luz_segurada = COORDS["luz_esquerda"]
                env.capture.segurar_botao(*luz_segurada)
                env._sync_porta_por_pixel("porta_esquerda")
            elif env.luz_dir:
                luz_segurada = COORDS["luz_direita"]
                env.capture.segurar_botao(*luz_segurada)
                env._sync_porta_por_pixel("porta_direita")

            # (d) Captura a ÁREA CLIENTE → referência 1280x720 → frame 84x84 do dataset.
            frame = env.capture.capturar_tela(regiao_cliente(melhor_janela(WINDOW_TITLE)))
            frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_ref   = cv2.resize(frame_cinza, env._ref_size)
            frame_pequeno = cv2.resize(frame_ref, (84, 84))

            # (e) Ameaças pela implementação do treino (estado HELD dentro do env).
            env._atualizar_ameaca(frame_ref)

            # (f) Solta a luz depois da captura (igual ao step).
            if luz_segurada is not None:
                env.capture.soltar_botao(*luz_segurada)

            # (g) Energia: simulação wall-clock re-ancorada pela FOTO (photo-primary).
            env._atualizar_energia()
            env.energia = validar_leitura_energia(env._ler_energia(frame_ref), env.energia)

            # (h) Relógios e cooldowns (0.25s/tick — o dimensionamento original dos
            # cooldowns, que cobrem as animações de ~0.75-1.0s).
            env._atualizar_cooldowns()
            env._atualizar_tempo()
            if env.camera_aberta:
                env._t_ultima_camera = time.perf_counter()

            # (i) Registro pendente lido DIRETO do env (campos idênticos aos do RL).
            caminho_frame = f"dados/{pasta}/frames/{frame_idx:06d}.png"
            cv2.imwrite(caminho_frame, frame_pequeno)
            registro_pendente = {
                "frame":            caminho_frame,
                "noite":            noite,
                "porta_esq":        int(env.porta_esq),
                "porta_dir":        int(env.porta_dir),
                "luz_esq":          int(env.luz_esq),
                "luz_dir":          int(env.luz_dir),
                "camera_aberta":    int(env.camera_aberta),
                "camera_ativa":     env.camera_ativa,
                "energia":          round(env.energia, 2),
                "tempo_ep":         round(min(env.tempo_jogo, 535.0), 2),
                "tempo_sem_camera": round(env._tempo_sem_camera(), 2),
                "idade_info_esq":   round(env._idade_info("esq"), 2),  # run 3: idade da info
                "idade_info_dir":   round(env._idade_info("dir"), 2),  # (13º-14º estados)
                "ameaca_esq":       int(env.ameaca_esq),
                "ameaca_dir":       int(env.ameaca_dir),
            }
            frame_idx += 1

            # (j) Compasso do loop (captura/coreografia têm custo variável).
            time.sleep(max(0.0, _LOOP_DT - (time.perf_counter() - t_iter)))

    finally:
        keyboard.unhook_all()

    # Último frame pendente: fecha com a próxima ação da fila (ou nada)
    if registro_pendente is not None:
        nome_acao = fila.popleft() if fila else "nada"
        registro_pendente["acao"] = acao_para_numero(nome_acao)
        registro_pendente["nome"] = nome_acao
        dados.append(registro_pendente)

    # Salva dataset
    caminho_json = f"dados/{pasta}/dataset.json"
    with open(caminho_json, "w") as f:
        json.dump(dados, f, indent=2)

    # Resumo
    from collections import Counter
    contagem = Counter(d["nome"] for d in dados)

    print(f"\n=== Gravação finalizada! ===")
    print(f"Noite: {noite} | Total de frames: {frame_idx}")
    print(f"Dataset salvo em: {caminho_json}")
    print(f"\nDistribuição de ações:")
    for nome, qtd in contagem.most_common():
        print(f"  {nome}: {qtd}")


if __name__ == "__main__":
    gravar()
