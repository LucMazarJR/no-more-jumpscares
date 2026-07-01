"""Grava gameplay humano para Behavioral Cloning (dataset.json + frames 84x84).

Uso:
    python -m src.utils.gravar_gameplay --noite N
        N = noite que VOCÊ vai jogar nesta sessão (1..7). Uma noite por sessão — o
        número vai em cada registro e vira o estado `noite` que a política enxerga.

Fluxo: rode o comando, navegue no jogo até a noite carregar, aperte F9 para começar a
gravar, jogue até o 6AM (pelas TECLAS abaixo, não pelo mouse!) e aperte F10 para parar.

O gravador espelha o pipeline do FNAFEnv de ponta a ponta, senão o BC aprende de uma
distribuição que o RL nunca vê:
  • captura a ÁREA CLIENTE da janela e redimensiona p/ a referência 1280x720 — mesma
    região/escala de _capturar_janela (a janela inteira incluía barra de título/bordas);
  • ameaças (Bonnie/Chica) rotuladas pela MESMA implementação do env (_atualizar_ameaca
    de um FNAFEnv "fantasma" — init só carrega templates, não abre nem controla o jogo);
  • tempo_sem_camera em segundos reais (12º estado da observação);
  • rotulagem SEM shift: cada registro é (frame_t, ação decidida VENDO frame_t) — a ação
    apertada entre a captura t e a t+1 fecha o registro t (antes a ação era associada ao
    frame capturado DEPOIS do efeito dela, invertendo a causalidade que o BC deve clonar).
"""
import cv2
import json
import os
import sys
import time
from datetime import datetime

import keyboard
import pyautogui
import pygetwindow as gw

from src.utils.capture import GameCapture, melhor_janela, regiao_cliente
from src.environment.fnaf_env import (
    FNAFEnv, COORDS, ACOES, LADO_POR_ACAO, SIDE_SWITCH_DELAY, WINDOW_TITLE, MAX_NOITE,
)

cap = GameCapture()

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

# ─── Índice de câmera — espelha a fórmula do env (acao_num - 5) ──
CAMERA_ID = {nome: num - 5 for num, nome in ACOES.items() if nome.startswith("camera_")}

# Intervalo de captura do loop (segundos). Mais rápido que o step do RL (~0.7s) de
# propósito: rende mais rótulos das ações RARAS (portas/luzes); o desbalanceio de
# classes resultante é corrigido no treino do BC (WeightedRandomSampler).
_LOOP_DT = 0.25

# Delay de virada de cabeça em frames de gravação (espelha SIDE_SWITCH_DELAY do env).
# Esse delay também é o único bloqueio para re-pressionamento — não há cooldown separado.
_PORTA_DELAY_FRAMES = max(1, round(SIDE_SWITCH_DELAY / _LOOP_DT))


class EstadoJogo:
    """Máquina de estado que espelha o FNAFEnv para registrar os estados internos
    (portas, luzes, câmera, energia, tempo, tempo_sem_camera) ao lado de cada frame.

    Portas: se o personagem precisa virar a câmera para o lado da porta
    (SIDE_SWITCH_DELAY), o estado só é atualizado após _PORTA_DELAY_FRAMES
    iterações. Enquanto essa mudança está pendente, novos presses são ignorados.
    O delay em si é o único bloqueio — não há cooldown separado por steps.
    """

    def __init__(self):
        self.porta_esq:     bool  = False
        self.porta_dir:     bool  = False
        self.luz_esq:       bool  = False
        self.luz_dir:       bool  = False
        self.camera_aberta: bool  = False
        self.camera_ativa:  int   = 0       # 0 = nenhuma; 1–11 = câmera selecionada
        self.energia:       float = 100.0
        self._inicio:       float = 0.0
        self._t_ultima_camera: float = 0.0  # base do tempo_sem_camera (12º estado)
        self.lado_atual:    str   = "centro"
        # Cada item: [frames_restantes, "esq"|"dir", novo_valor]
        # A existência de uma pendência para um lado bloqueia novos presses daquele lado.
        self._pendencias:   list  = []

    def iniciar(self) -> None:
        self._inicio = time.perf_counter()
        self._t_ultima_camera = self._inicio   # noite começa "recém-checada", igual ao env

    def _agendar_porta(self, lado: str, novo_valor: bool, delay: int) -> None:
        if delay <= 0:
            if lado == "esq":
                self.porta_esq = novo_valor
            else:
                self.porta_dir = novo_valor
        else:
            self._pendencias.append([delay, lado, novo_valor])

    def ao_pressionar(self, nome_acao: str) -> None:
        lado_alvo = LADO_POR_ACAO.get(nome_acao)

        if nome_acao == "porta_esquerda":
            # Bloqueado apenas enquanto há uma virada de cabeça pendente para esse lado
            if any(p[1] == "esq" for p in self._pendencias):
                return
            delay = _PORTA_DELAY_FRAMES if (lado_alvo and self.lado_atual != lado_alvo) else 0
            self._agendar_porta("esq", not self.porta_esq, delay)

        elif nome_acao == "porta_direita":
            if any(p[1] == "dir" for p in self._pendencias):
                return
            delay = _PORTA_DELAY_FRAMES if (lado_alvo and self.lado_atual != lado_alvo) else 0
            self._agendar_porta("dir", not self.porta_dir, delay)

        elif nome_acao == "luz_esquerda":
            if self.luz_esq:
                self.luz_esq = False
            else:
                self.luz_esq = True
                self.luz_dir = False          # mutuamente exclusivo

        elif nome_acao == "luz_direita":
            if self.luz_dir:
                self.luz_dir = False
            else:
                self.luz_dir = True
                self.luz_esq = False          # mutuamente exclusivo

        elif nome_acao == "abrir_fechar_camera":
            self.camera_aberta = not self.camera_aberta
            if self.camera_aberta:
                self.luz_esq = False
                self.luz_dir = False
            else:
                self.camera_ativa = 0

        elif nome_acao in CAMERA_ID:
            if self.camera_aberta:
                self.camera_ativa = CAMERA_ID[nome_acao]

        # O personagem começa a virar no momento do clique
        if lado_alvo:
            self.lado_atual = lado_alvo

    def atualizar(self) -> None:
        """Chama a cada iteração do loop — processa pendências de porta e energia."""
        proximas = []
        for p in self._pendencias:
            p[0] -= 1
            if p[0] <= 0:
                if p[1] == "esq":
                    self.porta_esq = p[2]
                else:
                    self.porta_dir = p[2]
            else:
                proximas.append(p)
        self._pendencias = proximas

        if self.camera_aberta:
            self._t_ultima_camera = time.perf_counter()   # mesma regra do env

        itens = (
            int(self.porta_esq) + int(self.porta_dir)
            + int(self.luz_esq) + int(self.luz_dir)
            + int(self.camera_aberta)
        )
        consumo = (0.104 + min(itens, 3) * 0.100) * _LOOP_DT
        self.energia = max(0.0, self.energia - consumo)

    def tempo_ep(self) -> float:
        return time.perf_counter() - self._inicio

    def tempo_sem_camera(self) -> float:
        """Segundos reais desde a última câmera aberta (0 se aberta) — espelha o env."""
        if self.camera_aberta:
            return 0.0
        return time.perf_counter() - self._t_ultima_camera

    def como_dict(self) -> dict:
        return {
            "porta_esq":        int(self.porta_esq),
            "porta_dir":        int(self.porta_dir),
            "luz_esq":          int(self.luz_esq),
            "luz_dir":          int(self.luz_dir),
            "camera_aberta":    int(self.camera_aberta),
            "camera_ativa":     self.camera_ativa,
            "energia":          round(self.energia, 2),
            "tempo_ep":         round(min(self.tempo_ep(), 535.0), 2),
            "tempo_sem_camera": round(self.tempo_sem_camera(), 2),
        }

def acao_para_numero(nome_acao: str) -> int:
    for numero, nome in ACOES.items():
        if nome == nome_acao:
            return numero
    return 0

def executar_acao_no_jogo(nome_acao: str):
    """Executa a ação no jogo — clique simples ou arrasto para câmera."""
    if nome_acao == "abrir_fechar_camera":
        # Câmera precisa de clique e arrasto para cima
        x, y = COORDS["abrir_fechar_camera"]
        pyautogui.mouseDown(x, y)
        time.sleep(0.05)
        pyautogui.moveTo(x, y - 200, duration=0.2)  # arrasta para cima
        pyautogui.mouseUp()

    elif nome_acao in COORDS:
        x, y = COORDS[nome_acao]
        pyautogui.click(x, y)

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

    # Env "fantasma" p/ rotular ameaças com a MESMA implementação/templates do treino.
    # O __init__ do FNAFEnv só carrega templates e monta espaços — não abre o jogo.
    env_cv = FNAFEnv()

    print(f"=== Gravação da NOITE {noite} ===")
    print("Mapeamento de teclas (jogue por elas, não pelo mouse):")
    for tecla, acao in TECLAS.items():
        print(f"  [{tecla}] → {acao}")
    print("\nNavegue no jogo até a noite carregar e aperte [F9] para COMEÇAR a gravar.")
    print("[F10] → PARA a gravação (se o jogo fechar sozinho — Golden Freddy — aperte F10;")
    print("        a sessão parcial continua válida para o BC).\n")

    estado    = EstadoJogo()
    dados     = []
    frame_idx = 0
    acao_atual = "nada"
    registro_pendente = None

    # Registra handlers para cada tecla
    def fazer_handler(nome_acao):
        def handler(event):
            nonlocal acao_atual
            acao_atual = nome_acao
            estado.ao_pressionar(nome_acao)
            executar_acao_no_jogo(nome_acao)
            print(f"  [{event.name}] → {nome_acao} | E:{estado.energia:.1f}% "
                  f"cam:{estado.camera_aberta} porta:{int(estado.porta_esq)}/{int(estado.porta_dir)}")
        return handler

    keyboard.wait("f9")            # hooks só DEPOIS do F9: menu/carregamento não contamina
    hooks = []
    for tecla, nome_acao in TECLAS.items():
        h = keyboard.on_press_key(tecla, fazer_handler(nome_acao))
        hooks.append(h)

    estado.iniciar()
    print(">>> Gravando! Jogue até o 6AM e aperte F10. <<<\n")

    try:
        while not keyboard.is_pressed("f10"):
            t0 = time.perf_counter()

            # Fecha o registro do frame ANTERIOR com a ação apertada desde aquela captura:
            # par (obs_t, ação decidida vendo obs_t) — a convenção do MDP que o BC clona.
            if registro_pendente is not None:
                registro_pendente["acao"] = acao_para_numero(acao_atual)
                registro_pendente["nome"] = acao_atual
                dados.append(registro_pendente)
                registro_pendente = None
            acao_atual = "nada"

            # Captura a ÁREA CLIENTE (mesma seleção de janela do env) e leva à referência
            # 1280x720 — dela saem o frame 84x84 (dataset) e a detecção de ameaça.
            if not gw.getWindowsWithTitle(WINDOW_TITLE):
                print("Jogo não encontrado! Aguardando... (F10 encerra)")
                time.sleep(1)
                continue
            frame = cap.capturar_tela(regiao_cliente(melhor_janela(WINDOW_TITLE)))
            frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_ref   = cv2.resize(frame_cinza, env_cv._ref_size)
            frame_pequeno = cv2.resize(frame_ref, (84, 84))

            # Ameaça pela implementação do env: espelha câmera/luz (gates da detecção)
            # e mantém o estado HELD entre frames dentro do env fantasma.
            env_cv.camera_aberta = estado.camera_aberta
            env_cv.luz_dir       = estado.luz_dir
            env_cv._atualizar_ameaca(frame_ref)

            caminho_frame = f"dados/{pasta}/frames/{frame_idx:06d}.png"
            cv2.imwrite(caminho_frame, frame_pequeno)

            registro_pendente = {
                "frame":      caminho_frame,
                "noite":      noite,
                "ameaca_esq": int(env_cv.ameaca_esq),
                "ameaca_dir": int(env_cv.ameaca_dir),
            }
            registro_pendente.update(estado.como_dict())

            frame_idx += 1
            estado.atualizar()
            # Mantém o período do loop estável (captura+CV têm custo variável)
            time.sleep(max(0.0, _LOOP_DT - (time.perf_counter() - t0)))

    finally:
        # Remove todos os hooks de teclado
        keyboard.unhook_all()

    # Último frame pendente: fecha com a ação acumulada até o F10
    if registro_pendente is not None:
        registro_pendente["acao"] = acao_para_numero(acao_atual)
        registro_pendente["nome"] = acao_atual
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
