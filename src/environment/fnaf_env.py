import gymnasium as gym
import numpy as np
import cv2
import time
import os
import subprocess
import unicodedata
from pathlib import Path
from gymnasium import spaces
from src.utils.capture import GameCapture, regiao_cliente, melhor_janela

def _carregar_env(caminho: str = ".env") -> None:
    if not os.path.exists(caminho):
        return

    with open(caminho, "r", encoding="utf-8") as arquivo:
        for linha in arquivo:
            conteudo = linha.strip()
            if not conteudo or conteudo.startswith("#") or "=" not in conteudo:
                continue

            chave, valor = conteudo.split("=", 1)
            chave = chave.strip()
            valor = valor.strip().strip('"').strip("'")
            os.environ.setdefault(chave, valor)


_carregar_env()

LARGURA = 84
ALTURA  = 84

ACOES = {
    0:  "nada",
    1:  "porta_esquerda",
    2:  "porta_direita",
    3:  "luz_esquerda",
    4:  "luz_direita",
    5:  "abrir_fechar_camera",
    6:  "camera_1a",
    7:  "camera_1b",
    8:  "camera_1c",
    9:  "camera_2a",
    10: "camera_2b",
    11: "camera_3",
    12: "camera_4a",
    13: "camera_4b",
    14: "camera_5",
    15: "camera_6",
    16: "camera_7",
}

ACOES_CAMERA = {acao for acao in ACOES.values() if acao.startswith("camera_")}
ACOES_LADO_ESQUERDO = {"porta_esquerda", "luz_esquerda"}
ACOES_LADO_DIREITO = {"porta_direita", "luz_direita", "abrir_fechar_camera"} | ACOES_CAMERA

LADO_POR_ACAO = {acao: "esquerdo" for acao in ACOES_LADO_ESQUERDO}
LADO_POR_ACAO.update({acao: "direito" for acao in ACOES_LADO_DIREITO})

def _env_int_obrigatorio(nome: str) -> int:
    valor = os.getenv(nome)
    if valor is None or valor.strip() == "":
        return 0
    try:
        return int(valor)
    except ValueError:
        return 0


def _env_int_opcional(nome: str, padrao: int) -> int:
    valor = os.getenv(nome)
    if valor is None or valor.strip() == "":
        return padrao
    try:
        return int(valor)
    except ValueError:
        return padrao


def _env_str_obrigatorio(nome: str) -> str:
    valor = os.getenv(nome)
    if valor is None or valor.strip() == "":
        return ""
    return valor.strip()


def _env_float_opcional(nome: str, padrao: float) -> float:
    valor = os.getenv(nome)
    if valor is None or valor.strip() == "":
        return padrao

    try:
        convertido = float(valor)
    except ValueError:
        raise ValueError(f"Valor invalido para {nome}: {valor}")

    if convertido < 0:
        raise ValueError(f"Valor invalido para {nome}: {valor}. Use numero >= 0")

    return convertido

def _env_str_opcional(nome: str, padrao: str = "") -> str:
    valor = os.getenv(nome)
    if valor is None:
        return padrao
    return valor.strip() or padrao



def _env_coord(acao: str) -> tuple[int, int]:
    prefixo = f"FNAF_COORD_{acao.upper()}".replace("-", "_")
    x = _env_int_obrigatorio(f"{prefixo}_X")
    y = _env_int_obrigatorio(f"{prefixo}_Y")
    return x, y


WINDOW_TITLE = _env_str_obrigatorio("FNAF_WINDOW_TITLE")
GAME_EXECUTABLE_PATH = _env_str_opcional("FNAF_EXECUTABLE_PATH", "")
REABRIR_ESPERA_SEGUNDOS = max(1, _env_int_opcional("FNAF_REABRIR_ESPERA_SEGUNDOS", 10))
POS_ALT_ENTER_ESPERA_SEGUNDOS = max(1, _env_int_opcional("FNAF_POS_ALT_ENTER_ESPERA_SEGUNDOS", 3))
# Método de reset (Decisão 7) — o MENU (com New Game / Continue) só aparece depois de uma MORTE;
# VENCER emenda direto na próxima noite (sem menu, sem clique). Por isso a escolha New Game vs
# Continue só acontece na morte:
#   new_game = currículo clássico: morte → New Game (Noite 1) ; vitória → próxima noite.
#   continue = MIRA FNAF_NOITE_DESEJADA: morte na/abaixo da alvo → Continue (retoma a noite),
#              morte ACIMA da alvo → New Game (reescala do 1). Vencer a alvo obriga jogar D+1 até
#              morrer (auto-avanço, não há menu pra voltar antes) — custo inerente da mecânica.
# A noite é rastreada internamente pelo desfecho (sem ler a tela). Falsa vitória/derrota é rara e
# se reancora sozinha na próxima morte; o 1º reset de cada execução SEMPRE faz New Game (Noite 1).
RESET_METODO = _env_str_opcional("FNAF_RESET_METODO", "new_game").lower()
MAX_NOITE = 7  # p/ normalizar noite/MAX_NOITE no estado (FNAF1: noites 1-6 + custom)
# Noite alvo do modo "continue" (1..MAX_NOITE). O modo new_game ignora. Default 1.
NOITE_DESEJADA = max(1, min(_env_int_opcional("FNAF_NOITE_DESEJADA", 1), MAX_NOITE))
# Espera (s) após uma VITÓRIA, enquanto o jogo avança sozinho pra próxima noite (6AM → intro).
VITORIA_ESPERA_SEGUNDOS = max(1, _env_int_opcional("FNAF_VITORIA_ESPERA_SEGUNDOS", 20))

# Botões do menu (obrigatórios) — calibre os dois com: python -m src.utils.calibrar_por_passos
NEW_GAME_CLICK = (_env_int_obrigatorio("FNAF_NEW_GAME_CLICK_X"), _env_int_obrigatorio("FNAF_NEW_GAME_CLICK_Y"))
CONTINUE_CLICK = (_env_int_obrigatorio("FNAF_CONTINUE_CLICK_X"), _env_int_obrigatorio("FNAF_CONTINUE_CLICK_Y"))


def decidir_reset(metodo: str, resultado: str | None, noite_atual: int,
                  noite_desejada: int, primeiro_reset: bool) -> tuple[str, int]:
    """Decide a AÇÃO de reset e a noite do próximo episódio (Decisão 7). Pura — testável offline.

    Ações: "new_game" (clica New Game → Noite 1), "continue" (clica Continue → retoma a noite onde
    morreu) e "nenhum" (NÃO clica: o jogo já emendou na próxima noite após a vitória — não há menu).

      • 1º reset da execução → New Game (ancora o save na Noite 1).
      • vitória → "nenhum" (auto-avanço pra próxima noite); vale p/ os dois métodos.
      • morte E truncado/None → re-clica o botão do menu (new_game/continue). O TRUNCADO entra aqui
        DE PROPÓSITO: um episódio truncado pode ser um FANTASMA preso no menu (o clique de início
        falhou no timing pós-crash do Golden Freddy) — re-clicar RESTAURA a auto-correção que o
        reset original tinha (ele sempre clicava). Antes o truncado virava "nenhum" e o agente
        ficava preso jogando contra o menu, truncando em LOOP. Só a vitória REAL (acima) emenda
        sem clicar.
          new_game → New Game (Noite 1).
          continue → mira `noite_desejada`: ACIMA da alvo → New Game (reescala do 1);
                     na/abaixo da alvo → Continue (retoma essa noite).
    """
    if primeiro_reset:
        return "new_game", 1
    if resultado == "vitoria":
        return "nenhum", min(noite_atual + 1, MAX_NOITE)
    if metodo == "new_game":
        return "new_game", 1
    alvo = max(1, min(noite_desejada, MAX_NOITE))
    if noite_atual > alvo:
        return "new_game", 1                      # morreu/truncou acima da alvo → reescala do 1
    return "continue", noite_atual                # morreu/truncou na/abaixo da alvo → retoma a noite
STEP_DELAY = _env_float_opcional("FNAF_STEP_DELAY", 0.35)
SIDE_SWITCH_DELAY = _env_float_opcional("FNAF_SIDE_SWITCH_DELAY", 0.85)
CAMERA_EXIT_DELAY = _env_float_opcional("FNAF_CAMERA_EXIT_DELAY", 0.65)
CAMERA_DRAG_PIXELS   = _env_int_opcional("FNAF_CAMERA_DRAG_PIXELS", 80)
CAMERA_DRAG_DURATION = _env_float_opcional("FNAF_CAMERA_DRAG_DURATION", 0.15)

COORDS = {
    acao: _env_coord(acao)
    for acao in ACOES.values()
    if acao != "nada"
}

# Checkpoints de energia por horário da noite (tempo em segundos, energia esperada %)
LIMIAR_AMEACA = 0.70  # match acima disto = animatrônico no vão (Decisão 4A)
DEBOUNCE_VAZIO = 4    # leituras de "vazio" ACUMULADAS p/ limpar a ameaça — absorve a estática
# ROI do rosto por lado (left,top,larg,alt em 1280x720). O rosto aparece em posição fixa
# (Bonnie x120-270/y160-330; Chica x800-950/y155-325), então o matchTemplate roda só na ROI:
# frame inteiro ≈ 44 ms/lado, ROI ≈ 9 ms — a auditoria alerta que latência por step distorce
# a energia/tempo (que correm no relógio real).
ROI_AMEACA = {"esquerdo": (0, 0, 420, 500), "direito": (700, 0, 580, 500)}

# Fator de desconto — FONTE ÚNICA: train.py importa daqui (PPO/VecNormalize) e o shaping
# potential-based (Decisão 4) usa o mesmo valor. Precisam casar p/ o shaping telescopar.
# Decisão 6: 0.997 (era 0.995) → horizonte ~333 steps (noite ~700), a vitória propaga melhor.
# Mudar SÓ em treino fresco: VecNormalize/crítico ficam presos ao gamma antigo ao retomar.
GAMMA = 0.997

# Detecção do Bonnie à esquerda — estado HELD (Decisão 4). A "sombra" do Bonnie no vão é
# escura IGUAL à luz apagada (ele PROJETA a sombra, e no escuro o vão também é escuro), então
# NÃO dá p/ detectar presença direto nem separar do escuro. Em vez disso o estado de perigo só
# muda por CONFIRMAÇÃO POSITIVA, senão MANTÉM o último valor:
#   - rosto do Bonnie casado (porta aberta) ............ ameaca_esq = True  (achou o Bonnie)
#   - corredor VAZIO iluminado (std do vão ALTO) ........ ameaca_esq = False (saiu → pode reabrir)
#   - sombra/escuro (std baixo, sem rosto) ............. mantém (não dá p/ confirmar)
# Assim some a dependência de detector de luz: no escuro nada confirma → mantém o estado, que
# só foi setado se você VIU o Bonnie antes. std medido no vão (x290-480): vazio iluminado ≈
# 11.65 ; Bonnie/escuro ≈ 9.0-9.3.
SOMBRA_REGIAO = (290, 80, 190, 440)  # (left, top, larg, alt) em 1280x720 — vão da porta esq
LIMIAR_VAZIO = 11.0       # std do vão ACIMA disto = corredor vazio iluminado confirmado
DEBOUNCE_PRESENCA = 2     # frames de rosto seguidos p/ setar a ameaça (absorve estática)

# Estado real da PORTA pela cor do botão DOOR do painel (HUD) — Decisão 4B, mesma convenção
# do _verificar_botao_porta:  verde (G>R) = FECHADA, vermelho (R>G) = ABERTA. Usado p/ o Φ do
# shaping (bloqueado = ameaça presente E porta fechada). Regiões em 1280x720 (left,top,larg,alt).
BOTAO_DOOR = {"esquerdo": (45, 300, 70, 55), "direito": (1170, 300, 70, 75)}
DOOR_COR_MARGEM = 15      # quanto G precisa superar R (ou vice-versa) p/ não ser ambíguo

# Leitura de energia (Decisão 4B) — dígitos de "Power left: XX%" (energia começa em 99, máx 2 dígitos)
ENERGIA_CELULAS_X = (185, 203)
ENERGIA_CELULA_W = 17
ENERGIA_Y = (620, 650)
ENERGIA_LIMIAR_GLIFO = 0.40  # match mínimo p/ aceitar como dígito (abaixo = ambíguo)


def validar_leitura_energia(lido, estimativa: float) -> float:
    """Filtro photo-primary: a foto é a verdade. A simulação (estimativa) desalinha
    fácil e o Foxy tira grandes pedaços de uma vez, então NÃO se rejeita queda por
    magnitude. Só barra o impossível (energia subir) e segura quando não há leitura
    (None); todo decréscimo — inclusive grande — é aceito e re-ancora a estimativa.
    O reader já devolve None em vez de meio número, então não chega leitura partida."""
    if lido is None:
        return estimativa                          # sem leitura (câmera/flicker/0%): mantém
    if lido > estimativa + 1.0:
        return estimativa                          # subiu: impossível → erro de leitura
    return float(lido)                             # decréscimo (inclui Foxy) → re-ancora na foto

CHECKPOINTS_NOITE = [
    (0,   100.0),
    (89,   85.0),   # 1AM
    (178,  60.0),   # 2AM
    (267,  40.0),   # 3AM
    (356,  25.0),   # 4AM
    (445,  15.0),   # 5AM
    (535,   5.0),   # 6AM
]

# ── Recompensa (redesenho NÃO-PRESCRITIVO) ────────────────────────────────────────────
# Princípio: premiar o OBJETIVO e os recursos reais do jogo, nunca ações específicas. Toda
# dica de COMO jogar entra por shaping potential-based (γ·Φ'−Φ, telescopa → NÃO move o ótimo,
# Ng/Harada/Russell 1999), nunca por bônus/penalidade por ação — que vicia e cria ótimos
# degenerados (ex.: acampar 100% na câmera). O sinal denso é SOBREVIVÊNCIA POR TEMPO REAL
# (não por nº de steps: pagar por step premia spammar ação rápida) e o orçamento total da
# noite fica << vitória, então VENCER domina "só durar".
# Terminais — magnitudes do desfecho. RECOMPENSA_MORTE é compartilhada com o caminho de
# interrupção (_interromper_episodio): o Golden Freddy FECHA o jogo (crash-jumpscare) em vez
# de mostrar Game Over, então "janela sumiu no meio da noite" é uma MORTE e leva o mesmo -100.
RECOMPENSA_MORTE      = -100.0 # derrota (terminal): qualquer animatrônico, INCLUI Golden Freddy
RECOMPENSA_VITORIA    = 500.0  # vitória (terminal): sobreviver à noite (>> orçamento denso)
DURACAO_NOITE         = 535.0  # ~6h em segundos reais (último checkpoint) — base do progresso
RECOMPENSA_NOITE      = 60.0   # orçamento denso de uma noite inteira (Σ ≈ 60 << vitória 500)
BONUS_MARCO_HORA      = 3.0    # marco por hora alcançada (flat, SEM peso de energia): 6×3 = 18
PESO_AMEACA_BLOQUEADA = 0.5    # Φ: +0.5 por lado com ameaça PRESENTE e porta FECHADA
PESO_FOXY             = 0.5    # Φ: penaliza câmera negligenciada (proxy do Foxy), só após PACIENCIA
FOXY_PACIENCIA        = 20     # steps sem câmera tolerados antes de o risco do Foxy subir no Φ


class FNAFEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()

        self.capture          = GameCapture()
        self.render_mode      = render_mode
        self.contador_vitoria = 0
        self._contador_menu   = 0      # frames seguidos vendo o menu (debounce, Decisão 8)
        self._crash_menu      = False  # último desfecho foi Bonnie/crash pro menu
        self._carregar_templates()

        self.observation_space = spaces.Dict({
            "imagem": spaces.Box(
                low=0, high=255,
                shape=(ALTURA, LARGURA, 1),
                dtype=np.uint8
            ),
            "estados": spaces.Box(
                low=0, high=1,
                shape=(11,),                 # +1 = noite (Decisão 7)
                dtype=np.float32
            )
        })
        self.action_space = spaces.Discrete(len(ACOES))

        self.passos    = 0
        self.max_passos = 10_000
        self.energia   = 100.0
        self._apagou   = False  # energia chegou a 0 nesta noite (luzes forçadas a apagar)
        self.tempo_jogo = 0.0
        self.luz_esq = False
        self.luz_dir = False
        self.porta_esq = False
        self.porta_dir = False
        self.camera_aberta = False
        self.camera_ativa = 0
        self.vivo      = True
        self.lado_atual = "centro"
        self.ultima_acao = None
        self.penultima_acao = None
        self.contador_nada = 0
        self.episode_start_time = None
        self._t_ultima_energia = None
        self._dt_step                 = 0.7  # Δt do step (recalculado em _atualizar_energia)
        self.passos_sem_camera        = 0
        self._botao_luz_pressionado   = None
        self.cooldown_porta_esq       = 0  # só bloqueia porta (animação ~0.6s)
        self.cooldown_porta_dir       = 0  # só bloqueia porta (animação ~0.6s)
        self.cooldown_camera          = 0  # bloqueia abrir/fechar câmera (animação ~1.0s)
        self._pixel_antes_porta: tuple | None = None  # (B, G, R) do botão antes do clique
        self._template_camera_discordancia = 0  # nº de checks consecutivos discordantes
        self._episodio_num        = 0
        self._count_sync_camera   = 0
        self._count_porta_falha   = 0
        self._count_sync_porta    = 0
        self._log_desyncs_path    = "logs/desyncs.log"
        self._horas_bonificadas: set = set()
        self._total_bonus_hora: float = 0.0
        self.ameaca_esq = False
        self.ameaca_dir = False
        self._vazio_esq = 0
        self._vazio_dir = 0
        self._presenca_esq = 0
        # Noite (Decisão 7) — rastreada internamente pelo desfecho do episódio anterior.
        # NÃO é zerada no reset() (persiste entre episódios; só a transição abaixo a muda).
        self.noite = 1
        self._resultado_episodio = None   # "vitoria" | "morte" | None (interrompido/truncado)
        self._primeiro_reset = True       # 1º reset da execução ancora o save na Noite 1 (New Game)
        self._acao_reset = "new_game"     # ação ("new_game"|"continue"|"nenhum") decidida por _preparar_reset

    def _janela_do_jogo_aberta(self) -> bool:
        import pygetwindow as gw
        return bool(gw.getWindowsWithTitle(WINDOW_TITLE))

    def _camera_aberta_por_template(self) -> bool | None:
        """Verifica estado real da câmera via matchTemplate no indicador 'YOU' do mapa.
        Retorna None se o template não foi carregado."""
        if self.template_camera_aberta is None:
            return None
        frame = self.capture.capturar_tela()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resultado = cv2.matchTemplate(gray, self.template_camera_aberta, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(resultado)
        return max_val > 0.75

    def _verificar_botao_porta(self, nome_acao: str) -> bool:
        """Verifica se o clique de porta registrou pelo delta de cor do botão (vermelho↔verde).
        Tenta até 3 vezes antes de desistir. Se todas falharem, lê a cor atual para
        sincronizar o estado interno com o que o jogo mostra em vez de reverter às cegas."""
        if self._pixel_antes_porta is None:
            return True
        b1, g1, r1 = self._pixel_antes_porta
        x, y = COORDS[nome_acao]

        for tentativa in range(3):
            frame = self.capture.capturar_tela()
            h, w = frame.shape[:2]
            if not (0 <= y < h and 0 <= x < w):
                return True
            b2, g2, r2 = int(frame[y, x][0]), int(frame[y, x][1]), int(frame[y, x][2])
            dom_antes = "verde" if g1 > r1 + 50 else ("vermelho" if r1 > g1 + 50 else None)
            dom_depois = "verde" if g2 > r2 + 50 else ("vermelho" if r2 > g2 + 50 else None)
            if dom_antes is not None and dom_depois is not None and dom_antes != dom_depois:
                return True  # cor dominante inverteu — clique registrado
            if tentativa < 2:
                self.capture.clicar(x, y)
                time.sleep(0.15)

        # Todas as tentativas falharam — lê estado real pela cor do botão
        self._count_porta_falha += 1
        if g2 > r2 + 50:        # predominantemente verde → porta fechada
            estado_real = True
        elif r2 > g2 + 50:      # predominantemente vermelho → porta aberta
            estado_real = False
        else:                   # cor ambígua — reverte conservadoramente
            estado_real = None

        if estado_real is not None:
            if nome_acao == "porta_esquerda":
                self.porta_esq = estado_real
                self.cooldown_porta_esq = 0
            else:
                self.porta_dir = estado_real
                self.cooldown_porta_dir = 0
        else:
            if nome_acao == "porta_esquerda":
                self.porta_esq = not self.porta_esq
                self.cooldown_porta_esq = 0
            else:
                self.porta_dir = not self.porta_dir
                self.cooldown_porta_dir = 0
        return False

    def _verificar_e_focar_janela(self) -> bool:
        import pygetwindow as gw
        janelas = gw.getWindowsWithTitle(WINDOW_TITLE)
        if not janelas:
            return False
        win = janelas[0]
        try:
            if not win.isActive:
                win.activate()
                time.sleep(0.15)  # 0.15s: Windows precisa de tempo para processar o foco
        except Exception:
            pass
        return True

    @staticmethod
    def _normalizar_texto(texto: str) -> str:
        texto = unicodedata.normalize("NFKD", texto)
        texto = "".join(char for char in texto if not unicodedata.combining(char))
        return texto.lower()

    @staticmethod
    def _caminhos_desktop() -> list[Path]:
        candidatos = [
            Path.home() / "Desktop",
            Path(os.getenv("USERPROFILE", "")) / "Desktop",
            Path(os.getenv("PUBLIC", "")) / "Desktop",
        ]

        one_drive = os.getenv("OneDrive")
        if one_drive:
            candidatos.append(Path(one_drive) / "Desktop")

        unicos: list[Path] = []
        vistos: set[str] = set()
        for caminho in candidatos:
            chave = str(caminho).strip().lower()
            if not chave or chave in vistos:
                continue
            vistos.add(chave)
            unicos.append(caminho)
        return unicos

    def _descobrir_atalho_desktop(self) -> Path | None:
        palavras_chave = ("five nights", "freddy", "fnaf")
        extensoes_validas = {".lnk", ".url", ".exe"}

        for desktop in self._caminhos_desktop():
            if not desktop.exists() or not desktop.is_dir():
                continue

            arquivos = sorted(
                [arquivo for arquivo in desktop.iterdir() if arquivo.is_file()],
                key=lambda item: item.name.lower(),
            )

            for arquivo in arquivos:
                if arquivo.suffix.lower() not in extensoes_validas:
                    continue

                nome_normalizado = self._normalizar_texto(arquivo.stem)
                if any(chave in nome_normalizado for chave in palavras_chave):
                    return arquivo

        return None

    def _resolver_caminho_jogo(self) -> Path | None:
        if GAME_EXECUTABLE_PATH:
            texto_expandido = os.path.expandvars(os.path.expanduser(GAME_EXECUTABLE_PATH))
            caminho = Path(texto_expandido)
            if caminho.exists() and caminho.is_file():
                return caminho

            nome_apenas = Path(GAME_EXECUTABLE_PATH).name
            for desktop in self._caminhos_desktop():
                candidato = desktop / nome_apenas
                if candidato.exists() and candidato.is_file():
                    return candidato

            print(
                "[FALLBACK] Caminho invalido em FNAF_EXECUTABLE_PATH: "
                f"{GAME_EXECUTABLE_PATH}"
            )

        encontrado = self._descobrir_atalho_desktop()
        if encontrado is not None:
            print(f"[FALLBACK] Usando atalho detectado na area de trabalho: {encontrado}")
        return encontrado

    @staticmethod
    def _abrir_arquivo(path_arquivo: Path) -> bool:
        try:
            if os.name == "nt":
                os.startfile(str(path_arquivo))
            else:
                subprocess.Popen([str(path_arquivo)], cwd=str(path_arquivo.parent))
            return True
        except Exception:
            return False

    def _abrir_jogo_fallback(self) -> bool:
        caminho = self._resolver_caminho_jogo()
        if caminho is None:
            print(
                "[FALLBACK] Nao foi encontrado executavel/atalho do jogo. "
                "Configure FNAF_EXECUTABLE_PATH com .exe ou .lnk."
            )
            return False

        if not self._abrir_arquivo(caminho):
            print(f"[FALLBACK] Falha ao abrir jogo automaticamente: {caminho}")
            return False

        print("[FALLBACK] Jogo fechado detectado. Relancando executavel...")
        time.sleep(REABRIR_ESPERA_SEGUNDOS)

        if not self.capture.focar_janela(WINDOW_TITLE):
            print("[FALLBACK] Janela nao encontrada apos relancamento.")
            return False

        self.capture.atalho("alt", "enter")
        time.sleep(POS_ALT_ENTER_ESPERA_SEGUNDOS)
        self.capture.focar_janela(WINDOW_TITLE)
        print("[FALLBACK] Jogo recolocado em modo janela (ALT+ENTER).")
        return True

    def _interromper_episodio(self, motivo: str, como_morte: bool = False):
        """Encerra o episódio quando o jogo some no meio da noite e tenta reabri-lo. Ambos os
        casos levam RECOMPENSA_MORTE (derrota — não premia ficar travado na tela inicial):

        como_morte=True — janela FECHADA = Golden Freddy fechou o jogo (crash-jumpscare). Conta
        como MORTE de verdade (morreu=True, desfecho 'morte'): o jogo reabre no MENU e o reset
        segue a mecânica de noite via decidir_reset — Continue retoma a noite onde morreu, New Game
        volta pra Noite 1 (conforme FNAF_RESET_METODO).
        como_morte=False — falha de captura com a janela ainda ABERTA (glitch transitório): marcado
        'interrompido' (fora da métrica de skill) e reancora na Noite 1 (estado incerto)."""
        if self._botao_luz_pressionado:
            try:
                self.capture.soltar_botao(*self._botao_luz_pressionado)
            except Exception:
                pass
            self._botao_luz_pressionado = None

        recuperado = self._abrir_jogo_fallback()
        if not recuperado:
            motivo = f"{motivo} (fallback sem sucesso)"

        recompensa = RECOMPENSA_MORTE
        if como_morte:
            self._resultado_episodio = "morte"   # mecânica de noite: Continue retoma / New Game volta 1
        else:
            self._primeiro_reset = True           # estado incerto → reancora na Noite 1

        info = {
            "passos":       self.passos,
            "energia":      self.energia,
            "tempo":        self.tempo_jogo,
            "tempo_real":   time.perf_counter() - (self.episode_start_time or time.perf_counter()),
            "porta_esq":    self.porta_esq,
            "porta_dir":    self.porta_dir,
            "camera_aberta": self.camera_aberta,
            "camera_ativa": self.camera_ativa,
            "morreu":       como_morte,
            "interrompido": not como_morte,
            "ocorrido":     motivo,
            "noite":        self.noite,
        }

        observacao = {
            "imagem": np.zeros((ALTURA, LARGURA, 1), dtype=np.uint8),
            # Deriva do espaço (10 estados após a ameaça da Decisão 4) p/ não divergir
            "estados": np.zeros(self.observation_space["estados"].shape, dtype=np.float32)
        }
        return observacao, recompensa, True, False, info

    def _preparar_reset(self) -> None:
        """Decide a noite e o BOTÃO do menu do próximo episódio (Decisão 7) via decidir_reset,
        a partir do desfecho anterior, do método (new_game/continue) e da noite alvo. Consome o
        flag de 1º reset (que força New Game) e o resultado do episódio. Testável offline."""
        self._acao_reset, self.noite = decidir_reset(
            RESET_METODO, self._resultado_episodio, self.noite,
            NOITE_DESEJADA, self._primeiro_reset,
        )
        self._primeiro_reset = False
        self._resultado_episodio = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if not WINDOW_TITLE:
            raise RuntimeError(
                "FNAF_WINDOW_TITLE nao configurado no .env. "
                "Configure as variaveis obrigatorias antes de executar."
            )

        self._preparar_reset()   # decide self.noite e self._acao_reset pelo desfecho anterior (Decisão 7)

        self.passos           = 0
        self.energia          = 100.0
        self._apagou          = False  # vira True se a energia zerar (separa vitória/morte por apagão)
        self.tempo_jogo       = 0.0
        self.luz_esq          = False
        self.luz_dir          = False
        self.porta_esq        = False
        self.porta_dir        = False
        self.camera_aberta    = False
        self.camera_ativa     = 0
        self.vivo             = True
        self.lado_atual       = "centro"
        self.ultima_acao      = None
        self.penultima_acao   = None
        self.contador_nada    = 0
        self.contador_vitoria  = 0
        self._contador_menu    = 0      # debounce da detecção de menu (Decisão 8)
        self._crash_menu       = False
        self.episode_start_time    = None
        self.passos_sem_camera        = 0
        self._botao_luz_pressionado   = None
        self.cooldown_porta_esq       = 0
        self.cooldown_porta_dir       = 0
        self.cooldown_camera          = 0
        self._pixel_antes_porta        = None
        self._template_camera_discordancia = 0
        self._episodio_num           += 1
        self._count_sync_camera       = 0
        self._count_porta_falha       = 0
        self._count_sync_porta        = 0
        self._horas_bonificadas       = set()
        self._total_bonus_hora        = 0.0
        self.ameaca_esq               = False
        self.ameaca_dir               = False
        self._vazio_esq               = 0
        self._vazio_dir               = 0
        self._presenca_esq            = 0

        if not self._janela_do_jogo_aberta():
            self._abrir_jogo_fallback()

        if not self.capture.focar_janela(WINDOW_TITLE):
            raise RuntimeError(
                "Janela do jogo nao encontrada. "
                "Configure FNAF_EXECUTABLE_PATH no .env para fallback automatico."
            )
        time.sleep(0.5)

        # Ação decidida por _preparar_reset. "nenhum" = vitória: o jogo já emendou na próxima noite
        # sozinho (sem menu) — só espera a transição 6AM→intro, sem clicar. Caso contrário está no
        # menu (morte ou 1º reset): clica New Game ou Continue (duplo-clique com sleeps p/ robustez).
        if self._acao_reset == "nenhum":
            time.sleep(VITORIA_ESPERA_SEGUNDOS)
        else:
            botao_reset = NEW_GAME_CLICK if self._acao_reset == "new_game" else CONTINUE_CLICK
            self.capture.clicar(*botao_reset)
            time.sleep(15)
            self.capture.clicar(*botao_reset)
            time.sleep(20)
        self.episode_start_time = time.perf_counter()
        self._t_ultima_energia  = self.episode_start_time

        print("Reset completo — noite iniciada!")
        observacao = self._capturar_observacao()
        return observacao, {}

    def step(self, acao: int):
        self.passos += 1
        self._atualizar_tempo()

        if not self._verificar_e_focar_janela():
            # Janela fechada no meio da noite = Golden Freddy fechou o jogo → MORTE.
            return self._interromper_episodio("janela fechada (Golden Freddy)", como_morte=True)

        # ── Verificação de sincronia via template "YOU" do mapa de câmeras ────
        # Checado a cada 3 steps, apenas fora do cooldown (evita falsa correção
        # durante a animação de abertura/fechamento da tablet).
        if self.template_camera_aberta is not None and self.passos % 3 == 0 and self.cooldown_camera == 0:
            real = self._camera_aberta_por_template()
            if real is not None and real != self.camera_aberta:
                self._template_camera_discordancia += 1
                if self._template_camera_discordancia >= 2:
                    self.camera_aberta = real
                    if not real:
                        self.camera_ativa = 0
                    self._count_sync_camera += 1
                    self._template_camera_discordancia = 0
            else:
                self._template_camera_discordancia = 0

        # ─────────────────────────────────────────────────────────────────────

        # Quando energia acabou, desliga tudo mas não encerra ainda —
        # no FNAF1 o Freddy demora alguns segundos para aparecer após a
        # energia zerar. Encerrar aqui causaria reset() durante a animação
        # de morte, corrompendo o estado do jogo. O episódio termina quando
        # _detectar_morte() confirmar via template, igual ao caminho normal.
        if self.energia <= 0:
            # Verificação passiva: se câmera estava aberta e YOU ainda aparece,
            # o jogo ainda tem energia — o estado interno zerou cedo demais.
            if (self.camera_aberta
                    and self.template_camera_aberta is not None
                    and self.passos % 3 == 0):
                if self._camera_aberta_por_template():
                    self.energia = 5.0
            if self.energia <= 0:
                self._apagou = True   # apagão confirmado: separa vitória/morte por energia esgotada
                self.porta_esq = False
                self.porta_dir = False
                self.luz_esq = False
                self.luz_dir = False
                self.camera_aberta = False

        # Φ antes da ação, para o shaping potential-based (Decisão 4): usa a ameaça
        # da observação anterior e o estado de porta pré-ação.
        phi_antes = self._potencial_seguranca()
        acao_valida = self._executar_acao(acao)
        time.sleep(STEP_DELAY)

        # Verifica cor do botão da porta após o clique: vermelho↔verde confirma registro.
        if self._pixel_antes_porta is not None:
            if not self._verificar_botao_porta(ACOES[acao]):
                acao_valida = False
            self._pixel_antes_porta = None

        # Pressiona o botão de luz ativa antes de capturar a observação —
        # mantém o corredor iluminado na imagem e sincroniza passivamente a porta do mesmo lado.
        if self.luz_esq:
            lx, ly = COORDS["luz_esquerda"]
            self.capture.segurar_botao(lx, ly)
            self._botao_luz_pressionado = (lx, ly)
            _frame_l = self.capture.capturar_tela()
            _px, _py = COORDS["porta_esquerda"]
            _h, _w = _frame_l.shape[:2]
            if 0 <= _py < _h and 0 <= _px < _w:
                _gl, _rl = int(_frame_l[_py, _px][1]), int(_frame_l[_py, _px][2])
                _real = True if _gl > _rl + 50 else (False if _rl > _gl + 50 else None)
                if _real is not None and _real != self.porta_esq:
                    self.porta_esq = _real
                    self._count_sync_porta += 1
        elif self.luz_dir:
            lx, ly = COORDS["luz_direita"]
            self.capture.segurar_botao(lx, ly)
            self._botao_luz_pressionado = (lx, ly)
            _frame_l = self.capture.capturar_tela()
            _px, _py = COORDS["porta_direita"]
            _h, _w = _frame_l.shape[:2]
            if 0 <= _py < _h and 0 <= _px < _w:
                _gl, _rl = int(_frame_l[_py, _px][1]), int(_frame_l[_py, _px][2])
                _real = True if _gl > _rl + 50 else (False if _rl > _gl + 50 else None)
                if _real is not None and _real != self.porta_dir:
                    self.porta_dir = _real
                    self._count_sync_porta += 1

        try:
            observacao = self._capturar_observacao()
            morreu     = self._detectar_morte()
            sobreviveu = self._detectar_vitoria()
            # Bonnie/crash devolveu ao menu sem tela de morte (Decisão 8) → conta como morte.
            # O jogo já está no menu, então o reset (que clica New Game) funciona igual.
            self._crash_menu = self._detectar_menu_crash()
            if self._crash_menu:
                morreu = True
        except Exception as erro:
            # Janela sumiu = Golden Freddy fechou o jogo → MORTE. Janela ainda aberta = glitch
            # transitório de captura → interrompido (também derrota, mas reancora na Noite 1).
            if not self._janela_do_jogo_aberta():
                return self._interromper_episodio(f"janela fechada (Golden Freddy): {erro}", como_morte=True)
            return self._interromper_episodio(f"falha ao capturar estado: {erro}")

        if self._botao_luz_pressionado:
            self.capture.soltar_botao(*self._botao_luz_pressionado)
            self._botao_luz_pressionado = None

        luz_esq_step = self.luz_esq
        luz_dir_step = self.luz_dir

        self._atualizar_energia()
        self._atualizar_cooldowns()

        if self.camera_aberta:
            self.passos_sem_camera = 0
        else:
            self.passos_sem_camera += 1

        recompensa = self._calcular_recompensa(morreu, sobreviveu, acao, acao_valida)
        terminado  = morreu or sobreviveu
        # Shaping potential-based (Decisão 4, Opção B): soma a variação de segurança.
        # Telescopa (Σ γ·Φ' − Φ ≈ 0 no episódio), então guia sem mover o ótimo. Φ'=0 no
        # terminal. Convive com VecNormalize: entra na recompensa crua, antes de normalizar.
        phi_depois = 0.0 if terminado else self._potencial_seguranca()
        recompensa += GAMMA * phi_depois - phi_antes
        # Trava de segurança por tempo: a noite dura ~535s. Se a detecção de
        # vitória falhar (template), encerra em vez de ficar preso por horas.
        truncado   = self.passos >= self.max_passos or self.tempo_jogo > 700.0

        if terminado or truncado:
            self._escrever_log_desyncs(morreu, sobreviveu)
        # Registra o desfecho p/ a transição de noite no próximo reset (Decisão 7).
        # truncado sem morte/vitória → None (mantém a noite).
        if terminado:
            self._resultado_episodio = "vitoria" if sobreviveu else "morte"

        info = {
            "passos":         self.passos,
            "energia":        self.energia,
            "tempo":          self.tempo_jogo,
            "tempo_real":     time.perf_counter() - (self.episode_start_time or time.perf_counter()),
            "luz_esq":        luz_esq_step,
            "luz_dir":        luz_dir_step,
            "porta_esq":      self.porta_esq,
            "porta_dir":      self.porta_dir,
            "camera_aberta":  self.camera_aberta,
            "camera_ativa":   self.camera_ativa,
            "morreu":         morreu,
            "acao_valida":    acao_valida,
            "acao_nome":      ACOES[acao],
            "bonus_hora":     self._total_bonus_hora,
            "noite":          self.noite,
            "causa":          self._classificar_desfecho(morreu, sobreviveu),
        }

        return observacao, recompensa, terminado, truncado, info

    def _classificar_desfecho(self, morreu: bool, sobreviveu: bool) -> str | None:
        """Rotula o desfecho terminal p/ separar SKILL de SORTE na métrica (None se não terminou):
          • vitoria_gerida    — venceu com a energia ainda de pé (gestão real do recurso);
          • vitoria_apagao    — venceu DEPOIS da energia zerar (luzes apagadas; o 6 AM chegou
                                antes do Freddy — vitória de alta variância, depende do RNG);
          • morte_energia     — energia esgotou e o Freddy pegou no apagão (problema de gestão);
          • morte_animatronico— morreu COM energia (um animatrônico passou a porta — defesa/timing).
        Usa a flag self._apagou (energia chegou a 0), não a energia do step terminal: é o estado
        de apagão de fato, imune a ruído de leitura e ao 6 AM esperado em ~5%."""
        if not (morreu or sobreviveu):
            return None
        if sobreviveu:
            return "vitoria_apagao" if self._apagou else "vitoria_gerida"
        # Decisão 8: morte detectada pelo menu (Bonnie/crash sem tela de Game Over).
        # É morte legítima (animatrônico passou), mas rotulada à parte p/ medir quão
        # frequente a detecção normal por template falha.
        if self._crash_menu:
            return "menu_crash"
        return "morte_energia" if self._apagou else "morte_animatronico"

    def _executar_acao(self, acao: int) -> bool:
        """Executa ação e retorna True se teve efeito, False se foi inválida."""
        nome_acao = ACOES[acao]
        lado_alvo = LADO_POR_ACAO.get(nome_acao)

        if nome_acao == "nada":
            self.penultima_acao = self.ultima_acao
            self.ultima_acao = nome_acao
            self.contador_nada += 1
            return True

        # Ações de porta/luz só funcionam quando NÃO está na câmera
        if nome_acao in ["porta_esquerda", "porta_direita", "luz_esquerda", "luz_direita"]:
            if self.camera_aberta:
                return False

            if nome_acao in {"porta_esquerda", "porta_direita"}:
                # Lê cor do botão antes do toggle para corrigir desync de estado.
                x_pre, y_pre = COORDS[nome_acao]
                frame_pre = self.capture.capturar_tela()
                h_pre, w_pre = frame_pre.shape[:2]
                if 0 <= y_pre < h_pre and 0 <= x_pre < w_pre:
                    g_pre = int(frame_pre[y_pre, x_pre][1])
                    r_pre = int(frame_pre[y_pre, x_pre][2])
                    if g_pre > r_pre + 50:
                        estado_real_pre = True   # verde → fechada
                    elif r_pre > g_pre + 50:
                        estado_real_pre = False  # vermelho → aberta
                    else:
                        estado_real_pre = None
                    if estado_real_pre is not None:
                        estado_atual_pre = self.porta_esq if nome_acao == "porta_esquerda" else self.porta_dir
                        if estado_real_pre != estado_atual_pre:
                            if nome_acao == "porta_esquerda":
                                self.porta_esq = estado_real_pre
                            else:
                                self.porta_dir = estado_real_pre
                            self._count_sync_porta += 1

            if nome_acao == "porta_esquerda":
                if self.cooldown_porta_esq > 0:
                    return False
                self.porta_esq = not self.porta_esq
                self.cooldown_porta_esq = 3  # 3 steps × 0.25s = 0.75s — cobre animação da porta
            elif nome_acao == "porta_direita":
                if self.cooldown_porta_dir > 0:
                    return False
                self.porta_dir = not self.porta_dir
                self.cooldown_porta_dir = 3
            elif nome_acao == "luz_esquerda":
                if self.luz_esq:
                    self.luz_esq = False
                else:
                    self.luz_dir = False  # só uma luz por vez
                    self.luz_esq = True
            elif nome_acao == "luz_direita":
                if self.luz_dir:
                    self.luz_dir = False
                else:
                    self.luz_esq = False  # só uma luz por vez
                    self.luz_dir = True

        # Abrir/fechar câmera — bloqueado durante cooldown para aguardar animação (~1.0s)
        elif nome_acao == "abrir_fechar_camera":
            if self.cooldown_camera > 0:
                return False
            self.camera_aberta = not self.camera_aberta
            self.cooldown_camera = 4  # 4 steps × 0.25s = 1.0s — cobre animação completa
            if not self.camera_aberta:
                self.camera_ativa = 0
            # Ao abrir câmera, desliga luzes
            if self.camera_aberta:
                self.luz_esq = False
                self.luz_dir = False

        # Trocar de câmera só funciona se câmera estiver aberta
        elif nome_acao.startswith("camera_"):
            if not self.camera_aberta:
                return False  # Ação inválida - câmera fechada
            self.camera_ativa = acao - 5

        if nome_acao in COORDS:
            x, y = COORDS[nome_acao]

            _saindo_camera = (self.ultima_acao in ACOES_CAMERA or
                              self.ultima_acao == "abrir_fechar_camera" or
                              (not self.camera_aberta and self.cooldown_camera > 0))
            _indo_porta_luz = nome_acao in {
                "luz_esquerda", "luz_direita", "porta_esquerda", "porta_direita"
            }

            # Saindo de qualquer ação de câmera para porta/luz OU para fechar/abrir câmera:
            # aguarda a animação da prancheta terminar antes do próximo clique.
            if _saindo_camera and (_indo_porta_luz or nome_acao == "abrir_fechar_camera"):
                # Para câmera: pré-posiciona ACIMA do botão (nunca no botão).
                # O botão dispara o toggle por hover — mover para (x,y) causaria
                # um toggle acidental antes do arrastar intencional (double-trigger = desync).
                if nome_acao == "abrir_fechar_camera":
                    self.capture.mover_mouse(x, y - CAMERA_DRAG_PIXELS)
                else:
                    self.capture.mover_mouse(x, y)
                time.sleep(CAMERA_EXIT_DELAY)
            # Ao trocar de lado sem vir de câmera, aguarda a virada de cabeça.
            elif lado_alvo and self.lado_atual and lado_alvo != self.lado_atual:
                # Para câmera: não entra no hitbox do botão antes do drag intencional.
                # O mesmo cuidado do _saindo_camera branch: posiciona ACIMA do botão.
                if nome_acao == "abrir_fechar_camera":
                    self.capture.mover_mouse(x, y - CAMERA_DRAG_PIXELS)
                else:
                    self.capture.mover_mouse(x, y)
                time.sleep(SIDE_SWITCH_DELAY)

            # Re-verifica foco após qualquer delay — outros processos podem ter
            # roubado o foco durante CAMERA_EXIT_DELAY ou SIDE_SWITCH_DELAY.
            self._verificar_e_focar_janela()

            if nome_acao == "abrir_fechar_camera":
                # Hover puro (sem mouseDown): evita que a UI de câmera capture o gesto
                # como pan, o que impedia o toggle de registrar ao fechar a câmera.
                self.capture.arrastar_para(x, y, duration=CAMERA_DRAG_DURATION)
                time.sleep(0.08)
                self.capture.arrastar_para(x, y - CAMERA_DRAG_PIXELS, duration=CAMERA_DRAG_DURATION)
            elif nome_acao in {"luz_esquerda", "luz_direita"}:
                pass  # press e SYNC-LUZ acontecem em step() antes da observação
            else:
                # Captura pixel antes do clique como referência para verificação pós-clique.
                if nome_acao in {"porta_esquerda", "porta_direita"}:
                    frame = self.capture.capturar_tela()
                    h, w = frame.shape[:2]
                    if 0 <= y < h and 0 <= x < w:
                        self._pixel_antes_porta = (int(frame[y, x][0]), int(frame[y, x][1]), int(frame[y, x][2]))
                self.capture.clicar(x, y)

            if lado_alvo:
                self.lado_atual = lado_alvo

        self.penultima_acao = self.ultima_acao
        self.ultima_acao = nome_acao
        self.contador_nada = 0
        return True

    def _atualizar_cooldowns(self):
        if self.cooldown_porta_esq > 0:
            self.cooldown_porta_esq -= 1
        if self.cooldown_porta_dir > 0:
            self.cooldown_porta_dir -= 1
        if self.cooldown_camera > 0:
            self.cooldown_camera -= 1
    
    def _atualizar_energia(self):
        # O jogo drena energia por segundo REAL (wall-clock). Usar STEP_DELAY
        # como proxy subestimava o dreno em ~50% (o step real leva ~0.7s com
        # animações), fazendo a observação de energia mentir para o agente.
        agora = time.perf_counter()
        delta = agora - (self._t_ultima_energia or agora)
        self._t_ultima_energia = agora
        self._dt_step = delta   # Δt de relógio do step — base do sinal denso de sobrevivência

        itens_ativos = (int(self.porta_esq) + int(self.porta_dir)
                        + int(self.luz_esq) + int(self.luz_dir)
                        + int(self.camera_aberta))
        itens_ativos = min(itens_ativos, 3)

        consumo_por_segundo = 0.104 + itens_ativos * 0.100
        self.energia -= consumo_por_segundo * delta
        self.energia = max(0.0, self.energia)

    def _atualizar_tempo(self):
        # Tempo real do episódio — o jogo roda em wall-clock, então o relógio
        # interno (checkpoints de hora, energia esperada, progresso) precisa
        # acompanhar o relógio do jogo, não a soma de STEP_DELAY.
        self.tempo_jogo = time.perf_counter() - (self.episode_start_time or time.perf_counter())
    
    def _energia_esperada(self) -> float:
        t = self.tempo_jogo
        for i in range(len(CHECKPOINTS_NOITE) - 1):
            t0, e0 = CHECKPOINTS_NOITE[i]
            t1, e1 = CHECKPOINTS_NOITE[i + 1]
            if t <= t1:
                frac = (t - t0) / (t1 - t0)
                return e0 + frac * (e1 - e0)
        return 5.0

    def _potencial_seguranca(self) -> float:
        """Φ(estado) do shaping potential-based (Ng, Harada & Russell 1999): o treino recompensa
        só a variação γ·Φ(depois)−Φ(antes) (somada em step()), que telescopa ao longo do episódio
        e por isso NÃO move o ótimo — guia SEM prescrever a jogada, só faz o crédito chegar cedo.
        Φ mede "quão segura é a situação AGORA":
          • +PESO_AMEACA_BLOQUEADA por lado com ameaça PRESENTE e porta FECHADA (lidou com ela);
            0 se exposta (porta aberta) ou sem ameaça — não premia fechar porta à toa.
          • −PESO_FOXY conforme a câmera é negligenciada ALÉM de FOXY_PACIENCIA (proxy do Foxy, que
            corre quanto menos se olha as câmeras). Por ser potential-based, checar a câmera sobe Φ
            e negligenciar baixa, e isso telescopa: NÃO há ganho líquido em acampar (≠ da antiga
            penalidade unilateral, que só punia NÃO-checar → viciava em ficar na câmera). Abaixo da
            paciência o termo é 0, então não há empurrão para a câmera no caso comum.
        Φ=0 no terminal (forçado em step())."""
        phi = 0.0
        if self.ameaca_esq and self.porta_esq:
            phi += PESO_AMEACA_BLOQUEADA
        if self.ameaca_dir and self.porta_dir:
            phi += PESO_AMEACA_BLOQUEADA
        excesso_sem_camera = max(0, self.passos_sem_camera - FOXY_PACIENCIA)
        risco_foxy = min(excesso_sem_camera / FOXY_PACIENCIA, 1.0)
        phi -= PESO_FOXY * risco_foxy
        return phi

    def _calcular_recompensa(self, morreu: bool, sobreviveu: bool, acao: int, acao_valida: bool) -> float:
        # ── Objetivo verdadeiro (terminal) ────────────────────────────────────────
        # Magnitudes moderadas p/ o crítico não explodir. É AQUI que "jogar bem" se separa
        # de "jogar mal": vencer >> morrer. O shaping (γ·Φ'−Φ, somado em step(), fora daqui)
        # só acelera o aprendizado SEM mover este ótimo — por isso não há bônus por ação.
        if morreu:
            return RECOMPENSA_MORTE
        if sobreviveu:
            return RECOMPENSA_VITORIA

        # ── Sinal denso = SOBREVIVÊNCIA POR TEMPO REAL (não por nº de steps) ──────────
        # Pagar por step fixo num env de tempo real (step de duração variável) paga MAIS
        # quem spamma ações rápidas/baratas — foi um dos vetores do vício de câmera. Atrelar
        # ao relógio (Δt) faz 10s sobrevividos valerem o mesmo, seja qual for a ação tomada.
        # O total da noite (RECOMPENSA_NOITE ≈ 60) fica << vitória (500), então VENCER domina
        # "só durar" (antes o denso somava ~525 e rivalizava a vitória — durar acampado pagava
        # quase tanto quanto vencer).
        dt = getattr(self, "_dt_step", 0.7)
        recompensa = (dt / DURACAO_NOITE) * RECOMPENSA_NOITE

        # Marco por hora alcançada — densifica o OBJETIVO (sobreviver), não a estratégia: é
        # FLAT (sem peso de energia, que seria dizer "conserve"). Gerência de energia emerge
        # do terminal (ficar sem energia → Freddy → morte), não de um alvo de energia cravado.
        bonus_hora = 0.0
        for t_cp, _ in CHECKPOINTS_NOITE[1:]:
            if t_cp not in self._horas_bonificadas and self.tempo_jogo >= t_cp:
                self._horas_bonificadas.add(t_cp)
                bonus_hora += BONUS_MARCO_HORA
        self._total_bonus_hora += bonus_hora
        recompensa += bonus_hora

        # Ação SEM EFEITO (câmera trocada com a tablet fechada, porta em cooldown, porta/luz
        # durante a câmera): sinal FIEL de "isso não fez nada" — não prescreve COMO jogar, só
        # remove o ruído de ações impossíveis. Pequeno e simétrico.
        if not acao_valida:
            recompensa -= 0.1

        return recompensa

    def _capturar_observacao(self) -> dict:
        # Captura apenas a janela do jogo — mesma região usada na detecção de
        # morte/vitória e na gravação de gameplay (BC). Capturar a tela inteira
        # diluía o jogo em meio ao desktop no frame 84x84.
        frame_cinza = self._capturar_janela()
        self._atualizar_ameaca(frame_cinza)
        # Decisão 4B — energia REAL: corrige a energia simulada com a leitura visual do
        # "Power left: XX%" (photo-primary: None mantém a simulação, subida é rejeitada,
        # queda re-ancora na foto). A simulação (_atualizar_energia) preenche entre leituras.
        self.energia = validar_leitura_energia(self._ler_energia(frame_cinza), self.energia)
        frame = cv2.resize(frame_cinza, (LARGURA, ALTURA))
        frame = np.expand_dims(frame, axis=-1)

        estados = np.array([
            float(self.porta_esq),
            float(self.porta_dir),
            float(self.luz_esq),
            float(self.luz_dir),
            float(self.camera_aberta),
            float(self.camera_ativa) / 11.0,
            float(self.energia) / 100.0,
            min(self.tempo_jogo / 535.0, 1.0),
            float(self.ameaca_esq),
            float(self.ameaca_dir),
            float(self.noite) / MAX_NOITE,        # Decisão 7: dificuldade da noite
        ], dtype=np.float32)

        return {"imagem": frame, "estados": estados}

    def _carregar_templates(self):
        refs = Path(__file__).parent.parent / "utils" / "referencias"

        def _ler_primeira_existente(*nomes):
            for nome in nomes:
                caminho = refs / nome
                if caminho.exists():
                    imagem = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)
                    if imagem is not None:
                        return imagem, nome
            return None, None

        morte_img, morte_nome = _ler_primeira_existente("morte.png", "morte.jpg", "morte.jpeg")
        vitoria_img, vitoria_nome = _ler_primeira_existente("vitoria.png", "vitoria.jpg", "vitoria.jpeg")

        camera_img, camera_nome = _ler_primeira_existente("camera_aberta.png")
        if camera_img is not None:
            self.template_camera_aberta = camera_img
            print(f"Template de câmera carregado: {camera_nome}")
        else:
            self.template_camera_aberta = None
            print("Template de câmera não encontrado — execute: python -m src.utils.calibrar camera_aberta")

        faltando = []
        if morte_img is None:
            faltando.append("morte.(png/jpg)")
        if vitoria_img is None:
            faltando.append("vitoria.(png/jpg)")

        if faltando:
            raise FileNotFoundError(
                "Imagens de referência não encontradas em src/utils/referencias/: "
                + ", ".join(faltando)
                + ". Rode: python -m src.utils.calibrar morte e python -m src.utils.calibrar vitoria"
            )

        print(f"Referências carregadas: {morte_nome}, {vitoria_nome}")

        # Resolução das referências — o frame capturado será redimensionado para isso
        self._ref_size = (morte_img.shape[1], morte_img.shape[0])  # (w, h) = (1280, 720)

        # Recorta só o texto "Game Over" (canto inferior direito de morte.jpg)
        h, w = morte_img.shape
        self.template_morte = morte_img[int(h * 0.88):, int(w * 0.82):]

        # Recorta só o texto "6 AM" (centro de vitoria.png)
        h, w = vitoria_img.shape
        self.template_vitoria = vitoria_img[int(h * 0.38):int(h * 0.58), int(w * 0.38):int(w * 0.62)]

        # Template do MENU PRINCIPAL (Decisão 8): Bonnie/crash volta pro menu SEM a tela
        # de "Game Over", então _detectar_morte() nunca dispara e o episódio ficava preso
        # até truncar por tempo. Opcional — se faltar, a detecção fica desligada.
        # Recorta só as letras "New Game": o fundo tem estática aleatória que não
        # correlaciona entre frames, então as letras constantes é que sustentam o match.
        # Recorte justo = maior fração de pixel-letra = match mais alto. Coordenadas
        # medidas na grade (debug/menu_grade.png) em 1280x720: x 172-379, y 402-439.
        # Exclui as setas ">>" (cursor de seleção, x 104-146) — o cursor pode não estar
        # sempre sobre New Game, mas as letras estão sempre lá.
        menu_img, menu_nome = _ler_primeira_existente("menu.png", "menu.jpg", "menu.jpeg")
        if menu_img is not None:
            menu_img = cv2.resize(menu_img, self._ref_size)  # garante 1280x720 antes do recorte por pixel
            self.template_menu = menu_img[402:439, 172:379]
            print(f"Template de menu carregado: {menu_nome} "
                  f"({self.template_menu.shape[1]}x{self.template_menu.shape[0]})")
        else:
            self.template_menu = None
            print("Template de menu não encontrado (opcional) — rode: python -m src.utils.calibrar menu")

        # Templates de ameaça (Decisão 4A): rosto do animatrônico no vão, por lado.
        # Opcionais — se faltarem, a detecção fica desligada (não quebra o treino).
        self.template_ameaca_esq = cv2.imread(str(refs / "ameaca_esquerda.png"), cv2.IMREAD_GRAYSCALE)
        self.template_ameaca_dir = cv2.imread(str(refs / "ameaca_direita.png"), cv2.IMREAD_GRAYSCALE)

        # Glifos 0-9 do power (Decisão 4B), binarizados. Opcionais.
        self.glifos_energia = {}
        for d in "0123456789":
            g = cv2.imread(str(refs / "digitos" / f"{d}.png"), cv2.IMREAD_GRAYSCALE)
            if g is not None:
                self.glifos_energia[d] = g

    def _capturar_janela(self) -> np.ndarray:
        """Captura apenas a janela do jogo e redimensiona para a resolução de referência."""
        win = melhor_janela(WINDOW_TITLE)  # mesma seleção do script de captura
        frame = self.capture.capturar_tela(regiao_cliente(win))  # área cliente: sem barra de título

        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(cinza, self._ref_size)

    def _detectar_morte(self) -> bool:
        # Ignora detecção nos primeiros ~40s reais de episódio para evitar
        # detectar a tela de Game Over do episódio anterior, que pode persistir
        # durante a transição de reset. (Gate por tempo real, não por passos —
        # a duração de cada step varia conforme as animações.)
        if self.tempo_jogo < 40.0:
            return False
        
        frame = self._capturar_janela()
        resultado = cv2.matchTemplate(frame, self.template_morte, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(resultado)
        return float(max_val) > 0.70

    def _detectar_vitoria(self) -> bool:
        # Ignora detecção nos primeiros ~10s reais para o jogo terminar de carregar
        if self.tempo_jogo < 10.0:
            return False
        
        frame = self._capturar_janela()
        resultado = cv2.matchTemplate(frame, self.template_vitoria, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(resultado)

        if float(max_val) > 0.70:
            self.contador_vitoria += 1
        else:
            self.contador_vitoria = 0

        return self.contador_vitoria >= 3

    def _detectar_menu_crash(self) -> bool:
        """Detecta o MENU PRINCIPAL no meio do episódio (Decisão 8). O Bonnie (e alguns
        crashes) devolve o jogo ao menu SEM passar pela tela de "Game Over", então
        _detectar_morte() nunca dispara e o episódio ficava rodando sobre o menu até
        truncar por tempo (700s), desperdiçando o episódio. Aqui, no meio do episódio,
        menu = morte. (Golden Freddy fecha a janela e já é tratado no except do step.)"""
        if self.template_menu is None:
            return False
        # Gate por tempo: cobre a transição do reset (o reset já clicou New Game e esperou
        # a noite carregar; 15s dá folga para não pegar resquício de menu do reset).
        if self.tempo_jogo < 15.0:
            return False

        frame = self._capturar_janela()
        resultado = cv2.matchTemplate(frame, self.template_menu, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(resultado)

        # Threshold 0.60 = ponto de máxima margem, medido na calibração: menu ao vivo
        # (estática nova) casa ~0.95 porque as letras brilhantes dominam a correlação;
        # telas não-menu (morte/vitória) ficam ~0.25. Debounce de 2 frames p/ garantir.
        if float(max_val) > 0.60:
            self._contador_menu += 1
        else:
            self._contador_menu = 0

        return self._contador_menu >= 2

    def _match_ameaca(self, frame_cinza: np.ndarray, lado: str) -> float:
        """Score (0–1) de casamento do rosto do animatrônico (Bonnie no vão à esquerda,
        Chica na janela à direita), só na ROI daquele lado — o rosto aparece em posição
        fixa, então restringir o matchTemplate à ROI corta ~75% do custo por step. Puro
        sobre a imagem — valida offline."""
        template = self.template_ameaca_esq if lado == "esquerdo" else self.template_ameaca_dir
        if template is None:
            return 0.0
        left, top, w, h = ROI_AMEACA[lado]
        roi = frame_cinza[top:top + h, left:left + w]
        if roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
            roi = frame_cinza  # ROI menor que o template (frame reduzido) → usa o frame todo
        resultado = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(resultado)
        return float(max_val)

    def _animatronico_no_vao(self, lado: str) -> float:
        """Score do animatrônico no vão. Confiável só com a luz daquele lado
        acesa — sem luz o vão fica escuro e o score cai (tratar com debounce)."""
        return self._match_ameaca(self._capturar_janela(), lado)

    def _sombra_no_vao(self, frame_cinza: np.ndarray, lado: str = "esquerdo") -> float:
        """Desvio-padrão do cinza na região do vão (textura). Corredor VAZIO iluminado tem
        textura (bricks/correntes) → std ALTO (> LIMIAR_VAZIO = vazio confirmado). Bonnie no
        vão OU luz apagada deixam o vão escuro e liso → std BAIXO (não confirma vazio). Puro
        sobre a imagem → validável offline. Só o lado esquerdo (Bonnie)."""
        left, top, w, h = SOMBRA_REGIAO
        reg = frame_cinza[top:top + h, left:left + w]
        if reg.size == 0:
            return 0.0
        return float(reg.std())

    def _porta_fechada_visual(self, frame_cor: np.ndarray, lado: str):
        """Estado real da porta pela cor do BOTÃO DOOR (mesma convenção do
        _verificar_botao_porta): verde=fechada → True, vermelho=aberta → False,
        ambíguo → None. Precisa do frame COLORIDO (BGR)."""
        left, top, w, h = BOTAO_DOOR[lado]
        reg = frame_cor[top:top + h, left:left + w]
        if reg.size == 0:
            return None
        g, r = float(reg[:, :, 1].mean()), float(reg[:, :, 2].mean())
        if g > r + DOOR_COR_MARGEM:
            return True
        if r > g + DOOR_COR_MARGEM:
            return False
        return None

    def _aplicar_deteccao_ameaca(self, est: str, vaz: str, presente: bool) -> None:
        """Debounce comum: 'presente' fixa a ameaça e zera o contador; 'vazio' só
        limpa após DEBOUNCE_VAZIO leituras vazias seguidas — absorve o flicker da
        estática (que lê como escuro/vazio)."""
        if presente:
            setattr(self, est, True)
            setattr(self, vaz, 0)
        else:
            n = getattr(self, vaz) + 1
            setattr(self, vaz, n)
            if n >= DEBOUNCE_VAZIO:
                setattr(self, est, False)

    def _atualizar_ameaca(self, frame_cinza: np.ndarray) -> None:
        """Atualiza ameaca_esq/dir.

        Bonnie (esquerda) é um estado HELD: a sombra dele no vão é escura IGUAL à luz
        apagada, então não dá p/ detectar presença direto. O perigo só muda por
        CONFIRMAÇÃO POSITIVA, senão MANTÉM:
          - corredor VAZIO iluminado (std > LIMIAR_VAZIO) → ameaca_esq = False (saiu);
          - rosto do Bonnie casado (porta aberta)         → ameaca_esq = True;
          - sombra/escuro (std baixo, sem rosto)          → mantém (não confirma nada).
        Debounce ACUMULADO (não precisa ser consecutivo): a estática pisca p/ escuro e cai no
        'mantém', que NÃO zera os contadores — só o sinal real oposto zera. Assim a confirmação
        acumula através do flicker (antes a estática zerava e a confirmação nunca fechava).

        Chica (direita): aparece na janela com porta aberta OU fechada → rosto, só com a
        luz direita acesa; some após DEBOUNCE_VAZIO leituras vazias.

        Com a CÂMERA ABERTA o frame é o MAPA da câmera, não o escritório — ler ameaça dali é
        ruído (limpa/seta errado) e tornava acampar na câmera falsamente "seguro" (Φ zerava a
        ameaça). Então segura o último estado derivado do escritório: a ameaça não some só
        porque o jogador ergueu a tablet, igual ao jogo real."""
        if self.camera_aberta:
            return
        # Bonnie (esquerda) — estado HELD, sem gate de luz (o "vazio confirmado" e o "rosto"
        # já exigem luz acesa; no escuro nada confirma e o estado se mantém).
        if self._sombra_no_vao(frame_cinza, "esquerdo") > LIMIAR_VAZIO:    # vazio confirmado
            self._presenca_esq = 0                                         # zera o oposto (real)
            self._vazio_esq += 1
            if self._vazio_esq >= DEBOUNCE_VAZIO:
                self.ameaca_esq = False
                self._vazio_esq = 0
        elif self._match_ameaca(frame_cinza, "esquerdo") > LIMIAR_AMEACA:  # rosto confirmado
            self._vazio_esq = 0                                            # zera o oposto (real)
            self._presenca_esq += 1
            if self._presenca_esq >= DEBOUNCE_PRESENCA:
                self.ameaca_esq = True
                self._presenca_esq = 0
        # else: sombra/escuro → MANTÉM estado e contadores (a estática não reseta a confirmação)

        # Chica (direita) — depende da luz direita; independe da porta (sempre rosto)
        if self.luz_dir:
            presente_dir = self._match_ameaca(frame_cinza, "direito") > LIMIAR_AMEACA
            self._aplicar_deteccao_ameaca("ameaca_dir", "_vazio_dir", presente_dir)

    def _celula_energia(self, frame_cinza: np.ndarray, i: int) -> np.ndarray:
        x = ENERGIA_CELULAS_X[i]
        reg = frame_cinza[ENERGIA_Y[0]:ENERGIA_Y[1], x:x + ENERGIA_CELULA_W]
        _, binr = cv2.threshold(reg, 130, 255, cv2.THRESH_BINARY)
        return binr

    def _digito_celula(self, frame_cinza: np.ndarray, i: int):
        """Lê o algarismo de uma célula. Retorna (dígito|None, vazia:bool).
        vazia=True quando a célula não tem pixels (nem dígito, nem ambígua)."""
        celula = self._celula_energia(frame_cinza, i)
        if int((celula > 0).sum()) < 8:
            return None, True                          # vazia (sem dígito)
        melhor, score = None, -1.0
        for d, g in self.glifos_energia.items():
            s = float(cv2.matchTemplate(celula, g, cv2.TM_CCOEFF_NORMED).max())
            if s > score:
                score, melhor = s, d
        if score < ENERGIA_LIMIAR_GLIFO:
            return None, False                         # tem pixels mas não casou: ambíguo
        return int(melhor), False

    def _ler_energia(self, frame_cinza: np.ndarray):
        """Lê 'Power left: XX%' por template binarizado (imune ao fundo). O número é
        alinhado à direita: unidade na célula 2, dezena na 1 só quando ≥10. Devolve
        None quando não dá pra confiar (câmera/flicker/0% apagado) em vez de chutar."""
        if not self.glifos_energia:
            return None
        unidade, _ = self._digito_celula(frame_cinza, 1)      # célula 2 = unidade
        if unidade is None:
            return None                                       # sem unidade: sem leitura
        dezena, dezena_vazia = self._digito_celula(frame_cinza, 0)  # célula 1 = dezena (opcional)
        if dezena is not None:
            return dezena * 10 + unidade
        if dezena_vazia:
            return unidade                                    # 1 dígito (dezena vazia)
        return None                                           # dezena ambígua: não chuta

    def _escrever_log_desyncs(self, morreu: bool, sobreviveu: bool) -> None:
        os.makedirs("logs", exist_ok=True)
        desfecho = "vitoria" if sobreviveu else ("morte" if morreu else "truncado")
        linha = (
            f"Ep {self._episodio_num:4d} | "
            f"steps {self.passos:5d} | "
            f"desfecho: {desfecho:8s} | "
            f"SYNC camera: {self._count_sync_camera:3d} | "
            f"SYNC porta: {self._count_sync_porta:3d} | "
            f"porta falha: {self._count_porta_falha:3d}\n"
        )
        with open(self._log_desyncs_path, "a", encoding="utf-8") as f:
            f.write(linha)

    def render(self):
        pass

    def close(self):
        pass