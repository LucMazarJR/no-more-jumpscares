import gymnasium as gym
import numpy as np
import cv2
import time
import os
import subprocess
import unicodedata
from pathlib import Path
from gymnasium import spaces
from src.utils.capture import GameCapture, regiao_cliente

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
RESET_CLICK = (
    _env_int_obrigatorio("FNAF_RESET_CLICK_X"),
    _env_int_obrigatorio("FNAF_RESET_CLICK_Y"),
)
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
DEBOUNCE_VAZIO = 2    # leituras vazias seguidas (luz acesa) p/ limpar a ameaça — absorve flicker

CHECKPOINTS_NOITE = [
    (0,   100.0),
    (89,   85.0),   # 1AM
    (178,  60.0),   # 2AM
    (267,  40.0),   # 3AM
    (356,  25.0),   # 4AM
    (445,  15.0),   # 5AM
    (535,   5.0),   # 6AM
]


class FNAFEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()

        self.capture          = GameCapture()
        self.render_mode      = render_mode
        self.contador_vitoria = 0
        self._carregar_templates()

        self.observation_space = spaces.Dict({
            "imagem": spaces.Box(
                low=0, high=255,
                shape=(ALTURA, LARGURA, 1),
                dtype=np.uint8
            ),
            "estados": spaces.Box(
                low=0, high=1,
                shape=(10,),
                dtype=np.float32
            )
        })
        self.action_space = spaces.Discrete(len(ACOES))

        self.passos    = 0
        self.max_passos = 10_000
        self.energia   = 100.0
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

    def _interromper_episodio(self, motivo: str):
        if self._botao_luz_pressionado:
            try:
                self.capture.soltar_botao(*self._botao_luz_pressionado)
            except Exception:
                pass
            self._botao_luz_pressionado = None

        recuperado = self._abrir_jogo_fallback()
        if not recuperado:
            motivo = f"{motivo} (fallback sem sucesso)"

        info = {
            "passos":       self.passos,
            "energia":      self.energia,
            "tempo_real":   time.perf_counter() - (self.episode_start_time or time.perf_counter()),
            "porta_esq":    self.porta_esq,
            "porta_dir":    self.porta_dir,
            "camera_aberta": self.camera_aberta,
            "camera_ativa": self.camera_ativa,
            "morreu":       False,
            "interrompido": True,
            "ocorrido":     motivo,
        }

        observacao = {
            "imagem": np.zeros((ALTURA, LARGURA, 1), dtype=np.uint8),
            "estados": np.zeros(8, dtype=np.float32)
        }
        return observacao, 0.0, True, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if not WINDOW_TITLE:
            raise RuntimeError(
                "FNAF_WINDOW_TITLE nao configurado no .env. "
                "Configure as variaveis obrigatorias antes de executar."
            )

        self.passos           = 0
        self.energia          = 100.0
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

        if not self._janela_do_jogo_aberta():
            self._abrir_jogo_fallback()

        if not self.capture.focar_janela(WINDOW_TITLE):
            raise RuntimeError(
                "Janela do jogo nao encontrada. "
                "Configure FNAF_EXECUTABLE_PATH no .env para fallback automatico."
            )
        time.sleep(0.5)

        self.capture.clicar(*RESET_CLICK)
        time.sleep(15)
        self.capture.clicar(*RESET_CLICK)
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
            return self._interromper_episodio("janela do jogo nao encontrada")

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
                self.porta_esq = False
                self.porta_dir = False
                self.luz_esq = False
                self.luz_dir = False
                self.camera_aberta = False

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
        except Exception as erro:
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
        # Trava de segurança por tempo: a noite dura ~535s. Se a detecção de
        # vitória falhar (template), encerra em vez de ficar preso por horas.
        truncado   = self.passos >= self.max_passos or self.tempo_jogo > 700.0

        if terminado or truncado:
            self._escrever_log_desyncs(morreu, sobreviveu)

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
        }

        return observacao, recompensa, terminado, truncado, info

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

    def _calcular_recompensa(self, morreu: bool, sobreviveu: bool, acao: int, acao_valida: bool) -> float:
        # Magnitudes terminais moderadas: valores extremos (−500/+1000) dominavam
        # a função de valor e criavam alvos com variância enorme em relação às
        # recompensas por step (~0.5), dificultando a convergência do crítico.
        # O retorno continua monotônico no tempo de sobrevivência.
        if morreu:
            return -100.0

        if sobreviveu:
            return +500.0

        # Bônus por checkpoint de hora — tempo_jogo agora é o tempo real do episódio
        tempo_ep = self.tempo_jogo
        bonus_hora = 0.0
        for t_cp, e_cp in CHECKPOINTS_NOITE[1:]:
            if t_cp not in self._horas_bonificadas and tempo_ep >= t_cp:
                self._horas_bonificadas.add(t_cp)
                ratio = min(self.energia / e_cp, 1.5) if e_cp > 0 else (1.0 if self.energia > 0 else 0.0)
                bonus_hora += max(ratio * 8.0, 1.0)  # 1 a 12
        self._total_bonus_hora += bonus_hora

        if not acao_valida:
            return -0.5 + bonus_hora

        # Base de sobrevivência: cada step vivo vale mais do que morrer cedo
        recompensa = bonus_hora + 0.5

        # Bônus por progresso no tempo (incentiva sobreviver mais)
        progresso = self.tempo_jogo / 535.0
        recompensa += progresso * 0.5  # até +0.5 adicional no final da noite

        nome_acao = ACOES[acao]

        # Penalidade por ação repetida (verifica se é igual à ação do step anterior)
        if nome_acao in ["porta_esquerda", "porta_direita", "luz_esquerda", "luz_direita"]:
            if nome_acao == self.penultima_acao:
                recompensa -= 0.15  # ~1/3 da base; desencoraja sem dominar
        elif nome_acao == "nada":
            # Permite descanso até 8 steps (~2s); penaliza inação prolongada
            if self.contador_nada > 8:
                recompensa -= min((self.contador_nada - 8) * 0.05, 0.5)
        elif nome_acao == self.penultima_acao:
            # Câmera ou toggle repetidos: penaliza só quando realmente repetido
            recompensa -= 0.1

        # Pequena penalidade por usar luzes (gasta energia sem observar)
        if nome_acao in ["luz_esquerda", "luz_direita"]:
            recompensa -= 0.05

        # Penalidade por ter ambas as portas fechadas (raramente necessário)
        if self.porta_esq and self.porta_dir:
            recompensa -= 0.1

        # Penalidade por inatividade da câmera — Foxy corre a cada ~5s sem câmera
        if self.passos_sem_camera > 20:
            excesso = self.passos_sem_camera - 20
            recompensa -= min(excesso * 0.02, 0.3)

        # Penalidade por energia abaixo do esperado para o momento da noite
        deficit = max(0.0, self._energia_esperada() - self.energia)
        recompensa -= deficit * 0.02

        recompensa = max(recompensa, -1.0)

        return recompensa

    def _capturar_observacao(self) -> dict:
        # Captura apenas a janela do jogo — mesma região usada na detecção de
        # morte/vitória e na gravação de gameplay (BC). Capturar a tela inteira
        # diluía o jogo em meio ao desktop no frame 84x84.
        frame_cinza = self._capturar_janela()
        self._atualizar_ameaca(frame_cinza)
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

        # Templates de ameaça (Decisão 4A): rosto do animatrônico no vão, por lado.
        # Opcionais — se faltarem, a detecção fica desligada (não quebra o treino).
        self.template_ameaca_esq = cv2.imread(str(refs / "ameaca_esquerda.png"), cv2.IMREAD_GRAYSCALE)
        self.template_ameaca_dir = cv2.imread(str(refs / "ameaca_direita.png"), cv2.IMREAD_GRAYSCALE)

    def _capturar_janela(self) -> np.ndarray:
        """Captura apenas a janela do jogo e redimensiona para a resolução de referência."""
        import pygetwindow as gw
        janelas = gw.getWindowsWithTitle(WINDOW_TITLE)
        if not janelas:
            raise RuntimeError("janela do jogo nao encontrada")

        win = janelas[0]
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

    def _match_ameaca(self, frame_cinza: np.ndarray, lado: str) -> float:
        """Score (0–1) de casamento do animatrônico no vão do lado dado.
        Pura sobre a imagem — dá para validar offline com frames salvos."""
        template = self.template_ameaca_esq if lado == "esquerdo" else self.template_ameaca_dir
        if template is None:
            return 0.0
        resultado = cv2.matchTemplate(frame_cinza, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(resultado)
        return float(max_val)

    def _animatronico_no_vao(self, lado: str) -> float:
        """Score do animatrônico no vão. Confiável só com a luz daquele lado
        acesa — sem luz o vão fica escuro e o score cai (tratar com debounce)."""
        return self._match_ameaca(self._capturar_janela(), lado)

    def _atualizar_ameaca(self, frame_cinza: np.ndarray) -> None:
        """Atualiza ameaca_esq/dir com debounce. Só lê o lado cuja luz está acesa
        (sem luz o vão é escuro); segura o último estado com a luz apagada. Uma vez
        detectada, a ameaça só some após DEBOUNCE_VAZIO leituras vazias seguidas com
        a luz acesa — absorve o flicker da estática (que lê como escuro)."""
        for lado, luz, est, vaz in (
            ("esquerdo", self.luz_esq, "ameaca_esq", "_vazio_esq"),
            ("direito",  self.luz_dir, "ameaca_dir", "_vazio_dir"),
        ):
            if not luz:
                continue
            if self._match_ameaca(frame_cinza, lado) > LIMIAR_AMEACA:
                setattr(self, est, True)
                setattr(self, vaz, 0)
            else:
                n = getattr(self, vaz) + 1
                setattr(self, vaz, n)
                if n >= DEBOUNCE_VAZIO:
                    setattr(self, est, False)

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