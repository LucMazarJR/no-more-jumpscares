"""Roda os blocos da Fase 1 (M1/M2 e, opcional, BC+RL) em sequência, sem supervisão.

Cada bloco é um `main.py jogar ... --episodios 5` (5 eps da noite alvo). Antes de cada bloco o
jogo é REABERTO do zero (fecha, abre, ALT+ENTER p/ modo janela): garante o menu limpo mesmo se o
bloco anterior terminou numa vitória (o jogo emenda sozinho na noite seguinte) ou travou.

Ordem intercalada (M1, M2, RL, M1, M2, RL, ...): deriva de máquina ao longo da noite (temperatura,
updates) cai igual nas três configurações. Bloco que quebra é registrado e o próximo segue;
3 falhas seguidas encerram tudo.

Uso (de um PowerShell, na raiz do projeto, com o jogo FECHADO ou no menu):
    venv\\Scripts\\python scripts\\rodar_fase1.py                       # 4 rodadas: M1, M2, RL
    venv\\Scripts\\python scripts\\rodar_fase1.py --rodadas 3 --sem-rl
    venv\\Scripts\\python scripts\\rodar_fase1.py --ate 18:00           # não começa bloco que passe disso
    venv\\Scripts\\python scripts\\rodar_fase1.py --blocos M1,RL,RL340  # curva BC → 190k → 340k

Saídas: logs/analise/avaliacoes.log (resultados, lido pelo resumo_avaliacoes.py),
logs/analise/fase1_blocos.txt (console de todos os blocos + tracebacks), debug/fase1_bloco_<n>.png.
"""
import argparse
import ctypes
import hashlib
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

ctypes.windll.user32.SetProcessDPIAware()  # mesmas coordenadas físicas que o pyautogui do env

import mss
import mss.tools
import pyautogui
import pygetwindow as gw

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RAIZ)
sys.path.insert(0, RAIZ)

from src.environment.fnaf_env import GAME_EXECUTABLE_PATH, RESET_METODO, WINDOW_TITLE  # noqa: E402

PYTHON = os.path.join("venv", "Scripts", "python.exe")
BC = os.path.join("modelos", "fnaf_bc.zip")
BC_SHA256 = "74436089BB1AA395FB4D28FBA683FCF9558F3C152EFE8E1FD3F3B0E176DCFF70"
RL = os.path.join("modelos", "pc_luciano_run4", "pc4_fnaf_ppo_190000_steps.zip")  # fim da run 4
# Continuação da run 4 (10–11/ago, ~150k steps sem log): mede se o PPO seguiu subindo ou estagnou.
RL340 = os.path.join("modelos", "pc_luciano_run4", "pc4_fnaf_ppo_340000_steps.zip")
LOG = os.path.join("logs", "analise", "fase1_blocos.txt")
TELA_CALIBRADA = (1920, 1080)    # monitor primário em que o .env deste PC foi calibrado
JANELA_CALIBRADA = (312, 160)    # canto da janela do jogo em modo janela nessa tela
DURACAO_BLOCO = timedelta(minutes=75)  # 5 eps de N1 + os de N2 de brinde, com folga

CONFIGS = {
    "M1": [BC],
    "M2": [BC, "--ablacao", "imagem"],
    "RL": [RL],
    "RL340": [RL340],
}


def registrar(msg: str) -> None:
    linha = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(linha, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(linha + "\n")


def checar_ambiente(ordem: list[str]) -> list[str]:
    erros = []
    if tuple(pyautogui.size()) != TELA_CALIBRADA:
        erros.append(f"monitor primário em {tuple(pyautogui.size())}, mas o .env foi calibrado em "
                     f"{TELA_CALIBRADA}: as coordenadas dos cliques não bateriam.")
    if RESET_METODO != "continue":
        erros.append(f"FNAF_RESET_METODO={RESET_METODO} no .env (precisa ser continue).")
    if not (os.getenv("PC") or "").strip():
        erros.append("PC vazio no .env.")
    with open(BC, "rb") as f:
        if hashlib.sha256(f.read()).hexdigest().upper() != BC_SHA256:
            erros.append(f"hash do {BC} diferente do BC da Fase 1.")
    # Todo modelo da fila precisa existir ANTES de começar: um checkpoint faltando só quebraria
    # horas depois, no bloco dele, com a máquina sozinha.
    for nome in dict.fromkeys(ordem):
        if not os.path.exists(CONFIGS[nome][0]):
            erros.append(f"{CONFIGS[nome][0]} não existe (bloco {nome}).")
    return erros


def janela():
    ws = gw.getWindowsWithTitle(WINDOW_TITLE)
    return ws[0] if ws else None


def focar(w) -> None:
    """SetForegroundWindow só é aceito de quem tem o foco; um toque no ALT libera (truque padrão)."""
    if w.isMinimized:
        w.restore()
    u = ctypes.windll.user32
    u.keybd_event(0x12, 0, 0, 0)
    u.keybd_event(0x12, 0, 2, 0)
    u.SetForegroundWindow(w._hWnd)
    time.sleep(0.5)


def preparar_jogo(n_bloco: int) -> bool:
    """Fecha e reabre o jogo, põe em modo janela e confere a geometria. True se ficou pronto."""
    for tentativa in range(1, 4):
        subprocess.run(["taskkill", "/IM", os.path.basename(GAME_EXECUTABLE_PATH), "/F"],
                       capture_output=True)
        time.sleep(3)
        os.startfile(GAME_EXECUTABLE_PATH)
        time.sleep(12)
        w = janela()
        if w is None:
            registrar(f"  jogo não abriu (tentativa {tentativa}/3)")
            continue
        focar(w)
        if w.width <= 1280:  # abriu em tela cheia (resolução 1280x720) → modo janela
            pyautogui.hotkey("alt", "enter")
            time.sleep(4)
            w = janela()
            focar(w)
        pyautogui.moveTo(960, 1000)  # tira o mouse de cima dos botões do menu
        # A calibração do .env vale para a janela nesta posição exata (centralizada no primário).
        ok = (abs(w.left - JANELA_CALIBRADA[0]) <= 10 and abs(w.top - JANELA_CALIBRADA[1]) <= 10
              and w.width > 1280)
        registrar(f"  janela {w.left},{w.top} {w.width}x{w.height} | primário {tuple(pyautogui.size())} "
                  f"({'ok' if ok else f'ERRADA, esperado {JANELA_CALIBRADA}'})")
        try:
            with mss.mss() as s:
                img = s.grab({"left": w.left, "top": w.top, "width": w.width, "height": w.height})
                mss.tools.to_png(img.rgb, img.size, output=os.path.join("debug", f"fase1_bloco_{n_bloco:02d}.png"))
        except Exception:
            pass
        if ok:
            return True
    return False


def rodar_bloco(n: int, nome: str, episodios: int) -> int:
    cmd = [PYTHON, "-u", "main.py", "jogar", "--modelo", *CONFIGS[nome],
           "--estocastico", "--noite", "1", "--episodios", str(episodios)]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"\n----- bloco {n} {nome}: {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                text=True, encoding="utf-8", errors="replace")
        for linha in proc.stdout:
            sys.stdout.write(linha)
            f.write(linha)
            f.flush()
        return proc.wait()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rodadas", type=int, default=4, help="rodadas de (M1, M2[, RL])")
    ap.add_argument("--episodios", type=int, default=5)
    ap.add_argument("--sem-rl", action="store_true", help="só M1 e M2")
    ap.add_argument("--blocos", help=f"ordem de UMA rodada, separada por vírgula (opções: "
                                     f"{', '.join(CONFIGS)}); ex.: M1,RL,RL340")
    ap.add_argument("--ate", help="HH:MM: não começa bloco que termine depois disso")
    args = ap.parse_args()

    os.makedirs("debug", exist_ok=True)
    if args.blocos:
        ordem = [b.strip() for b in args.blocos.split(",") if b.strip()]
        desconhecidos = [b for b in ordem if b not in CONFIGS]
        if desconhecidos:
            sys.exit(f"[ERRO] blocos desconhecidos: {desconhecidos} (opções: {', '.join(CONFIGS)})")
    else:
        ordem = ["M1", "M2"] + ([] if args.sem_rl else ["RL"])
    fila = ordem * args.rodadas
    limite = None
    if args.ate:
        h, m = map(int, args.ate.split(":"))
        limite = datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
        if limite <= datetime.now():
            limite += timedelta(days=1)

    erros = checar_ambiente(ordem)
    if erros:
        for e in erros:
            print("[ERRO]", e)
        sys.exit(1)

    registrar(f"=== Fase 1: {len(fila)} blocos {fila} | PC {os.getenv('PC')} | até {limite or '-'}")
    falhas_seguidas = 0
    try:
        for n, nome in enumerate(fila, 1):
            if limite and datetime.now() + DURACAO_BLOCO > limite:
                registrar(f"parando: o bloco {n} passaria do limite {limite:%H:%M}")
                break
            registrar(f"bloco {n}/{len(fila)} {nome}: preparando o jogo")
            if not preparar_jogo(n):
                registrar("  não consegui deixar o jogo em modo janela; encerrando")
                break
            inicio = time.time()
            codigo = rodar_bloco(n, nome, args.episodios)
            pyautogui.mouseUp()  # se o bloco caiu segurando a luz, solta o botão
            registrar(f"bloco {n} {nome}: saída {codigo} em {(time.time() - inicio) / 60:.0f} min")
            falhas_seguidas = falhas_seguidas + 1 if codigo != 0 else 0
            if falhas_seguidas >= 3:
                registrar("3 blocos seguidos falharam; encerrando (ver tracebacks acima)")
                break
    except KeyboardInterrupt:
        pyautogui.mouseUp()
        registrar("interrompido (Ctrl+C)")

    if not os.path.exists(os.path.join("logs", "analise", "avaliacoes.log")):
        registrar("nenhum bloco gravou avaliação ainda; sem resumo")
        return
    resumo = subprocess.run([PYTHON, os.path.join("scripts", "resumo_avaliacoes.py")],
                            env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    registrar("resumo:\n" + resumo.stdout + resumo.stderr)


if __name__ == "__main__":
    main()
