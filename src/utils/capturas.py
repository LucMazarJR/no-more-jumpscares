"""Capturas HD das noites do AGENTE, para treinar a percepção por aprendizado mais tarde.

O env grava cada step (1280x720, JPEG) numa pasta da noite. No fim da noite, três camadas
decidem o que fica:

  1. EVENTOS (sinais que o env já tem): morte, Golden Freddy, apagão, queda de energia (Foxy),
     prancheta abaixada pelo jogo, porta que não respondeu, Bonnie/Chica na porta, vitória.
     Noites ≥ 3 inteiras. Vão para notaveis/ e ficam.
  2. NOVIDADE: cada quadro é comparado (k-NN em miniaturas) com um banco de quadros normais do
     mesmo contexto (câmera mostrada, luz, escritório). O que foge do padrão vai para
     notaveis/<noite>/novidade/ — pega Foxy saindo, pôster do Golden Freddy, "IT'S ME", palco
     esvaziando, sem regra escrita à mão.
  3. AMOSTRA DO NORMAL: poucos quadros por contexto e noite (o modelo precisa saber o que é "ok").
     Os mesmos alimentam o banco da novidade.

As 4 últimas noites ficam inteiras em recentes/; o resto sai por rotação e por cota de disco.

Uso:
  python -m src.utils.capturas resumo [--ultimas N]
  python -m src.utils.capturas reprocessar [dados/gameplay_*]   (roda nas demos HD; semeia o banco)
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import sys
import threading
import time
import zlib
from pathlib import Path

import cv2
import numpy as np

RAIZ_PADRAO = Path("capturas")
MANTER_RECENTES = 4
QUALIDADE_JPEG = 90
TAMANHO_HD = (1280, 720)
FILA_MAX = 64                 # quadros esperando o encode; cheia = descarta (nunca segura o step)

# Classe do quadro (medido nas 3 demos HD de nov/2026)
LIMIAR_BRANCO = 230
LINHAS_BLIP = 20              # troca/abertura de câmera: faixas de linhas inteiras brancas. Nas demos,
                              # quadro normal tem 0–4 dessas linhas (borda da câmera); blip, ≥ 75
LIMIAR_ESTATICA = 58.0        # |laplaciano| médio em 320x180: estática ≥ 60; câmeras p99 ≤ 49
LIMIAR_CANTO_CLARO = 20.0     # canto do mapa no 84x84 com câmera fechada (= _cortar_jumpscare)

# Novidade
MINI = (64, 36)
K_VIZINHO = 3
BANCO_MAX = 400
BANCO_MIN = 20                # abaixo disto o contexto não julga novidade
NOVIDADE_FATOR = 1.2
NOVIDADE_PISO = 12.0          # distância RMS mínima (escala 0-255) para contar como novo
NOVIDADE_MAX_NOITE = 30
NOVIDADE_INTERVALO_S = 3.0
NOVIDADE_INICIO_S = 25.0      # a noite surgindo (fade da tela) virava "novo" no escritório nas demos
AMOSTRAS_POR_CONTEXTO = 5
BANCO_POR_NOITE = 30

# Eventos: (segundos antes, segundos depois)
JANELA_MORTE = 20.0
JANELA_VITORIA = 5.0
JANELA_QUEDA = 8.0
JANELA_SYNC = 5.0
JANELA_PORTA = 10.0
QUEDA_ENERGIA_MIN = 4.0
MAX_POR_TIPO = 10
NOITE_INTEIRA_A_PARTIR = 3
DESFECHOS_MORTE = {"morte_animatronico", "morte_energia", "menu_crash", "morte_golden"}

NOMES_CAMERA = {1: "1A", 2: "1B", 3: "1C", 4: "2A", 5: "2B", 6: "3", 7: "4A", 8: "4B",
                9: "5", 10: "6", 11: "7"}


# ── classificação e contexto ────────────────────────────────────────────────────────────────

def classificar_quadro(cinza: np.ndarray, camera_aberta: bool) -> str:
    """"transicao" (blip da troca de câmera), "estatica" (pós-morte, câmera interrompida),
    "claro_fechada" (jumpscare ou fim da animação da prancheta) ou "normal"."""
    if ((cinza > LIMIAR_BRANCO).mean(axis=1) > 0.9).sum() >= LINHAS_BLIP:
        return "transicao"
    p = cv2.resize(cinza, (320, 180), interpolation=cv2.INTER_AREA).astype(np.float32)
    if np.abs(cv2.Laplacian(p, cv2.CV_32F)).mean() > LIMIAR_ESTATICA:
        return "estatica"
    if not camera_aberta and cv2.resize(cinza, (84, 84))[50:80, 52:82].mean() > LIMIAR_CANTO_CLARO:
        return "claro_fechada"
    return "normal"


def contexto(meta: dict) -> str:
    if meta.get("energia", 100) <= 0:
        return "apagao"
    if meta.get("camera_aberta"):
        return "cam_" + NOMES_CAMERA.get(meta.get("camera_mostrada") or 1, "1A")
    portas = ("_pe" if meta.get("porta_esq") else "") + ("_pd" if meta.get("porta_dir") else "")
    if meta.get("luz_esq"):
        return "luz_esq" + ("_pe" if meta.get("porta_esq") else "")
    if meta.get("luz_dir"):
        return "luz_dir" + ("_pd" if meta.get("porta_dir") else "")
    return "escritorio" + portas


def miniatura(cinza: np.ndarray) -> np.ndarray:
    m = cv2.resize(cinza, MINI, interpolation=cv2.INTER_AREA).astype(np.float32)
    m = cv2.GaussianBlur(m, (3, 3), 0)
    return (m - m.mean()).ravel()      # tira o brilho global (piscar da luz, estática fraca)


# ── camada 1: eventos ───────────────────────────────────────────────────────────────────────

def _intervalo(ts: np.ndarray, t0: float, t1: float) -> tuple[int, int] | None:
    idx = np.where((ts >= t0) & (ts <= t1))[0]
    return (int(idx[0]), int(idx[-1])) if len(idx) else None


def janelas_de_eventos(metas: list[dict], desfecho: str | None) -> list[tuple[str, int, int]]:
    """Lista (tipo, i0, i1), índices inclusivos em `metas`. Janelas do mesmo tipo que se tocam
    viram uma; no máximo MAX_POR_TIPO por tipo."""
    if not metas:
        return []
    ts = np.array([m["t"] for m in metas], dtype=float)
    fim = ts[-1]
    brutas: list[tuple[str, int, int]] = []

    def add(tipo, t0, t1):
        r = _intervalo(ts, t0, t1)
        if r:
            brutas.append((tipo, *r))

    if desfecho in DESFECHOS_MORTE:
        add(desfecho, fim - JANELA_MORTE, fim)
    elif desfecho and desfecho.startswith("vitoria"):
        add(desfecho, fim - JANELA_VITORIA, fim)

    zerou = next((k for k, m in enumerate(metas) if m.get("energia", 100) <= 0), None)
    if zerou is not None:
        add("apagao", ts[zerou] - JANELA_VITORIA, fim)

    pendentes_ameaca: dict[str, int] = {}
    for k, m in enumerate(metas):
        t = ts[k]
        if k and metas[k - 1].get("energia", 100) - m.get("energia", 100) >= QUEDA_ENERGIA_MIN:
            add("queda_energia", t - JANELA_QUEDA, t + JANELA_QUEDA)
        if m.get("sync_camera"):
            add("prancheta_abaixada", t - JANELA_SYNC, t + JANELA_SYNC)
        if m.get("porta_falhou"):
            add("porta_falhou", t - JANELA_PORTA, t + JANELA_PORTA)
        for lado in ("esq", "dir"):
            chave = f"ameaca_{lado}"
            if m.get(chave) and not (k and metas[k - 1].get(chave)):
                pendentes_ameaca[lado] = 3
            if pendentes_ameaca.get(lado) and m.get(f"luz_{lado}"):
                brutas.append((f"ameaca_{lado}", k, k))
                pendentes_ameaca[lado] -= 1

    if metas[0].get("noite", 1) >= NOITE_INTEIRA_A_PARTIR:
        brutas.append(("noite_inteira", 0, len(metas) - 1))

    # junta janelas do mesmo tipo que se sobrepõem ou encostam
    janelas: list[tuple[str, int, int]] = []
    for tipo in dict.fromkeys(b[0] for b in brutas):
        doms = sorted((a, b) for t, a, b in brutas if t == tipo)
        juntas = [list(doms[0])]
        for a, b in doms[1:]:
            if a <= juntas[-1][1] + 1:
                juntas[-1][1] = max(juntas[-1][1], b)
            else:
                juntas.append([a, b])
        janelas += [(tipo, a, b) for a, b in juntas[:MAX_POR_TIPO]]
    return janelas


# ── camada 2: novidade ──────────────────────────────────────────────────────────────────────

def _dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distância RMS entre as linhas de a e de b (matriz len(a) × len(b))."""
    a2 = (a * a).sum(1)[:, None]
    b2 = (b * b).sum(1)[None, :]
    d2 = np.maximum(a2 + b2 - 2.0 * a @ b.T, 0.0) / a.shape[1]
    return np.sqrt(d2)


def score_novidade(vetores: np.ndarray, banco: np.ndarray) -> np.ndarray:
    """Distância ao K-ésimo vizinho no banco (nan se o banco é pequeno demais)."""
    if len(vetores) == 0:
        return np.zeros(0)
    if banco is None or len(banco) < BANCO_MIN:
        return np.full(len(vetores), np.nan)
    d = _dist(vetores.astype(np.float32), banco.astype(np.float32))
    return np.partition(d, K_VIZINHO - 1, axis=1)[:, K_VIZINHO - 1]


def limiar_banco(banco: np.ndarray) -> float:
    """p99 do score leave-one-out do próprio banco × NOVIDADE_FATOR, com piso absoluto."""
    if banco is None or len(banco) < BANCO_MIN:
        return float("inf")
    b = banco.astype(np.float32)
    d = _dist(b, b)
    np.fill_diagonal(d, np.inf)
    loo = np.partition(d, K_VIZINHO - 1, axis=1)[:, K_VIZINHO - 1]
    return max(float(np.percentile(loo, 99)) * NOVIDADE_FATOR, NOVIDADE_PISO)


def escolher_novidades(ts, ctxs, scores, limiares, elegivel) -> list[int]:
    """Índices acima do limiar do seu contexto, os mais fora do padrão primeiro, um por
    NOVIDADE_INTERVALO_S, no máximo NOVIDADE_MAX_NOITE."""
    razao = np.array([s / limiares[c] if (e and np.isfinite(s) and np.isfinite(limiares[c])) else 0.0
                      for s, c, e in zip(scores, ctxs, elegivel)])
    escolhidos: list[int] = []
    for i in np.argsort(-razao):
        if razao[i] <= 1.0 or len(escolhidos) >= NOVIDADE_MAX_NOITE:
            break
        if all(abs(ts[i] - ts[j]) >= NOVIDADE_INTERVALO_S for j in escolhidos):
            escolhidos.append(int(i))
    return sorted(escolhidos)


# ── camada 3: amostra do normal ─────────────────────────────────────────────────────────────

def escolher_espalhados(ts, ctxs, normais, por_contexto: int) -> dict[str, list[int]]:
    """Até `por_contexto` índices por contexto, espalhados no tempo, entre os `normais`."""
    saida: dict[str, list[int]] = {}
    for c in sorted(set(ctxs)):
        idx = [i for i in range(len(ts)) if ctxs[i] == c and normais[i]]
        if not idx:
            continue
        pos = np.unique(np.linspace(0, len(idx) - 1, min(por_contexto, len(idx))).round().astype(int))
        saida[c] = [idx[p] for p in pos]
    return saida


# ── disco: rotação e cota ───────────────────────────────────────────────────────────────────

def rotacionar(pasta_recentes: Path, manter: int = MANTER_RECENTES, proteger: Path | None = None) -> list[Path]:
    """Apaga as noites mais antigas de recentes/ (o nome começa pela data), mantendo `manter`."""
    if not pasta_recentes.exists():
        return []
    noites = sorted(p for p in pasta_recentes.iterdir() if p.is_dir() and p != proteger)
    apagar = noites[:-manter] if manter else noites
    for p in apagar:
        shutil.rmtree(p, ignore_errors=True)
    return apagar


def uso_disco(raiz: Path) -> int:
    """Bytes ocupados sob `raiz`, contando hardlinks uma vez só."""
    vistos, total = set(), 0
    for dirpath, _, arquivos in os.walk(raiz):
        for nome in arquivos:
            try:
                st = os.stat(os.path.join(dirpath, nome))
            except OSError:
                continue
            chave = (st.st_dev, st.st_ino) if st.st_ino else os.path.join(dirpath, nome)
            if chave not in vistos:
                vistos.add(chave)
                total += st.st_size
    return total


def candidatos_cota(raiz: Path, proteger: Path | None = None) -> list[Path]:
    """Ordem de remoção quando a cota estoura: noites recentes (antigas primeiro), novidades
    antigas, noites inteiras antigas, e só então os eventos antigos (morte, Golden, apagão)."""
    def noites(sub):
        p = raiz / sub
        return sorted(x for x in p.iterdir() if x.is_dir()) if p.exists() else []

    ordem = [p for p in noites("recentes") if p != proteger]
    notaveis = noites("notaveis")
    ordem += [n / "novidade" for n in notaveis if (n / "novidade").exists()]
    ordem += [n / "noite_inteira" for n in notaveis if (n / "noite_inteira").exists()]
    ordem += notaveis
    return ordem


def aplicar_cota(raiz: Path, limite_bytes: int, proteger: Path | None = None) -> list[Path]:
    apagadas = []
    for alvo in candidatos_cota(raiz, proteger):
        if uso_disco(raiz) <= limite_bytes:
            break
        shutil.rmtree(alvo, ignore_errors=True)
        apagadas.append(alvo)
    return apagadas


def _linkar(origem: Path, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists():
        return
    try:
        os.link(origem, destino)          # hardlink: não ocupa disco enquanto a origem existir
    except OSError:
        shutil.copy2(origem, destino)


# ── folha de contato ────────────────────────────────────────────────────────────────────────

def folha_de_contato(itens: list[tuple[Path, str]], destino: Path, colunas: int = 6) -> None:
    if not itens:
        return
    w, h = 320, 180
    tiles = []
    for caminho, rotulo in itens[:48]:
        im = cv2.imread(str(caminho))
        im = np.zeros((h, w, 3), np.uint8) if im is None else cv2.resize(im, (w, h))
        cv2.rectangle(im, (0, 0), (w, 22), (0, 0, 0), -1)
        cv2.putText(im, rotulo[:40], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(im)
    while len(tiles) % colunas:
        tiles.append(np.zeros((h, w, 3), np.uint8))
    linhas = [np.hstack(tiles[k:k + colunas]) for k in range(0, len(tiles), colunas)]
    destino.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destino), np.vstack(linhas), [cv2.IMWRITE_JPEG_QUALITY, 85])


# ── gravador ────────────────────────────────────────────────────────────────────────────────

class GravadorCapturas:
    """Grava os quadros da noite numa thread (o step não espera o encode) e, no fim da noite,
    seleciona eventos, novidades e amostras. Nada aqui pode derrubar o treino: falhas viram aviso."""

    def __init__(self, raiz: Path | str = RAIZ_PADRAO, cota_gb: float = 10.0, sincrono: bool = False,
                 banco_por_noite: int = BANCO_POR_NOITE):
        self.raiz = Path(raiz)
        self.banco_por_noite = banco_por_noite
        self.limite = int(cota_gb * 1024 ** 3)
        self.sincrono = sincrono
        self.pasta: Path | None = None
        self.desfecho: str | None = None
        self._i = 0
        self._descartados = 0
        self._minis: dict[int, np.ndarray] = {}
        self._fila: queue.Queue = queue.Queue(maxsize=FILA_MAX)
        self._fechamento: threading.Thread | None = None
        self._avisou = False
        if not sincrono:
            threading.Thread(target=self._trabalhar, daemon=True).start()

    # -- ciclo da noite --
    def iniciar_noite(self, ep: int, noite: int, nome: str | None = None) -> None:
        self._esperar_fechamento()
        nome = nome or f"{time.strftime('%Y%m%d_%H%M%S')}_ep{ep:04d}_n{noite}"
        self.pasta = self.raiz / "recentes" / nome
        (self.pasta / "quadros").mkdir(parents=True, exist_ok=True)
        self.desfecho = None
        self._i = 0
        self._descartados = 0
        self._minis = {}

    def registrar(self, frame_bgr: np.ndarray | None, meta: dict, origem: Path | None = None) -> None:
        """`origem`: JPEG já existente (demos) — vira hardlink em vez de encode."""
        if self.pasta is None or (frame_bgr is None and origem is None):
            return
        item = (self.pasta, self._i, frame_bgr, dict(meta), origem)
        self._i += 1
        if self.sincrono:
            self._processar(item)
            return
        try:
            self._fila.put_nowait(item)
        except queue.Full:
            self._descartados += 1

    def marcar_desfecho(self, desfecho: str | None) -> None:
        if self.pasta is not None:
            self.desfecho = desfecho

    def encerrar_noite(self, assincrono: bool = False) -> None:
        if self.pasta is None:
            return
        pasta, desfecho, minis = self.pasta, self.desfecho, None
        self.pasta = None
        if not self.sincrono:
            self._fila.join()
        minis, self._minis = self._minis, {}
        if assincrono and not self.sincrono:
            self._fechamento = threading.Thread(
                target=self._fechar_seguro, args=(pasta, desfecho, minis, self._descartados), daemon=True)
            self._fechamento.start()
        else:
            self._fechar_seguro(pasta, desfecho, minis, self._descartados)

    def _esperar_fechamento(self) -> None:
        if self._fechamento is not None:
            self._fechamento.join()
            self._fechamento = None

    # -- worker --
    def _trabalhar(self) -> None:
        while True:
            item = self._fila.get()
            try:
                self._processar(item)
            except Exception as erro:
                self._avisar(f"captura falhou: {erro}")
            finally:
                self._fila.task_done()

    def _processar(self, item) -> None:
        pasta, i, frame, meta, origem = item
        arquivo = pasta / "quadros" / f"{i:06d}.jpg"
        if origem is not None:
            _linkar(Path(origem), arquivo)
            cinza = cv2.imread(str(origem), cv2.IMREAD_GRAYSCALE)
        else:
            if frame.shape[1::-1] != TAMANHO_HD:
                frame = cv2.resize(frame, TAMANHO_HD, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(arquivo), frame, [cv2.IMWRITE_JPEG_QUALITY, QUALIDADE_JPEG])
            cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if cinza is None:
            return
        meta.update(i=i, arquivo=arquivo.name,
                    classe=classificar_quadro(cinza, bool(meta.get("camera_aberta"))))
        self._minis[i] = miniatura(cinza).astype(np.float16)
        with open(pasta / "meta.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    def _avisar(self, msg: str) -> None:
        if not self._avisou:
            print(f"[capturas] {msg} (avisos seguintes omitidos)")
            self._avisou = True

    # -- fim da noite --
    def _fechar_seguro(self, pasta, desfecho, minis, descartados) -> None:
        try:
            self._fechar(pasta, desfecho, minis, descartados)
        except Exception as erro:
            self._avisar(f"fechamento da noite falhou: {erro}")

    def _carregar_banco(self, ctx: str) -> tuple[np.ndarray | None, int]:
        p = self.raiz / "banco" / f"{ctx}.npz"
        if not p.exists():
            return None, 0
        z = np.load(p)
        return z["vetores"], int(z["vistos"])

    def _salvar_banco(self, ctx: str, novos: list[np.ndarray], rng) -> None:
        banco, vistos = self._carregar_banco(ctx)
        banco = list(banco) if banco is not None else []
        for v in novos:                    # amostragem reservatório: o banco não envelhece num só dia
            vistos += 1
            if len(banco) < BANCO_MAX:
                banco.append(v)
            else:
                j = int(rng.integers(0, vistos))
                if j < BANCO_MAX:
                    banco[j] = v
        (self.raiz / "banco").mkdir(parents=True, exist_ok=True)
        np.savez(self.raiz / "banco" / f"{ctx}.npz", vetores=np.array(banco, dtype=np.float16), vistos=vistos)

    def _fechar(self, pasta: Path, desfecho, minis, descartados) -> dict:
        meta_path = pasta / "meta.jsonl"
        metas = [json.loads(l) for l in open(meta_path, encoding="utf-8")] if meta_path.exists() else []
        metas = [m for m in metas if m["i"] in minis]
        metas.sort(key=lambda m: m["i"])
        if not metas:                      # reset que falhou, noite sem nenhum step
            shutil.rmtree(pasta, ignore_errors=True)
            return {}
        nome = pasta.name
        notaveis = self.raiz / "notaveis" / nome
        folha: list[tuple[Path, str]] = []
        resumo = {"noite": nome, "desfecho": desfecho or "?", "quadros": len(metas),
                  "descartados": descartados, "eventos": {}, "novidades": [], "blip_pos_troca": None}

        if metas:
            quadro = lambda m: pasta / "quadros" / m["arquivo"]
            ts = np.array([m["t"] for m in metas], dtype=float)
            ctxs = [contexto(m) for m in metas]
            classes = [m.get("classe") for m in metas]
            vet = np.array([minis[m["i"]] for m in metas], dtype=np.float32)

            # 1. eventos
            for tipo, a, b in janelas_de_eventos(metas, desfecho):
                resumo["eventos"][tipo] = resumo["eventos"].get(tipo, 0) + 1
                destino = notaveis / ("noite_inteira" if tipo == "noite_inteira" else f"{tipo}_{ts[a]:05.0f}s")
                for m in metas[a:b + 1]:
                    _linkar(quadro(m), destino / m["arquivo"])
                if tipo != "noite_inteira":
                    for k in sorted({a, (a + b) // 2, b}):
                        folha.append((quadro(metas[k]), f"{tipo} t={ts[k]:.0f}s"))

            # 2. novidade (contra o banco de noites ANTERIORES; só depois esta noite entra nele)
            normal = np.array([c == "normal" for c in classes])
            elegivel_novidade = normal & (ts >= NOVIDADE_INICIO_S)
            scores = np.full(len(metas), np.nan)
            limiares = {}
            for c in set(ctxs):
                idx = [k for k in range(len(metas)) if ctxs[k] == c]
                banco, _ = self._carregar_banco(c)
                limiares[c] = limiar_banco(banco)
                scores[idx] = score_novidade(vet[idx], banco)
            novas = escolher_novidades(ts, ctxs, scores, limiares, elegivel_novidade)
            for k in novas:
                rot = f"{ctxs[k]}_{ts[k]:05.0f}s_x{scores[k] / limiares[ctxs[k]]:.1f}"
                _linkar(quadro(metas[k]), notaveis / "novidade" / f"{rot}.jpg")
                folha.append((quadro(metas[k]), f"novo {ctxs[k]} t={ts[k]:.0f}s x{scores[k] / limiares[ctxs[k]]:.1f}"))
                resumo["novidades"].append(f"{ctxs[k]}@{ts[k]:.0f}s")

            # 3. amostra do normal + banco (sem ameaça, sem blip, sem novidade)
            sem_ameaca = np.array([not (m.get("ameaca_esq") or m.get("ameaca_dir")) for m in metas])
            dentro = np.array([not (np.isfinite(s) and s > limiares[c]) for s, c in zip(scores, ctxs)])
            ok = normal & sem_ameaca & dentro
            for c, idx in escolher_espalhados(ts, ctxs, ok, AMOSTRAS_POR_CONTEXTO).items():
                for k in idx:
                    _linkar(quadro(metas[k]), self.raiz / "amostras" / c / f"{nome}_{metas[k]['i']:06d}.jpg")
            rng = np.random.default_rng(zlib.crc32(nome.encode()))
            for c, idx in escolher_espalhados(ts, ctxs, ok, self.banco_por_noite).items():
                self._salvar_banco(c, [vet[k].astype(np.float16) for k in idx], rng)

            # o que o agente VÊ depois de trocar de câmera: o step captura ~0,35 s após o clique
            trocas = [m for m in metas if str(m.get("acao", "")).startswith("camera_") and m.get("camera_aberta")]
            if trocas:
                resumo["blip_pos_troca"] = round(sum(m.get("classe") == "transicao" for m in trocas) / len(trocas), 2)

        if metas:
            notaveis.mkdir(parents=True, exist_ok=True)
            shutil.copy2(meta_path, notaveis / "meta.jsonl")
            folha_de_contato(folha, notaveis / "folha.jpg")
        self._escrever_resumo(resumo)
        rotacionar(self.raiz / "recentes")
        aplicar_cota(self.raiz, self.limite)
        return resumo

    def _escrever_resumo(self, r: dict) -> None:
        self.raiz.mkdir(parents=True, exist_ok=True)
        eventos = ", ".join(f"{k}×{v}" for k, v in r["eventos"].items()) or "-"
        blip = "-" if r["blip_pos_troca"] is None else f"{r['blip_pos_troca']:.0%}"
        linha = (f"| {r['noite']} | {r['desfecho']} | {r['quadros']} | {eventos} | "
                 f"{len(r['novidades'])}: {' '.join(r['novidades'][:8])} | {blip} | {r['descartados']} |\n")
        caminho = self.raiz / "resumo.md"
        novo = not caminho.exists()
        with open(caminho, "a", encoding="utf-8") as f:
            if novo:
                f.write("| noite | desfecho | quadros | eventos | novidades | blip pós-troca | descartados |\n"
                        "|---|---|---|---|---|---|---|\n")
            f.write(linha)
        with open(self.raiz / "indice.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[capturas] {r['noite']}: {eventos} | novidades {len(r['novidades'])} | blip pós-troca {blip}")


# ── linha de comando ────────────────────────────────────────────────────────────────────────

def _metas_da_demo(dados: list[dict]) -> list[dict]:
    """Converte registros do gravador de demos no formato de meta do env."""
    metas, cam = [], 1
    for r in dados:
        if r.get("camera_ativa"):
            cam = r["camera_ativa"]
        metas.append({
            "t": r["tempo_ep"], "noite": r.get("noite", 1), "acao": r.get("nome"),
            "energia": r.get("energia", 100), "porta_esq": r.get("porta_esq"), "porta_dir": r.get("porta_dir"),
            "luz_esq": r.get("luz_esq"), "luz_dir": r.get("luz_dir"), "camera_aberta": r.get("camera_aberta"),
            "camera_mostrada": cam, "ameaca_esq": r.get("ameaca_esq"), "ameaca_dir": r.get("ameaca_dir"),
        })
    return metas


def reprocessar(pastas: list[str], raiz: Path = RAIZ_PADRAO) -> None:
    """Roda a seleção nas demos HD. Quadro de demo é normal conferido por gente, então cada uma
    semeia o banco com mais exemplos (150 por contexto) do que uma noite do agente."""
    g = GravadorCapturas(raiz, sincrono=True, banco_por_noite=150)
    for p in pastas:
        p = Path(p)
        dados = json.load(open(p / "dataset.json", encoding="utf-8"))
        if not dados or "frame_hd" not in dados[0]:
            print(f"{p.name}: sem frames_hd, pulando")
            continue
        g.iniciar_noite(0, dados[0].get("noite", 1), nome="demo_" + p.name.replace("gameplay_", ""))
        for r, m in zip(dados, _metas_da_demo(dados)):
            origem = p / "frames_hd" / Path(r["frame_hd"]).name
            if origem.exists():
                g.registrar(None, m, origem=origem)
        venceu = dados[-1]["tempo_ep"] >= 520
        g.marcar_desfecho("vitoria_gerida" if venceu else "morte_animatronico")
        g.encerrar_noite()


def resumo(ultimas: int, raiz: Path = RAIZ_PADRAO) -> None:
    caminho = raiz / "resumo.md"
    if not caminho.exists():
        print("Nenhuma noite capturada ainda.")
        return
    linhas = open(caminho, encoding="utf-8").read().splitlines()
    print("\n".join(linhas[:2] + linhas[2:][-ultimas:]))
    print(f"\nUso de disco: {uso_disco(raiz) / 1024 ** 3:.2f} GB em {raiz}/")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "reprocessar":
        import glob
        alvos = args[1:] or sorted(glob.glob("dados/gameplay_*"))
        reprocessar(alvos)
    elif args and args[0] == "resumo":
        n = int(args[args.index("--ultimas") + 1]) if "--ultimas" in args else 10
        resumo(n)
    else:
        print(__doc__)
