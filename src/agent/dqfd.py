"""DQfD (Deep Q-learning from Demonstrations, Hester et al., 2018) — braço B do treino do zero.

FASE 1 (este arquivo, por enquanto): pré-treino OFFLINE só com as demos, para validar antes de
gastar jogo. Critério registrado (PROPOSTA_TREINO_DO_ZERO): igualar o BC na validação.

Por que DQfD aqui: o PPO usa cada amostra ~4 vezes e descarta (≈55 updates de política em 226k
steps na run 6), e as demos só entram como init, que o RL erode. O DQfD guarda tudo num replay e
deixa as demos no buffer para sempre, com uma perda de MARGEM que mantém a ação humana acima das
outras no Q enquanto o TD ensina o resto.

Transições das demos:
  • TICK: as demos têm ~0,3 s por registro e o agente ~0,9 s por step. Cada transição agrupa
    REGISTROS_POR_STEP registros: obs = o 1º, ação = a 1ª ação real da janela (senão "nada").
  • RECOMPENSA reconstruída com as mesmas fórmulas do env: denso por tempo real, marco por hora,
    shaping γΦ'−Φ (Φ de _potencial_seguranca, a partir dos campos gravados) e terminal (vitória se
    a demo chegou ao 6 AM, morte se acabou antes). A penalidade de ação inválida não entra (o
    gravador não registra validade); é −0,1 e só existe no RL.
  • Reabrir porta com a flag acesa vira "nada" (mesma regra do BC, _neutralizar_reabertura_com_flag).

Perdas (por transição de demo): TD de 1 passo (double DQN) + TD de N_PASSOS + margem grande
(MARGEM) na ação humana + L2 (weight decay). A margem é ponderada pelos pesos de classe do BC
(desvio do artigo): com 'nada' em ~80% das janelas, sem peso a margem quase só ensina 'nada'.

Rodar (sem o jogo; use a GPU):  python main.py dqfd --pre
"""
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agent.behavioral_cloning import (
    NUM_ACOES, _corrigir_caminhos_frames, _estados_do_frame, _imagem_do_frame,
    _neutralizar_reabertura_com_flag, _reconstruir_idade_info,
)
from src.agent.multimodal_policy import MultimodalExtractor
from src.environment.fnaf_env import (
    ACOES, BONUS_MARCO_HORA, CHECKPOINTS_NOITE, DURACAO_NOITE, FOXY_PACIENCIA_S, GAMMA,
    INFO_SATURACAO_S, PESO_AMEACA_BLOQUEADA, PESO_ENERGIA, PESO_FOXY, PESO_INFO,
    RECOMPENSA_MORTE, RECOMPENSA_NOITE, RECOMPENSA_VITORIA,
)

REGISTROS_POR_STEP = 3      # ~0,3 s/registro × 3 ≈ 0,9 s, o step do agente
N_PASSOS = 10               # retorno de n passos (artigo: 10)
MARGEM = 0.8                # perda de margem grande (artigo: 0,8)
LAMBDA_N = 1.0              # peso do TD de n passos
LAMBDA_E = 1.0              # peso da margem
L2 = 1e-5                   # weight decay
ESCALA_RECOMPENSA = 0.01    # sem escala (vitória 500) o Q oscilou de 3 a 54; com 0,01 fica estável
FIM_VITORIA_S = 525.0       # demo que chegou até aqui terminou no 6 AM (a noite dura ~535 s)
CAMINHO_PRE = "modelos/dqfd_pre.pt"


# ── Recompensa reconstruída a partir dos campos gravados (mesmas fórmulas do env) ─────────────
def phi_registro(d: dict) -> float:
    """Φ do env (_potencial_seguranca) calculado de um registro de demo."""
    phi = 0.0
    if d.get("ameaca_esq") and d.get("porta_esq"):
        phi += PESO_AMEACA_BLOQUEADA
    if d.get("ameaca_dir") and d.get("porta_dir"):
        phi += PESO_AMEACA_BLOQUEADA
    excesso = max(0.0, float(d.get("tempo_sem_camera", 0.0)) - FOXY_PACIENCIA_S)
    phi -= PESO_FOXY * min(excesso / FOXY_PACIENCIA_S, 1.0)
    phi += PESO_ENERGIA * (float(d.get("energia", 100.0)) / 100.0)
    for lado in ("esq", "dir"):
        phi -= PESO_INFO * min(float(d.get(f"idade_info_{lado}", 0.0)) / INFO_SATURACAO_S, 1.0)
    return phi


def recompensa_transicao(d0: dict, d1: dict | None, venceu: bool) -> float:
    """Recompensa do env entre os registros d0 e d1 (d1=None: terminal depois de d0)."""
    if d1 is None:
        return RECOMPENSA_VITORIA if venceu else RECOMPENSA_MORTE   # Φ terminal = 0 → shaping −Φ ≈ irrelevante
    t0, t1 = float(d0["tempo_ep"]), float(d1["tempo_ep"])
    r = (t1 - t0) / DURACAO_NOITE * RECOMPENSA_NOITE
    r += BONUS_MARCO_HORA * sum(1 for t_cp, _ in CHECKPOINTS_NOITE[1:] if t0 < t_cp <= t1)
    return r + GAMMA * phi_registro(d1) - phi_registro(d0)


def demo_venceu(dados: list[dict]) -> bool:
    """Chegou ao 6 AM? O relógio da demo parte do F9, então uma vitória pode acabar antes dos 535 s
    (N2 de 06/11: 521,5 s, e seria rotulada como morte). Com frames HD, vale também a tela de 6 AM
    nos últimos registros, com o template e o limiar do env. Demos sem HD: só o relógio."""
    if float(dados[-1]["tempo_ep"]) >= FIM_VITORIA_S:
        return True
    if "frame_hd" not in dados[-1]:
        return False
    import cv2
    from src.environment.fnaf_env import FNAFEnv
    template = FNAFEnv().template_vitoria
    casou = 0
    for d in dados[-8:]:
        img = cv2.imread(d["frame_hd"], cv2.IMREAD_GRAYSCALE)
        if img is not None:
            casou += cv2.minMaxLoc(cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED))[1] > 0.70
    return casou >= 3


# ── Transições das demos ─────────────────────────────────────────────────────────────────────
def carregar_transicoes(caminhos_json: list[str], fase: int = 0) -> list[dict]:
    """Um dict por demo com os tensores das transições no tick do agente (ver docstring do módulo).
    `fase` desloca o início das janelas: as fases 0, 1 e 2 juntas usam todos os registros."""
    episodios = []
    for caminho in sorted(caminhos_json):
        with open(caminho, "r") as f:
            dados = json.load(f)
        _corrigir_caminhos_frames(dados, caminho)
        _reconstruir_idade_info(dados)
        _neutralizar_reabertura_com_flag(dados)
        venceu = demo_venceu(dados)
        idx = list(range(fase, len(dados), REGISTROS_POR_STEP))
        acoes, recompensas = [], []
        for k, i in enumerate(idx):
            janela = dados[i:i + REGISTROS_POR_STEP]
            acoes.append(next((int(d["acao"]) for d in janela if int(d["acao"]) != 0), 0))
            prox = dados[idx[k + 1]] if k + 1 < len(idx) else None
            recompensas.append(recompensa_transicao(dados[i], prox, venceu))
        T = len(idx)
        recompensas = np.array(recompensas, dtype=np.float32)
        # Retorno de n passos: soma descontada das n recompensas seguintes (cortada no terminal)
        # e o índice de bootstrap (−1 se o episódio acaba antes).
        ret_n = np.zeros(T, dtype=np.float32)
        boot_n = np.full(T, -1, dtype=np.int64)
        for t in range(T):
            fim = min(t + N_PASSOS, T)
            ret_n[t] = sum(GAMMA ** (j - t) * recompensas[j] for j in range(t, fim))
            if t + N_PASSOS < T:
                boot_n[t] = t + N_PASSOS
        episodios.append({
            "nome": Path(caminho).parent.name,
            "noite": int(dados[0].get("noite", 1)),
            "minutos": float(dados[-1]["tempo_ep"]) / 60.0,
            "venceu": venceu,
            "imagem": torch.from_numpy(np.stack([_imagem_do_frame(dados[i]) for i in idx])),
            "estados": torch.from_numpy(np.stack([_estados_do_frame(dados[i]) for i in idx])),
            "acoes": torch.tensor(acoes, dtype=torch.int64),
            "recompensas": torch.from_numpy(recompensas),
            "ret_n": torch.from_numpy(ret_n),
            "boot_n": torch.from_numpy(boot_n),
        })
        print(f"Demo {episodios[-1]['nome']}: {T} transições, {'vitória' if venceu else 'morte'}, "
              f"ações reais {int((episodios[-1]['acoes'] != 0).sum())}")
    return episodios


# ── Rede Q (dueling) sobre o mesmo extractor do PPO ──────────────────────────────────────────
class RedeQ(nn.Module):
    def __init__(self, observation_space):
        super().__init__()
        self.extractor = MultimodalExtractor(observation_space)
        dim = self.extractor.features_dim
        self.valor = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.vantagem = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, NUM_ACOES))

    def forward(self, obs: dict) -> torch.Tensor:
        # O SB3 divide a imagem por 255 antes do extractor (preprocess_obs); aqui é por nossa conta.
        z = self.extractor({"imagem": obs["imagem"].float() / 255.0, "estados": obs["estados"]})
        a = self.vantagem(z)
        return self.valor(z) + a - a.mean(dim=1, keepdim=True)


def _lote(ep_list, refs, device):
    """Monta o lote a partir de (episódio, t)."""
    def pegar(chave, ts=None):
        return torch.stack([ep_list[e][chave][t if ts is None else ts[k]] for k, (e, t) in enumerate(refs)]).to(device)
    T = [ep_list[e]["acoes"].shape[0] for e, _ in refs]
    prox = [min(t + 1, T[k] - 1) for k, (_, t) in enumerate(refs)]
    feito = torch.tensor([float(t + 1 >= T[k]) for k, (_, t) in enumerate(refs)], device=device)
    boot = [int(ep_list[e]["boot_n"][t]) for e, t in refs]
    tem_boot = torch.tensor([float(b >= 0) for b in boot], device=device)
    boot = [max(b, 0) for b in boot]
    return {
        "obs": {"imagem": pegar("imagem"), "estados": pegar("estados")},
        "obs1": {"imagem": pegar("imagem", prox), "estados": pegar("estados", prox)},
        "obsn": {"imagem": pegar("imagem", boot), "estados": pegar("estados", boot)},
        "acao": pegar("acoes"), "r": pegar("recompensas"), "feito": feito,
        "ret_n": pegar("ret_n"), "tem_boot": tem_boot,
    }


def perdas_dqfd(q_net, alvo_net, b, pesos_classe, escala: float = ESCALA_RECOMPENSA,
                lambda_e: float = LAMBDA_E) -> tuple[torch.Tensor, dict]:
    """`escala` multiplica as recompensas (a vitória vale 500 e o denso ~0,1 por step: sem escala
    o Q oscilou de 3 a 54 no 1º pré-treino)."""
    q = q_net(b["obs"])
    q_a = q.gather(1, b["acao"].unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        a1 = q_net(b["obs1"]).argmax(dim=1, keepdim=True)                     # double DQN
        y1 = escala * b["r"] + GAMMA * (1 - b["feito"]) * alvo_net(b["obs1"]).gather(1, a1).squeeze(1)
        an = q_net(b["obsn"]).argmax(dim=1, keepdim=True)
        yn = escala * b["ret_n"] + (GAMMA ** N_PASSOS) * b["tem_boot"] * alvo_net(b["obsn"]).gather(1, an).squeeze(1)
    j_td = F.smooth_l1_loss(q_a, y1)
    j_n = F.smooth_l1_loss(q_a, yn)
    margem = torch.full_like(q, MARGEM)
    margem.scatter_(1, b["acao"].unsqueeze(1), 0.0)
    w = pesos_classe[b["acao"]]
    j_e = ((q + margem).max(dim=1).values - q_a)
    j_e = (w * j_e).sum() / w.sum().clamp_min(1e-8)
    total = j_td + LAMBDA_N * j_n + lambda_e * j_e
    return total, {"td": j_td.item(), "n": j_n.item(), "margem": j_e.item(), "q_medio": q_a.mean().item()}


@torch.no_grad()
def avaliar(q_net, ep: dict, device) -> dict:
    """Ação gulosa nas transições de um episódio: top-1, macro-recall e ritmo (ações/min) × humano."""
    acertos, total = Counter(), Counter()
    preds = []
    for ini in range(0, ep["acoes"].shape[0], 256):
        obs = {"imagem": ep["imagem"][ini:ini + 256].to(device), "estados": ep["estados"][ini:ini + 256].to(device)}
        preds.append(q_net(obs).argmax(dim=1).cpu())
    pred = torch.cat(preds)
    for a, p in zip(ep["acoes"].tolist(), pred.tolist()):
        total[a] += 1
        acertos[a] += int(a == p)
    recall = {c: acertos[c] / n for c, n in total.items()}
    return {"top1": sum(acertos.values()) / max(sum(total.values()), 1),
            "macro": sum(recall.values()) / max(len(recall), 1),
            "ritmo": float((pred != 0).sum()) / max(ep["minutos"], 1e-9),
            "ritmo_humano": float((ep["acoes"] != 0).sum()) / max(ep["minutos"], 1e-9),
            "recall": recall}


def pretreinar(caminhos_json: list[str], passos: int = 30_000, lote: int = 64, lr: float = 1e-4,
               alvo_cada: int = 1000, avaliar_cada: int = 1000, seed: int = 0,
               escala: float = ESCALA_RECOMPENSA, lambda_e: float = LAMBDA_E, caminho_saida: str = CAMINHO_PRE,
               fases_extra: tuple = ()) -> dict:
    """Pré-treino só com as demos. Segura 1 episódio de validação pela mesma regra do BC (a noite
    com mais demos, o episódio com menos ações reais) e guarda o melhor checkpoint pelo score
    (harmônica de top-1 e macro, a mesma do BC)."""
    from src.agent.behavioral_cloning import _pesos_classe, _score_clone
    from src.environment.fnaf_env import FNAFEnv
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    episodios = carregar_transicoes(caminhos_json)
    noites = Counter(ep["noite"] for ep in episodios)
    noite_val = max(noites, key=lambda n: (noites[n], -n))
    i_val = min((i for i, ep in enumerate(episodios) if ep["noite"] == noite_val),
                key=lambda i: int((episodios[i]["acoes"] != 0).sum()))
    val = episodios[i_val]
    treino = [ep for i, ep in enumerate(episodios) if i != i_val]
    for fase in fases_extra:   # mais transições das mesmas demos; a validação segue só na fase 0
        treino += [ep for ep in carregar_transicoes(caminhos_json, fase) if ep["nome"] != val["nome"]]
    refs = [(e, t) for e, ep in enumerate(treino) for t in range(ep["acoes"].shape[0])]
    pesos = _pesos_classe(treino).to(device)
    print(f"\n{len(refs)} transições de treino em {len(treino)} demos; validação '{val['nome']}' "
          f"| device {device}")

    espaco = FNAFEnv().observation_space
    q_net, alvo_net = RedeQ(espaco).to(device), RedeQ(espaco).to(device)
    alvo_net.load_state_dict(q_net.state_dict())
    opt = torch.optim.Adam(q_net.parameters(), lr=lr, weight_decay=L2)
    melhor, melhor_info = -1.0, None
    avaliar_cada = min(avaliar_cada, passos)
    for passo in range(1, passos + 1):
        b = _lote(treino, random.sample(refs, lote), device)
        perda, info = perdas_dqfd(q_net, alvo_net, b, pesos, escala, lambda_e)
        opt.zero_grad()
        perda.backward()
        nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
        opt.step()
        if passo % alvo_cada == 0:
            alvo_net.load_state_dict(q_net.state_dict())
        if passo % avaliar_cada == 0:
            q_net.eval()
            m = avaliar(q_net, val, device)
            q_net.train()
            score = _score_clone(m["top1"], m["macro"])
            print(f"passo {passo:6d} | td {info['td']:.3f} n {info['n']:.3f} margem {info['margem']:.3f} "
                  f"Q {info['q_medio']:7.2f} | val top-1 {m['top1']*100:5.1f}% macro {m['macro']*100:5.1f}% "
                  f"score {score*100:5.1f} | ritmo {m['ritmo']:5.1f}/min (humano {m['ritmo_humano']:.1f})")
            if score > melhor:
                melhor, melhor_info = score, m
                torch.save({"estado": q_net.state_dict(), "passo": passo, "score": score,
                            "demos": [ep["nome"] for ep in episodios], "validacao": val["nome"]},
                           caminho_saida)
    print(f"\n>>> melhor score {melhor*100:.1f} (top-1 {melhor_info['top1']*100:.1f}% | macro "
          f"{melhor_info['macro']*100:.1f}% | ritmo {melhor_info['ritmo']:.1f}/min) salvo em {caminho_saida}")
    print("Recall por classe (validação):")
    for c in sorted(melhor_info["recall"]):
        print(f"    {ACOES.get(c, c):<20} {melhor_info['recall'][c]*100:5.1f}%")
    caminho_val = next(c for c in caminhos_json if Path(c).parent.name == val["nome"])
    for rotulo, zip_bc in (("BC bruto", "modelos/fnaf_bc_bruto.zip"), ("BC calibrado", "modelos/fnaf_bc.zip")):
        if Path(zip_bc).exists():
            m = avaliar_bc_no_tick(zip_bc, caminho_val, device)
            print(f"Comparação, {rotulo} nas mesmas transições: top-1 {m['top1']*100:5.1f}% | macro "
                  f"{m['macro']*100:5.1f}% | score {_score_clone(m['top1'], m['macro'])*100:5.1f} | "
                  f"ritmo {m['ritmo']:5.1f}/min")
    return melhor_info


@torch.no_grad()
def avaliar_bc_no_tick(zip_bc: str, caminho_json: str, device) -> dict:
    """O BC (recorrente) prevê por REGISTRO; para comparar com o DQfD no tick do agente, vale a
    previsão (argmax) no 1º registro da janela, com o estado oculto propagado por todos."""
    from sb3_contrib import RecurrentPPO
    from src.agent.behavioral_cloning import _carregar_episodios, _chunks_episodio, _estado_lstm_zero
    modelo = RecurrentPPO.load(zip_bc, device=device)
    policy = modelo.policy
    policy.set_training_mode(False)
    ep = _carregar_episodios([caminho_json])[0]
    h, preds = _estado_lstm_zero(policy, modelo.device), []
    for obs, starts, _ in _chunks_episodio(ep, 256, modelo.device):
        dist, h = policy.get_distribution(obs, h, starts)
        preds.append(dist.distribution.probs.argmax(dim=1).cpu())
    pred, rot = torch.cat(preds).tolist(), ep["acoes"].tolist()
    acertos, total, n_reais = Counter(), Counter(), 0
    for i in range(0, len(rot), REGISTROS_POR_STEP):
        a = next((x for x in rot[i:i + REGISTROS_POR_STEP] if x != 0), 0)
        p = pred[i]   # só o 1º registro: é a obs que o DQfD vê (a janela inteira daria 3 chances ao BC)
        total[a] += 1
        acertos[a] += int(a == p)
        n_reais += int(p != 0)
    recall = {c: acertos[c] / n for c, n in total.items()}
    return {"top1": sum(acertos.values()) / max(sum(total.values()), 1),
            "macro": sum(recall.values()) / max(len(recall), 1),
            "ritmo": n_reais / max(ep["minutos"], 1e-9)}


# ── Versão recorrente (estilo R2D3): a mesma memória do BC ───────────────────────────────────
# Hipótese: a diferença para o BC no macro vem de o DQfD feedforward ver um frame só. Sequências
# com burn-in: os primeiros BURN_IN passos só aquecem a LSTM (sem perda), porque o estado oculto
# real daquele ponto não está guardado; ele parte de zero.
BURN_IN = 20
TREINO_SEQ = 40


class RedeQRecorrente(nn.Module):
    def __init__(self, observation_space, oculto: int = 128):
        super().__init__()
        self.extractor = MultimodalExtractor(observation_space)
        self.lstm = nn.LSTM(self.extractor.features_dim, oculto, batch_first=True)
        self.valor = nn.Linear(oculto, 1)
        self.vantagem = nn.Linear(oculto, NUM_ACOES)

    def forward(self, imagem, estados, h=None):
        """imagem (B,T,84,84,1), estados (B,T,14) → Q (B,T,NUM_ACOES), h."""
        B, T = estados.shape[:2]
        z = self.extractor({"imagem": imagem.reshape(B * T, *imagem.shape[2:]).float() / 255.0,
                            "estados": estados.reshape(B * T, -1)}).reshape(B, T, -1)
        s, h = self.lstm(z, h)
        a = self.vantagem(s)
        return self.valor(s) + a - a.mean(dim=2, keepdim=True), h


def _lote_seq(episodios, n_seq, device):
    """Janelas de BURN_IN + TREINO_SEQ + N_PASSOS, com início sorteado em todo o episódio. O que
    passa do fim repete o último passo e fica de fora da perda (máscara 0). Janela que começa em
    t=0 não precisa de burn-in: o estado zero é o estado real."""
    L = BURN_IN + TREINO_SEQ + N_PASSOS
    lote = {k: [] for k in ("imagem", "estados", "acao", "r", "ret_n", "feito", "tem_boot", "perda")}
    for _ in range(n_seq):
        ep = random.choice(episodios)
        T = ep["acoes"].shape[0]
        t0 = random.randint(0, max(T - BURN_IN - 1, 0))
        pos = torch.arange(t0, t0 + L)
        idx = pos.clamp_max(T - 1)
        lote["imagem"].append(ep["imagem"][idx])
        lote["estados"].append(ep["estados"][idx])
        lote["acao"].append(ep["acoes"][idx])
        lote["r"].append(ep["recompensas"][idx])
        lote["ret_n"].append(ep["ret_n"][idx])
        lote["feito"].append((pos >= T - 1).float())
        lote["tem_boot"].append((ep["boot_n"][idx] >= 0).float())
        na_janela = torch.zeros(L)
        na_janela[BURN_IN if t0 > 0 else 0:BURN_IN + TREINO_SEQ] = 1.0
        lote["perda"].append(na_janela * (pos < T).float())
    return {k: torch.stack(v).to(device) for k, v in lote.items()}


def perdas_recorrente(q_net, alvo_net, b, pesos_classe, escala=ESCALA_RECOMPENSA, lambda_e=LAMBDA_E):
    q, _ = q_net(b["imagem"], b["estados"])
    with torch.no_grad():
        q_alvo, _ = alvo_net(b["imagem"], b["estados"])
    L = q.shape[1]
    i = torch.arange(L, device=q.device)
    i1, iN = (i + 1).clamp_max(L - 1), (i + N_PASSOS).clamp_max(L - 1)
    q_a = q.gather(2, b["acao"].unsqueeze(2)).squeeze(2)
    with torch.no_grad():
        a1 = q[:, i1].argmax(dim=2, keepdim=True)                              # double DQN
        y1 = escala * b["r"] + GAMMA * (1 - b["feito"]) * q_alvo[:, i1].gather(2, a1).squeeze(2)
        an = q[:, iN].argmax(dim=2, keepdim=True)
        yn = escala * b["ret_n"] + (GAMMA ** N_PASSOS) * b["tem_boot"] * q_alvo[:, iN].gather(2, an).squeeze(2)
    m = b["perda"]
    n = m.sum().clamp_min(1.0)
    j_td = (F.smooth_l1_loss(q_a, y1, reduction="none") * m).sum() / n
    j_n = (F.smooth_l1_loss(q_a, yn, reduction="none") * m).sum() / n
    margem = torch.full_like(q, MARGEM)
    margem.scatter_(2, b["acao"].unsqueeze(2), 0.0)
    w = pesos_classe[b["acao"]] * m
    j_e = (w * ((q + margem).max(dim=2).values - q_a)).sum() / w.sum().clamp_min(1e-8)
    total = j_td + LAMBDA_N * j_n + lambda_e * j_e
    return total, {"td": j_td.item(), "n": j_n.item(), "margem": j_e.item(),
                   "q_medio": ((q_a * m).sum() / n).item()}


@torch.no_grad()
def avaliar_recorrente(q_net, ep: dict, device) -> dict:
    """Como avaliar(), mas propagando o estado oculto pelo episódio inteiro, como no jogo."""
    acertos, total, preds, h = Counter(), Counter(), [], None
    for ini in range(0, ep["acoes"].shape[0], 256):
        q, h = q_net(ep["imagem"][ini:ini + 256].unsqueeze(0).to(device),
                     ep["estados"][ini:ini + 256].unsqueeze(0).to(device), h)
        preds.append(q[0].argmax(dim=1).cpu())
    pred = torch.cat(preds)
    for a, p in zip(ep["acoes"].tolist(), pred.tolist()):
        total[a] += 1
        acertos[a] += int(a == p)
    recall = {c: acertos[c] / n for c, n in total.items()}
    return {"top1": sum(acertos.values()) / max(sum(total.values()), 1),
            "macro": sum(recall.values()) / max(len(recall), 1),
            "ritmo": float((pred != 0).sum()) / max(ep["minutos"], 1e-9),
            "ritmo_humano": float((ep["acoes"] != 0).sum()) / max(ep["minutos"], 1e-9),
            "recall": recall}


def pretreinar_recorrente(caminhos_json: list[str], passos: int = 10_000, n_seq: int = 16, lr: float = 1e-4,
                          alvo_cada: int = 500, avaliar_cada: int = 500, seeds: tuple = (0, 1, 2, 3),
                          caminho_saida: str = "modelos/dqfd_pre_lstm.pt") -> dict:
    """Pré-treino recorrente, com o mesmo split e o mesmo score do feedforward. Quatro seeds, porque
    o BC fica com o melhor de 4 inits: a comparação precisa dar a mesma chance aos dois."""
    from src.agent.behavioral_cloning import _pesos_classe, _score_clone
    from src.environment.fnaf_env import FNAFEnv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    episodios = carregar_transicoes(caminhos_json)
    noites = Counter(ep["noite"] for ep in episodios)
    noite_val = max(noites, key=lambda n: (noites[n], -n))
    i_val = min((i for i, ep in enumerate(episodios) if ep["noite"] == noite_val),
                key=lambda i: int((episodios[i]["acoes"] != 0).sum()))
    val = episodios[i_val]
    treino = [ep for i, ep in enumerate(episodios) if i != i_val]
    pesos = _pesos_classe(treino).to(device)
    espaco = FNAFEnv().observation_space
    print(f"\nvalidação '{val['nome']}' | {len(treino)} demos de treino | device {device}", flush=True)
    melhor_global, melhor_info = -1.0, None
    for seed in seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        q_net, alvo_net = RedeQRecorrente(espaco).to(device), RedeQRecorrente(espaco).to(device)
        alvo_net.load_state_dict(q_net.state_dict())
        opt = torch.optim.Adam(q_net.parameters(), lr=lr, weight_decay=L2)
        melhor = -1.0
        for passo in range(1, passos + 1):
            perda, info = perdas_recorrente(q_net, alvo_net, _lote_seq(treino, n_seq, device), pesos)
            opt.zero_grad()
            perda.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            opt.step()
            if passo % alvo_cada == 0:
                alvo_net.load_state_dict(q_net.state_dict())
            if passo % avaliar_cada == 0:
                m = avaliar_recorrente(q_net, val, device)
                score = _score_clone(m["top1"], m["macro"])
                melhor = max(melhor, score)
                print(f"seed {seed} passo {passo:6d} | td {info['td']:.3f} n {info['n']:.3f} margem "
                      f"{info['margem']:.3f} Q {info['q_medio']:6.2f} | val top-1 {m['top1']*100:5.1f}% "
                      f"macro {m['macro']*100:5.1f}% score {score*100:5.1f} | ritmo {m['ritmo']:5.1f}/min "
                      f"(humano {m['ritmo_humano']:.1f})", flush=True)
                if score > melhor_global:
                    melhor_global, melhor_info = score, m
                    torch.save({"estado": q_net.state_dict(), "seed": seed, "passo": passo, "score": score,
                                "demos": [ep["nome"] for ep in episodios], "validacao": val["nome"]},
                               caminho_saida)
        print(f">>> seed {seed}: melhor score {melhor*100:.1f}", flush=True)
    print(f"\n>>> MELHOR GERAL score {melhor_global*100:.1f} (top-1 {melhor_info['top1']*100:.1f}% | macro "
          f"{melhor_info['macro']*100:.1f}% | ritmo {melhor_info['ritmo']:.1f}/min) salvo em {caminho_saida}")
    for c in sorted(melhor_info["recall"]):
        print(f"    {ACOES.get(c, c):<20} {melhor_info['recall'][c]*100:5.1f}%")
    return melhor_info


# ── Jogar com a rede pré-treinada (`main.py jogar --modelo modelos/<...>.pt`) ────────────────
EPSILON_JOGO = 0.01         # com --estocastico: a política de exploração mínima do DQN


class AgenteDQfD:
    """Adaptador com a interface .predict do SB3, para o `jogar` reaproveitar trace e log. Só a
    rede sem memória (RedeQ). Gulosa; com deterministic=False, ε-greedy com EPSILON_JOGO. O DQN não
    sorteia como o PPO: sem ε, um estado repetido dá sempre a mesma ação."""

    def __init__(self, caminho: str, observation_space):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(caminho, map_location=self.device)
        self.rede = RedeQ(observation_space).to(self.device)
        self.rede.load_state_dict(ck["estado"])
        self.rede.eval()

    @torch.no_grad()
    def predict(self, obs: dict, deterministic: bool = True):
        if not deterministic and random.random() < EPSILON_JOGO:
            return np.array(random.randrange(NUM_ACOES)), None
        t = {"imagem": torch.as_tensor(obs["imagem"], device=self.device).unsqueeze(0),
             "estados": torch.as_tensor(obs["estados"], device=self.device).unsqueeze(0)}
        return np.array(int(self.rede(t).argmax(dim=1))), None
