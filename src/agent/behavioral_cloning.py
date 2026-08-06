import json
import random
from collections import Counter, defaultdict

import numpy as np
import cv2
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from stable_baselines3 import PPO
from src.environment.fnaf_env import FNAFEnv, MAX_NOITE, ACOES, FOXY_SATURACAO_S, INFO_SATURACAO_S
from src.agent.multimodal_policy import MultimodalExtractor
from pathlib import Path

NUM_ACOES = 17


def _reconstruir_idade_info(dados: list[dict]) -> int:
    """Preenche idade_info_esq/dir (estados 13-14, run 3) em datasets gravados ANTES deles.

    Proxy honesto da regra do env (que re-ancora a idade a cada leitura DEFINITIVA do lado):
    aqui a âncora é "luz do lado acesa com a câmera fechada" (a leitura definitiva exige luz
    nos dois lados) ou uma TRANSIÇÃO da flag de ameaça (transição implica confirmação).
    O relógio parte de "fresco" no 1º frame (espelha o reset do env) e anda por tempo_ep.
    Grava segundos CRUS — __getitem__ normaliza por INFO_SATURACAO_S (fonte única).
    Datasets novos (gravador run 3) já trazem os campos → retorna 0 sem tocar em nada."""
    if not dados or "idade_info_esq" in dados[0]:
        return 0
    t_conf = {"esq": 0.0, "dir": 0.0}
    ameaca_prev = {"esq": None, "dir": None}
    for d in dados:
        t = float(d.get("tempo_ep", 0.0))
        cam_fechada = not d.get("camera_aberta", 0)
        for lado in ("esq", "dir"):
            ameaca = d.get(f"ameaca_{lado}")
            if (cam_fechada and d.get(f"luz_{lado}", 0)) or \
               (ameaca_prev[lado] is not None and ameaca != ameaca_prev[lado]):
                t_conf[lado] = t
            ameaca_prev[lado] = ameaca
            d[f"idade_info_{lado}"] = round(max(t - t_conf[lado], 0.0), 2)
    return len(dados)


def _imagem_do_frame(dado: dict) -> np.ndarray:
    """(84,84,1) uint8 — espelha FNAFEnv._capturar_observacao(). Frame ausente = preto."""
    frame = cv2.imread(dado["frame"], cv2.IMREAD_GRAYSCALE)
    if frame is None:
        frame = np.zeros((84, 84), dtype=np.uint8)
    return np.expand_dims(frame, axis=-1)


def _estados_do_frame(dado: dict) -> np.ndarray:
    """14 estados normalizados — FONTE ÚNICA compartilhada pelo BC feedforward e o recorrente
    (espelha FNAFEnv._capturar_observacao()). Datasets antigos sem os campos usam valores
    neutros: energia=100, tempo=0, demais=0. Divisores importados do env (se o env mudar, o
    BC acompanha)."""
    return np.array([
        float(dado.get("porta_esq", 0)),
        float(dado.get("porta_dir", 0)),
        float(dado.get("luz_esq", 0)),
        float(dado.get("luz_dir", 0)),
        float(dado.get("camera_aberta", 0)),
        float(dado.get("camera_ativa", 0)) / 11.0,
        float(dado.get("energia", 100)) / 100.0,
        min(float(dado.get("tempo_ep", 0)) / 535.0, 1.0),
        float(dado.get("ameaca_esq", 0)),
        float(dado.get("ameaca_dir", 0)),
        float(dado.get("noite", 1)) / MAX_NOITE,
        min(float(dado.get("tempo_sem_camera", 0.0)) / FOXY_SATURACAO_S, 1.0),  # 12º
        min(float(dado.get("idade_info_esq", 0.0)) / INFO_SATURACAO_S, 1.0),    # 13º (run 3)
        min(float(dado.get("idade_info_dir", 0.0)) / INFO_SATURACAO_S, 1.0),    # 14º (run 3)
    ], dtype=np.float32)


class GameplayDataset(Dataset):
    def __init__(self, caminhos_json: list[str]):
        self.dados = []

        reconstruidos = 0
        for caminho in caminhos_json:
            with open(caminho, "r") as f:
                dados = json.load(f)
                reconstruidos += _reconstruir_idade_info(dados)   # datasets pré-run-3 (12 estados)
                self.dados.extend(dados)
                print(f"Carregado: {caminho} ({len(dados)} frames)")

        print(f"\nTotal combinado: {len(self.dados)} frames")
        if reconstruidos:
            print(f"[run 3] idade_info_esq/dir RECONSTRUÍDA em {reconstruidos} frames "
                  f"(datasets gravados antes dos estados 13-14 — proxy: tempo desde a última "
                  f"luz acesa do lado com câmera fechada)")

        acoes_reais = [d for d in self.dados if d["acao"] != 0]
        print(f"Frames com ação real: {len(acoes_reais)}")

        sem_estados = sum(1 for d in self.dados if "porta_esq" not in d)
        if sem_estados:
            pct = sem_estados / len(self.dados) * 100
            print(f"\n[AVISO] {sem_estados} frames ({pct:.1f}%) sem estados internos "
                  f"— campos ausentes usarão valores neutros (estados=zeros, energia=100).")
            print("  Use a versão atualizada de gravar_gameplay.py para novos datasets.\n")

        # Campos NOVOS do gravador corrigido. Sem eles o BC ensina errado: sem "noite" tudo
        # vira Noite 1 (a política aprende a IGNORAR a dificuldade); sem "tempo_sem_camera"
        # o 12º estado fica 0 no BC e ativo no RL (distribuições divergem). Regrave.
        for campo in ("noite", "tempo_sem_camera", "ameaca_esq"):
            faltando = sum(1 for d in self.dados if campo not in d)
            if faltando:
                pct = faltando / len(self.dados) * 100
                print(f"[AVISO] {faltando} frames ({pct:.1f}%) sem '{campo}' — dataset ANTIGO. "
                      f"Regrave com: python -m src.utils.gravar_gameplay --noite N")

        contagem = Counter(d["nome"] for d in self.dados)
        print("\nDistribuição de ações:")
        for nome, qtd in contagem.most_common():
            print(f"  {nome}: {qtd}")

    def __len__(self):
        return len(self.dados)

    def __getitem__(self, idx):
        dado = self.dados[idx]
        obs = {
            "imagem": torch.ByteTensor(_imagem_do_frame(dado)),
            "estados": torch.FloatTensor(_estados_do_frame(dado)),
        }
        return obs, torch.LongTensor([int(dado["acao"])])[0]


def _split_estratificado(dataset: GameplayDataset, frac_val: float = 0.10, seed: int = 42):
    """Separa 10% p/ validação POR CLASSE (senão as ações raras — portas — caem todas
    no treino e o recall de validação delas fica indefinido). Classes com < 2 exemplos
    ficam inteiras no treino."""
    por_classe: dict[int, list[int]] = defaultdict(list)
    for i, d in enumerate(dataset.dados):
        por_classe[int(d["acao"])].append(i)

    rng = random.Random(seed)
    treino, val = [], []
    for indices in por_classe.values():
        rng.shuffle(indices)
        n_val = int(len(indices) * frac_val) if len(indices) >= 2 else 0
        val.extend(indices[:n_val])
        treino.extend(indices[n_val:])
    return treino, val


def _sampler_balanceado(dataset: GameplayDataset, indices: list[int]) -> WeightedRandomSampler:
    """Peso 1/freq(classe), CAPADO em 10× a mediana dos pesos de classe: o oceano de
    "nada" deixa de afogar as ações raras (portas), sem descartar dados nem deixar uma
    classe de 3 exemplos dominar o gradiente (o cap segura o extremo)."""
    contagem = Counter(int(dataset.dados[i]["acao"]) for i in indices)
    peso_classe = {c: 1.0 / n for c, n in contagem.items()}
    cap = 10.0 * float(np.median(list(peso_classe.values())))
    peso_classe = {c: min(w, cap) for c, w in peso_classe.items()}
    pesos = [peso_classe[int(dataset.dados[i]["acao"])] for i in indices]
    return WeightedRandomSampler(pesos, num_samples=len(indices), replacement=True)


@torch.no_grad()
def _avaliar(policy, loader, device) -> tuple[float, float, dict[int, float]]:
    """(top-1, macro-recall, recall por classe) no conjunto de validação, com argmax
    determinístico — mede o que a política FARIA, não a probabilidade média."""
    certos_por_classe: Counter = Counter()
    total_por_classe:  Counter = Counter()
    certos = total = 0
    for obs_batch, acoes in loader:
        obs_device = {
            "imagem": obs_batch["imagem"].to(device),
            "estados": obs_batch["estados"].to(device),
        }
        acoes = acoes.to(device)
        dist = policy.get_distribution(obs_device)
        pred = dist.distribution.probs.argmax(dim=1)
        certos += int((pred == acoes).sum().item())
        total += len(acoes)
        for a, p in zip(acoes.tolist(), pred.tolist()):
            total_por_classe[a] += 1
            if a == p:
                certos_por_classe[a] += 1

    recall = {c: certos_por_classe[c] / n for c, n in total_por_classe.items()}
    top1 = certos / total if total else 0.0
    macro = sum(recall.values()) / len(recall) if recall else 0.0
    return top1, macro, recall


def treinar_bc(caminhos_json: list[str], epochs: int = 200, lr: float = 1e-3,
               paciencia: int = 15):
    print("=== Behavioral Cloning ===\n")

    dataset = GameplayDataset(caminhos_json)
    idx_treino, idx_val = _split_estratificado(dataset)
    print(f"\nSplit estratificado: {len(idx_treino)} treino / {len(idx_val)} validação")

    loader = DataLoader(
        Subset(dataset, idx_treino), batch_size=32,
        sampler=_sampler_balanceado(dataset, idx_treino),   # balanceado (sem shuffle)
    )
    loader_val = DataLoader(Subset(dataset, idx_val), batch_size=64)

    print("\nCriando modelo PPO com arquitetura multimodal...")
    env = FNAFEnv()

    policy_kwargs = dict(
        features_extractor_class=MultimodalExtractor,
    )

    # O learning_rate do PPO aqui é irrelevante: o treino de BC usa um Adam próprio (lr
    # abaixo) — o PPO só fornece a arquitetura da política e o formato de checkpoint .zip.
    modelo = PPO(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        device="auto",
    )

    policy    = modelo.policy
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    print(f"\nTreinando por até {epochs} épocas (early stop: {paciencia} sem melhora "
          f"no macro-recall da validação)...\n")

    melhor_macro = -1.0
    melhor_estado = None
    sem_melhora = 0

    for epoch in range(epochs):
        total_loss  = 0
        total_certo = 0
        total       = 0

        for obs_batch, acoes in loader:
            # Move observações para device
            obs_device = {
                "imagem": obs_batch["imagem"].to(modelo.device),
                "estados": obs_batch["estados"].to(modelo.device)
            }
            acoes = acoes.to(modelo.device)

            distribution = policy.get_distribution(obs_device)
            log_probs    = distribution.log_prob(acoes)
            loss         = -log_probs.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            with torch.no_grad():
                acoes_pred = distribution.distribution.probs.argmax(dim=1)
                total_certo += (acoes_pred == acoes).sum().item()
                total       += len(acoes)

        acuracia = total_certo / total * 100
        top1_val, macro_val, recall = _avaliar(policy, loader_val, modelo.device)
        print(f"Época {epoch+1:3d}/{epochs} | Loss: {total_loss/len(loader):.4f} | "
              f"Acurácia treino: {acuracia:.1f}% | Val top-1: {top1_val*100:.1f}% | "
              f"Val macro-recall: {macro_val*100:.1f}%")

        if macro_val > melhor_macro:
            melhor_macro = macro_val
            melhor_estado = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
            sem_melhora = 0
            print("  Recall por classe (validação):")
            for c in sorted(recall):
                print(f"    {ACOES.get(c, c):<20} {recall[c]*100:5.1f}%")
        else:
            sem_melhora += 1
            if sem_melhora >= paciencia:
                print(f"\nEarly stop: {paciencia} épocas sem melhora no macro-recall.")
                break

    # O checkpoint salvo é o MELHOR da validação, não o último (o fim do treino já
    # está memorizando o conjunto de treino — macro-recall do val é o critério).
    if melhor_estado is not None:
        policy.load_state_dict(melhor_estado)
    print(f"\nMelhor macro-recall (validação): {melhor_macro*100:.1f}%")
    print("Critérios de aceite (plano): portas >= 60%, câmera >= 50%, luzes >= 40%, top-1 >= 70%.")

    caminho_saida = "modelos/fnaf_bc.zip"
    modelo.save(caminho_saida)
    print(f"Modelo BC salvo em: {caminho_saida}")

    env.close()
    return modelo


def _carregar_episodios(caminhos_json: list[str]) -> list[dict]:
    """Cada dataset.json é UM episódio (noite contínua). Devolve tensores por episódio,
    preservando a ordem temporal (o BC recorrente precisa da sequência inteira)."""
    episodios = []
    for caminho in sorted(caminhos_json):
        with open(caminho, "r") as f:
            dados = json.load(f)
        _reconstruir_idade_info(dados)   # datasets pré-run-3: reconstrói idade_info_esq/dir
        episodios.append({
            "imagem":  torch.from_numpy(np.stack([_imagem_do_frame(d) for d in dados])),   # (T,84,84,1) uint8
            "estados": torch.from_numpy(np.stack([_estados_do_frame(d) for d in dados])),  # (T,14) float32
            "acoes":   torch.from_numpy(np.array([int(d["acao"]) for d in dados], dtype=np.int64)),
            "noite":   int(dados[0].get("noite", 1)),
            "nome":    Path(caminho).parent.name,
        })
        print(f"Carregado: {caminho} ({len(dados)} frames, noite {episodios[-1]['noite']})")
    return episodios


def _pesos_classe(episodios: list[dict], cap_mult: float = 10.0,
                  modo: str = "sqrt") -> torch.Tensor:
    """Peso por classe da perda do BC (não dá p/ reamostrar frames soltos sem quebrar a sequência
    da LSTM, então o balanceamento vira peso na perda), capado em cap_mult× a mediana.

    `modo` controla a AGRESSIVIDADE — e isso é um trade-off real descoberto na auditoria:
      • "inv"  = 1/freq: equilibra as classes ao máximo, mas empurra a política pra DISTRIBUIÇÃO
        UNIFORME (entropia → ln17 = 2.83). Foi o que produziu clones indecisos (H≈2.4-2.8),
        inúteis como warmstart — o RL SORTEIA a ação, então H alto = agente quase aleatório.
      • "sqrt" = 1/sqrt(freq) (DEFAULT): ainda tira as ações raras do fundo do poço, mas mantém
        o clone perto da distribuição REAL do humano — que é o objetivo de um warmstart (imitar),
        não o de um classificador balanceado (maximizar macro-recall).
      • "nenhum" = máxima verossimilhança pura: fidelidade máxima ao humano, mas o oceano de
        "nada" (86% dos frames) pode afogar portas/câmeras.
    Se o clone não passar no aceite por H alto, use um modo mais brando; se não passar por macro
    baixo, use um mais agressivo."""
    cont = Counter()
    for ep in episodios:
        cont.update(ep["acoes"].tolist())
    if modo == "nenhum":
        return torch.ones(NUM_ACOES)
    expoente = 1.0 if modo == "inv" else 0.5
    peso = {c: 1.0 / (n ** expoente) for c, n in cont.items()}
    cap = cap_mult * float(np.median(list(peso.values())))
    vec = torch.ones(NUM_ACOES)
    for c, w in peso.items():
        vec[c] = min(w, cap)
    return vec


def _estado_lstm_zero(policy, device):
    """Estado oculto zerado (h, c) do LSTM do ator, n_seq=1 — início de episódio."""
    forma = (policy.lstm_actor.num_layers, 1, policy.lstm_actor.hidden_size)
    return (torch.zeros(forma, device=device), torch.zeros(forma, device=device))


def _chunks_episodio(ep: dict, chunk: int, device):
    """Itera os chunks de BPTT truncado de um episódio. `starts` marca 1 no PRIMEIRO frame do
    episódio (reset do estado oculto) e 0 no resto — a LSTM propaga a memória dentro do episódio."""
    T = ep["acoes"].shape[0]
    for ini in range(0, T, chunk):
        fim = min(ini + chunk, T)
        obs = {"imagem": ep["imagem"][ini:fim].to(device),
               "estados": ep["estados"][ini:fim].to(device)}
        starts = torch.zeros(fim - ini, device=device)
        if ini == 0:
            starts[0] = 1.0
        yield obs, starts, ep["acoes"][ini:fim].to(device)


@torch.no_grad()
def _avaliar_recorrente(policy, ep: dict, device, chunk: int):
    """(top-1, macro-recall, entropia, recall por classe) num episódio de validação,
    teacher-forced com argmax determinístico e o estado oculto propagado como no jogo.

    A ENTROPIA média da política é medida junto porque o RL SORTEIA a ação (não usa argmax):
    um clone com top-1 alto mas H≈2.8 (uniforme) sorteia errado quase sempre e não serve de
    warmstart. É o número que denuncia um clone inútil antes de gastar dias de treino."""
    h = _estado_lstm_zero(policy, device)
    total, certo = Counter(), Counter()
    soma_h, n_h = 0.0, 0
    for obs, starts, acoes in _chunks_episodio(ep, chunk, device):
        dist, h = policy.get_distribution(obs, h, starts)
        h = (h[0].detach(), h[1].detach())
        probs = dist.distribution.probs
        soma_h += float((-(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=1)).sum())
        n_h += probs.shape[0]
        pred = probs.argmax(dim=1)
        for a, p in zip(acoes.tolist(), pred.tolist()):
            total[a] += 1
            certo[a] += int(a == p)
    recall = {c: certo[c] / n for c, n in total.items()}
    top1 = sum(certo.values()) / max(sum(total.values()), 1)
    macro = sum(recall.values()) / len(recall) if recall else 0.0
    entropia = soma_h / max(n_h, 1)
    return top1, macro, entropia, recall


def _score_clone(top1: float, macro: float) -> float:
    """Média harmônica de top-1 e macro-recall — o critério de seleção do checkpoint.

    Selecionar por macro-recall PURO (como fazíamos) escolhe modelos DEGENERADOS: quem spamma
    uma classe rara ganha 100% de recall nela e 0 no resto, pontuando mais que um modelo
    equilibrado. Foi exatamente o que aconteceu (clone salvo com macro 12.8% mas top-1 2.8%,
    argmax preso em luz_direita; baseline 'sempre nada' = 86.5%). A harmônica exige os DOIS:
    imitar o humano na maioria dos frames E não ignorar as ações raras."""
    if top1 <= 0.0 or macro <= 0.0:
        return 0.0
    return 2.0 * top1 * macro / (top1 + macro)


def treinar_bc_recorrente(caminhos_json: list[str], epochs: int = 200, lr: float = 1e-3,
                          paciencia: int = 15, chunk: int = 256, restarts: int = 4,
                          modo_pesos: str = "sqrt"):
    """BC RECORRENTE (run 3): clona as demos como SEQUÊNCIAS numa política RecurrentPPO, com
    teacher forcing e BPTT truncado. Diferente do BC feedforward (embaralha frames soltos → só o
    extractor transfere pra LSTM), aqui a própria LSTM + cabeças são treinadas, então o warmstart
    transfere ~100% pra run LSTM — mata o cold-start que deixou a run 3 aleatória (H~2.0 a 175k).

    Estado oculto zera no início de cada episódio e propaga (detached) entre chunks de `chunk`
    frames (BPTT truncado: mais janelas de gradiente, sem retropropagar ~1500 passos de uma vez).
    Desbalanceamento (portas raras) via PERDA PONDERADA por classe.

    `restarts`: com POUCAS sequências (~5 episódios), o resultado depende muito da inicialização
    aleatória (uma init ruim estagna a perda ~2.3 em vez de ~1.7 → clone de 13% em vez de 37%).
    Treina `restarts` vezes com inits independentes e FICA COM O MELHOR macro-recall — troca a
    loteria por um clone confiável. É offline/barato; suba se a variância ainda incomodar."""
    from sb3_contrib import RecurrentPPO

    print("=== Behavioral Cloning RECORRENTE (LSTM) ===\n")
    episodios = _carregar_episodios(caminhos_json)
    if len(episodios) < 2:
        raise SystemExit("BC recorrente precisa de >= 2 episódios (datasets). Grave mais noites.")

    # Split POR EPISÓDIO (estratificar por classe quebraria a sequência): segura 1 episódio de
    # validação da noite com MAIS gravações (removê-lo ainda deixa a noite no treino); empate →
    # o de menos ações (menos custoso remover). Preserva a diversidade de noites no treino.
    noites = Counter(ep["noite"] for ep in episodios)
    noite_val = max(noites, key=lambda n: (noites[n], -n))
    candidatos = [i for i, ep in enumerate(episodios) if ep["noite"] == noite_val]
    idx_val = min(candidatos, key=lambda i: int((episodios[i]["acoes"] != 0).sum()))
    val = episodios[idx_val]
    treino = [ep for i, ep in enumerate(episodios) if i != idx_val]
    print(f"\nSplit por episódio: {len(treino)} treino / 1 validação "
          f"('{val['nome']}', noite {val['noite']})")

    pesos_cpu = _pesos_classe(episodios, modo=modo_pesos)
    print(f"Balanceamento de classes: modo '{modo_pesos}' "
          f"(sqrt = equilibra sem empurrar o clone pra uniforme)")

    print("\nCriando RecurrentPPO (LSTM hidden 128, 1 camada, critic_lstm) — mesma arquitetura da run...")
    env = FNAFEnv()
    policy_kwargs = dict(
        features_extractor_class=MultimodalExtractor,
        lstm_hidden_size=128, n_lstm_layers=1, enable_critic_lstm=True,
    )

    def _treinar_uma_vez():
        """Uma inicialização completa (fresh model). Retorna (melhor_macro, melhor_estado_cpu,
        modelo). O modelo volta com os pesos do MELHOR epoch já carregados."""
        modelo = RecurrentPPO("MultiInputLstmPolicy", env=env,
                              policy_kwargs=policy_kwargs, verbose=0, device="auto")
        policy, device = modelo.policy, modelo.device
        pesos = pesos_cpu.to(device)
        optimizer = optim.Adam(policy.parameters(), lr=lr)
        melhor_score, melhor_info, melhor_estado, sem_melhora = -1.0, None, None, 0
        for epoch in range(epochs):
            random.shuffle(treino)
            perda_epoca, n_chunks = 0.0, 0
            for ep in treino:
                h = _estado_lstm_zero(policy, device)
                for obs, starts, acoes in _chunks_episodio(ep, chunk, device):
                    dist, h = policy.get_distribution(obs, h, starts)
                    w = pesos[acoes]
                    perda = -(w * dist.log_prob(acoes)).sum() / w.sum().clamp_min(1e-8)  # NLL ponderada
                    optimizer.zero_grad()
                    perda.backward()
                    optimizer.step()
                    h = (h[0].detach(), h[1].detach())   # BPTT truncado
                    perda_epoca += perda.item()
                    n_chunks += 1

            top1_val, macro_val, h_val, recall = _avaliar_recorrente(policy, val, device, chunk)
            score = _score_clone(top1_val, macro_val)
            print(f"Época {epoch+1:3d}/{epochs} | Loss: {perda_epoca/max(n_chunks,1):.4f} | "
                  f"top-1: {top1_val*100:.1f}% | macro: {macro_val*100:.1f}% | "
                  f"H: {h_val:.2f} | score: {score*100:.1f}")
            if score > melhor_score:
                melhor_score = score
                melhor_info = (top1_val, macro_val, h_val)
                melhor_estado = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
                sem_melhora = 0
                print("  Recall por classe (validação):")
                for c in sorted(recall):
                    print(f"    {ACOES.get(c, c):<20} {recall[c]*100:5.1f}%")
            else:
                sem_melhora += 1
                if sem_melhora >= paciencia:
                    print(f"\nEarly stop: {paciencia} épocas sem melhora no score.")
                    break
        if melhor_estado is not None:
            policy.load_state_dict(melhor_estado)
        return melhor_score, melhor_info, modelo

    melhor_global, melhor_info, melhor_modelo = -1.0, None, None
    for r in range(restarts):
        print(f"\n===== Restart {r + 1}/{restarts} (init independente) =====")
        score, info, modelo = _treinar_uma_vez()
        top1, macro, h_clone = info if info else (0.0, 0.0, float("nan"))
        print(f"[restart {r + 1}] score {score*100:.1f} (top-1 {top1*100:.1f}% | "
              f"macro {macro*100:.1f}% | H {h_clone:.2f})")
        if score > melhor_global:
            melhor_global, melhor_info, melhor_modelo = score, info, modelo

    top1, macro, h_clone = melhor_info
    print(f"\n>>> MELHOR de {restarts} restarts: score {melhor_global*100:.1f} "
          f"(top-1 {top1*100:.1f}% | macro {macro*100:.1f}% | H {h_clone:.2f})")

    # Aceite do clone (auditoria 08/2026): o warmstart só vale se o clone IMITA (top-1) e é
    # CONFIANTE (H baixo — o RL sorteia a ação, não usa argmax). Um clone com H≈2.8 injeta uma
    # política quase uniforme e reproduz o cold-start que travou a run 3.
    problemas = []
    if top1 < 0.70:
        problemas.append(f"top-1 {top1*100:.1f}% < 70% (não imita o humano)")
    if h_clone > 1.3:
        problemas.append(f"H {h_clone:.2f} > 1.3 (clone indeciso — warmstart fraco)")
    if macro < 0.25:
        problemas.append(f"macro {macro*100:.1f}% < 25% (ignora ações raras: portas/câmeras)")
    if problemas:
        print("[AVISO] Clone ABAIXO do aceite — NÃO use pra treinar ainda:")
        for p in problemas:
            print(f"   - {p}")
        print("   Ações: gravar mais demos de NOITE 1 (frugalidade de energia) ou subir 'restarts'.")
    else:
        print("[OK] Clone dentro do aceite (top-1 >= 70%, H <= 1.3, macro >= 25%).")

    caminho_saida = "modelos/fnaf_bc.zip"
    melhor_modelo.save(caminho_saida)
    print(f"Modelo BC (recorrente) salvo em: {caminho_saida}")
    env.close()
    return melhor_modelo


def transferir_pesos(modelo_rl, caminho_origem: str = "modelos/fnaf_bc.zip") -> None:
    """Warmstart: copia os pesos da POLÍTICA de um checkpoint SB3 (.zip) para um modelo de RL
    fresco, via load_state_dict(strict=False). Reusa só o que casa por nome E shape — o
    MultimodalExtractor (CNN+MLP) SEMPRE casa; as cabeças pi/vf casam quando o net_arch coincide.

    Compatível com a LSTM: o BC treina um PPO feedforward, mas ao transferir pra um RecurrentPPO
    só o MultimodalExtractor bate (as chaves da LSTM e do mlp pós-LSTM têm shape diferente e são
    ignoradas) — ou seja, aquece a PERCEPÇÃO, que é a parte cara, e deixa a memória pro RL aprender.

    É o caminho CORRETO de combinar BC + RL: inicializar a partir dos pesos, NÃO fazer média entre
    redes de inicializações diferentes (que quebra a política — os neurônios não se correspondem).
    Substitui o antigo combinar_bc_com_ppo. É INIT, não recompensa: o RL fica livre p/ divergir."""
    from stable_baselines3.common.save_util import load_from_zip_file

    _, params, _ = load_from_zip_file(caminho_origem, device="auto")
    estado_origem = params.get("policy") if isinstance(params, dict) else None
    if estado_origem is None:
        raise RuntimeError(f"Checkpoint sem state_dict de 'policy': {caminho_origem}")

    alvo = modelo_rl.policy.state_dict()
    casados = {k: v for k, v in estado_origem.items()
               if k in alvo and alvo[k].shape == v.shape}
    modelo_rl.policy.load_state_dict(casados, strict=False)

    print(f"[BC warmstart] transferidos {len(casados)}/{len(alvo)} tensores de {caminho_origem}")
    nao_vieram = sorted({k.split('.')[0] for k in alvo if k not in casados})
    if nao_vieram:
        print(f"[BC warmstart] init aleatório (não transferido): {', '.join(nao_vieram)}")


if __name__ == "__main__":
    treinar_bc([
        "dados/gameplay_teste/dataset.json",
    ], epochs=200)

    print("\nPronto! Para usar o BC como ponto de partida do PPO:")
    print("  Rode: python main.py treino --novo --bc modelos/fnaf_bc.zip")
