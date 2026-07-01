import json
import random
from collections import Counter, defaultdict

import numpy as np
import cv2
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from stable_baselines3 import PPO
from src.environment.fnaf_env import FNAFEnv, MAX_NOITE, ACOES, FOXY_SATURACAO_S
from src.agent.multimodal_policy import MultimodalExtractor
from pathlib import Path

NUM_ACOES = 17

class GameplayDataset(Dataset):
    def __init__(self, caminhos_json: list[str]):
        self.dados = []

        for caminho in caminhos_json:
            with open(caminho, "r") as f:
                dados = json.load(f)
                self.dados.extend(dados)
                print(f"Carregado: {caminho} ({len(dados)} frames)")

        print(f"\nTotal combinado: {len(self.dados)} frames")

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

        # Carrega imagem
        frame = cv2.imread(dado["frame"], cv2.IMREAD_GRAYSCALE)
        if frame is None:
            frame = np.zeros((84, 84), dtype=np.uint8)
        frame = np.expand_dims(frame, axis=-1)  # (84, 84, 1)

        # 12 estados normalizados — espelha FNAFEnv._capturar_observacao().
        # Datasets antigos sem os campos de estado usam valores neutros:
        # energia=100 (cheio), tempo=0 (início), demais=0 (inativo). Divisores
        # importados do env (fonte única) — se o env mudar, o BC acompanha.
        estados = np.array([
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
        ], dtype=np.float32)

        acao = int(dado["acao"])

        obs = {
            "imagem": torch.ByteTensor(frame),
            "estados": torch.FloatTensor(estados)
        }
        return obs, torch.LongTensor([acao])[0]


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

    modelo = PPO(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
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
