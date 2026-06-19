import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from stable_baselines3 import PPO
from src.environment.fnaf_env import FNAFEnv, MAX_NOITE
from src.agent.multimodal_policy import MultimodalExtractor
from pathlib import Path
from collections import Counter

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

        # 10 estados normalizados — espelha FNAFEnv._capturar_observacao().
        # Datasets antigos sem os campos de estado usam valores neutros:
        # energia=100 (cheio), tempo=0 (início), demais=0 (inativo). A ameaça
        # (Decisão 4A) não é gravada e o frame salvo é 84x84 (pequeno p/ o template
        # 1280x720), então fica neutra=0 no BC; o RL a computa ao vivo.
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
            float(dado.get("noite", 1)) / MAX_NOITE,   # Decisão 7; datasets antigos → noite 1
        ], dtype=np.float32)

        acao = int(dado["acao"])
        
        obs = {
            "imagem": torch.ByteTensor(frame),
            "estados": torch.FloatTensor(estados)
        }
        return obs, torch.LongTensor([acao])[0]


def treinar_bc(caminhos_json: list[str], epochs: int = 50, lr: float = 1e-3):
    print("=== Behavioral Cloning ===\n")

    dataset = GameplayDataset(caminhos_json)
    loader  = DataLoader(dataset, batch_size=32, shuffle=True)

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

    print(f"\nTreinando por {epochs} épocas...\n")

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
        print(f"Época {epoch+1:3d}/{epochs} | Loss: {total_loss/len(loader):.4f} | Acurácia: {acuracia:.1f}%")

    caminho_saida = "modelos/fnaf_bc.zip"
    modelo.save(caminho_saida)
    print(f"\nModelo BC salvo em: {caminho_saida}")

    env.close()
    return modelo


def combinar_bc_com_ppo(caminho_bc: str = "modelos/fnaf_bc.zip",
                         caminho_ppo: str = "modelos/fnaf_merged.zip"):
    """DESCONTINUADO: média de pesos entre o modelo BC e um modelo PPO
    treinado de outra inicialização NÃO combina o aprendizado — os neurônios
    das duas redes não se correspondem e o resultado é uma política quebrada.

    O fluxo correto é usar o modelo BC diretamente como ponto de partida do
    PPO: treine o BC, copie/renomeie modelos/fnaf_bc.zip para a pasta
    modelos/ como único modelo e rode `python main.py treino` — o treino
    continua a partir dele.
    """
    raise RuntimeError(
        "combinar_bc_com_ppo foi descontinuado: a média de pesos entre redes "
        "de inicializações diferentes quebra a política. Continue o treino "
        "PPO diretamente a partir de modelos/fnaf_bc.zip."
    )


if __name__ == "__main__":
    treinar_bc([
        "dados/gameplay_teste/dataset.json",
    ], epochs=200)

    print("\nPronto! Para usar o BC como ponto de partida do PPO:")
    print("  1. Deixe modelos/fnaf_bc.zip como o modelo mais recente da pasta modelos/")
    print("     (ou mova os outros .zip para um backup)")
    print("  2. Rode: python main.py treino")