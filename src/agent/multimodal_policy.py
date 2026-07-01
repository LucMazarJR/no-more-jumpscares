import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class MultimodalExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=256)

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample_img = torch.zeros(1, 1, 84, 84)
            cnn_output_dim = self.cnn(sample_img).shape[1]

        self.fc_estados = nn.Sequential(
            # Deriva do espaço (hoje 12: Decisão 7 +noite, +tempo_sem_camera) — nunca hardcodar:
            # o shape errado aqui só estoura em runtime, bem depois da construção.
            nn.Linear(observation_space["estados"].shape[0], 32),
            nn.ReLU(),
        )

        self.fc_combined = nn.Sequential(
            nn.Linear(cnn_output_dim + 32, 256),
            nn.ReLU(),
        )

    def forward(self, observations):
        # NÃO dividir por 255 aqui: o SB3 (extract_features → preprocess_obs)
        # já normaliza espaços de imagem uint8 para [0, 1] antes de chamar o
        # extractor. Dividir de novo deixava os pixels em [0, 0.004] — a CNN
        # recebia entrada quase nula e o agente ficava cego para a imagem.
        imagem = observations["imagem"].float()
        estados = observations["estados"]

        if imagem.dim() == 3:
            imagem = imagem.unsqueeze(0)

        # Aceita channels-last (N, 84, 84, 1) — usado pelo behavioral cloning —
        # e channels-first (N, 1, 84, 84) — formato do VecTransposeImage do SB3.
        if imagem.shape[-1] == 1:
            imagem = imagem.permute(0, 3, 1, 2)

        features_img = self.cnn(imagem)
        features_estados = self.fc_estados(estados)

        combined = torch.cat([features_img, features_estados], dim=1)
        return self.fc_combined(combined)
