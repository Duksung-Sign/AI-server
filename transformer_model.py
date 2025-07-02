# transformer_model.py

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SignTransformer(nn.Module):
    def __init__(self, input_dim=232, model_dim=256, num_classes=3, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()
        # 입력 차원 확장 (232 -> 256)
        self.input_fc = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)

        # Transformer 인코더 레이어
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 분류기
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_classes)
        )

    def forward(self, x):
        # x: (B, T, 174) 형태로 이미 들어옴
        x = self.input_fc(x)  # (B, T, model_dim)
        x = self.pos_encoder(x)  # positional encoding 추가
        x = self.transformer_encoder(x)  # transformer 인코더 통과
        x = x.mean(dim=1)  # temporal 평균 pooling
        return self.classifier(x)  # 최종 분류
