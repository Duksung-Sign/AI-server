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


class TransformerCTC(nn.Module):
    def __init__(self, input_dim=244, model_dim=256, num_classes=13, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()

        print(f"🎯 Transformer CTC 모델 초기화")
        print(f"📊 입력 차원: {input_dim}")
        print(f"🏷️ 클래스 수: {num_classes}")
        print(f"🧠 모델 차원: {model_dim}")
        print(f"👁️ 헤드 수: {num_heads}")
        print(f"🏗️ 레이어 수: {num_layers}")

        # 244차원 → model_dim 변환
        self.input_fc = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            activation='gelu'  # 더 나은 활성화 함수
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 프레임별 클래스 예측 (CTC용)
        self.classifier = nn.Linear(model_dim, num_classes + 1)  # +1은 CTC blank label

        # 드롭아웃 추가
        self.dropout = nn.Dropout(dropout)

        # 파라미터 수 계산
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"📈 총 파라미터: {total_params:,}")
        print(f"🎓 학습 가능 파라미터: {trainable_params:,}")

    def forward(self, x):
        """
        Args:
            x: (B, T, 244) - 배치, 시퀀스, 244차원 특징

        Returns:
            logits: (B, T, num_classes+1) - CTC용 출력
        """
        # 입력 차원 확인
        assert x.shape[-1] == 244, f"입력 차원이 244가 아님: {x.shape[-1]}"

        # 244차원 → model_dim 변환
        x = self.input_fc(x)  # (B, T, model_dim)
        x = self.dropout(x)  # 드롭아웃

        # Positional Encoding 추가
        x = self.pos_encoder(x)  # (B, T, model_dim)

        # Transformer Encoder
        x = self.transformer_encoder(x)  # (B, T, model_dim)

        # 분류기
        logits = self.classifier(x)  # (B, T, num_classes+1)

        return logits

    def get_model_info(self):
        """모델 정보 반환"""
        return {
            'input_dim': 244,
            'num_classes': 13,
            'model_dim': self.input_fc.out_features,
            'num_heads': self.transformer_encoder.layers[0].self_attn.num_heads,
            'num_layers': len(self.transformer_encoder.layers),
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


# 모델 테스트 함수
def test_model():
    """모델 작동 테스트"""
    print("\n🧪 모델 테스트")
    print("=" * 30)

    # 모델 생성
    model = TransformerCTC(
        input_dim=244,
        model_dim=256,
        num_classes=13,
        num_heads=8,
        num_layers=6,
        dropout=0.1
    )

    # 더미 데이터 생성 (batch_size=4, seq_len=30, features=244)
    dummy_input = torch.randn(4, 30, 244)

    print(f"\n📥 입력 형태: {dummy_input.shape}")

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"📤 출력 형태: {output.shape}")
    print(f"✅ 예상 형태: (4, 30, 14) - 13클래스 + 1blank")

    if output.shape == (4, 30, 14):
        print("🎉 모델 테스트 성공!")
    else:
        print("❌ 모델 출력 형태 오류")

    return model


if __name__ == "__main__":
    model = test_model()
    print(f"\n📊 모델 정보:")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")