import torch

MODEL_PATH = "model/250904_best_ctc_transformer_244.pt"  # 경로를 필요에 따라 수정하세요


def check_model_layers(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')

    print("📂 체크포인트 로딩 완료!")

    if "model_state_dict" not in checkpoint:
        print("❌ model_state_dict가 체크포인트에 없습니다.")
        return

    state_dict = checkpoint["model_state_dict"]
    print(f"🔍 저장된 파라미터 수: {len(state_dict)}")

    transformer_layers = [
        k for k in state_dict.keys() if k.startswith("transformer_encoder.layers")
    ]

    # 레이어 인덱스 추출
    layer_indices = sorted(set(
        int(k.split('.')[2]) for k in transformer_layers
    ))

    print(f"🧱 Transformer 레이어 개수: {len(layer_indices)}")
    print("📑 레이어 인덱스 목록:", layer_indices)

    # 상세 출력
    print("\n📋 주요 키 (일부만 출력):")
    for k in list(state_dict.keys())[:10]:
        print("  ", k)
    if len(state_dict) > 10:
        print("  ...")


if __name__ == "__main__":
    check_model_layers(MODEL_PATH)
