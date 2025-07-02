from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import torch
from transformer_model import SignTransformer

# === FastAPI 앱 생성 ===
app = FastAPI()

# === 하이퍼파라미터 및 클래스 정의 ===
SEQ_LEN = 30
INPUT_DIM = 236
CLASS_NAMES = ['thx','call','none']  # ← 모델 학습 시 사용한 클래스 순서와 동일하게

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 모델 및 정규화 값 로드 ===
MODEL_PATH = "model/best_transformer_236.pt"
MEAN_PATH = "model/mean.npy"
STD_PATH = "model/std.npy"

model = SignTransformer(input_dim=INPUT_DIM, num_classes=len(CLASS_NAMES)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

mean = np.load(MEAN_PATH)
std = np.load(STD_PATH)
std[std == 0] = 1e-6  # 0으로 나누는 오류 방지

# === 입력 데이터 스키마 정의 ===
class InputData(BaseModel):
    sequence: list[list[float]]  # (30, 236) float 배열

# === 루트 라우트 ===
@app.get("/")
async def root():
    return {"message": "Transformer-based Sign Language API is running!"}

# === 테스트용 라우트 ===
@app.post("/test")
async def test(data: dict):
    return {"received": data, "message": "Test successful!"}

# === 예측 라우트 ===
@app.post("/predict",
          summary="수어 예측",
          description="30프레임, 236차원 시퀀스를 입력받아 수어 클래스 예측 결과를 반환합니다.")
async def predict(data: InputData):
    try:
        input_array = np.array(data.sequence, dtype=np.float32)

        if input_array.shape != (SEQ_LEN, INPUT_DIM):
            return {"error": f"Input shape must be ({SEQ_LEN}, {INPUT_DIM}), but got {input_array.shape}"}

        # 정규화
        input_array = (input_array - mean) / std

        # 모델 입력 형태로 변환
        input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # 예측
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        predicted_label = CLASS_NAMES[pred_idx]
        probabilities = dict(zip(CLASS_NAMES, probs.tolist()))

        return {
            "prediction": predicted_label,
            "probabilities": probabilities
        }

    except Exception as e:
        return {"error": str(e)}

# === 로컬 테스트 실행용 ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
