from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import torch
from transformer_ctc_model import TransformerCTC  # ✅ CTC 모델로 변경

# === FastAPI 앱 생성 ===
app = FastAPI()

# === 하이퍼파라미터 및 클래스 정의 ===
SEQ_LEN = 30
INPUT_DIM = 244

CLASS_NAMES = [
    "thx",         # 0
    "study",       # 1
    "okay",        # 2
    "me",          # 3
    "you",         # 4
    "arrive",      # 5
    "nicetomeet",  # 6
    "none",        # 7
    "hate",        # 8
    "hello",       # 9
    "call",        # 10
    "goodnice",    # 11
    "sorry"        # 12
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 모델 경로 및 로드 ===
MODEL_PATH = "model/250904_best_ctc_transformer_244.pt"

model = TransformerCTC(input_dim=INPUT_DIM, num_classes=len(CLASS_NAMES), num_layers=6).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# === 입력 데이터 스키마 정의 ===
class InputData(BaseModel):
    sequence: list[list[float]]  # (30, 244) 형태

# === 루트 라우트 ===
@app.get("/")
async def root():
    return {"message": "✅ 244D CTC Transformer Sign Language API is running!"}

# === 테스트 라우트 ===
@app.post("/test")
async def test(data: dict):
    return {"received": data, "message": "Test successful!"}

# === 예측 라우트 ===
@app.post("/predict",
          summary="수어 예측 (244차원, CTC 모델)",
          description="30프레임, 244차원 시퀀스를 입력받아 수어 클래스 예측 결과를 반환합니다.")
async def predict(data: InputData):
    try:
        input_array = np.array(data.sequence, dtype=np.float32)

        if input_array.shape != (SEQ_LEN, INPUT_DIM):
            return {"error": f"❌ 입력 차원 오류: ({SEQ_LEN}, {INPUT_DIM}) 이어야 합니다. 현재: {input_array.shape}"}

        # === 모델 입력 형태로 변환
        input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, 30, 244)

        # === 예측
        with torch.no_grad():
            output = model(input_tensor)  # (1, 30, num_classes + 1)
            probs = torch.softmax(output, dim=2)[0].cpu().numpy()  # (30, 14)

        # === CTC 평균 확률 기반 예측 (blank 제외)
        avg_probs = probs.mean(axis=0)  # (14,)
        pred_idx = int(np.argmax(avg_probs[:len(CLASS_NAMES)]))  # blank 제외
        predicted_label = CLASS_NAMES[pred_idx]
        probabilities = dict(zip(CLASS_NAMES, avg_probs[:len(CLASS_NAMES)].tolist()))

        return {
            "prediction": predicted_label,
            "probabilities": probabilities
        }

    except Exception as e:
        return {"error": str(e)}

# === 로컬 실행 ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
