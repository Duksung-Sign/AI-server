# main.py
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import os
import numpy as np
import tensorflow as tf

# === FastAPI 앱 ===
app = FastAPI(title="Keras LSTM Sign API (244D)",
              description="입력 시퀀스(30x244)를 받아 LSTM(.h5) 모델로 예측합니다.",
              version="1.0.0")

# ================== 고정 상수 (244D 스키마) ==================
SEQ_LEN = 30
FEATURE_DIM = 244

# === 클래스 순서 (실시간 코드와 동일) ===
CLASS_NAMES = [
    "thx",         # 0
    "study",       # 1
    "okay",        # 2
    "me",          # 3
    "you",         # 4
    "arrive",      # 5
    "nicetomeet",  # 6
    "none",        # 7
    "love",        # 8
    "hate",        # 9
    "hello",       # 10
    "call",        # 11
    "goodnice",    # 12
    "sorry"        # 13
]

# === 환경변수로 경로/키 오버라이드 가능 ===
MODEL_PATH = os.getenv("MODEL_PATH", "model/20250902_ver01.h5")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "secret")  # /reload 보호용(선택)

# === 모델 로드 ===
model = None
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    # warm-up (옵션)
    _ = model.predict(np.zeros((1, SEQ_LEN, FEATURE_DIM), dtype=np.float32), verbose=0)

def ensure_model():
    if model is None:
        load_model()

# === 입력 데이터 스키마 ===
class InputData(BaseModel):
    sequence: list[list[float]]  # (T, 244). 보통 T=30

# === 유틸: 길이 보정(선택) ===
def temporal_interpolate(seq_np: np.ndarray, target_len: int = SEQ_LEN) -> np.ndarray:
    """(T, D)->(target_len, D) 선형보간. T가 30이 아닐 때만 사용."""
    T, D = seq_np.shape
    if T == target_len:
        return seq_np
    x_old = np.linspace(0.0, 1.0, T)
    x_new = np.linspace(0.0, 1.0, target_len)
    out = np.empty((target_len, D), dtype=np.float32)
    for d in range(D):
        out[:, d] = np.interp(x_new, x_old, seq_np[:, d])
    return out

# === 라우트 ===
@app.on_event("startup")
def _startup():
    load_model()

@app.get("/")
def root():
    return {"message": "✅ Keras LSTM Sign API is running!", "model_path": MODEL_PATH}

@app.post("/test")
async def test(data: dict):
    return {"received": data, "message": "Test successful!"}

@app.get("/health")
def health():
    try:
        ensure_model()
        return {"ok": True, "classes": len(CLASS_NAMES)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/reload")
def reload_model(x_api_key: str = Header(None)):
    if x_api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")
    load_model()
    return {"reloaded": True, "model_path": MODEL_PATH}

@app.post("/predict",
          summary="수어 예측 (244D, LSTM .h5)",
          description="T프레임(일반적으로 30), 244차원 시퀀스를 입력받아 확률과 예측 라벨을 반환합니다.")
def predict(data: InputData):
    try:
        # 입력 → np
        arr = np.asarray(data.sequence, dtype=np.float32)  # (T, 244)
        if arr.ndim != 2 or arr.shape[1] != FEATURE_DIM:
            return {"error": f"❌ 입력 차원 오류: (*, {FEATURE_DIM})이어야 합니다. 현재: {arr.shape}"}

        # 길이 보정(선택): T!=30인 경우 선형보간으로 30으로 맞춤
        if arr.shape[0] != SEQ_LEN:
            arr = temporal_interpolate(arr, target_len=SEQ_LEN)

        # 배치 차원 추가
        x = np.expand_dims(arr, axis=0)  # (1, 30, 244)

        ensure_model()
        probs = model.predict(x, verbose=0)[0]  # (num_classes,)
        probs = probs.astype(np.float32)

        pred_idx = int(np.argmax(probs))
        predicted_label = CLASS_NAMES[pred_idx]
        probabilities = {cls: float(p) for cls, p in zip(CLASS_NAMES, probs.tolist())}

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