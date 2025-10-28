import os
import numpy as np
import uvicorn
from collections import deque
from typing import Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import tensorflow as tf
from tensorflow.keras import layers as L, models as M
import joblib  # ✅ 추가

# === FastAPI 앱 설정 ===
app = FastAPI(
    title="Sign Language LSTM API (276D, WebSocket)",
    description="Unity에서 276차원 원본 프레임을 받아 LSTM 모델로 예측합니다.",
    version="5.0.0"
)

# === 모델 및 데이터 설정 ===
SEQ_LEN = 30
NUM_FEATURES = 276
NUM_CLASSES = 13
MODEL_PATH = os.getenv("MODEL_PATH", "model/20251025_ver05_best_lstm_276.h5")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler_276.pkl")  # ✅ 추가

CLASS_NAMES = [
    "thx", "study", "okay", "me", "you", "arrive", "ntmu",
    "none", "dislike", "hello", "call", "like", "sorry"
]

model = None
scaler = None  # ✅ 추가


# === 모델 로드 ===
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")

    def Inception1D(x, filters=64):
        b3 = L.Conv1D(filters, 3, padding='same', activation='relu')(x)
        b5 = L.Conv1D(filters, 5, padding='same', activation='relu')(x)
        b7 = L.Conv1D(filters, 7, padding='same', activation='relu')(x)
        z = L.Concatenate()([b3, b5, b7])
        z = L.LayerNormalization()(z)
        z = L.Dropout(0.15)(z)
        z = L.Conv1D(filters, 1, padding='same', activation='relu')(z)

        se = L.GlobalAveragePooling1D()(z)
        se = L.Dense(filters // 2, activation='relu')(se)
        se = L.Dense(filters, activation='sigmoid')(se)
        se = L.Multiply()([z, L.Reshape((1, filters))(se)])

        skip = L.Conv1D(filters, 1, padding='same')(x)
        out = L.Add()([se, skip])
        return out

    def DilatedBlock(x, filters=64, dilation=2):
        z = L.Conv1D(filters, 3, padding='same',
                     dilation_rate=dilation, activation='relu')(x)
        z = L.LayerNormalization()(z)
        z = L.Dropout(0.15)(z)
        if x.shape[-1] != filters:
            x = L.Conv1D(filters, 1, padding='same')(x)
        return L.Add()([x, z])

    inp = L.Input(shape=(SEQ_LEN, NUM_FEATURES))
    x = Inception1D(inp, filters=64)
    x = DilatedBlock(x, filters=64, dilation=2)
    x = DilatedBlock(x, filters=64, dilation=4)
    x = L.Bidirectional(L.LSTM(128, return_sequences=True))(x)
    x = L.Dropout(0.3)(x)
    x = L.Bidirectional(L.LSTM(64))(x)
    x = L.LayerNormalization()(x)
    x = L.Dense(96, activation='relu')(x)
    x = L.Dropout(0.5)(x)
    out = L.Dense(NUM_CLASSES, activation='softmax')(x)

    model = M.Model(inp, out)
    model.load_weights(MODEL_PATH)

    # Warm-up
    _ = model.predict(np.zeros((1, SEQ_LEN, NUM_FEATURES),
                     dtype=np.float32), verbose=0)
    print(f"✅ LSTM(276D) 모델 로드 완료: {MODEL_PATH}")


# === WebSocket 연결 관리 ===
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[WebSocket, deque] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[websocket] = deque(maxlen=SEQ_LEN)
        print(f"새 클라이언트 연결: {websocket.client}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            del self.active_connections[websocket]
            print(f"클라이언트 연결 종료: {websocket.client}")

    def add_frame(self, websocket: WebSocket, frame_data: list):
        if websocket in self.active_connections:
            self.active_connections[websocket].append(frame_data)

    def get_buffer(self, websocket: WebSocket) -> deque:
        return self.active_connections.get(websocket)


manager = ConnectionManager()


# === 서버 시작 시 모델 & 스케일러 로드 ===
@app.on_event("startup")
def _startup():
    global scaler
    load_model()
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"SCALER_PATH not found: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    print(f"✅ StandardScaler 로드 완료: {SCALER_PATH}")


@app.get("/")
def root():
    return {"message": "✅ 276D LSTM Sign API running",
            "model_path": MODEL_PATH,
            "scaler_path": SCALER_PATH,
            "expected_input_dim": NUM_FEATURES,
            "seq_len": SEQ_LEN}


# === WebSocket 엔드포인트 ===
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            frame = data.get("frame")

            if not isinstance(frame, list) or len(frame) != NUM_FEATURES:
                await websocket.send_json(
                    {"error": f"Invalid frame dimension. Expected {NUM_FEATURES}, got {len(frame)}"}
                )
                continue

            manager.add_frame(websocket, frame)
            buffer = manager.get_buffer(websocket)

            if len(buffer) == SEQ_LEN:
                sequence = np.array(list(buffer), dtype=np.float32)

                # ✅ 표준화 적용 (훈련 때 사용한 scaler 그대로)
                sequence = scaler.transform(sequence.reshape(-1, NUM_FEATURES)) \
                                 .reshape(SEQ_LEN, NUM_FEATURES)

                # 🔹 (선택) 분포 확인용 샘플 저장 — 필요 시만 켜기
                # np.save("unity_input_sample.npy", sequence)

                x = np.expand_dims(sequence, axis=0)
                probs = model.predict(x, verbose=0)[0]

                pred_idx = int(np.argmax(probs))
                predicted_label = CLASS_NAMES[pred_idx]
                probabilities = {cls: float(p)
                                 for cls, p in zip(CLASS_NAMES, probs.tolist())}

                await websocket.send_json({
                    "prediction": predicted_label,
                    "probabilities": probabilities
                })

                # 슬라이딩 윈도우: 한 프레임 제거
                buffer.popleft()

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"⚠️ 예측 중 오류 발생: {e}")
        manager.disconnect(websocket)


# === 로컬 실행 ===
if __name__ == "__main__":
    uvicorn.run("main_ws_276:app", host="0.0.0.0", port=8000, reload=True)
