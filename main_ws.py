# main_ws.py (최종본)

import os
import numpy as np
import uvicorn
from collections import deque
from typing import Dict, List

# FastAPI 관련
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Keras 모델 구조를 만들기 위한 라이브러리
import tensorflow as tf
from tensorflow.keras import layers as L, models as M

# === FastAPI 앱 ===
app = FastAPI(title="Keras LSTM Sign API (WebSocket, 최종)",
              description="웹소켓으로 실시간 시퀀스를 받아 Keras 모델로 예측합니다.",
              version="3.0.0")

# ================== 모델 설정 (학습 코드와 동일하게 맞춤) ==================
SEQ_LEN = 30
NUM_FEATURES = 244
NUM_CLASSES = 14

# 모델이 학습한 클래스 순서 (14개)
CLASS_NAMES = [
    "thx", "study", "okay", "me", "you", "arrive", "nicetomeet",
    "none", "love", "hate", "hello", "call", "goodnice", "sorry"
]

# === 환경변수 및 모델 경로 ===
MODEL_PATH = os.getenv("MODEL_PATH", "model/20250902_ver01.h5")

# === 모델 로드 함수 (최종 수정본) ===
model = None


def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")

    # --- 친구의 모델 구조 코드 ---
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
        z = L.Conv1D(filters, 3, padding='same', dilation_rate=dilation, activation='relu')(x)
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
    # --- 모델 구조 정의 끝 ---

    # 구조가 아닌 가중치(weights)만 불러옵니다.
    model.load_weights(MODEL_PATH)

    # warm-up
    _ = model.predict(np.zeros((1, SEQ_LEN, NUM_FEATURES), dtype=np.float32), verbose=0)
    print("✅ 모델 로딩 및 준비 완료!")


# === 웹소켓 연결 관리 ===
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[WebSocket, deque] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[websocket] = deque(maxlen=SEQ_LEN)
        print(f"새 클라이언트 연결: {websocket.client}")

    def disconnect(self, websocket: WebSocket):
        del self.active_connections[websocket]
        print(f"클라이언트 연결 종료: {websocket.client}")

    def add_frame(self, websocket: WebSocket, frame_data: list):
        if websocket in self.active_connections:
            self.active_connections[websocket].append(frame_data)

    def get_buffer(self, websocket: WebSocket) -> deque:
        return self.active_connections.get(websocket)


manager = ConnectionManager()


# === FastAPI 라우트 ===
@app.on_event("startup")
def _startup():
    load_model()


@app.get("/")
def root():
    return {"message": "✅ Keras LSTM Sign API (WebSocket) is running!", "model_path": MODEL_PATH}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_json()
            frame = data.get('frame')

            if not isinstance(frame, list) or len(frame) != NUM_FEATURES:
                await websocket.send_json({"error": f"Invalid frame dimension. Expected {NUM_FEATURES}"})
                continue

            manager.add_frame(websocket, frame)
            buffer = manager.get_buffer(websocket)

            if len(buffer) == SEQ_LEN:
                sequence = np.array(list(buffer), dtype=np.float32)
                x = np.expand_dims(sequence, axis=0)

                probs = model.predict(x, verbose=0)[0]

                pred_idx = int(np.argmax(probs))
                predicted_label = CLASS_NAMES[pred_idx]
                probabilities = {cls: float(p) for cls, p in zip(CLASS_NAMES, probs.tolist())}

                await websocket.send_json({
                    "prediction": predicted_label,
                    "probabilities": probabilities
                })

                # --- ❗️ 슬라이딩 윈도우 적용 부분 ❗️ ---
                # 가장 오래된 프레임 하나를 버려서 버퍼를 한 칸씩 이동시킴
                buffer.popleft()

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"오류 발생: {e}")
        manager.disconnect(websocket)


# === 로컬 실행 ===
if __name__ == "__main__":
    uvicorn.run("main_ws:app", host="0.0.0.0", port=8000, reload=True)