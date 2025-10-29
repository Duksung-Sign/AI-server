import os
import numpy as np
import uvicorn
from collections import deque
from typing import Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import tensorflow as tf
import joblib

# === FastAPI 앱 설정 ===
app = FastAPI(
    title="Sign Language LSTM API (276D, WebSocket)",
    description="Unity에서 276차원 원본 프레임을 받아 LSTM 모델로 예측합니다.",
    version="6.0.0"  # 버전 업데이트
)

# === 모델 및 데이터 설정 ===
SEQ_LEN = 30
NUM_FEATURES = 276
NUM_CLASSES = 13  # ★ 1. 학습 스크립트와 클래스 개수 일치 (13개)

# ★ 2. 학습 스크립트가 생성할 파일명과 일치시켜야 합니다.
# 예: "model/20251030_276D_ver01_best_276d.h5" (학습 후 실제 파일명으로 변경)
MODEL_PATH = os.getenv("MODEL_PATH", "model/20251030_276D_ver01_best_276d.h5")

# ★ 학습 스크립트가 생성할 "scaler_276.pkl"
SCALER_PATH = os.getenv("SCALER_PATH", "scaler_276_1030.pkl")

# ★ 3. [가장 중요] 학습 스크립트의 'classes' 리스트와 순서/내용이 100% 일치해야 함
CLASS_NAMES = [
    '감사합니다_flipped', '공부하다_flipped', '괜찮다_flipped', '나_flipped', '당신_flipped', '도착하다_flipped',
    '반갑습니다_flipped', '비수어_flipped', '싫다_flipped', '안녕하세요_flipped', '연락해주세요_flipped',
    '좋다멋지다_flipped', '죄송합니다_flipped'
]

model = None
scaler = None


# === 모델 로드 ===
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")

    try:
        # compile=False가 .h5 로딩 시 더 빠르고 안정적입니다.
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        print(f"❌ 모델 로드 중 심각한 오류 발생: {e}")
        raise e

    # Warm-up
    try:
        _ = model.predict(np.zeros((1, SEQ_LEN, NUM_FEATURES),
                                   dtype=np.float32), verbose=0)
        print(f"✅ LSTM(276D) 모델 로드 및 워밍업 완료: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ 모델 워밍업 실패. 입력 shape({SEQ_LEN}, {NUM_FEATURES})가 모델과 맞는지 확인하세요. 오류: {e}")
        raise e


# === WebSocket 연결 관리 ===
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[WebSocket, deque] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        # deque(maxlen=SEQ_LEN)가 자동 슬라이딩 윈도우 역할을 합니다.
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

            # 1. 최신 프레임을 덱(deque)에 추가 (오래된 것은 자동 삭제됨)
            manager.add_frame(websocket, frame)
            buffer = manager.get_buffer(websocket)

            # 2. 버퍼가 30개 찰 때까지 기다림
            if len(buffer) == SEQ_LEN:
                # 3. (30, 276) 형태의 Raw 데이터 배열 생성
                sequence = np.array(list(buffer), dtype=np.float32)

                # 4. [핵심] 스케일러 적용
                # (30, 276) -> (30, 276)
                sequence_scaled = scaler.transform(sequence)

                # 5. 모델 입력 형태로 변경
                # (30, 276) -> (1, 30, 276)
                x = np.expand_dims(sequence_scaled, axis=0)

                # 6. 예측
                probs = model.predict(x, verbose=0)[0]

                pred_idx = int(np.argmax(probs))
                predicted_label = CLASS_NAMES[pred_idx]
                probabilities = {cls: float(p)
                                 for cls, p in zip(CLASS_NAMES, probs.tolist())}

                await websocket.send_json({
                    "prediction": predicted_label,
                    "probabilities": probabilities
                })

                # ★ 4. [수정] 'popleft()' 제거 ★
                # popleft()를 하면 30프레임마다 1번 예측하는 '텀블링 윈도우'가 됩니다.
                # popleft()를 제거해야 매 프레임 예측하는 '슬라이딩 윈도우'가 됩니다.
                # buffer.popleft() # <-- 이 줄 삭제

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"⚠️ 예측 중 오류 발생: {e}")
        manager.disconnect(websocket)


# === 로컬 실행 ===
if __name__ == "__main__":
    # 이 파일 이름을 'main_server_276.py' 등으로 저장했다면
    # uvicorn.run("main_server_276:app", ...)

    # 지금 파일명이 main_ws_276.py 라면
    uvicorn.run("main_ws:app", host="0.0.0.0", port=8000, reload=True)