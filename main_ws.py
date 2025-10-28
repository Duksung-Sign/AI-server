import os
import numpy as np
import uvicorn
from collections import deque
from typing import Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import tensorflow as tf
# from tensorflow.keras import layers as L, models as M  # <-- 더 이상 필요 없음
import joblib

# === FastAPI 앱 설정 ===
app = FastAPI(
    title="Sign Language LSTM API (276D, WebSocket)",
    description="Unity에서 276차원 원본 프레임을 받아 LSTM 모델로 예측합니다.",
    version="5.0.1"  # 버전 업데이트
)

# === 모델 및 데이터 설정 ===
SEQ_LEN = 30
NUM_FEATURES = 276
NUM_CLASSES = 13
MODEL_PATH = os.getenv("MODEL_PATH", "model/20251025_ver05_best_lstm_276.h5")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler_276.pkl")

CLASS_NAMES = [
    "thx", "study", "okay", "me", "you", "arrive", "ntmu",
    "none", "dislike", "hello", "call", "like", "sorry"
]

model = None
scaler = None


# === 모델 로드 (수정된 부분) ===
def load_model():
    """
    .h5 파일에서 모델 구조와 가중치를 한 번에 로드합니다.
    """
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")

    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    # 기존의 Inception1D, DilatedBlock 등 복잡한 구조 정의를 모두 삭제하고
    # 로컬 테스트 코드와 동일하게 load_model을 사용합니다.
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        print(f"❌ 모델 로드 중 심각한 오류 발생: {e}")
        print("모델 .h5 파일이 구조를 포함하여 저장되었는지 확인하세요.")
        raise e
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    # Warm-up (모델이 로드되었는지 확인)
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
    load_model()  # 수정된 함수 호출
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
                # 1. Deque를 리스트로 변환 후 NumPy 배열 생성
                sequence = np.array(list(buffer), dtype=np.float32)

                # ▼▼▼▼▼ 디버그 코드 (매 예측 시 최신 프레임 출력) ▼▼▼▼▼
                raw_from_unity = sequence[-1]  # Unity가 보낸 최신 프레임
                print("\n" + "=" * 30)
                print(">>> [SERVER: RAW 276] (스케일러 적용 전) <<<")
                print(raw_from_unity.tolist())
                print("=" * 30 + "\n")
                # ▲▲▲▲▲ 디버그 코드 ▲▲▲▲▲


                # 2. 스케일러 적용 (중요!)
                # (SEQ_LEN, NUM_FEATURES) -> (SEQ_LEN * NUM_FEATURES,) 1D 배열로 변환 불필요
                # .transform()은 2D 배열을 기대함
                sequence_scaled = scaler.transform(sequence)  # (30, 276) 형태 그대로 변환

                # 3. 모델 입력 형태로 변경
                # (30, 276) -> (1, 30, 276)
                x = np.expand_dims(sequence_scaled, axis=0)

                # 4. 예측
                probs = model.predict(x, verbose=0)[0]

                pred_idx = int(np.argmax(probs))
                predicted_label = CLASS_NAMES[pred_idx]
                probabilities = {cls: float(p)
                                 for cls, p in zip(CLASS_NAMES, probs.tolist())}

                await websocket.send_json({
                    "prediction": predicted_label,
                    "probabilities": probabilities
                })

                # 슬라이딩 윈도우: 가장 오래된 프레임 제거
                buffer.popleft()

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"⚠️ 예측 중 오류 발생: {e}")
        manager.disconnect(websocket)


# === 로컬 실행 ===
if __name__ == "__main__":
    # main_ws_276.py 파일이므로 "main_ws_276:app"이 맞습니다.
    uvicorn.run("main_ws_276:app", host="0.0.0.0", port=8000, reload=True)