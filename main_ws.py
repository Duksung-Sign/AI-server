import os
import numpy as np
import uvicorn
import joblib
from collections import deque
from typing import Dict, List

# FastAPI 관련
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Keras 모델 구조를 만들기 위한 라이브러리
import tensorflow as tf
from tensorflow.keras import layers as L, models as M

# === FastAPI 앱 ===
app = FastAPI(title="Keras Sign API (244 & 276 통합)",
              description="웹소켓으로 244 피처와 276 피처를 각각 다른 라우트에서 받아 예측합니다.",
              version="10.0.0")

# ================== 244 모델 설정 ==================
SEQ_LEN_244 = 30
NUM_FEATURES_244 = 244
NUM_CLASSES_244 = 14
CLASS_NAMES_244 = [
    "thx", "study", "okay", "me", "you", "arrive", "nicetomeet",
    "none", "love", "hate", "hello", "call", "goodnice", "sorry"
]
# 환경변수 이름을 분리합니다.
MODEL_PATH_244 = os.getenv("MODEL_PATH_244", "model/20251010_ver0001.h5")
model_244 = None

# ================== 276 모델 설정 ==================
SEQ_LEN_276 = 30
NUM_FEATURES_276 = 276
NUM_CLASSES_276 = 13
CLASS_NAMES_276 = [
    "thx", "study", "okay", "me", "you", "arrive", "nicetomeet",
    "none",  "hate", "hello", "call", "goodnice", "sorry"
]
# 환경변수 이름을 분리합니다.
MODEL_PATH_276 = os.getenv("MODEL_PATH_276", "model/20251030_276D_ver01_best_276d.h5")
SCALER_PATH_276 = os.getenv("SCALER_PATH_276", "scaler_276_1030.pkl")
model_276 = None
scaler_276 = None


# === 모델 로드 함수 (244) ===
def load_model_244():
    global model_244
    if not os.path.exists(MODEL_PATH_244):
        raise FileNotFoundError(f"MODEL_PATH_244 not found: {MODEL_PATH_244}")

    # --- 244 모델 구조 코드 ---
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

    inp = L.Input(shape=(SEQ_LEN_244, NUM_FEATURES_244))
    x = Inception1D(inp, filters=64)
    x = DilatedBlock(x, filters=64, dilation=2)
    x = DilatedBlock(x, filters=64, dilation=4)
    x = L.Bidirectional(L.LSTM(128, return_sequences=True))(x)
    x = L.Dropout(0.3)(x)
    x = L.Bidirectional(L.LSTM(64))(x)
    x = L.LayerNormalization()(x)
    x = L.Dense(96, activation='relu')(x)
    x = L.Dropout(0.5)(x)
    out = L.Dense(NUM_CLASSES_244, activation='softmax')(x)

    model_244 = M.Model(inp, out)
    # --- 모델 구조 정의 끝 ---

    model_244.load_weights(MODEL_PATH_244)

    # warm-up
    _ = model_244.predict(np.zeros((1, SEQ_LEN_244, NUM_FEATURES_244), dtype=np.float32), verbose=0)
    print(f"✅ [244] 모델 로딩 및 준비 완료: {MODEL_PATH_244}")


# === 모델 로드 함수 (276) ===
def load_model_276_and_scaler():
    global model_276, scaler_276

    # 1. 모델 로드
    if not os.path.exists(MODEL_PATH_276):
        raise FileNotFoundError(f"MODEL_PATH_276 not found: {MODEL_PATH_276}")
    try:
        model_276 = tf.keras.models.load_model(MODEL_PATH_276, compile=False)
        # Warm-up
        _ = model_276.predict(np.zeros((1, SEQ_LEN_276, NUM_FEATURES_276), dtype=np.float32), verbose=0)
        print(f"✅ [276] 모델 로드 및 워밍업 완료: {MODEL_PATH_276}")
    except Exception as e:
        print(f"❌ [276] 모델 로드 또는 워밍업 실패. 오류: {e}")
        raise e

    # 2. 스케일러 로드
    if not os.path.exists(SCALER_PATH_276):
        raise FileNotFoundError(f"SCALER_PATH_276 not found: {SCALER_PATH_276}")
    scaler_276 = joblib.load(SCALER_PATH_276)
    print(f"✅ [276] StandardScaler 로드 완료: {SCALER_PATH_276}")


# === 웹소켓 연결 관리 (공통 클래스) ===
class ConnectionManager:
    def __init__(self, seq_len: int):
        self.active_connections: Dict[WebSocket, deque] = {}
        self.seq_len = seq_len

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        # deque(maxlen=...)가 자동 슬라이딩 윈도우 역할을 합니다.
        self.active_connections[websocket] = deque(maxlen=self.seq_len)
        print(f"새 클라이언트 연결 (SEQ_LEN={self.seq_len}): {websocket.client}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            del self.active_connections[websocket]
            print(f"클라이언트 연결 종료: {websocket.client}")

    def add_frame(self, websocket: WebSocket, frame_data: list):
        if websocket in self.active_connections:
            # maxlen 덕분에 꽉 찼으면 자동으로 맨 왼쪽(가장 오래된) 데이터가 밀려납니다.
            self.active_connections[websocket].append(frame_data)

    def get_buffer(self, websocket: WebSocket) -> deque:
        return self.active_connections.get(websocket)


# === 매니저 인스턴스 분리 ===
manager_244 = ConnectionManager(seq_len=SEQ_LEN_244)
manager_276 = ConnectionManager(seq_len=SEQ_LEN_276)


# === FastAPI 라우트 ===
@app.on_event("startup")
def _startup():
    # 서버 시작 시 두 모델을 모두 로드합니다.
    load_model_244()
    load_model_276_and_scaler()


@app.get("/")
def root():
    return {"message": "✅ Keras Sign API (244 & 276) is running!",
            "model_244": MODEL_PATH_244,
            "model_276": MODEL_PATH_276,
            "scaler_276": SCALER_PATH_276,
            "endpoints": ["/ws/244", "/ws/276"]}


# === 엔드포인트 1: 244 피처 모델 ===
@app.websocket("/ws/244")
async def websocket_endpoint_244(websocket: WebSocket):
    await manager_244.connect(websocket)

    try:
        while True:
            data = await websocket.receive_json()
            frame = data.get('frame')

            if not isinstance(frame, list) or len(frame) != NUM_FEATURES_244:
                await websocket.send_json({"error": f"Invalid frame dimension. Expected {NUM_FEATURES_244}"})
                continue

            # 1. 덱(deque)에 프레임 추가 (maxlen에 의해 자동 슬라이딩)
            manager_244.add_frame(websocket, frame)
            buffer = manager_244.get_buffer(websocket)

            # 2. 버퍼가 30개 찰 때까지 기다림
            if len(buffer) == SEQ_LEN_244:
                sequence = np.array(list(buffer), dtype=np.float32)
                x = np.expand_dims(sequence, axis=0)

                # 3. [244 모델]로 예측
                probs = model_244.predict(x, verbose=0)[0]

                pred_idx = int(np.argmax(probs))
                predicted_label = CLASS_NAMES_244[pred_idx]
                probabilities = {cls: float(p) for cls, p in zip(CLASS_NAMES_244, probs.tolist())}

                await websocket.send_json({
                    "prediction": predicted_label,
                    "probabilities": probabilities
                })

                # 'deque(maxlen=...)'이 자동 슬라이딩을 하므로 수동 popleft()는 필요 없습니다.

    except WebSocketDisconnect:
        manager_244.disconnect(websocket)
    except Exception as e:
        print(f"[244] 오류 발생: {e}")
        manager_244.disconnect(websocket)


# === 엔드포인트 2: 276 피처 모델 ===
@app.websocket("/ws/276")
async def websocket_endpoint_276(websocket: WebSocket):
    await manager_276.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            frame = data.get("frame")

            if not isinstance(frame, list) or len(frame) != NUM_FEATURES_276:
                await websocket.send_json(
                    {"error": f"Invalid frame dimension. Expected {NUM_FEATURES_276}, got {len(frame)}"}
                )
                continue

            # 1. 덱(deque)에 프레임 추가 (maxlen에 의해 자동 슬라이딩)
            manager_276.add_frame(websocket, frame)
            buffer = manager_276.get_buffer(websocket)

            # 2. 버퍼가 30개 찰 때까지 기다림
            if len(buffer) == SEQ_LEN_276:
                # 3. (30, 276) 형태의 Raw 데이터 배열 생성
                sequence = np.array(list(buffer), dtype=np.float32)

                # 4. [핵심] 276용 스케일러 적용
                sequence_scaled = scaler_276.transform(sequence)

                # 5. 모델 입력 형태로 변경
                x = np.expand_dims(sequence_scaled, axis=0)

                # 6. [276 모델]로 예측
                probs = model_276.predict(x, verbose=0)[0]

                pred_idx = int(np.argmax(probs))
                predicted_label = CLASS_NAMES_276[pred_idx]
                probabilities = {cls: float(p)
                                 for cls, p in zip(CLASS_NAMES_276, probs.tolist())}

                await websocket.send_json({
                    "prediction": predicted_label,
                    "probabilities": probabilities
                })

                # 'deque(maxlen=...)'이 자동 슬라이딩을 하므로 수동 popleft()는 필요 없습니다.

    except WebSocketDisconnect:
        manager_276.disconnect(websocket)
    except Exception as e:
        print(f"⚠️ [276] 예측 중 오류 발생: {e}")
        manager_276.disconnect(websocket)


# === 로컬 실행 ===
if __name__ == "__main__":
    # 이 파일 이름을 'main_merged.py'로 저장했다면
    # uvicorn.run("main_merged:app", host="0.0.0.0", port=8000, reload=True)

    # 만약 'main_ws.py'로 저장했다면 아래 코드를 사용하세요.
    uvicorn.run("main_ws:app", host="0.0.0.0", port=8000, reload=True)