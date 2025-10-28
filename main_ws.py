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
    version="5.1.0 (Server-Side Smoothing)"  # 버전 업데이트
)

# === 모델 및 데이터 설정 ===
SEQ_LEN = 30
NUM_FEATURES = 276
NUM_CLASSES = 13
MODEL_PATH = os.getenv("MODEL_PATH", "model/20251029_ver07_best_lstm_276.h5")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler_276_1029.pkl")

# ★★★★★ 수정된 부분 1: 스무딩 윈도우 크기 추가 ★★★★★
SMOOTHING_WINDOW = 5  # 로컬 테스트 코드(realtime_predict_lstm_276.py)와 동일하게 설정

CLASS_NAMES = [
    "thx", "study", "okay", "me", "you", "arrive", "ntmu",
    "none", "dislike", "hello", "call", "like", "sorry"
]

model = None
scaler = None


# === 모델 로드 (수정 없음) ===
def load_model():
    """
    .h5 파일에서 모델 구조와 가중치를 한 번에 로드합니다.
    """
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")

    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        print(f"❌ 모델 로드 중 심각한 오류 발생: {e}")
        print("모델 .h5 파일이 구조를 포함하여 저장되었는지 확인하세요.")
        raise e

    # Warm-up (모델이 로드되었는지 확인)
    try:
        _ = model.predict(np.zeros((1, SEQ_LEN, NUM_FEATURES),
                                   dtype=np.float32), verbose=0)
        print(f"✅ LSTM(276D) 모델 로드 및 워밍업 완료: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ 모델 워밍업 실패. 입력 shape({SEQ_LEN}, {NUM_FEATURES})가 모델과 맞는지 확인하세요. 오류: {e}")
        raise e


# === WebSocket 연결 관리 (수정됨) ===
class ConnectionManager:
    def __init__(self):
        # ★★★★★ 수정된 부분 2: 클라이언트별로 "프레임 큐"와 "예측 큐"를 딕셔너리로 저장 ★★★★★
        self.active_connections: Dict[WebSocket, Dict[str, deque]] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        # ★★★★★ 수정된 부분 3: 두 개의 큐를 함께 생성 ★★★★★
        self.active_connections[websocket] = {
            "frames": deque(maxlen=SEQ_LEN),  # 30프레임짜리 특징(feature) 큐
            "preds": deque(maxlen=SMOOTHING_WINDOW)  # 5프레임짜리 예측(prediction) 큐
        }
        print(f"새 클라이언트 연결: {websocket.client}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            del self.active_connections[websocket]
            print(f"클라이언트 연결 종료: {websocket.client}")

    # ★★★★★ 수정된 부분 4: 클라이언트의 버퍼 딕셔너리를 통째로 가져오는 함수 ★★★★★
    def get_buffers(self, websocket: WebSocket) -> Dict[str, deque] | None:
        return self.active_connections.get(websocket)


manager = ConnectionManager()


# === 서버 시작 시 모델 & 스케일러 로드 (수정 없음) ===
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
    return {"message": "✅ 276D LSTM Sign API running (Server Smoothing)",
            "model_path": MODEL_PATH,
            "scaler_path": SCALER_PATH,
            "expected_input_dim": NUM_FEATURES,
            "seq_len": SEQ_LEN}


# === WebSocket 엔드포인트 (수정됨) ===
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

            # ★★★★★ 수정된 부분 5: 해당 클라이언트의 버퍼 딕셔너리를 가져옴 ★★★★★
            buffers = manager.get_buffers(websocket)
            if not buffers:
                # (혹시 모를 예외처리: 연결이 끊겼으면 루프 종료)
                break

            # 두 개의 큐를 각각 가져옴
            frame_buffer = buffers["frames"]
            pred_buffer = buffers["preds"]

            # 1. 프레임 큐에 현재 프레임 추가
            frame_buffer.append(frame)

            if len(frame_buffer) == SEQ_LEN:
                # 2. Deque를 리스트로 변환 후 NumPy 배열 생성
                sequence = np.array(list(frame_buffer), dtype=np.float32)

                # ▼▼▼▼▼ 디버그 코드 (수정 없음) ▼▼▼▼▼
                raw_from_unity = sequence[-1]
                print("\n" + "=" * 30)
                print(">>> [SERVER: RAW 276] (스케일러 적용 전) <<<")
                print(raw_from_unity.tolist())
                print("=" * 30 + "\n")
                # ▲▲▲▲▲ 디버그 코드 ▲▲▲▲▲

                # 3. 스케일러 적용
                sequence_scaled = scaler.transform(sequence)

                # 4. 모델 입력 형태로 변경
                x = np.expand_dims(sequence_scaled, axis=0)

                # 5. 예측
                probs = model.predict(x, verbose=0)[0]

                # ★★★★★ 수정된 부분 6: 스무딩 로직 적용 ★★★★★
                idx = int(np.argmax(probs))  # 현재 프레임의 최고 예측 인덱스

                # 예측 큐에 현재 예측 추가
                pred_buffer.append(idx)

                # 예측 큐(최근 5개)에서 가장 많이 나온 값(다수결)을 찾음
                idx_major = max(set(pred_buffer), key=list(pred_buffer).count)

                # 최종 라벨을 다수결로 결정
                predicted_label = CLASS_NAMES[idx_major]
                probabilities = {cls: float(p)
                                 for cls, p in zip(CLASS_NAMES, probs.tolist())}

                await websocket.send_json({
                    "prediction": predicted_label,  # 스무딩된 결과 전송
                    "probabilities": probabilities
                })

                # 슬라이딩 윈도우: 가장 오래된 프레임 제거
                frame_buffer.popleft()

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"⚠️ 예측 중 오류 발생: {e}")
        manager.disconnect(websocket)


# === 로컬 실행 (수정 없음) ===
if __name__ == "__main__":
    # 이 파일 이름을 main_ws_276_smoothing.py로 저장했다면
    # uvicorn.run("main_ws_276_smoothing:app", host="0.0.0.0", port=8000, reload=True)
    # 로 실행해야 합니다.

    # 원래 파일 이름(main_ws_276.py)을 덮어썼다면 아래 코드가 맞습니다.
    uvicorn.run("main_ws_276:app", host="0.0.0.0", port=8000, reload=True)