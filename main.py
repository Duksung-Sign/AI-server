from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import uvicorn

# === FastAPI 앱 생성 ===
app = FastAPI()

# === 모델 로드 ===
model = tf.keras.models.load_model("sign_language_lstm_model_none.h5", compile=False)
class_names = ['hello', 'sorry', 'thank', 'none']

# === 입력 데이터 스키마 정의 ===
class InputData(BaseModel):
    sequence: list[list[float]]  # (30, 148)

# === 루트 라우트 (GET) ===
@app.get("/")
async def root():
    return {"message": "FastAPI on EC2!"}

# === 테스트 라우트 (POST) ===
@app.post("/test")
async def test(data: dict):
    return {
        "received": data,
        "message": "It works!"
    }

# === 예측 라우트 (POST) ===
@app.post("/predict",
          summary="수어 시퀀스 예측",
          description="30프레임짜리 수어 좌표 시퀀스를 입력받아 해당 수어를 예측합니다."
          )
async def predict(data: InputData):
    try:
        input_array = np.array(data.sequence, dtype=np.float32)

        if input_array.shape != (30, 148):
            return {"error": f"Input shape must be (30, 148), but got {input_array.shape}"}

        input_array = np.expand_dims(input_array, axis=0)  # (1, 30, 148)
        prediction = model.predict(input_array, verbose=0)[0]

        predicted_label = class_names[int(np.argmax(prediction))]
        probabilities = dict(zip(class_names, prediction.tolist()))

        return {
            "prediction": predicted_label,
            "probabilities": probabilities
        }

    except Exception as e:
        return {"error": str(e)}
