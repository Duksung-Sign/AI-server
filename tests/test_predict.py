import requests
import numpy as np

# === 실제 데이터 불러오기 ===
# 반드시 (30, 244) 형태의 샘플이여야 합니다.
X = np.load("model/X_unified_244_aug.npy")  # 경로 확인 필요
sample = X[0].tolist()  # 리스트로 변환하여 JSON 전송

# === POST 요청 ===
url = "http://127.0.0.1:8000/predict"
res = requests.post(url, json={"sequence": sample})

# === 결과 출력 ===
print("✅ 예측 결과:", res.json())
