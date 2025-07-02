# test_predict.py
import requests
import numpy as np

# 실제 데이터 불러오기
X = np.load("model/X_holistic_58_mask.npy")  # 경로 주의
sample = X[0].tolist()  # (30, 236)

res = requests.post("http://127.0.0.1:8000/predict", json={"sequence": sample})
print(res.json())
