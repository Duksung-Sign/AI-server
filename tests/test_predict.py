import requests
import numpy as np

url = "http://127.0.0.1:5000/predict"

# 30프레임 × 148개 특징의 더미 입력
dummy = np.random.rand(30, 148).tolist()

res = requests.post(url, json={"sequence": dummy})

print("Status Code:", res.status_code)
print("Response:", res.json())
