## 테스트용 특징벡터 추출하는 코드
import numpy as np
import json

# === 특징 벡터 파일 로드 ===
X = np.load("model/X_unified_244_aug.npy")  # ← 경로에 맞게 수정
y = np.load("model/y_unified_244_aug.npy")  # ← 클래스 인덱스 저장한 라벨 파일

# === 클래스 인덱스 매핑 ===
CLASS_NAMES = [
    "thx", "study", "okay", "me", "you", "arrive", "nicetomeet",
    "none", "hate", "hello", "call", "goodnice", "sorry"
]

# === "죄송합니다" 클래스 인덱스 찾기
target_class = "sorry"
target_index = CLASS_NAMES.index(target_class)

# === 해당 클래스의 첫 번째 샘플 추출
sample = X[y == target_index][0]  # (30, 244)

# === JSON 저장
output = {"sequence": sample.tolist()}

with open("sample_sorry_input.json", "w") as f:
    json.dump(output, f, indent=2)

print("✅ '죄송합니다' 특징 벡터 저장 완료: sample_sorry_input.json")
