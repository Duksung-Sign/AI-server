import numpy as np

# === 파일 경로 ===
UNITY_INPUT_PATH = "unity_input_sample.npy"         # Unity에서 저장된 입력 (30, 276)
TRAIN_DATA_PATH  = "X_train_scaled_gyro_276.npy"    # 학습 시 사용한 정규화된 데이터 (N, 30, 276)

# === 데이터 로드 ===
unity_input = np.load(UNITY_INPUT_PATH)
train_data  = np.load(TRAIN_DATA_PATH)

print(f"✅ Unity 입력 shape: {unity_input.shape}")
print(f"✅ 학습 데이터 shape: {train_data.shape}")

if unity_input.shape != (30, 276):
    print(f"⚠️ Unity 입력 shape 예상과 다름: {unity_input.shape}")

# === 통계 계산 함수 ===
def describe(arr, name):
    print(f"\n📊 [{name}]")
    print(f"  min:  {arr.min():.4f}")
    print(f"  max:  {arr.max():.4f}")
    print(f"  mean: {arr.mean():.4f}")
    print(f"  std:  {arr.std():.4f}")

# === 분포 비교 ===
describe(train_data, "Train Data (Scaled)")
describe(unity_input, "Unity Input (Received)")

mean_diff = abs(unity_input.mean() - train_data.mean())
std_diff  = abs(unity_input.std()  - train_data.std())

print("\n⚖️ 평균 차이:", round(mean_diff, 4))
print("⚖️ 표준편차 차이:", round(std_diff, 4))

if mean_diff > 0.1 or std_diff > 0.1:
    print("🚨 Unity 입력 분포가 학습 데이터와 다릅니다! (스케일 불일치 가능성 높음)")
else:
    print("✅ Unity 입력 분포가 학습 데이터와 유사합니다.")
