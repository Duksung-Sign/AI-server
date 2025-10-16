# client_test.py

import asyncio
import websockets
import json
import random
import time

# --- 서버 설정과 동일하게 맞춤 ---
SEQ_LEN = 30
FEATURE_DIM = 244
SERVER_URL = "ws://localhost:8000/ws"


# --------------------------------

async def run_test():
    """테스트 클라이언트 실행"""
    # with 블록을 사용해 웹소켓 서버에 접속합니다.
    async with websockets.connect(SERVER_URL) as websocket:
        print(f"✅ 서버에 연결되었습니다: {SERVER_URL}")

        # 30개의 프레임을 0.05초 간격으로 순차적으로 전송합니다.
        for i in range(SEQ_LEN):
            # 실제로는 미디어파이프 등에서 나온 244차원 벡터여야 합니다.
            # 여기서는 테스트를 위해 랜덤 값으로 채웁니다.
            fake_frame = [random.random() for _ in range(FEATURE_DIM)]

            # 서버가 받을 수 있는 JSON 형태로 데이터를 만듭니다.
            payload = {"frame": fake_frame}
            await websocket.send(json.dumps(payload))

            print(f"-> 프레임 {i + 1}/{SEQ_LEN} 전송 완료")
            time.sleep(0.05)  # 실제 스트리밍처럼 약간의 딜레이를 줍니다.

        print("\n✅ 30개 프레임 전송 완료. 서버의 예측 결과를 기다립니다...")

        # 서버로부터 예측 결과 메시지를 수신합니다.
        response = await websocket.recv()
        prediction_data = json.loads(response)

        print("\n===== 서버 예측 결과 (Prediction Result) =====")
        print(f"예측 라벨 (Predicted Label): {prediction_data.get('prediction')}")
        # print("각 클래스 확률 (Probabilities):", prediction_data.get('probabilities'))
        print("==============================================")


if __name__ == "__main__":
    # `websockets` 라이브러리가 없다면 설치해주세요: pip install websockets
    try:
        asyncio.run(run_test())
    except ConnectionRefusedError:
        print("❌ 연결 실패! 서버가 실행 중인지 확인해주세요.")