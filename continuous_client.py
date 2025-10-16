# continuous_client.py

import asyncio
import websockets
import json
import random
import time

# --- 서버 설정 ---
SEQ_LEN = 30
FEATURE_DIM = 244
SERVER_URL = "ws://localhost:8000/ws"
FRAME_RATE = 1/30  # 50ms마다 프레임 생성 (초당 20프레임)
# ------------------

# 서버로부터 오는 메시지를 계속 받아서 출력하는 역할
async def receive_messages(websocket):
    try:
        while True:
            response = await websocket.recv()
            prediction_data = json.loads(response)
            print(f"\n🔥 서버로부터 예측 결과 수신! -> {prediction_data.get('prediction')}\n")
    except websockets.exceptions.ConnectionClosed:
        print(" 서버와의 연결이 끊겼습니다.")

# 서버로 프레임을 계속 보내는 역할
async def send_frames(websocket):
    frame_count = 0
    try:
        while True:
            frame_count += 1
            fake_frame = [random.random() for _ in range(FEATURE_DIM)]
            payload = {"frame": fake_frame}
            await websocket.send(json.dumps(payload))
            print(f"-> 프레임 #{frame_count} 전송")
            await asyncio.sleep(FRAME_RATE)
    except websockets.exceptions.ConnectionClosed:
        print("프레임 전송 중단.")


async def main():
    async with websockets.connect(SERVER_URL) as websocket:
        print(f"✅ 서버에 연결되었습니다: {SERVER_URL}")

        # 메시지 수신과 프레임 전송을 동시에 실행
        receive_task = asyncio.create_task(receive_messages(websocket))
        send_task = asyncio.create_task(send_frames(websocket))

        # 두 작업이 끝날 때까지 기다림 (실제로는 무한 실행)
        await asyncio.gather(receive_task, send_task)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n사용자에 의해 프로그램이 종료되었습니다.")
    except ConnectionRefusedError:
        print("❌ 연결 실패! 서버가 실행 중인지 확인해주세요.")