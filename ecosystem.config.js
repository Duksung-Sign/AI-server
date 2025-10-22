// ecosystem.config.js
module.exports = {
  apps : [{
    name   : "my-sign-api", // pm2에서 보여질 앱 이름
    script : "/home/ubuntu/AI-server/venv/bin/uvicorn", // uvicorn 실행 파일 전체 경로
    args   : "main_ws:app --host 0.0.0.0 --port 8000", // uvicorn에게 전달할 인자들
    // interpreter: "/home/ubuntu/AI-server/venv/bin/python", // 사용할 파이썬 경로
    cwd    : "/home/ubuntu/AI-server/", // 앱 실행 디렉토리
    watch  : false, // 파일 변경 감지 비활성화 (개발 중이면 true)
    env    : {
        "NODE_ENV": "production",
        // 필요한 다른 환경 변수 추가 가능 (예: MODEL_PATH)
        // "MODEL_PATH": "/app/model/your_model.h5"
    }
  }]
}