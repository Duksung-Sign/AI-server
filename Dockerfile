# Dockerfile

# 1. 베이스 이미지 선택 (파이썬 3.10)
FROM python:3.10-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. requirements.txt 파일을 먼저 복사해서 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 프로젝트의 나머지 파일들을 모두 복사
COPY . .

# 5. 서버 실행 (8000번 포트 개방)
EXPOSE 8000

# ✅ "--factory" 옵션 사용 명령어로 변경
CMD ["uvicorn", "--factory", "main_ws:app", "--host", "0.0.0.0", "--port", "8000"]