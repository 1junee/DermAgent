#!/bin/bash

# GPT-4o로 피부과 진단 실행
# .env 파일에서 API 키 로드

# 스크립트 디렉토리로 이동
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# .env 파일에서 환경변수 로드
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "Loaded environment variables from .env"
else
    echo "Error: .env file not found in $SCRIPT_DIR"
    echo "Please create a .env file with OPENAI_API_KEY=your-api-key"
    echo "You can copy .env.template to .env and edit it:"
    echo "  cp .env.template .env"
    exit 1
fi

# API 키 확인
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY is not set in .env file"
    echo "Please add OPENAI_API_KEY=your-api-key to .env"
    exit 1
fi

# 경로 설정
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Starting GPT-4o dermatology diagnosis..."
echo "Note: Paths are managed by project_path.py"

# baseline.py 실행 (기본값 사용)
python "${SCRIPT_DIR}/baseline.py" \
    --model gpt \
    --api_key "${OPENAI_API_KEY}"

echo "Done!"
