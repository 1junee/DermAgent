#!/bin/bash

# Qwen3VL-8B 에이전트 비교 실험 실행 스크립트

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 설정
CSV_PATH="/home/work/wonjun/DermAgent/dataset/Derm1M/Derm1M_v2_pretrain_ontology_sampled_1000.csv"
IMAGE_DIR="/home/work/wonjun/DermAgent/dataset/Derm1M"
MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_DIR="./results/qwen3vl_experiments"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 인자 파싱
START_IDX=0
END_IDX=100

while [[ $# -gt 0 ]]; do
    case $1 in
        --start)
            START_IDX="$2"
            shift 2
            ;;
        --end)
            END_IDX="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --gpu)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --start N      Start from sample N (default: 0)"
            echo "  --end N        End at sample N (default: 100)"
            echo "  --output DIR   Output directory (default: ./results/qwen3vl_experiments)"
            echo "  --gpu DEVICES  GPU devices (default: 0,1)"
            echo "  -h, --help     Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run all 100 samples"
            echo "  $0 --start 0 --end 10                 # Run first 10 samples"
            echo "  $0 --start 50 --end 100               # Run last 50 samples"
            echo "  $0 --gpu 0                            # Use only GPU 0"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# GPU 설정
if [ -z "$CUDA_DEVICES" ]; then
    CUDA_DEVICES="0,1"
fi

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         Qwen3VL-8B Agent Comparison Experiment                 ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# 환경 확인
echo -e "${YELLOW}🔍 Checking environment...${NC}"

# Python 확인
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python3: $(python3 --version)${NC}"

# CSV 파일 확인
if [ ! -f "$CSV_PATH" ]; then
    echo -e "${RED}❌ CSV file not found: $CSV_PATH${NC}"
    exit 1
fi
echo -e "${GREEN}✓ CSV file found${NC}"

# 이미지 디렉토리 확인
if [ ! -d "$IMAGE_DIR" ]; then
    echo -e "${RED}❌ Image directory not found: $IMAGE_DIR${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Image directory found${NC}"

# 필수 패키지 확인
echo -e "${YELLOW}📦 Checking required packages...${NC}"
REQUIRED_PACKAGES=("torch" "transformers" "qwen_vl_utils" "tqdm" "PIL")
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if [ "$pkg" = "PIL" ]; then
        python3 - <<'PY' 2>/dev/null && ok=1 || ok=0
import PIL
from PIL import Image  # verify core submodule
PY
    else
        python3 -c "import ${pkg}" 2>/dev/null && ok=1 || ok=0
    fi

    if [ "$ok" -eq 1 ]; then
        echo -e "${GREEN}✓ ${pkg} installed${NC}"
    else
        # pip 패키지명 안내 (PIL → Pillow)
        pip_name=$([ "$pkg" = "PIL" ] && echo "Pillow" || echo "$pkg")
        echo -e "${RED}❌ ${pkg} not installed${NC}"
        echo -e "${YELLOW}Install with: pip install ${pip_name}${NC}"
        exit 1
    fi
done

# GPU 확인
echo -e "${YELLOW}🖥️  GPU Configuration:${NC}"
echo -e "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | head -n 2

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    Experiment Configuration                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo -e "📁 CSV Path:    ${CSV_PATH}"
echo -e "🖼️  Image Dir:   ${IMAGE_DIR}"
echo -e "🤖 Model:       ${MODEL_PATH}"
echo -e "💾 Output:      ${OUTPUT_DIR}"
echo -e "📊 Samples:     ${START_IDX} to ${END_IDX} (total: $((END_IDX - START_IDX)))"
echo -e "🎮 GPU:         ${CUDA_DEVICES}"
echo ""

# 확인 프롬프트
echo -e "${YELLOW}Press Enter to start, or Ctrl+C to cancel...${NC}"
read

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

# 시작 시간 기록
START_TIME=$(date +%s)

# 실험 실행
echo ""
echo -e "${GREEN}🚀 Starting experiment...${NC}"
echo ""

CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python3 "${SCRIPT_DIR}/run_qwen3vl_experiments.py" \
    --csv "$CSV_PATH" \
    --image_dir "$IMAGE_DIR" \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_DIR" \
    --start "$START_IDX" \
    --end "$END_IDX"

EXIT_CODE=$?

# 종료 시간 기록
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                  Experiment Completed! ✓                       ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo -e "⏱️  Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo -e "💾 Results saved in: ${OUTPUT_DIR}"
    echo ""

    # 결과 파일 확인
    if [ -d "$OUTPUT_DIR" ]; then
        echo -e "${BLUE}📄 Output files:${NC}"
        ls -lh "$OUTPUT_DIR" | tail -n +2
    fi
else
    echo -e "${RED}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                  Experiment Failed! ✗                          ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo -e "Exit code: $EXIT_CODE"
fi

echo ""
