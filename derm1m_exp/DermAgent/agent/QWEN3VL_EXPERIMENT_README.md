# Qwen3VL-8B 에이전트 비교 실험 가이드

Qwen3VL-8B 모델로 DermatologyAgent와 ReActAgent를 비교하는 실험입니다.

## 📋 개요

- **모델**: Qwen/Qwen3-VL-8B-Instruct
- **데이터**: 100개 샘플 (Derm1M_v2_pretrain_ontology_sampled_100.csv)
- **비교 대상**:
  - 🔹 **DermatologyAgent**: 고정 5단계 파이프라인
  - 🔸 **ReActAgent**: 동적 추론 기반

## 🚀 빠른 시작

### 1. 모든 100개 샘플 실행

```bash
cd /home/work/wonjun/DermAgent/derm1m_exp/DermAgent/agent
./run_qwen3vl_experiment.sh
```

### 2. 특정 범위만 실행

```bash
# 처음 10개만
./run_qwen3vl_experiment.sh --start 0 --end 10

# 50번째부터 100번째까지
./run_qwen3vl_experiment.sh --start 50 --end 100

# 특정 샘플 하나 (예: 5번)
./run_qwen3vl_experiment.sh --start 5 --end 6
```

### 3. GPU 설정

```bash
# GPU 0만 사용
./run_qwen3vl_experiment.sh --gpu 0

# GPU 2,3 사용
./run_qwen3vl_experiment.sh --gpu 2,3
```

### 4. 출력 디렉토리 지정

```bash
./run_qwen3vl_experiment.sh --output ./my_results
```

## 📊 출력 형식

### 화면 출력 예시

```
================================================================================
📊 SAMPLE 1/100
================================================================================

📷 Image: youtube/TyY1qef8dIM_frame_562_0.jpg
🏷️  Ground Truth: allergic contact dermatitis
📝 Hierarchical: inflammatory, non-infectious, eczema, contact dermatitis, allergic contact dermatitis

🔹 DermatologyAgent (Fixed 5-Step)────────────────────────────────────────────
Prediction: allergic contact dermatitis
Confidence: 0.85
Path: inflammatory > eczema > contact dermatitis > allergic contact dermatitis
✓ Exact Match: 1
✓ Hierarchical F1: 1.000
✓ Distance: 0.00

🔸 ReActAgent (Dynamic)───────────────────────────────────────────────────────
Prediction: contact dermatitis
Confidence: 0.78
Path: inflammatory > eczema > contact dermatitis
Steps: 6
✓ Exact Match: 0
✓ Hierarchical F1: 0.857
✓ Distance: 1.00

📈 Comparison─────────────────────────────────────────────────────────────────
🏆 DermatologyAgent wins (F1: 1.000 vs 0.857)
================================================================================
```

### 저장 파일

실험 결과는 `./results/qwen3vl_experiments/` 디렉토리에 저장됩니다:

1. **중간 결과** (10개마다 자동 저장)
   - `results_interim_0_10.json`
   - `results_interim_0_20.json`
   - ...

2. **최종 결과**
   - `results_final_0_100_20251202_143000.json`

3. **로그 파일**
   - `qwen3vl_experiment_20251202_143000.log`

### 결과 JSON 구조

```json
[
  {
    "sample_idx": 0,
    "filename": "youtube/TyY1qef8dIM_frame_562_0.jpg",
    "ground_truth": "allergic contact dermatitis",
    "ground_truth_hierarchical": "inflammatory, non-infectious, eczema, contact dermatitis, allergic contact dermatitis",
    "dermatology_agent": {
      "prediction": "allergic contact dermatitis",
      "confidence": 0.85,
      "ontology_path": ["inflammatory", "eczema", "contact dermatitis", "allergic contact dermatitis"],
      "metrics": {
        "exact_match": 1,
        "hierarchical_f1": 1.0,
        "hierarchical_distance": 0.0,
        "partial_credit": 1.0
      }
    },
    "react_agent": {
      "prediction": "contact dermatitis",
      "confidence": 0.78,
      "ontology_path": ["inflammatory", "eczema", "contact dermatitis"],
      "reasoning_steps": 6,
      "metrics": {
        "exact_match": 0,
        "hierarchical_f1": 0.857,
        "hierarchical_distance": 1.0,
        "partial_credit": 0.9
      }
    }
  }
]
```

## 📈 최종 통계

실험 완료 후 다음과 같은 통계가 출력됩니다:

```
================================================================================
📊 FINAL STATISTICS
================================================================================
Total samples: 100

Wins:
  🔹 DermatologyAgent: 45 (45.0%)
  🔸 ReActAgent: 38 (38.0%)
  🤝 Ties: 17 (17.0%)

Average Hierarchical F1:
  🔹 DermatologyAgent: 0.782
  🔸 ReActAgent: 0.756

💾 Final results saved to: ./results/qwen3vl_experiments/results_final_0_100_20251202_143000.json
================================================================================
```

## 🔧 고급 사용법

### Python 스크립트 직접 실행

```bash
python run_qwen3vl_experiments.py \
    --csv /path/to/data.csv \
    --image_dir /path/to/images \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --output ./results \
    --start 0 \
    --end 100
```

### 배치 실행 (여러 GPU에서 병렬)

```bash
# GPU 0에서 0-50
CUDA_VISIBLE_DEVICES=0 python run_qwen3vl_experiments.py \
    --csv data.csv --image_dir images --output results/gpu0 \
    --start 0 --end 50 &

# GPU 1에서 50-100
CUDA_VISIBLE_DEVICES=1 python run_qwen3vl_experiments.py \
    --csv data.csv --image_dir images --output results/gpu1 \
    --start 50 --end 100 &

wait
echo "Both done!"
```

## 📝 주의사항

1. **메모리 요구사항**: Qwen3VL-8B는 약 16GB VRAM 필요
2. **실행 시간**: 샘플당 약 30-60초 소요 (하드웨어에 따라 다름)
3. **중단/재개**: Ctrl+C로 중단 가능. `--start` 옵션으로 재개 가능
4. **로그 확인**: 상세 로그는 `results/` 디렉토리의 `.log` 파일 참조

## 🐛 문제 해결

### 모델 로딩 실패
```bash
# Hugging Face 캐시 확인
ls ~/.cache/huggingface/hub/

# 수동 다운로드
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen3-VL-8B-Instruct')"
```

### CUDA Out of Memory
```bash
# 배치 크기 줄이기 또는 더 큰 GPU 사용
# 또는 모델을 INT8/INT4로 양자화
```

### 이미지 경로 오류
```bash
# CSV와 이미지 디렉토리 구조 확인
head -3 /path/to/data.csv
ls /path/to/images/youtube/
```

## 📧 문의

문제가 발생하면 로그 파일과 함께 문의해주세요.
