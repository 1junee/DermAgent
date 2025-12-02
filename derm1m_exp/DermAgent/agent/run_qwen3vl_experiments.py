#!/usr/bin/env python3
"""
Qwen3VL-8B로 100개 샘플에 대해 두 에이전트 비교 실험
각 샘플의 결과를 화면에 출력하고 파일로도 저장
"""

import os
import sys
import csv
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from tqdm import tqdm

# 경로 설정
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "eval"))

from dermatology_agent import DermatologyAgent
from react_agent import ReActDermatologyAgent
from evaluation_metrics import HierarchicalEvaluator


class QwenVLM:
    """Qwen3-VL 모델 래퍼"""

    def __init__(self, model_path: str, logger=None):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor
        from qwen_vl_utils import process_vision_info

        print(f"Loading Qwen3VL model from {model_path}...")
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.process_vision_info = process_vision_info
        self.logger = logger
        print("Model loaded successfully!")

    def _log(self, message: str):
        if self.logger:
            self.logger.info(message)

    def chat_img(self, prompt: str, image_paths: List[str], max_tokens: int = 1024) -> str:
        """이미지와 함께 채팅"""
        messages = [{"role": "user", "content": []}]

        # 이미지 추가
        for path in image_paths:
            if os.path.exists(path):
                messages[0]["content"].append({"type": "image", "image": f"file://{path}"})

        # 텍스트 프롬프트 추가
        messages[0]["content"].append({"type": "text", "text": prompt})

        self._log(f"\n{'='*60}")
        self._log(f"Qwen3VL Request")
        self._log(f"{'='*60}")
        self._log(f"Prompt (first 300 chars): {prompt[:300]}...")
        self._log(f"Images: {image_paths}")
        self._log(f"{'='*60}\n")

        try:
            # 프롬프트 생성
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = self.process_vision_info(messages)

            # 입력 준비
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)

            # 생성
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
            generated = outputs[0][inputs.input_ids.shape[1]:]
            response = self.processor.decode(generated, skip_special_tokens=True)

            self._log(f"\n{'='*60}")
            self._log(f"Qwen3VL Response")
            self._log(f"{'='*60}")
            self._log(response)
            self._log(f"{'='*60}\n")

            return response

        except Exception as e:
            self._log(f"Qwen3VL Error: {e}")
            return "{}"


class GPT4oVLM:
    """GPT-4o 래퍼 (OpenAI API)"""

    def __init__(self, api_key: str = None, model: str = "gpt-4o", logger=None):
        import openai
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for GPT-4o backend")
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.logger = logger

    def _log(self, message: str):
        if self.logger:
            self.logger.info(message)

    def chat_img(self, prompt: str, image_paths: List[str], max_tokens: int = 1024) -> str:
        import base64

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        for path in image_paths:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })

        self._log(f"[GPT4o] prompt preview: {prompt[:200]}...")

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens
            )
            return resp.choices[0].message.content
        except Exception as e:
            self._log(f"GPT4o Error: {e}")
            return "{}"


def setup_logging(output_dir: Path, experiment_name: str):
    """로깅 설정"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"{experiment_name}_{timestamp}.log"

    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # 파일 핸들러
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)

    return logger, log_file


def print_result_summary(
    sample_idx: int,
    sample: Dict,
    derm_result: Dict,
    react_result: Dict,
    derm_eval: Dict,
    react_eval: Dict
):
    """결과 요약을 화면에 출력"""
    print("\n" + "="*80)
    print(f"📊 SAMPLE {sample_idx + 1}/100")
    print("="*80)

    # 샘플 정보
    print(f"\n📷 Image: {sample['filename']}")
    print(f"🏷️  Ground Truth: {sample.get('disease_label', 'N/A')}")
    print(f"📝 Hierarchical: {sample.get('hierarchical_disease_label', 'N/A')}")

    # DermatologyAgent 결과
    print(f"\n{'🔹 DermatologyAgent (Fixed 5-Step)':-<80}")
    derm_pred = derm_result.get('final_diagnosis', ['N/A'])[0] if derm_result.get('final_diagnosis') else 'N/A'
    derm_conf = derm_result.get('confidence_scores', {}).get(derm_pred, 0.0)
    derm_path = derm_result.get('diagnosis_path', [])
    print(f"Prediction: {derm_pred}")
    print(f"Confidence: {derm_conf:.2f}")
    print(f"Path: {' > '.join(derm_path)}")

    # 평가 지표 - evaluate_single returns metrics directly
    print(f"✓ Exact Match: {derm_eval.get('exact_match', 0)}")
    print(f"✓ Hierarchical F1: {derm_eval.get('hierarchical_f1', 0):.3f}")
    print(f"✓ Distance: {derm_eval.get('avg_min_distance', 0):.2f}")

    # ReActAgent 결과
    print(f"\n{'🔸 ReActAgent (Dynamic)':-<80}")
    print(f"Prediction: {react_result.get('primary_diagnosis', 'N/A')}")
    print(f"Confidence: {react_result.get('confidence', 0):.2f}")
    print(f"Path: {' > '.join(react_result.get('ontology_path', []))}")
    print(f"Steps: {len(react_result.get('reasoning_chain', []))}")

    # 평가 지표 - evaluate_single returns metrics directly
    print(f"✓ Exact Match: {react_eval.get('exact_match', 0)}")
    print(f"✓ Hierarchical F1: {react_eval.get('hierarchical_f1', 0):.3f}")
    print(f"✓ Distance: {react_eval.get('avg_min_distance', 0):.2f}")

    # 비교
    print(f"\n{'📈 Comparison':-<80}")
    derm_f1 = derm_eval.get('hierarchical_f1', 0)
    react_f1 = react_eval.get('hierarchical_f1', 0)

    if derm_f1 > react_f1:
        winner = "🏆 DermatologyAgent wins"
    elif react_f1 > derm_f1:
        winner = "🏆 ReActAgent wins"
    else:
        winner = "🤝 Tie"

    print(f"{winner} (F1: {derm_f1:.3f} vs {react_f1:.3f})")
    print("="*80)


def run_experiment(
    csv_path: str,
    image_dir: str,
    model_path: str,
    output_dir: Path,
    backend: str = "auto",
    api_key: str = None,
    start_idx: int = 0,
    end_idx: int = None
):
    """실험 실행"""

    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    # 로거 설정
    main_logger, log_file = setup_logging(output_dir, "qwen3vl_experiment")

    print("\n" + "="*80)
    print("🚀 Starting Qwen3VL Agent Comparison Experiment")
    print("="*80)
    print(f"📁 CSV: {csv_path}")
    print(f"🖼️  Images: {image_dir}")
    print(f"🤖 Model: {model_path}")
    print(f"🧠 Backend: {backend}")
    print(f"💾 Output: {output_dir}")
    print(f"📄 Log: {log_file}")
    print("="*80 + "\n")

    # VLM 모델 로드
    if backend == "auto":
        backend = "gpt4o" if model_path.lower().startswith("gpt") else "qwen"

    if backend == "gpt4o":
        vlm = GPT4oVLM(api_key=api_key, model=model_path, logger=main_logger)
    elif backend == "qwen":
        vlm = QwenVLM(model_path, logger=main_logger)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # 평가기 초기화
    evaluator = HierarchicalEvaluator()

    # CSV 로드
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        samples = list(reader)

    if end_idx is None:
        end_idx = len(samples)

    samples = samples[start_idx:end_idx]

    print(f"📊 Processing {len(samples)} samples ({start_idx} to {end_idx})\n")

    # 결과 저장
    all_results = []

    # 각 샘플 처리
    for idx, sample in enumerate(tqdm(samples, desc="Processing samples")):

        # 이미지 경로
        image_path = Path(image_dir) / sample['filename']
        if not image_path.exists():
            print(f"⚠️  Image not found: {image_path}")
            continue

        # Ground truth
        gt_label = sample.get('disease_label', '')
        gt_hierarchical = sample.get('hierarchical_disease_label', '')

        main_logger.info(f"\n{'='*80}")
        main_logger.info(f"Sample {start_idx + idx + 1}/{len(samples)}")
        main_logger.info(f"Image: {sample['filename']}")
        main_logger.info(f"Ground Truth: {gt_label}")
        main_logger.info(f"{'='*80}\n")

        # 1. DermatologyAgent 실행
        main_logger.info("Running DermatologyAgent...")
        try:
            derm_agent = DermatologyAgent(
                ontology_path=None,
                vlm_model=vlm,
                verbose=False
            )
            derm_result = derm_agent.diagnose(str(image_path))
            if not isinstance(derm_result, dict):
                raise TypeError("DermatologyAgent should return dict")
        except Exception as e:
            print(f"⚠️  Error in DermatologyAgent: {e}")
            main_logger.error(f"DermatologyAgent failed for {sample['filename']}: {e}")
            derm_result = {
                "primary_diagnosis": "error",
                "confidence": 0.0,
                "final_diagnosis": ["error"],
                "ontology_path": []
            }

        # 2. ReActAgent 실행
        main_logger.info("Running ReActAgent...")
        try:
            react_agent = ReActDermatologyAgent(
                ontology_path=None,
                vlm_model=vlm,
                max_steps=8
            )
            react_result_obj = react_agent.diagnose(str(image_path))
            react_result = react_result_obj.to_dict()  # Convert DiagnosisResult object to dict
        except Exception as e:
            print(f"⚠️  Error in ReActAgent: {e}")
            main_logger.error(f"ReActAgent failed for {sample['filename']}: {e}")
            react_result = {
                "primary_diagnosis": "error",
                "confidence": 0.0,
                "differential_diagnoses": [],
                "ontology_path": []
            }

        # 3. 평가
        # DermatologyAgent returns final_diagnosis (list), confidence_scores (dict), diagnosis_path (list)
        derm_primary = derm_result.get('final_diagnosis', ['unknown'])[0] if derm_result.get('final_diagnosis') else 'unknown'
        derm_confidence = derm_result.get('confidence_scores', {}).get(derm_primary, 0.0)
        derm_path = derm_result.get('diagnosis_path', [])

        # ReActAgent returns primary_diagnosis (str), confidence (float), ontology_path (list)
        react_primary = react_result.get('primary_diagnosis', 'unknown')
        react_confidence = react_result.get('confidence', 0.0)
        react_path = react_result.get('ontology_path', [])

        derm_pred = [derm_primary]
        react_pred = [react_primary]
        gt = [gt_label] if gt_label else ['unknown']

        derm_eval = evaluator.evaluate_single(gt, derm_pred)
        react_eval = evaluator.evaluate_single(gt, react_pred)

        # 4. 결과 저장
        result_entry = {
            'sample_idx': start_idx + idx,
            'filename': sample['filename'],
            'ground_truth': gt_label,
            'ground_truth_hierarchical': gt_hierarchical,
            'dermatology_agent': {
                'prediction': derm_primary,
                'confidence': derm_confidence,
                'ontology_path': derm_path,
                'metrics': derm_eval  # evaluate_single returns metrics directly, not nested
            },
            'react_agent': {
                'prediction': react_primary,
                'confidence': react_confidence,
                'ontology_path': react_path,
                'reasoning_steps': len(react_result.get('reasoning_chain', [])),
                'metrics': react_eval  # evaluate_single returns metrics directly, not nested
            }
        }
        all_results.append(result_entry)

        # 5. 화면 출력
        print_result_summary(
            start_idx + idx, sample,
            derm_result, react_result,
            derm_eval, react_eval
        )

        # 중간 저장 (10개마다)
        if (idx + 1) % 10 == 0:
            interim_file = output_dir / f"results_interim_{start_idx}_{start_idx + idx + 1}.json"
            with open(interim_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Interim results saved to {interim_file}")

    # 최종 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_file = output_dir / f"results_final_{start_idx}_{end_idx}_{timestamp}.json"
    with open(final_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 통계 출력
    print("\n" + "="*80)
    print("📊 FINAL STATISTICS")
    print("="*80)

    # Safe dictionary access with nested .get()
    derm_wins = 0
    react_wins = 0
    ties = 0

    derm_f1_scores = []
    react_f1_scores = []

    for r in all_results:
        # metrics are stored directly, not nested under 'metrics' key
        derm_f1 = r.get('dermatology_agent', {}).get('metrics', {}).get('hierarchical_f1', 0)
        react_f1 = r.get('react_agent', {}).get('metrics', {}).get('hierarchical_f1', 0)

        derm_f1_scores.append(derm_f1)
        react_f1_scores.append(react_f1)

        if derm_f1 > react_f1:
            derm_wins += 1
        elif react_f1 > derm_f1:
            react_wins += 1
        else:
            ties += 1

    derm_avg_f1 = sum(derm_f1_scores) / len(derm_f1_scores) if derm_f1_scores else 0.0
    react_avg_f1 = sum(react_f1_scores) / len(react_f1_scores) if react_f1_scores else 0.0

    print(f"Total samples: {len(all_results)}")
    print(f"\nWins:")
    if all_results:
        print(f"  🔹 DermatologyAgent: {derm_wins} ({derm_wins/len(all_results)*100:.1f}%)")
        print(f"  🔸 ReActAgent: {react_wins} ({react_wins/len(all_results)*100:.1f}%)")
        print(f"  🤝 Ties: {ties} ({ties/len(all_results)*100:.1f}%)")
        print(f"\nAverage Hierarchical F1:")
        print(f"  🔹 DermatologyAgent: {derm_avg_f1:.3f}")
        print(f"  🔸 ReActAgent: {react_avg_f1:.3f}")
    else:
        print("  No results to analyze")
    print(f"\n💾 Final results saved to: {final_file}")
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3VL Agent Comparison Experiment")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--image_dir", required=True, help="Directory containing images")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct", help="Model path or OpenAI model name")
    parser.add_argument("--output", default="./results", help="Output directory")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=None, help="End index")
    parser.add_argument("--backend", choices=["auto", "qwen", "gpt4o"], default="auto",
                        help="VLM backend selection (auto: infer from model name)")
    parser.add_argument("--api_key", default=None, help="API key (required for GPT-4o backend)")

    args = parser.parse_args()

    run_experiment(
        csv_path=args.csv,
        image_dir=args.image_dir,
        model_path=args.model,
        output_dir=Path(args.output),
        backend=args.backend,
        api_key=args.api_key,
        start_idx=args.start,
        end_idx=args.end
    )
