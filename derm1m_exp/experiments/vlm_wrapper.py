"""
VLM 래퍼 모듈

GPT-4o를 독립적으로 사용할 수 있는 래퍼 클래스
qwen_vl_utils 등 의존성 문제를 피하기 위해 별도로 구현
"""

import os
import base64
from pathlib import Path
from typing import List, Optional
from openai import OpenAI
import openai


def encode_image(image_path: str) -> str:
    """이미지를 base64로 인코딩"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def compress_image(image_path: str, output_path: str, quality: int = 50) -> str:
    """이미지 압축"""
    try:
        from PIL import Image
        img = Image.open(image_path)

        # RGBA -> RGB 변환 (JPEG는 알파 채널 지원 안함)
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')

        # 크기 조정 (필요시)
        max_size = 1024
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path, 'JPEG', quality=quality)
        return output_path
    except Exception as e:
        print(f"이미지 압축 실패: {e}")
        return image_path


class GPT4oWrapper:
    """
    GPT-4o 래퍼 클래스

    기존 model.py의 GPT4o 클래스와 호환되는 인터페이스 제공
    """

    def __init__(
        self,
        api_key: str,
        use_labels_prompt: bool = True,
        model: str = "gpt-4o",
        disease_labels_path: Optional[str] = None
    ):
        """
        Args:
            api_key: OpenAI API 키
            use_labels_prompt: 질병 라벨 목록 사용 여부
            model: 사용할 모델 (기본값: gpt-4o)
            disease_labels_path: 질병 라벨 파일 경로
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model

        # 시스템 프롬프트 설정
        if use_labels_prompt:
            disease_labels_str = self._load_disease_labels(disease_labels_path)
            self.instruction = f"""You are a dermatology expert. You are provided with a skin image and a question about it.
Please analyze the image carefully and provide a detailed diagnosis or answer based on your expertise.
Focus on identifying skin conditions, lesions, or abnormalities visible in the image.

When identifying skin conditions, the disease_label should be one of the following: {disease_labels_str}

Provide a clear and professional response."""
        else:
            self.instruction = """You are a dermatology expert. You are provided with a skin image and a question about it.
Analyze the image carefully and provide a detailed, professional answer about visible skin findings.
Do NOT assume or reference any predefined disease label list. If uncertain, state the likely differentials."""

    def _load_disease_labels(self, disease_labels_path: Optional[str] = None) -> str:
        """질병 라벨 목록 로드"""
        if disease_labels_path is None:
            # 기본 경로 탐색
            script_dir = Path(__file__).parent
            possible_paths = [
                script_dir.parent / "baseline" / "extracted_node_names.txt",
                script_dir.parent.parent / "derm1m_exp" / "baseline" / "extracted_node_names.txt",
            ]
            for path in possible_paths:
                if path.exists():
                    disease_labels_path = str(path)
                    break

        if disease_labels_path and os.path.exists(disease_labels_path):
            with open(disease_labels_path, "r") as f:
                disease_labels = [
                    line.strip().split("→")[1] if "→" in line else line.strip()
                    for line in f.readlines() if line.strip()
                ]
            return ", ".join(disease_labels)
        else:
            # 온톨로지에서 직접 로드 시도
            try:
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
                from ontology_utils import OntologyTree
                tree = OntologyTree()
                return ", ".join(sorted(tree.valid_nodes))
            except Exception:
                return ""

    def chat_img(
        self,
        input_text: str,
        image_path: List[str],
        max_tokens: int = 512
    ) -> str:
        """
        이미지와 텍스트로 대화

        Args:
            input_text: 질문 텍스트
            image_path: 이미지 경로 리스트
            max_tokens: 최대 토큰 수

        Returns:
            모델 응답
        """
        try:
            # 이미지 인코딩
            base64_images = []
            for image in image_path:
                if os.path.exists(image):
                    base64_image = encode_image(image)
                    base64_images.append(base64_image)

            # 메시지 구성
            messages = [
                {"role": "system", "content": self.instruction},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input_text},
                    ]
                }
            ]

            # 이미지 추가
            for image in base64_images:
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image}"
                    }
                })

            # API 호출
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens
            )

            # None 응답 처리 - API가 content를 반환하지 않을 수 있음
            content = completion.choices[0].message.content
            if content is None:
                print("[VLM Warning] API returned None content, retrying once...")
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens
                )
                content = completion.choices[0].message.content
                if content is None:
                    print("[VLM Warning] API returned None content again; returning empty string.")
                    return ""
            return content

        except openai.APIStatusError as e:
            if "413" in str(e):
                print("이미지가 너무 큽니다. 압축 후 재시도...")
                compressed_images = []
                # tmp_file 경로를 experiments 폴더 내부로 설정
                tmp_dir = Path(__file__).parent / "tmp_file"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                for i, image in enumerate(image_path):
                    compressed_path = str(tmp_dir / f"compressed_image_{i}.jpg")
                    compressed_images.append(compress_image(image, compressed_path, quality=50))
                return self.chat_img(input_text, compressed_images, max_tokens)
            else:
                raise e


# GPT4o 별칭 (기존 코드 호환성)
GPT4o = GPT4oWrapper


def test_vlm():
    """VLM 래퍼 테스트"""
    print("=" * 60)
    print("VLM Wrapper Test")
    print("=" * 60)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("테스트를 건너뜁니다.")
        return

    print("GPT4oWrapper 초기화 테스트...")
    vlm = GPT4oWrapper(api_key=api_key, use_labels_prompt=False)
    print(f"  모델: {vlm.model}")
    print(f"  시스템 프롬프트 길이: {len(vlm.instruction)} chars")
    print("초기화 성공!")


if __name__ == "__main__":
    test_vlm()
