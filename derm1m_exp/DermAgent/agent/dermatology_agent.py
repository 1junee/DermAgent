"""
Dermatology Diagnosis Agent Framework

온톨로지 기반 계층적 탐색과 도구 기반 추론을 활용한 피부과 진단 에이전트
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

# 경로 설정 - eval 폴더의 모듈을 import하기 위해
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "eval"))

from ontology_utils import OntologyTree


class DiagnosisStep(Enum):
    """진단 단계"""
    INITIAL_ASSESSMENT = "initial_assessment"
    CATEGORY_CLASSIFICATION = "category_classification"
    SUBCATEGORY_CLASSIFICATION = "subcategory_classification"
    DIFFERENTIAL_DIAGNOSIS = "differential_diagnosis"
    FINAL_DIAGNOSIS = "final_diagnosis"


# 신뢰도 임계값 상수
MIN_CATEGORY_CONFIDENCE = 0.4
MIN_SUBCATEGORY_CONFIDENCE = 0.3
MIN_DIFFERENTIAL_CONFIDENCE = 0.3
FALLBACK_CONFIDENCE_FLAG = 0.25


@dataclass
class ObservationResult:
    """관찰 결과"""
    morphology: List[str] = field(default_factory=list)  # 형태학적 특징
    color: List[str] = field(default_factory=list)       # 색상
    distribution: List[str] = field(default_factory=list) # 분포 패턴
    location: str = ""                                    # 신체 위치
    surface: List[str] = field(default_factory=list)     # 표면 특징
    symptoms: List[str] = field(default_factory=list)    # 증상
    raw_description: str = ""                            # 원본 설명


@dataclass
class DiagnosisState:
    """진단 상태 추적"""
    current_step: DiagnosisStep = DiagnosisStep.INITIAL_ASSESSMENT
    current_path: List[str] = field(default_factory=list)  # 현재 온톨로지 경로
    candidates: List[str] = field(default_factory=list)    # 후보 질환들
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    observations: Optional[ObservationResult] = None
    reasoning_history: List[Dict] = field(default_factory=list)
    final_diagnosis: List[str] = field(default_factory=list)

    # 오류 추적 필드
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    has_fallback: bool = False
    vlm_failures: int = 0
    low_confidence_steps: List[str] = field(default_factory=list)

    # Backtracking 필드
    explored_paths: set = field(default_factory=set)  # Set[Tuple[str, ...]]
    backtrack_count: int = 0
    max_backtracks: int = 3
    backtrack_history: List[Dict[str, Any]] = field(default_factory=list)


class BaseTool(ABC):
    """도구 기본 클래스"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        pass


class OntologyNavigator(BaseTool):
    """온톨로지 탐색 도구"""
    
    def __init__(self, tree: OntologyTree):
        self.tree = tree
    
    @property
    def name(self) -> str:
        return "ontology_navigator"
    
    @property
    def description(self) -> str:
        return "Navigate the disease ontology tree to find relevant categories and diseases"
    
    def execute(self, action: str, node: str = "root") -> Dict:
        """
        Actions:
        - get_children: 자식 노드들 반환
        - get_path: 루트까지 경로 반환
        - get_siblings: 형제 노드들 반환
        - get_descendants: 모든 자손 반환
        - validate: 노드 유효성 검증
        """
        if action == "get_children":
            children = self.tree.get_children(node)
            return {"node": node, "children": children, "count": len(children)}
        
        elif action == "get_path":
            path = self.tree.get_path_to_root(node)
            return {"node": node, "path": path, "depth": len(path) - 1}
        
        elif action == "get_siblings":
            siblings = self.tree.get_siblings(node)
            return {"node": node, "siblings": siblings}
        
        elif action == "get_descendants":
            descendants = list(self.tree.get_all_descendants(node))
            return {"node": node, "descendants": descendants[:50], "total": len(descendants)}
        
        elif action == "validate":
            is_valid = self.tree.is_valid_node(node)
            canonical = self.tree.get_canonical_name(node) if is_valid else None
            return {"node": node, "valid": is_valid, "canonical_name": canonical}
        
        else:
            return {"error": f"Unknown action: {action}"}


class DifferentialDiagnosisTool(BaseTool):
    """VLM 기반 동적 감별 진단 도구"""

    def __init__(self, tree: OntologyTree, vlm_model=None, system_instruction: str = ""):
        self.tree = tree
        self.vlm = vlm_model
        self.system_instruction = system_instruction.strip()
    
    @property
    def name(self) -> str:
        return "differential_diagnosis"

    @property
    def description(self) -> str:
        return "Compare clinical features with candidate diseases using VLM-based dynamic comparison"

    def execute(
        self,
        candidates: List[str],
        observations: ObservationResult,
        image_path: str = None
    ) -> Dict[str, float]:
        """
        VLM을 사용하여 후보 질환들과 관찰 결과를 비교

        Args:
            candidates: 후보 질환 목록
            observations: 관찰 결과
            image_path: 이미지 경로 (VLM 사용 시 필요)

        Returns:
            {disease: score} 형태의 딕셔너리
        """
        if self.vlm is None or image_path is None:
            # VLM 없으면 균등 점수 (모든 후보에 동일 기회)
            return {candidate: 0.5 for candidate in candidates}

        # VLM으로 배치 비교
        scores = self._compare_with_vlm_batch(candidates, observations, image_path)

        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    def _compare_with_vlm_batch(
        self,
        candidates: List[str],
        observations: ObservationResult,
        image_path: str
    ) -> Dict[str, float]:
        """
        한 번의 VLM 호출로 모든 후보 질환 비교

        비용 효율적이고 상대적 비교 가능
        """
        if not candidates:
            return {}

        # 후보 목록 포맷팅
        candidates_list = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)])

        # 관찰 결과 포맷팅
        obs_text = f"""
Morphology: {', '.join(observations.morphology) if observations.morphology else 'not specified'}
Color: {', '.join(observations.color) if observations.color else 'not specified'}
Distribution: {', '.join(observations.distribution) if observations.distribution else 'not specified'}
Surface: {', '.join(observations.surface) if observations.surface else 'not specified'}
Location: {observations.location if observations.location else 'not specified'}
"""

        # Focused instruction (토큰 최적화 - full system instruction 제외)
        focused_instruction = (
            "You are a dermatology expert. Compare this skin lesion with the candidate diagnoses listed below. "
            "Only consider the diseases in the provided list."
        )
        prompt = f"""{focused_instruction}

Compare this skin lesion with the following candidate diagnoses and rate each one.

Candidate Diagnoses:
{candidates_list}

Observed Clinical Features:
{obs_text}

For EACH candidate diagnosis, evaluate:
1. How well do the observed features match the typical presentation of this disease?
2. What features support this diagnosis?
3. What features contradict this diagnosis?
4. Overall likelihood score (0-10)

Respond in JSON format:
{{
    "comparisons": [
        {{
            "disease": "exact disease name from the list",
            "likelihood_score": 0-10,
            "supporting_features": ["feature1", "feature2"],
            "contradicting_features": ["feature1", "feature2"],
            "brief_reasoning": "one sentence explanation"
        }},
        ... (one entry for each candidate)
    ]
}}

IMPORTANT: Include ALL {len(candidates)} candidates in your response. Provide ONLY the JSON output."""

        try:
            response = self.vlm.chat_img(prompt, [image_path], max_tokens=2048)
            # JSON 파싱
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                parsed = json.loads(json_match.group())
                comparisons = parsed.get("comparisons", [])

                scores = {}
                for comp in comparisons:
                    if not isinstance(comp, dict):
                        continue
                    disease = comp.get("disease", "").strip()
                    if not disease:  # 빈 disease 이름 무시
                        continue
                    likelihood = comp.get("likelihood_score", 5)
                    # 0-10 스케일을 0-1로 변환, 타입 체크
                    try:
                        scores[disease] = float(likelihood) / 10.0
                    except (TypeError, ValueError):
                        scores[disease] = 0.5

                # 누락된 후보들에게 중립 점수
                for candidate in candidates:
                    if candidate not in scores:
                        scores[candidate] = 0.5

                return scores
            else:
                # 파싱 실패 시 균등 점수
                return {candidate: 0.5 for candidate in candidates}

        except Exception:
            # VLM 호출 실패 시 균등 점수
            return {candidate: 0.5 for candidate in candidates}


class DermatologyAgent:
    """피부과 진단 에이전트"""
    
    def __init__(
        self,
        ontology_path: Optional[str] = None,
        vlm_model: Any = None,  # Vision-Language Model (GPT4o, QwenVL, etc.)
        verbose: bool = True
    ):
        self.tree = OntologyTree(ontology_path)  # None이면 자동 탐색
        self.vlm = vlm_model
        self.verbose = verbose
        
        # 도구 초기화
        self.leaf_diseases = sorted([n for n in self.tree.valid_nodes if not self.tree.get_children(n)])
        self.system_instruction = self._build_system_instruction()
        self.tools = {
            "navigator": OntologyNavigator(self.tree),
            "differential": DifferentialDiagnosisTool(self.tree, self.vlm, self.system_instruction),
        }
        
        # 루트 카테고리 (Level 1)
        self.root_categories = self.tree.ontology.get("root", [])
        
        # 프롬프트 템플릿
        self._load_prompts()

    def _build_system_instruction(self) -> str:
        """기본 시스템 프롬프트 구성 (disease list 제외 - 토큰 최적화)"""
        return (
            "You are a board-certified dermatology expert. You are provided with a skin image and may receive a related question. "
            "Analyze the image carefully and provide a detailed, professional assessment. Focus on identifying skin conditions, lesions, or abnormalities. "
            "When identifying skin conditions, choose the disease_label from the provided list of valid diagnoses in the prompt. "
            "Call out any emergent or high-risk findings explicitly (e.g., melanoma, necrotizing infection, Stevens-Johnson syndrome). "
            "If uncertain, share the top differentials rather than inventing new labels. "
            "Always select from the specific options provided in each prompt."
        )
    
    def _load_prompts(self):
        """프롬프트 템플릿 로드"""
        self.prompts = {
            "initial_assessment": """Analyze this dermatological image and describe what you observe.

IMPORTANT: If there is NO visible skin lesion or disease in the image, respond with:
{
    "morphology": ["no visible lesion"],
    "color": ["not observed"],
    "distribution": ["not observed"],
    "surface": ["not observed"],
    "border": ["not observed"],
    "location": "not observed",
    "additional_notes": "No definitive skin lesion visible in the image"
}

Focus on PRIMARY LESION MORPHOLOGY - be VERY specific:
1. Morphology (primary lesion type):
   - Flat lesions: macule, patch
   - Raised solid lesions: papule, plaque, nodule, wheal, tumor
   - Fluid-filled lesions: vesicle, bulla, pustule
   - Loss of tissue: erosion, ulcer, fissure, excoriation
   - Other: cyst, comedo, abscess
2. Color: red, pink, brown, black, white, yellow, purple, blue, skin-colored, etc.
3. Distribution pattern: localized, generalized, symmetric, asymmetric, clustered, linear, dermatomal, follicular, etc.
4. Surface features: smooth, scaly, crusted, rough, verrucous, ulcerated, eroded, lichenified, etc.
5. Border: well-defined, ill-defined, regular, irregular, raised, rolled
6. Body location: face, trunk, extremities, hands, feet, oral cavity, genitals, scalp, etc.

Important: Do NOT leave any list empty. If a feature is not visible, include "not observed" in that list.

Provide your observations in JSON format:
{
    "morphology": ["list of PRIMARY lesion types - be very specific"],
    "color": ["list of colors observed"],
    "distribution": ["distribution patterns"],
    "surface": ["surface features"],
    "border": ["border characteristics"],
    "location": "body location",
    "additional_notes": "any other relevant observations"
}

Provide ONLY the JSON output.""",

            "category_classification": """Based on the clinical features observed in this skin image, 
classify this condition into ONE of the following major categories:

Categories:
{categories}

Consider the morphology, distribution, and clinical presentation.
Respond with JSON:
{{
    "selected_category": "the most likely category",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

Provide ONLY the JSON output.""",

            "subcategory_classification": """Given that this skin condition belongs to the "{parent_category}" category,
further classify it into one of these subcategories:

Subcategories:
{subcategories}

Based on the image features:
{observations}

Respond with JSON:
{{
    "selected_subcategory": "the most likely subcategory",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

Provide ONLY the JSON output.""",

            "final_diagnosis": """Based on the hierarchical classification path:
{path}

And the clinical observations:
{observations}

Select the most likely specific diagnosis from these candidates:
{candidates}

IMPORTANT: If the clinical observations indicate "no visible lesion" or if you cannot confidently identify a specific disease, set primary_diagnosis to "no definitive diagnosis".

Respond with JSON:
{{
    "primary_diagnosis": "most likely diagnosis (or 'no definitive diagnosis' if lesion is not clear)",
    "confidence": 0.0-1.0,
    "differential_diagnoses": ["other possible diagnoses in order of likelihood"],
    "reasoning": "clinical reasoning for your diagnosis"
}}

Provide ONLY the JSON output."""
        }
    
    def _call_vlm(self, prompt: str, image_path: str, state: DiagnosisState = None, step: str = "unknown") -> Tuple[str, bool]:
        """
        VLM 모델 호출

        Returns:
            Tuple[str, bool]: (response, success_flag)
        """
        if self.vlm is None:
            if state:
                state.vlm_failures += 1
                self._record_error(state, step, "NO_VLM", "VLM model is None")
            return "{}", False

        full_prompt = f"{self.system_instruction}\n\n{prompt}"

        try:
            response = self.vlm.chat_img(full_prompt, [image_path], max_tokens=1024)
            return response, True
        except Exception as e:
            if state:
                state.vlm_failures += 1
                self._record_error(state, step, "VLM_EXCEPTION", str(e), {"exception_type": type(e).__name__})
            if self.verbose:
                print(f"VLM Error: {e}")
            return "{}", False
    
    def _parse_json_response(self, response: str) -> Dict:
        """JSON 응답 파싱"""
        try:
            # JSON 부분 추출
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
        return {}
    
    def _log(self, message: str):
        """로깅"""
        if self.verbose:
            print(f"[Agent] {message}")

    def _record_error(self, state: DiagnosisState, step: str, error_type: str, message: str, details: Dict = None):
        """진단 상태에 오류 기록"""
        error_entry = {
            "step": step,
            "type": error_type,
            "message": message,
            "details": details or {}
        }
        state.errors.append(error_entry)
        self._log(f"ERROR [{step}]: {message}")

    def _record_warning(self, state: DiagnosisState, step: str, message: str, details: Dict = None):
        """진단 상태에 경고 기록"""
        warning_entry = {
            "step": step,
            "message": message,
            "details": details or {}
        }
        state.warnings.append(warning_entry)
        self._log(f"WARNING [{step}]: {message}")

    def _check_confidence(self, state: DiagnosisState, step: str, confidence: float, threshold: float) -> bool:
        """신뢰도가 임계값을 만족하는지 확인하고, 그렇지 않으면 경고 기록"""
        if confidence < threshold:
            state.low_confidence_steps.append(step)
            self._record_warning(
                state,
                step,
                f"Low confidence: {confidence:.2f} < {threshold:.2f}",
                {"confidence": confidence, "threshold": threshold}
            )
            return False
        return True

    # ============ Backtracking 메서드 ============

    def _should_backtrack(self, state: DiagnosisState) -> bool:
        """
        감별진단 품질 기반으로 backtracking 필요 여부 판단

        Criteria:
        - 모든 후보의 신뢰도 < MIN_DIFFERENTIAL_CONFIDENCE
        - backtrack_count < max_backtracks
        """
        if state.backtrack_count >= state.max_backtracks:
            self._log(f"  Max backtracks ({state.max_backtracks}) reached, no backtracking")
            return False

        # Check if we have any good differential candidates
        if not state.confidence_scores:
            return True

        # Get confidence scores for current candidates
        candidate_scores = [
            state.confidence_scores.get(c, 0.0)
            for c in state.candidates
        ]

        if not candidate_scores:
            return True

        # Safe max/avg calculation for non-empty list
        max_candidate_score = max(candidate_scores) if candidate_scores else 0.0
        avg_candidate_score = sum(candidate_scores) / len(candidate_scores) if candidate_scores else 0.0

        # Backtrack if all scores are too low
        should_backtrack = (
            max_candidate_score < MIN_DIFFERENTIAL_CONFIDENCE or
            (max_candidate_score < 0.5 and avg_candidate_score < 0.3)
        )

        if should_backtrack:
            self._log(f"  Backtracking triggered: max_score={max_candidate_score:.2f}, avg_score={avg_candidate_score:.2f}")

        return should_backtrack

    def _backtrack_to_subcategory(self, state: DiagnosisState) -> bool:
        """
        하위 카테고리 레벨로 돌아가서 대체 경로 시도

        Returns:
            True if backtracking succeeded, False if no alternatives available
        """
        if len(state.current_path) < 2:
            self._log("  Cannot backtrack: path too short")
            return False

        # Record current path as explored
        state.explored_paths.add(tuple(state.current_path))

        # Get parent category (one level up)
        state.current_path.pop()  # Remove current subcategory
        parent_category = state.current_path[-1]

        # Get all children of parent
        all_children = self.tree.get_children(parent_category)

        # Filter out already explored children
        unexplored_children = []
        for child in all_children:
            test_path = tuple(state.current_path + [child])
            if test_path not in state.explored_paths:
                unexplored_children.append(child)

        if not unexplored_children:
            self._log(f"  No unexplored children for {parent_category}")

            # Try backtracking to category level if possible
            if len(state.current_path) >= 2:
                return self._backtrack_to_category(state)
            return False

        # Select next best alternative
        # Sort by previous confidence if available
        unexplored_children.sort(
            key=lambda x: state.confidence_scores.get(x, 0.0),
            reverse=True
        )

        next_subcategory = unexplored_children[0]
        state.current_path.append(next_subcategory)
        state.backtrack_count += 1

        # Record backtracking event
        backtrack_record = {
            "backtrack_level": "subcategory",
            "from_path": list(list(state.explored_paths)[-1]) if state.explored_paths else [],
            "to_path": state.current_path.copy(),
            "parent": parent_category,
            "new_subcategory": next_subcategory,
            "alternatives_remaining": len(unexplored_children) - 1
        }
        state.backtrack_history.append(backtrack_record)

        self._log(f"  Backtracked to: {' → '.join(state.current_path)}")
        self._log(f"  Alternatives remaining: {len(unexplored_children) - 1}")

        return True

    def _backtrack_to_category(self, state: DiagnosisState) -> bool:
        """
        카테고리 레벨로 돌아가서 대체 루트 카테고리 시도

        Returns:
            True if backtracking succeeded, False if no alternatives available
        """
        if len(state.current_path) < 1:
            self._log("  Cannot backtrack to category: no path")
            return False

        # Record current path as explored
        state.explored_paths.add(tuple(state.current_path))

        # Reset to before category selection
        state.current_path.clear()

        # Get unexplored root categories
        unexplored_categories = []
        for category in self.root_categories:
            test_path = tuple([category])
            if test_path not in state.explored_paths:
                unexplored_categories.append(category)

        if not unexplored_categories:
            self._log("  No unexplored root categories available")
            return False

        # Select next category (sorted by previous confidence)
        unexplored_categories.sort(
            key=lambda x: state.confidence_scores.get(x, 0.0),
            reverse=True
        )

        next_category = unexplored_categories[0]
        state.current_path.append(next_category)
        state.backtrack_count += 1

        # Record backtracking event
        backtrack_record = {
            "backtrack_level": "category",
            "from_path": list(list(state.explored_paths)[-1]) if state.explored_paths else [],
            "to_path": state.current_path.copy(),
            "new_category": next_category,
            "alternatives_remaining": len(unexplored_categories) - 1
        }
        state.backtrack_history.append(backtrack_record)

        self._log(f"  Backtracked to category: {next_category}")
        self._log(f"  Category alternatives remaining: {len(unexplored_categories) - 1}")

        return True

    # ============ 진단 단계별 메서드 ============
    
    def step_initial_assessment(
        self,
        image_path: str,
        state: DiagnosisState
    ) -> DiagnosisState:
        """Step 1: 초기 평가 - 이미지에서 특징 추출"""
        self._log("Step 1: Initial Assessment")

        prompt = self.prompts["initial_assessment"]
        response, success = self._call_vlm(prompt, image_path, state, "initial_assessment")
        parsed = self._parse_json_response(response)

        def _normalize_list(values):
            return values if values else ["not observed"]
        
        observations = ObservationResult(
            morphology=_normalize_list(parsed.get("morphology", [])),
            color=_normalize_list(parsed.get("color", [])),
            distribution=_normalize_list(parsed.get("distribution", [])),
            surface=_normalize_list(parsed.get("surface", [])),
            location=parsed.get("location", ""),
            raw_description=response
        )

        state.observations = observations
        state.reasoning_history.append({
            "step": "initial_assessment",
            "observations": parsed,
            "raw_response": response[:500]
        })

        self._log(f"  Observed morphology: {observations.morphology}")
        self._log(f"  Observed colors: {observations.color}")
        self._log(f"  Location: {observations.location}")

        # Check if no visible lesion was detected
        if "no visible lesion" in [m.lower() for m in observations.morphology]:
            self._log("  No visible lesion detected - skipping to final diagnosis")
            state.final_diagnosis = ["no definitive diagnosis"]
            state.confidence_scores["no definitive diagnosis"] = 0.0
            state.current_step = DiagnosisStep.FINAL_DIAGNOSIS
            return state

        state.current_step = DiagnosisStep.CATEGORY_CLASSIFICATION
        return state
    
    def step_category_classification(
        self,
        image_path: str,
        state: DiagnosisState
    ) -> DiagnosisState:
        """Step 2: 대분류 - 루트 카테고리 선택"""
        self._log("Step 2: Category Classification (Level 1)")

        categories_desc = "\n".join([f"- {cat}" for cat in self.root_categories])
        prompt = self.prompts["category_classification"].format(categories=categories_desc)

        response, success = self._call_vlm(prompt, image_path, state, "category_classification")

        if not success:
            self._record_warning(state, "category_classification", "VLM call failed, using fallback")
            state.has_fallback = True
            state.current_path.append("inflammatory")
            state.confidence_scores["inflammatory"] = FALLBACK_CONFIDENCE_FLAG
            state.current_step = DiagnosisStep.SUBCATEGORY_CLASSIFICATION
            return state

        parsed = self._parse_json_response(response)

        if not parsed:
            self._record_warning(state, "category_classification", "Failed to parse VLM response, using fallback")
            state.has_fallback = True
            state.current_path.append("inflammatory")
            state.confidence_scores["inflammatory"] = FALLBACK_CONFIDENCE_FLAG
            state.current_step = DiagnosisStep.SUBCATEGORY_CLASSIFICATION
            return state

        selected = parsed.get("selected_category", "")
        confidence = parsed.get("confidence", 0.5)

        # Check confidence threshold
        self._check_confidence(state, "category_classification", confidence, MIN_CATEGORY_CONFIDENCE)

        # 유효한 카테고리인지 확인
        canonical = self.tree.get_canonical_name(selected)
        if canonical and canonical in self.root_categories:
            state.current_path.append(canonical)
            state.confidence_scores[canonical] = confidence
        else:
            # Invalid category selected
            self._record_warning(
                state,
                "category_classification",
                f"Invalid category '{selected}', using fallback to 'inflammatory'",
                {"selected": selected, "valid_categories": self.root_categories}
            )
            state.has_fallback = True
            state.current_path.append("inflammatory")
            state.confidence_scores["inflammatory"] = FALLBACK_CONFIDENCE_FLAG

        state.current_step = DiagnosisStep.SUBCATEGORY_CLASSIFICATION
        state.reasoning_history.append({
            "step": "category_classification",
            "selected": state.current_path[-1],
            "confidence": state.confidence_scores.get(state.current_path[-1], confidence),
            "reasoning": parsed.get("reasoning", ""),
            "is_fallback": state.has_fallback
        })

        self._log(f"  Selected category: {state.current_path[-1]} (conf: {state.confidence_scores.get(state.current_path[-1], confidence):.2f})")

        return state
    
    def step_subcategory_classification(
        self, 
        image_path: str, 
        state: DiagnosisState,
        max_depth: int = 3
    ) -> DiagnosisState:
        """Step 3: 중분류/소분류 - 하위 카테고리 탐색"""
        current_depth = len(state.current_path)
        
        while current_depth < max_depth:
            current_node = state.current_path[-1]
            children = self.tree.get_children(current_node)
            
            if not children:
                break  # 리프 노드 도달
            
            self._log(f"Step 3.{current_depth}: Subcategory Classification")
            self._log(f"  Current: {current_node}")
            self._log(f"  Options: {children[:10]}...")
            
            # 자식이 많으면 VLM에게 선택 요청
            if len(children) > 1:
                subcategories_desc = "\n".join([f"- {child}" for child in children])
                obs_desc = json.dumps({
                    "morphology": state.observations.morphology if state.observations else [],
                    "color": state.observations.color if state.observations else [],
                    "location": state.observations.location if state.observations else ""
                }, indent=2)

                prompt = self.prompts["subcategory_classification"].format(
                    parent_category=current_node,
                    subcategories=subcategories_desc,
                    observations=obs_desc
                )

                response, success = self._call_vlm(prompt, image_path, state, f"subcategory_level_{current_depth}")
                parsed = self._parse_json_response(response)

                selected = parsed.get("selected_subcategory", "")
                confidence = parsed.get("confidence", 0.5)

                # Check confidence threshold
                self._check_confidence(state, f"subcategory_level_{current_depth}", confidence, MIN_SUBCATEGORY_CONFIDENCE)

                # 유효성 확인
                canonical = self.tree.get_canonical_name(selected)
                if canonical and canonical in children:
                    state.current_path.append(canonical)
                    state.confidence_scores[canonical] = confidence
                else:
                    # Invalid subcategory selected
                    self._record_warning(
                        state,
                        f"subcategory_level_{current_depth}",
                        f"Invalid subcategory '{selected}', using fallback to first child: {children[0]}",
                        {"selected": selected, "valid_children": children, "fallback": children[0]}
                    )
                    state.has_fallback = True
                    state.current_path.append(children[0])
                    state.confidence_scores[children[0]] = FALLBACK_CONFIDENCE_FLAG

                state.reasoning_history.append({
                    "step": f"subcategory_level_{current_depth}",
                    "parent": current_node,
                    "selected": state.current_path[-1],
                    "confidence": state.confidence_scores.get(state.current_path[-1], confidence),
                    "is_fallback": state.has_fallback
                })

                self._log(f"  Selected: {state.current_path[-1]} (conf: {state.confidence_scores.get(state.current_path[-1], confidence):.2f})")
            else:
                # 자식이 하나면 그대로 선택
                if children:
                    state.current_path.append(children[0])
                    state.confidence_scores[children[0]] = 0.9
                else:
                    self._log(f"  Warning: Node '{current_node}' has no children")
                    break
            
            current_depth = len(state.current_path)
        
        state.current_step = DiagnosisStep.DIFFERENTIAL_DIAGNOSIS
        return state
    
    def step_differential_diagnosis(
        self,
        image_path: str,
        state: DiagnosisState
    ) -> DiagnosisState:
        """Step 4: 감별 진단 - 후보 질환들 비교"""
        self._log("Step 4: Differential Diagnosis")

        if not state.current_path:
            self._record_error(state, "differential_diagnosis", "EMPTY_PATH", "Current path is empty")
            self._log("  Error: Current path is empty, cannot perform differential diagnosis")
            state.current_step = DiagnosisStep.FINAL_DIAGNOSIS
            return state

        current_node = state.current_path[-1]
        
        # 현재 노드의 자손들을 후보로
        descendants = self.tree.get_all_descendants(current_node)
        
        # 리프 노드들만 필터링 (실제 질환)
        leaf_candidates = [d for d in descendants if not self.tree.get_children(d)]
        
        # 현재 노드도 후보에 포함 (리프인 경우)
        if not self.tree.get_children(current_node):
            leaf_candidates.append(current_node)
        
        if not leaf_candidates:
            leaf_candidates = [current_node]
        
        state.candidates = leaf_candidates[:20]  # 상위 20개만
        
        self._log(f"  Candidates: {len(state.candidates)} diseases")
        
        # 감별 진단 도구 사용 (VLM 기반)
        if state.observations:
            diff_scores = self.tools["differential"].execute(
                state.candidates,
                state.observations,
                image_path  # VLM이 이미지를 직접 보고 비교
            )
            state.confidence_scores.update(diff_scores)

            # Check if all scores are too low
            top_scores = sorted(diff_scores.values(), reverse=True)[:3]
            if top_scores:
                self._log(f"  Top 3 scores: {[f'{s:.2f}' for s in top_scores]}")

                # Record low confidence in differential
                if top_scores[0] < MIN_DIFFERENTIAL_CONFIDENCE:
                    self._record_warning(
                        state,
                        "differential_diagnosis",
                        f"All differential candidates have low confidence (max: {top_scores[0]:.2f})",
                        {"top_scores": top_scores, "candidate_count": len(state.candidates)}
                    )

        state.current_step = DiagnosisStep.FINAL_DIAGNOSIS
        return state
    
    def step_final_diagnosis(
        self, 
        image_path: str, 
        state: DiagnosisState
    ) -> DiagnosisState:
        """Step 5: 최종 진단"""
        self._log("Step 5: Final Diagnosis")
        
        # VLM에게 최종 결정 요청
        path_str = " → ".join(state.current_path)
        candidates_str = "\n".join([f"- {c}" for c in state.candidates[:15]])
        obs_str = json.dumps({
            "morphology": state.observations.morphology if state.observations else [],
            "color": state.observations.color if state.observations else [],
            "distribution": state.observations.distribution if state.observations else [],
            "location": state.observations.location if state.observations else ""
        }, indent=2)
        
        prompt = self.prompts["final_diagnosis"].format(
            path=path_str,
            observations=obs_str,
            candidates=candidates_str
        )

        response, success = self._call_vlm(prompt, image_path, state, "final_diagnosis")
        parsed = self._parse_json_response(response)
        
        primary = parsed.get("primary_diagnosis", "")
        differentials = parsed.get("differential_diagnoses", [])
        confidence = parsed.get("confidence", 0.5)
        
        # 유효성 확인 및 최종 진단 설정
        canonical_primary = self.tree.get_canonical_name(primary)
        if canonical_primary:
            state.final_diagnosis.append(canonical_primary)
            state.confidence_scores[canonical_primary] = confidence
        
        for diff in differentials[:3]:
            canonical = self.tree.get_canonical_name(diff)
            if canonical and canonical not in state.final_diagnosis:
                state.final_diagnosis.append(canonical)
        
        # 후보 점수 기반 백업
        if not state.final_diagnosis:
            sorted_candidates = sorted(
                state.confidence_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            for candidate, score in sorted_candidates[:3]:
                if self.tree.is_valid_node(candidate):
                    canonical = self.tree.get_canonical_name(candidate)
                    if canonical and canonical not in state.final_diagnosis:
                        state.final_diagnosis.append(canonical)
        
        state.reasoning_history.append({
            "step": "final_diagnosis",
            "primary": primary,
            "differentials": differentials,
            "confidence": confidence,
            "reasoning": parsed.get("reasoning", "")
        })
        
        self._log(f"  Final diagnosis: {state.final_diagnosis}")
        self._log(f"  Confidence: {confidence:.2f}")
        
        return state
    
    # ============ 메인 진단 메서드 ============
    
    def diagnose(
        self, 
        image_path: str,
        max_depth: int = 4
    ) -> Dict[str, Any]:
        """
        전체 진단 파이프라인 실행
        
        Args:
            image_path: 피부 이미지 경로
            max_depth: 온톨로지 탐색 최대 깊이
        
        Returns:
            진단 결과 딕셔너리
        """
        self._log(f"\n{'='*50}")
        self._log(f"Starting diagnosis for: {image_path}")
        self._log(f"{'='*50}")
        
        state = DiagnosisState()

        # Step 1: 초기 평가
        state = self.step_initial_assessment(image_path, state)

        # Check if no visible lesion was detected - skip remaining steps
        if state.current_step == DiagnosisStep.FINAL_DIAGNOSIS and "no definitive diagnosis" in state.final_diagnosis:
            self._log("No visible lesion detected - diagnosis complete")
        else:
            # Step 2: 대분류
            state = self.step_category_classification(image_path, state)

            # Step 3: 중분류/소분류
            state = self.step_subcategory_classification(image_path, state, max_depth)

            # Step 4-5: 감별 진단 및 최종 진단 (with backtracking)
            backtrack_attempts = 0
            while backtrack_attempts <= state.max_backtracks:
                # Step 4: 감별 진단
                state = self.step_differential_diagnosis(image_path, state)

                # Check if we should backtrack
                if self._should_backtrack(state):
                    self._log(f"\n{'='*50}")
                    self._log(f"Backtracking attempt {backtrack_attempts + 1}/{state.max_backtracks}")
                    self._log(f"{'='*50}")

                    # Try to backtrack to subcategory level
                    if self._backtrack_to_subcategory(state):
                        # Re-run subcategory classification from new position
                        state.current_step = DiagnosisStep.SUBCATEGORY_CLASSIFICATION
                        state = self.step_subcategory_classification(image_path, state, max_depth)
                        backtrack_attempts += 1
                        continue
                    else:
                        # No alternatives available, proceed with current best
                        self._log("No backtracking alternatives available, proceeding with current path")
                        break
                else:
                    # Differential diagnosis quality is acceptable
                    break

            # Step 5: 최종 진단
            state = self.step_final_diagnosis(image_path, state)
        
        # 결과 정리
        result = {
            "image_path": image_path,
            "final_diagnosis": state.final_diagnosis,
            "diagnosis_path": state.current_path,
            "confidence_scores": state.confidence_scores,
            "observations": {
                "morphology": state.observations.morphology if state.observations else [],
                "color": state.observations.color if state.observations else [],
                "distribution": state.observations.distribution if state.observations else [],
                "location": state.observations.location if state.observations else "",
            },
            "reasoning_history": state.reasoning_history,
            "candidates_considered": state.candidates,

            # 새로운 진단 품질 정보
            "errors": state.errors,
            "warnings": state.warnings,
            "has_fallback": state.has_fallback,
            "vlm_failures": state.vlm_failures,
            "low_confidence_steps": state.low_confidence_steps,
            "backtrack_count": state.backtrack_count,
            "backtrack_history": state.backtrack_history,
            "explored_paths": [list(p) for p in state.explored_paths],
        }
        
        self._log(f"\n{'='*50}")
        self._log(f"Diagnosis complete")
        self._log(f"Result: {state.final_diagnosis}")
        self._log(f"Path: {' → '.join(state.current_path)}")
        self._log(f"{'='*50}\n")
        
        return result
    
    def diagnose_batch(
        self, 
        image_paths: List[str],
        max_depth: int = 4
    ) -> List[Dict[str, Any]]:
        """배치 진단"""
        results = []
        for path in image_paths:
            result = self.diagnose(path, max_depth)
            results.append(result)
        return results


def demo():
    """데모 (VLM 없이 구조 테스트)"""
    print("=== Dermatology Agent Demo (without VLM) ===\n")

    try:
        # VLM 없이 에이전트 생성 (자동 경로)
        agent = DermatologyAgent(ontology_path=None, vlm_model=None, verbose=True)
        print(f"✓ Ontology loaded from: {agent.tree.ontology_path}\n")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    print("\n[Ontology Navigator Tool Demo]")
    nav = agent.tools["navigator"]
    
    print("\nRoot categories:")
    result = nav.execute("get_children", "root")
    for child in result["children"]:
        print(f"  - {child}")
    
    print("\nInflammatory subcategories:")
    result = nav.execute("get_children", "inflammatory")
    for child in result["children"]:
        print(f"  - {child}")
    
    print("\nFungal diseases:")
    result = nav.execute("get_children", "fungal")
    for child in result["children"]:
        print(f"  - {child}")
    
    print("\nPath for 'Tinea corporis':")
    result = nav.execute("get_path", "Tinea corporis")
    print(f"  {' → '.join(result['path'])}")
    
    print("\n[Differential Diagnosis Tool Demo]")
    diff_tool = agent.tools["differential"]
    
    # 가상의 관찰 결과
    obs = ObservationResult(
        morphology=["annular", "scaly", "plaque"],
        color=["red", "erythematous"],
        distribution=["localized"],
        location="trunk",
        raw_description="circular red scaly patch with raised border"
    )
    
    candidates = ["Tinea corporis", "Psoriasis", "Eczema", "Cellulitis"]
    scores = diff_tool.execute(candidates, obs)
    
    print(f"\nObservations: morphology={obs.morphology}, color={obs.color}")
    print("Differential diagnosis scores:")
    for disease, score in scores.items():
        print(f"  {disease}: {score:.3f}")


if __name__ == "__main__":
    demo()
