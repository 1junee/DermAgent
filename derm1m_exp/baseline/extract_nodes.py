import json
from pathlib import Path

def extract_nodes_excluding_root_and_children(ontology_path, output_path=None):
    """
    온톨로지에서 루트와 루트의 직계 자식 노드들을 제외한 나머지 노드들을 추출합니다.

    Args:
        ontology_path: ontology.json 파일 경로
        output_path: 추출된 노드를 저장할 파일 경로 (옵션)

    Returns:
        dict: 추출된 노드들 (키: 노드명, 값: 자식 노드 리스트)
    """
    # ontology 로드
    with open(ontology_path, 'r', encoding='utf-8') as f:
        ontology = json.load(f)

    # 루트의 자식 노드들 가져오기
    root_children = ontology.get('root', [])

    # 제외할 노드 목록: 'root' + 루트의 자식들
    excluded_nodes = {'root'} | set(root_children)

    # 루트와 루트의 자식들을 제외한 노드들 추출
    extracted_nodes = {
        node: children
        for node, children in ontology.items()
        if node not in excluded_nodes
    }

    # 통계 출력
    print(f"전체 노드 개수: {len(ontology)}")
    print(f"제외된 노드: {len(excluded_nodes)}개")
    print(f"  - 루트 노드: 1개")
    print(f"  - 루트의 자식: {len(root_children)}개")
    print(f"추출된 노드 개수: {len(extracted_nodes)}개")
    print()
    print("루트의 자식 노드들:")
    for child in root_children:
        print(f"  - {child}")

    # 결과 저장 (옵션)
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(extracted_nodes, f, indent=2, ensure_ascii=False)
        print(f"\n추출된 노드들이 저장되었습니다: {output_path}")

    return extracted_nodes


if __name__ == "__main__":
    # 파일 경로 설정
    ontology_path = Path(__file__).parent.parent / "dataset" / "Derm1M" / "ontology.json"
    output_path = Path(__file__).parent / "extracted_nodes.json"

    # 노드 추출
    extracted_nodes = extract_nodes_excluding_root_and_children(
        ontology_path=ontology_path,
        output_path=output_path
    )

    # 추출된 노드 이름 목록 저장
    node_names_path = Path(__file__).parent / "extracted_node_names.txt"
    with open(node_names_path, 'w', encoding='utf-8') as f:
        for node_name in sorted(extracted_nodes.keys()):
            f.write(f"{node_name}\n")

    print(f"노드 이름 목록이 저장되었습니다: {node_names_path}")
