import csv
import random
import shutil
from pathlib import Path

# 경로 설정
base_dir = Path('/home/heodnjswns/burnskin/dataset/Derm1M')
csv_path = base_dir / 'Derm1M_v2_pretrain.csv'
output_dir = base_dir / 'random_samples_100'
output_csv_path = output_dir / 'sampled_data.csv'
output_images_dir = output_dir / 'images'

# 출력 디렉토리 생성
output_dir.mkdir(exist_ok=True)
output_images_dir.mkdir(exist_ok=True)

print(f"CSV 파일 읽는 중: {csv_path}")

# CSV 파일 읽기 (UTF-8 BOM 처리)
with open(csv_path, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    all_rows = list(reader)

print(f"전체 데이터 개수: {len(all_rows)}")

# 100개 랜덤 샘플링 (random seed 고정)
random.seed(42)
sampled_rows = random.sample(all_rows, 100)
print(f"샘플링된 데이터 개수: {len(sampled_rows)}")

# 샘플링된 데이터를 CSV로 저장
with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
    if sampled_rows:
        fieldnames = sampled_rows[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampled_rows)

print(f"샘플링된 CSV 저장 완료: {output_csv_path}")

# 이미지 파일 복사
print("\n이미지 파일 복사 중...")
copied_count = 0
missing_count = 0
missing_files = []

for row in sampled_rows:
    filename = row['filename']
    source_path = base_dir / filename

    # 파일명만 추출 (경로 제외)
    image_filename = Path(filename).name
    dest_path = output_images_dir / image_filename

    if source_path.exists():
        shutil.copy2(source_path, dest_path)
        copied_count += 1
        if copied_count % 10 == 0:
            print(f"  복사 진행: {copied_count}/100")
    else:
        missing_count += 1
        missing_files.append(str(source_path))
        print(f"  파일 없음: {source_path}")

print(f"\n완료!")
print(f"복사된 이미지: {copied_count}개")
print(f"누락된 이미지: {missing_count}개")

if missing_files:
    print(f"\n누락된 파일 목록:")
    for f in missing_files[:10]:  # 최대 10개만 출력
        print(f"  - {f}")
    if len(missing_files) > 10:
        print(f"  ... 외 {len(missing_files) - 10}개")

print(f"\n결과 저장 위치:")
print(f"  CSV: {output_csv_path}")
print(f"  이미지: {output_images_dir}")
