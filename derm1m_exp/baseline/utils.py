
import base64
from PIL import Image
import json
import os
import csv
import subprocess
import re
import uuid
from pathlib import Path
from tqdm import tqdm
import cv2


def count_csv_rows(file_path):
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        row_count = sum(1 for row in reader)
    return row_count - 1


def ensure_csv_has_column(file_path: str, column_name: str) -> None:
    if not os.path.exists(file_path):
        return
    with open(file_path, mode='r', encoding='utf-8', newline='') as file:
        rows = list(csv.reader(file))
    if not rows:
        return
    header = rows[0]
    if column_name in header:
        return
    header.append(column_name)
    for row in rows[1:]:
        row.append('')
    with open(file_path, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def compress_video(input_path, output_path, crf=28):
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    command = [
        "ffmpeg", "-i", input_path, "-vcodec", "libx264", "-crf", str(crf), output_path
    ]
    subprocess.run(command, check=True)
    return output_path


def compress_image(image_path, output_path, quality=50):
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with Image.open(image_path) as img:
        img.save(output_path, format="JPEG", quality=quality)
    return output_path

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_image =  base64.b64encode(image_file.read()).decode('utf-8')
    return base64_image


def encode_video(video_path):
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def sort_files_by_number_in_name(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # 파일명에서 숫자 추출 (언더스코어 여부 관계없이)
    def extract_number(filename):
        basename = os.path.basename(filename)
        # 파일명에서 모든 숫자를 찾음
        numbers = re.findall(r'\d+', basename)
        if not numbers:
            return 0
        # 언더스코어가 있으면 마지막 숫자 사용 (예: 256_276.jpg -> 276)
        # 언더스코어가 없으면 첫 번째 숫자 사용 (예: 100.jpg -> 100)
        if '_' in basename:
            return int(numbers[-1])  # 마지막 숫자
        else:
            return int(numbers[0])   # 첫 번째 숫자

    files.sort(key=extract_number)
    stride = max(1, len(files) // 25)  # stride가 0이 되는 것을 방지
    files = files[::stride]
    return files


def videolist2imglist(video_path, num):
    base64_videos = []
    assign = []
    video_num = len(video_path)
    for i in range(video_num):
        assign.append(num / video_num)
    j = 0
    for video in video_path:
        video = cv2.VideoCapture(video)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        base64_images = []
        fps = assign[j]
        interval = int(frame_count / fps)
        for i in range(0, frame_count, interval):
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.read()
            if ret:
                _, buffer = cv2.imencode(".jpg", frame)
                base64_images.append(base64.b64encode(buffer).decode("utf-8"))
        base64_videos.append(base64_images)
        j += 1
    return base64_videos


def sample_video_frames(video_path: str, max_frames: int = 8, output_dir: str | None = None) -> list[str]:
    """Save uniformly sampled frames from a video and return their file paths."""
    if output_dir is None:
        output_dir = os.path.join("tmp_file", "internvl_frames")
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    interval = max(1, frame_count // max_frames) if frame_count else 1

    saved_paths: list[str] = []
    frame_idx = 0
    while len(saved_paths) < max_frames and (frame_count == 0 or frame_idx < frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        filename = f"{Path(video_path).stem}_{len(saved_paths)}_{uuid.uuid4().hex}.jpg"
        file_path = os.path.join(output_dir, filename)
        cv2.imwrite(file_path, frame)
        saved_paths.append(file_path)
        frame_idx += interval

    cap.release()
    return saved_paths
