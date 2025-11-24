
# Python 코드로 다운로드
from huggingface_hub import snapshot_download

# 전체 데이터셋 다운로드
snapshot_download(
    repo_id="redlessone/Derm1M",
    repo_type="dataset",
    local_dir="./Derm1M",
    local_dir_use_symlinks=False
)