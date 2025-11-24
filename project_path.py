import os
from pathlib import Path

# Get the absolute path of this file
THIS_FILE = Path(__file__).resolve()

# Project root is the parent directory of this file
PROJECT_PATH = str(THIS_FILE.parent)

# Dataset paths
DATASET_ROOT = os.path.join(PROJECT_PATH, "dataset")
DERM1M_ROOT = os.path.join(DATASET_ROOT, "Derm1M")

# Experiment paths
EXP_ROOT = os.path.join(PROJECT_PATH, "derm1m_exp")
BASELINE_ROOT = os.path.join(EXP_ROOT, "baseline")

# Common data paths
RANDOM_SAMPLES_100 = os.path.join(DERM1M_ROOT, "random_samples_100")
SAMPLED_DATA_CSV = os.path.join(RANDOM_SAMPLES_100, "sampled_data.csv")

# Output paths
OUTPUTS_ROOT = os.path.join(BASELINE_ROOT, "outputs")

if __name__ == "__main__":
    print(f"PROJECT_PATH: {PROJECT_PATH}")
    print(f"DATASET_ROOT: {DATASET_ROOT}")
    print(f"DERM1M_ROOT: {DERM1M_ROOT}")
    print(f"BASELINE_ROOT: {BASELINE_ROOT}")
    print(f"SAMPLED_DATA_CSV: {SAMPLED_DATA_CSV}")
