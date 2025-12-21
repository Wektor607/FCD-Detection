from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# DATA_DIR = PROJECT_ROOT / "meld_graph" / "data" / "input" / "data4sharing"
DATA_DIR = PROJECT_ROOT / "data" / "input" / "data4sharing"
DEFAULT_DEMOGRAPHIC_FILE = DATA_DIR / "demographics_qc_allgroups_withH27H28H101.csv"
UPLOAD_DIR = PROJECT_ROOT / "data" / "input" / "data4sharing" / "meld_combats" # Path("uploads")
RESULT_DIR = Path("results")
# T1_FILE = "./meld_graph/data/input/data4sharing/fsaverage_sym/mri/brainmask.mgz"
T1_FILE = "./data/input/data4sharing/fsaverage_sym/mri/brainmask.mgz"

def ensure_dirs():
    UPLOAD_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)
