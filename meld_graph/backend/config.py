from pathlib import Path
import os


# =========================
# Project structure
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data"
INPUT_DIR = DATA_ROOT / "input"
OUTPUT_DIR = DATA_ROOT / "output"

UPLOAD_DIR = INPUT_DIR / "meld_combats"
RESULT_DIR = OUTPUT_DIR / "results"


# =========================
# Demographics
# =========================

DEFAULT_DEMOGRAPHIC_FILE = (
    INPUT_DIR / "demographics_qc_allgroups_withH27H28H101.csv"
)


# =========================
# FreeSurfer
# =========================

FREESURFER_HOME = Path(
    os.environ.get("FREESURFER_HOME", "/opt/freesurfer-7.2.0")
)

SUBJECTS_DIR = Path(
    os.environ.get("SUBJECTS_DIR", FREESURFER_HOME / "subjects")
)

T1_FILE = SUBJECTS_DIR / "fsaverage_sym" / "mri" / "brainmask.mgz"
# =========================

def ensure_dirs():
    UPLOAD_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)
