import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from meld_graph.paths import FEATURE_PATH

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from languidemedseg_meld.inference import inference

from .config import (DEFAULT_DEMOGRAPHIC_FILE, OUTPUT_DIR, RESULT_DIR, T1_FILE,
                     UPLOAD_DIR, ensure_dirs)
from .plotting_utils import plot_and_save
from .utils import parse_id

ensure_dirs()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# mount results dir as static files
app.mount("/opt", StaticFiles(directory="/opt"), name="opt")
app.mount("/results", StaticFiles(directory=RESULT_DIR), name="results")


@app.post("/predict")
async def predict(file: UploadFile, 
                  description: str = Form(""), 
                  model_type: str = Form("")):
    
    file_name = parse_id(file.filename)
    input_path = UPLOAD_DIR / f"{file_name}.hdf5"
    out_dir = RESULT_DIR / file_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # save uploaded file
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cmd = [
        "bash",
        "/app/meldgraph.sh",
        "run_script_prediction_inference.py",
        "--id", str(file_name),
        "--aug_mode", "test",
        "--return_result", "True",
    ]

    if Path(DEFAULT_DEMOGRAPHIC_FILE).exists():
        cmd.extend(["--demographic_file", str(DEFAULT_DEMOGRAPHIC_FILE)])

    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"Script failed: {result.stderr}")

    subject_path = Path(FEATURE_PATH) / file_name / "features" / "result.npz"

    if not subject_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found for subject {file_name} at {subject_path}"
        )

    with np.load(subject_path, allow_pickle=False) as npz:
        preds = npz["result"].astype("float32")

        subject_data = {
            file_name: {
                "result": preds
            }
        }


    img_nii, epi_dict = inference(subject_data, description, model_type)

    if isinstance(img_nii, dict):
        if file_name not in img_nii:
            raise KeyError(f"No prediction for subject {file_name}")
        img_nii = img_nii[file_name]
    elif isinstance(img_nii, list):
        img_nii = img_nii[0]

    plot_and_save(img_nii, epi_dict, file_name, out_dir, T1_FILE, model_type)

    text = f"{epi_dict['report']}"

    return {
        "text": text,
        "result_2dpng": f"/results/{file_name}/{file_name}.png",
        "result_3dpng": f"/results/{file_name}/{file_name}_surface_combined.png",
        "result_nii": f"/results/{file_name}/{file_name}.nii.gz",
        "t1_bg": T1_FILE,
        "download_2dpng": f"/download/2dpng/{file_name}",
        "download_3dpng": f"/download/3dpng/{file_name}",
        "download_nii": f"/download/nii/{file_name}",
    }


@app.get("/download/2dpng/{file_name}")
async def download_png(file_name: str):
    file_path = RESULT_DIR / file_name / f"{file_name}.png"
    return FileResponse(file_path, media_type="image/png", filename=f"{file_name}.png")

@app.get("/download/3dpng/{file_name}")
async def download_png(file_name: str):
    file_path = RESULT_DIR / file_name / f"{file_name}_surface_combined.png"
    return FileResponse(file_path, media_type="image/png", filename=f"{file_name}.png")

@app.get("/download/nii/{file_name}")
async def download_nii(file_name: str):
    file_path = RESULT_DIR / file_name / f"{file_name}.nii.gz"
    return FileResponse(file_path, media_type="application/gzip", filename=f"{file_name}.nii.gz")
