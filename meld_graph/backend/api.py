import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

# ensure package imports for meld_graph and project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "meld_graph")))

from fastapi import FastAPI, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from languidemedseg_meld.inference import inference

from .config import (DEFAULT_DEMOGRAPHIC_FILE, RESULT_DIR, T1_FILE, UPLOAD_DIR,
                     ensure_dirs)
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
app.mount("/results", StaticFiles(directory=RESULT_DIR), name="results")


@app.post("/predict")
async def predict(file: UploadFile, description: str = Form(...), model_type: str = Form(None)):
    file_name = parse_id(file.filename)
    input_path = UPLOAD_DIR / f"{file_name}.hdf5"
    out_dir = RESULT_DIR / file_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # save uploaded file
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cmd = [
        "./meld_graph/meldgraph.sh",
        "run_script_prediction_meld.py",
        "--id", str(file_name),
        "--demographic_file", str(DEFAULT_DEMOGRAPHIC_FILE),
        "--aug_mode", "test",
        "--return_result", "True",
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"Script failed: {result.stderr}")

    subject_path = result.stdout.strip().splitlines()[-1]
    with open(subject_path, "rb") as f:
        data_dict = pickle.load(f)

    img_nii, epi_dict = inference(data_dict, description, model_type)
    if isinstance(img_nii, list):
        img_nii = img_nii[0]

    plot_and_save(img_nii, epi_dict, file_name, out_dir, T1_FILE)

    text = f"{epi_dict['report']}"

    return {
        "text": text,
        "result_png": f"/results/{file_name}/{file_name}.png",
        "download_png": f"/download/png/{file_name}",
        "download_nii": f"/download/nii/{file_name}",
    }


@app.get("/download/png/{file_name}")
async def download_png(file_name: str):
    file_path = RESULT_DIR / file_name / f"{file_name}.png"
    return FileResponse(file_path, media_type="image/png", filename=f"{file_name}.png")


@app.get("/download/nii/{file_name}")
async def download_nii(file_name: str):
    file_path = RESULT_DIR / file_name / f"{file_name}.nii.gz"
    return FileResponse(file_path, media_type="application/gzip", filename=f"{file_name}.nii.gz")
