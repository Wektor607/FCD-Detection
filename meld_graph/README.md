# Multimodal FCD Detection Model  
### Docker Setup & Training Guide

This repository contains the code for a **new multimodal architecture for FCD detection**,  
built on top of the [**MELD Graph**](https://github.com/MELDProject/meld_graph/tree/main?tab=readme-ov-file) pipeline and fully wrapped in Docker.

⚠️ **Important notes**

* **Memory requirements:** at least **20 GB RAM** is recommended  
  (especially for MELD preprocessing and training inside Docker)
* Please check the official [**MELD documentation**](https://meld-graph.readthedocs.io/en/latest/install_docker.html)
* If the instructions below are unclear, we highly recommend watching the official  
  [**installation video**](https://www.youtube.com/watch?v=oduOe6NDXLA)
* For model training, GPUs such as **NVIDIA A40 or A100** are recommended,  
  or other GPUs with a **similar amount of VRAM**.

---

## 1. Prerequisites

### 1.1 Install Docker

Follow the official instructions:

* [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)

Verify:

```bash
docker --version
```

---

### 1.2 Install NVIDIA Container Toolkit (GPU support)

Required to run training on GPU.

Follow:
[https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Verify:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi
```

---

### 1.3 Get FreeSurfer License

* Register and download license here:
  [https://surfer.nmr.mgh.harvard.edu/registration.html](https://surfer.nmr.mgh.harvard.edu/registration.html)

You will receive a file called:

```
license.txt
```

Move the license file to the **meld_graph root folder** (where `Dockerfile` is located):

```
meld_graph/
├── Dockerfile
├── compose.yml
├── license.txt   ← HERE
```

---

### 1.4 MELD License

You [**must fill out the MELD form**](https://docs.google.com/forms/d/e/1FAIpQLSdocMWtxbmh9T7Sv8NT4f0Kpev-tmRI-kngDhUeBF9VcZXcfg/viewform) to obtain the MELD license.
Place it next to `license.txt`:

```
meld_license.txt
```

---

## 2. Data Setup

### 2.1 Update Data Path (IMPORTANT)

Edit `compose.yml`:

```yaml
volumes:
  - {YOUR_PATH}/meld_graph/data:/data
```

⚠️ This path **must point to your local data directory**.

---

### 2.2 Required Files

The following file is required:

```

/data/MELD_splits.csv

```

This file must contain **two columns**:

```

subject_id,split
patient_1,test
patient_2,trainval
...

```

where:
- `subject_id` — unique subject identifier  
- `split` — data split assignment (`trainval` or `test`)

⚠️ **Note:** A demographic file is **not required** for this pipeline.


---

## 3. Model Setup

### 3.1 Using MELD Pretrained Model

Run once:

```bash
DOCKER_USER="$(id -u):$(id -g)" \
docker compose run meld_graph \
python scripts/new_patient_pipeline/prepare_classifier.py
```

---

### 3.2 Using Pretrained Models

Create the following directory and place the pretrained model checkpoints in this directory:

```bash
data/saved_models/
```

📦 **Pretrained models availability**

* A public download link will be provided later.
* Until then, please contact the author via email: **[mikhelson.g@gmail.com](mailto:mikhelson.g@gmail.com)**
---

### 3.3 Feature Generation (MELD Graph)

Before starting model training, make sure that **surface-based features at multiple hierarchy levels**
have been generated from the HDF5 data using the MELD Graph pipeline.

If your data are already available in **HDF5 format**, run the following command to generate
multi-level features:

```bash
./meldgraph.sh run_script_prediction_meld.py \
  --list_ids input/data4sharing/demographics_qc_allgroups_withH27H28H101.csv \
  --demographic_file input/data4sharing/demographics_qc_allgroups_withH27H28H101.csv \
  --aug_mode train
```

This step generates surface-based features at different icosphere levels, which are required
by the multimodal model during training and inference.

⚠️ **Note:**
If you only have **raw MRI data** and need to convert them into the MELD-compatible HDF5 format,
please follow the official MELD Graph harmonisation and preprocessing instructions:
[https://meld-graph.readthedocs.io/en/latest/harmonisation.html](https://meld-graph.readthedocs.io/en/latest/harmonisation.html)

---

### 3.4 Text Generation

For the multimodal setup, **textual descriptions must be generated** for each subject
before training and inference.

Instructions and scripts for generating text descriptions are provided here:

```

meld_graph/utils/text_generation/README.md

```

Please follow the steps described in this README to generate the required text inputs,
which are then used by the language-guided component of the model.

---

## 4. Training

### 4.1 Build Image

```bash
DOCKER_BUILDKIT=0 docker compose -f compose.yml build
```

---

### 4.2 Start Container

```bash
docker compose up -d meld_graph
```

---

### 4.3 Enter Container

```bash
docker compose exec meld_graph bash
```

---

### 4.4 Run Training

```bash
WANDB_MODE=disabled \
python languidemedseg_meld/train_Kfold.py \
  --config languidemedseg_meld/config/training.yaml \
  --job_name exp2
```

⚠️ **Note on ensemble training and random seeds**

During our experiments, we trained an **ensemble of 5 models**.
To increase ensemble diversity, **random seeds were manually changed between runs**.

Although the training script defines a base seed (`SEED = 42`), each cross-validation fold
is trained with a **fold-specific seed**:

```
fold_seed = SEED + fold
```

This seed affects data shuffling, sampling, model initialization, and all other stochastic
components of training.

If you plan to reproduce the ensemble setup, make sure to **vary the base seed across runs**
(e.g. by modifying the `SEED` value in the training script).
Running multiple trainings with the same seed will result in highly correlated models and
significantly reduce the effectiveness of the ensemble.

---

### 4.5 Run Testing

```markdown
```bash
WANDB_MODE=disabled \
python3 languidemedseg_meld/test_Kfold.py \
  --config languidemedseg_meld/config/training.yaml \
  --ckpt_prefix saved_models/exp2
````

⚠️ **Note on ensemble evaluation**

The testing script supports **ensemble inference** by loading multiple checkpoints
corresponding to different cross-validation folds.

By default, the ensemble is constructed using the following logic in the test script:

```python
ckpt_paths = [
    ckpt_prefix.parent / f"{ckpt_prefix.name}_fold{i+1}.ckpt"
    for i in range(5)
]
```

This assumes that:

* models were trained using **5-fold cross-validation**, and
* checkpoint files follow the naming pattern
  `<ckpt_prefix>_fold1.ckpt, ..., <ckpt_prefix>_fold5.ckpt`.

If you trained a different number of folds or used a custom naming scheme,
please adjust this section of the code accordingly.

---

## 5. Web Interface

### 5.1 Run Web Stack

```bash
DOCKER_BUILDKIT=0 \
docker compose -f docker-web-compose.yml up --build
```

---

## 6. Common Issues

### ❌ GPU not detected

Check:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi
```

---

### ❌ Bus error / DataLoader crash

Cause: insufficient shared memory (`/dev/shm`)

Fix:

```bash
docker compose up -d --shm-size=8g
```

---

## 7. Cleanup (Disk Space)

Remove stopped containers:

```bash
docker container prune
```

Remove unused images:

```bash
docker image prune
```

Full cleanup:

```bash
docker system prune -a
```

---

## 8. System Requirements

| Resource | Minimum  | Recommended  |
| -------- | -------- | ------------ |
| RAM | ❌ 8 GB | ✅ ≥20 GB (recommended 32 GB) |
| GPU      | Optional | ✅ NVIDIA A40 / A100 GPU |
| Disk     | 50+ GB   | 100+ GB      |

---

## Contact / Notes

If something breaks — **it’s usually paths, licenses, or RAM/GPU**.  
Please double-check those first.

For questions, bug reports, or access to pretrained models, feel free to contact:  
**mikhelson.g@gmail.com**
