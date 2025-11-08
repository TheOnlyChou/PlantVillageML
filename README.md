# Concordia — Machine Learning Projects

**Computer Science students @ EFREI Paris & Concordia**

This repository contains a reproducible pipeline for training and running a plant disease classification model using TensorFlow. The model was trained on the PlantVillage dataset and outputs:

1. The predicted class (e.g. `Tomato___Late_blight`, `Pepper__healthy`)
2. A simplified health label: `healthy` or `not_healthy` (added in the inference code)

---

## Tech stack

- Language: Python
- ML / Deep Learning: TensorFlow, Keras
- Other: scikit-learn, OpenCV (used in data preparation / experiments)
- Tools: Git, Jupyter, Docker (optional), WSL2 (Ubuntu), TensorBoard
- Target environment used for development: Windows 11 + WSL2, NVIDIA GPU (CUDA)

---

## Quick start (recommended: WSL2 / Ubuntu)

These instructions assume you are using WSL2 (Ubuntu) on Windows and want to run the repo there. If you prefer native Windows, adapt the commands accordingly.

1. Update the system and install prerequisites:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential python3 python3-venv python3-pip git
```

2. (Optional) Verify GPU and NVIDIA drivers are available from WSL:

```bash
# on the Windows host, ensure NVIDIA drivers & WSL integration are installed
# inside WSL:
nvidia-smi
```

3. Create a Python virtual environment and install requirements:

```bash
# from inside WSL
cd ~
python3 -m venv ~/mlproject_venv
source ~/mlproject_venv/bin/activate
pip install --upgrade pip
# from the repo root (see Project setup below), run:
pip install -r requirements.txt
```

Note: TensorFlow GPU support requires matching CUDA/cuDNN libraries in your WSL environment; follow the official TensorFlow + NVIDIA documentation for the correct CUDA version.

---

## Project setup (from WSL)

If your project lives on the Windows filesystem (e.g. under `C:\Users\...`), you can access it from WSL under `/mnt/c/...`.

```bash
# open the repo from WSL
cd /mnt/c/Users/Alexandre/PycharmProjects/MachineLearningProject
source ~/mlproject_venv/bin/activate
pip install -r requirements.txt
```

Tip: For best IO performance, you can clone the repo inside WSL's home directory and work there.

---

## Train the model

The training notebook is `notebooks/02_train_model.ipynb`. You can convert it to a script or run it directly in a Jupyter session.

To convert the notebook to a script:

```bash
cd notebooks
jupyter nbconvert --to script 02_train_model.ipynb
```

To run the training script (if you converted it) or the training module from the repo root:

```bash
python notebooks/02_train_model.py
# or, if the repo exposes a train entrypoint
python -m src.train
```

By default the trained model is saved to:

```
models/plant_disease.keras
```

---

## Monitor training (TensorBoard)

During training the logs are written to `logs/`. Start TensorBoard from the repo root:

```bash
tensorboard --logdir logs
```

Open the URL shown in the terminal (usually http://localhost:6006).

---

## Inference demo

The inference demo notebook is `notebooks/03_inference_demo.ipynb`. The simplified inference flow is implemented in `src/infer.py`.

The inference step does the following (high level):

- Load the trained model (default path: `models/plant_disease.keras`)
- Load the class names (if available)
- Predict a class for a single image
- Map the predicted class to a binary health label: if the predicted class name contains the substring `healthy` (case-insensitive) the result is `healthy`, otherwise `not_healthy`.

Example output:

```
Predicted: Tomato___healthy (confidence: 0.982)
Health: healthy
```

You can run the demo notebook in Jupyter (launch from the repo root) or build a small script that imports `src.infer` and calls `predict_single_image()`.

---

## Repository structure

```
MachineLearningProject/
├── data/                   # raw and processed datasets (ignored by git)
│   ├── raw/                # original archives
│   └── processed/          # train/val splits
├── models/                 # saved models (.keras) (ignored by git)
├── logs/                   # TensorBoard logs (ignored by git)
├── notebooks/              # Jupyter notebooks (training, inference)
├── src/                    # data loader, model, training, inference
└── requirements.txt
```

---

## Requirements

- Python >= 3.12
- TensorFlow >= 2.20 (or the version pinned in `requirements.txt`)
- Optional: NVIDIA GPU + CUDA (for GPU-accelerated training)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Notes

- Large datasets, virtual environments and model checkpoints should not be committed. Make sure `.gitignore` excludes `data/`, `models/`, `.venv/` and similar directories.
- If a file that should be ignored is already tracked by Git, remove it from the index with `git rm --cached <path>` and commit the change.

---

If you want, I can also:

- Update the inference notebook to print the binary health label alongside the predicted class.
- Add a short example script `scripts/infer_example.py` that runs a single image inference from the command line.