from pathlib import Path

"""
Configuration settings for the machine learning project.
Instead of hardcoding values throughout the codebase, we define them here for easy management.
"""
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10