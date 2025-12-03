import sys
from fastapi import FastAPI, HTTPException, UploadFile, File
from pathlib import Path
import shutil

# assume the app is runned from the root of the repo (if run from docker it should be the case)
sys.path.insert(0, './src')
import config
from infer import load_model, predict_single_image


# use the PlantVillage validation directory to get all the classes
val_dir = config.DATA_PROCESSED_DIR / "PlantVillage" / "val"
if not val_dir.exists():
    raise FileNotFoundError(f"Validation directory not found: {val_dir}")

class_names = sorted([d.name for d in val_dir.iterdir() if d.is_dir()])
print(f"Found {len(class_names)} classes: {class_names[:5]}...")


print("Loading model...")
model = load_model()
print("Model loaded successfully.")


app = FastAPI(title="plantvillageml backend")


@app.post("/predict_image")
async def create_upload_file(file: UploadFile = File(...)):
    # create a directory to store uploads if it doesn't exist
    upload_dir = Path("uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    # define the path to save the uploaded file
    file_location = upload_dir / file.filename

    # save the uploaded file
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_single_image(model, file_location, class_names)

    return {
        "predicted": result.get('predicted_class', 'Unknown'),
        "confidence": result.get('confidence', 0),
        "health": result.get('health', 'n/a')
    }
