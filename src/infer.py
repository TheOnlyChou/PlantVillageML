import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image
import config


def load_model(model_path=None):
    if model_path is None:
        model_path = config.MODELS_DIR / "plant_disease.keras"
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    return tf.keras.models.load_model(model_path)


def preprocess_image(img_path, target_size=None):
    if target_size is None:
        target_size = config.IMG_SIZE
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img, dtype="float32")
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def is_healthy(class_name: str) -> bool:
    if not class_name:
        return False
    return "healthy" in class_name.lower()


def predict_single_image(model, img_path, class_names=None):
    img_array = preprocess_image(img_path)
    preds = model.predict(img_array, verbose=0)
    pred_idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][pred_idx])

    result = {
        "predicted_index": pred_idx,
        "confidence": confidence,
        "all_probabilities": preds[0].tolist(),
    }

    predicted_class = None
    if class_names is not None:
        predicted_class = class_names[pred_idx]
        result["predicted_class"] = predicted_class

    # Binary health assessment
    if predicted_class is not None and is_healthy(predicted_class):
        result["health"] = "healthy"
    else:
        result["health"] = "not_healthy"

    return result


def batch_predict(model, image_paths, class_names=None):
    results = []
    for img_path in image_paths:
        try:
            out = predict_single_image(model, img_path, class_names)
            out["image_path"] = str(img_path)
            results.append(out)
        except Exception as e:
            results.append({"image_path": str(img_path), "error": str(e)})
    return results
