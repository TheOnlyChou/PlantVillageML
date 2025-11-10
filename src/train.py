import datetime
from pathlib import Path

import tensorflow as tf
from . import config
from .model import build_model


def train_model(
    train_ds,
    val_ds,
    num_classes: int,
    epochs: int = 10,
    save_path=None,
):
    model = build_model(num_classes)

    if save_path is None:
        save_path = config.MODELS_DIR / "plant_disease.keras"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    logs_dir = getattr(config, "LOGS_DIR", Path("logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = logs_dir / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=run_dir,
        histogram_freq=1,
        write_graph=False,
        write_images=False,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            monitor="val_loss",
            save_best_only=True,
        ),
        tensorboard_cb,
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    if not save_path.exists():
        model.save(save_path)

    return model, history