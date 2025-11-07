import tensorflow as tf
from src.model import build_model

def train_model(train_ds,
                val_ds,
                num_classes: int,
                epochs: int = 10,
                save_path: str = "models/plant_disease.keras"):
    """
    Train a TensorFlow model on the provided training and validation datasets.
    """
    # Build the model
    model = build_model(num_classes)

    # Perform data prefetching for performance optimization
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # Save the trained model
    model.save(save_path)

    return model, history
