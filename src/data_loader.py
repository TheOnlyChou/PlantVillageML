import tensorflow as tf
import config


def get_train_val_ds():
    base_dir = config.DATA_PROCESSED_DIR / "PlantVillage"
    train_dir = base_dir / "train"
    val_dir = base_dir / "val"

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE, shuffle=False
    )

    # Get class names from training dataset
    class_names = train_ds.class_names

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)

    return train_ds, val_ds, class_names
