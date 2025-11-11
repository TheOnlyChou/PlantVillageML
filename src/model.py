import tensorflow as tf
import config


def build_model(num_classes: int) -> tf.keras.Model:
    """
    Builds and compiles a transfer learning model using EfficientNetB0.
    Args:
        num_classes (int): number of output classes.
    Returns:
        tf.keras.Model: compiled Keras model ready for training.
    """

    # Base model (EfficientNet pretrained on ImageNet)
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, input_shape=(*config.IMG_SIZE, 3), weights="imagenet"
    )
    base_model.trainable = False  # Freeze pretrained layers

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
        ],
        name="data_augmentation",
    )

    inputs = tf.keras.Input(shape=(*config.IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="plant_disease_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
