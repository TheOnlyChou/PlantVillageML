import tensorflow as tf
from src import config

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
        include_top=False,
        input_shape=(*config.IMG_SIZE, 3),
        weights="imagenet"
    )
    base_model.trainable = False  # Freeze pretrained layers

    # Model architecture
    inputs = tf.keras.Input(shape=(*config.IMG_SIZE, 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)  # prevent overfitting
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    # Compilation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model