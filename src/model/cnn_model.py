import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

from src.config import NUM_CLASSES


def build_model(input_length):

    model = tf.keras.Sequential([

        layers.Input(shape=(input_length,)),

        # Embedding
        layers.Embedding(
            input_dim=20000,
            output_dim=32
        ),

        # CNN Block 1
        layers.Conv1D(
            64,
            kernel_size=5,
            activation="relu",
            kernel_regularizer=l2(0.0005)
        ),

        layers.MaxPooling1D(2),

        layers.Dropout(0.3),

        # CNN Block 2
        layers.Conv1D(
            128,
            kernel_size=5,
            activation="relu",
            kernel_regularizer=l2(0.0005)
        ),

        layers.MaxPooling1D(2),

        layers.Dropout(0.3),

        # CNN Block 3
        layers.Conv1D(
            256,
            kernel_size=3,
            activation="relu",
            kernel_regularizer=l2(0.0005)
        ),

        layers.GlobalMaxPooling1D(),

        layers.Dropout(0.4),

        # Dense
        layers.Dense(
            128,
            activation="relu"
        ),

        layers.Dropout(0.4),

        layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model