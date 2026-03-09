import matplotlib.pyplot as plt
import os
from tensorflow.keras.callbacks import EarlyStopping
from src.data_preprocessing import preprocess_data
from src.model.cnn_model import build_model
from src.config import EPOCHS, BATCH_SIZE, MODEL_PATH, PLOTS_DIR


def train_model():

    os.makedirs(PLOTS_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = preprocess_data()

    model = build_model(X_train.shape[1])
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[early_stop]
    )

    model.save(MODEL_PATH)

    # ==============================
    # Accuracy Graph
    # ==============================

    plt.figure(figsize=(8,5))

    plt.plot(history.history['accuracy'], label="Train Accuracy")
    plt.plot(history.history['val_accuracy'], label="Validation Accuracy")

    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig(PLOTS_DIR + "training_accuracy.png")
    plt.close()

    # ==============================
    # Loss Graph
    # ==============================

    plt.figure(figsize=(8,5))

    plt.plot(history.history['loss'], label="Train Loss")
    plt.plot(history.history['val_loss'], label="Validation Loss")

    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig(PLOTS_DIR + "training_loss.png")
    plt.close()

    return model, X_test, y_test