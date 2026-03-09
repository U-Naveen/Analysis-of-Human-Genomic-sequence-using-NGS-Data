import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

from src.config import PLOTS_DIR


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    predicted_classes = np.argmax(predictions, axis=1)

    print(classification_report(y_test, predicted_classes))

    # ==============================
    # Confusion Matrix
    # ==============================

    cm = confusion_matrix(y_test, predicted_classes)

    plt.figure(figsize=(8,6))

    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        fmt="d"
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.savefig(PLOTS_DIR + "confusion_matrix.png")

    plt.close()