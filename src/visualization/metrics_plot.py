import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import classification_report
from src.config import PLOTS_DIR


def plot_class_metrics(y_true, y_pred):

    report = classification_report(y_true, y_pred, output_dict=True)

    classes = list(report.keys())[:-3]

    precision = [report[c]['precision'] for c in classes]
    recall = [report[c]['recall'] for c in classes]
    f1 = [report[c]['f1-score'] for c in classes]

    x = np.arange(len(classes))

    width = 0.25

    plt.figure(figsize=(10,6))

    plt.bar(x - width, precision, width, label="Precision")
    plt.bar(x, recall, width, label="Recall")
    plt.bar(x + width, f1, width, label="F1-score")

    plt.xticks(x, classes)

    plt.xlabel("Class")
    plt.ylabel("Score")

    plt.title("Class-wise Performance Metrics")

    plt.legend()

    plt.savefig(PLOTS_DIR + "class_metrics.png")

    plt.close()