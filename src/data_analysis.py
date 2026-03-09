import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import DATA_PATH, PLOTS_DIR


def run_analysis():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH, sep="\t")

    print("Dataset shape:", df.shape)
    print(df.head())

    # ===============================
    # Class Distribution
    # ===============================

    plt.figure()

    sns.countplot(x="class", data=df)

    plt.title("Class Distribution")

    plt.savefig(PLOTS_DIR + "class_distribution.png")

    plt.close()

    # ===============================
    # Sequence Length Distribution
    # ===============================

    df["length"] = df["sequence"].apply(len)

    plt.figure()

    sns.histplot(df["length"], bins=50)

    plt.title("Sequence Length Distribution")

    plt.savefig(PLOTS_DIR + "sequence_length_distribution.png")

    plt.close()

    # ===============================
    # Nucleotide Frequency
    # ===============================

    nucleotides = ["A", "C", "G", "T"]

    counts = {n: 0 for n in nucleotides}

    for seq in df["sequence"]:

        for n in nucleotides:

            counts[n] += seq.count(n)

    plt.figure()

    plt.bar(counts.keys(), counts.values())

    plt.title("Nucleotide Frequency")

    plt.savefig(PLOTS_DIR + "nucleotide_frequency.png")

    plt.close()

    print("Analysis plots saved in outputs/plots/")