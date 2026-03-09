import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from src.utils.dna_encoding import generate_kmers
from src.config import DATA_PATH, TEST_SIZE, RANDOM_STATE


def preprocess_data():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH, sep="\t")

    # ===============================
    # Generate k-mers
    # ===============================

    df["kmers"] = df["sequence"].apply(lambda x: " ".join(generate_kmers(x, k=3)))

    # ===============================
    # Tokenize k-mers
    # ===============================

    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(df["kmers"])

    sequences = tokenizer.texts_to_sequences(df["kmers"])

    # ===============================
    # Pad sequences
    # ===============================

    MAX_LEN = 500

    X = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    y = df["class"]

    print("Encoded shape:", X.shape)

    # ===============================
    # Train Test Split
    # ===============================

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y

    )

    return X_train, X_test, y_train, y_test