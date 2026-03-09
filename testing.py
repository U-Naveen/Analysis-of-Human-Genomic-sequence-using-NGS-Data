import pandas as pd

df = pd.read_csv("data/raw/Human Data Sequnence.txt", sep="\t")

print(df["class"].value_counts())