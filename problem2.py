import numpy as np
import math
import pandas as pd

dataset = pd.read_csv("coursework_ml\dataset_iris.csv", sep=";", encoding="utf-8", header=None)
df = pd.DataFrame(dataset)

amount_train = round(0.7*len(df))
titles = df.iloc[:, 0].unique()
one_hot_titles = []
zeroes_string = "0"*len(titles)
for i in range(len(zeroes_string)):
    category = zeroes_string
print(zeroes_string)
features_train = df.iloc[:, 1:][:amount_train]
features_labels = df.iloc[:, 0][:amount_train]

validation_train = df.iloc[:, 1:][amount_train:]
validation_labels = df.iloc[:, 0][amount_train:]

