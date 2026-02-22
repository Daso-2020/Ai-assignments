import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ---  Loading the three datasets ---
file1 = r"C:\Users\Daso-PC\Desktop\AI assigments\Assignment 3\Trail1_extracted_features_acceleration_m1ai1-1.csv"
file2 = r"C:\Users\Daso-PC\Desktop\AI assigments\Assignment 3\Trail2_extracted_features_acceleration_m1ai1.csv"
file3 = r"C:\Users\Daso-PC\Desktop\AI assigments\Assignment 3\Trail3_extracted_features_acceleration_m2ai0.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)

print("Trail1 shape:", df1.shape)
print("Trail2 shape:", df2.shape)
print("Trail3 shape:", df3.shape)

print("\nColumns preview:\n", df1.columns)
print("\nEvent values example:\n", df1["event"].value_counts().head(10))

# --- Combine the datasets ---
df = pd.concat([df1, df2, df3], ignore_index=True)

print("Combined dataset shape:", df.shape)