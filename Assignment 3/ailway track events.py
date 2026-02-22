import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix


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

# --- Combining the datasets ---
df = pd.concat([df1, df2, df3], ignore_index=True)

print("Combined dataset shape:", df.shape)

# ---  Removing unwanted columns  ---
cols_to_drop = ["start_time", "axle", "cluster", "tsne_1", "tsne_2"]

existing_to_drop = [c for c in cols_to_drop if c in df.columns]
missing = [c for c in cols_to_drop if c not in df.columns]

df = df.drop(columns=existing_to_drop)

print("Dropped columns:", existing_to_drop)
print("Missing (not found, OK):", missing)
print("New shape after drop:", df.shape)
print("Remaining columns:\n", df.columns)

# ---  Convert event column to binary label ---
df["label"] = (df["event"] != "normal").astype(int)

print("Label distribution (0=normal, 1=event):")
print(df["label"].value_counts())

#  see which event types became 1
print("\nEvent types:")
print(df["event"].value_counts().head(15))

# ---  Separate features and labels ---
X = df.drop(columns=["event", "label"])
y = df["label"]

print("X shape:", X.shape)
print("y shape:", y.shape)

# --- Normalize the dataset ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Normalized X shape:", X_scaled.shape)

# --- Split data (80% train / 20% test) ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# --- Train SVM using Train/Test Split ---
svm_model = SVC(kernel='rbf')

svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("SVM Accuracy (Train/Test Split):", accuracy)

# ---  5-Fold Cross Validation on training set ---
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    SVC(kernel='rbf'),
    X_train,
    y_train,
    cv=kfold,
    scoring="accuracy"
)

print("CV accuracy scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())
print("Std CV accuracy:", cv_scores.std())

print("\n--- Train/Test Evaluation ---")
print("Accuracy:", accuracy)
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))

# ---  Keep original feature matrix with names (for feature selection) ---
X_df = df.drop(columns=["event", "label"])   # DataFrame with feature names
y = df["label"]

feature_names = X_df.columns.tolist()
print("Number of features:", len(feature_names))
print("Features:", feature_names)