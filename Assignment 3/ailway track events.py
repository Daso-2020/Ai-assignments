import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier



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

# ---  Pearson Correlation Feature Selection ---
corr_with_label = X_df.copy()
corr_with_label["label"] = y

correlations = corr_with_label.corr()["label"].drop("label").abs()

top_features_pearson = correlations.sort_values(ascending=False).head(8).index

print("\nTop 8 Pearson Features:")
print(top_features_pearson)

# ---  Evaluate SVM with Pearson-selected features (Top 8) ---
X_pearson = X_df[top_features_pearson]

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_pearson, y, test_size=0.2, random_state=42
)

svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf"))
])

svm_pipe.fit(X_train_p, y_train_p)
y_pred_p = svm_pipe.predict(X_test_p)

acc_p = accuracy_score(y_test_p, y_pred_p)

print("SVM Accuracy with Pearson Top-8:", acc_p)
print("Confusion matrix:\n", confusion_matrix(y_test_p, y_pred_p))
print(classification_report(y_test_p, y_pred_p, digits=4))

# ---  Chi-Square Feature Selection ---
scaler_mm = MinMaxScaler()
X_scaled_mm = scaler_mm.fit_transform(X_df)

chi_selector = SelectKBest(score_func=chi2, k=8)
X_chi = chi_selector.fit_transform(X_scaled_mm, y)

chi_features = X_df.columns[chi_selector.get_support()]

print("\nTop 8 Chi-Square Features:")
print(chi_features)

# ---  Evaluate SVM with Chi-Square-selected features (Top 8) ---
X_chi_df = X_df[chi_features]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_chi_df, y, test_size=0.2, random_state=42
)

svm_pipe.fit(X_train_c, y_train_c)
y_pred_c = svm_pipe.predict(X_test_c)

acc_c = accuracy_score(y_test_c, y_pred_c)

print("SVM Accuracy with Chi-Square Top-8:", acc_c)
print("Confusion matrix:\n", confusion_matrix(y_test_c, y_pred_c))
print(classification_report(y_test_c, y_pred_c, digits=4))

# --- RFE Feature Selection ---
svm_linear = SVC(kernel="linear")

rfe_selector = RFE(estimator=svm_linear, n_features_to_select=8)
rfe_selector.fit(X_df, y)

rfe_features = X_df.columns[rfe_selector.support_]

print("\nTop 8 RFE Features:")
print(rfe_features)

# ---  Evaluate SVM with RFE-selected features (Top 8) ---
X_rfe = X_df[rfe_features]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_rfe, y, test_size=0.2, random_state=42
)

svm_pipe.fit(X_train_r, y_train_r)
y_pred_r = svm_pipe.predict(X_test_r)

acc_r = accuracy_score(y_test_r, y_pred_r)

print("SVM Accuracy with RFE Top-8:", acc_r)
print("Confusion matrix:\n", confusion_matrix(y_test_r, y_pred_r))
print(classification_report(y_test_r, y_pred_r, digits=4))

# ---  Random Forest Feature Importance ---
rf = RandomForestClassifier(random_state=42)
rf.fit(X_df, y)

importances = rf.feature_importances_

rf_importances = pd.Series(importances, index=X_df.columns)
top_features_rf = rf_importances.sort_values(ascending=False).head(8).index

print("\nTop 8 RF Importance Features:")
print(top_features_rf)

# --- Evaluate SVM with RF-selected features (Top 8) ---
X_rf = X_df[top_features_rf]

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_rf, y, test_size=0.2, random_state=42
)

svm_pipe.fit(X_train_f, y_train_f)
y_pred_f = svm_pipe.predict(X_test_f)

acc_f = accuracy_score(y_test_f, y_pred_f)

print("SVM Accuracy with RF Top-8:", acc_f)
print("Confusion matrix:\n", confusion_matrix(y_test_f, y_pred_f))
print(classification_report(y_test_f, y_pred_f, digits=4))