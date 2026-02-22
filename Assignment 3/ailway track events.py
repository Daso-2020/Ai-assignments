# --- Data handling ---
import pandas as pd  # for loading and manipulating CSV data
import numpy as np   # for numeric operations (means, std, etc.)

# --- Model + evaluation ---
from sklearn.model_selection import train_test_split           # for 80/20 split
from sklearn.model_selection import StratifiedKFold            # for k-fold CV with class balance
from sklearn.model_selection import cross_val_score            # for CV scoring
from sklearn.pipeline import Pipeline                          # to prevent leakage (scaling inside model pipeline)
from sklearn.preprocessing import StandardScaler               # standardize features (mean=0, std=1)
from sklearn.svm import SVC                                    # SVM classifier
from sklearn.metrics import accuracy_score                     # accuracy metric
from sklearn.metrics import classification_report, confusion_matrix  # detailed metrics + confusion matrix

# --- Feature selection methods ---
from sklearn.feature_selection import SelectKBest, chi2         # chi-square selection
from sklearn.preprocessing import MinMaxScaler                  # chi-square needs non-negative features
from sklearn.feature_selection import RFE                       # recursive feature elimination (wrapper)
from sklearn.ensemble import RandomForestClassifier             # embedded importance method


#  File paths
# ------------------------------
file1 = r"C:\Users\Daso-PC\Desktop\AI assigments\Assignment 3\Trail1_extracted_features_acceleration_m1ai1-1.csv"
file2 = r"C:\Users\Daso-PC\Desktop\AI assigments\Assignment 3\Trail2_extracted_features_acceleration_m1ai1.csv"
file3 = r"C:\Users\Daso-PC\Desktop\AI assigments\Assignment 3\Trail3_extracted_features_acceleration_m2ai0.csv"
# ------------------------------


# Load, combine, clean, label the data
# ----------------------------------------
df1 = pd.read_csv(file1)  # load Trail1
df2 = pd.read_csv(file2)  # load Trail2
df3 = pd.read_csv(file3)  # load Trail3

df = pd.concat([df1, df2, df3], ignore_index=True)  # combine all into one dataset

# Columns that must be removed
cols_to_drop = ["start_time", "axle", "cluster", "tsne_1", "tsne_2"]

# Drop only columns that exist
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Convert event to binary label:
# normal -> 0, anything else -> 1
df["label"] = (df["event"] != "normal").astype(int)

# Build feature matrix X (only numeric features) and target y (label)
X_df = df.drop(columns=["event", "label"])  # 16 feature columns
y = df["label"]                             # target labels

# Quick sanity prints (for tests)
print("Combined shape:", df.shape)
print("X shape:", X_df.shape, "| y shape:", y.shape)
print("Label counts:\n", y.value_counts())
# ----------------------------------------



# Define ONE leak-free SVM pipeline
# ----------------------------------------
# StandardScaler is inside Pipeline  (to aensure no leakage)
svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf"))
])
# ----------------------------------------

# Helper: evaluate any feature set in a consistent way
# -------------------------------------------------------
def evaluate_feature_set(X_features: pd.DataFrame, y_labels: pd.Series, title: str) -> float:
    """
    Evaluate a given feature subset using the SAME 80/20 stratified split + pipeline.
    Returns test accuracy.
    """
    # Stratified split keeps the class balance similar in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_features,
        y_labels,
        test_size=0.2,
        random_state=42,
        stratify=y_labels
    )

    # Fit model on training only (scaler fits only on X_train inside pipeline)
    svm_pipe.fit(X_train, y_train)

    # Predict on test set
    y_pred = svm_pipe.predict(X_test)

    # Compute accuracy
    acc = accuracy_score(y_test, y_pred)

    # Print results
    print("\n==============================")
    print(f"{title}")
    print("==============================")
    print("Test accuracy:", acc)
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))

    return acc
# -------------------------------------------------------


#  Train/Test (leak-free bug FIXED)
# ----------------------------------------
acc_all = evaluate_feature_set(X_df, y, "SVM with ALL 16 features (Leak-free)")
# ----------------------------------------


#  5-fold CV (leak-free bug FIXED)
# ----------------------------------------
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    svm_pipe,     # pipeline => scaler happens inside each fold correctly
    X_df,
    y,
    cv=kfold,
    scoring="accuracy"
)

print("\n==============================")
print("5-Fold Cross-Validation (Leak-free)")
print("==============================")
print("CV scores:", np.round(cv_scores, 4))
print("CV mean  :", cv_scores.mean())
print("CV std   :", cv_scores.std())
# ----------------------------------------


#  Feature Selection (4 methods)
# ----------------------------------------

# ---- Method 1: Pearson correlation (filter) ----
corr_temp = X_df.copy()          # copy features
corr_temp["label"] = y           # add label to compute correlation
pearson_scores = corr_temp.corr()["label"].drop("label").abs()  # abs correlation with label
top_pearson = pearson_scores.sort_values(ascending=False).head(8).index  # choose top 8

acc_pearson = evaluate_feature_set(X_df[top_pearson], y, "SVM with Pearson Top-8 features")


# ---- Method 2: Chi-square (filter) ----
# Chi-square requires non-negative values using MinMaxScaler here only for selection step
mm = MinMaxScaler()                          # scale features to [0,1]
X_mm = mm.fit_transform(X_df)                # fit on full X_df (OK because this is ONLY for ranking)
chi_selector = SelectKBest(score_func=chi2, k=8)  # select top 8 by chi2 score
chi_selector.fit(X_mm, y)                    # compute chi2 ranking
top_chi = X_df.columns[chi_selector.get_support()]  # get chosen feature names

acc_chi = evaluate_feature_set(X_df[top_chi], y, "SVM with Chi-Square Top-8 features")


# ---- Method 3: RFE (wrapper) ----
# Use linear SVM as estimator for ranking features
svm_linear = SVC(kernel="linear")
rfe = RFE(estimator=svm_linear, n_features_to_select=8)
rfe.fit(X_df, y)
top_rfe = X_df.columns[rfe.support_]

acc_rfe = evaluate_feature_set(X_df[top_rfe], y, "SVM with RFE Top-8 features")


# ---- Method 4: Random Forest importance (embedded) ----
rf = RandomForestClassifier(random_state=42)
rf.fit(X_df, y)
rf_importances = pd.Series(rf.feature_importances_, index=X_df.columns)
top_rf = rf_importances.sort_values(ascending=False).head(8).index

acc_rf = evaluate_feature_set(X_df[top_rf], y, "SVM with Random Forest Importance Top-8 features")
# ----------------------------------------


# Final summary table
# ----------------------------------------
print("\n==============================")
print("FINAL SUMMARY (Leak-free)")
print("==============================")
summary = pd.DataFrame({
    "Feature Set": [
        "All 16 features",
        "Pearson Top-8",
        "Chi-Square Top-8",
        "RFE Top-8",
        "RF Importance Top-8"
    ],
    "Test Accuracy": [
        acc_all,
        acc_pearson,
        acc_chi,
        acc_rfe,
        acc_rf
    ]
})
print(summary.to_string(index=False))
# ----------------------------------------
