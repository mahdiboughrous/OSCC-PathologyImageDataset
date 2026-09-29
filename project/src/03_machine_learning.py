import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")

# 1. Load data
df = pd.read_csv("data/processed_data.csv")
df.replace("/", np.nan, inplace=True)

# Format types and feature engineering
num_cols = ["Age(Y)", "Weight(kg)", "Height(cm)"]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["BMI"] = df["Weight(kg)"] / ((df["Height(cm)"] / 100) ** 2)

binary_cols = [
    "Gender(0male/1female)",
    "AlcoholHistory(0no/1yes)",
    "SmokingHistory(0no/1yes)",
    "BetelNutHistory(0no/1yes)",
    "Diabetes(0no/1yes)",
    "RespiratoryDisease(0no/1yes)",
    "CardiovascularDisease(0no/1yes)",
    "MedControlledHypertension(0no/1yes)",
    "TD", "TI", "CE", "PI",
]
for col in binary_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["REC"] = df["REC"].astype(int)

# 2. Define feature sets
clinical_num = ["Age(Y)", "BMI"]
clinical_cat = [
    "Gender(0male/1female)",
    "SmokingHistory(0no/1yes)",
    "AlcoholHistory(0no/1yes)",
    "BetelNutHistory(0no/1yes)",
    "ASAGrade",
    "Surgery Site",
    "Diabetes(0no/1yes)",
    "CardiovascularDisease(0no/1yes)",
    "MedControlledHypertension(0no/1yes)",
]

pathology_vars = ["TD", "TI", "CE", "PI"]

# Split data using predefined splits
train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "valid"]
test_df = df[df["split"] == "test"]

print(f"Split sizes: train={len(train_df)}, valid={len(val_df)}, test={len(test_df)}")
print(f"Train REC dist: {(train_df['REC']==0).sum()} neg, {(train_df['REC']==1).sum()} pos")

# 3. Model setup helper
def make_pipeline(classifier, num_features, cat_features):
    transformers = []
    if num_features:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_pipe, num_features))
    if cat_features:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers.append(("cat", cat_pipe, cat_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([("preprocessor", preprocessor), ("clf", classifier)])

feature_configs = {
    "Clinical": (clinical_num, clinical_cat),
    "Pathology": ([], pathology_vars),
    "Combined": (clinical_num, clinical_cat + pathology_vars),
}

models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=2024
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=4, class_weight="balanced", random_state=2024
    ),
}

# 4. Training and Evaluation
results = []
y_train = train_df["REC"]
y_test = test_df["REC"]

for feat_name, (num_f, cat_f) in feature_configs.items():
    all_f = num_f + cat_f
    X_train = train_df[all_f]
    X_test = test_df[all_f]

    for model_name, clf in models.items():
        pipe = make_pipeline(clf, num_f, cat_f)
        pipe.fit(X_train, y_train)

        probs = pipe.predict_proba(X_test)[:, 1]
        preds = pipe.predict(X_test)

        results.append({
            "Model": model_name,
            "Features": feat_name,
            "ROC-AUC": round(roc_auc_score(y_test, probs), 3),
            "AUPRC": round(average_precision_score(y_test, probs), 3),
            "F1": round(f1_score(y_test, preds), 3),
            "Recall": round(recall_score(y_test, preds), 3),
            "Brier": round(brier_score_loss(y_test, probs), 3),
        })

res_df = pd.DataFrame(results)
res_df.to_csv("results/tables/main_model_comparison.csv", index=False)
print("\nSaved results/tables/main_model_comparison.csv\n")
print(res_df.round(3).to_string(index=False))
