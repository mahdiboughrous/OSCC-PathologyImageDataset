import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    brier_score_loss, precision_recall_curve, confusion_matrix,
    accuracy_score, recall_score, f1_score, precision_score
)
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.size": 11})

# ---- 1. Load & Clean ------------------------------------------------
df = pd.read_csv("data/processed_data.csv")
df.replace("/", np.nan, inplace=True)

num_cols = ["Age(Y)", "Weight(kg)", "Height(cm)"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["BMI"] = df["Weight(kg)"] / ((df["Height(cm)"] / 100) ** 2)

cat_cols = [
    "Gender(0male/1female)", "AlcoholHistory(0no/1yes)",
    "SmokingHistory(0no/1yes)", "BetelNutHistory(0no/1yes)",
    "ASAGrade", "Surgery Site", "Diabetes(0no/1yes)",
    "CardiovascularDisease(0no/1yes)", "MedControlledHypertension(0no/1yes)",
    "TD", "TI", "CE", "PI",
]
for c in cat_cols:
    if c != "Surgery Site":
        df[c] = pd.to_numeric(df[c], errors="coerce")
df["REC"] = df["REC"].astype(int)

clinical_num = ["Age(Y)", "BMI"]
clinical_cat = [
    "Gender(0male/1female)", "SmokingHistory(0no/1yes)",
    "AlcoholHistory(0no/1yes)", "BetelNutHistory(0no/1yes)",
    "ASAGrade", "Surgery Site", "Diabetes(0no/1yes)",
    "CardiovascularDisease(0no/1yes)", "MedControlledHypertension(0no/1yes)",
]
pathology_vars = ["TD", "TI", "CE", "PI"]

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "valid"].copy()
test_df = df[df["split"] == "test"].copy()
y_train = train_df["REC"].values
y_test = test_df["REC"].values
n_test = len(y_test)
prevalence = y_test.mean()
print(f"Test set: n={n_test}, REC prevalence={prevalence:.3f}")

# ---- TD encoding check ------------------------------------------------
print(f"\nTD encoding in training: {train_df['TD'].value_counts().sort_index().to_dict()}")
print(f"TI encoding: {train_df['TI'].value_counts().sort_index().to_dict()}")
print(f"CE encoding: {train_df['CE'].value_counts().sort_index().to_dict()}")
print(f"PI encoding: {train_df['PI'].value_counts().sort_index().to_dict()}")

# ---- 2. Build pipelines ------------------------------------------------
def make_pipe(classifier, num_f, cat_f):
    transformers = []
    if num_f:
        transformers.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler())
        ]), num_f))
    if cat_f:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_f))
    return Pipeline([
        ("prep", ColumnTransformer(transformers=transformers, remainder="drop")),
        ("clf", classifier)
    ])

configs = {
    "Clinical LR": (clinical_num, clinical_cat,
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=2024)),
    "Clinical RF": (clinical_num, clinical_cat,
        RandomForestClassifier(n_estimators=200, max_depth=4, class_weight="balanced", random_state=2024)),
    "Pathology LR": ([], pathology_vars,
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=2024)),
    "Pathology RF": ([], pathology_vars,
        RandomForestClassifier(n_estimators=200, max_depth=4, class_weight="balanced", random_state=2024)),
    "Combined LR": (clinical_num, clinical_cat + pathology_vars,
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=2024)),
    "Combined RF": (clinical_num, clinical_cat + pathology_vars,
        RandomForestClassifier(n_estimators=200, max_depth=4, class_weight="balanced", random_state=2024)),
}

# ---- 3. Fit & store predictions -----------------------------------------
probs_all = {}
for name, (n_f, c_f, model) in configs.items():
    pipe = make_pipe(model, n_f, c_f)
    pipe.fit(train_df[n_f + c_f], y_train)
    probs_all[name] = pipe.predict_proba(test_df[n_f + c_f])[:, 1]

# ---- 4. Threshold selection via validation set --------------------------
for name, (n_f, c_f, model) in configs.items():
    pipe = make_pipe(model, n_f, c_f)
    pipe.fit(train_df[n_f + c_f], y_train)
    val_probs = pipe.predict_proba(val_df[n_f + c_f])[:, 1]
    prec, rec, thresholds = precision_recall_curve(val_df["REC"].values, val_probs)
    f1_vals = 2 * prec * rec / (prec + rec + 1e-8)
    best_idx = int(np.nanargmax(f1_vals))
    best_thresh = float(thresholds[min(best_idx, len(thresholds) - 1)])
    best_thresh = max(0.0, min(1.0, best_thresh))
    probs_all[f"{name}_thresh"] = best_thresh
    print(f"{name:20s}: optimal threshold (F1-val) = {best_thresh:.3f}")

# ---- 5. Evaluate all models at default (0.5) and optimal thresholds -----
def calc_metrics(y_true, y_prob, thresh):
    y_pred = (y_prob >= thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "NPV": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "AUC": float(roc_auc_score(y_true, y_prob)),
        "AUPRC": float(average_precision_score(y_true, y_prob)),
        "Brier": float(brier_score_loss(y_true, y_prob)),
    }

results = []
for name in configs.keys():
    probs = probs_all[name]
    opt_thresh = probs_all[f"{name}_thresh"]
    
    # Default threshold 0.5
    m = calc_metrics(y_test, probs, 0.5)
    m["Model"] = name
    m["Threshold"] = 0.5
    results.append(m)
    
    # Optimal threshold
    m = calc_metrics(y_test, probs, opt_thresh)
    m["Model"] = name
    m["Threshold"] = round(opt_thresh, 3)
    results.append(m)

res_df = pd.DataFrame(results)
res_df.to_csv("results/tables/full_evaluation.csv", index=False)
print("\nSaved results/tables/full_evaluation.csv")
print(res_df[["Model", "Threshold", "AUC", "AUPRC", "Precision", "Recall", "Specificity", "F1", "Brier"]].to_string(index=False))

# ---- 6. Bootstrap CIs for all metrics ------------------------------------
np.random.seed(2024)
ci_rows = []
for name in configs.keys():
    probs = probs_all[name]
    base_auc = roc_auc_score(y_test, probs)
    base_auprc = average_precision_score(y_test, probs)
    base_brier = brier_score_loss(y_test, probs)
    
    boot_auc, boot_auprc, boot_brier = [], [], []
    for _ in range(1000):
        idx = np.random.choice(n_test, size=n_test, replace=True)
        if len(np.unique(y_test[idx])) > 1:
            boot_auc.append(roc_auc_score(y_test[idx], probs[idx]))
            boot_auprc.append(average_precision_score(y_test[idx], probs[idx]))
        boot_brier.append(brier_score_loss(y_test[idx], probs[idx]))
    
    ci_rows.append({
        "Model": name,
        "AUC_point": round(base_auc, 4),
        "AUC_CI_low": round(np.percentile(boot_auc, 2.5), 4),
        "AUC_CI_high": round(np.percentile(boot_auc, 97.5), 4),
        "AUPRC_point": round(base_auprc, 4),
        "AUPRC_CI_low": round(np.percentile(boot_auprc, 2.5), 4),
        "AUPRC_CI_high": round(np.percentile(boot_auprc, 97.5), 4),
        "Brier_point": round(base_brier, 4),
        "Brier_CI_low": round(np.percentile(boot_brier, 2.5), 4),
        "Brier_CI_high": round(np.percentile(boot_brier, 97.5), 4),
    })

ci_df = pd.DataFrame(ci_rows)
ci_df.to_csv("results/tables/bootstrap_confidence_intervals.csv", index=False)
print("\nSaved results/tables/bootstrap_confidence_intervals.csv")
print(ci_df.round(4).to_string(index=False))

# ---- 7. DeLong-style paired AUC comparison (bootstrap diff) -------------
def paired_auc_test(probs_a, probs_b, y_true, n_boot=1000):
    auc_a = roc_auc_score(y_true, probs_a)
    auc_b = roc_auc_score(y_true, probs_b)
    diff_obs = auc_a - auc_b
    
    boot_diffs = []
    np.random.seed(42)
    for _ in range(n_boot):
        idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) > 1:
            a = roc_auc_score(y_true[idx], probs_a[idx])
            b = roc_auc_score(y_true[idx], probs_b[idx])
            boot_diffs.append(a - b)
    
    ci_low = np.percentile(boot_diffs, 2.5)
    ci_high = np.percentile(boot_diffs, 97.5)
    # Two-sided p-value: proportion of bootstrap diffs with opposite sign
    p_val = np.mean(np.sign(boot_diffs) != np.sign(diff_obs)) * 2
    p_val = min(p_val, 1.0)
    
    return {"auc_A": auc_a, "auc_B": auc_b, "diff": diff_obs,
            "CI_low": ci_low, "CI_high": ci_high, "p_value": p_val}

comparisons = [
    ("Pathology RF", "Combined RF", "RF: Pathology vs Combined"),
    ("Pathology LR", "Combined LR", "LR: Pathology vs Combined"),
    ("Pathology RF", "Clinical RF", "RF: Pathology vs Clinical"),
    ("Pathology LR", "Clinical LR", "LR: Pathology vs Clinical"),
]

delong_results = []
for na, nb, label in comparisons:
    r = paired_auc_test(probs_all[na], probs_all[nb], y_test)
    r["Comparison"] = label
    delong_results.append(r)

delong_df = pd.DataFrame(delong_results)
delong_df.to_csv("results/tables/delong_pairwise_comparisons.csv", index=False)
print("\nSaved results/tables/delong_pairwise_comparisons.csv")
print(delong_df.round(4).to_string(index=False))

# ---- 8. Decision-curve analysis -----------------------------------------
def net_benefit(y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (preds == 1))
    fp = np.sum((y_true == 0) & (preds == 1))
    n = len(y_true)
    return (tp / n) - (fp / n) * (threshold / (1 - threshold))

thresholds_dca = np.linspace(0.05, 0.50, 46)
dca_data = {"Threshold": thresholds_dca}

for name in ["Pathology RF", "Pathology LR", "Combined RF", "Clinical RF"]:
    dca_data[name] = [net_benefit(y_test, probs_all[name], t) for t in thresholds_dca]

# Always-treat line: NB = prevalence - threshold/(1-threshold) * (1-prevalence)
dca_data["Always treat"] = [
    prevalence - (t / (1 - t)) * (1 - prevalence) for t in thresholds_dca
]
# Never-treat line: NB = 0
dca_data["Never treat"] = [0.0] * len(thresholds_dca)

plt.figure(figsize=(9, 6))
colors_dca = {"Pathology RF": "#d62728", "Pathology LR": "#ff7f0e",
              "Combined RF": "#2ca02c", "Clinical RF": "#1f77b4",
              "Always treat": "gray", "Never treat": "black"}
for key, col in colors_dca.items():
    plt.plot(thresholds_dca, dca_data[key], label=key, color=col, lw=1.8)

plt.axhline(0, color="black", linestyle="--", lw=0.8)
plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
plt.title("Decision-Curve Analysis (Test Cohort, n=200, prevalence=20.8%)")
plt.legend(loc="lower right", frameon=True)
plt.tight_layout()
plt.savefig("results/figures/decision_curve_analysis.png", dpi=300)
plt.close()
print("Saved results/figures/decision_curve_analysis.png")

# ---- 9. Summary table for report ----------------------------------------
summary = res_df[res_df["Threshold"] == 0.5][
    ["Model", "AUC", "AUPRC", "Precision", "Recall", "Specificity", "F1", "Brier"]
].copy()
summary.to_csv("results/tables/report_summary_default_threshold.csv", index=False)
print("\nSaved results/tables/report_summary_default_threshold.csv")
print(summary.round(3).to_string(index=False))

# ---- 10. Random seed impact check (2 runs) ------------------------------
print("\n--- Random seed robustness check ---")
seeds = [2024, 42, 123]
seed_results = []
for seed in seeds:
    rf = RandomForestClassifier(n_estimators=200, max_depth=4, class_weight="balanced", random_state=seed)
    pipe = make_pipe(rf, [], pathology_vars)
    pipe.fit(train_df[pathology_vars], y_train)
    probs_s = pipe.predict_proba(test_df[pathology_vars])[:, 1]
    auc_s = roc_auc_score(y_test, probs_s)
    auprc_s = average_precision_score(y_test, probs_s)
    seed_results.append({"Seed": seed, "AUC": round(auc_s, 4), "AUPRC": round(auprc_s, 4)})
    print(f"  Seed={seed}: AUC={auc_s:.4f}, AUPRC={auprc_s:.4f}")

pd.DataFrame(seed_results).to_csv("results/tables/seed_robustness.csv", index=False)
print("Saved results/tables/seed_robustness.csv")

print("\n=== Phase 5 complete ===")
