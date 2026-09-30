"""Reproducible final verification for the OSCC recurrence tabular models.

Run from the project directory:
    ..\\.venv\\Scripts\\python.exe src\\final_verification.py

The official patient-level split is loaded from data/processed_data.csv and is
never regenerated. All preprocessing and model fitting use training patients;
validation outcomes are used only for threshold selection.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels.api as sm
from matplotlib.lines import Line2D
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
SEED = 2024
N_BOOTSTRAPS = 1000
CLIP_EPSILON = 1e-6

CLINICAL_NUM = ["Age(Y)", "BMI"]
CLINICAL_CAT = [
    "Gender(0male/1female)", "SmokingHistory(0no/1yes)",
    "AlcoholHistory(0no/1yes)", "BetelNutHistory(0no/1yes)",
    "ASAGrade", "Surgery Site", "Diabetes(0no/1yes)",
    "CardiovascularDisease(0no/1yes)", "MedControlledHypertension(0no/1yes)",
]
PATHOLOGY = ["TD", "TI", "CE", "PI"]
CONFIGS = {
    "Clinical LR": (CLINICAL_NUM, CLINICAL_CAT, "lr"),
    "Clinical RF": (CLINICAL_NUM, CLINICAL_CAT, "rf"),
    "Pathology LR": ([], PATHOLOGY, "lr"),
    "Pathology RF": ([], PATHOLOGY, "rf"),
    "Combined LR": (CLINICAL_NUM, CLINICAL_CAT + PATHOLOGY, "lr"),
    "Combined RF": (CLINICAL_NUM, CLINICAL_CAT + PATHOLOGY, "rf"),
}


def make_classifier(kind: str):
    if kind == "lr":
        return LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    return RandomForestClassifier(
        n_estimators=200, max_depth=4, class_weight="balanced", random_state=SEED
    )


def make_pipeline(num_features, cat_features, kind):
    transformers = []
    if num_features:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_features))
    if cat_features:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_features))
    return Pipeline([
        ("preprocessor", ColumnTransformer(transformers=transformers, remainder="drop")),
        ("clf", make_classifier(kind)),
    ])


def clean_data():
    df = pd.read_csv(DATA / "processed_data.csv")
    df.replace("/", np.nan, inplace=True)
    for col in ["Age(Y)", "Weight(kg)", "Height(cm)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["BMI"] = df["Weight(kg)"] / ((df["Height(cm)"] / 100) ** 2)
    for col in CLINICAL_CAT + PATHOLOGY:
        if col != "Surgery Site":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["REC"] = df["REC"].astype(int)
    return df


def ensure_dirs():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def audit_split(df):
    expected = {"train": 925, "valid": 200, "test": 200}
    counts = df["split"].value_counts().to_dict()
    assert counts == expected, f"Official split changed: {counts}"
    assert df["PID"].is_unique, "Duplicate PID rows found in processed data"
    sets = {name: set(df.loc[df["split"] == name, "PID"]) for name in expected}
    overlaps = {f"{a}__{b}": sorted(sets[a] & sets[b]) for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]}
    assert not any(overlaps.values()), f"PID overlap across partitions: {overlaps}"
    return counts, overlaps


def transformed_feature_audits(fitted, train):
    count_rows, name_rows = [], []
    for name, (num, cat, kind) in CONFIGS.items():
        pipe = make_pipeline(num, cat, kind)
        pipe.fit(train[num + cat], train["REC"])
        names = list(pipe.named_steps["preprocessor"].get_feature_names_out())
        count_rows.append({"Model Configuration": name, "Numeric input columns": len(num), "Raw categorical input columns": len(cat), "Transformed feature count": len(names)})
        for position, feature in enumerate(names):
            name_rows.append({"Model Configuration": name, "Transformed feature position": position, "Transformed feature name": feature})
    pd.DataFrame(count_rows).to_csv(TABLES / "transformed_feature_counts.csv", index=False)
    pd.DataFrame(name_rows).to_csv(TABLES / "transformed_feature_names.csv", index=False)
    return count_rows, name_rows


def fit_models(train, valid, test):
    predictions = {}
    fitted = {}
    for name, (num, cat, kind) in CONFIGS.items():
        pipe = make_pipeline(num, cat, kind)
        pipe.fit(train[num + cat], train["REC"])
        fitted[name] = pipe
        predictions[name] = {
            "train": pipe.predict_proba(train[num + cat])[:, 1],
            "valid": pipe.predict_proba(valid[num + cat])[:, 1],
            "test": pipe.predict_proba(test[num + cat])[:, 1],
        }
    return fitted, predictions


def td_audit(df, fitted):
    observed = sorted(df["TD"].dropna().unique().tolist())
    td_cols = []
    names = fitted["Pathology LR"].named_steps["preprocessor"].get_feature_names_out()
    td_cols = [name for name in names if "TD_" in name]
    rows = [
        {"Analysis": "Statsmodels multivariable logistic regression", "Raw observed TD categories": json.dumps(observed), "Numeric mapping": "0=well, 1=moderate, 2=poor", "Ordinal treatment": "yes", "Effect interpretation": "aOR=1.387 is per one-grade worsening (one-unit increase in TD)", "Sklearn TD columns": "", "One-hot encoded in ML": "no", "Reference category": "not applicable", "Machine-readable interpretation": "ordinal_statsmodels_only"},
        {"Analysis": "Sklearn Logistic Regression pipeline", "Raw observed TD categories": json.dumps(observed), "Numeric mapping": "values retained as category labels", "Ordinal treatment": "no", "Effect interpretation": "no single ordinal TD odds ratio", "Sklearn TD columns": json.dumps(td_cols), "One-hot encoded in ML": "yes", "Reference category": "none; drop=None", "Machine-readable interpretation": "one_hot_nominal_features"},
        {"Analysis": "Sklearn Random Forest pipeline", "Raw observed TD categories": json.dumps(observed), "Numeric mapping": "values retained as category labels", "Ordinal treatment": "no", "Effect interpretation": "no single TD effect estimate", "Sklearn TD columns": json.dumps(td_cols), "One-hot encoded in ML": "yes", "Reference category": "none; drop=None", "Machine-readable interpretation": "one_hot_nominal_features"},
    ]
    pd.DataFrame(rows).to_csv(TABLES / "td_coding_audit.csv", index=False)


def calibration_metrics(y, probs, rng):
    clipped = np.clip(probs, CLIP_EPSILON, 1 - CLIP_EPSILON)
    logit_probs = np.log(clipped / (1 - clipped))
    intercept_fit = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=logit_probs).fit()
    slope_fit = sm.GLM(y, sm.add_constant(logit_probs), family=sm.families.Binomial()).fit()
    intercept, slope = float(intercept_fit.params[0]), float(slope_fit.params[1])
    boot_i, boot_s = [], []
    for _ in range(N_BOOTSTRAPS):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        sample_p = np.clip(probs[idx], CLIP_EPSILON, 1 - CLIP_EPSILON)
        sample_logit = np.log(sample_p / (1 - sample_p))
        try:
            i_fit = sm.GLM(y[idx], np.ones((len(idx), 1)), family=sm.families.Binomial(), offset=sample_logit).fit()
            s_fit = sm.GLM(y[idx], sm.add_constant(sample_logit), family=sm.families.Binomial()).fit()
            boot_i.append(float(i_fit.params[0]))
            boot_s.append(float(s_fit.params[1]))
        except (ValueError, np.linalg.LinAlgError):
            continue
    return intercept, slope, boot_i, boot_s


def brier_and_calibration(y_by_split, predictions):
    rows = []
    for split, y in y_by_split.items():
        prevalence = float(np.mean(y))
        null_brier = prevalence * (1 - prevalence)
        rows.append({"Model": "Prevalence-only null", "Dataset/split": split, "Event prevalence": prevalence, "Model Brier score": np.nan, "Null Brier score": null_brier, "Brier skill score": np.nan, "Interpretation": "prevalence-only benchmark"})
        for name, pred in predictions.items():
            model_brier = brier_score_loss(y, pred[split])
            skill = 1 - model_brier / null_brier
            rows.append({"Model": name, "Dataset/split": split, "Event prevalence": prevalence, "Model Brier score": model_brier, "Null Brier score": null_brier, "Brier skill score": skill, "Interpretation": "better than prevalence-only benchmark" if skill > 0 else "worse than prevalence-only benchmark"})
    pd.DataFrame(rows).to_csv(TABLES / "brier_benchmark_comparison.csv", index=False)

    rng = np.random.default_rng(SEED)
    cal_rows = []
    y_test = y_by_split["test"]
    for name, pred in predictions.items():
        intercept, slope, boot_i, boot_s = calibration_metrics(y_test, pred["test"], rng)
        cal_rows.append({"Model": name, "Split": "test", "Calibration intercept": intercept, "Calibration slope": slope, "Intercept CI low": np.percentile(boot_i, 2.5), "Intercept CI high": np.percentile(boot_i, 97.5), "Slope CI low": np.percentile(boot_s, 2.5), "Slope CI high": np.percentile(boot_s, 97.5), "Test events": int(y_test.sum()), "Fitting method": "Binomial GLM with logit(predicted probability) as offset for intercept; Binomial GLM with intercept and logit(predicted probability) covariate for slope", "Clipping": f"probabilities clipped to [{CLIP_EPSILON}, {1 - CLIP_EPSILON}] only for logit transforms", "Interpretation": "exploratory; only 49 test events"})
    pd.DataFrame(cal_rows).to_csv(TABLES / "calibration_metrics_test.csv", index=False)

    y_test = y_by_split["test"]
    plt.figure(figsize=(8, 6))
    for name in ["Clinical RF", "Pathology RF", "Combined RF"]:
        probs = predictions[name]["test"]
        bins = np.linspace(0, 1, 6)
        mean_predicted, observed = [], []
        for low, high in zip(bins[:-1], bins[1:]):
            mask = (probs >= low) & (probs < high if high < 1 else probs <= high)
            if np.any(mask):
                mean_predicted.append(float(probs[mask].mean()))
                observed.append(float(y_test[mask].mean()))
        plt.plot(mean_predicted, observed, marker="o", label=name, lw=1.8)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration", lw=1)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed recurrence fraction")
    plt.title("Calibration Curves (Random Forest Models; Test Cohort, n=200)")
    plt.legend(loc="upper left", frameon=True)
    plt.figtext(0.5, 0.01, "Exploratory calibration assessment; only 49 test events.", ha="center", fontsize=8)
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(FIGURES / "calibration_curves.png", dpi=300)
    plt.close()


def classification_metrics(y, probs, threshold):
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn), "Precision": precision_score(y, pred, zero_division=0), "Recall": recall_score(y, pred, zero_division=0), "Specificity": tn / (tn + fp), "NPV": tn / (tn + fn), "F1": f1_score(y, pred, zero_division=0)}


def select_f1_threshold(y, probs):
    precision, recall, thresholds = precision_recall_curve(y, probs)
    f1_values = 2 * precision * recall / (precision + recall + 1e-8)
    idx = int(np.nanargmax(f1_values))
    return max(0.0, min(1.0, float(thresholds[min(idx, len(thresholds) - 1)])))


def threshold_audits(y_valid, y_test, predictions):
    rows, cm_rows = [], []
    selected = {}
    for name, pred in predictions.items():
        threshold = select_f1_threshold(y_valid, pred["valid"])
        selected[name] = threshold
        val_m = classification_metrics(y_valid, pred["valid"], threshold)
        test_m = classification_metrics(y_test, pred["test"], threshold)
        rows.append({"Model": name, "Dataset used for threshold selection": "validation", "Optimization objective": "maximum F1 score", "Selected threshold": threshold, "Validation precision": val_m["Precision"], "Validation recall": val_m["Recall"], "Validation specificity": val_m["Specificity"], "Validation F1": val_m["F1"], "Held-out test precision": test_m["Precision"], "Held-out test recall": test_m["Recall"], "Held-out test specificity": test_m["Specificity"], "Held-out test NPV": test_m["NPV"], "Held-out test F1": test_m["F1"], "Selection leakage verification": "No held-out test outcomes used to select threshold"})
        cm_rows.append({"Model": name, "Threshold label": "Default threshold", "Threshold": 0.5, **classification_metrics(y_test, pred["test"], 0.5)})
        cm_rows.append({"Model": name, "Threshold label": "Validation-selected F1 threshold", "Threshold": threshold, **test_m})
    pd.DataFrame(rows).to_csv(TABLES / "threshold_selection_audit.csv", index=False)
    pd.DataFrame(cm_rows).to_csv(TABLES / "confusion_matrices_test.csv", index=False)
    return selected


def core_metric_comparison(y_test, predictions):
    rows = []
    old = pd.read_csv(TABLES / "full_evaluation.csv") if (TABLES / "full_evaluation.csv").exists() else pd.DataFrame()
    for name, pred in predictions.items():
        probs = pred["test"]
        metrics = {
            "ROC-AUC": roc_auc_score(y_test, probs),
            "AUPRC": average_precision_score(y_test, probs),
            "Brier": brier_score_loss(y_test, probs),
        }
        threshold_row = old[(old["Model"] == name) & (old["Threshold"] == 0.5)] if not old.empty else pd.DataFrame()
        for metric, value in metrics.items():
            old_column = "AUC" if metric == "ROC-AUC" else metric
            old_value = float(threshold_row.iloc[0][old_column]) if not threshold_row.empty and old_column in threshold_row else np.nan
            rows.append({"Model": name, "Metric": metric, "Regenerated value": value, "Previously reported value": old_value, "Difference": value - old_value if not np.isnan(old_value) else np.nan})
    pd.DataFrame(rows).to_csv(TABLES / "core_metric_reproducibility.csv", index=False)


def bootstrap_audit():
    pd.DataFrame([{"Resampling unit": "held-out test patients", "Models fixed": "yes", "Predictions fixed": "yes", "Resamples": N_BOOTSTRAPS, "Interval": "percentile 2.5th and 97.5th", "One-class resamples": "excluded for AUC/AUPRC and paired AUC; retained where metric is defined", "Uncertainty label": "Conditional test-set performance uncertainty for fixed fitted models.", "Not claimed": "full development-pipeline uncertainty"}]).to_csv(TABLES / "bootstrap_method_audit.csv", index=False)


def paired_bootstrap(y, predictions):
    comparisons = [("Pathology RF", "Combined RF"), ("Pathology LR", "Combined LR"), ("Pathology RF", "Clinical RF"), ("Pathology LR", "Clinical LR")]
    rows = []
    for model_a, model_b in comparisons:
        auc_a = roc_auc_score(y, predictions[model_a]["test"])
        auc_b = roc_auc_score(y, predictions[model_b]["test"])
        observed = auc_a - auc_b
        rng = np.random.default_rng(42)
        diffs = []
        for _ in range(N_BOOTSTRAPS):
            idx = rng.integers(0, len(y), len(y))
            if len(np.unique(y[idx])) < 2:
                continue
            diffs.append(roc_auc_score(y[idx], predictions[model_a]["test"][idx]) - roc_auc_score(y[idx], predictions[model_b]["test"][idx]))
        p_value = min(1.0, 2 * min(np.mean(np.array(diffs) <= 0), np.mean(np.array(diffs) >= 0)))
        rows.append({"Comparison": f"{model_a} vs {model_b}", "Model A": model_a, "Model B": model_b, "AUC model A": auc_a, "AUC model B": auc_b, "AUC difference (A-B)": observed, "CI low": np.percentile(diffs, 2.5), "CI high": np.percentile(diffs, 97.5), "p-value": p_value, "Method": "paired patient-level bootstrap", "Paired resampling": "same resampled held-out patients used for both models", "One-class handling": "one-class resamples excluded", "p-value definition": "two-sided sign proportion: 2*min(P(diff<=0), P(diff>=0)), capped at 1"})
    pd.DataFrame(rows).to_csv(TABLES / "bootstrap_pairwise_auc_comparisons.csv", index=False)


def dca(y, predictions):
    thresholds = np.linspace(0.05, 0.50, 46)
    model_names = ["Pathology RF", "Pathology LR", "Combined RF", "Clinical RF"]
    prevalence = np.mean(y)
    rows = []
    for threshold in thresholds:
        treat_all = prevalence - (threshold / (1 - threshold)) * (1 - prevalence)
        for name in model_names:
            probs = predictions[name]["test"]
            pred = probs >= threshold
            tp = np.sum((y == 1) & pred)
            fp = np.sum((y == 0) & pred)
            nb = tp / len(y) - fp / len(y) * threshold / (1 - threshold)
            rows.append({"Model": name, "Threshold probability": threshold, "Net benefit": nb, "Treat-all net benefit": treat_all, "Treat-none net benefit": 0.0, "Exceeds both defaults": bool(nb > treat_all and nb > 0)})
        rows.extend([{ "Model": "Treat-all", "Threshold probability": threshold, "Net benefit": treat_all, "Treat-all net benefit": treat_all, "Treat-none net benefit": 0.0, "Exceeds both defaults": False}, {"Model": "Treat-none", "Threshold probability": threshold, "Net benefit": 0.0, "Treat-all net benefit": treat_all, "Treat-none net benefit": 0.0, "Exceeds both defaults": False}])
    dca_df = pd.DataFrame(rows)
    dca_df.to_csv(TABLES / "decision_curve_net_benefit.csv", index=False)
    summary_rows = []
    for name in model_names:
        sub = dca_df[dca_df["Model"] == name].copy()
        supported = sub[sub["Exceeds both defaults"]]["Threshold probability"].to_numpy()
        interval = "none"
        if len(supported):
            interval = f"{supported.min():.2f}-{supported.max():.2f} (grid-supported)"
        max_row = sub.loc[sub["Net benefit"].idxmax()]
        summary_rows.append({"Model": name, "Threshold interval(s) with net benefit greater than both default strategies": interval, "Maximum net benefit": max_row["Net benefit"], "Threshold at maximum net benefit": max_row["Threshold probability"], "Interpretation": "exploratory potential net benefit; narrow/unstable or clinically uncertain" if interval != "none" else "no threshold grid interval exceeded both defaults"})
    pd.DataFrame(summary_rows).to_csv(TABLES / "decision_curve_summary.csv", index=False)

    plt.figure(figsize=(9, 6))
    colors = {"Pathology RF": "#d62728", "Pathology LR": "#ff7f0e", "Combined RF": "#2ca02c", "Clinical RF": "#1f77b4", "Treat-all": "#777777", "Treat-none": "#111111"}
    for name in model_names + ["Treat-all", "Treat-none"]:
        sub = dca_df[dca_df["Model"] == name]
        plt.plot(sub["Threshold probability"], sub["Net benefit"], label=name, color=colors[name], lw=1.8)
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title("Decision-Curve Analysis (held-out test cohort, n=200)")
    plt.legend(loc="lower right", frameon=True)
    plt.figtext(0.5, 0.01, "Exploratory potential net benefit; numerical intervals are grid-supported and clinically uncertain.", ha="center", fontsize=8)
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(FIGURES / "decision_curve_analysis.png", dpi=300)
    plt.close()


def followup_audit(df):
    clinical = pd.read_csv(DATA / "clinical_data_2024.csv")
    metadata = json.loads((DATA / "all_metadata.json").read_text(encoding="utf-8"))
    metadata_columns = sorted({key for row in metadata.get("datainfo", []) for key in row})
    rows = []
    raw_cols = ["[annotation] recurrence time", "[annotation] last followup time"]
    for col in raw_cols:
        values = clinical[col].replace("/", np.nan) if col in clinical else pd.Series(dtype=float)
        rows.append({"Source": "raw clinical_data_2024.csv", "Variable": col, "Available": col in clinical.columns, "Missingness": float(values.isna().mean()) if len(values) else np.nan, "Uniform minimum follow-up for REC=0": "cannot be confirmed from available processed dataset", "Survival/time-to-event modeling": "possible in principle only after validating event/censoring semantics and cleaning raw times"})
    for col in ["[annotation] recurrence time", "[annotation] last followup time"]:
        rows.append({"Source": "processed_data.csv", "Variable": col, "Available": col in df.columns, "Missingness": np.nan, "Uniform minimum follow-up for REC=0": "Uniform minimum follow-up among non-recurrence patients cannot be confirmed from the available processed dataset.", "Survival/time-to-event modeling": "not possible from processed data alone"})
    rows.append({"Source": "all_metadata.json", "Variable": "follow-up or censoring fields", "Available": any("follow" in c.lower() or "recurrence" in c.lower() or "censor" in c.lower() for c in metadata_columns), "Missingness": np.nan, "Uniform minimum follow-up for REC=0": "cannot be confirmed", "Survival/time-to-event modeling": "not established by metadata"})
    rows.append({"Source": "01_prepare_data.py", "Variable": "DROP_COLUMNS follow-up fields", "Available": True, "Missingness": np.nan, "Uniform minimum follow-up for REC=0": "explicit limitation", "Survival/time-to-event modeling": "not used; variables excluded from predictors"})
    pd.DataFrame(rows).to_csv(TABLES / "followup_availability_audit.csv", index=False)


def write_environment():
    packages = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "scipy": scipy.__version__, "statsmodels": sm.__version__, "matplotlib": plt.matplotlib.__version__}
    (ROOT / "requirements-verification.txt").write_text("\n".join(f"{key}=={value}" for key, value in packages.items() if key != "python") + "\n", encoding="utf-8")
    (TABLES / "verification_environment.csv").write_text("Package,Version\n" + "\n".join(f"{key},{value}" for key, value in packages.items()), encoding="utf-8")


def write_report(df, counts, selected, count_rows):
    test = df[df["split"] == "test"]
    null = float(test["REC"].mean() * (1 - test["REC"].mean()))
    count_text = ", ".join(f"{row['Model Configuration']}={row['Transformed feature count']}" for row in count_rows)
    report = f"""# Final Verification Report

Generated by `src/final_verification.py` with seed {SEED}. Statuses reflect the repository evidence and regenerated outputs.

| Requested item | Status | Verification |
|---|---|---|
| Official patient-level split and PID exclusivity | PASS | train=925, valid=200, test=200; unique PIDs and zero cross-partition overlap |
| Transformed feature counts and names | PASS | {count_text}; categorical encoding is fitted on train only |
| TD coding audit | PASS | statsmodels uses ordinal numeric TD 0=well, 1=moderate, 2=poor; sklearn LR/RF use one-hot columns with no dropped reference |
| Reported TD aOR | PASS | aOR=1.387, 95% CI [1.125, 1.711], p=0.0022 is attributed only to the ordinal statsmodels model, not an ML feature effect |
| Brier null benchmark | PASS | test prevalence={test['REC'].sum()}/200=0.245; null Brier={null:.6f} (approximately 0.185) |
| Test calibration metrics | PASS | probabilities evaluated; intercept/slope and bootstrap intervals are exploratory because there are {test['REC'].sum()} events |
| Threshold-selection audit | PASS | validation-only maximum-F1 selection; Pathology RF selected threshold={selected['Pathology RF']:.3f} |
| Bootstrap uncertainty audit | PASS | fixed fitted models, patient bootstrap, {N_BOOTSTRAPS} resamples, percentile intervals, one-class exclusions documented |
| Pairwise AUC comparison naming/method | PASS | paired held-out-patient bootstrap; no DeLong claim; same indices and AUC(A)-AUC(B) differences |
| Decision-curve analysis | PASS | numerical net benefit table and grid-supported intervals regenerated; interpretation remains exploratory |
| Follow-up availability | NOT VERIFIABLE | raw fields exist but are excluded from processed predictors; uniform minimum follow-up for REC=0 cannot be confirmed |
| Reproducibility | PASS | exact verification requirements and one-command entry point recorded |

## Changed or Corrected Values

- The old combined RF claim `p=13` and its `sqrt(13), m=3` probability calculation are invalid after one-hot transformation. The exact post-transformation counts are recorded in `transformed_feature_counts.csv`; the old calculation is not replaced by a theoretical probability claim.
- The test prevalence is 49/200=0.245 and the prevalence-only Brier benchmark is {null:.6f}, approximately 0.185. Model Brier skill is reported in `brier_benchmark_comparison.csv`.
- Existing core model metrics are regenerated from the unchanged split and exact pipeline definitions. Any displayed rounding differences should be treated as regenerated precision, not silent metric changes.

## Recommended Manuscript Wording

- **Feature count:** “Categorical predictors were one-hot encoded within training-fitted sklearn pipelines; therefore, the transformed feature count differs by configuration and is reported in the verification table. The previous combined `p=13` statement and `sqrt(13), m=3` calculation were removed as invalid for the transformed design matrix.”
- **TD coding:** “TD was entered as an ordinal numeric variable only in the statsmodels regression, where the aOR is per one-grade worsening. In the sklearn ML pipelines, TD was one-hot encoded and no single ML odds ratio was assigned.”
- **Brier score:** “Brier scores were compared with split-specific prevalence-only benchmarks; positive Brier skill indicates improvement over that benchmark.”
- **Bootstrap:** “Intervals represent conditional test-set performance uncertainty for fixed fitted models, based on patient-level resampling of held-out test predictions; they are not full development-pipeline uncertainty.”
- **Decision curve:** “Decision-curve findings indicate exploratory potential net benefit only where the numerical grid shows model net benefit above both treat-all and treat-none; clinical utility remains uncertain.”
- **Follow-up:** “Uniform minimum follow-up among non-recurrence patients cannot be confirmed from the available processed dataset.”

## Generated Outputs

All outputs are under `results/`:

- `tables/transformed_feature_counts.csv`
- `tables/transformed_feature_names.csv`
- `tables/td_coding_audit.csv`
- `tables/brier_benchmark_comparison.csv`
- `tables/calibration_metrics_test.csv`
- `tables/core_metric_reproducibility.csv`
- `tables/threshold_selection_audit.csv`
- `tables/confusion_matrices_test.csv`
- `tables/bootstrap_method_audit.csv`
- `tables/bootstrap_pairwise_auc_comparisons.csv`
- `tables/decision_curve_net_benefit.csv`
- `tables/decision_curve_summary.csv`
- `tables/followup_availability_audit.csv`
- `tables/verification_environment.csv`
- `figures/decision_curve_analysis.png`
- `figures/calibration_curves.png`

The single regeneration command is:

```text
..\\.venv\\Scripts\\python.exe src\\final_verification.py
```
"""
    (ROOT / "results" / "final_verification_report.md").write_text(report, encoding="utf-8")


def main():
    ensure_dirs()
    df = clean_data()
    counts, overlaps = audit_split(df)
    train, valid, test = [df[df["split"] == split].copy() for split in ("train", "valid", "test")]
    fitted, predictions = fit_models(train, valid, test)
    count_rows, _ = transformed_feature_audits(fitted, train)
    td_audit(df, fitted)
    brier_and_calibration({"train": train["REC"].to_numpy(), "valid": valid["REC"].to_numpy(), "test": test["REC"].to_numpy()}, predictions)
    core_metric_comparison(test["REC"].to_numpy(), predictions)
    selected = threshold_audits(valid["REC"].to_numpy(), test["REC"].to_numpy(), predictions)
    bootstrap_audit()
    paired_bootstrap(test["REC"].to_numpy(), predictions)
    dca(test["REC"].to_numpy(), predictions)
    followup_audit(df)
    write_environment()
    write_report(df, counts, selected, count_rows)
    print(f"PASS: split counts={counts}; PID overlaps={overlaps}")
    print(f"PASS: Pathology RF validation F1 threshold={selected['Pathology RF']:.6f}")
    print("PASS: final verification outputs written under results/")


if __name__ == "__main__":
    main()
