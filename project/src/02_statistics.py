import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# 1. Load data and clean missing markers
df = pd.read_csv("data/processed_data.csv")
df.replace("/", np.nan, inplace=True)

num_cols = ["Age(Y)", "Weight(kg)", "Height(cm)"]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["BMI"] = df["Weight(kg)"] / ((df["Height(cm)"] / 100) ** 2)

cat_cols = [
    "Gender(0male/1female)", "AlcoholHistory(0no/1yes)", "SmokingHistory(0no/1yes)",
    "BetelNutHistory(0no/1yes)", "ASAGrade", "HPV(0/1)", "Diabetes(0no/1yes)",
    "RespiratoryDisease(0no/1yes)", "CardiovascularDisease(0no/1yes)",
    "MedControlledHypertension(0no/1yes)", "TD", "TI", "CE", "PI",
]
for col in cat_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["REC"] = df["REC"].astype(int)
print(f"Loaded: {len(df)} patients, REC={((df['REC']==1).sum())} positive")

rec0, rec1 = df[df["REC"] == 0], df[df["REC"] == 1]
table1_rows = []

for col in ["Age(Y)", "Weight(kg)", "Height(cm)", "BMI"]:
    v0, v1 = rec0[col].dropna(), rec1[col].dropna()
    if len(v0) > 0 and len(v1) > 0:
        stat, p_val = stats.mannwhitneyu(v0, v1, alternative="two-sided")
    else:
        p_val = np.nan
    table1_rows.append({
        "Variable": col,
        "Overall": f"{df[col].median():.1f}",
        "REC=0": f"{v0.median():.1f} (n={len(v0)})",
        "REC=1": f"{v1.median():.1f} (n={len(v1)})",
        "Test": "Mann-Whitney U",
        "p-value": f"{p_val:.4f}"
    })

for col in cat_cols + ["Surgery Site"]:
    ct = pd.crosstab(df[col], df["REC"])
    if 0 not in ct.columns: ct[0] = 0
    if 1 not in ct.columns: ct[1] = 0
    chi2, p_val, dof, exp = stats.chi2_contingency(ct)
    for cat in ct.index:
        n_all = (df[col].notna() & (df[col] == cat)).sum()
        if n_all == 0: continue
        n0 = ct.loc[cat, 0]
        n1 = ct.loc[cat, 1]
        table1_rows.append({
            "Variable": f"{col}: {cat}",
            "Overall": f"{n_all}",
            "REC=0": f"{n0}",
            "REC=1": f"{n1}",
            "Test": "Chi-Square",
            "p-value": f"{p_val:.4f}"
        })

t1 = pd.DataFrame(table1_rows)
t1.to_csv("results/tables/table1_characteristics.csv", index=False)
print("Saved table1_characteristics.csv")
print(t1.head(12).to_string(index=False))

model_vars = [
    "Age(Y)", "Gender(0male/1female)", "SmokingHistory(0no/1yes)",
    "AlcoholHistory(0no/1yes)", "BetelNutHistory(0no/1yes)",
    "ASAGrade", "Diabetes(0no/1yes)", "CardiovascularDisease(0no/1yes)",
    "TD", "TI", "CE", "PI",
]
reg_df = df[model_vars + ["REC"]].dropna()
print(f"\nRegression on {len(reg_df)} complete cases")
X = sm.add_constant(reg_df[model_vars])
y = reg_df["REC"]
logit_model = sm.Logit(y, X).fit(disp=0)
results_df = pd.DataFrame({
    "OR": np.exp(logit_model.params),
    "2.5% CI": np.exp(logit_model.conf_int()[0]),
    "97.5% CI": np.exp(logit_model.conf_int()[1]),
    "p-value": logit_model.pvalues
})
results_df.to_csv("results/tables/multivariable_logistic_regression.csv", index=True)
print("Saved multivariable_logistic_regression.csv")
print(results_df.round(3).to_string())
