import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

# 1. Load Data
df = pd.read_csv('data/processed_data.csv')
df.replace('/', np.nan, inplace=True)

num_cols = ['Age(Y)', 'Weight(kg)', 'Height(cm)']
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['BMI'] = df['Weight(kg)'] / ((df['Height(cm)'] / 100) ** 2)

cat_cols = [
    'Gender(0male/1female)',
    'AlcoholHistory(0no/1yes)',
    'SmokingHistory(0no/1yes)',
    'BetelNutHistory(0no/1yes)',
    'ASAGrade',
    'Surgery Site',
    'Diabetes(0no/1yes)',
    'CardiovascularDisease(0no/1yes)',
    'MedControlledHypertension(0no/1yes)',
    'TD', 'TI', 'CE', 'PI',
]
for c in cat_cols:
    if c != 'Surgery Site':
        df[c] = pd.to_numeric(df[c], errors='coerce')
df['REC'] = df['REC'].astype(int)

clinical_num = ['Age(Y)', 'BMI']
clinical_cat = [
    'Gender(0male/1female)',
    'SmokingHistory(0no/1yes)',
    'AlcoholHistory(0no/1yes)',
    'BetelNutHistory(0no/1yes)',
    'ASAGrade',
    'Surgery Site',
    'Diabetes(0no/1yes)',
    'CardiovascularDisease(0no/1yes)',
    'MedControlledHypertension(0no/1yes)',
]
pathology_vars = ['TD', 'TI', 'CE', 'PI']

train_df = df[df['split'] == 'train']
test_df = df[df['split'] == 'test']

def build_pipeline(clf, num_vars, cat_vars):
    transformers = []
    if num_vars:
        transformers.append(('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scl', StandardScaler())]), num_vars))
    if cat_vars:
        transformers.append(('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_vars))
    return Pipeline([('prep', ColumnTransformer(transformers=transformers)), ('clf', clf)])

configs = {
    'Clinical LR': (clinical_num, clinical_cat, LogisticRegression(class_weight='balanced', max_iter=1000, random_state=2024)),
    'Clinical RF': (clinical_num, clinical_cat, RandomForestClassifier(n_estimators=200, max_depth=4, class_weight='balanced', random_state=2024)),
    'Pathology LR': ([], pathology_vars, LogisticRegression(class_weight='balanced', max_iter=1000, random_state=2024)),
    'Pathology RF': ([], pathology_vars, RandomForestClassifier(n_estimators=200, max_depth=4, class_weight='balanced', random_state=2024)),
    'Combined LR': (clinical_num, clinical_cat + pathology_vars, LogisticRegression(class_weight='balanced', max_iter=1000, random_state=2024)),
    'Combined RF': (clinical_num, clinical_cat + pathology_vars, RandomForestClassifier(n_estimators=200, max_depth=4, class_weight='balanced', random_state=2024)),
}

y_train = train_df['REC']
y_test = test_df['REC'].values
preds_dict = {}
ci_records = []

np.random.seed(2024)
for name, (n_f, c_f, model) in configs.items():
    pipe = build_pipeline(model, n_f, c_f)
    pipe.fit(train_df[n_f + c_f], y_train)
    probs = pipe.predict_proba(test_df[n_f + c_f])[:, 1]
    preds_dict[name] = probs

    boot_aucs = []
    for _ in range(1000):
        idx = np.random.choice(len(y_test), size=len(y_test), replace=True)
        if len(np.unique(y_test[idx])) > 1:
            boot_aucs.append(roc_auc_score(y_test[idx], probs[idx]))
    ci_low, ci_high = np.percentile(boot_aucs, [2.5, 97.5])
    point_auc = roc_auc_score(y_test, probs)
    ci_records.append({'Model Configuration': name, 'ROC-AUC': point_auc, '95% CI': f'[{ci_low:.3f} - {ci_high:.3f}]'})

ci_df = pd.DataFrame(ci_records)
ci_df.to_csv('results/tables/auc_confidence_intervals.csv', index=False)
print('Saved results/tables/auc_confidence_intervals.csv')
print(ci_df.to_string(index=False))

# 2. Figure 1: ROC Curves
plt.figure(figsize=(8, 6))
colors = ['#7f7f7f', '#bcbd22', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for (name, probs), col in zip(preds_dict.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', color=col, lw=1.8)
plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.500)', lw=1)
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curves by Input Modality on Test Cohort (n=200)')
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.savefig('results/figures/roc_curves.png', dpi=300)
plt.close()

# 3. Figure 2: Calibration Curves
plt.figure(figsize=(8, 6))
for name in ['Clinical RF', 'Pathology RF', 'Combined RF']:
    prob_true, prob_pred = calibration_curve(y_test, preds_dict[name], n_bins=5, strategy='uniform')
    plt.plot(prob_pred, prob_true, marker='o', label=name, lw=1.8)
plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', lw=1)
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Observed Recurrence Fraction')
plt.title('Calibration Curves (Random Forest Models)')
plt.legend(loc='upper left', frameon=True)
plt.tight_layout()
plt.savefig('results/figures/calibration_curves.png', dpi=300)
plt.close()

# 4. Figure 3: Forest Plot
mlr = pd.read_csv('results/tables/multivariable_logistic_regression.csv')
mlr = mlr[mlr['Unnamed: 0'] != 'const'].copy()
mlr.sort_values(by='OR', inplace=True)
plt.figure(figsize=(9, 6))
y_pos = np.arange(len(mlr))
plt.errorbar(mlr['OR'], y_pos, xerr=[mlr['OR'] - mlr['2.5% CI'], mlr['97.5% CI'] - mlr['OR']], fmt='o', color='#1f77b4', ecolor='#1f77b4', elinewidth=1.8, capsize=4)
plt.axvline(1.0, color='gray', linestyle='--', lw=1)
plt.yticks(y_pos, mlr['Unnamed: 0'])
plt.xlabel('Adjusted Odds Ratio (95% CI)')
plt.title('Multivariable Logistic Regression Predictors of Recurrence')
plt.tight_layout()
plt.savefig('results/figures/forest_plot_odds_ratios.png', dpi=300)
plt.close()
print('All Phase 4 figures generated in results/figures/.')
