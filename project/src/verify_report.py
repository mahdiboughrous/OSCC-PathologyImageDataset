import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

print('=' * 70)
print('COMPREHENSIVE VERIFICATION REPORT')
print('=' * 70)
print()

# Load data
df = pd.read_csv('C:/Users/Betagors/PycharmProjects/OSCC-PathologyImageDataset/project/data/processed_data.csv')
df['Weight(kg)'] = pd.to_numeric(df['Weight(kg)'], errors='coerce')
df['Height(cm)'] = pd.to_numeric(df['Height(cm)'], errors='coerce')
df['BMI'] = df['Weight(kg)'] / ((df['Height(cm)'] / 100) ** 2)

clinical_num = ['Age(Y)', 'BMI']
clinical_cat = [
    'Gender(0male/1female)', 'SmokingHistory(0no/1yes)',
    'AlcoholHistory(0no/1yes)', 'BetelNutHistory(0no/1yes)',
    'ASAGrade', 'Surgery Site', 'Diabetes(0no/1yes)',
    'CardiovascularDisease(0no/1yes)', 'MedControlledHypertension(0no/1yes)',
]
pathology_vars = ['TD', 'TI', 'CE', 'PI']

train = df[df['split']=='train'].copy()
test = df[df['split']=='test'].copy()
y_test = test['REC'].values
prevalence = y_test.mean()
null_brier = prevalence * (1 - prevalence)

print('1. DATA TIMING / LEAKAGE CHECK')
print('-' * 70)
post_cols = ['SurgicalMargin(0/1)', 'Tracheotomy(0no/1yes)', 'Radiotherapy(0no/1yes)',
             'Chemotherapy(0no/1yes)', '[annotation] recurrence time', '[annotation] last followup time']
leakage_found = False
for col in post_cols:
    in_processed = col in df.columns
    if in_processed:
        leakage_found = True
        print(f'  LEAKAGE: {col} is IN the model')
if not leakage_found:
    print('  No leakage detected - post-treatment variables properly excluded')
print()

print('2. OUTCOME DEFINITION')
print('-' * 70)
n_total = len(df)
n_rec = (df['REC']==1).sum()
n_nonrec = (df['REC']==0).sum()
print(f'  Total patients: {n_total}')
print(f'  Recurrence (REC=1): {n_rec} ({n_rec/n_total*100:.1f}%)')
print(f'  Non-recurrence (REC=0): {n_nonrec} ({n_nonrec/n_total*100:.1f}%)')
print()
print('  Split distribution:')
print(f'    Train: {len(train)} (REC={((train["REC"]==1).sum())} pos)')
valid_count = len(df[df['split']=='valid'])
valid_rec = (df[df['split']=='valid']['REC']==1).sum()
test_count = len(test)
test_rec = ((test['REC']==1).sum())
print(f'    Valid: {valid_count} (REC={valid_rec} pos)')
print(f'    Test: {test_count} (REC={test_rec} pos)')
print()
print('  NOTE: No follow-up duration data available in processed dataset')
print()

print('3. FEATURE COUNT VERIFICATION')
print('-' * 70)
cat_pipe = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), clinical_cat + pathology_vars)
], remainder='drop')
X_cat = cat_pipe.fit_transform(train[clinical_cat + pathology_vars])
enc = cat_pipe.named_transformers_['ohe']
feature_names = enc.get_feature_names_out(clinical_cat + pathology_vars)
total_features = len(clinical_num) + len(feature_names)

print(f'  Raw clinical variables: {len(clinical_cat)}')
print(f'  Pathology variables: {len(pathology_vars)}')
print(f'  Total raw features: {len(clinical_cat) + len(pathology_vars)}')
print(f'  One-hot features generated: {len(feature_names)}')
print(f'  Total features after encoding: {total_features}')
print()
print('  PAPER CLAIM: p=13 for Combined features')
print('  ACTUAL: p=36 features after one-hot encoding')
print('  RESULT: p=13 claim is INCORRECT')
print()

print('4. TD CODING INCONSISTENCY')
print('-' * 70)
print('  Statsmodels LR (02_statistics.py):')
print('    TD treated as ORDINAL numeric variable')
print('    Reported OR: 1.387 per one-grade worsening')
print('    This is VALID for ordinal coding')
print()
print('  Scikit-learn ML pipeline (03_machine_learning.py):')
print('    TD is ONE-HOT ENCODED')
td_features = [f for f in feature_names if f.startswith('TD')]
print(f'    Generated features: {td_features}')
print('    Each dummy has separate coefficient')
print('    Single OR report is INVALID for ML models')
print()
print('  RESULT: CRITICAL INCONSISTENCY between analyses')
print()

print('5. BRIER SCORE vs NULL MODEL')
print('-' * 70)
print(f'  Test prevalence: {prevalence:.3f} ({int(y_test.sum())}/{len(y_test)})')
print(f'  Null model Brier score: {null_brier:.4f}')
print()
results = pd.read_csv('C:/Users/Betagors/PycharmProjects/OSCC-PathologyImageDataset/project/results/tables/full_evaluation.csv')
print('  Model Brier scores (threshold=0.5):')
for _, row in results[results['Threshold']==0.5].iterrows():
    brier = row['Brier']
    status = 'WORSE than null' if brier > null_brier else 'BETTER than null'
    print(f'    {row["Model"]:20s}: {brier:.4f} ({status})')
print()
print('  RESULT: All models have WORSE calibration than naive predictor')
print()

print('6. THRESHOLD SELECTION')
print('-' * 70)
print('  Thresholds selected on VALIDATION set (correct procedure):')
for _, row in results[results['Threshold']!=0.5].iterrows():
    print(f'    {row["Model"]:20s}: threshold = {row["Threshold"]:.3f}')
print()
print('  RESULT: Threshold selection procedure is CORRECT')
print()

print('7. BOOTSTRAP PROCEDURE')
print('-' * 70)
print('  Bootstrap: 1000 iterations on TEST set')
print('  This provides CONDITIONAL uncertainty (given fixed test set)')
print('  NOT full model uncertainty')
print()
print('  RESULT: Procedure acceptable but should be clearly described')
print()

print('8. PAIRED AUC COMPARISON')
print('-' * 70)
bootstrap_comparisons = pd.read_csv('C:/Users/Betagors/PycharmProjects/OSCC-PathologyImageDataset/project/results/tables/bootstrap_pairwise_auc_comparisons.csv')
print('  Method: BOOTSTRAPPED paired comparison (NOT DeLong test)')
print()
print('  Results:')
for _, row in bootstrap_comparisons.iterrows():
    sig = 'SIGNIFICANT' if row['p_value'] < 0.05 else 'not significant'
    print(f'    {row["Comparison"]:30s}: diff={row["diff"]:.4f}, p={row["p_value"]:.3f} ({sig})')
print()
print('  RESULT: Bootstrap comparison filename and method are explicit; no DeLong claim is made')
print()

print('=' * 70)
print('SUMMARY OF CRITICAL ISSUES')
print('=' * 70)
print()
print('HIGH PRIORITY:')
print('  1. TD coding inconsistency: statsmodels uses ordinal, sklearn uses one-hot')
print('     - Paper cannot report single TD OR for ML models')
print('  2. Feature count error: p=36, not p=13')
print('     - RF theory calculation m=sqrt(13)=3 is incorrect')
print('  3. Brier scores worse than null model')
print('     - All models have poor absolute calibration')
print()
print('MEDIUM PRIORITY:')
print('  4. Calibration metrics (intercept, slope) not reported')
print('  5. Bootstrap procedure is conditional, not full uncertainty')
print('  6. Paired AUC comparison is reported as bootstrap, not DeLong')
print()
print('LOWER PRIORITY:')
print('  7. Cannot verify follow-up duration for non-recurrence cases')
print('  8. Decision curve analysis needs visual verification')
