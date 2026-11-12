# Oral Defense Guide (Strict 10-Minute Timing)
## Final Evaluation: Statistics & Data Analysis Modules (FST Béni Mellal — USMS)
**Session C01-2 (09:15 – 09:30) — Saturday, November 14, 2026**  
**Topic:** *From Clinical Characteristics to Surgical Pathology: Statistical and Machine-Learning Analysis of Recurrence Risk in Oral Squamous Cell Carcinoma*  
**Supervisors:** Prof. Aziz OUHINOU & Prof. Hicham MOUSSA  
**Team Members:** Yassin Oulhammous, Mahdi Boughrous, Yasmine Elasri, Mohamed Taha Benhassoun, Mustapha Amarzoug  

---

### Critical Examination Guidelines (Faculty Notice)
- **Strict Format:** 15 minutes total per team (10 to 12 minutes presentation $+ 3$ to 5 minutes jury Q&A).
- **Rule:** *“Aucun dépassement ne sera toléré afin de respecter le planning.”* (Zero overtime tolerated).
- **Language:** English presentation slides and speech.
- **Speakers:** Maximum 1 or 2 designated speakers.

---

## ⏱️ Slide-by-Slide Stopwatch Breakdown (Total: 10 min 00 s)

```
[00:00 - 00:45]  Slide 1  : Title, Authors, Affiliations & Institutional Scoping
[00:45 - 01:45]  Slide 2  : Clinical Context, Recurrence Burden & The Information Timeline Dilemma
[01:45 - 02:30]  Slide 3  : Three Research Questions & Methodological Flow (Inference vs ML)
[02:30 - 03:30]  Slide 4  : Multi-OSCC Cohort, Split Integrity & Data Leakage Prevention
[03:30 - 04:30]  Slide 5  : Bivariate Statistical Associations: Etiology vs Tumor Pathology
[04:30 - 05:30]  Slide 6  : Multivariable Logistic Regression & Forest Plot Analysis
[05:30 - 06:15]  Slide 7  : Machine Learning Architecture (Configurations & Balanced Weights)
[06:15 - 07:30]  Slide 8  : Held-Out Test Set Performance & ROC Curve Analysis
[07:30 - 08:30]  Slide 9  : Probabilistic Calibration (Brier Score) & Threshold Decision Dilemma
[08:30 - 09:15]  Slide 10 : Clinical Utility: Decision Curve Analysis (DCA & Net Benefit)
[09:15 - 09:45]  Slide 11 : Methodological Synthesis & Study Limitations
[09:45 - 10:15]  Slide 12 : Conclusion, Clinical Impact & Multimodal Perspectives (WSI)
[10:15 - 10:30]  Slide 13 : Acknowledgments & Opening of the Jury Q&A Session
------------------------------------------------------------------------------------------------
[RESERVE / Q&A]  Slide A1 : Backup - Detailed Team Contributions (CRediT Taxonomy)
[RESERVE / Q&A]  Slide A2 : Backup - Statistical Uncertainty, Bootstrap CIs & Seed Robustness
```

---

## 🎤 Word-for-Word Oral Presentation Script (English)

### Slide 1: Title & Research Team (00:00 - 00:45)
* **Spoken English:**  
  *“Distinguished members of the jury, professors, good morning. We are pleased to present our mini-project entitled: 'From Clinical Characteristics to Surgical Pathology: Statistical and Machine-Learning Analysis of Recurrence Risk in Oral Squamous Cell Carcinoma', conducted under the supervision of Professor Aziz OUHINOU and Professor Hicham MOUSSA for the Statistics and Data Analysis modules. Our research team brings together a multidisciplinary synergy across Artificial Intelligence, Biology, Organic Chemistry, and Mathematics.”*
* **Speaker Cue:** Stand straight, speak at a calm, deliberate pace, and acknowledge the committee.

---

### Slide 2: Clinical Context & The Temporal Information Problem (00:45 - 01:45)
* **Spoken English:**  
  *“Oral Squamous Cell Carcinoma (OSCC) accounts for over 90% of all oral malignancies worldwide, representing approximately 377,000 cases and 177,000 deaths annually. In surgically treated patients, tumor recurrence within two years is the paramount clinical endpoint, heavily dictating post-operative surveillance and patient mortality.*  
  *However, a frequent methodological pitfall in the literature is treating all medical variables as if they were available simultaneously. In clinical practice, demographic and lifestyle characteristics are known pre-operatively, whereas surgical histopathology—such as tumor differentiation, depth of invasion, or cancer emboli—is determined only after tumor resection. Our study explicitly models this temporal progression.”*

---

### Slide 3: Research Questions & Dual-Perspective Design (01:45 - 02:30)
* **Spoken English:**  
  *“To rigorously evaluate this prognostic continuum, we formulate three sequential scientific questions:*  
  *First: Which baseline clinical and lifestyle characteristics correlate with recurrence?*  
  *Second: Do surgical-pathology markers provide independent prognostic signal after multivariable adjustment for clinical factors?*  
  *Third: Can machine learning algorithms reliably generalize recurrence prediction to unseen, held-out patients?*  
  *Throughout this study, we maintain a strict distinction between statistical inference—which explains cohort associations—and machine learning prediction, which quantifies out-of-sample utility.”*

---

### Slide 4: Cohort Architecture & Leakage Prevention (02:30 - 03:30)
* **Spoken English:**  
  *“We investigated the Multi-OSCC benchmark cohort comprising 1,325 surgically treated patients, containing 275 recurrence events—representing an event prevalence of 20.75% and a 4-to-1 class imbalance.*  
  *To prevent subtle data leakage, all post-treatment variables—including surgical margin status, flap reconstruction, tracheostomy, and adjuvant chemoradiotherapy—were strictly excluded from predictors.*  
  *We adopted the official benchmark patient-level partition with zero patient ID overlap: 925 patients for training, 200 for validation and hyperparameter tuning, and a held-out test cohort of 200 patients containing 49 events. All preprocessing transformations—median imputation, scaling, and one-hot encoding—were fitted strictly on the training partition.”*

---

### Slide 5: Exploratory Statistical Associations (03:30 - 04:30)
* **Spoken English:**  
  *“Our bivariate statistical analysis revealed a striking, counter-intuitive clinical finding: classical etiologic exposures—namely smoking history, alcohol use, and betel-nut chewing—showed no statistically significant association with post-operative recurrence, with all p-values exceeding 0.45.*  
  *In oncological biology, this distinction is crucial: lifestyle carcinogens drive primary oncogenesis, but once a tumor is resected, recurrence is governed by intrinsic cellular aggressiveness rather than initial exposures.*  
  *In sharp contrast, all surgical pathology variables demonstrated strong unadjusted associations: Tumor Differentiation (p < 0.001), Tumor Invasion (p = 0.009), Cancer Embolus (p = 0.024), and Perineural Invasion (p = 0.003).”*

---

### Slide 6: Multivariable Logistic Regression & Forest Plot (04:30 - 05:30)
* **Spoken English:**  
  *“We fitted a multivariable logistic regression model on 1,317 complete cases. As illustrated in the Forest Plot, only one pathological factor retained statistically independent significance after controlling for all clinical and surgical covariates: Tumor Differentiation (TD).*  
  *Its adjusted odds ratio reached 1.387 [95% CI: 1.125 to 1.711, p = 0.0022], indicating that each single-grade worsening in histological differentiation increases the adjusted odds of recurrence by approximately 39%. Conversely, tumor invasion, vascular emboli, and perineural invasion attenuated toward the null once differentiation was accounted for.”*

---

### Slide 7: Machine Learning Pipeline Architecture (05:30 - 06:15)
* **Spoken English:**  
  *“To transition from explanatory inference to predictive modeling, we evaluated L2-regularized Logistic Regression and Random Forest classifiers across three distinct feature configurations:*  
  *Clinical-only (27 features post one-hot encoding), Pathology-only (9 features), and Combined (36 features).*  
  *To handle the 4-to-1 class imbalance without artificial synthetic oversampling, models incorporated balanced class weighting. Furthermore, decision thresholds were optimized strictly on the validation set via F1 maximization, leaving the test set completely untainted.”*

---

### Slide 8: Held-Out Test Set Performance & ROC Analysis (06:15 - 07:30)
* **Spoken English:**  
  *“On the strictly held-out test cohort of 200 patients, our findings are definitive: Clinical-only models demonstrated near-random discrimination, with ROC-AUC values between 0.534 and 0.537.*  
  *The Pathology-only Random Forest achieved the highest observed discrimination with an ROC-AUC of 0.627 [bootstrap 95% CI: 0.528 to 0.724].*  
  *Crucially, adding the 27 clinical features to pathology degraded the Random Forest ROC-AUC from 0.627 down to 0.557. This confirms that injecting noisy, non-informative clinical variables into an expanded feature space dilutes predictive signal in finite cohorts.”*

---

### Slide 9: Probabilistic Calibration & Threshold Dilemma (07:30 - 08:30)
* **Spoken English:**  
  *“Examining probabilistic reliability reveals an essential trade-off: all models exhibited Brier scores between 0.239 and 0.259, exceeding the null prevalence benchmark of 0.185. This occurs because balanced class weighting artificially inflates predicted probabilities, producing negative calibration intercepts around minus 1.1.*  
  *Moreover, applying the validation-optimal decision threshold of 0.453 increased test recall to 83.7%—capturing 41 out of 49 recurrences—but at the expense of generating 104 false alarms out of 151 healthy patients. Shifting thresholds alters the clinical trade-off between missed cases and over-investigation, without altering the underlying discrimination.”*

---

### Slide 10: Clinical Utility: Decision Curve Analysis (08:30 - 09:15)
* **Spoken English:**  
  *“To assess real-world clinical actionability, we conducted Decision Curve Analysis (DCA). As shown in the net benefit plot, the Pathology Random Forest provides a positive net benefit over default 'treat-all' and 'treat-none' strategies only within a narrow threshold probability window between 20% and 35%.*  
  *From a clinical perspective, this indicates that the model is appropriate for guiding intensified, non-invasive surveillance—such as shortened follow-up intervals or targeted MRI scans—but should not be used to justify toxic, aggressive adjuvant therapies.”*

---

### Slide 11: Methodological Discussion & Study Limitations (09:15 - 09:45)
* **Spoken English:**  
  *“In summary, this study provides three core lessons: First, initial etiologic factors diverge biologically from post-operative recurrence drivers. Second, statistically significant variables do not automatically yield high out-of-sample discrimination. And third, four coarse categorical pathology fields cannot fully capture the spatial architecture of the tumor microenvironment.*  
  *We acknowledge key limitations: the test set contains 49 recurrence events producing wide confidence intervals, HPV status was excluded due to 22.4% missingness, and the uncalibrated probability output requires post-hoc isotonic recalibration.”*

---

### Slide 12: Conclusion & Future Perspectives (09:45 - 10:15)
* **Spoken English:**  
  *“In conclusion, surgical pathology significantly outperforms pre-operative clinical features for recurrence risk stratification in OSCC, driven predominantly by tumor differentiation.*  
  *Moving forward, the most promising frontier is multimodal deep learning: fusing structured tabular clinical data with deep vision transformers operating directly on digitized Whole-Slide Histopathology Images (WSI), followed by multicentric prospective validation.”*

---

### Slide 13: Closing & Acknowledgments (10:15 - 10:30)
* **Spoken English:**  
  *“We express our sincere gratitude to our supervisors, Professor Aziz OUHINOU and Professor Hicham MOUSSA, and to the distinguished members of the jury for their time and attention. We are now delighted to answer your questions.”*

---

## 🎯 Strategic Answers to Anticipated Jury Questions

1. **Jury Question:** *“Why are major carcinogens like tobacco and alcohol non-significant for recurrence?”*  
   **Recommended Answer:**  
   *“We must distinguish primary carcinogenesis from post-resection prognosis. Tobacco and alcohol induce mucosal DNA damage leading to primary tumor inception. However, once the primary tumor is surgically excised, recurrence within two years is governed by residual microscopic tumor biology, cellular de-differentiation, and vascular invasiveness. Lifestyle habits no longer dictate short-term recurrence kinetics after resection.”*

2. **Jury Question:** *“Why did combining clinical variables with pathology reduce Random Forest performance from 0.627 to 0.557?”*  
   **Recommended Answer:**  
   *“This illustrates feature space dilation and noise injection. The 27 clinical features have near-zero predictive power (AUC ~ 0.53). Adding them quadrupled the feature dimensionality from 9 to 36, causing tree splits at random non-informative cutoffs and overfitting the training split under finite sample sizes.”*

3. **Jury Question:** *“How was the work distributed among team members?”*  
   $\implies$ **Action:** Switch to **Backup Slide A1**!  
   *“As detailed in our written report and displayed on this backup slide, our contributions follow the CRediT taxonomy aligned with our doctoral specialties: Yassin and Mahdi led the AI modeling, calibration, and reproducibility verification; Mohamed Taha provided biological and oncological interpretation; Mustapha reviewed chemical carcinogenesis and biochemical plausibility; and Yasmine formulated the statistical hypothesis testing and multivariable regressions.”*

4. **Jury Question:** *“Is the difference between Pathology and Combined models statistically significant?”*  
   $\implies$ **Action:** Switch to **Backup Slide A2**!  
   *“As audited on Backup Slide A2, we conducted paired bootstrap resampling over 1,000 iterations. The 95% confidence interval for the AUC difference spans from -0.018 to +0.160, which includes zero. Therefore, we scientifically emphasize that the superiority of the pathology model is an empirical observation rather than a statistically confirmed superiority.”*
