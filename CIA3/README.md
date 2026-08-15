# CIA-3: ML for Social Good Ensemble Challenge
## Emergency Ambulance Call Triage / Prioritization (Health)

**Author:** Arnav (2547115)

An end-to-end, reproducible ML pipeline that scores incoming emergency-dispatch phone calls with an
independent urgency/triage level — **Critical / Urgent / Non-urgent** — using only information a
phone dispatcher could realistically obtain from a caller, with no vitals-monitor access. Built on
the KTAS (Korean Triage and Acuity Scale) dataset.

See `2547115_CIA3.ipynb` for the full, executed notebook (all five rubric sections, all outputs
generated from real runs — nothing fabricated).

---

## Problem summary

When several emergency calls arrive at once and ambulances are limited, dispatchers must rank
callers using only phone-call information. This project trains a model that assigns each call an
independent urgency score from caller-obtainable signals (chief complaint, self-reported pain,
mental status, age, sex, arrival mode, injury flag, caller relationship) — explicitly **excluding**
device-measured vitals (blood pressure, heart rate, SpO2, respiratory rate, temperature), which a
phone dispatcher never has access to. A secondary ablation experiment trains the same models *with*
vitals included, to measure the information gap this constraint costs.

Full impact framing, beneficiaries, target definition, and responsible-use limitations are in
**Section 1 of the notebook**.

## Dataset

- **Source:** Kaggle, "Korean Triage and Acuity Scale (KTAS)", uploaded by **ilkeryildiz**:
  `https://www.kaggle.com/datasets/ilkeryildiz/ktas`
- **Local copy:** `data.csv` in this folder (semicolon-delimited, `cp1252` encoding — see notes
  below). Verified in Section 2 of the notebook to match the public dataset: **1,267 rows, 24
  columns**, `KTAS_expert` target distribution of 26 / 220 / 487 / 459 / 75 across KTAS levels 1–5.
- **Underlying data:** anonymized ED triage records from two academic hospitals in South Korea
  (2016–2017), independently re-graded by a panel of triage experts.
- **Known data-quality quirks** (documented and handled in Section 2 of the notebook):
  - `"??"` placeholder strings in vitals columns (SBP/DBP/HR/RR/BT/Saturation) for unreadable
    device values — coerced to `NaN`.
  - `"#BOÞ!"` (an Excel export corruption artifact) in 556/1267 rows of `NRS_pain` — coerced to
    `NaN` and median-imputed.
  - `Saturation` (SpO2) missing in 688/1267 rows (~54%) — a real device-availability gap.
  - `KTAS duration_min` uses a decimal comma (`"5,00"`) — irrelevant, column dropped as leakage.
  - File encoding is `cp1252`, not UTF-8 (a stray byte breaks UTF-8 decoding) — load with
    `pd.read_csv("data.csv", sep=";", encoding="cp1252")`.

## Repository layout

```
CIA3/
├── 2547115_CIA3.ipynb          # main notebook — all 5 rubric sections, executed with real outputs
├── data.csv                    # KTAS dataset (semicolon-delimited, cp1252)
├── README.md                   # this file
├── section5_pitch_script.md    # 3-minute video pitch script
├── models/
│   └── triage_pipeline.joblib  # saved, reproducible trained pipeline (best model: Stacking Ensemble)
├── results/
│   ├── model_comparison_primary.csv
│   ├── vitals_ablation.csv
│   └── figures/                # every plot referenced in the notebook (EDA, confusion matrices,
│                                # comparison chart, SHAP plots, fairness chart)
└── scripts/
    ├── data_dictionary.py      # raw column meanings + caller-obtainable/device-measured split
    ├── build_notebook.py       # generates the notebook programmatically (for reproducibility/audit)
    ├── patch_narrative.py, patch_narrative2.py  # fills notebook narrative with real run results
    └── predict_new_call.py     # loads the saved pipeline and predicts on a new call record
```

## Setup

```bash
pip install pandas numpy scikit-learn lightgbm imbalanced-learn shap matplotlib seaborn joblib jupyter nbconvert
```

Tested with: pandas 2.2, scikit-learn 1.5, lightgbm 4.7, shap 0.52, Python 3.12.

## Reproducing the results

```bash
# from inside the CIA3/ folder
jupyter nbconvert --to notebook --execute --inplace 2547115_CIA3.ipynb --ExecutePreprocessor.timeout=1800
```

This re-runs every cell — data loading/audit, EDA, preprocessing pipeline, all four models
(tuned via cross-validated random search), the vitals ablation, SHAP explanations, and the
fairness audit — and re-saves `models/triage_pipeline.joblib` and every figure in
`results/figures/`. Random seed is fixed (`RANDOM_SEED = 42`) throughout for reproducibility.

## Predicting on a new call

```bash
python scripts/predict_new_call.py
```

Loads `models/triage_pipeline.joblib` and scores one example synthetic call record. Edit the
`example_call` dict in `scripts/predict_new_call.py` (or import `predict_call()` from it) to score
your own record — required fields: `Age`, `NRS_pain`, `Sex`, `Arrival mode`, `Injury`, `Mental`,
`Chief_complain_group`, `Caller_relationship`.

## Results summary (from the executed notebook)

**Primary experiment (caller-obtainable features only), test-set F1-macro:**

| Model | F1-macro | ROC-AUC (OVR macro) | Critical recall | Critical precision |
|---|---|---|---|---|
| **Stacking Ensemble** (winner) | **0.7229** | 0.8652 | 0.649 | 0.667 |
| LightGBM (boosting) | 0.7091 | 0.8702 | 0.649 | 0.600 |
| Logistic Regression (baseline) | 0.7077 | 0.8469 | 0.676 | 0.625 |
| Random Forest (bagging) | 0.6628 | 0.8601 | 0.595 | 0.667 |

The stacking ensemble (baseline + Random Forest + LightGBM, with a Logistic Regression
meta-learner trained on leakage-safe, out-of-fold base-learner predictions) beats the baseline by
**+0.0152 F1-macro (+2.2% relative)** — the only model to beat the baseline on every base metric
simultaneously. Random Forest alone underperforms the baseline, illustrating that a single
bagging model isn't automatically better than a well-regularized linear model on a dataset this
size (~885 training rows); stacking is what recovers and extends the gain.

**Vitals ablation (caller-obtainable vs. + device-measured vitals):** the gap is small and
inconsistent in direction — mean change of **-0.011 F1-macro** across the four model types.
Logistic Regression and Random Forest improve marginally with vitals added (+0.007, +0.002);
LightGBM and the Stacking Ensemble get slightly *worse* (-0.027, -0.025), likely because six
additional noisy/missing-heavy numeric columns increase overfitting risk faster than they add
signal at this sample size. Practically, this means the caller-obtainable-only design is not
leaving much performance on the table — full details and interpretation in notebook §3.6.

**Fairness audit:** Critical-class recall is **0.77 for female callers vs. 0.47 for male callers**
in the test set — a real, notable 30-point gap flagged for further investigation (see notebook
§4.4) rather than glossed over. Age-band recall shows no comparable collapse.

Full metrics, confusion matrices, SHAP explanations, and the complete ethics discussion are in the
notebook.

## Responsible-use notice

This model is **decision support only**, is **not validated for real emergency deployment**, is
trained on a small (1,267-row) sample of Korean ED data from 2016–2017, and has not undergone
clinical or regulatory review. See notebook Section 4.4 for the full ethics and deployment-limits
discussion, including the recommended human-review probability band and the false-negative/
false-positive cost-asymmetry argument for how the decision threshold should be tuned in any real
deployment.
