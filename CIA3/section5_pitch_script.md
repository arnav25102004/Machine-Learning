# Section 5 — Three-Minute Product Pitch and Live Demo Script

**Project:** Emergency Ambulance Call Triage / Prioritization (KTAS-based)
**Total runtime:** 3:00

---

## 0:00–0:35 — Problem and beneficiaries (35s)

**Spoken:**
> "Imagine a dispatch center with three ambulances and five emergency calls ringing at once. The
> dispatcher has no vitals monitor, no doctor on scene — often not even the patient on the line,
> just a relative or a bystander describing what they see. Who gets the ambulance first? Get it
> wrong, and someone who needed help immediately waits behind someone who didn't. We built a model
> that gives each incoming call an independent urgency score — Critical, Urgent, or Non-urgent —
> using only what's realistically knowable over a phone call, so dispatchers can rank simultaneous
> calls and send limited ambulances where they're needed most first."

**On-screen elements:**
- Title card: project name + "Health — Emergency Dispatch Triage"
- Simple diagram: 3 ambulances icon, 5 phone-call icons converging on one dispatcher icon
- Cut to: repo folder structure in VS Code / file explorer (`README.md`, notebook, `models/`)

---

## 0:35–1:15 — Data cleaning, EDA, feature engineering highlights (40s)

**Spoken:**
> "We used the KTAS dataset — real, anonymized emergency-department triage records from Korea,
> 1,267 patient visits, each independently graded by clinical experts on a 5-level urgency scale,
> which we collapsed to three classes matching a real dispatch decision: send now, queue, or defer.
> The data had real problems — corrupted placeholder values in over half the pain-score column,
> missing oxygen readings in more than half the rows — all documented and cleaned in the notebook.
> But the key design decision was this: we split every feature into two groups — what a caller can
> actually tell a dispatcher over the phone, like symptoms, pain level, consciousness, age — versus
> what only a hospital device can measure, like blood pressure and heart rate. Our primary models
> train on caller-obtainable data ONLY, to simulate the real information a dispatcher has. Vitals
> are only used in a separate side-by-side comparison, to measure exactly what's lost by respecting
> that real-world constraint."

**On-screen elements:**
- Notebook Section 2 scrolling: class-balance bar chart (`class_balance.png`)
- Highlight the caller-obtainable vs. device-measured table/markdown cell (§2.4)
- Quick flash of `chief_complaint_distribution.png` and `pain_vs_triage.png`

---

## 1:15–2:10 — Baseline vs. bagging vs. boosting vs. stacking results (55s)

**Spoken:**
> "We trained four models on the caller-obtainable features: a Logistic Regression baseline, a
> tuned Random Forest, a tuned LightGBM boosting model, and a stacking ensemble combining all
> three with a meta-learner — trained with cross-validation so the meta-learner never sees data its
> base models were trained on, avoiding leakage. On the held-out test set, the stacking ensemble
> won, with an F1-macro of 0.72 versus 0.71 for the baseline — a real, if modest, gain, and
> critically, it didn't sacrifice detection of critical cases to get there. We also ran the vitals
> comparison: surprisingly, adding device vitals barely helped — and for two of four models it
> actually hurt slightly, given our sample size. That's a genuinely useful finding: the
> caller-obtainable-only design isn't leaving much on the table."

**On-screen elements:**
- `model_comparison_chart.png` — the F1/ROC-AUC bar chart across all four models
- `results/model_comparison_primary.csv` table view
- `vitals_ablation_chart.png` — the vitals gap comparison

---

## 2:10–3:00 — Live prediction + SHAP explanation + ethics caveat (50s)

**Spoken:**
> "Here's the model in action on one realistic call: a 72-year-old, reported by a family member,
> difficulty breathing, arriving by private ambulance, pain 8 out of 10, only responding to verbal
> prompts. The model predicts Critical with 76% confidence. SHAP tells us why: the low mental-status
> response and the respiratory complaint are the two biggest drivers — exactly what a trained
> dispatcher would flag. But — and this matters — this model is decision support only. It's not
> validated for real emergency use, it's trained on a relatively small Korean hospital dataset from
> 2016 to 2017, and our own fairness audit found a real gap in how reliably it catches critical
> cases across male versus female callers. A human dispatcher always makes the final call. This
> is a proof of concept for the methodology, not a deployable system."

**On-screen elements:**
- Terminal running `python scripts/predict_new_call.py` live, showing the predicted class + probabilities
- `shap_waterfall_A_critical_example.png` — the SHAP waterfall for the closest matching example
- Closing slide: "Decision support only — not validated for deployment" + dataset citation

---

## Production notes

- Keep spoken pace brisk — total word count above is ~430 words, roughly matching a 3-minute
  spoken pace (140–150 wpm).
- All numbers quoted in the script are pulled directly from the executed notebook's real output
  (see `README.md` results table) — do not substitute placeholder or rounded-differently numbers
  when recording.
- If recording live rather than narrating over screen capture, run
  `python scripts/predict_new_call.py` once beforehand to confirm the exact console output timing.
