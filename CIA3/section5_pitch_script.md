# Section 5 — Three-Minute Product Pitch and Live Demo Script

**Project:** Emergency Ambulance Call Triage / Prioritization (KTAS-based)
**Total runtime:** 3:00

---

## 0:00–0:35 — Problem and beneficiaries (35s)

**Spoken:**
> "Say a dispatch center has three ambulances, and five calls come in at the same time. The
> dispatcher has no vitals monitor, no doctor standing there. A lot of the time it's not even the
> patient on the phone — it's a family member or a stranger describing what they see. So who gets
> the ambulance first? If you get that wrong, someone who needed help right away ends up waiting.
> That's the problem we're solving. We built a model that gives every incoming call its own urgency
> score — Critical, Urgent, or Non-urgent — using only what a dispatcher could actually know from a
> phone call. That way they can rank calls coming in together and send ambulances to the worst
> cases first."

**On-screen elements:**
- Title card: project name + "Health — Emergency Dispatch Triage"
- Simple diagram: 3 ambulances icon, 5 phone-call icons converging on one dispatcher icon
- Cut to: repo folder structure in VS Code / file explorer (`README.md`, notebook, `models/`)

---

## 0:35–1:15 — Data cleaning, EDA, feature engineering highlights (40s)

**Spoken:**
> "We used the KTAS dataset — real emergency-room triage records from Korea, about 1,267 patients,
> each one graded by actual doctors on a 5-level urgency scale. We simplified that down to three
> levels that match what a dispatcher actually decides: send help now, queue it, or it can wait.
> The data wasn't clean — over half the pain-score entries were corrupted, and more than half the
> oxygen readings were just missing. We found all that and fixed it in the notebook. But the real
> decision we made was this: we split every column into two groups — stuff a caller can actually
> tell you over the phone, like symptoms, pain, whether they're conscious, their age — versus stuff
> only a hospital machine can measure, like blood pressure or heart rate. Our main models only get
> trained on the phone-call stuff. The device readings are only used later, in a separate test, just
> to see how much we're giving up by not having them."

**On-screen elements:**
- Notebook Section 2 scrolling: class-balance bar chart (`class_balance.png`)
- Highlight the caller-obtainable vs. device-measured table/markdown cell (§2.4)
- Quick flash of `chief_complaint_distribution.png` and `pain_vs_triage.png`

---

## 1:15–2:10 — Baseline vs. bagging vs. boosting vs. stacking results (55s)

**Spoken:**
> "We tried four models. First a simple Logistic Regression as our baseline. Then a Random Forest.
> Then LightGBM, a boosting model. And finally we stacked all three together with a meta-learner on
> top — basically a model that learns how to combine the other three. We made sure the meta-learner
> never saw predictions from a model that was trained on the same data, so there's no cheating going
> on there. On the test set, the stacked model won — 0.72 versus 0.71 for the baseline. Not a huge
> jump, but a real one, and it didn't get worse at catching critical cases to achieve it. We also
> checked what happens if we add the vitals back in — and honestly, it barely made a difference. For
> two of the four models it actually got slightly worse. Which tells us the phone-call-only approach
> isn't giving up as much as you'd think."

**On-screen elements:**
- `model_comparison_chart.png` — the F1/ROC-AUC bar chart across all four models
- `results/model_comparison_primary.csv` table view
- `vitals_ablation_chart.png` — the vitals gap comparison

---

## 2:10–3:00 — Live prediction + SHAP explanation + ethics caveat (50s)

**Spoken:**
> "Let's run it live on one example: a 72-year-old, a family member is calling it in, trouble
> breathing, came by private ambulance, pain 8 out of 10, only reacting when someone talks to them.
> The model says Critical, 76% confident. And we can actually see why — it's mostly picking up on
> the fact that they're not fully responsive, plus the breathing problem. That's exactly what a real
> dispatcher would key in on too. But — and this is important — this is decision support, not a
> replacement for a dispatcher. It's not tested for real emergency use, it's trained on a fairly
> small dataset from Korean hospitals a few years back, and when we checked, we actually found it
> catches critical cases less reliably for male callers than female ones. So a real dispatcher still
> makes the final call, always. This is a proof of concept, not something you'd plug into a real
> dispatch center as-is."

**On-screen elements:**
- Terminal running `python scripts/predict_new_call.py` live, showing the predicted class + probabilities
- `shap_waterfall_A_critical_example.png` — the SHAP waterfall for the closest matching example
- Closing slide: "Decision support only — not validated for deployment" + dataset citation

---

## Production notes

- Talk at a normal, brisk pace — the script is about 420 words, which fits 3 minutes comfortably.
- Every number in this script comes straight from the notebook's actual output (also in
  `README.md`) — don't round differently or swap in placeholder numbers when recording.
- If you're running the prediction live instead of just showing a recording, run
  `python scripts/predict_new_call.py` once before recording so you know what it'll print and how
  long it takes.
