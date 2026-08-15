"""Patches placeholder narrative markdown cells in the executed notebook with the real numbers."""
import json
import os

NB_PATH = os.path.join(os.path.dirname(__file__), "..", "2547115_CIA3.ipynb")

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)


def find_cell(marker):
    for c in nb["cells"]:
        if c["cell_type"] == "markdown" and marker in "".join(c["source"]):
            return c
    raise ValueError(f"marker not found: {marker}")


# --- 3.5 model comparison narrative ---
c = find_cell("*(The specific winner and margin are read off")
c["source"] = r"""**Result:** the **Stacking Ensemble** wins on the test set with **F1-macro = 0.7229**,
beating the Logistic Regression baseline (F1-macro = 0.7077) by **+0.0152 absolute (+2.2%
relative)**. Ranked by F1-macro: Stacking Ensemble (0.723) > LightGBM (0.709) > Logistic Regression
baseline (0.708) > Random Forest (0.663). Random Forest alone actually falls *below* the linear
baseline here — with only ~885 training rows split three ways for CV, a single bagged-tree model
does not have enough data to out-perform a well-regularized linear model, but stacking it together
with LightGBM and the baseline still lets the ensemble beat every individual model, including the
baseline. The stacking ensemble also achieves the best Critical-class precision (0.667, tied with
Random Forest) while matching LightGBM's Critical-class recall (0.649) — i.e. it isn't just winning
on the macro-average, it specifically doesn't sacrifice detection of the safety-critical minority
class to get there. ROC-AUC (OVR macro) is closely clustered across all four models (0.847–0.870),
which is expected given the modest dataset size — the F1-macro/per-class breakdown is the more
diagnostic metric here for a safety-critical, imbalanced target.
""".splitlines(keepends=True)

# --- 3.6 ablation gap narrative ---
c = find_cell("*(Read the printed gap once executed and summarize here")
c["source"] = r"""**Result:** the vitals gap is **small and inconsistent in sign** across models — average change
of **-0.011 F1-macro** across the four model types when vitals are added. Logistic Regression and
Random Forest both improve marginally with vitals (+0.007 and +0.002 respectively), but LightGBM and
the Stacking Ensemble actually get *slightly worse* with vitals added (-0.027 and -0.025). This is a
genuinely interesting and somewhat counter-intuitive result worth taking seriously rather than
dismissing: with only ~190 training rows per class after the 70/15/15 split, adding six more noisy,
54%-missing (`Saturation`) numeric columns increases the feature space and the model's capacity to
overfit faster than it increases real signal, especially for the more flexible tree-boosting models.
It does **not** mean vitals are clinically uninformative in general — they clearly are, in any normal
ED context — it means that in this particular sample size regime, the caller-obtainable features
alone already capture most of the learnable signal for *this* target definition (the 3-class
collapse), and the added vitals mostly add noise rather than a clean accuracy gain. Operationally,
this is a reassuring result for the phone-triage use case: the caller-obtainable-only models are not
leaving a large amount of performance on the table by design, they are already close to (and for two
of four model types, better than) what the same models achieve with full vitals.
""".splitlines(keepends=True)

# --- 4.2 local explanation narrative ---
c = find_cell("**Plain-language read of the three local explanations**")
c["source"] = r"""**Plain-language read of the three local explanations (actual model output):**

- **Example A (68 y/o, unresponsive, cardiac/chest complaint, NRS pain 9):** predicted **Critical**
  with probability 0.85 (Non-urgent 0.09, Urgent 0.06) — a confident, correct-shaped prediction. The
  SHAP waterfall shows this driven mainly by *unresponsive mental status* and the *cardiac/chest*
  complaint category — despite family-member (not self) reporting, which the model correctly treats
  as low-signal on its own. This matches the "possible cardiac event with altered mental status" red
  flag a real dispatcher is trained to listen for.
- **Example B (24 y/o, alert, walking in, mild pain, self-reporting):** predicted **Non-urgent**
  with probability 0.70 (Critical 0.16, Urgent 0.14) — driven by *alert mental status*, *low
  self-reported pain (2/10)*, and *walking* arrival mode, consistent with a stable, ambulatory
  patient profile.
- **Example C (45 y/o, alert, abdominal complaint, pain 6/10, bystander caller):** predicted
  **Urgent** with probability 0.52 (Non-urgent 0.33, Critical 0.14) — the deliberately ambiguous
  case lands almost exactly on the Urgent/Non-urgent boundary (0.52 vs. 0.33, a 19-point margin),
  driven by the moderate-high pain score and the abdominal complaint category, which can range from
  benign to surgical emergency. This is precisely the kind of borderline probability spread that
  should trigger the human-review band recommended in §4.4 rather than being auto-resolved.
""".splitlines(keepends=True)

# --- 4.4 subgroup fairness narrative ---
c = find_cell("*(Once executed: state explicitly whether the subgroup recall numbers")
c["source"] = r"""**Result — a real fairness gap worth flagging, not dismissing:** Critical-class recall is
**0.77 for Female callers (17/22 correctly flagged) vs. only 0.47 for Male callers (7/15 correctly
flagged)** in the test set — a **30-point gap**. This means, in this test sample, the model misses
more than half of truly critical presentations in male patients while catching over three-quarters
in female patients. By age band, recall ranges from 0.58 (51–70) to 1.00 (<=30, though only n=2 —
too small to trust), with no band showing a fairness-breaking collapse to near-zero. **The
sex-based gap is the more concerning and more statistically grounded of the two** (larger subgroup
sizes: 22 and 15 Critical cases respectively) and should be treated as a real finding requiring
investigation before any deployment — possible causes include differences in how male vs. female
patients or their callers described symptoms in this dataset, or simply small-sample noise (37
total Critical test cases split two ways is still a small base rate to estimate recall from
precisely). Either way, this is exactly the kind of disparity a deployment-readiness review must
resolve — with a larger dataset and a formal statistical test for the gap — before this model could
be trusted operationally.
""".splitlines(keepends=True)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Patched narrative cells with real results.")
