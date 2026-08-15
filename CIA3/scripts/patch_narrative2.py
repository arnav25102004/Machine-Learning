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


c = find_cell("**Reading the global plot in plain language:**")
c["source"] = r"""**Reading the global plot in plain language (actual SHAP output):** the strongest driver of a
Critical prediction is the **Cardiac/Chest chief-complaint category** — rows with this complaint
push the prediction sharply toward Critical, matching real triage priorities where chest pain is a
classic red flag. **Arrival mode** is the next strongest block: arriving by **Private Ambulance**
pushes toward Critical while arriving by **Private Vehicle** pushes away from it — sensible, since a
family already decided to call an ambulance rather than drive. **Age** has a clear but graded effect
(older pushes toward Critical, younger pushes away). **Mental status = Alert** is a strong
*negative* driver of Critical (blue/low values push toward Critical, i.e. being *not* alert pushes
toward Critical) — consistent with consciousness level being one of the clearest triage signals a
caller can convey. **Sex** and **NRS_pain** show smaller but non-trivial effects, and — notably —
the synthetic `Caller_relationship` field barely registers (Bystander/Stranger appears far down the
list with a small effect), which is a reassuring sanity check: the model is not leaning heavily on
our simulated field, it is driven mainly by real clinical signal (complaint category, arrival mode,
age, mental status).
""".splitlines(keepends=True)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Patched SHAP global summary narrative.")
