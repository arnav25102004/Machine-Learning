"""
Data dictionary / column semantics for the KTAS (Korean Triage and Acuity Scale) dataset.

Source (cite exactly this in README):
    Kaggle dataset: "Korean Triage and Acuity Scale (KTAS)"
    Uploader: ilkeryildiz
    URL: https://www.kaggle.com/datasets/ilkeryildiz/ktas
    Underlying clinical study: Korean Triage and Acuity Scale data collected at two
    academic EDs in South Korea (2016-2017), described in:
    Ok M. et al., "Predicting the Korean Triage and Acuity Scale Level..." and related
    KTAS validation literature.

This file documents raw column meaning and encodes the caller-obtainable vs.
device-measured split used throughout the notebook.
"""

# Raw column -> plain-language meaning
COLUMN_MEANING = {
    "Group": "Site/hospital group indicator (1 or 2) — administrative, not predictive.",
    "Sex": "Patient sex, coded 1/2 in source (mapped to Male/Female).",
    "Age": "Patient age in years, as reported by caller/patient.",
    "Patients number per hour": "ED census/crowding indicator at time of arrival — NOT available to a phone dispatcher, excluded from modeling.",
    "Arrival mode": "How the patient arrived (1=Walking,2=Public Ambulance,3=Private vehicle,4=Private ambulance,5=Wheelchair,6=Other) — caller can state this or dispatcher decides it, so treated as caller-obtainable.",
    "Injury": "Injury vs. non-injury presentation flag (1=Yes,2=No) — caller-obtainable.",
    "Chief_complain": "Free-text chief complaint as recorded by triage nurse — caller-obtainable (this is exactly what a caller describes over the phone).",
    "Mental": "Mental status (1=Alert,2=Verbal Response,3=Pain Response,4=Unresponsive) — caller-obtainable (a bystander/caller can describe consciousness level).",
    "Pain": "Binary flag: patient reports pain (1) or not (0/2) — caller-obtainable.",
    "NRS_pain": "Numeric Rating Scale pain score (0-10), patient/caller self-report — caller-obtainable.",
    "SBP": "Systolic blood pressure (mmHg) — DEVICE-MEASURED, not available over a phone call.",
    "DBP": "Diastolic blood pressure (mmHg) — DEVICE-MEASURED.",
    "HR": "Heart rate (bpm) — DEVICE-MEASURED.",
    "RR": "Respiratory rate (breaths/min) — DEVICE-MEASURED.",
    "BT": "Body temperature (deg C) — DEVICE-MEASURED.",
    "Saturation": "SpO2 oxygen saturation (%) — DEVICE-MEASURED, largest missingness (~54%).",
    "KTAS_RN": "Triage level (1-5) assigned by the triage nurse at first contact — NOT a feature, informational only (we predict the expert label, not this).",
    "Diagnosis in ED": "Final ED diagnosis text — only known AFTER treatment, leakage if used as a feature. Excluded.",
    "Disposition": "Discharge/admission outcome — only known AFTER treatment, leakage if used as a feature. Excluded.",
    "KTAS_expert": "Gold-standard triage acuity level (1=most urgent/resuscitation ... 5=least urgent), assigned retrospectively by triage experts. THIS IS THE PREDICTION TARGET.",
    "Error_group": "Study metadata about RN-vs-expert mistriage category — leakage (derived from target), excluded.",
    "Length of stay_min": "ED length of stay in minutes — only known AFTER the visit, leakage, excluded.",
    "KTAS duration_min": "Time to triage — process metadata, only known after the visit; excluded.",
    "mistriage": "Study flag for whether RN mistriaged vs. expert — leakage (derived from target), excluded.",
}

# Features realistically obtainable by a phone dispatcher / bystander caller,
# with NO physical device and NO hands-on exam of the patient.
CALLER_OBTAINABLE_NUMERIC = ["Age", "NRS_pain"]
CALLER_OBTAINABLE_CATEGORICAL = ["Sex", "Arrival mode", "Injury", "Mental", "Chief_complain_group", "Caller_relationship"]

# Features that require a physical device / hands-on measurement at the bedside.
DEVICE_MEASURED_NUMERIC = ["SBP", "DBP", "HR", "RR", "BT", "Saturation"]

# Columns dropped entirely: administrative, leakage (post-outcome), or not
# realistically available at dispatch time.
LEAKAGE_OR_IRRELEVANT_COLUMNS = [
    "Group",
    "Patients number per hour",
    "KTAS_RN",
    "Diagnosis in ED",
    "Disposition",
    "Error_group",
    "Length of stay_min",
    "KTAS duration_min",
    "mistriage",
]

TARGET_RAW = "KTAS_expert"
