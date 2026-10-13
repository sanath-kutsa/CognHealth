"""CognHealth- Alzheimer's risk awareness model.

Real labelled data (ADNI / OASIS / NACC) needs approved access, so this trains on a SYNTHETIC
cohort whose risk factors follow published relative risks (Lancet Commission-style log-RRs).
To use real data, replace make_data() and keep the rest.
Run: pip install scikit-learn numpy && python cognhealth_model.py
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# feature -> log relative risk used to simulate the cohort (same weights as the website)
TRUE_W = {"age_decades_over_60": .6, "low_education": .47, "family_history": .59, "poor_sleep": .26,
          "low_activity": .34, "high_bp": .47, "diabetes": .4, "smoking": .47, "hearing_loss": .64,
          "memory_complaints": .5, "social_isolation": .47, "low_mood": .5,
          "head_injury": .59, "heavy_alcohol": .18, "obesity": .47, "high_ldl": .53, "vision_loss": .4,
          "low_mental_activity": .3, "poor_diet": .2, "language_trouble": .5, "recall_deficit": .25}
INTERCEPT = -3.6
rng = np.random.default_rng(42)

def make_data(n=6000):
    names = list(TRUE_W)
    X = (rng.random((n, len(names))) < 0.25).astype(float)          # binary risk factors
    X[:, 0] = rng.integers(0, 4, n)                                  # age decades over 60 (0-3)
    X[:, -1] = rng.integers(0, 6, n)                                 # words missed out of 5
    logit = INTERCEPT + X @ np.array(list(TRUE_W.values()))
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return names, X, y

names, X, y = make_data()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)
model = LogisticRegression(C=10, max_iter=1000).fit(Xtr, ytr)
print("AUC on held-out data:", round(roc_auc_score(yte, model.predict_proba(Xte)[:, 1]), 3))
for n, w in zip(names, model.coef_[0]):
    print(f"{n:22s} learned {w:+.2f}   published-style {TRUE_W[n]:+.2f}")

def risk_score(row):
    """Return a 0-100 Alzheimer's risk awareness score for one feature vector."""
    return round(100 * model.predict_proba([row])[0, 1], 1)

json.dump({"intercept": float(model.intercept_[0]), "weights": dict(zip(names, map(float, model.coef_[0])))},
          open("cognhealth_weights.json", "w"), indent=2)
print("Saved cognhealth_weights.json")
