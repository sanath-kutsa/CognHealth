# CognHealth
# 🧠 CognHealth
**Early Alzheimer's risk awareness and cognitive training**

Live site: https://sanath-kutsa.github.io/CognHealth/

CognHealth is a web app that helps people understand their Alzheimer's risk awareness and train their reasoning skills. It is an educational, screening-style tool. It is **not** a medical device and does **not** diagnose or predict Alzheimer's disease.

## Features
- **Alzheimer's risk check:** adaptive questionnaire with 15+ questions. Follow-up questions change based on earlier answers, plus a word-recall memory test.
- **Risk awareness score:** a logistic regression model built on published modifiable risk factors (age, education, family history, sleep, activity, blood pressure, diabetes, hearing loss, and more), with a chart showing what contributed most.
- **Cognitive skills estimate:** 20 adaptive questions across numeric, pattern, verbal and logic skills. Difficulty rises and falls with your answers, and results are on a 70–130 index.
- **Logic Arena:** three reasoning games (Matrix Reasoning, Order Logic, Number Logic) with 50 levels each, XP, daily goals and streaks.
- **Brain games:** Memory Match, Sequence Recall, N-Back and adaptive puzzles, with game recommendations based on your weakest areas.
- **Articles:** curated reading on Alzheimer's, neuroscience, brain health and AI from sources such as The Lancet, WHO and MIT.
- Terms, Privacy Policy, medical disclaimer, consent screen and contact page.

## Machine learning
- **Model:** logistic regression, `score = 100 / (1 + e^-(b + Σ wᵢxᵢ))`
- **Weights:** log relative risks inspired by the 2024 Lancet Commission on dementia prevention.
- **Training script:** `cognhealth_model.py` (Python, scikit-learn, NumPy) trains the model on a **synthetic** cohort, reports AUC on held-out data and exports the learned weights to JSON.
- **Adaptive logic:** rule-based question branching and a staircase difficulty model similar to item response theory.

## Tech stack
HTML, CSS, JavaScript (single-file app, no backend), Canvas animation, localStorage for progress, Python for the model.

## Limitations
The model weights are illustrative and are not calibrated or validated on patient data. Real research uses datasets such as ADNI, OASIS and NACC, plus clinical assessment. The score is an awareness indicator, not a probability of developing Alzheimer's. The cognitive index is not a clinical IQ score. Research on whether brain training improves general intelligence is mixed, and no improvement is guaranteed.

## Contact
cognhealth@gmail.com
