# Healthcare Risk Prediction (Heart Disease Demo)

A compact, health-style classification demo using a public Heart dataset.

## What it shows
- EDA: distributions & correlation heatmap
- Models: Logistic Regression, Random Forest, XGBoost (auto-select best by CV ROC AUC)
- Metrics: ROC AUC, PR AUC, confusion matrix, classification report
- Interpretability: SHAP summary plot

## How to run
Open `healthcare_risk_prediction.ipynb` and run cells top→bottom.
Artifacts are saved under `outputs/`:
- `eda_distributions.png`, `corr_matrix.png`
- `roc_pr_curves.png`, `confusion_matrix.png`
- `shap_summary.png`
- `metrics.json`

# Healthcare Risk Prediction (Heart Disease) — No-Widget Static Pipeline

A clean, reproducible ML workflow for heart-disease risk prediction:
- Robust EDA (saved as PNG)
- Model selection (LogReg, RandomForest, optional XGBoost)
- Evaluation (ROC, PR, confusion matrix)
- Explainability with SHAP (high-res PNG + SVG)

**Why it’s useful:** Static PNG/SVG artifacts (no Jupyter widgets) → easy to review in interviews and share on GitHub/LinkedIn.

## Quickstart
```bash
conda create -n heartds python=3.10 -y
conda activate heartds
pip install -r requirements.txt
python healthcare_risk_prediction_static.py

Artifacts saved to outputs/:

eda_distributions.png, corr_matrix.png

roc_pr_curves.png, confusion_matrix.png

shap_summary-600dpi.png + .svg, shap_top_features-600dpi.png + .svg

metrics.json

Headline Results (sample)

CV ROC AUC (LogReg winner): 0.903

Test ROC AUC: 0.926

Test PR AUC: 0.812

Notes

Works offline after first run (if data is cached locally).

If xgboost is hard to install, remove it from requirements.txt—the script handles it gracefully.


3) **Ensure `requirements.txt` is clean** (no prose):
```bash
notepad requirements.txt

