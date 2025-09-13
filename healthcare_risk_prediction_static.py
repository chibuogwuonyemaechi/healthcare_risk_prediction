# -*- coding: utf-8 -*-
"""
Healthcare Risk Prediction (Heart Disease)
- No widgets, no inline plotting
- Saves all figures to outputs/ as static PNGs
- Prints metrics to console and writes metrics.json
"""


import os
os.environ["MPLBACKEND"] = "Agg"  # force static backend (no widgets)

import warnings, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams["figure.dpi"]  = 150      # notebook/preview size
mpl.rcParams["savefig.dpi"] = 360      # file resolution (crisper PNGs)
mpl.rcParams["font.size"]   = 11

import matplotlib as mpl
mpl.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 600,       # <- crisp PNGs
    "font.size": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 12,
})



from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve,
    ConfusionMatrixDisplay, f1_score
)

warnings.filterwarnings("ignore")
np.random.seed(42)

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

def load_heart_df():
    candidates = [
        "https://storage.googleapis.com/download.tensorflow.org/data/heart.csv",
        "https://raw.githubusercontent.com/dataprofessor/data/master/heart-disease.csv",
    ]
    for url in candidates:
        try:
            df = pd.read_csv(url)
            print(f"✅ Loaded: {url} | shape={df.shape}")
            return df
        except Exception as e:
            print(f"⚠️ Failed {url}: {e}")
    local = Path("data/heart.csv")
    if local.exists():
        df = pd.read_csv(local)
        print(f"✅ Loaded local: {local} | shape={df.shape}")
        return df
    raise RuntimeError("Could not load heart dataset. Provide data/heart.csv with a 'target' column.")

def eda(df: pd.DataFrame):
    print("Target balance:\n", df["target"].value_counts())
    num_cols = [c for c in df.columns if c != "target"]

    # histograms
    cols = 4
    rows = max(1, (len(num_cols) + cols - 1)//cols)
    plt.figure(figsize=(16, rows*3.0))
    for i, c in enumerate(num_cols, start=1):
        plt.subplot(rows, cols, i)
        try:
            df[c].hist(bins=30)
        except Exception:
            pass
        plt.title(c)
    plt.tight_layout()
    plt.savefig(OUT/"eda_distributions.png", dpi=160)
    plt.close()

    # correlation
    corr = df[num_cols + ["target"]].corr(numeric_only=True)
    plt.figure(figsize=(9,7))
    plt.imshow(corr, cmap="RdBu", vmin=-1, vmax=1)
    plt.colorbar(shrink=0.8)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(OUT/"corr_matrix.png", dpi=160)
    plt.close()

def select_model(X_train, y_train, pre):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = []

    # Logistic Regression
    pipe_lr = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"))])
    gs_lr = GridSearchCV(pipe_lr, {"clf__C":[0.1,1.0,5.0]}, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
    gs_lr.fit(X_train, y_train)
    candidates.append(("LogReg", gs_lr.best_estimator_, gs_lr.best_score_))

    # Random Forest
    pipe_rf = Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced", n_jobs=-1))])
    gs_rf = GridSearchCV(pipe_rf, {"clf__max_depth":[None,6,10,14], "clf__min_samples_leaf":[1,2,5]}, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
    gs_rf.fit(X_train, y_train)
    candidates.append(("RandomForest", gs_rf.best_estimator_, gs_rf.best_score_))

    # Optional XGBoost
    try:
        import xgboost as xgb
        pos, neg = int((y_train==1).sum()), int((y_train==0).sum())
        spw = max(1.0, neg/max(1,pos))
        pipe_xgb = Pipeline([("pre", pre), ("clf", xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, random_state=42, objective="binary:logistic",
            eval_metric="auc", tree_method="hist", scale_pos_weight=spw, n_jobs=-1))])
        gs_xgb = GridSearchCV(pipe_xgb, {"clf__max_depth":[3,4,5], "clf__min_child_weight":[1,3,5]}, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
        gs_xgb.fit(X_train, y_train)
        candidates.append(("XGBoost", gs_xgb.best_estimator_, gs_xgb.best_score_))
    except Exception as e:
        print("⚠️ XGBoost unavailable/failed; skipping. Reason:", e)

    best_name, best_model, best_cv = sorted(candidates, key=lambda t: t[2], reverse=True)[0]
    print(f"✅ Best by CV ROC AUC: {best_name} (cv_auc={best_cv:.3f})")
    return best_name, best_model, best_cv

def evaluate(best_model, X, y, X_test, y_test):
    clf = best_model.named_steps["clf"]
    pos_idx = int(np.where(clf.classes_ == 1)[0][0])

    proba = best_model.predict_proba(X_test)[:, pos_idx]
    pred  = (proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, proba)
    pr_auc  = average_precision_score(y_test, proba)

    print(f"Test ROC AUC: {roc_auc:.3f} | PR AUC: {pr_auc:.3f}")
    print("\nClassification report:\n", classification_report(y_test, pred, digits=3))

    cm = confusion_matrix(y_test, pred)
    print("Confusion matrix:\n", cm)

    # curves
    fpr, tpr, _ = roc_curve(y_test, proba)
    prec, rec, _ = precision_recall_curve(y_test, proba)

    plt.figure(figsize=(11,4))
    plt.subplot(1,2,1); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--'); plt.title(f"ROC (AUC={roc_auc:.3f})"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.subplot(1,2,2); plt.plot(rec, prec); plt.title(f"Precision-Recall (AP={pr_auc:.3f})"); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.tight_layout(); plt.savefig(OUT/"roc_pr_curves.png", dpi=160); plt.close()

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion matrix (threshold=0.5)")
    plt.tight_layout(); plt.savefig(OUT/"confusion_matrix.png", dpi=160); plt.close()

    # stability
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_roc = cross_val_score(best_model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    prec2, rec2, thr = precision_recall_curve(y_test, proba)
    f1s = (2*prec2*rec2)/(prec2+rec2+1e-12)
    best_t = 0.5 if len(thr)==0 else thr[np.nanargmax(f1s[:-1])]

    # metrics.json
    with open(OUT/"metrics.json","w") as f:
        json.dump({
            "cv_roc_auc_mean": float(cv_roc.mean()),
            "test_roc_auc": float(roc_auc),
            "test_pr_auc": float(pr_auc),
            "best_threshold_by_f1": float(best_t)
        }, f, indent=2)

    print("✅ Saved:",
          (OUT/"roc_pr_curves.png").as_posix(),
          (OUT/"confusion_matrix.png").as_posix(),
          (OUT/"metrics.json").as_posix())

def shap_plots(best_model, X_train, X_test, OUT):
    """Generate high-res (PNG+SVG) SHAP plots; no widgets used."""
    import numpy as np
    import matplotlib.pyplot as plt
    try:
        import shap
    except Exception as e:
        print("⚠️ SHAP not installed; skipping:", e)
        return

    try:
        # Transform features exactly as the model sees them
        pre = best_model.named_steps["pre"]
        base_clf = best_model.named_steps["clf"]
        X_train_trans = pre.transform(X_train)
        X_test_trans  = pre.transform(X_test)
        feat_names    = pre.transformers_[0][2]

        # Choose explainer
        RFCls = None
        try:
            from sklearn.ensemble import RandomForestClassifier as RFCls
        except Exception:
            pass

        xgb = None
        try:
            import xgboost as xgb
        except Exception:
            pass

        sv = None
        if RFCls is not None and isinstance(base_clf, RFCls):
            explainer   = shap.TreeExplainer(base_clf)
            shap_values = explainer.shap_values(X_test_trans)
            sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        elif xgb is not None and hasattr(xgb, "XGBClassifier") and isinstance(base_clf, xgb.XGBClassifier):
            explainer   = shap.TreeExplainer(base_clf)
            shap_values = explainer.shap_values(X_test_trans)
            sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        else:
            # Linear first; fallback to Kernel (sampled) for LR/others
            background = shap.kmeans(X_train_trans, 20)
            try:
                explainer = shap.LinearExplainer(
                    base_clf, background, feature_perturbation="interventional"
                )
                sv = explainer.shap_values(X_test_trans)
            except Exception:
                explainer = shap.KernelExplainer(
                    lambda z: base_clf.predict_proba(z)[:, 1], background
                )
                n  = min(200, X_test_trans.shape[0])
                sv = explainer.shap_values(X_test_trans[:n], nsamples=100)
                X_test_trans = X_test_trans[:n]

        if sv is None:
            print("⚠️ Could not compute SHAP values; skipping plots.")
            return

        # ---- SHAP summary (high-res + vector) ----
        shap.summary_plot(
            sv,
            X_test_trans,
            feature_names=feat_names,
            plot_type="dot",
            show=False,
            max_display=12,
            plot_size=(14, 10),
        )
        plt.tight_layout()
        (OUT / "outputs").mkdir(exist_ok=True)  # in case OUT is project root
        plt.savefig(OUT / "shap_summary-600dpi.png", dpi=600, bbox_inches="tight", pad_inches=0.05)
        plt.savefig(OUT / "shap_summary.svg",        bbox_inches="tight", pad_inches=0.05)
        plt.savefig(OUT / "shap_summary.pdf",        bbox_inches="tight", pad_inches=0.05)
        plt.close()

        # ---- Top-10 bar (high-res + vector) ----
        mean_abs = np.abs(sv).mean(0)
        idx      = np.argsort(mean_abs)[::-1][:10]
        labels   = np.array(feat_names)[idx][::-1]
        vals     = mean_abs[idx][::-1]

        plt.figure(figsize=(10, 6))
        plt.barh(labels, vals)
        plt.xlabel("mean |SHAP value| (impact)")
        plt.title("Top 10 features by mean |SHAP|")
        plt.tight_layout()
        plt.savefig(OUT / "shap_top_features-600dpi.png", dpi=600, bbox_inches="tight", pad_inches=0.05)
        plt.savefig(OUT / "shap_top_features.svg",        bbox_inches="tight", pad_inches=0.05)
        plt.close()

        print("✅ Saved SHAP plots:",
              (OUT / "shap_summary-600dpi.png").as_posix(),
              (OUT / "shap_top_features-600dpi.png").as_posix())
    except Exception as e:
        print("⚠️ SHAP plotting failed; continuing. Reason:", e)


def main():
    print("Matplotlib backend:", matplotlib.get_backend())  # should be 'Agg'
    print("Saving to:", OUT.resolve())

    # 1) Load
    df = load_heart_df()
    df.columns = [c.strip().lower() for c in df.columns]
    assert "target" in df.columns, "Dataset must include a 'target' column."
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 2) EDA (static images)
    eda(df)

    # 3) Split + preprocess
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)
    X = df.drop(columns=["target"]); y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    num_features = X.columns.tolist()
    pre = ColumnTransformer(
        transformers=[("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale",  StandardScaler())
        ]), num_features)]
    )

    # 4) Model selection
    best_name, best_model, best_cv = select_model(X_train, y_train, pre)
    print(f"Winner: {best_name} | CV ROC AUC: {best_cv:.3f}")

    # 5) Evaluation (static images + metrics.json)
    evaluate(best_model, X, y, X_test, y_test)

    # 6) SHAP (static images)
    shap_plots(best_model, X_train, X_test, OUT)

    print("\nAll done. Open the 'outputs/' folder for PNGs and metrics.json.")

if __name__ == "__main__":
    main()
