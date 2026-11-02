"""
Heart-disease prediction: Majority baseline vs Logistic Regression vs Decision Tree vs KNN.
Leakage-resistant workflow (all preprocessing inside sklearn Pipelines):
  stratified 75/25 split -> 5-fold CV tuning on train -> nested CV for model comparison
  -> single final evaluation on the held-out test set -> bootstrap CIs, McNemar tests,
  cost-sensitive threshold analysis (threshold chosen from out-of-fold training predictions only).
Run:  python src/analysis.py        (from repository root)
"""
import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.model_selection import (train_test_split, StratifiedKFold, GridSearchCV,
                                     cross_validate, cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, balanced_accuracy_score,
                             matthews_corrcoef, make_scorer)

warnings.filterwarnings("ignore")
SEED = 42
ROOT = Path(__file__).resolve().parents[1]
FIG, RES = ROOT / "figures", ROOT / "results"
FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight"})
COL = {"Majority baseline": "#7f7f7f", "Logistic regression": "#2E86AB",
       "Decision tree": "#E67E22", "KNN": "#27AE60"}

# ------------------------------------------------------------------ data
df = pd.read_csv(ROOT / "data" / "heart.csv")
n_chol0, n_bp0 = int((df.Cholesterol == 0).sum()), int((df.RestingBP == 0).sum())
# zero cholesterol / resting BP are physiologically impossible -> missing
df["Cholesterol"] = df["Cholesterol"].replace(0, np.nan)
df["RestingBP"] = df["RestingBP"].replace(0, np.nan)
TARGET = "HeartDisease"
NUM = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
CAT = ["Sex", "ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope"]
X, y = df[NUM + CAT], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
info = {"n": len(df), "n_train": len(X_train), "n_test": len(X_test),
        "pos_rate_all": float(y.mean()), "pos_rate_train": float(y_train.mean()),
        "pos_rate_test": float(y_test.mean()), "n_pos_test": int(y_test.sum()),
        "n_neg_test": int((1 - y_test).sum()), "chol_zero": n_chol0, "bp_zero": n_bp0,
        "chol_missing_pos_rate": float(df.loc[df.Cholesterol.isna(), TARGET].mean()),
        "chol_observed_pos_rate": float(df.loc[df.Cholesterol.notna(), TARGET].mean())}


def make_prep(indicator=False):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=indicator)),
                          ("sc", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), CAT)])


def make_pipe(model, indicator=False):
    return Pipeline([("prep", make_prep(indicator)), ("model", model)])


# ------------------------------------------------------------------ models + grids
K_GRID = list(range(3, 32, 2))
C_GRID = [float(c) for c in np.logspace(-3, 2, 11)]
DEPTH_GRID, LEAF_GRID = list(range(1, 11)), [1, 5, 10, 20]
MODELS = {
    "Majority baseline": (DummyClassifier(strategy="most_frequent"), None),
    "Logistic regression": (LogisticRegression(max_iter=1000, tol=1e-4, random_state=SEED),
                            {"model__C": C_GRID}),
    "Decision tree": (DecisionTreeClassifier(random_state=SEED),
                      {"model__max_depth": DEPTH_GRID, "model__min_samples_leaf": LEAF_GRID}),
    "KNN": (KNeighborsClassifier(metric="euclidean"), {"model__n_neighbors": K_GRID}),
}
inner = StratifiedKFold(5, shuffle=True, random_state=SEED)
outer = StratifiedKFold(5, shuffle=True, random_state=SEED + 1)
spec = make_scorer(recall_score, pos_label=0)
SCORING = {"accuracy": "accuracy", "precision": make_scorer(precision_score, zero_division=0),
           "recall": "recall", "specificity": spec, "f1": "f1", "roc_auc": "roc_auc"}


def tuned(name, indicator=False):
    est, grid = MODELS[name]
    pipe = make_pipe(est, indicator)
    if grid is None:
        return pipe
    return GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=-1, refit=True)


# ------------------------------------------------------------------ nested CV (training partition only)
nested = {}
for name in MODELS:
    r = cross_validate(tuned(name), X_train, y_train, cv=outer, scoring=SCORING, n_jobs=1)
    nested[name] = {m: (float(r[f"test_{m}"].mean()), float(r[f"test_{m}"].std(ddof=1))) for m in SCORING}
    print("nested", name, {m: round(v[0], 3) for m, v in nested[name].items()})

# ------------------------------------------------------------------ final fit on full training set
gs, fitted, tune_time, fit_time, pred_time, best = {}, {}, {}, {}, {}, {}
for name in MODELS:
    t0 = time.perf_counter(); m = tuned(name).fit(X_train, y_train); tune_time[name] = time.perf_counter() - t0
    if hasattr(m, "best_params_"):
        gs[name] = m; best[name] = m.best_params_; final = m.best_estimator_
    else:
        best[name] = {}; final = m
    t0 = time.perf_counter()
    for _ in range(20):
        final.fit(X_train, y_train)
    fit_time[name] = (time.perf_counter() - t0) / 20
    t0 = time.perf_counter()
    for _ in range(20):
        final.predict_proba(X_test)
    pred_time[name] = (time.perf_counter() - t0) / 20
    fitted[name] = final
    print("best", name, best[name])

P = {n: m.predict_proba(X_test)[:, 1] for n, m in fitted.items()}
YP = {n: m.predict(X_test) for n, m in fitted.items()}      # default 0.50 rule
yt = y_test.values


def metric_row(yt_, yp, pr):
    tn, fp, fn, tp = confusion_matrix(yt_, yp, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(yt_, yp), "precision": precision_score(yt_, yp, zero_division=0),
            "recall": recall_score(yt_, yp), "specificity": tn / (tn + fp), "f1": f1_score(yt_, yp),
            "balanced_acc": balanced_accuracy_score(yt_, yp), "mcc": matthews_corrcoef(yt_, yp),
            "roc_auc": roc_auc_score(yt_, pr), "pr_auc": average_precision_score(yt_, pr),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


test_tab = pd.DataFrame({n: metric_row(yt, YP[n], P[n]) for n in MODELS}).T
test_tab.to_csv(RES / "test_metrics.csv")

# bootstrap CIs on the test set (predictions fixed, rows resampled)
rng = np.random.default_rng(SEED); B = 2000
boot = {n: {k: [] for k in ["accuracy", "recall", "specificity", "f1", "roc_auc"]} for n in MODELS}
for _ in range(B):
    idx = rng.integers(0, len(yt), len(yt))
    if yt[idx].min() == yt[idx].max():
        continue
    for n in MODELS:
        yp, pr = YP[n][idx], P[n][idx]
        tn, fp, fn, tp = confusion_matrix(yt[idx], yp, labels=[0, 1]).ravel()
        boot[n]["accuracy"].append((tn + tp) / len(idx)); boot[n]["recall"].append(tp / (tp + fn))
        boot[n]["specificity"].append(tn / (tn + fp)); boot[n]["f1"].append(f1_score(yt[idx], yp))
        boot[n]["roc_auc"].append(roc_auc_score(yt[idx], pr))
ci = {n: {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in d.items()}
      for n, d in boot.items()}

# exact McNemar test on correctness of default-threshold predictions
names = list(MODELS); mcn = {}
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        ca, cb = YP[a] == yt, YP[b] == yt
        b01, b10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
        p = binomtest(min(b01, b10), b01 + b10, 0.5).pvalue if b01 + b10 > 0 else 1.0
        mcn[f"{a} vs {b}"] = {"only_second_correct": b01, "only_first_correct": b10, "p": float(p)}
# paired bootstrap of AUC differences among the real models
auc_diff = {}
for a, b in [("Logistic regression", "Decision tree"), ("Logistic regression", "KNN"), ("KNN", "Decision tree")]:
    d = []
    for _ in range(B):
        idx = rng.integers(0, len(yt), len(yt))
        if yt[idx].min() == yt[idx].max():
            continue
        d.append(roc_auc_score(yt[idx], P[a][idx]) - roc_auc_score(yt[idx], P[b][idx]))
    auc_diff[f"{a} - {b}"] = {"mean": float(np.mean(d)), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}

# ------------------------------------------------------------------ cost-sensitive threshold (OOF on training only)
C_FN, C_FP = 5, 1
grid_t = np.linspace(0.02, 0.98, 49)
thr, cost_rows = {}, {}
for n in ["Logistic regression", "Decision tree", "KNN"]:
    oof = cross_val_predict(fitted[n], X_train, y_train, cv=inner, method="predict_proba")[:, 1]
    costs = []
    for t in grid_t:
        tn, fp, fn, tp = confusion_matrix(y_train, (oof >= t).astype(int), labels=[0, 1]).ravel()
        costs.append(C_FN * fn + C_FP * fp)
    costs = np.array(costs)
    ties = np.flatnonzero(costs == costs.min())
    t_star = float(grid_t[ties[len(ties) // 2]])
    thr[n] = {"t_star": t_star, "oof_cost_curve": costs.tolist()}
    row = {}
    for lab, t in [("0.50", 0.5), ("tuned", t_star)]:
        tn, fp, fn, tp = confusion_matrix(yt, (P[n] >= t).astype(int), labels=[0, 1]).ravel()
        row[lab] = {"t": t, "FN": int(fn), "FP": int(fp), "cost": int(C_FN * fn + C_FP * fp),
                    "recall": float(tp / (tp + fn)), "specificity": float(tn / (tn + fp))}
    cost_rows[n] = row
b = confusion_matrix(yt, YP["Majority baseline"], labels=[0, 1]).ravel()
cost_rows["Majority baseline"] = {"0.50": {"FN": int(b[2]), "FP": int(b[1]), "cost": int(C_FN * b[2] + C_FP * b[1])}}

# ------------------------------------------------------------------ ablation: missing-value indicator
abl = {}
for n in ["Logistic regression", "Decision tree", "KNN"]:
    r = cross_validate(tuned(n, indicator=True), X_train, y_train, cv=outer, scoring="roc_auc")
    abl[n] = {"with_indicator": float(r["test_score"].mean()), "without": nested[n]["roc_auc"][0]}

# ------------------------------------------------------------------ interpretability
lr_pipe = fitted["Logistic regression"]
fn_lr = [f.split("__")[1] for f in lr_pipe.named_steps["prep"].get_feature_names_out()]
coefs = pd.Series(lr_pipe.named_steps["model"].coef_[0], index=fn_lr).sort_values()
tree = fitted["Decision tree"].named_steps["model"]
tnames = [f.split("__")[1] for f in fitted["Decision tree"].named_steps["prep"].get_feature_names_out()]
imp = pd.Series(tree.feature_importances_, index=tnames).sort_values(ascending=False)
(RES / "tree_rules.txt").write_text(export_text(tree, feature_names=tnames, max_depth=4))

# ------------------------------------------------------------------ figures
# Fig 1: EDA
fig, ax = plt.subplots(1, 3, figsize=(10, 3))
vc = y.value_counts().sort_index()
ax[0].bar(["No (0)", "Yes (1)"], vc.values, color=["#2ECC71", "#E74C3C"])
for i, v in enumerate(vc.values):
    ax[0].text(i, v + 8, f"{v} ({100 * v / len(y):.1f}%)", ha="center")
ax[0].set_title("Class distribution"); ax[0].set_ylabel("Patients"); ax[0].set_ylim(0, 620)
cp = pd.crosstab(df.ST_Slope, df[TARGET], normalize="index")[1].reindex(["Up", "Flat", "Down"])
ex = pd.crosstab(df.ExerciseAngina, df[TARGET], normalize="index")[1]
cpt = pd.crosstab(df.ChestPainType, df[TARGET], normalize="index")[1]
lab = [f"ST slope: {k}" for k in cp.index] + [f"Angina: {k}" for k in ex.index] + [f"Chest pain: {k}" for k in cpt.index]
val = list(cp.values) + list(ex.values) + list(cpt.values)
ax[1].barh(lab[::-1], val[::-1], color="#2E86AB"); ax[1].axvline(y.mean(), ls="--", c="k", lw=.8)
ax[1].set_title("Heart-disease rate by category"); ax[1].set_xlabel("Proportion with heart disease")
ch = pd.Series({"Cholesterol\nobserved": info["chol_observed_pos_rate"],
                "Cholesterol\nrecorded as 0": info["chol_missing_pos_rate"]})
ax[2].bar(ch.index, ch.values, color=["#2E86AB", "#E67E22"]); ax[2].axhline(y.mean(), ls="--", c="k", lw=.8)
for i, v in enumerate(ch.values):
    ax[2].text(i, v + .02, f"{v:.2f}", ha="center")
ax[2].set_ylim(0, 1.05); ax[2].set_title("Outcome rate vs. missingness"); ax[2].set_ylabel("Proportion with heart disease")
plt.tight_layout(); plt.savefig(FIG / "fig1_eda.png"); plt.close()

# Fig 2: tuning curves
fig, ax = plt.subplots(1, 3, figsize=(10, 3))
r = pd.DataFrame(gs["Logistic regression"].cv_results_)
ax[0].errorbar(np.log10(r.param_model__C.astype(float)), r.mean_test_score, r.std_test_score,
               color=COL["Logistic regression"], marker="o", ms=3, capsize=2)
ax[0].axvline(np.log10(gs["Logistic regression"].best_params_["model__C"]), c="r", ls=":")
ax[0].set_xlabel(r"$\log_{10} C$"); ax[0].set_title("Logistic regression"); ax[0].set_ylabel("Mean 5-fold CV ROC AUC")
r = pd.DataFrame(gs["Decision tree"].cv_results_)
for leaf, g in r.groupby("param_model__min_samples_leaf"):
    ax[1].plot(g.param_model__max_depth.astype(int), g.mean_test_score, marker="o", ms=3, label=f"min leaf = {leaf}")
ax[1].set_xlabel("max_depth"); ax[1].set_title("Decision tree"); ax[1].legend(fontsize=7, frameon=False)
r = pd.DataFrame(gs["KNN"].cv_results_)
ax[2].errorbar(r.param_model__n_neighbors.astype(int), r.mean_test_score, r.std_test_score,
               color=COL["KNN"], marker="o", ms=3, capsize=2)
ax[2].axvline(gs["KNN"].best_params_["model__n_neighbors"], c="r", ls=":")
ax[2].set_xlabel("k (neighbours)"); ax[2].set_title("KNN")
plt.tight_layout(); plt.savefig(FIG / "fig2_tuning.png"); plt.close()
tuning_tbl = {n: {"best_cv_auc": float(g.best_score_),
                  "best_params": {k: (float(v) if isinstance(v, (np.floating, float)) else int(v))
                                  for k, v in g.best_params_.items()}} for n, g in gs.items()}

# Fig 3: nested-CV metrics
mets = ["accuracy", "recall", "specificity", "f1", "roc_auc"]; mlab = ["Accuracy", "Recall", "Specificity", "F1", "ROC AUC"]
fig, ax = plt.subplots(figsize=(8, 3.2)); w = .2
for i, n in enumerate(MODELS):
    ax.bar(np.arange(5) + (i - 1.5) * w, [nested[n][m][0] for m in mets], w,
           yerr=[nested[n][m][1] for m in mets], capsize=2, color=COL[n], label=n)
ax.set_xticks(range(5)); ax.set_xticklabels(mlab); ax.set_ylim(0, 1.12); ax.set_ylabel("Nested-CV mean ± SD")
ax.legend(ncol=4, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.14))
plt.tight_layout(); plt.savefig(FIG / "fig3_nested_cv.png"); plt.close()

# Fig 4: confusion matrices
fig, ax = plt.subplots(1, 4, figsize=(11, 2.9))
for a, n in zip(ax, MODELS):
    cm = confusion_matrix(yt, YP[n], labels=[0, 1]); a.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        a.text(j, i, v, ha="center", va="center", fontsize=12, color="white" if v > cm.max() / 2 else "black")
    a.set_xticks([0, 1]); a.set_yticks([0, 1]); a.set_xticklabels(["No", "Yes"]); a.set_yticklabels(["No", "Yes"])
    a.set_xlabel("Predicted"); a.set_title(n, fontsize=9); a.spines[:].set_visible(False)
ax[0].set_ylabel("Actual"); plt.tight_layout(); plt.savefig(FIG / "fig4_confusion.png"); plt.close()

# Fig 5: ROC + PR
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for n in MODELS:
    fpr, tpr, _ = roc_curve(yt, P[n])
    ax[0].plot(fpr, tpr, color=COL[n], lw=1.6, label=f"{n} (AUC {roc_auc_score(yt, P[n]):.3f})")
    pr_, rc_, _ = precision_recall_curve(yt, P[n])
    ax[1].plot(rc_, pr_, color=COL[n], lw=1.6, label=f"{n} (AP {average_precision_score(yt, P[n]):.3f})")
ax[0].plot([0, 1], [0, 1], "k:", lw=.8); ax[0].set_xlabel("1 − Specificity"); ax[0].set_ylabel("Sensitivity")
ax[0].set_title("ROC curves (test set)"); ax[0].legend(fontsize=7, frameon=False, loc="lower right")
ax[1].axhline(yt.mean(), c="k", ls=":", lw=.8); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
ax[1].set_title("Precision–recall curves (test set)"); ax[1].set_ylim(0.4, 1.02)
ax[1].legend(fontsize=7, frameon=False, loc="lower left")
plt.tight_layout(); plt.savefig(FIG / "fig5_roc_pr.png"); plt.close()

# Fig 6: interpretability
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].barh(coefs.index, coefs.values, color=np.where(coefs.values > 0, "#E74C3C", "#2E86AB")); ax[0].axvline(0, c="k", lw=.6)
ax[0].set_title("Logistic regression coefficients"); ax[0].tick_params(axis="y", labelsize=7)
top = imp.head(8)[::-1]
ax[1].barh(top.index, top.values, color=COL["Decision tree"]); ax[1].set_title("Decision-tree feature importance")
ax[1].tick_params(axis="y", labelsize=7)
plt.tight_layout(); plt.savefig(FIG / "fig6_interpretability.png"); plt.close()

# Fig 7: tree
fig, ax = plt.subplots(figsize=(11, 4.6))
ann = plot_tree(tree, feature_names=tnames, class_names=["No", "Yes"], filled=True, impurity=False,
          max_depth=3, fontsize=7, ax=ax, rounded=True)
for a in ann:
    if a.get_text().startswith("samples = 32"):
        x, y = a.get_position()
        a.set_position((x + 0.045, y))
plt.savefig(FIG / "fig7_tree.png"); plt.close()

# Fig 8: cost curves
fig, ax = plt.subplots(figsize=(5.2, 3.2))
for n in thr:
    ax.plot(grid_t, thr[n]["oof_cost_curve"], color=COL[n], label=n)
    ax.plot(thr[n]["t_star"], min(thr[n]["oof_cost_curve"]), "o", color=COL[n])
ax.axvline(C_FP / (C_FP + C_FN), c="k", ls=":", lw=.8)
ax.set_xlabel("Decision threshold"); ax.set_ylabel(f"OOF cost ({C_FN}·FN + {C_FP}·FP), training")
ax.legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig(FIG / "fig8_cost.png"); plt.close()

# ------------------------------------------------------------------ save
out = {"info": info,
       "nested_cv": {n: {k: list(v) for k, v in d.items()} for n, d in nested.items()},
       "best_params": {n: {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in p.items()}
                       for n, p in best.items()},
       "tuning": tuning_tbl, "test": test_tab.round(4).to_dict("index"), "ci": ci, "mcnemar": mcn,
       "auc_diff": auc_diff, "cost": cost_rows, "thresholds": {n: v["t_star"] for n, v in thr.items()},
       "ablation": abl,
       "timing_ms": {n: {"grid_search_s": tune_time[n], "fit_ms": 1e3 * fit_time[n], "predict_ms": 1e3 * pred_time[n]}
                     for n in MODELS},
       "lr_coef": coefs.round(3).to_dict(), "tree_importance": imp.head(8).round(3).to_dict(),
       "tree_depth": int(tree.get_depth()), "tree_leaves": int(tree.get_n_leaves()),
       "grids": {"C": C_GRID, "depth": DEPTH_GRID, "leaf": LEAF_GRID, "k": K_GRID},
       "costs": {"C_FN": C_FN, "C_FP": C_FP}}
(RES / "results.json").write_text(json.dumps(out, indent=2, default=float))
pd.DataFrame({n: {m: f"{v[0]:.3f} ± {v[1]:.3f}" for m, v in d.items()} for n, d in nested.items()}).T.to_csv(RES / "nested_cv.csv")
print(test_tab.round(3)); print(json.dumps(out["cost"], indent=1, default=float)); print(abl); print(mcn); print(auc_diff)
