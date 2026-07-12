
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

HERE = Path(__file__).resolve()
AI_ROOT = None
for parent in HERE.parents:
    if (parent / "modelExperiments").is_dir():
        AI_ROOT = parent
        break
if AI_ROOT is None:
    AI_ROOT = HERE.parents[2]

DDR = AI_ROOT / "modelExperiments" / "DDR"

AUC_CSV = DDR / "outputs_slic_auc_eval" / "reports" / \
    "slic_insertion_deletion_detailed_DDR_test.csv"

FID_CSV = DDR / "outputs_fidelity_stability_eval" / "reports" / \
    "fidelity_per_image_DDR_test.csv"

STAB_CSV = DDR / "outputs_fidelity_stability_eval" / "reports" / \
    "stability_per_image_DDR_test.csv"

OUTPUT_DIR = DDR / "outputs_significance_test"
REPORTS_DIR = OUTPUT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TXT_PATH = REPORTS_DIR / "significance_test_DDR_test.txt"
CSV_PATH = REPORTS_DIR / "significance_test_DDR_test.csv"

GC, PP, XGC = "gradcam", "gradcampp", "xgradcam"
PAIRS = [(GC, PP), (GC, XGC), (PP, XGC)]

ALPHA = 0.05
N_COMPARISONS = len(PAIRS)
ALPHA_BONF = ALPHA / N_COMPARISONS


def paired_report(rows, lines, name, a, b, label_a, label_b, higher_better=True):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b

    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0

    better_a = (diff.mean() > 0) == higher_better
    win = label_a if better_a else label_b
    sig = "DA" if (w_p < ALPHA_BONF) else "nu"

    line = (f"{name:<16} {label_a:>10} vs {label_b:<10} n={len(a):>4}  "
            f"dif medie={diff.mean():+.4f}  t p={t_p:.4f}  Wilcoxon p={w_p:.4f}  "
            f"d={d:+.3f}  -> {win}  (semnificativ: {sig})")
    print(line)
    lines.append(line)

    rows.append({
        "metric": name,
        "method_a": label_a,
        "method_b": label_b,
        "n": int(len(a)),
        "mean_diff": round(float(diff.mean()), 6),
        "t_stat": round(float(t_stat), 6),
        "t_pvalue": round(float(t_p), 6),
        "wilcoxon_stat": round(float(w_stat), 6) if not np.isnan(w_stat) else np.nan,
        "wilcoxon_pvalue": round(float(w_p), 6) if not np.isnan(w_p) else np.nan,
        "cohens_d": round(float(d), 6),
        "higher_better": higher_better,
        "winner": win,
        "alpha_bonferroni": round(ALPHA_BONF, 6),
        "significant": sig == "DA",
    })


def main():
    rows = []
    lines = []

    def emit(text=""):
        print(text)
        lines.append(text)

    emit("=" * 100)
    emit("  TEST PERECHE Grad-CAM vs Grad-CAM++ vs X-Grad-CAM")
    emit(f"  Corectie Bonferroni: {N_COMPARISONS} comparatii -> prag semnificativ p < {ALPHA_BONF:.4f}")
    emit("=" * 100)

    if AUC_CSV.exists():
        auc = pd.read_csv(AUC_CSV)
        piv = auc.pivot_table(index="index", columns="method",
                              values=["deletion_auc", "insertion_auc"]).dropna()
        for m1, m2 in PAIRS:
            paired_report(rows, lines, "Deletion AUC",
                          piv[("deletion_auc", m1)], piv[("deletion_auc", m2)],
                          m1, m2, higher_better=False)
        for m1, m2 in PAIRS:
            paired_report(rows, lines, "Insertion AUC",
                          piv[("insertion_auc", m1)], piv[("insertion_auc", m2)],
                          m1, m2, higher_better=True)
    else:
        emit(f"[!] Lipseste {AUC_CSV.name} - ruleaza intai compare_slic_auc.py")

    if FID_CSV.exists():
        fid = pd.read_csv(FID_CSV)
        for col in ["top5", "top10", "top15", "top20"]:
            p = fid.pivot_table(index="index", columns="method", values=col).dropna()
            for m1, m2 in PAIRS:
                paired_report(rows, lines, f"Fidelitate {col}",
                              p[m1], p[m2], m1, m2, higher_better=True)
    else:
        emit(f"[i] Lipseste {FID_CSV.name}.")
        emit("    Ruleaza compare_fidelity_stability.py ca sa generezi CSV-ul per-imagine.")

    if STAB_CSV.exists():
        stab = pd.read_csv(STAB_CSV)
        for col in ["corr", "ssim"]:
            p = stab.pivot_table(index="index", columns="method", values=col).dropna()
            for m1, m2 in PAIRS:
                paired_report(rows, lines, f"Stabilitate {col}",
                              p[m1], p[m2], m1, m2, higher_better=True)
    else:
        emit(f"[i] Lipseste {STAB_CSV.name} - ruleaza intai compare_fidelity_stability.py")

    emit("=" * 100)

    with open(TXT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    if rows:
        pd.DataFrame(rows).to_csv(CSV_PATH, index=False)

    print("\nSaved TXT:", TXT_PATH)
    print("Saved CSV:", CSV_PATH)


if __name__ == "__main__":
    main()