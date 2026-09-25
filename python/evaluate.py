import numpy as np
from pathlib import Path
from config import DATA_DIR
from decode import load_bundle, score_audio


def collect_scores(bundle, folder: Path):
    return [score_audio(bundle, str(f)) for f in sorted(folder.glob("*.wav"))]


def delta_of(s):
    others = [s["filler"]]
    if "silence" in s:
        others.append(s["silence"])
    return s["word"] - max(others)


def metrics_at(threshold, pos_d, neg_d):
    tp = int((pos_d > threshold).sum())
    fn = int((pos_d <= threshold).sum())
    fp = int((neg_d > threshold).sum())
    tn = int((neg_d <= threshold).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1, tp, fp, fn, tn


def tune_threshold(bundle):
    val_pos = collect_scores(bundle, DATA_DIR / "verification" / "positive")
    val_neg = collect_scores(bundle, DATA_DIR / "verification" / "negative")
    if not val_pos or not val_neg:
        raise SystemExit("Faltam dados em data/verification/")

    pos_d = np.array([delta_of(s) for s in val_pos])
    neg_d = np.array([delta_of(s) for s in val_neg])
    all_d = np.concatenate([pos_d, neg_d])
    grid = np.linspace(all_d.min(), all_d.max(), 400)

    best = (-1, 0.0, 0, 0)
    for t in grid:
        p, r, f1, *_ = metrics_at(t, pos_d, neg_d)
        if f1 > best[0]:
            best = (f1, t, p, r)
    f1, t, p, r = best
    print(f"[verification] melhor F1={f1:.3f} em threshold={t:.4f} "
          f"(P={p:.3f}, R={r:.3f})")
    return t


def report_test(bundle, threshold):
    te_pos = collect_scores(bundle, DATA_DIR / "test" / "positive")
    te_neg = collect_scores(bundle, DATA_DIR / "test" / "negative")
    if not te_pos or not te_neg:
        raise SystemExit("Faltam dados em data/test/")

    pos_d = np.array([delta_of(s) for s in te_pos])
    neg_d = np.array([delta_of(s) for s in te_neg])
    p, r, f1, tp, fp, fn, tn = metrics_at(threshold, pos_d, neg_d)
    print(f"[test]         threshold={threshold:.4f} "
          f"P={p:.3f} R={r:.3f} F1={f1:.3f} "
          f"(TP={tp} FP={fp} FN={fn} TN={tn})")
    return {"threshold": threshold, "P": p, "R": r, "F1": f1}


if __name__ == "__main__":
    bundle = load_bundle()
    t = tune_threshold(bundle)
    report_test(bundle, t)
    print(f"\nUse THRESHOLD = {t:.4f} no config.py")