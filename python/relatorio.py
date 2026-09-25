# relatorio.py
from decode import load_bundle, score_audio, decide
from config import DATA_DIR, THRESHOLD
from pathlib import Path

bundle = load_bundle()

def evaluate(split):
    rows = []
    for cls, label in [("positive", True), ("negative", False)]:
        folder = DATA_DIR / split / cls
        for f in sorted(folder.glob("*.wav")):
            s = score_audio(bundle, str(f))
            det, delta = decide(s, THRESHOLD)
            rows.append({"split": split, "arquivo": f.name,
                         "classe": cls, "esperado": label,
                         "detectado": det, "delta": delta,
                         "score_word": s["word"],
                         "score_filler": s["filler"]})
    return rows

for split in ["verification", "test"]:
    r = evaluate(split)
    n_ok = sum(1 for x in r if x["esperado"] == x["detectado"])
    print(f"[{split}] {n_ok}/{len(r)} corretos = {n_ok/len(r):.3f}")