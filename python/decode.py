import pickle
import numpy as np
from config import MODEL_DIR, THRESHOLD
from mfcc import extract_mfcc


def load_bundle():
    with open(MODEL_DIR / "hmm_bundle.pkl", "rb") as f:
        return pickle.load(f)


def score_audio(bundle, path: str) -> dict:
    X = extract_mfcc(path)
    T = X.shape[0]
    scores = {
        "word": bundle["word"].score(X) / T,
        "filler": bundle["filler"].score(X) / T,
    }
    if bundle.get("silence") is not None:
        scores["silence"] = bundle["silence"].score(X) / T
    return scores


def decide(scores: dict, threshold: float = THRESHOLD):
    word = scores["word"]
    competitors = [scores["filler"]]
    if "silence" in scores:
        competitors.append(scores["silence"])
    delta = word - max(competitors)
    return delta > threshold, delta


def detect(bundle, path: str, threshold: float = THRESHOLD):
    s = score_audio(bundle, path)
    detected, delta = decide(s, threshold)
    return {"detected": detected, "delta": delta, "scores": s}


if __name__ == "__main__":
    import sys
    bundle = load_bundle()
    print(detect(bundle, sys.argv[1]))