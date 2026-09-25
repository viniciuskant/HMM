import pickle
import json
import numpy as np
from pathlib import Path
from config import (DATA_DIR, MODEL_DIR, N_STATES_WORD, N_STATES_FILLER, N_STATES_SIL, N_MIX, N_ITER, USE_DELTAS, N_MFCC)
from mfcc import extract_mfcc
from hmm import DiagonalGMMHMM, estimate_self_prob

DIM = N_MFCC * 3 if USE_DELTAS else N_MFCC

def load_folder(folder: Path) -> list[np.ndarray]:
    files = sorted(folder.glob("*.wav"))
    if not files:
        return []
    feats = []
    for f in files:
        feats.append(extract_mfcc(str(f)))
    return feats


def train_single_hmm(feats_list, n_states, n_mix, seed=0):
    #pega o número de frames de cada mfcc
    lengths = [f.shape[0] for f in feats_list] 

    #concatena todas as mfccs
    X = np.vstack(feats_list)
    avg_frames = float(np.mean(lengths))
    self_prob = estimate_self_prob(avg_frames, n_states)
    print(f"    seqs={len(feats_list)} frames_mean={avg_frames:.1f} "
          f"self_prob={self_prob:.3f}")

    model = DiagonalGMMHMM(
        n_components=n_states,
        n_mix=n_mix,
        n_iter=N_ITER,
        tol=1e-4,
        seed=seed,
        self_prob=self_prob,
    )
    model.fit(X, lengths)
    return model

def hmm_to_dict(h):
    return {
        "n_states":   int(h.n_components),
        "n_mix":      int(h.n_mix),
        "n_features": int(h.means_.shape[2]),
        "startprob":  h.startprob_.astype(np.float32).tolist(),
        "transmat":   h.transmat_ .astype(np.float32).tolist(),
        "weights":    h.weights_  .astype(np.float32).tolist(),
        "means":      h.means_    .astype(np.float32).tolist(),
        "covars":     h.covars_   .astype(np.float32).tolist(),
    }

def main():
    print("Carregando dados de treino...")

    pos = load_folder(DATA_DIR / "train" / "positive")
    neg = load_folder(DATA_DIR / "train" / "negative")
    sil_folder = DATA_DIR / "train" / "silence"

    sil = load_folder(sil_folder) if sil_folder.exists() else []

    print(f"Positivos: {len(pos)} | Negativos: {len(neg)} | Silêncio: {len(sil)}")
    if not pos or not neg:
        raise SystemExit("Nengum dado de treino. Rode prepare_data.py primeiro.")

    print("Treinando HMM da palavra...")
    hmm_word = train_single_hmm(pos, N_STATES_WORD, N_MIX, seed=1)

    print("Treinando HMM de filler...")
    hmm_filler = train_single_hmm(neg, N_STATES_FILLER, N_MIX, seed=2)

    hmm_sil = None
    if sil:
        print("Treinando HMM de silêncio...")
        hmm_sil = train_single_hmm(sil, N_STATES_SIL, N_MIX, seed=3)

    bundle = {"word": hmm_word, "filler": hmm_filler, "silence": hmm_sil,
              "dim": DIM}
    out = MODEL_DIR / "hmm_bundle.pkl"
    with open(out, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Salvo em {out} ({out.stat().st_size/1024:.1f} KB)")

    export = {
        "dim": DIM,
        "word":    hmm_to_dict(hmm_word),
        "filler":  hmm_to_dict(hmm_filler),
        "silence": hmm_to_dict(hmm_sil) if hmm_sil is not None else None,
    }
    with open(MODEL_DIR / "hmm_bundle.json", "w") as f:
        json.dump(export, f, indent=2)
    print(f"JSON salvo em {MODEL_DIR / 'hmm_bundle.json'}")

if __name__ == "__main__":
    main()