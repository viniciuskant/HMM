import argparse
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
from joblib import Parallel, delayed
from tqdm import tqdm

ROOT = Path(__file__).parent
DATA = ROOT / ".." / "data"
UNIFICADO_DEFAULT = Path("/datasets/audios/unificado")
SR = 16000


def list_wavs(folder: Path):
    return sorted(folder.glob("*.wav"))


def load(path):
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y


def save(path: Path, y: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, SR)


def extract_speaker(path) -> str:
    name = Path(path).stem
    return name.split("-")[0] if "-" in name else name


def group_by_speaker(files):
    groups = {}
    for f in files:
        groups.setdefault(extract_speaker(f), []).append(f)
    return groups


def split_speakers(speakers, p_train, p_test, seed):
    rng = random.Random(seed)
    spks = sorted(speakers)
    rng.shuffle(spks)
    n = len(spks)
    n_train = int(round(n * p_train))
    n_test = int(round(n * p_test))
    train = set(spks[:n_train])
    test = set(spks[n_train:n_train + n_test])
    verif = set(spks[n_train + n_test:])
    return train, test, verif


def find_offset(orig, alig, hop=512):
    if len(alig) > len(orig):
        return None
    frame = 1024
    o_env = librosa.feature.rms(y=orig, frame_length=frame, hop_length=hop)[0]
    a_env = librosa.feature.rms(y=alig, frame_length=frame, hop_length=hop)[0]
    if len(a_env) > len(o_env):
        return None
    o_env = o_env - o_env.mean()
    a_env = a_env - a_env.mean()
    corr = np.correlate(o_env, a_env, mode="valid")
    if len(corr) == 0:
        return None
    best = int(np.argmax(corr))
    if corr[best] < 0.3 * (np.linalg.norm(o_env) * np.linalg.norm(a_env) + 1e-9):
        return None
    return best * hop


def cut_negatives(orig, alig, n_cuts, cut_sec, rng):
    cut_len = int(cut_sec * SR)
    if len(orig) < cut_len + len(alig):
        return []
    off = find_offset(orig, alig)
    if off is None:
        starts = [rng.randint(0, len(orig) - cut_len) for _ in range(n_cuts)]
    else:
        end = off + len(alig)
        regioes = [(0, off), (end, len(orig))]
        starts = []
        for _ in range(n_cuts):
            rng.shuffle(regioes)
            for a, b in regioes:
                if b - a >= cut_len:
                    starts.append(rng.randint(a, b - cut_len))
                    break
    return [orig[s:s + cut_len] for s in starts]


def symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())



def process_negative(args):
    f, alin_dir, n_cuts, cut_sec, seed = args
    alig_path = alin_dir / f.name
    if not alig_path.exists():
        return (None, [])
    try:
        orig = load(f)
        alig = load(alig_path)
    except Exception as e:
        return ("__erro__", (f.name, str(e)))
    rng = random.Random(seed)
    segs = cut_negatives(orig, alig, n_cuts, cut_sec, rng)
    return (extract_speaker(f), [(f, i, s) for i, s in enumerate(segs)])



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=float, default=0.30)
    ap.add_argument("--test", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-cuts", type=int, default=3)
    ap.add_argument("--cut-sec", type=float, default=1.0)
    ap.add_argument("--unificado", type=Path, default=UNIFICADO_DEFAULT)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--max-unificado", type=int, default=800)
    args = ap.parse_args()

    p_train, p_test = args.train, args.test
    p_verif = 1.0 - p_train - p_test
    if p_verif <= 0:
        sys.exit(f"Proporções inválidas: train={p_train} test={p_test}")
    print(f"Split: train={p_train:.0%} test={p_test:.0%} verif={p_verif:.0%} "
          f"(agrupado por speaker)")

    orig_dir = DATA / "original"
    alin_dir = DATA / "alinhados"
    if not orig_dir.is_dir() or not alin_dir.is_dir():
        sys.exit(f"Faltam {orig_dir} e/ou {alin_dir}")

    train_dir = DATA / "train"
    test_dir = DATA / "test"
    verif_dir = DATA / "verification"
    bases = {"train": train_dir, "test": test_dir, "verif": verif_dir}

    if args.clean:
        for d in (train_dir, test_dir, verif_dir):
            if d.exists():
                shutil.rmtree(d)

    for split in (train_dir, test_dir, verif_dir):
        (split / "positive").mkdir(parents=True, exist_ok=True)
        (split / "negative").mkdir(parents=True, exist_ok=True)
    (train_dir / "silence").mkdir(parents=True, exist_ok=True)

    #POSITIVOS: split por speaker 
    alinhados = list_wavs(alin_dir)
    if not alinhados:
        sys.exit(f"Nenhum .wav em {alin_dir}")
    spk_groups_pos = group_by_speaker(alinhados)
    print(f"[positivos] {len(alinhados)} arqs, {len(spk_groups_pos)} speakers")

    train_spks, test_spks, verif_spks = split_speakers(
        spk_groups_pos.keys(), p_train, p_test, args.seed)
    print(f"[speakers] train={len(train_spks)} test={len(test_spks)} "
          f"verif={len(verif_spks)}")

    # Validação de disjunção
    assert not (train_spks & test_spks), "vazamento train/test"
    assert not (train_spks & verif_spks), "vazamento train/verif"
    assert not (test_spks & verif_spks), "vazamento test/verif"

    def split_of(spk):
        if spk in train_spks: return "train"
        if spk in test_spks:  return "test"
        if spk in verif_spks: return "verif"
        return None

    cont_pos = {"train": 0, "test": 0, "verif": 0}
    for f in alinhados:
        dest = split_of(extract_speaker(f))
        if dest is None:
            continue
        shutil.copy2(f, bases[dest] / "positive" / f.name)
        cont_pos[dest] += 1
    print(f"[positivos] train={cont_pos['train']} test={cont_pos['test']} "
          f"verif={cont_pos['verif']}")

    # NEGATIVOS a partir do original
    originals = list_wavs(orig_dir)
    print(f"[negativos] processando {len(originals)} originais "
          f"com {args.jobs if args.jobs > 0 else 'todos os'} núcleos...")

    tasks = [(f, alin_dir, args.n_cuts, args.cut_sec, args.seed + i)
             for i, f in enumerate(originals)]
    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(process_negative)(t) for t in tqdm(tasks, desc="negativos")
    )

    cont_neg = {"train": 0, "test": 0, "verif": 0, "skip": 0}
    for spk, items in results:
        if spk == "__erro__":
            print(f"  ! erro em {items[0]}: {items[1]}")
            continue
        if spk is None:
            continue
        dest = split_of(spk)
        if dest is None:
            cont_neg["skip"] += len(items)
            continue
        for f, i, seg in items:
            out = bases[dest] / "negative" / f"{f.stem}_neg{i:02d}.wav"
            save(out, seg)
            cont_neg[dest] += 1
    print(f"[negativos] train={cont_neg['train']} test={cont_neg['test']} "
          f"verif={cont_neg['verif']} (skip={cont_neg['skip']})")

    #NEGATIVOS do unificado agrupado por pasta
    if args.max_unificado > 0 and args.unificado.is_dir():
        uni_groups = {}
        for f in args.unificado.rglob("*.wav"):
            spk = f.parent.name          # pasta = speaker
            uni_groups.setdefault(spk, []).append(f)

        if uni_groups:
            print(f"[unificado] {len(uni_groups)} speakers, "
                  f"{sum(len(v) for v in uni_groups.values())} arquivos")

            u_tr, u_te, u_ve = split_speakers(
                uni_groups.keys(), p_train, p_test, args.seed + 1)

            def collect(spks):
                fs = []
                for s in spks:
                    fs.extend(uni_groups[s])
                return fs

            tr = collect(u_tr)
            te = collect(u_te)
            ve = collect(u_ve)

            total = len(tr) + len(te) + len(ve)
            if total > args.max_unificado:
                rng = random.Random(args.seed)
                rng.shuffle(tr); rng.shuffle(te); rng.shuffle(ve)
                n_tr = int(round(args.max_unificado * p_train))
                n_te = int(round(args.max_unificado * p_test))
                n_ve = args.max_unificado - n_tr - n_te
                tr, te, ve = tr[:n_tr], te[:n_te], ve[:n_ve]

            for f in tqdm(tr, desc="uni->train"):
                symlink(f, train_dir / "negative" / f"uni_{f.parent.name}_{f.name}")
            for f in tqdm(te, desc="uni->test"):
                symlink(f, test_dir / "negative" / f"uni_{f.parent.name}_{f.name}")
            for f in tqdm(ve, desc="uni->verif"):
                symlink(f, verif_dir / "negative" / f"uni_{f.parent.name}_{f.name}")
            print(f"[unificado] train={len(tr)} test={len(te)} verif={len(ve)}")

    # SILÊNCIO 
    n_sil = 0
    for f in originals:
        if split_of(extract_speaker(f)) != "train":
            continue
        try:
            y = load(f)
        except Exception:
            continue
        head = y[:int(0.3 * SR)]
        if np.sqrt(np.mean(head ** 2)) < 0.01:
            save(train_dir / "silence" / f"s_{f.stem}.wav", head)
            n_sil += 1
    print(f"[silêncio] trechos: {n_sil}")

    #VERIFICAÇÃO FINAL DOS FALANTES
    print("\n=== Verificação de speakers ===")
    spk_por_split = {}
    for split in ("train", "test", "verification"):
        spks = set()
        for cls in ("positive", "negative", "silence"):
            d = DATA / split / cls
            if not d.is_dir():
                continue
            for f in d.glob("*.wav"):
                spks.add(extract_speaker(f))
        spk_por_split[split] = spks
        print(f"  {split:13s}: {len(spks)} speakers")

    t = spk_por_split["train"]
    e = spk_por_split["test"]
    v = spk_por_split["verification"]
    print(f"\n  train ∩ test         = {len(t & e)}")
    print(f"  train ∩ verification = {len(t & v)}")
    print(f"  test  ∩ verification = {len(e & v)}")
    print(f"  train ∩ test ∩ verif = {len(t & e & v)}")
    
    print("\n=== Resumo de arquivos ===")
    for split in ("train", "test", "verification"):
        for cls in ("positive", "negative", "silence"):
            d = DATA / split / cls
            if d.is_dir():
                k = len(list(d.glob("*.wav")))
                print(f"  {split:13s}/{cls:9s}: {k:5d} wavs")


if __name__ == "__main__":
    main()