import argparse
import re
from pathlib import Path
from itertools import combinations

ROOT = Path(__file__).parent
DATA_DIR = ROOT / ".." / "data"


def extrair_speaker(nome: str, regex=None, sep=None) -> str | None:
    if regex:
        m = re.match(regex, nome)
        return m.group(1) if m else None
    if sep:
        return nome.split(sep)[0]
    if "-" in nome:
        return nome.split("-")[0]
    return nome.rsplit(".", 1)[0]


def listar_speakers(pasta: Path, regex, sep) -> dict[str, list[Path]]:
    resultado: dict[str, list[Path]] = {}
    if not pasta.is_dir():
        return resultado
    for f in sorted(pasta.rglob("*.wav")):
        spk = extrair_speaker(f.name, regex=regex, sep=sep)
        if spk is None:
            continue
        resultado.setdefault(spk, []).append(f)
    return resultado


def jaccard(a: set, b: set) -> float:
    inter = a & b
    uniao = a | b
    return len(inter) / len(uniao) if uniao else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regex", default=None,
                    help="Regex com grupo 1 capturando o speaker")
    ap.add_argument("--sep", default=None,
                    help="Separador simples (ex.: '-')")
    ap.add_argument("--splits", nargs="+",
                    default=["train", "test", "verification"])
    ap.add_argument("--subpastas", nargs="+",
                    default=["positive", "negative"],
                    help="Subpastas de cada split a varrer")
    args = ap.parse_args()

    #coleta speakers
    por_split: dict[str, set] = {}
    arquivos_por_split: dict[str, dict] = {}

    for split in args.splits:
        spks_split = set()
        arqs_split = {}
        for sub in args.subpastas:
            pasta = DATA_DIR / split / sub
            spk_map = listar_speakers(pasta, args.regex, args.sep)
            for spk, fs in spk_map.items():
                spks_split.add(spk)
                arqs_split.setdefault(spk, []).extend(fs)
        por_split[split] = spks_split
        arquivos_por_split[split] = arqs_split

    #tamanhos absolutos
    print("=" * 70)
    print("SPEAKERS POR SPLIT")
    print("=" * 70)
    for split in args.splits:
        n_spk = len(por_split[split])
        n_arq = sum(len(v) for v in arquivos_por_split[split].values())
        print(f"  {split:14s}  speakers={n_spk:4d}  arquivos={n_arq:6d}")

    # interseções par a par
    print("\n" + "=" * 70)
    print("INTERSEÇÕES (par a par)")
    print("=" * 70)
    for a, b in combinations(args.splits, 2):
        A, B = por_split[a], por_split[b]
        inter = A & B
        rel_a = len(inter) / len(A) if A else 0.0
        rel_b = len(inter) / len(B) if B else 0.0
        rel_uni = jaccard(A, B)
        print(f"\n  {a}  ∩  {b}")
        print(f"    |{a}|              = {len(A):5d}")
        print(f"    |{b}|              = {len(B):5d}")
        print(f"    |{a} ∩ {b}|        = {len(inter):5d}   (absoluto)")
        print(f"    |{a} ∩ {b}| / |{a}| = {rel_a:.3f}   (relativo a {a})")
        print(f"    |{a} ∩ {b}| / |{b}| = {rel_b:.3f}   (relativo a {b})")
        print(f"    Jaccard            = {rel_uni:.3f}")
        if inter and len(inter) <= 30:
            print(f"    speakers comuns    : {sorted(inter)}")
        elif inter:
            amostra = sorted(inter)[:15]
            print(f"    speakers comuns    : {amostra} ... (+{len(inter)-15})")

    # interseção tripla
    if len(args.splits) >= 3:
        tripla = set.intersection(*(por_split[s] for s in args.splits))
        print("\n" + "=" * 70)
        print("INTERSEÇÃO TRIPLA")
        print("=" * 70)
        print(f"  |{ ' ∩ '.join(args.splits) }| = {len(tripla)}")
        if tripla and len(tripla) <= 30:
            print(f"  speakers: {sorted(tripla)}")

    print("\n" + "=" * 70)
    print("RESULTADO")
    print("=" * 70)
    pares = list(combinations(args.splits, 2))
    com_sobreposicao = []
    for a, b in pares:
        inter = por_split[a] & por_split[b]
        if inter:
            com_sobreposicao.append((a, b, len(inter)))

    if not com_sobreposicao:
        print("  Nenhuma sobreposição. O split é limpo por speaker.")
    else:
        print("  Há sobreposição de speakers entre splits:")
        for a, b, n in com_sobreposicao:
            print(f"     - {a} × {b}: {n} speakers em comum")
        print("\n  Recomendação: refaça o split com GroupShuffleSplit")
        print("  agrupando por speaker (sklearn.model_selection.GroupShuffleSplit).")


if __name__ == "__main__":
    main()