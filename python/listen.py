#!/usr/bin/env python3
"""
Escuta o microfone em tempo real e detecta a palavra-alvo.
Mostra feedback visual do nível e do score a cada hop.
"""
import argparse
import sys
import time
from collections import deque
from math import gcd

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

from config import SAMPLE_RATE, THRESHOLD
from decode import load_bundle
from mfcc import extract_mfcc_from_array


# ---------------- feedback visual ----------------

BAR_CHARS = " ▁▂▃▄▅▆▇█"

def bar(valor, vmin, vmax, largura=20):
    """Barra tipo ▁▂▃▄▅▆▇█ proporcional ao valor."""
    if vmax <= vmin:
        return " " * largura
    p = (valor - vmin) / (vmax - vmin)
    p = min(max(p, 0.0), 1.0)
    n = int(round(p * largura))
    if n <= 0:
        return " " * largura
    nivel = min(len(BAR_CHARS) - 1, max(1, int(round(p * (len(BAR_CHARS) - 1)))))
    return BAR_CHARS[nivel] * n + " " * (largura - n)


def formata_linha(rms, delta, detectado, contagem, thr):
    """
    Formato:
      [RMS ▁▂▃▄▅___]  delta +0.412  thr 0.499  o
    """
    rms_bar = bar(rms, 0.0, 0.05, 14)
    # delta: mapear [-1, +2] para barra
    delta_bar = bar(delta, -1.0, 2.0, 20)
    marca = "●" if detectado else "○"
    delta_str = f"{delta:+.3f}"
    return (f"[{rms_bar}] {delta_bar}  "
            f"delta={delta_str}  thr={thr:+.3f}  {marca} "
            f"({contagem})")


def obter_taxa_nativa(device=None) -> int:
    try:
        info = sd.query_devices(device, "input")
        return int(info["default_samplerate"])
    except Exception:
        return 48000


def scores_for(bundle, X):
    T = X.shape[0]
    s = {
        "word":   bundle["word"].score(X) / T,
        "filler": bundle["filler"].score(X) / T,
    }
    if bundle.get("silence") is not None:
        s["silence"] = bundle["silence"].score(X) / T
    return s


def decide(scores, thr):
    comp = [scores["filler"]]
    if "silence" in scores:
        comp.append(scores["silence"])
    delta = scores["word"] - max(comp)
    return delta > thr, delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--hop", type=float, default=0.25)
    ap.add_argument("--cooldown", type=float, default=1.0)
    ap.add_argument("--min-rms", type=float, default=0.005)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--verboso", action="store_true",
                    help="Mostra scores word/filler/silence a cada hop")
    args = ap.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    bundle = load_bundle()
    taxa_nativa = obter_taxa_nativa(args.device)

    print(f"Dispositivo      : {args.device if args.device is not None else 'padrão'}")
    print(f"Taxa nativa      : {taxa_nativa} Hz")
    print(f"Taxa do modelo   : {SAMPLE_RATE} Hz (após resample)")
    print(f"Threshold        : {args.threshold:+.4f}")
    print(f"Janela / Hop     : {args.window:.2f}s / {args.hop:.2f}s")
    print(f"RMS mínimo       : {args.min_rms}")
    print("\n┌─ Legenda ────────────────────────────────────────┐")
    print("│  ▁▂▃▄▅▆▇█  = nível do RMS (esquerda)             │")
    print("│  ▁▂▃▄▅▆▇█  = delta (direita)                     │")
    print("│  ● = detectado    ○ = não detectado              │")
    print("└──────────────────────────────────────────────────┘")
    print("\nFale ao microfone. Ctrl+C para sair.\n")

    win_n_nat = int(args.window * taxa_nativa)
    hop_n_nat = int(args.hop * taxa_nativa)
    n_hops = max(1, int(np.ceil(win_n_nat / hop_n_nat)))

    try:
        stream = sd.InputStream(
            samplerate=taxa_nativa, channels=1, dtype="float32",
            blocksize=hop_n_nat, device=args.device,
        )
    except Exception as e:
        sys.exit(f"Erro abrindo microfone: {e}")

    ring = deque(maxlen=n_hops)
    count = 0
    last_detect = 0.0
    n_baixo = 0
    ultima_impressao = 0.0
    INTERVALO_MS = 150   # limita a 6–7 atualizações por segundo

    stream.start()
    try:
        while True:
            chunk, overflowed = stream.read(hop_n_nat)
            y = chunk.flatten().astype(np.float32)
            ring.append(y)
            if len(ring) < n_hops:
                continue

            buf = np.concatenate(list(ring))
            if len(buf) > win_n_nat:
                buf = buf[-win_n_nat:]

            rms = float(np.sqrt(np.mean(buf ** 2)))

            # feedback imediato do nível (mesmo abaixo do min_rms)
            now = time.time()
            if (now - ultima_impressao) * 1000 < INTERVALO_MS:
                continue
            ultima_impressao = now

            if args.min_rms > 0 and rms < args.min_rms:
                n_baixo += 1
                linha = formata_linha(rms, 0.0, False, count, args.threshold)
                sys.stdout.write(f"\r{linha}   silêncio")
                sys.stdout.flush()
                continue

            # reamostragem
            if taxa_nativa != SAMPLE_RATE:
                g = gcd(taxa_nativa, SAMPLE_RATE)
                up, down = SAMPLE_RATE // g, taxa_nativa // g
                buf = resample_poly(buf, up, down).astype(np.float32)

            try:
                X = extract_mfcc_from_array(buf, SAMPLE_RATE)
            except Exception as e:
                sys.stdout.write(f"\r  ! erro mfcc: {e}\n")
                continue
            if X is None or X.shape[0] < 5:
                continue

            s = scores_for(bundle, X)
            detected, delta = decide(s, args.threshold)

            if detected and (now - last_detect) > args.cooldown:
                count += 1
                last_detect = now
                # imprime em nova linha para registrar no histórico
                extra = ""
                if "silence" in s:
                    extra = f"  sil={s['silence']:.3f}"
                sys.stdout.write("\r" + " " * 100 + "\r")
                print(f"● Detectado {count} vezes   "
                      f"delta={delta:+.3f}  "
                      f"word={s['word']:.3f}  "
                      f"filler={s['filler']:.3f}{extra}")
                continue

            linha = formata_linha(rms, delta, detected, count, args.threshold)
            if args.verboso:
                linha += (f"  w={s['word']:+.3f} f={s['filler']:+.3f}")
                if "silence" in s:
                    linha += f" s={s['silence']:+.3f}"
            sys.stdout.write(f"\r{linha}")
            sys.stdout.flush()

    except KeyboardInterrupt:
        sys.stdout.write("\r" + " " * 100 + "\r")
        print(f"\n\nTotal de detecções: {count}")
        print(f"Frames ignorados por baixo RMS: {n_baixo}")
    finally:
        stream.stop()
        stream.close()


if __name__ == "__main__":
    main()