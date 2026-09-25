from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / ".." / "data"
MODEL_DIR = ROOT / ".." / "model"
MODEL_DIR.mkdir(exist_ok=True)

# Áudio
SAMPLE_RATE = 16000
FRAME_LEN = 0.025
FRAME_HOP = 0.010
N_MFCC = 12
N_FFT = 512
N_MELS = 40
USE_DELTAS = False

# HMM
N_STATES_WORD = 8
N_STATES_FILLER = 12
N_STATES_SIL = 3
N_MIX = 8
N_ITER = 30

# Decodificação
THRESHOLD = 0.5420

# Splits
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
VERIF_DIR = DATA_DIR / "verification"