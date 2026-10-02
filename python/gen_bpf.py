from scipy.signal import butter

SAMPLE_RATE  = 16000
HIGHPASS_HZ  = 80.0
LOWPASS_HZ   = 7000.0
FILTER_ORDER = 4

sos = butter(FILTER_ORDER,
             [HIGHPASS_HZ / (SAMPLE_RATE / 2), LOWPASS_HZ / (SAMPLE_RATE / 2)],
             btype="band", output="sos")

print("static const double BPF_SOS[][6] = {")
for row in sos:
    print("    {" + ", ".join(f"{x:.17g}" for x in row) + "},")
print("};")