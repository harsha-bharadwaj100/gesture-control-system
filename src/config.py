"""
Project Configuration: Single-Channel sEMG Gesture Recognition System
Centralizes cohort parameters, sampling rates, gestures, and filter configurations.
"""

# Hardware & Sampling Specs
SAMPLE_RATE = 1000  # Hz
CHANNEL = "Dev1/ai0"
TOTAL_TRIAL_DURATION = 4.0  # seconds (4000 samples)

# 5 Active Gesture Classes (Target Domain)
GESTURES = ["Fist", "Open palm", "Pinch", "Wrist up", "Wrist down"]

# Active Contraction Plateau (excluding reaction latency and early release)
ACTIVE_START_SEC = 0.6  # sample 600
ACTIVE_END_SEC = 3.4    # sample 3400

# Sliding Window Parameters
WINDOW_SIZE_MS = 500    # 500 ms = 500 samples
WINDOW_STEP_MS = 100    # 100 ms stride (80% overlap)

# Filter Specifications
NOTCH_FREQ = 50.0       # Hz (AC mains powerline)
NOTCH_Q = 30.0          # Quality factor
BANDPASS_LOW = 20.0     # Hz (Motion artifact cutoff)
BANDPASS_HIGH = 450.0   # Hz (Nyquist guard)
BANDPASS_ORDER = 4      # 4th order Butterworth (effective 8th order with zero-phase)
FILTER_PADLEN = 150     # Zero-phase reflection padding samples

# Validated Cohort Definitions
ALL_PARTICIPANTS = [
    "Arundhati", "Tanishka_", "Thrisha", "chinmaya", "guru_raj",
    "halu", "harsha", "harshitha_nc", "keerthana_k", "keerthi",
    "lakshmi_maam", "priyanka_maam", "samiksha", "vandith", "vishal"
]

EXCLUDED_PARTICIPANTS = {
    "keerthi": "Electrode contact failure (flatline RMS <= 6.1 mV, DCR = 1.21x)",
    "vishal": "Electrode contact failure (flatline RMS <= 6.8 mV, DCR = 1.25x)",
    "priyanka_maam": "Protocol task cue inversion (rested during active prompt, contracted during rest)"
}

CLEAN_COHORT = [p for p in ALL_PARTICIPANTS if p not in EXCLUDED_PARTICIPANTS]
