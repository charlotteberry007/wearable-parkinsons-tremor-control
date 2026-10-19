"""
feature_extractor.py  –  Scale-invariant 18-feature IMU extractor
==================================================================
All features are ratios, frequencies, or normalised statistics —
NO absolute magnitudes.  This makes the features independent of
the physical unit or LSB-scale of the sensor so the same model
works for:
  • Training dataset  (±2g / ±2000 dps, 16.4 LSB/dps)
  • ESP32 live data   (±2g / ±250  dps, 131  LSB/dps)

Feature index map (18 features)
---------------------------------
 0  acc_peak_freq          Dominant frequency of |accel| signal  [Hz]
 1  acc_tremor_power_ratio 3.5–7.5 Hz fraction of 0.5–20 Hz band (ratio)
 2  acc_spectral_entropy   Normalised Shannon entropy of PSD (low=rhythmic)
 3  acc_spectral_flatness  Geometric/arithmetic mean ratio of PSD (low=tonal)
 4  acc_harmonic_ratio     Power at 2×peak_freq / power at peak_freq
 5  acc_zcr                Zero-crossing rate of DC-removed |accel|  [crossings/s]
 6  acc_cv                 Coefficient of variation of |accel| (std/mean)
 7  acc_skewness           Skewness of |accel| samples
 8  acc_kurtosis           Excess kurtosis of |accel| samples
 9  acc_jerk_zcr           ZCR of derivative of |accel|  [crossings/s]
10  gyro_peak_freq         Dominant frequency of |gyro|  [Hz]
11  gyro_tremor_power_ratio 3.5–7.5 Hz fraction of 0.5–20 Hz band (ratio)
12  gyro_spectral_entropy  Normalised Shannon entropy of gyro PSD
13  gyro_zcr               Zero-crossing rate of DC-removed |gyro|  [crossings/s]
14  gyro_cv                Coefficient of variation of |gyro| (std/mean)
15  gyro_skewness          Skewness of |gyro| samples
16  freq_match             min(acc_peak, gyro_peak)/max → 1.0 when frequencies agree
17  total_tremor_evidence  acc_tpr * gyro_tpr * (1 - acc_spectral_flatness)
"""

import numpy as np
from scipy.signal import welch, butter, filtfilt
from scipy.stats import skew, kurtosis as scipy_kurtosis

# ─── constants ────────────────────────────────────────────────────────────────
FS = 100            # sample rate [Hz]  (ESP32 firmware targets 100 Hz)
WINDOW_S = 2.0      # window length [s]
WINDOW_N = int(FS * WINDOW_S)   # 200 samples
TREMOR_LO = 3.5     # tremor band lower bound [Hz]
TREMOR_HI = 7.5     # tremor band upper bound [Hz]
BROAD_LO  = 0.5     # broad band lower bound [Hz]
BROAD_HI  = 20.0    # broad band upper bound [Hz]
N_FEATURES = 18

# ─── helpers ──────────────────────────────────────────────────────────────────

def _bandpower(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    """Integrate PSD between lo and hi Hz (trapezoidal)."""
    mask = (freqs >= lo) & (freqs <= hi)
    if mask.sum() < 2:
        return 0.0
    _integrate = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    return float(_integrate(psd[mask], freqs[mask]))


def _spectral_entropy(psd: np.ndarray) -> float:
    """Normalised Shannon entropy of a PSD (0 = single tone, 1 = white noise)."""
    p = psd / (psd.sum() + 1e-12)
    h = -np.sum(p * np.log(p + 1e-12))
    return float(h / np.log(len(p) + 1e-12))


def _spectral_flatness(psd: np.ndarray) -> float:
    """Wiener entropy: geometric mean / arithmetic mean of PSD (0 = tonal, 1 = flat)."""
    eps = 1e-12
    gm = np.exp(np.mean(np.log(psd + eps)))
    am = np.mean(psd) + eps
    return float(np.clip(gm / am, 0.0, 1.0))


def _zcr(signal: np.ndarray, fs: float = FS) -> float:
    """Zero-crossing rate in crossings/second (after DC removal)."""
    dc = signal - signal.mean()
    zc = np.sum(np.diff(np.sign(dc)) != 0)
    return float(zc / (len(signal) / fs))


def _cv(signal: np.ndarray) -> float:
    """Coefficient of variation = std / |mean|  (clipped to [0, 10])."""
    mu = np.abs(signal.mean())
    if mu < 1e-6:
        return 0.0
    return float(np.clip(signal.std() / mu, 0.0, 10.0))


def _multi_axis_linear_psd(signals, fs: float = FS):
    """
    Compute total linear PSD as the sum of PSDs across linear DC-removed channels.
    This preserves the true oscillation frequency (3.5 - 7.5 Hz) and avoids
    the frequency doubling / rectification that happens when computing the PSD
    of Euclidean vector norms like sqrt(x^2 + y^2 + z^2).
    """
    nperseg = min(len(signals[0]), 128)
    tot_psd = None
    freqs = None
    for sig in signals:
        dc = sig - sig.mean()
        f, p = welch(dc, fs=fs, nperseg=nperseg, window='hann')
        if tot_psd is None:
            tot_psd = p.copy()
            freqs = f
        else:
            tot_psd += p

    mask = (freqs >= BROAD_LO) & (freqs <= BROAD_HI)
    if mask.sum() == 0:
        return 0.0, freqs, tot_psd
    peak_idx = np.argmax(tot_psd[mask])
    peak_freq = float(freqs[mask][peak_idx])
    return peak_freq, freqs, tot_psd


def _dominant_axis_zcr(signals, fs: float = FS) -> float:
    """Zero-crossing rate of the highest-variance DC-removed axis."""
    max_var = -1.0
    best = signals[0]
    for s in signals:
        v = float(np.var(s))
        if v > max_var:
            max_var = v
            best = s
    return _zcr(best, fs)


# ─── public API ───────────────────────────────────────────────────────────────

def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Extract 18 scale-invariant features from a single IMU window.

    Parameters
    ----------
    window : np.ndarray, shape (N, 6)
        Columns: [aX, aY, aZ, gX, gY, gZ]  in raw LSB units.
        N must be >= 16; ideally WINDOW_N (200 samples at 100 Hz).

    Returns
    -------
    feats : np.ndarray, shape (18,)
        Float features; NaN-free (any NaN replaced with 0.0).
    """
    if window.shape[0] < 16:
        return np.zeros(N_FEATURES, dtype=np.float32)

    ax, ay, az = window[:, 0], window[:, 1], window[:, 2]
    gx, gy, gz = window[:, 3], window[:, 4], window[:, 5]

    # ── accel linear multi-axis spectral ───────────────────────────────────
    acc_peak_freq, a_freqs, a_psd = _multi_axis_linear_psd([ax, ay, az])

    a_broad = _bandpower(a_freqs, a_psd, BROAD_LO, BROAD_HI)
    a_tremor = _bandpower(a_freqs, a_psd, TREMOR_LO, TREMOR_HI)
    acc_tpr = float(a_tremor / (a_broad + 1e-12))

    acc_spec_entropy = _spectral_entropy(a_psd)
    acc_spec_flatness = _spectral_flatness(a_psd)

    # harmonic ratio: power at 2xpeak / power at peak (+/-0.5 Hz bands)
    peak2 = acc_peak_freq * 2.0
    h1 = _bandpower(a_freqs, a_psd, acc_peak_freq - 0.5, acc_peak_freq + 0.5)
    h2 = _bandpower(a_freqs, a_psd, peak2 - 0.5, peak2 + 0.5)
    acc_harmonic_ratio = float(h2 / (h1 + 1e-12))

    acc_mag = np.sqrt(ax**2 + ay**2 + az**2).astype(np.float64)
    acc_zcr = _dominant_axis_zcr([ax, ay, az])
    acc_cv = _cv(acc_mag)
    acc_skew = float(skew(acc_mag))
    acc_kurt = float(scipy_kurtosis(acc_mag, fisher=True))

    # jerk ZCR (derivative of dominant dynamic accel axis)
    acc_jerk_zcr = _dominant_axis_zcr([np.diff(ax), np.diff(ay), np.diff(az)])

    # ── gyro linear multi-axis spectral ────────────────────────────────────
    gx_dc = gx - gx.mean()
    gy_dc = gy - gy.mean()
    gz_dc = gz - gz.mean()
    gyro_mag = np.sqrt(gx_dc**2 + gy_dc**2 + gz_dc**2).astype(np.float64)

    gyro_peak_freq, g_freqs, g_psd = _multi_axis_linear_psd([gx_dc, gy_dc, gz_dc])

    g_broad = _bandpower(g_freqs, g_psd, BROAD_LO, BROAD_HI)
    g_tremor = _bandpower(g_freqs, g_psd, TREMOR_LO, TREMOR_HI)
    gyro_tpr = float(g_tremor / (g_broad + 1e-12))

    gyro_spec_entropy = _spectral_entropy(g_psd)
    gyro_zcr = _dominant_axis_zcr([gx_dc, gy_dc, gz_dc])
    gyro_cv = _cv(gyro_mag + 1.0)
    gyro_skew = float(skew(gyro_mag))

    # ── cross-modal features ────────────────────────────────────────────────
    denom = max(acc_peak_freq, gyro_peak_freq, 0.01)
    numer = min(acc_peak_freq, gyro_peak_freq)
    freq_match = float(numer / denom)

    total_tremor_evidence = float(acc_tpr * gyro_tpr * (1.0 - acc_spec_flatness))

    # ── assemble & sanitise ─────────────────────────────────────────────────
    feats = np.array([
        acc_peak_freq,          # 0
        acc_tpr,                # 1
        acc_spec_entropy,       # 2
        acc_spec_flatness,      # 3
        acc_harmonic_ratio,     # 4
        acc_zcr,                # 5
        acc_cv,                 # 6
        acc_skew,               # 7
        acc_kurt,               # 8
        acc_jerk_zcr,           # 9
        gyro_peak_freq,         # 10
        gyro_tpr,               # 11
        gyro_spec_entropy,      # 12
        gyro_zcr,               # 13
        gyro_cv,                # 14
        gyro_skew,              # 15
        freq_match,             # 16
        total_tremor_evidence,  # 17
    ], dtype=np.float32)

    # replace any NaN / Inf with 0
    feats = np.where(np.isfinite(feats), feats, 0.0).astype(np.float32)
    return feats


FEATURE_NAMES = [
    "acc_peak_freq",
    "acc_tremor_power_ratio",
    "acc_spectral_entropy",
    "acc_spectral_flatness",
    "acc_harmonic_ratio",
    "acc_zcr",
    "acc_cv",
    "acc_skewness",
    "acc_kurtosis",
    "acc_jerk_zcr",
    "gyro_peak_freq",
    "gyro_tremor_power_ratio",
    "gyro_spectral_entropy",
    "gyro_zcr",
    "gyro_cv",
    "gyro_skewness",
    "freq_match",
    "total_tremor_evidence",
]


# ─── self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    t = np.arange(WINDOW_N) / FS

    # Simulated 5 Hz tremor window
    ax = 2000 * np.sin(2 * np.pi * 5.0 * t) + rng.normal(0, 200, WINDOW_N)
    ay = 1500 * np.sin(2 * np.pi * 5.0 * t + 0.5) + rng.normal(0, 200, WINDOW_N)
    az = rng.normal(-13338, 300, WINDOW_N)
    gx = rng.normal(5002, 50, WINDOW_N)   # DC bias (training-scale)
    gy = 800 * np.sin(2 * np.pi * 5.0 * t) + rng.normal(-239, 100, WINDOW_N)
    gz = 500 * np.sin(2 * np.pi * 5.0 * t) + rng.normal(275, 100, WINDOW_N)

    window = np.stack([ax, ay, az, gx, gy, gz], axis=1)
    feats = extract_features(window)

    print("=== Tremor window ===")
    for name, val in zip(FEATURE_NAMES, feats):
        print(f"  {name:<30s}: {val:.4f}")
    print(f"\nacc_peak_freq={feats[0]:.2f} Hz  (expected ~5 Hz)")
    print(f"acc_tremor_power_ratio={feats[1]:.3f}  (expected >0.5)")

    # Simulated voluntary motion (1.5 Hz, large amplitude)
    ax2 = 5000 * np.sin(2 * np.pi * 1.5 * t) + rng.normal(0, 300, WINDOW_N)
    ay2 = 4000 * np.sin(2 * np.pi * 1.5 * t) + rng.normal(0, 300, WINDOW_N)
    az2 = rng.normal(-13338, 400, WINDOW_N)
    gy2 = 2000 * np.sin(2 * np.pi * 1.5 * t) + rng.normal(-239, 200, WINDOW_N)
    gz2 = 1500 * np.sin(2 * np.pi * 1.5 * t) + rng.normal(275, 200, WINDOW_N)
    window2 = np.stack([ax2, ay2, az2, gx, gy2, gz2], axis=1)
    feats2 = extract_features(window2)
    print("\n=== Voluntary motion window ===")
    print(f"  acc_peak_freq={feats2[0]:.2f} Hz  (expected ~1.5 Hz)")
    print(f"  acc_tremor_power_ratio={feats2[1]:.3f}  (expected <0.3)")
