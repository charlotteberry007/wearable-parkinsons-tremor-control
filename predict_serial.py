"""
predict_serial.py  -  Real-time Parkinson's tremor detection & motor control
==============================================================================
* Reads raw IMU data from ESP32 over serial (mode 'r' -> CSV format)
* Extracts 18 scale-invariant features every ~2 s window
* Runs XGBoost + RandomForest ensemble
* Computes tremor intensity -> maps to PWM duty cycle (variable motor speed)
* Sends 'V'+<intensity_byte> then 'G' to ESP32 for antiphase motor suppression
* Sends 'D' to ESP32 when no tremor (safe state)

Usage:
    python predict_serial.py                    # auto-detect port
    python predict_serial.py --port COM5        # explicit port
    python predict_serial.py --simulate         # offline test (no serial)

ESP32 firmware commands (firmware v3.7):
    b'r'  -> enter raw CSV mode
    b'V'  -> next byte sets motor intensity (PWM 0-240)
    b'G'  -> enable ML phase-pulse antiphase feedback (motors ON)
    b'D'  -> disable ML gate (motors OFF / safe)
    b'S'  -> emergency stop

Motor Speed / Intensity Mapping:
    The motor intensity varies with tremor severity. Computed from:
      - ML ensemble probability (p_ml)
      - Accelerometer tremor power ratio (acc_tpr)
      - Tremor evidence score (total_tremor_evidence)
    Mapped to PWM duty: 80 (gentle) to 220 (strong) range.
"""

import sys
import os
import time
import argparse
import threading
import queue
import numpy as np

# -- Lazy serial import (not needed for --simulate) --
try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

import joblib
from feature_extractor import (
    extract_features, FEATURE_NAMES, WINDOW_N, FS, N_FEATURES
)

WORKSPACE = os.path.dirname(os.path.abspath(__file__))

# --- configuration ---
BAUD_RATE         = 115200
SERIAL_TIMEOUT    = 2.0    # seconds

STEP_N            = 50     # samples to advance per evaluation (50% overlap)
STREAK_TRIGGER    = 2      # consecutive positive windows before motor fires
STREAK_RELEASE    = 3      # consecutive negative windows before motor stops

# 3-gate thresholds
ACC_CV_REST_THR   = 0.02   # below -> resting, skip ML
PEAK_FREQ_VOL_THR = 2.8    # Hz below -> voluntary/slow, skip ML
ML_PROB_THR       = 0.40   # relaxed ensemble probability threshold
PHYS_FREQ_LO      = 3.0    # Hz - physiological tremor lower bound
PHYS_FREQ_HI      = 8.0    # Hz - physiological tremor upper bound
PHYS_TPR_THR      = 0.18   # acc_tremor_power_ratio lower bound

# Motor intensity mapping
PWM_MIN           = 80     # minimum PWM duty (just overcomes motor friction)
PWM_MAX           = 220    # maximum PWM duty (safe for 3V ERM motors)
PWM_OFF           = 0      # motors off

# --- model loading ---

def load_models():
    scaler_path = os.path.join(WORKSPACE, "scaler.pkl")
    xgb_path    = os.path.join(WORKSPACE, "xgb_model.pkl")
    rf_path     = os.path.join(WORKSPACE, "rf_model.pkl")

    missing = [p for p in [scaler_path, xgb_path, rf_path] if not os.path.exists(p)]
    if missing:
        print("[ERROR] Model files not found:")
        for m in missing:
            print(f"  {m}")
        print("  Run: python train_ensemble.py")
        sys.exit(1)

    scaler    = joblib.load(scaler_path)
    xgb_model = joblib.load(xgb_path)
    rf_model  = joblib.load(rf_path)
    print(f"[OK] Models loaded from {WORKSPACE}")
    return scaler, xgb_model, rf_model


# --- serial port helpers ---

def auto_detect_port():
    if not HAS_SERIAL:
        return None
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if any(kw in (p.description or "").upper()
               for kw in ("CP210", "CH340", "FTDI", "USB", "UART", "ESP")):
            print(f"[PORT] Auto-detected: {p.device}  ({p.description})")
            return p.device
    if ports:
        print(f"[PORT] Using first available port: {ports[0].device}")
        return ports[0].device
    return None


def open_serial(port: str):
    ser = serial.Serial(port, BAUD_RATE, timeout=SERIAL_TIMEOUT)
    time.sleep(2.0)     # allow ESP32 to boot / reset
    ser.reset_input_buffer()
    # Switch to raw CSV mode
    ser.write(b'r')
    time.sleep(0.1)
    print(f"[SERIAL] Opened {port} @ {BAUD_RATE} baud, sent 'r' (raw CSV mode)")
    return ser


# --- serial reader thread ---

def serial_reader(ser, data_queue: queue.Queue, stop_event: threading.Event):
    """
    Background thread: reads lines from serial, parses IMU CSV rows,
    puts parsed numpy arrays [aX,aY,aZ,gX,gY,gZ] into data_queue.

    Expected firmware format (raw CSV mode):
        ax*4, ay*4, az*4, gx, gy, gz          (6 integers)
        or: timestamp, ax*4, ay*4, az*4, gx, gy, gz  (7 integers)
    Lines starting with '#' or '[' are comments -> skip.
    """
    while not stop_event.is_set():
        try:
            line = ser.readline().decode("utf-8", errors="replace").strip()
        except Exception:
            continue

        if not line or line[0] in ('#', '['):
            continue

        parts = line.split(',')
        try:
            vals = [float(v.strip()) for v in parts]
        except ValueError:
            continue

        if len(vals) == 6:
            data_queue.put(np.array(vals, dtype=np.float32))
        elif len(vals) == 7:
            data_queue.put(np.array(vals[1:], dtype=np.float32))  # strip timestamp


# --- simulate IMU data ---

def simulate_imu_stream(data_queue: queue.Queue, stop_event: threading.Event):
    """
    Generates synthetic IMU samples at FS Hz.
    Cycles through: resting -> voluntary -> tremor with varying intensity -> repeat
    Uses TRAINING-SCALE values (accel 16384 LSB/g, gyro +/-2000 dps).
    """
    rng = np.random.default_rng(99)
    t = 0.0
    dt = 1.0 / FS
    phase = 0.0

    # name, duration_s, freq_hz, amp_accel, amp_gyro
    scenario = [
        ("RESTING",            5.0,  0.0,    0,     0),
        ("VOLUNTARY 1.5Hz",    8.0,  1.5, 8000,  6000),
        ("TREMOR-MILD 5Hz",    8.0,  5.2,  800,   500),    # mild -> low PWM
        ("TREMOR-MED 5Hz",     8.0,  5.2, 1500,  1000),    # medium -> mid PWM
        ("TREMOR-STRONG 4Hz", 10.0,  4.2, 3000,  2200),    # strong -> high PWM
        ("VOLUNTARY 1.8Hz",    5.0,  1.8, 6000,  4000),
        ("RESTING",            3.0,  0.0,    0,     0),
        ("TREMOR-WEAK 6Hz",    6.0,  6.0,  600,   400),    # weak -> low PWM
    ]

    si = 0
    elapsed = 0.0
    print(f"\n[SIM] Starting: {scenario[si][0]}")

    while not stop_event.is_set():
        label, duration, freq, amp_a, amp_g = scenario[si]

        if freq > 0:
            phase += 2 * np.pi * freq * dt
            ax = amp_a * np.sin(phase) + rng.normal(0, max(amp_a*0.08, 100))
            ay = amp_a * np.sin(phase + 0.5) + rng.normal(0, max(amp_a*0.08, 100))
            az = rng.normal(-13338, 200)
            gx = rng.normal(5002, 60)
            gy = amp_g * np.sin(phase) + rng.normal(-239, max(amp_g*0.08, 60))
            gz = amp_g * np.sin(phase + 0.4) + rng.normal(275, max(amp_g*0.08, 60))
        else:
            ax = rng.normal(54, 100)
            ay = rng.normal(5756, 100)
            az = rng.normal(-13338, 100)
            gx = rng.normal(5002, 40)
            gy = rng.normal(-239, 40)
            gz = rng.normal(275, 40)

        data_queue.put(np.array([ax, ay, az, gx, gy, gz], dtype=np.float32))

        t += dt
        elapsed += dt

        if elapsed >= duration:
            elapsed = 0.0
            si = (si + 1) % len(scenario)
            print(f"\n[SIM] Switching to: {scenario[si][0]}")

        time.sleep(dt)


# --- intensity computation ---

def compute_motor_intensity(p_ml: float, feats: np.ndarray) -> int:
    """
    Compute motor PWM duty (0-255) based on tremor severity indicators.

    The intensity scales with:
      1. ML confidence (p_ml): higher confidence -> stronger response
      2. Tremor power ratio (acc_tpr, gyro_tpr): more power in tremor band -> stronger
      3. Acceleration variability (acc_cv)

    Returns PWM value: PWM_MIN..PWM_MAX (80..230)
    """
    acc_tpr  = float(feats[1])   # feature 1: acc_tremor_power_ratio
    gyro_tpr = float(feats[11])  # feature 11: gyro_tremor_power_ratio
    acc_cv   = float(feats[6])   # feature 6: acc_cv

    # Severity score bounded [0.0, 1.0]
    severity = np.clip(0.50 * p_ml + 0.25 * acc_tpr + 0.25 * gyro_tpr + 0.20 * min(acc_cv * 10.0, 1.0), 0.0, 1.0)

    # Linear interpolation: [0.35, 1.0] -> [PWM_MIN, PWM_MAX]
    t = np.clip((severity - 0.35) / 0.65, 0.0, 1.0)
    pwm = int(PWM_MIN + t * (PWM_MAX - PWM_MIN))
    return max(PWM_MIN, min(PWM_MAX, pwm))


# --- decision logic ---

def decide(feats: np.ndarray, scaler, xgb_model, rf_model):
    """
    Robust tremor decision logic:
    1. Resting check: Very low kinetic energy -> RESTING
    2. ML Classification: Evaluate 18 scale-invariant features with calibrated XGBoost + RF ensemble
    3. Physiological consistency: Reject if pure voluntary slow motion (< 2.5 Hz with no tremor power)

    Returns (prediction, p_ml, intensity_pwm, reason_str)
    """
    acc_peak_freq  = float(feats[0])    # feature 0
    acc_tpr        = float(feats[1])    # feature 1
    acc_cv         = float(feats[6])    # feature 6
    gyro_peak_freq = float(feats[10])   # feature 10
    gyro_tpr       = float(feats[11])   # feature 11

    # Gate 1: Completely resting?
    if acc_cv < ACC_CV_REST_THR and acc_tpr < 0.10:
        return 0, 0.0, PWM_OFF, "RESTING (sensor still)"

    # Evaluate ML Ensemble
    X_scaled = scaler.transform(feats.reshape(1, -1))
    p_xgb = float(xgb_model.predict_proba(X_scaled)[0, 1])
    p_rf  = float(rf_model.predict_proba(X_scaled)[0, 1])
    p_ml  = 0.6 * p_xgb + 0.4 * p_rf

    # Gate 2: Voluntary motion rejection
    # If dominant frequencies are strictly in voluntary band (< 2.5 Hz) and ML probability is low
    if (acc_peak_freq < PEAK_FREQ_VOL_THR and gyro_peak_freq < PEAK_FREQ_VOL_THR) and p_ml < 0.40:
        return 0, p_ml, PWM_OFF, f"VOLUNTARY GESTURE (f={acc_peak_freq:.1f}Hz, ML%={p_ml*100:.0f})"

    # Gate 3: Tremor Detection
    # ML positive with sufficient confidence (ML threshold = 0.40)
    if p_ml >= ML_PROB_THR:
        intensity = compute_motor_intensity(p_ml, feats)
        dom_f = acc_peak_freq if (PHYS_FREQ_LO <= acc_peak_freq <= PHYS_FREQ_HI) else gyro_peak_freq
        return 1, p_ml, intensity, f"TREMOR DETECTED (p={p_ml:.2f}, dom_f={dom_f:.1f}Hz, PWM={intensity})"

    return 0, p_ml, PWM_OFF, f"NORMAL (ML%={p_ml*100:.0f}, f={acc_peak_freq:.1f}Hz)"


# --- motor command sender ---

def send_motor_cmd(ser_or_none, cmd: bytes, simulate: bool, intensity: int = 0):
    """Send motor command to ESP32. For 'G', also sends 'V'+intensity first."""
    if cmd == b'G' and intensity > 0:
        # Send variable intensity BEFORE enabling the gate
        intensity_byte = max(60, min(240, intensity))
        if simulate:
            print(f"    [CMD] SET INTENSITY PWM={intensity_byte}")
        else:
            if ser_or_none:
                try:
                    ser_or_none.write(b'V' + bytes([intensity_byte]))
                except Exception as e:
                    print(f"[WARN] Serial write failed: {e}")
        # Small delay to ensure V is processed before G
        time.sleep(0.005)

    if simulate:
        if cmd == b'G':
            print(f"    [CMD] >> MOTORS ON  (antiphase, PWM={intensity})")
        elif cmd == b'D':
            print(f"    [CMD] >> MOTORS OFF")
        return

    if ser_or_none:
        try:
            ser_or_none.write(cmd)
        except Exception as e:
            print(f"[WARN] Serial write failed: {e}")


# --- main predict loop ---

def predict_loop(ser_or_none, simulate: bool, scaler, xgb_model, rf_model):
    ring = np.zeros((WINDOW_N, 6), dtype=np.float32)
    ring_ptr = 0
    filled = 0

    streak_pos = 0      # consecutive positive windows
    streak_neg = 0      # consecutive negative windows
    motor_on = False
    current_intensity = 0

    data_queue = queue.Queue(maxsize=2000)
    stop_event = threading.Event()

    # start appropriate data source
    if simulate:
        t = threading.Thread(target=simulate_imu_stream, args=(data_queue, stop_event), daemon=True)
    else:
        t = threading.Thread(target=serial_reader, args=(ser_or_none, data_queue, stop_event), daemon=True)
    t.start()

    print("\n" + "="*65)
    print("  Tremor Detection Running  (Ctrl+C to stop)")
    print("  Motor speed varies with tremor intensity (PWM 80-220)")
    print("="*65)

    samples_since_eval = 0
    eval_count = 0

    try:
        while True:
            # -- collect one sample --
            try:
                sample = data_queue.get(timeout=3.0)
            except queue.Empty:
                print("[WARN] No data for 3 s - is ESP32 in raw CSV mode? (sent 'r'?)")
                if ser_or_none:
                    ser_or_none.write(b'r')  # retry
                continue

            # -- fill ring buffer --
            ring[ring_ptr % WINDOW_N] = sample
            ring_ptr += 1
            filled = min(filled + 1, WINDOW_N)
            samples_since_eval += 1

            # -- evaluate every STEP_N samples once buffer is full --
            if filled < WINDOW_N or samples_since_eval < STEP_N:
                continue

            samples_since_eval = 0
            eval_count += 1

            # build ordered window (oldest first)
            if ring_ptr >= WINDOW_N:
                start = ring_ptr % WINDOW_N
                window = np.vstack([ring[start:], ring[:start]])
            else:
                window = ring.copy()

            # -- extract features --
            feats = extract_features(window)

            # -- decision --
            pred, p_ml, intensity, reason = decide(feats, scaler, xgb_model, rf_model)

            # -- streak logic --
            if pred == 1:
                streak_pos += 1
                streak_neg = 0
            else:
                streak_neg += 1
                streak_pos = 0

            new_motor_state = motor_on
            if not motor_on and streak_pos >= STREAK_TRIGGER:
                new_motor_state = True
            elif motor_on and streak_neg >= STREAK_RELEASE:
                new_motor_state = False

            # -- motor command with variable speed --
            if new_motor_state:
                # Send intensity + gate on every positive evaluation (heartbeat)
                send_motor_cmd(ser_or_none, b'G', simulate, intensity)
                current_intensity = intensity
            else:
                if motor_on and not new_motor_state:
                    send_motor_cmd(ser_or_none, b'D', simulate)
                    current_intensity = 0
            motor_on = new_motor_state

            # -- display --
            if motor_on:
                status = f"[!] TREMOR PWM={current_intensity}"
            else:
                status = "[ok] normal"
            print(
                f"[#{eval_count:04d}] {status}  "
                f"freq={feats[0]:.1f}Hz  tpr={feats[1]:.2f}  "
                f"cv={feats[6]:.3f}  ML%={p_ml*100:.0f}  "
                f"streak+={streak_pos} -{streak_neg}  | {reason}"
            )

    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C - sending DISABLE + STOP to ESP32 ...")
        send_motor_cmd(ser_or_none, b'D', simulate)
        time.sleep(0.1)
        if ser_or_none:
            try:
                ser_or_none.write(b'S')
            except Exception:
                pass
        stop_event.set()
        print("[STOP] Done.")


# --- entry point ---

def main():
    parser = argparse.ArgumentParser(description="Parkinson's Tremor Detector")
    parser.add_argument("--port",     default=None, help="Serial port (e.g. COM5)")
    parser.add_argument("--simulate", action="store_true", help="Run offline simulation")
    args = parser.parse_args()

    # Load models
    scaler, xgb_model, rf_model = load_models()

    if args.simulate:
        print("[SIM] Simulation mode - no serial port required.")
        predict_loop(None, simulate=True, scaler=scaler,
                     xgb_model=xgb_model, rf_model=rf_model)
        return

    # Serial mode
    if not HAS_SERIAL:
        print("[ERROR] pyserial not installed.  Run: pip install pyserial")
        sys.exit(1)

    port = args.port or auto_detect_port()
    if not port:
        print("[ERROR] No serial port found.  Use --port COMX or --simulate")
        sys.exit(1)

    try:
        ser = open_serial(port)
    except serial.SerialException as e:
        print(f"[ERROR] Cannot open {port}: {e}")
        sys.exit(1)

    try:
        predict_loop(ser, simulate=False, scaler=scaler,
                     xgb_model=xgb_model, rf_model=rf_model)
    finally:
        try:
            ser.close()
        except Exception:
            pass
        print("[SERIAL] Port closed.")


if __name__ == "__main__":
    main()