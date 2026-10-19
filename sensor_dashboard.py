"""
============================================================================
Real-Time MPU6050 Kinematic Waveform Viewer & Tremor ML Inference Dashboard
Target: ESP32 + MPU6050 Wearable Parkinson's Tremor System
============================================================================
Features:
- Live 4-panel visualizer (Accel, Gyro, 3.5-7.5 Hz Filtered Tremor, Welch PSD Spectrum)
- Real-time 16-feature XGBoost + SVM ensemble tremor risk scoring
- Interactive keyboard motor test keys: '1' (Motor ON), '0' (Motor OFF), 's' (Emergency Stop)
- Hardware stream or synthetic simulated preview (--simulate)
"""

import argparse
import csv
from collections import deque
from datetime import datetime
from pathlib import Path
import sys
import time

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

FS = 100           # 100 Hz sampling frequency
WIN = 100          # 1.0 second window (100 samples)
HOP = 50           # 0.5 second evaluation hop
SCALE = np.array([16384.0, 16384.0, 16384.0, 131.0, 131.0, 131.0])

def decode_line(s):
    """
    Parses CSV line: returns (timestamp_us_or_None, [ax, ay, az, gx, gy, gz])
    """
    s = s.strip()
    if not s or s.startswith('#'):
        return None
    parts = s.split(',')
    if len(parts) not in (6, 7):
        return None
    try:
        values = [int(float(x.strip())) for x in parts]
    except ValueError:
        return None
    if not all(-32768 <= x <= 32767 for x in values[-6:]):
        return None
    return (values[0] if len(values) == 7 else None, values[-6:])

def demo_stream():
    """Synthetic generator for offline preview without hardware plugged in."""
    t0 = time.monotonic()
    k = 0
    rng = np.random.default_rng(22)
    while True:
        t = k / FS
        # 5.0 Hz synthetic oscillation + noise
        vals = [
            int(1200 * np.sin(2 * np.pi * 5.0 * t) + rng.normal(0, 60)),
            int(800 * np.cos(2 * np.pi * 5.0 * t) + rng.normal(0, 60)),
            int(16384 + 1400 * np.sin(2 * np.pi * 5.0 * t) + rng.normal(0, 60)),
            int(1800 * np.sin(2 * np.pi * 5.0 * t) + rng.normal(0, 10)),
            int(1200 * np.cos(2 * np.pi * 5.0 * t) + rng.normal(0, 10)),
            int(900 * np.sin(2 * np.pi * 5.0 * t + 0.5) + rng.normal(0, 10))
        ]
        yield (int(k * 10000), vals)
        k += 1
        sleep_dur = max(0.0, (t0 + k / FS) - time.monotonic())
        if sleep_dur > 0:
            time.sleep(sleep_dur)

def load_ml_classifier(project_dir):
    """Loads 16-feature fitted scaler and XGBoost+SVM ensemble."""
    import joblib
    from feature_extractor import extract_imu_features
    
    scaler_file = project_dir / 'scaler.pkl' if (project_dir / 'scaler.pkl').exists() else project_dir / 'models' / 'scaler_imu_only.pkl'
    xgb_file = project_dir / 'xgb_model.pkl' if (project_dir / 'xgb_model.pkl').exists() else project_dir / 'models' / 'xgb_imu_only.pkl'
    svm_file = project_dir / 'svm_model.pkl' if (project_dir / 'svm_model.pkl').exists() else project_dir / 'models' / 'svm_imu_only.pkl'
    
    scaler = joblib.load(scaler_file)
    xgb = joblib.load(xgb_file)
    svm = joblib.load(svm_file)
    
    if scaler.n_features_in_ != 16 or xgb.n_features_in_ != 16 or svm.n_features_in_ != 16:
        raise ValueError(f"Expected 16 features. Got {scaler.n_features_in_}.")
        
    def classify(raw_window):
        feat = extract_imu_features(np.asarray(raw_window, dtype=np.float64), fs=FS)
        feat_scaled = scaler.transform(feat.reshape(1, -1))
        score = 0.55 * xgb.predict_proba(feat_scaled)[0, 1] + 0.45 * svm.predict_proba(feat_scaled)[0, 1]
        peak_freq = float(feat[6])
        tpr = float(feat[8])
        return float(score), peak_freq, tpr
        
    return classify

def estimate_dominant_frequency(samples):
    """Finds peak frequency and power in the 3.5 - 7.5 Hz tremor band."""
    if len(samples) < WIN:
        return (float('nan'), 0.0, 3)
    x = np.asarray(samples[-WIN:], dtype=float)[:, 3:6] / 131.0
    x = x - x.mean(axis=0)
    freqs, power = signal.welch(x, fs=FS, nperseg=WIN, axis=0)
    band = (freqs >= 3.5) & (freqs <= 7.5)
    if not np.any(band):
        return (float('nan'), 0.0, 3)
    peak = power[band].max(axis=0)
    gyro_axis = int(np.argmax(peak))
    dom_idx = np.argmax(power[band, gyro_axis])
    dominant = float(freqs[band][dom_idx])
    rms = float(np.sqrt(np.mean(x[:, gyro_axis]**2)))
    return dominant, rms, gyro_axis + 3

def run(args):
    ser = None
    if args.simulate:
        source = demo_stream()
        print("\n>>> SIMULATED SIGNAL MODE (No ESP32 Hardware Required) <<<")
    else:
        import serial
        from serial.tools import list_ports
        port = args.port
        if not port:
            ports = list(list_ports.comports())
            for p in ports:
                if any(k in p.description.upper() for k in ["CP210", "CH340", "USB", "UART"]):
                    port = p.device
                    break
            if not port and ports:
                port = ports[0].device
        if not port:
            print("No serial port found. Specify with --port COMx or run with --simulate")
            sys.exit(1)
        print(f"Opening serial port {port} at 115200 baud...")
        ser = serial.Serial(port, 115200, timeout=0.03)
        time.sleep(2.0)
        ser.reset_input_buffer()
        ser.write(b'R')  # Request raw CSV streaming
        print(f"Connected to {port} at 115200 baud.")
        source = None

    project_dir = Path(__file__).resolve().parent
    classify = None if args.no_ml else load_ml_classifier(project_dir)
    outfile = Path(args.output) if args.output else project_dir / f'mpu_session_{datetime.now():%Y%m%d_%H%M%S}.csv'
    outfile.parent.mkdir(exist_ok=True, parents=True)

    times = deque(maxlen=400)
    raw = deque(maxlen=400)
    model_window = deque(maxlen=WIN)
    ts_dev = deque(maxlen=300)
    unwrapped_dev_us = 0
    prev_dev_us = None
    last_plot = 0
    score = float('nan')
    received = 0
    rejected = 0
    motor_state_str = "OFF"
    host_t0 = time.monotonic()

    plt.ion()
    fig, axs = plt.subplots(4, 1, figsize=(11, 8.5), constrained_layout=True)
    fig.canvas.manager.set_window_title("Wearable Parkinson's Tremor System | Real-Time Dashboard")

    trace = [axs[0].plot([], [], label=n)[0] for n in ('Ax (g)', 'Ay (g)', 'Az (g)')]
    gtrace = [axs[1].plot([], [], label=n)[0] for n in ('Gx (dps)', 'Gy (dps)', 'Gz (dps)')]
    filt_line, = axs[2].plot([], [], color='crimson', label='3.5 - 7.5 Hz Bandpassed Tremor Wave')
    fft_line, = axs[3].plot([], [], color='teal', label='Welch Power Spectral Density (PSD)')

    axs[0].set_ylabel('Accel (g)')
    axs[1].set_ylabel('Gyro (deg/s)')
    axs[2].set_ylabel('Tremor Band (deg/s)')
    axs[3].set_ylabel('PSD ((deg/s)^2/Hz)')
    axs[3].set_xlabel('Frequency (Hz)')
    axs[0].set_xlabel('Time (s)')
    axs[1].set_xlabel('Time (s)')
    axs[2].set_xlabel('Time (s)')

    for ax in axs[:2]:
        ax.legend(loc='upper right', ncol=3, fontsize=8)
    axs[2].legend(loc='upper right', fontsize=8)
    axs[3].axvspan(3.5, 7.5, color='orange', alpha=0.2, label='Parkinsonian Band (3.5 - 7.5 Hz)')
    axs[3].set_xlim(0, 20)
    axs[3].legend(loc='upper right', fontsize=8)

    sos = signal.butter(3, [3.5, 7.5], btype='bandpass', fs=FS, output='sos')

    # Interactive key binding for motor testing
    def on_key(event):
        nonlocal motor_state_str
        if ser is None:
            return
        k = event.key.lower() if event.key else ''
        if k == '1':
            ser.write(b'1')
            ser.flush()
            motor_state_str = "ON (Manual)"
            print("\n>>> MANUAL MOTOR COMMAND: '1' -> VIBRATION ON <<<")
        elif k == '0':
            ser.write(b'0')
            ser.flush()
            motor_state_str = "OFF"
            print("\n>>> MANUAL MOTOR COMMAND: '0' -> VIBRATION OFF <<<")
        elif k == 't':
            ser.write(b'T')
            ser.flush()
            motor_state_str = "DIAGNOSTIC TEST"
            print("\n>>> RUNNING 7-STAGE MOTOR DIAGNOSTIC TEST ('T') <<<")
        elif k == 'i':
            ser.write(b'I')
            ser.flush()
            print("\n>>> TOGGLED CHANNEL 2 POLARITY INVERT ('I') <<<")
        elif k == 'q':
            ser.write(b'Q')
            ser.flush()
            print("\n>>> TOGGLED MOTOR MODE: DUAL H-BRIDGE <-> QUAD ALL-OUTS ('Q') <<<")
        elif k == 's':
            ser.write(b'S')
            ser.flush()
            motor_state_str = "EMERGENCY STOPPED"
            print("\n>>> EMERGENCY STOP COMMAND SENT ('S') <<<")

    fig.canvas.mpl_connect('key_press_event', on_key)

    print("\n" + "=" * 65)
    print("  DASHBOARD HOTKEYS (Click graph window to focus):")
    print("  * Press '1' -> Force ALL Motors ON")
    print("  * Press '0' -> Force ALL Motors OFF")
    print("  * Press 't' -> Run Comprehensive 7-Stage Motor Test")
    print("  * Press 'i' -> Toggle Channel 2 Polarity Invert (Flip Direction)")
    print("  * Press 'q' -> Toggle Mode (Dual H-Bridge vs Quad All-OUTs)")
    print("  * Press 's' -> Emergency Motor STOP")
    print("=" * 65 + "\n")

    try:
        with outfile.open('w', newline='', encoding='utf8') as f:
            writer = csv.writer(f)
            writer.writerow(['t_pc_s', 't_esp_us', 'ax_raw', 'ay_raw', 'az_raw', 'gx_raw', 'gy_raw', 'gz_raw'])

            while plt.fignum_exists(fig.number):
                if ser is not None:
                    line = ser.readline().decode('ascii', errors='ignore')
                    if line.startswith('#'):
                        continue
                    parsed = decode_line(line)
                    if parsed is None:
                        if line.strip():
                            rejected += 1
                        plt.pause(0.0001)
                        continue
                    dev_us, values = parsed
                else:
                    dev_us, values = next(source)

                host_s = time.monotonic() - host_t0
                if dev_us is not None:
                    if prev_dev_us is not None:
                        unwrapped_dev_us += (dev_us - prev_dev_us) % (2**32)
                        ts_dev.append(unwrapped_dev_us * 1e-6)
                    else:
                        ts_dev.append(0.0)
                    prev_dev_us = dev_us

                t = ts_dev[-1] if dev_us is not None else host_s
                writer.writerow([f'{host_s:.6f}', dev_us if dev_us is not None else '', *values])
                received += 1

                times.append(t)
                raw.append(values)
                model_window.append(values)

                # ML inference every HOP (50 samples)
                if classify and len(model_window) == WIN and received % HOP == 0:
                    try:
                        score, peak_f, tpr = classify(list(model_window))
                        if score >= 0.50 and ser is not None:
                            ser.write(b'1')
                            motor_state_str = "AUTO: ON (Tremor)"
                        elif score < 0.35 and ser is not None:
                            ser.write(b'0')
                            motor_state_str = "AUTO: OFF (Normal)"
                    except Exception as e:
                        print(f"ML Error: {e}")

                # Update plots at ~5 Hz refresh rate to avoid GUI lag
                if time.monotonic() - last_plot < 0.20 or len(times) < 16:
                    plt.pause(0.001)
                    continue

                last_plot = time.monotonic()
                x = np.array(list(raw), dtype=float) / SCALE
                tplot = np.asarray(times, dtype=float)

                for i in range(3):
                    trace[i].set_data(tplot, x[:, i])
                    gtrace[i].set_data(tplot, x[:, i+3])

                for ax in axs[:2]:
                    ax.relim()
                    ax.autoscale_view()

                dom_freq, _, filt_axis = estimate_dominant_frequency(list(raw))
                y = x[:, filt_axis]

                if len(y) > 24:
                    try:
                        filt = signal.sosfiltfilt(sos, y)
                    except ValueError:
                        filt = np.zeros(len(y))
                    filt_line.set_data(tplot, filt)
                    axs[2].relim()
                    axs[2].autoscale_view()

                if len(y) >= 100:
                    hz, psd = signal.welch(y[-100:] - np.mean(y[-100:]), fs=FS, nperseg=100)
                    fft_line.set_data(hz, psd)
                    axs[3].relim()
                    axs[3].autoscale_view(scalex=False)

                ml_txt = f"ML Tremor Risk: {score*100:.1f}%" if np.isfinite(score) else "ML: Initializing..."
                freq_txt = f"Dominant Peak: {dom_freq:.1f} Hz" if np.isfinite(dom_freq) else "Peak: --.- Hz"
                fig.suptitle(f"Frames: {received} | {freq_txt} | {ml_txt} | Actuator: [{motor_state_str}]", fontsize=11, fontweight='bold')
                plt.pause(0.005)

    except KeyboardInterrupt:
        print("\nDashboard closed by user.")
    finally:
        if ser is not None:
            try:
                ser.write(b'0')
                ser.close()
            except Exception:
                pass
        plt.close('all')
        print(f"Logged {received} kinematic frames to: {outfile.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="Live ESP32 MPU6050 Graphical Waveform & ML Dashboard")
    parser.add_argument("--port", help="Serial port (e.g. COM5)")
    parser.add_argument("--output", help="CSV log filename")
    parser.add_argument("--no-ml", action="store_true", help="Disable ML inference (graph waveforms only)")
    parser.add_argument("--simulate", action="store_true", help="Run with simulated preview data (no hardware needed)")
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()
