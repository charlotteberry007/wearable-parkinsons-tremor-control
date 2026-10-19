import numpy as np
import joblib
from feature_extractor import extract_imu_features

# Load Model Artifacts
xgb_model = joblib.load('xgb_model.pkl')
svm_model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

def generate_synthetic_window(fs=100, duration=1.0, motion_type="resting"):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    if motion_type == "tremor":
        # 5.0 Hz rhythmic oscillation in Parkinsonian tremor band
        ax = 15000 * np.sin(2 * np.pi * 5.0 * t) + np.random.normal(0, 300, len(t))
        ay = 12000 * np.cos(2 * np.pi * 5.0 * t) + np.random.normal(0, 300, len(t))
        az = 16384 + 10000 * np.sin(2 * np.pi * 5.0 * t) + np.random.normal(0, 300, len(t))
        gx = 4000 * np.sin(2 * np.pi * 5.0 * t) + np.random.normal(0, 100, len(t))
        gy = 3500 * np.cos(2 * np.pi * 5.0 * t) + np.random.normal(0, 100, len(t))
        gz = 2000 * np.sin(2 * np.pi * 5.0 * t) + np.random.normal(0, 100, len(t))
        
    elif motion_type == "voluntary_wave":
        # 1.2 Hz broad voluntary arm waving
        ax = 14000 * np.sin(2 * np.pi * 1.2 * t) + np.random.normal(0, 200, len(t))
        ay = 10000 * np.cos(2 * np.pi * 1.2 * t) + np.random.normal(0, 200, len(t))
        az = 16384 + 6000 * np.sin(2 * np.pi * 1.2 * t) + np.random.normal(0, 200, len(t))
        gx = 6000 * np.sin(2 * np.pi * 1.2 * t) + np.random.normal(0, 100, len(t))
        gy = 4500 * np.cos(2 * np.pi * 1.2 * t) + np.random.normal(0, 100, len(t))
        gz = 4000 * np.sin(2 * np.pi * 1.2 * t) + np.random.normal(0, 100, len(t))
        
    elif motion_type == "voluntary_reach":
        # 2.0 Hz reaching / gesture
        ax = 9000 * np.sin(2 * np.pi * 2.0 * t) + np.random.normal(0, 200, len(t))
        ay = 7000 * np.cos(2 * np.pi * 2.0 * t) + np.random.normal(0, 200, len(t))
        az = 16384 + 4000 * np.sin(2 * np.pi * 2.0 * t) + np.random.normal(0, 200, len(t))
        gx = 4000 * np.sin(2 * np.pi * 2.0 * t) + np.random.normal(0, 100, len(t))
        gy = 3000 * np.cos(2 * np.pi * 2.0 * t) + np.random.normal(0, 100, len(t))
        gz = 3500 * np.sin(2 * np.pi * 2.0 * t) + np.random.normal(0, 100, len(t))
        
    else: # resting
        ax = np.random.normal(0, 100, len(t))
        ay = np.random.normal(0, 100, len(t))
        az = 16384 + np.random.normal(0, 100, len(t))
        gx = np.random.normal(0, 30, len(t))
        gy = np.random.normal(0, 30, len(t))
        gz = np.random.normal(0, 30, len(t))
        
    return np.column_stack([ax, ay, az, gx, gy, gz])

normal_audio = np.array([0.02, 0.01, 0.01, 0.03, -25.0, 0.02, 0.01, 22.0, 0.02, 0.15] + [0.2]*8)

for test_name in ["resting", "voluntary_wave", "voluntary_reach", "tremor"]:
    imu = generate_synthetic_window(motion_type=test_name)
    imu_feats = extract_imu_features(imu)
    
    peak_freq = imu_feats[6]
    tpr = imu_feats[8]
    gyro_rms = imu_feats[15]
    
    fused_vector = np.concatenate([normal_audio, imu_feats]).reshape(1, -1)
    scaled = scaler.transform(fused_vector)
    p_ml = 0.55 * xgb_model.predict_proba(scaled)[0, 1] + 0.45 * svm_model.predict_proba(scaled)[0, 1]
    
    # Frequency-gated verification: Is it within 3.5 - 7.5 Hz with high TPR?
    is_tremor_band = (3.5 <= peak_freq <= 7.5) and (tpr >= 0.25) and (gyro_rms > 300)
    final_decision = (p_ml >= 0.50) and is_tremor_band
    
    print(f"{test_name:18s} | PeakFreq: {peak_freq:4.1f} Hz | TPR: {tpr:6.2%} | ML Score: {p_ml:6.2%} | Final Tremor Alert: {final_decision}")