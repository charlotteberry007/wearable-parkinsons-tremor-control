import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Feature Names (18 Audio + 16 IMU = 34 Total)
FEATURE_NAMES = [
    "Jitter_Local", "Jitter_RAP", "Jitter_PPQ5", "Shimmer_Local", "Shimmer_dB", 
    "Shimmer_APQ3", "Shimmer_APQ5", "HNR", "NHR", "RPDE", 
    "MFCC1", "MFCC2", "MFCC3", "MFCC4", "MFCC5", "MFCC6", "MFCC7", "Delta_MFCC1",
    "Acc_Mean", "Acc_Std", "Acc_Skew", "Acc_Kurtosis", "ZCR", "RMS_Jerk",
    "Peak_Freq_Hz", "Peak_PSD", "Tremor_Power_Ratio", "Spectral_Entropy", "ApEn",
    "Gyro_Mean", "Gyro_Std", "Gyro_Max", "Gyro_Min", "Gyro_RMS"
]

def explain_patient(sample_vector, output_filename="patient_shap_force_plot.png"):
    """
    Computes SHAP values for a single patient sample vector (shape: 1x34)
    and saves a force plot image.
    """
    xgb_model = joblib.load('xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    sample_scaled = scaler.transform(sample_vector.reshape(1, -1))
    
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(sample_scaled)
    
    # Render Force Plot
    try:
        shap.initjs()
    except Exception:
        pass

    fig = shap.plots.force(
        explainer.expected_value, 
        shap_values.values[0], 
        feature_names=FEATURE_NAMES,
        matplotlib=True,
        show=False
    )
    plt.title("Patient SHAP Diagnostic Attribution")
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Exported {output_filename}")

def explain_prediction_terminal(sample_vector, explainer=None, scaler=None, top_k=3):
    """
    Returns text summary of top-k biomarkers driving the prediction for live display.
    """
    if explainer is None:
        xgb_model = joblib.load('xgb_model.pkl')
        explainer = shap.TreeExplainer(xgb_model)
    if scaler is None:
        scaler = joblib.load('scaler.pkl')
        
    sample_scaled = scaler.transform(sample_vector.reshape(1, -1))
    shap_vals = explainer(sample_scaled).values[0]
    
    indices = np.argsort(np.abs(shap_vals))[::-1][:top_k]
    contributors = []
    for idx in indices:
        impact = shap_vals[idx]
        direction = "+" if impact > 0 else "-"
        contributors.append(f"{FEATURE_NAMES[idx]} ({direction}{abs(impact):.2f})")
    return ", ".join(contributors)

def generate_global_summary(X_test_scaled, output_filename="global_shap_summary.png"):
    xgb_model = joblib.load('xgb_model.pkl')
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_test_scaled)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_scaled, feature_names=FEATURE_NAMES, show=False)
    plt.title("Global Biomarker Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Exported {output_filename}")

if __name__ == "__main__":
    scaler = joblib.load('scaler.pkl')
    dummy_patient = np.random.randn(1, 34)
    explain_patient(dummy_patient)
    
    dummy_cohort = scaler.transform(np.random.randn(50, 34))
    generate_global_summary(dummy_cohort)
