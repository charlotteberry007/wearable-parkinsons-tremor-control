"""
=============================================================================
Project: Wearable Tremor Control - Quantitative Evaluation Tool
Purpose: Ingests serial CSV logs, computes spectral power, evaluates tremor
         attenuation between stimulation epochs, and visualizes phase tracking.
=============================================================================
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

def analyze_experiment_log(csv_filename):
    print(f"Loading experiment log: {csv_filename} ...")
    
    # Read CSV skipping comment lines
    df = pd.read_csv(csv_filename, comment='#')
    print(f"Successfully loaded {len(df)} samples.")
    
    fs = 50.0  # Decimated logging rate (Hz)
    t = (df['timestamp_ms'] - df['timestamp_ms'].iloc[0]) / 1000.0
    
    # Identify Stimulation ON vs Stimulation OFF intervals
    is_stim_on = df['pwm'] > 0
    
    rms = df['tremor_rms'].values
    freq = df['freq_hz'].values
    phase = df['phase_rad'].values
    phase_valid = df['phase_valid'].values
    
    # Quantitative Attenuation Calculation
    baseline_rms = np.median(rms[~is_stim_on]) if np.any(~is_stim_on) else 1.0
    stimulated_rms = np.median(rms[is_stim_on]) if np.any(is_stim_on) else 1.0
    
    attenuation_pct = 100.0 * (baseline_rms - stimulated_rms) / baseline_rms
    
    # Valid Phase Metrics
    valid_cycles_ratio = np.mean(phase_valid) * 100.0
    mean_freq = np.nanmean(freq[freq > 0])
    
    print("\n" + "="*55)
    print("      QUANTITATIVE EXPERIMENT BENCHMARK REPORT")
    print("="*55)
    print(f"Total Duration               : {t.iloc[-1]:.1f} seconds")
    print(f"Mean Tremor Frequency        : {mean_freq:.2f} Hz")
    print(f"Phase Validity Rate          : {valid_cycles_ratio:.1f}%")
    print(f"Median Baseline Tremor RMS   : {baseline_rms:.1f} LSB")
    print(f"Median Stimulated Tremor RMS : {stimulated_rms:.1f} LSB")
    print(f"Calculated Attenuation Ratio : {attenuation_pct:+.2f}%")
    print("="*55)
    
    # Plotting Diagnostics
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    # Subplot 1: Tremor RMS & Stimulation Epochs
    axes[0].plot(t, rms, color='navy', label='Tremor RMS (LSB)', lw=1.5)
    axes[0].fill_between(t, 0, np.max(rms)*1.1, where=is_stim_on, color='salmon', alpha=0.35, label='Stimulation Active')
    axes[0].set_ylabel('Tremor RMS (LSB)')
    axes[0].set_title('Tremor Amplitude & Adaptive Stimulation Epochs')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(loc='upper right')
    
    # Subplot 2: Estimated Tremor Frequency
    axes[1].plot(t, freq, color='darkgreen', label='Estimated Freq (Hz)', lw=1.2)
    axes[1].axhline(y=5.0, color='gray', linestyle=':', label='5.0 Hz Baseline')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_ylim(2.5, 9.0)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(loc='upper right')
    
    # Subplot 3: Phase Tracking
    axes[2].plot(t, phase, color='purple', label='Estimated Phase φ(t) [0, 2π)', lw=0.8)
    axes[2].set_ylabel('Phase (rad)')
    axes[2].set_xlabel('Elapsed Time (s)')
    axes[2].set_ylim(-0.2, 2*np.pi + 0.2)
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].legend(loc='upper right')
    
    plt.tight_layout()
    plot_output = "tremor_experiment_results.png"
    plt.savefig(plot_output, dpi=300)
    print(f"\nPlot saved successfully as '{plot_output}'\n")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        analyze_experiment_log(sys.argv[1])
    else:
        print("Usage: python evaluate_experiment.py <path_to_log.csv>")
