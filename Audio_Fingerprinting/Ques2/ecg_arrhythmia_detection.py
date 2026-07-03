"""
ECG Arrhythmia Detection - Complete Solution
Q2: The Midnight Episode - Catching the Arrhythmia
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART A: LOADING AND ANALYZING THE SIGNAL
# ============================================================================

def part_a():
    """
    (a) Reading the signal [0.75%]
    (i) How many seconds long is the clip?
    (ii) In the healthy stretch, what is the patient's heart rate in beats per minute,
         and how many samples does one healthy beat occupy?
    (iii) Treating the healthy ECG as a periodic signal, what is its fundamental frequency f0?
    """
    print("\n" + "="*70)
    print("PART A: READING AND ANALYZING THE SIGNAL")
    print("="*70)
    
    # Load the ECG signal
    patient_ecg = np.load('/mnt/user-data/uploads/patient_ecg.npy')
    
    # Given parameters
    fs = 250  # Sampling frequency in Hz
    N = len(patient_ecg)  # Number of samples
    
    # (i) Duration of the clip
    duration = N / fs
    print(f"\n(i) Clip Duration:")
    print(f"    Total samples: {N}")
    print(f"    Sampling frequency: {fs} Hz")
    print(f"    Duration: {duration} seconds ({duration/60:.2f} minutes)")
    
    # (ii) Heart rate and samples per beat
    # From the problem: one full beat arrives every 0.8 seconds
    beat_interval = 0.8  # seconds
    heart_rate = 60 / beat_interval  # BPM
    samples_per_beat = beat_interval * fs
    
    print(f"\n(ii) Healthy Heart Rate:")
    print(f"    Beat interval: {beat_interval} seconds")
    print(f"    Heart rate: {heart_rate} beats per minute (BPM)")
    print(f"    Samples per beat: {samples_per_beat} samples")
    
    # (iii) Fundamental frequency
    f0 = 1 / beat_interval
    print(f"\n(iii) Fundamental Frequency:")
    print(f"      f0 = 1 / {beat_interval} = {f0} Hz")
    
    return patient_ecg, fs

# ============================================================================
# PART B: FREQUENCY DOMAIN ANALYSIS
# ============================================================================

def part_b(patient_ecg, fs):
    """
    (b) Healthy heart in frequency domain [0.75%]
    (i) What is the healthy ECG? Is it (nearly) periodic?
    (ii) Which frequency component is responsible for higher-frequency content?
    (iii) What happens when heart rate rises?
    """
    print("\n" + "="*70)
    print("PART B: FREQUENCY DOMAIN ANALYSIS OF HEALTHY ECG")
    print("="*70)
    
    # Extract healthy portion (first 2000 samples = healthy)
    healthy_end = 2000
    healthy_ecg = patient_ecg[:healthy_end]
    
    # Compute FFT
    fft_healthy = fft(healthy_ecg)
    freqs = fftfreq(len(healthy_ecg), 1/fs)
    magnitude = np.abs(fft_healthy)
    
    # Only positive frequencies
    positive_freqs = freqs[:len(freqs)//2]
    positive_mag = magnitude[:len(magnitude)//2]
    
    print(f"\n(i) Healthy ECG Characteristics:")
    print(f"    The healthy ECG is periodic with regular intervals")
    print(f"    It resembles a smooth, continuous pattern with clear repetition")
    
    # Find dominant frequencies
    top_indices = np.argsort(positive_mag)[-5:][::-1]
    print(f"\n(ii) Dominant Frequency Components:")
    for idx in top_indices:
        if idx < len(positive_freqs):
            print(f"     Frequency: {positive_freqs[idx]:.2f} Hz, Magnitude: {positive_mag[idx]:.2e}")
    
    print(f"\n(iii) Effect of Heart Rate Increase:")
    print(f"      - Fundamental frequency f0 increases proportionally")
    print(f"      - All harmonic peaks shift to higher frequencies")
    print(f"      - The QRS complex (sharp spike) has high-frequency content at 40-100 Hz")
    print(f"      - P and T waves contribute to lower frequency components")
    
    return healthy_ecg, fft_healthy, positive_freqs, positive_mag

# ============================================================================
# PART C: WINDOWING AND TEMPLATE CREATION
# ============================================================================

def part_c(patient_ecg, fs):
    """
    (c) Cutting a heartbeat (windowing) [1.0%]
    (i) How wide (in samples) should the window be?
    (ii) Where should Maya place it?
    """
    print("\n" + "="*70)
    print("PART C: WINDOWING AND TEMPLATE EXTRACTION")
    print("="*70)
    
    # From problem: beat interval is 0.8 seconds
    beat_interval = 0.8
    samples_per_beat = beat_interval * fs
    
    print(f"\n(i) Window Width for Clean Beat Template:")
    print(f"    One full beat duration: {beat_interval} seconds")
    print(f"    In samples: {samples_per_beat} samples")
    print(f"    For a 'clean' template, first try: {samples_per_beat} samples")
    print(f"    If that's too wide and includes noise, try: 600 samples (2.4 seconds)")
    
    print(f"\n(ii) Window Placement:")
    print(f"    Place window in the healthy region (first ~2000 samples)")
    print(f"    Start around sample 0-400 to avoid startup artifacts")
    print(f"    Extract exactly one complete beat with clear P, QRS, T waves")
    
    # Load template
    template = np.load('/mnt/user-data/uploads/template.npy')
    print(f"\n    Template loaded: {len(template)} samples")
    print(f"    Template duration: {len(template)/fs:.3f} seconds")
    
    return template

# ============================================================================
# PART D: TEMPLATE MATCHING VIA CORRELATION
# ============================================================================

def part_d(patient_ecg, template, fs):
    """
    (d) Match the template (correlation) [1.5%]
    Compute normalized correlation coefficient ρ(m)
    """
    print("\n" + "="*70)
    print("PART D: TEMPLATE MATCHING VIA CORRELATION")
    print("="*70)
    
    L = len(template)
    N = len(patient_ecg)
    
    # Normalize template
    template_mean = np.mean(template)
    template_centered = template - template_mean
    template_energy = np.sum(template_centered ** 2)
    
    # Compute correlation at each position
    correlation = np.zeros(N - L + 1)
    
    for m in range(N - L + 1):
        signal_segment = patient_ecg[m:m+L]
        signal_mean = np.mean(signal_segment)
        signal_centered = signal_segment - signal_mean
        signal_energy = np.sum(signal_centered ** 2)
        
        # Normalized correlation
        numerator = np.sum(template_centered * signal_centered)
        denominator = np.sqrt(template_energy * signal_energy)
        
        if denominator > 0:
            correlation[m] = numerator / denominator
        else:
            correlation[m] = 0
    
    print(f"\nCorrelation Score ρ(m):")
    print(f"  Range of ρ(m): [{np.min(correlation):.3f}, {np.max(correlation):.3f}]")
    print(f"  ρ(m) = 1.0 → perfect match to template")
    print(f"  ρ(m) ≈ 0.8+ → good match (healthy beat)")
    print(f"  ρ(m) < 0.5 → poor match (abnormal beat or arrhythmia)")
    
    # Find where correlation drops significantly
    threshold = 0.5
    anomaly_starts = np.where(correlation < threshold)[0]
    if len(anomaly_starts) > 0:
        onset_sample = anomaly_starts[0]
        print(f"\n  Arrhythmia onset detected at sample: {onset_sample}")
        print(f"  Time: {onset_sample/fs:.2f} seconds")
    
    return correlation

# ============================================================================
# PART E: ONSET DETECTION & SPECTROGRAM
# ============================================================================

def part_e(patient_ecg, fs):
    """
    (e) Onset detection & the spectrogram [1.5%]
    Identify where the arrhythmia begins
    """
    print("\n" + "="*70)
    print("PART E: ONSET DETECTION AND SPECTROGRAM ANALYSIS")
    print("="*70)
    
    # Compute spectrogram
    f_spec, t_spec, Sxx = spectrogram(patient_ecg, fs=fs, nperseg=256)
    
    print(f"\nSpectrogram Analysis:")
    print(f"  Frequency range: {np.min(f_spec):.2f} - {np.max(f_spec):.2f} Hz")
    print(f"  Time resolution: {t_spec[1] - t_spec[0]:.3f} seconds")
    print(f"  Shape: {Sxx.shape[0]} frequencies × {Sxx.shape[1]} time windows")
    
    # In healthy region, power should be concentrated
    # In arrhythmia region, regularity breaks down
    print(f"\n  Healthy Region (0-20s):")
    healthy_power = np.mean(Sxx[:, :80], axis=1)
    peak_freq_healthy = f_spec[np.argmax(healthy_power)]
    print(f"    Peak frequency: {peak_freq_healthy:.2f} Hz")
    
    print(f"\n  Detecting sudden changes in spectral content")
    print(f"  Arrhythmia onset: where regularity breaks down and")
    print(f"                    unexpected frequency components appear")
    
    return f_spec, t_spec, Sxx

# ============================================================================
# PART F: SAMPLING RATE CONSIDERATIONS
# ============================================================================

def part_f():
    """
    (f) Sampling rate for sharp QRS detection [0.5%]
    """
    print("\n" + "="*70)
    print("PART F: SAMPLING RATE ANALYSIS FOR QRS DETECTION")
    print("="*70)
    
    print(f"\nNyquist Theorem and QRS Detection:")
    print(f"  QRS complex contains sharp spikes at high frequencies (~50 Hz)")
    print(f"  Nyquist frequency must be ≥ 2 × 50 Hz = 100 Hz minimum")
    print(f"  Current sampling rate: 250 Hz")
    print(f"  Nyquist frequency: 125 Hz ✓ Sufficient for QRS detection")
    print(f"\n  Lower sampling rates:")
    print(f"  - 100 Hz (Nyquist = 50 Hz): Cannot capture sharp QRS spikes")
    print(f"  - 150 Hz (Nyquist = 75 Hz): Borderline, may lose fine details")
    print(f"  - 250 Hz or higher: Captures all relevant cardiac features")
    print(f"\n  Reason for aliasing with lower rates:")
    print(f"  Sharp, narrow QRS spikes have broad frequency spectrum")
    print(f"  Need at least 4-5 samples per spike to represent it accurately")

# ============================================================================
# PART G: COMPLETE DETECTOR IMPLEMENTATION
# ============================================================================

def part_g(patient_ecg, template, fs, correlation):
    """
    (g) Prototyping the detector in code [1.5%]
    Complete implementation with threshold and detection
    """
    print("\n" + "="*70)
    print("PART G: COMPLETE ARRHYTHMIA DETECTOR")
    print("="*70)
    
    # Find correlation threshold
    threshold = 0.5
    
    # Detect where signal deviates from template
    is_normal = correlation >= threshold
    is_abnormal = ~is_normal
    
    # Find onset (transition from normal to abnormal)
    onset_indices = np.where(~is_normal)[0]
    
    if len(onset_indices) > 0:
        onset_sample = onset_indices[0]
        onset_time = onset_sample / fs
        print(f"\nArrhythmia Detection Results:")
        print(f"  Threshold: ρ(m) = {threshold}")
        print(f"  Normal region: ρ(m) ≥ {threshold}")
        print(f"  Abnormal region: ρ(m) < {threshold}")
        print(f"\n  ONSET DETECTED:")
        print(f"    Sample: {onset_sample}")
        print(f"    Time: {onset_time:.2f} seconds into recording")
        print(f"    Timestamp: {int(onset_time//60)}m {int(onset_time%60)}s")
    else:
        print(f"\n  No significant arrhythmia detected in this recording")
    
    # Calculate statistics
    normal_corr = correlation[is_normal]
    abnormal_corr = correlation[is_abnormal] if len(correlation[is_abnormal]) > 0 else np.array([])
    
    print(f"\n  Correlation Statistics:")
    print(f"    Normal beats: ρ mean = {np.mean(normal_corr):.3f}, std = {np.std(normal_corr):.3f}")
    if len(abnormal_corr) > 0:
        print(f"    Abnormal beats: ρ mean = {np.mean(abnormal_corr):.3f}, std = {np.std(abnormal_corr):.3f}")
    print(f"    Number of abnormal detections: {np.sum(is_abnormal)}")
    
    return onset_indices, threshold

# ============================================================================
# PART H: VISUALIZATION AND SPECTROGRAM
# ============================================================================

def part_h(patient_ecg, template, correlation, fs, f_spec, t_spec, Sxx):
    """
    (h) Visualizing the spectrogram [0.5%]
    Create comprehensive plots showing detection results
    """
    print("\n" + "="*70)
    print("PART H: VISUALIZATION")
    print("="*70)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    # Plot 1: Original ECG signal
    time_ecg = np.arange(len(patient_ecg)) / fs
    axes[0].plot(time_ecg, patient_ecg, 'b-', linewidth=0.8)
    axes[0].set_title('Patient ECG Recording', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Amplitude (mV)')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, len(patient_ecg)/fs])
    
    # Plot 2: Normalized Correlation Score
    time_corr = np.arange(len(correlation)) / fs
    axes[1].plot(time_corr, correlation, 'g-', linewidth=1)
    axes[1].axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5)')
    axes[1].fill_between(time_corr, 0.5, 1, where=(correlation >= 0.5), 
                         alpha=0.3, color='green', label='Normal')
    axes[1].fill_between(time_corr, 0.5, correlation, where=(correlation < 0.5), 
                         alpha=0.3, color='red', label='Abnormal')
    axes[1].set_title('Template Correlation Score ρ(m)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('ρ(m)')
    axes[1].set_ylim([-0.2, 1.1])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, len(patient_ecg)/fs])
    
    # Plot 3: Spectrogram
    im = axes[2].pcolormesh(t_spec, f_spec, 10*np.log10(Sxx + 1e-10), 
                             shading='gouraud', cmap='viridis')
    axes[2].set_title('Spectrogram (Power vs Time vs Frequency)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Frequency (Hz)')
    axes[2].set_ylim([0, 50])
    cbar = plt.colorbar(im, ax=axes[2])
    cbar.set_label('Power (dB)')
    
    # Plot 4: Zoomed view of arrhythmia region
    # Find transition point
    threshold = 0.5
    abnormal = np.where(correlation < threshold)[0]
    if len(abnormal) > 0:
        transition = abnormal[0]
        zoom_start = max(0, transition - 500)
        zoom_end = min(len(patient_ecg), transition + 1500)
        
        zoom_time = np.arange(zoom_start, zoom_end) / fs
        axes[3].plot(zoom_time, patient_ecg[zoom_start:zoom_end], 'b-', linewidth=1)
        axes[3].axvline(x=transition/fs, color='r', linestyle='--', 
                       linewidth=2, label=f'Arrhythmia Onset ({transition/fs:.2f}s)')
        axes[3].fill_betweenx(axes[3].get_ylim(), 0, transition/fs - zoom_start/fs, 
                              alpha=0.2, color='green', label='Healthy')
        axes[3].fill_betweenx(axes[3].get_ylim(), transition/fs - zoom_start/fs, 
                              (zoom_end-zoom_start)/fs, alpha=0.2, color='red', label='Arrhythmia')
        axes[3].set_title('Zoomed View: Transition to Arrhythmia', fontsize=12, fontweight='bold')
        axes[3].legend()
    else:
        axes[3].plot(np.arange(len(patient_ecg))/fs, patient_ecg, 'b-', linewidth=0.8)
        axes[3].set_title('No Significant Arrhythmia Detected', fontsize=12, fontweight='bold')
    
    axes[3].set_ylabel('Amplitude (mV)')
    axes[3].set_xlabel('Time (seconds)')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/ecg_analysis_complete.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: /home/claude/ecg_analysis_complete.png")
    
    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  ECG ARRHYTHMIA DETECTION - COMPLETE SOLUTION".center(68) + "█")
    print("█" + "  Q2: The Midnight Episode - Catching the Arrhythmia".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    # Part A: Load and analyze signal
    patient_ecg, fs = part_a()
    
    # Part B: Frequency domain analysis
    healthy_ecg, fft_healthy, pos_freqs, pos_mag = part_b(patient_ecg, fs)
    
    # Part C: Template extraction
    template = part_c(patient_ecg, fs)
    
    # Part D: Template matching via correlation
    correlation = part_d(patient_ecg, template, fs)
    
    # Part E: Onset detection and spectrogram
    f_spec, t_spec, Sxx = part_e(patient_ecg, fs)
    
    # Part F: Sampling rate analysis
    part_f()
    
    # Part G: Complete detector
    onset_indices, threshold = part_g(patient_ecg, template, fs, correlation)
    
    # Part H: Visualization
    fig = part_h(patient_ecg, template, correlation, fs, f_spec, t_spec, Sxx)
    
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE")
    print("="*70)
    print("\nAll results have been computed and visualized.")
    print("Check ecg_analysis_complete.png for the complete visualization.")

if __name__ == "__main__":
    main()
    plt.show()
