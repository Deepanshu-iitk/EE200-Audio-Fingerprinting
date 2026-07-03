"""
experiments.py -- All experiments for Q3A report
=================================================
Generates spectrograms, constellation maps, offset histograms,
and runs robustness tests (noise, pitch shift, time stretch).
All plots are saved to the  report/  directory.
"""

import os
import sys
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# -- project imports -------------------------------------------------------------
from fingerprint import (
    load_audio, compute_spectrogram, find_peaks,
    generate_hashes, generate_single_peak_hashes,
    match_hashes, get_offset_histogram,
    add_noise, pitch_shift, time_stretch,
    DEFAULT_SR, DEFAULT_N_FFT, DEFAULT_HOP,
)
from database import load_database, build_database

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SONG_DIR = os.path.join(SCRIPT_DIR, "Song database")
REPORT_DIR = os.path.join(SCRIPT_DIR, "report")
CLIPS_DIR = os.path.join(SCRIPT_DIR, "test_clips")
DB_PATH = os.path.join(SCRIPT_DIR, "fingerprint_db.pkl")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)

# -- Plotting helpers ------------------------------------------------------------

def _save(fig, name):
    fig.savefig(os.path.join(REPORT_DIR, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    [OK] saved {name}")


# ===============================================================================
# 0. Generate test clips by extracting segments from songs
# ===============================================================================

def generate_test_clips(n_clips=10, clip_duration=10.0, seed=42):
    """Extract short clips from random positions in random songs."""
    rng = random.Random(seed)
    song_files = sorted(f for f in os.listdir(SONG_DIR) if f.lower().endswith(".mp3"))
    chosen = rng.sample(song_files, min(n_clips, len(song_files)))

    clip_info = []
    for fname in chosen:
        y, sr = load_audio(os.path.join(SONG_DIR, fname))
        max_start = len(y) - int(clip_duration * sr)
        if max_start <= 0:
            start = 0
        else:
            start = rng.randint(0, max_start)
        end = start + int(clip_duration * sr)
        clip = y[start:end]

        clip_name = f"clip_{os.path.splitext(fname)[0]}.wav"
        clip_path = os.path.join(CLIPS_DIR, clip_name)
        import soundfile as sf
        sf.write(clip_path, clip, sr)
        clip_info.append((clip_path, os.path.splitext(fname)[0]))

    print(f"  Generated {len(clip_info)} test clips in {CLIPS_DIR}")
    return clip_info


# ===============================================================================
# 1. Spectrogram – short vs. long window
# ===============================================================================

def experiment_spectrogram_windows():
    """Compare spectrograms with different FFT window sizes."""
    print("\n[Experiment 1] Spectrogram window sizes")
    song_file = sorted(os.listdir(SONG_DIR))[0]  # first song alphabetically
    song_path = os.path.join(SONG_DIR, song_file)
    song_name = os.path.splitext(song_file)[0]

    y, sr = load_audio(song_path)
    # Use first 15 seconds
    y = y[: int(15 * sr)]

    windows = [256, 1024, 4096]
    labels = ["Short (256)", "Default (1024)", "Long (4096)"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Spectrogram Window Comparison -- "{song_name}"', fontsize=14, fontweight="bold")

    for ax, n_fft, label in zip(axes, windows, labels):
        hop = n_fft // 2
        S_db, freqs, times = compute_spectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop)
        img = ax.imshow(S_db, aspect="auto", origin="lower",
                        extent=[times[0], times[-1], freqs[0], freqs[-1]],
                        cmap="magma", vmin=-80, vmax=0)
        ax.set_title(f"Window = {label}", fontsize=11)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(0, 5000)
        fig.colorbar(img, ax=ax, label="dB")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "01_spectrogram_windows.png")
    return song_path, song_name


# ===============================================================================
# 2. Constellation map
# ===============================================================================

def experiment_constellation_map(song_path, song_name):
    """Plot spectrogram with detected peaks overlaid."""
    print("\n[Experiment 2] Constellation map")
    y, sr = load_audio(song_path)
    y = y[: int(15 * sr)]

    S_db, freqs, times = compute_spectrogram(y, sr=sr)
    peaks = find_peaks(S_db)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f'Constellation Map -- "{song_name}"', fontsize=14, fontweight="bold")

    # Left: spectrogram
    axes[0].imshow(S_db, aspect="auto", origin="lower",
                   extent=[times[0], times[-1], freqs[0], freqs[-1]],
                   cmap="magma", vmin=-80, vmax=0)
    axes[0].set_title("Spectrogram")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_ylim(0, 5000)

    # Right: peaks on spectrogram
    axes[1].imshow(S_db, aspect="auto", origin="lower",
                   extent=[times[0], times[-1], freqs[0], freqs[-1]],
                   cmap="magma", vmin=-80, vmax=0, alpha=0.5)
    peak_times = times[np.clip(peaks[:, 1], 0, len(times) - 1)]
    peak_freqs = freqs[np.clip(peaks[:, 0], 0, len(freqs) - 1)]
    axes[1].scatter(peak_times, peak_freqs, s=2, c="cyan", alpha=0.6)
    axes[1].set_title(f"Constellation ({len(peaks)} peaks)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_ylim(0, 5000)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "02_constellation_map.png")


# ===============================================================================
# 3. Offset histogram – correct vs wrong match
# ===============================================================================

def experiment_offset_histogram(database, song_names, clip_info):
    """Show offset histogram for a correct match and a wrong match."""
    print("\n[Experiment 3] Offset histogram (correct vs wrong)")
    clip_path, true_name = clip_info[0]
    y, sr = load_audio(clip_path)

    S_db, _, _ = compute_spectrogram(y, sr=sr)
    peaks = find_peaks(S_db)
    hashes = generate_hashes(peaks)

    best_id, best_count, offset_counts = match_hashes(hashes, database)

    # Find IDs
    correct_id = None
    wrong_id = None
    for sid, sname in song_names.items():
        if sname == true_name:
            correct_id = sid
        elif wrong_id is None and sname != true_name:
            wrong_id = sid

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Offset Histogram -- Correct vs Wrong Song", fontsize=14, fontweight="bold")

    for ax, sid, title_tag, color in [
        (axes[0], correct_id, f"CORRECT: {true_name}", "#00c853"),
        (axes[1], wrong_id, f"WRONG: {song_names.get(wrong_id, '?')}", "#ff1744"),
    ]:
        oc = offset_counts.get(sid, {})
        offsets, counts = get_offset_histogram(oc)
        if len(offsets) > 0:
            ax.bar(offsets, counts, width=1.0, color=color, alpha=0.8)
        ax.set_title(title_tag, fontsize=10)
        ax.set_xlabel("Time Offset (frames)")
        ax.set_ylabel("Number of matching hashes")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "03_offset_histogram.png")

    print(f"    Matched: {song_names.get(best_id, 'None')} (count={best_count})")
    return best_id, best_count


# ===============================================================================
# 4. Single peaks vs paired hashes
# ===============================================================================

def experiment_single_vs_paired(database, song_names, clip_info):
    """Compare matching with single-peak hashes vs paired hashes."""
    print("\n[Experiment 4] Single peaks vs paired hashes")

    clip_path, true_name = clip_info[0]
    y, sr = load_audio(clip_path)
    S_db, _, _ = compute_spectrogram(y, sr=sr)
    peaks = find_peaks(S_db)

    # -- Build a single-peak database --------------------------------------
    single_db = {}
    song_files = sorted(f for f in os.listdir(SONG_DIR) if f.lower().endswith(".mp3"))
    for idx, fname in enumerate(song_files):
        sy, ssr = load_audio(os.path.join(SONG_DIR, fname))
        sS, _, _ = compute_spectrogram(sy, sr=ssr)
        sp = find_peaks(sS)
        sh = generate_single_peak_hashes(sp)
        for h, off in sh:
            if h not in single_db:
                single_db[h] = []
            single_db[h].append((idx, off))

    # Match with single peaks
    q_single = generate_single_peak_hashes(peaks)
    s_best, s_count, s_offsets = match_hashes(q_single, single_db)

    # Match with paired hashes
    q_paired = generate_hashes(peaks)
    p_best, p_count, p_offsets = match_hashes(q_paired, database)

    # Find correct song IDs
    correct_id_single = None
    for sid, sname in song_names.items():
        if sname == true_name:
            correct_id_single = sid
            break

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Single Peaks vs Paired Hashes -- query from "{true_name}"',
                 fontsize=13, fontweight="bold")

    # Single peak histogram for correct song
    ax = axes[0]
    oc = s_offsets.get(correct_id_single, {})
    offsets, counts = get_offset_histogram(oc)
    if len(offsets) > 0:
        ax.bar(offsets, counts, width=1.0, color="#ff9800", alpha=0.8)
    ax.set_title(f"Single Peaks (best match: {song_names.get(s_best, '?')}, count={s_count})", fontsize=10)
    ax.set_xlabel("Offset")
    ax.set_ylabel("Matches")

    # Paired hash histogram for correct song
    ax = axes[1]
    oc = p_offsets.get(correct_id_single, {})
    offsets, counts = get_offset_histogram(oc)
    if len(offsets) > 0:
        ax.bar(offsets, counts, width=1.0, color="#2196f3", alpha=0.8)
    ax.set_title(f"Paired Hashes (best match: {song_names.get(p_best, '?')}, count={p_count})", fontsize=10)
    ax.set_xlabel("Offset")
    ax.set_ylabel("Matches")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "04_single_vs_paired.png")

    return s_best, s_count, p_best, p_count


# ===============================================================================
# 5. Noise robustness
# ===============================================================================

def experiment_noise_robustness(database, song_names, clip_info):
    """Test recognition under increasing noise levels."""
    print("\n[Experiment 5] Noise robustness")
    clip_path, true_name = clip_info[0]
    y_clean, sr = load_audio(clip_path)

    snr_levels = [20, 10, 5, 0, -5]
    results = []

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(f'Noise Robustness -- "{true_name}"', fontsize=14, fontweight="bold")

    # Clean clip spectrogram
    ax = axes[0, 0]
    S_db, freqs, times = compute_spectrogram(y_clean, sr=sr)
    ax.imshow(S_db, aspect="auto", origin="lower",
              extent=[times[0], times[-1], freqs[0], freqs[-1]],
              cmap="magma", vmin=-80, vmax=0)
    ax.set_title("Clean (no noise)", fontsize=10)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0, 5000)

    for i, snr in enumerate(snr_levels):
        y_noisy = add_noise(y_clean, snr_db=snr)
        S_db_n, freqs_n, times_n = compute_spectrogram(y_noisy, sr=sr)
        peaks = find_peaks(S_db_n)
        hashes = generate_hashes(peaks)
        best_id, best_count, _ = match_hashes(hashes, database)
        matched = song_names.get(best_id, "None")
        correct = matched == true_name
        results.append((snr, matched, correct, best_count))

        row, col = divmod(i + 1, 3)
        ax = axes[row, col]
        ax.imshow(S_db_n, aspect="auto", origin="lower",
                  extent=[times_n[0], times_n[-1], freqs_n[0], freqs_n[-1]],
                  cmap="magma", vmin=-80, vmax=0)
        status = "Y" if correct else "X"
        ax.set_title(f"SNR={snr} dB -> {matched} {status}", fontsize=10,
                     color="green" if correct else "red")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(0, 5000)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "05_noise_robustness.png")

    # Summary bar chart
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    snrs = [r[0] for r in results]
    counts_vals = [r[3] for r in results]
    colors = ["#00c853" if r[2] else "#ff1744" for r in results]
    ax2.bar(range(len(snrs)), counts_vals, color=colors, tick_label=[f"{s} dB" for s in snrs])
    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel("Best offset count")
    ax2.set_title("Noise Robustness -- Matching Confidence", fontweight="bold")
    _save(fig2, "05b_noise_bar.png")

    for snr, matched, correct, cnt in results:
        print(f"    SNR={snr:>3} dB  ->  {matched:<40s}  {'Y' if correct else 'X'}  (count={cnt})")

    return results


# ===============================================================================
# 6. Pitch shift robustness
# ===============================================================================

def experiment_pitch_shift(database, song_names, clip_info):
    """Test recognition under pitch shifting."""
    print("\n[Experiment 6] Pitch shift robustness")
    clip_path, true_name = clip_info[0]
    y_clean, sr = load_audio(clip_path)

    shifts = [-3, -2, -1, 0, 1, 2, 3]
    results = []

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle(f'Pitch Shift Robustness -- "{true_name}"', fontsize=14, fontweight="bold")

    for i, n_steps in enumerate(shifts):
        row, col = divmod(i, 4)
        ax = axes[row, col]

        if n_steps == 0:
            y_mod = y_clean.copy()
        else:
            y_mod = pitch_shift(y_clean, sr=sr, n_steps=n_steps)

        S_db, freqs, times = compute_spectrogram(y_mod, sr=sr)
        peaks = find_peaks(S_db)
        hashes = generate_hashes(peaks)
        best_id, best_count, _ = match_hashes(hashes, database)
        matched = song_names.get(best_id, "None")
        correct = matched == true_name
        results.append((n_steps, matched, correct, best_count))

        ax.imshow(S_db, aspect="auto", origin="lower",
                  extent=[times[0], times[-1], freqs[0], freqs[-1]],
                  cmap="magma", vmin=-80, vmax=0)
        status = "Y" if correct else "X"
        ax.set_title(f"Shift={n_steps:+d} st -> {status}", fontsize=10,
                     color="green" if correct else "red")
        ax.set_ylim(0, 5000)

    # Hide last subplot if odd
    if len(shifts) < 8:
        axes[1, 3].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "06_pitch_shift.png")

    for n_steps, matched, correct, cnt in results:
        print(f"    Shift={n_steps:+d} semitones  ->  {matched:<40s}  {'Y' if correct else 'X'}  (count={cnt})")

    return results


# ===============================================================================
# 7. Time stretch robustness
# ===============================================================================

def experiment_time_stretch(database, song_names, clip_info):
    """Test recognition under time stretching."""
    print("\n[Experiment 7] Time stretch robustness")
    clip_path, true_name = clip_info[0]
    y_clean, sr = load_audio(clip_path)

    rates = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]
    results = []

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle(f'Time Stretch Robustness -- "{true_name}"', fontsize=14, fontweight="bold")

    for i, rate in enumerate(rates):
        row, col = divmod(i, 4)
        ax = axes[row, col]

        if rate == 1.0:
            y_mod = y_clean.copy()
        else:
            y_mod = time_stretch(y_clean, rate=rate)

        S_db, freqs, times = compute_spectrogram(y_mod, sr=sr)
        peaks = find_peaks(S_db)
        hashes = generate_hashes(peaks)
        best_id, best_count, _ = match_hashes(hashes, database)
        matched = song_names.get(best_id, "None")
        correct = matched == true_name
        results.append((rate, matched, correct, best_count))

        ax.imshow(S_db, aspect="auto", origin="lower",
                  extent=[times[0], times[-1], freqs[0], freqs[-1]],
                  cmap="magma", vmin=-80, vmax=0)
        status = "Y" if correct else "X"
        ax.set_title(f"Rate={rate:.2f} -> {status}", fontsize=10,
                     color="green" if correct else "red")
        ax.set_ylim(0, 5000)

    if len(rates) < 8:
        axes[1, 3].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "07_time_stretch.png")

    for rate, matched, correct, cnt in results:
        print(f"    Rate={rate:.2f}  ->  {matched:<40s}  {'Y' if correct else 'X'}  (count={cnt})")

    return results


# ===============================================================================
# 8. Batch accuracy test
# ===============================================================================

def experiment_batch_accuracy(database, song_names, clip_info):
    """Run all test clips through the identifier and report accuracy."""
    print("\n[Experiment 8] Batch accuracy")
    correct = 0
    total = 0
    rows = []
    for clip_path, true_name in clip_info:
        y, sr = load_audio(clip_path)
        S_db, _, _ = compute_spectrogram(y, sr=sr)
        peaks = find_peaks(S_db)
        hashes = generate_hashes(peaks)
        best_id, best_count, _ = match_hashes(hashes, database)
        matched = song_names.get(best_id, "None")
        is_correct = matched == true_name
        if is_correct:
            correct += 1
        total += 1
        rows.append((os.path.basename(clip_path), true_name, matched, is_correct))
        print(f"    {os.path.basename(clip_path):>50s}  ->  {matched:<40s}  {'Y' if is_correct else 'X'}")

    acc = correct / total * 100 if total > 0 else 0
    print(f"\n    Accuracy: {correct}/{total} = {acc:.1f}%")
    return rows, acc


# ===============================================================================
# MAIN
# ===============================================================================

def main():
    print("=" * 70)
    print("  Q3A -- Sonic Signatures: Running All Experiments")
    print("=" * 70)

    # Build database if needed
    if not os.path.exists(DB_PATH):
        print("\n[Step 0] Building fingerprint database ...")
        database, song_names = build_database(SONG_DIR, db_path=DB_PATH)
    else:
        print("\n[Step 0] Loading existing fingerprint database ...")
        database, song_names = load_database(DB_PATH)
    print(f"  Database: {len(database):,} hashes, {len(song_names)} songs")

    # Generate test clips
    print("\n[Step 1] Generating test clips ...")
    clip_info = generate_test_clips(n_clips=10, clip_duration=10.0)

    # Run experiments
    song_path, song_name = experiment_spectrogram_windows()
    experiment_constellation_map(song_path, song_name)
    experiment_offset_histogram(database, song_names, clip_info)
    experiment_single_vs_paired(database, song_names, clip_info)
    noise_results = experiment_noise_robustness(database, song_names, clip_info)
    pitch_results = experiment_pitch_shift(database, song_names, clip_info)
    time_results = experiment_time_stretch(database, song_names, clip_info)
    batch_rows, batch_acc = experiment_batch_accuracy(database, song_names, clip_info)

    # Save experiment results as text
    with open(os.path.join(REPORT_DIR, "experiment_results.txt"), "w") as f:
        f.write("=== Noise Robustness ===\n")
        for snr, matched, correct, cnt in noise_results:
            f.write(f"SNR={snr:>3} dB  ->  {matched}  {'Y' if correct else 'X'}  count={cnt}\n")
        f.write("\n=== Pitch Shift ===\n")
        for n_steps, matched, correct, cnt in pitch_results:
            f.write(f"Shift={n_steps:+d}  ->  {matched}  {'Y' if correct else 'X'}  count={cnt}\n")
        f.write("\n=== Time Stretch ===\n")
        for rate, matched, correct, cnt in time_results:
            f.write(f"Rate={rate:.2f}  ->  {matched}  {'Y' if correct else 'X'}  count={cnt}\n")
        f.write(f"\n=== Batch Accuracy: {batch_acc:.1f}% ===\n")
        for fname, true_n, pred_n, ok in batch_rows:
            f.write(f"{fname}  ->  true={true_n}  pred={pred_n}  {'Y' if ok else 'X'}\n")

    print("\n" + "=" * 70)
    print("  All experiments completed!  Plots saved to report/")
    print("=" * 70)


if __name__ == "__main__":
    main()
