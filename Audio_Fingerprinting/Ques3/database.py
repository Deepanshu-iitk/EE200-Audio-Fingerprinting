"""
database.py — Fingerprint Db
"""

import os
import pickle
import time
from fingerprint import (
    load_audio, compute_spectrogram, find_peaks, generate_hashes,
    DEFAULT_SR, DEFAULT_N_FFT, DEFAULT_HOP
)


DB_FILENAME = "fingerprint_db.pkl"


def build_database(song_dir, db_path=None,
                   sr=DEFAULT_SR, n_fft=DEFAULT_N_FFT, hop_length=DEFAULT_HOP,
                   neighborhood_size=10, amp_min=-55,
                   fan_out=15, verbose=True):
    if db_path is None:
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DB_FILENAME)

    song_files = sorted(
        f for f in os.listdir(song_dir) if f.lower().endswith(('.mp3', '.wav'))
    )

    database = {}
    song_names = {}

    t0 = time.time()
    for idx, filename in enumerate(song_files):
        filepath = os.path.join(song_dir, filename)
        song_name = os.path.splitext(filename)[0]
        song_id = idx
        song_names[song_id] = song_name

        if verbose:
            print(f"  [{idx + 1:>2}/{len(song_files)}] Indexing: {song_name}")

        y, _ = load_audio(filepath, sr=sr)
        S_db, _, _ = compute_spectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        peaks = find_peaks(S_db, neighborhood_size=neighborhood_size, amp_min=amp_min)
        hashes = generate_hashes(peaks, fan_out=fan_out)

        for h, offset in hashes:
            if h not in database:
                database[h] = []
            database[h].append((song_id, offset))

    elapsed = time.time() - t0
    if verbose:
        total_entries = sum(len(v) for v in database.values())
        print(f"\n  Database built in {elapsed:.1f}s")
        print(f"  {len(song_names)} songs  |  {len(database):,} unique hashes  |  {total_entries:,} entries")

    data = {"database": database, "song_names": song_names}
    with open(db_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(f"  Saved to {db_path}\n")

    return database, song_names


def load_database(db_path=None):
    """Load a previously built fingerprint database."""
    if db_path is None:
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DB_FILENAME)
    with open(db_path, "rb") as f:
        data = pickle.load(f)
    return data["database"], data["song_names"]


def identify_clip(clip_path, database, song_names,
                  sr=DEFAULT_SR, n_fft=DEFAULT_N_FFT, hop_length=DEFAULT_HOP,
                  neighborhood_size=10, amp_min=-55, fan_out=15):
                      
    from fingerprint import match_hashes

    y, _ = load_audio(clip_path, sr=sr)
    S_db, _, _ = compute_spectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    peaks = find_peaks(S_db, neighborhood_size=neighborhood_size, amp_min=amp_min)
    hashes = generate_hashes(peaks, fan_out=fan_out)

    best_id, best_count, offset_counts = match_hashes(hashes, database)

    match_name = song_names.get(best_id) if best_id is not None else None
    return match_name, best_count, offset_counts


if __name__ == "__main__":
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    song_dir = os.path.join(script_dir, "Song database")

    if not os.path.isdir(song_dir):
        print(f"Song directory not found: {song_dir}")
        sys.exit(1)

    print("Building fingerprint database ...")
    build_database(song_dir)
    print("Done!")
