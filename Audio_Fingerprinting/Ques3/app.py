import os
import io
import csv
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from fingerprint import (
    load_audio, compute_spectrogram, find_peaks,
    generate_hashes, match_hashes, get_offset_histogram,
    DEFAULT_SR, DEFAULT_N_FFT, DEFAULT_HOP,
)
from database import load_database, build_database


st.set_page_config(
    page_title=" EE200 — Audio Fingerprinting",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .hero-header h1 {
        background: linear-gradient(90deg, #00d2ff, #3a7bd5, #ff6fd8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .hero-header p {
        color: #b0b0d0;
        font-size: 1.1rem;
        font-weight: 300;
    }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(0, 210, 255, 0.3);
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(0, 210, 255, 0.1);
    }
    .result-card h2 {
        color: #00d2ff;
        font-size: 1.6rem;
        margin-bottom: 0.5rem;
    }
    .result-card .song-name {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .result-card .confidence {
        color: #aaa;
        font-size: 0.95rem;
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #0f0c29;
    }

    /* Plot containers */
    .plot-container {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255,255,255,0.05);
    }

    /* Batch result */
    .batch-row {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        display: flex;
        justify-content: space-between;
        border-left: 3px solid #00d2ff;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(0,210,255,0.15);
        border-radius: 12px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# State & Database 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SONG_DIR = os.path.join(SCRIPT_DIR, "Song database")
DB_PATH = os.path.join(SCRIPT_DIR, "fingerprint_db.pkl")


@st.cache_resource(show_spinner="Building fingerprint database (first run only)...")
def get_database():
    """Load or build the fingerprint database."""
    if os.path.exists(DB_PATH):
        return load_database(DB_PATH)
    else:
        return build_database(SONG_DIR, db_path=DB_PATH)


database, song_names = get_database()

# Sidebar
with st.sidebar:
    st.markdown("Settings")
    mode = st.radio("Mode", ["Single Clip", "Batch Mode"],
                    help="Single-clip shows visualisations; batch processes multiple files.")
    st.markdown("---")
    st.markdown(f"Database: {len(song_names)} songs indexed")
    st.markdown(f"Unique hashes: {len(database):,}")
    st.markdown("---")
    st.markdown("Song Library")
    for sid in sorted(song_names, key=lambda k: song_names[k]):
        st.markdown(f"• {song_names[sid]}")


st.markdown("""
<div class="hero-header">
    <h1>EE200: Audio Fingerprinting</h1>
    <p>Upload an audio clip and identify the song</p>
</div>
""", unsafe_allow_html=True)


# Single-clip mode
if mode == "Single Clip":
    uploaded = st.file_uploader("Upload a query clip (MP3 or WAV)", type=["mp3", "wav"],
                                key="single_upload")
    if uploaded is not None:
        # Save temp file
        suffix = "." + uploaded.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        try:
            with st.spinner("Analysing audio..."):
                y, sr = load_audio(tmp_path)
                S_db, freqs, times = compute_spectrogram(y, sr=sr)
                peaks = find_peaks(S_db)
                hashes = generate_hashes(peaks)
                best_id, best_count, offset_counts = match_hashes(hashes, database)
                match_name = song_names.get(best_id, "Unknown")

            # Result card 
            st.markdown(f"""
            <div class="result-card">
                <h2>Match Found!</h2>
                <div class="song-name">{match_name}</div>
                <div class="confidence">Confidence score: {best_count} matching hash offsets</div>
            </div>
            """, unsafe_allow_html=True)

            # Audio playback 
            st.audio(tmp_path)

            # Visualisations 
            st.markdown("### Intermediate Steps")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("Spectrogram")
                fig1, ax1 = plt.subplots(figsize=(8, 4))
                fig1.patch.set_facecolor("#1a1a2e")
                ax1.set_facecolor("#1a1a2e")
                img = ax1.imshow(S_db, aspect="auto", origin="lower",
                                 extent=[times[0], times[-1], freqs[0], freqs[-1]],
                                 cmap="magma", vmin=-80, vmax=0)
                ax1.set_xlabel("Time (s)", color="white")
                ax1.set_ylabel("Frequency (Hz)", color="white")
                ax1.set_ylim(0, 5000)
                ax1.tick_params(colors="white")
                fig1.colorbar(img, ax=ax1, label="dB")
                fig1.tight_layout()
                st.pyplot(fig1)
                plt.close(fig1)

            with col2:
                st.markdown("Constellation Map")
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                fig2.patch.set_facecolor("#1a1a2e")
                ax2.set_facecolor("#1a1a2e")
                ax2.imshow(S_db, aspect="auto", origin="lower",
                           extent=[times[0], times[-1], freqs[0], freqs[-1]],
                           cmap="magma", vmin=-80, vmax=0, alpha=0.4)
                peak_t = times[np.clip(peaks[:, 1], 0, len(times) - 1)]
                peak_f = freqs[np.clip(peaks[:, 0], 0, len(freqs) - 1)]
                ax2.scatter(peak_t, peak_f, s=3, c="cyan", alpha=0.7)
                ax2.set_xlabel("Time (s)", color="white")
                ax2.set_ylabel("Frequency (Hz)", color="white")
                ax2.set_ylim(0, 5000)
                ax2.set_title(f"{len(peaks)} peaks detected", color="white", fontsize=10)
                ax2.tick_params(colors="white")
                fig2.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)

            # Offset histogram 
            st.markdown("Offset Histogram (Top Matches)")
            # Show histograms for top-3 songs by best offset count
            scored = []
            for sid, oc in offset_counts.items():
                mx = max(oc.values()) if oc else 0
                scored.append((mx, sid))
            scored.sort(reverse=True)
            top_n = min(3, len(scored))

            fig3, axes3 = plt.subplots(1, top_n, figsize=(6 * top_n, 4))
            fig3.patch.set_facecolor("#1a1a2e")
            if top_n == 1:
                axes3 = [axes3]
            for k, (mx, sid) in enumerate(scored[:top_n]):
                ax = axes3[k]
                ax.set_facecolor("#1a1a2e")
                oc = offset_counts[sid]
                offsets, counts = get_offset_histogram(oc)
                color = "#00d2ff" if sid == best_id else "#555"
                if len(offsets) > 0:
                    ax.bar(offsets, counts, width=1, color=color, alpha=0.85)
                ax.set_title(f"{song_names[sid]}\n(max={mx})",
                             color="white", fontsize=9)
                ax.set_xlabel("Offset", color="white")
                ax.set_ylabel("Matches", color="white")
                ax.tick_params(colors="white")
            fig3.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

            # Metrics 
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Peaks", f"{len(peaks):,}")
            m2.metric("Hashes", f"{len(hashes):,}")
            m3.metric("Best Count", best_count)
            m4.metric("Duration", f"{len(y)/sr:.1f}s")

        finally:
            os.unlink(tmp_path)



# Batch mode
else:
    st.markdown("Batch Mode")
    st.markdown("Upload multiple audio clips. The app will identify each and "
                "produce a `results.csv` with columns `filename` and `prediction`.")

    uploaded_files = st.file_uploader("Upload query clips",
                                      type=["mp3", "wav"],
                                      accept_multiple_files=True,
                                      key="batch_upload")

    if uploaded_files:
        results = []
        progress = st.progress(0, text="Processing clips...")

        for i, uf in enumerate(uploaded_files):
            suffix = "." + uf.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uf.read())
                tmp_path = tmp.name
            try:
                y, sr = load_audio(tmp_path)
                S_db, _, _ = compute_spectrogram(y, sr=sr)
                peaks = find_peaks(S_db)
                hashes = generate_hashes(peaks)
                best_id, best_count, _ = match_hashes(hashes, database)
                prediction = song_names.get(best_id, "Unknown")
                results.append({"filename": uf.name, "prediction": prediction})
            finally:
                os.unlink(tmp_path)
            progress.progress((i + 1) / len(uploaded_files),
                              text=f"Processed {i + 1}/{len(uploaded_files)}")

        progress.empty()

        # Display results
        st.markdown("Results")
        for r in results:
            st.markdown(f"""
            <div class="batch-row">
                <span style="color:#ccc">{r['filename']}</span>
                <span style="color:#00d2ff; font-weight:600">{r['prediction']}</span>
            </div>
            """, unsafe_allow_html=True)

        # Generate CSV
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=["filename", "prediction"])
        writer.writeheader()
        writer.writerows(results)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")

        st.download_button(
            label="Download results.csv",
            data=csv_bytes,
            file_name="results.csv",
            mime="text/csv",
        )
