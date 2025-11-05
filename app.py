import streamlit as st
import numpy as np
import soundfile as sf
import io
from scipy.signal import butter, lfilter
from pydub import AudioSegment
import tempfile

# -------------------------------
# Helper Functions
# -------------------------------

def db_to_gain(db):
    """Convert dB value to linear gain."""
    return 10 ** (db / 20)

def apply_filter(data, fs, filter_type, cutoff_low=None, cutoff_high=None, order=4):
    """Apply LPF/BPF/HPF filter."""
    nyq = 0.5 * fs
    if filter_type == 'low':
        normal_cutoff = cutoff_low / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
    elif filter_type == 'high':
        normal_cutoff = cutoff_high / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
    elif filter_type == 'band':
        normal_cutoff = [cutoff_low / nyq, cutoff_high / nyq]
        b, a = butter(order, normal_cutoff, btype='band', analog=False)
    else:
        return data
    return lfilter(b, a, data)

def mix_audio(audio1, audio2, gain1, gain2, pan1, pan2):
    """Mix two mono signals with gain and panning control."""
    # Apply gain
    audio1 = audio1 * db_to_gain(gain1)
    audio2 = audio2 * db_to_gain(gain2)

    # Stereo panning (simple balance control)
    left1 = audio1 * (1 - pan1)
    right1 = audio1 * (1 + pan1)
    left2 = audio2 * (1 - pan2)
    right2 = audio2 * (1 + pan2)

    # Combine into stereo
    left = left1 + left2
    right = right1 + right2

    stereo = np.vstack((left, right)).T
    return stereo

# -------------------------------
# Streamlit UI
# -------------------------------

st.title("🎚️ Software-Defined Audio Mixer & Equalizer")
st.write("Dibuat untuk Proyek DSK — menggunakan Streamlit & Python")

# Upload 2 audio files
file1 = st.file_uploader("Upload Audio Channel 1 (.wav)", type=["wav"])
file2 = st.file_uploader("Upload Audio Channel 2 (.wav)", type=["wav"])

if file1 and file2:
    # Load files
    data1, fs1 = sf.read(file1)
    data2, fs2 = sf.read(file2)

    if fs1 != fs2:
        st.error("Sample rate kedua file harus sama!")
    else:
        st.success(f"Sample Rate: {fs1} Hz")

        # Convert stereo to mono if needed
        if len(data1.shape) > 1:
            data1 = np.mean(data1, axis=1)
        if len(data2.shape) > 1:
            data2 = np.mean(data2, axis=1)

        # Controls
        st.subheader("🎚️ Channel Control")
        col1, col2 = st.columns(2)
        with col1:
            gain1 = st.slider("Gain Ch1 (dB)", -20, 12, 0)
            pan1 = st.slider("Pan Ch1 (-1=L, +1=R)", -1.0, 1.0, 0.0)
        with col2:
            gain2 = st.slider("Gain Ch2 (dB)", -20, 12, 0)
            pan2 = st.slider("Pan Ch2 (-1=L, +1=R)", -1.0, 1.0, 0.0)

        # Mix
        mixed = mix_audio(data1, data2, gain1, gain2, pan1, pan2)

        st.audio(sf.write(io.BytesIO(), mixed, fs1, format='wav'))

        # Equalizer Section
        st.subheader("🎛️ 3-Band Equalizer")
        bass_gain = st.slider("Bass (LPF @ 250Hz)", -12, 12, 0)
        mid_gain = st.slider("Mid (BPF 500Hz–4kHz)", -12, 12, 0)
        treble_gain = st.slider("Treble (HPF @ 5kHz)", -12, 12, 0)

        # Apply filters
        bass = apply_filter(mixed[:,0], fs1, 'low', cutoff_low=250) * db_to_gain(bass_gain)
        mid = apply_filter(mixed[:,0], fs1, 'band', cutoff_low=500, cutoff_high=4000) * db_to_gain(mid_gain)
        treble = apply_filter(mixed[:,0], fs1, 'high', cutoff_high=5000) * db_to_gain(treble_gain)

        # Combine EQ output
        eq_left = bass + mid + treble
        eq_right = eq_left  # same for simplicity
        eq_stereo = np.vstack((eq_left, eq_right)).T

        # Normalize output
        eq_stereo /= np.max(np.abs(eq_stereo))

        # Play output
        st.audio(sf.write(io.BytesIO(), eq_stereo, fs1, format='wav'))

        # Save output
        if st.button("💾 Simpan hasil Mixdown (.wav)"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, eq_stereo, fs1)
                st.download_button("Download hasil mixdown", tmp.read(), "mixdown_output.wav")

