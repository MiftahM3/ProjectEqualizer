import streamlit as st
import numpy as np
import soundfile as sf
import io
import tempfile
from scipy.signal import butter, lfilter, sawtooth, square
import matplotlib.pyplot as plt

# ==================================================
# 🔧 Helper Functions
# ==================================================
def db_to_gain(db):
    return 10 ** (db / 20)

def apply_filter(data, fs, filter_type, cutoff_low=None, cutoff_high=None, order=4):
    """Apply LPF, HPF, or BPF."""
    nyq = 0.5 * fs
    if filter_type == 'low':
        b, a = butter(order, cutoff_low / nyq, btype='low')
    elif filter_type == 'high':
        b, a = butter(order, cutoff_high / nyq, btype='high')
    elif filter_type == 'band':
        b, a = butter(order, [cutoff_low / nyq, cutoff_high / nyq], btype='band')
    else:
        return data
    return lfilter(b, a, data)

def mix_audio(audio1, audio2, gain1, gain2, pan1, pan2):
    """Mix two mono signals with gain (dB) and pan (-1..1)."""
    audio1 *= db_to_gain(gain1)
    audio2 *= db_to_gain(gain2)
    left = audio1 * (1 - pan1) + audio2 * (1 - pan2)
    right = audio1 * (1 + pan1) + audio2 * (1 + pan2)
    return np.vstack((left, right)).T

def generate_waveform(wave_type, freq, duration, fs):
    """Generate basic waveform."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    if wave_type == "Sine":
        y = np.sin(2 * np.pi * freq * t)
    elif wave_type == "Square":
        y = square(2 * np.pi * freq * t)
    elif wave_type == "Triangle":
        y = sawtooth(2 * np.pi * freq * t, 0.5)
    elif wave_type == "Noise":
        y = np.random.uniform(-1, 1, len(t))
    else:
        y = np.zeros_like(t)
    return y / np.max(np.abs(y))

def visualize_waveform(wave, fs, title="Waveform"):
    """Plot waveform for time-domain visualization."""
    t = np.linspace(0, len(wave)/fs, len(wave))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t[:1000], wave[:1000])
    ax.set_title(title)
    ax.set_xlabel("Waktu (detik)")
    ax.set_ylabel("Amplitudo")
    st.pyplot(fig)

# ==================================================
# 🧠 Streamlit App
# ==================================================
st.set_page_config(page_title="Software-Defined Audio Mixer", page_icon="🎚️", layout="centered")
st.title("🎚️ Software-Defined Audio Mixer + Generator + Equalizer")
st.write("Aplikasi proyek DSK: **Audio Mixer, Equalizer, dan Generator Sinyal Digital**")

# ==================================================
# 🔀 Tabs
# ==================================================
tab1, tab2 = st.tabs(["🎵 Mixer & Equalizer", "🎛️ Generator Sinyal"])

# ==================================================
# TAB 1: MIXER & EQ
# ==================================================
with tab1:
    st.header("🎧 Mixer & Equalizer")

    file1 = st.file_uploader("Upload Audio Channel 1 (.wav)", type=["wav"])
    file2 = st.file_uploader("Upload Audio Channel 2 (.wav)", type=["wav"])

    if file1 and file2:
        data1, fs1 = sf.read(file1)
        data2, fs2 = sf.read(file2)

        if fs1 != fs2:
            st.error("Sample rate kedua file harus sama!")
        else:
            if len(data1.shape) > 1:
                data1 = np.mean(data1, axis=1)
            if len(data2.shape) > 1:
                data2 = np.mean(data2, axis=1)

            st.subheader("🎚️ Channel Control")
            col1, col2 = st.columns(2)
            with col1:
                gain1 = st.slider("Gain Ch1 (dB)", -20, 12, 0)
                pan1 = st.slider("Pan Ch1 (-1=L, +1=R)", -1.0, 1.0, 0.0)
            with col2:
                gain2 = st.slider("Gain Ch2 (dB)", -20, 12, 0)
                pan2 = st.slider("Pan Ch2 (-1=L, +1=R)", -1.0, 1.0, 0.0)

            # Mix two audio signals
            mixed = mix_audio(data1, data2, gain1, gain2, pan1, pan2)

            # Play preview
            temp_mix = io.BytesIO()
            sf.write(temp_mix, mixed, fs1, format='wav')
            temp_mix.seek(0)
            st.audio(temp_mix, format='audio/wav')

            # Equalizer section
            st.subheader("🎛️ 3-Band Equalizer")
            bass_gain = st.slider("Bass (LPF @ 250Hz)", -12, 12, 0)
            mid_gain = st.slider("Mid (BPF 500Hz–4kHz)", -12, 12, 0)
            treble_gain = st.slider("Treble (HPF @ 5kHz)", -12, 12, 0)

            left = mixed[:, 0]
            bass = apply_filter(left, fs1, 'low', cutoff_low=250) * db_to_gain(bass_gain)
            mid = apply_filter(left, fs1, 'band', cutoff_low=500, cutoff_high=4000) * db_to_gain(mid_gain)
            treble = apply_filter(left, fs1, 'high', cutoff_high=5000) * db_to_gain(treble_gain)

            eq = bass + mid + treble
            eq_stereo = np.vstack((eq, eq)).T
            eq_stereo /= np.max(np.abs(eq_stereo))

            # Play EQ output
            temp_eq = io.BytesIO()
            sf.write(temp_eq, eq_stereo, fs1, format='wav')
            temp_eq.seek(0)
            st.audio(temp_eq, format='audio/wav')

            st.subheader("📈 Visualisasi (Setelah EQ)")
            visualize_waveform(eq, fs1, "Output EQ")

            # Download EQ result
            st.download_button(
                label="💾 Download hasil Mixdown (.wav)",
                data=temp_eq,
                file_name="mixdown_output.wav",
                mime="audio/wav"
            )

# ==================================================
# TAB 2: SIGNAL GENERATOR
# ==================================================
with tab2:
    st.header("🎛️ Signal Generator")

    col1, col2 = st.columns(2)
    with col1:
        wave_type = st.selectbox("Jenis Gelombang", ["Sine", "Square", "Triangle", "Noise"])
        fs = st.number_input("Sample Rate (Hz)", min_value=8000, max_value=96000, value=44100, step=1000)
    with col2:
        # Frekuensi manual
        freq = st.number_input("Frekuensi (Hz)", min_value=1, max_value=20000, value=440, step=1)
        # Durasi manual
        duration = st.number_input("Durasi (detik)", min_value=0.1, max_value=60.0, value=3.0, step=0.1)

    st.info("💡 Kamu bisa ketik nilai manual di kolom atas untuk hasil yang lebih presisi.")

    if st.button("🎵 Generate Audio"):
        wave = generate_waveform(wave_type, freq, duration, fs)
        stereo = np.vstack((wave, wave)).T

        buf = io.BytesIO()
        sf.write(buf, stereo, fs, format='wav')
        buf.seek(0)

        st.subheader("▶️ Preview Suara")
        st.audio(buf, format="audio/wav")

        st.download_button(
            label="💾 Download File .WAV",
            data=buf,
            file_name=f"{wave_type}_{freq}Hz_{duration}s.wav",
            mime="audio/wav"
        )

        def visualize_waveform(wave, fs, title="Waveform", duration_display=0.01):
    """
    Menampilkan visualisasi gelombang waktu-domain yang jelas dan informatif.
    wave: data audio (numpy array)
    fs: sample rate (Hz)
    title: judul plot
    duration_display: durasi bagian awal yang ditampilkan (detik)
    """
    import matplotlib.pyplot as plt

    # Hitung jumlah sampel yang akan ditampilkan (misal 0.01 detik = 10ms)
    samples_to_show = int(fs * duration_display)
    t = np.linspace(0, duration_display, samples_to_show, endpoint=False)
    wave_segment = wave[:samples_to_show]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, wave_segment, color='royalblue', linewidth=1.5)

    ax.set_title(f"{title}\n(Sample Rate: {fs} Hz, Durasi ditampilkan: {duration_display*1000:.1f} ms)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Waktu (detik)", fontsize=11)
    ax.set_ylabel("Amplitudo", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(-1.1, 1.1)

    st.pyplot(fig)

