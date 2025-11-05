# app.py
import streamlit as st
import numpy as np
import soundfile as sf
import io
from scipy.signal import butter, lfilter, sawtooth, square
import plotly.graph_objects as go

# ===========================
# Helper functions
# ===========================
def db_to_gain(db):
    return 10 ** (db / 20)

def apply_filter(data, fs, filter_type, cutoff_low=None, cutoff_high=None, order=4):
    """
    Apply LPF, HPF or BPF using Butterworth IIR (forward filtering).
    If cutoff values are invalid relative to Nyquist, returns original data.
    """
    nyq = 0.5 * fs
    try:
        if filter_type == 'low':
            if cutoff_low is None or cutoff_low <= 0 or cutoff_low >= nyq: return data
            b, a = butter(order, cutoff_low / nyq, btype='low')
        elif filter_type == 'high':
            if cutoff_high is None or cutoff_high <= 0 or cutoff_high >= nyq: return data
            b, a = butter(order, cutoff_high / nyq, btype='high')
        elif filter_type == 'band':
            if (cutoff_low is None or cutoff_high is None or
                cutoff_low <= 0 or cutoff_high >= nyq or cutoff_low >= cutoff_high): return data
            b, a = butter(order, [cutoff_low / nyq, cutoff_high / nyq], btype='band')
        else:
            return data
        return lfilter(b, a, data)
    except Exception:
        # On any numerical error, fallback to raw data
        return data

def mix_audio(audio1, audio2, gain1_db, gain2_db, pan1, pan2):
    """
    Mix two mono signals with dB gains and pan (-1..1).
    Returns a stereo Nx2 numpy array.
    """
    # Ensure same length
    n = max(len(audio1), len(audio2))
    a1 = np.zeros(n); a2 = np.zeros(n)
    a1[:len(audio1)] = audio1
    a2[:len(audio2)] = audio2

    a1 = a1 * db_to_gain(gain1_db)
    a2 = a2 * db_to_gain(gain2_db)

    left = a1 * (1 - pan1) + a2 * (1 - pan2)
    right = a1 * (1 + pan1) + a2 * (1 + pan2)
    stereo = np.vstack((left, right)).T
    # prevent clipping
    maxv = np.max(np.abs(stereo))
    if maxv > 1:
        stereo = stereo / maxv
    return stereo

def generate_waveform(wave_type, freq, duration, fs):
    """Generate Sine, Square, Triangle or Noise; normalized to ±1."""
    if duration <= 0 or fs <= 0:
        return np.array([])
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
    if np.max(np.abs(y)) == 0:
        return y
    return y / np.max(np.abs(y))

# ===========================
# Plotly visualizations
# ===========================
def visualize_waveform_plotly(wave, fs, title="Waveform", duration_display=0.02):
    """Interactive time-domain plot with Plotly."""
    if wave is None or len(wave) == 0:
        st.info("Tidak ada data sinyal untuk ditampilkan.")
        return
    samples_to_show = int(fs * duration_display)
    samples_to_show = min(samples_to_show, len(wave))
    if samples_to_show < 2:
        st.warning("Durasi tampilan terlalu kecil untuk divisualisasikan. Tingkatkan durasi.")
        return
    t = np.linspace(0, samples_to_show / fs, samples_to_show, endpoint=False)
    y = wave[:samples_to_show]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode='lines', line=dict(color='#1E88E5', width=1.6), name='Amplitudo'))
    # autoscale Y with small margin
    ymin, ymax = float(np.min(y)), float(np.max(y))
    margin = max(0.02, (ymax - ymin) * 0.1)
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis_title="Waktu (detik)",
        yaxis_title="Amplitudo",
        template="plotly_white",
        height=360,
        hovermode="x unified",
        margin=dict(l=50, r=30, t=50, b=40),
        yaxis=dict(range=[ymin - margin, ymax + margin])
    )
    st.plotly_chart(fig, use_container_width=True)

def visualize_spectrum_plotly(wave, fs, title="Spektrum Frekuensi", show_cutoff=True):
    """Interactive frequency-domain plot with Plotly. Normalized so peak = 0 dB."""
    if wave is None or len(wave) < 2:
        st.info("Tidak ada data sinyal cukup panjang untuk FFT.")
        return
    N = len(wave)
    f = np.fft.rfftfreq(N, 1.0/fs)
    mag = np.abs(np.fft.rfft(wave))
    # avoid divide by zero
    peak = np.max(mag) if np.max(mag) > 0 else 1.0
    fft_db = 20 * np.log10(mag / peak + 1e-12)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f, y=fft_db, mode='lines', line=dict(color='#E64A19', width=1.6), name='Magnitude (dB)'))

    if show_cutoff:
        fig.add_vrect(x0=0, x1=250, fillcolor="skyblue", opacity=0.10, layer="below", line_width=0)
        fig.add_vrect(x0=500, x1=4000, fillcolor="lightgreen", opacity=0.10, layer="below", line_width=0)
        fig.add_vrect(x0=5000, x1=fs/2, fillcolor="lightcoral", opacity=0.10, layer="below", line_width=0)
        fig.add_vline(x=250, line_dash="dash", line_color="blue", opacity=0.5)
        fig.add_vline(x=4000, line_dash="dash", line_color="green", opacity=0.5)
        fig.add_vline(x=5000, line_dash="dash", line_color="red", opacity=0.5)

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis_title="Frekuensi (Hz)",
        yaxis_title="Magnitude (dB, peak=0dB)",
        template="plotly_white",
        height=360,
        hovermode="x unified",
        margin=dict(l=50, r=30, t=50, b=40),
        xaxis=dict(range=[0, fs/2])
    )
    st.plotly_chart(fig, use_container_width=True)

# ===========================
# Streamlit app
# ===========================
st.set_page_config(page_title="Audio Mixer + EQ + Generator", page_icon="🎚️", layout="centered")
st.title("🎚️ Software-Defined Audio Mixer + Equalizer + Generator")
st.caption("Kelompok 2 • Digital Signal Processing (DSK) • 2025")

tab1, tab2 = st.tabs(["🎵 Mixer & Equalizer", "🎛️ Generator Sinyal"])

# ---------------------------
# TAB 1: Mixer & EQ
# ---------------------------
with tab1:
    st.header("🎧 Mixer & Equalizer")

    file1 = st.file_uploader("Upload Audio Channel 1 (.wav)", type=["wav"], key="upload_ch1")
    file2 = st.file_uploader("Upload Audio Channel 2 (.wav)", type=["wav"], key="upload_ch2")

    if file1 and file2:
        data1, fs1 = sf.read(file1)
        data2, fs2 = sf.read(file2)

        if fs1 != fs2:
            st.error("⚠️ Sample rate kedua file harus sama!")
        else:
            # convert to mono if stereo
            if len(data1.shape) > 1: data1 = np.mean(data1, axis=1)
            if len(data2.shape) > 1: data2 = np.mean(data2, axis=1)

            st.subheader("🎚️ Channel Control")
            col1, col2 = st.columns(2)
            with col1:
                gain1 = st.slider("Gain Ch1 (dB)", -20, 12, 0, key="gain_ch1")
                pan1 = st.slider("Pan Ch1 (-1=L, +1=R)", -1.0, 1.0, 0.0, key="pan_ch1")
            with col2:
                gain2 = st.slider("Gain Ch2 (dB)", -20, 12, 0, key="gain_ch2")
                pan2 = st.slider("Pan Ch2 (-1=L, +1=R)", -1.0, 1.0, 0.0, key="pan_ch2")

            # Mix and preview
            mixed = mix_audio(data1, data2, gain1, gain2, pan1, pan2)
            temp_mix = io.BytesIO()
            sf.write(temp_mix, mixed, fs1, format='wav')
            temp_mix.seek(0)
            st.audio(temp_mix, format='audio/wav')

            # Equalizer
            st.subheader("🎛️ 3-Band Equalizer")
            bass_gain = st.slider("Bass (LPF @250Hz, dB)", -12, 12, 0, key="bass_gain")
            mid_gain = st.slider("Mid (BPF 500–4kHz, dB)", -12, 12, 0, key="mid_gain")
            treble_gain = st.slider("Treble (HPF @5kHz, dB)", -12, 12, 0, key="treble_gain")

            left = mixed[:, 0]
            bass = apply_filter(left, fs1, 'low', cutoff_low=250) * db_to_gain(bass_gain)
            mid = apply_filter(left, fs1, 'band', cutoff_low=500, cutoff_high=4000) * db_to_gain(mid_gain)
            treble = apply_filter(left, fs1, 'high', cutoff_high=5000) * db_to_gain(treble_gain)

            eq = bass + mid + treble
            # avoid divide by zero
            if np.max(np.abs(eq)) > 0:
                eq = eq / np.max(np.abs(eq))

            eq_stereo = np.vstack((eq, eq)).T
            temp_eq = io.BytesIO()
            sf.write(temp_eq, eq_stereo, fs1, format='wav')
            temp_eq.seek(0)
            st.audio(temp_eq, format='audio/wav')

            # Analysis section
            st.subheader("📈 Analisis Sebelum & Sesudah EQ")
            st.markdown("🕒 **Durasi tampilan (detik)**: atur berapa panjang potongan sinyal (zoom).")
            zoom_dur = st.slider("Durasi tampilan (detik)", 0.001, 0.2, 0.02, step=0.001, key="zoom_eq")

            visualize_waveform_plotly(left, fs1, "Sebelum EQ (Left Channel)", duration_display=zoom_dur)
            visualize_waveform_plotly(eq, fs1, "Sesudah EQ (Left Channel)", duration_display=zoom_dur)

            st.subheader("📊 Spektrum Frekuensi (Interaktif)")
            visualize_spectrum_plotly(left, fs1, "Spektrum Sebelum EQ")
            visualize_spectrum_plotly(eq, fs1, "Spektrum Sesudah EQ")

            st.download_button(label="💾 Download hasil Mixdown (.wav)",
                               data=temp_eq, file_name="mixdown_output.wav", mime="audio/wav")

# ---------------------------
# TAB 2: Signal Generator
# ---------------------------
with tab2:
    st.header("🎛️ Signal Generator")

    col1, col2 = st.columns(2)
    with col1:
        wave_type = st.selectbox("Jenis Gelombang", ["Sine", "Square", "Triangle", "Noise"], key="gen_wave_type")
        fs = st.number_input("Sample Rate (Hz)", min_value=8000, max_value=96000, value=44100, step=1000, key="gen_fs")
    with col2:
        freq = st.number_input("Frekuensi (Hz)", min_value=1, max_value=20000, value=440, step=1, key="gen_freq")
        duration = st.number_input("Durasi (detik)", min_value=0.1, max_value=60.0, value=3.0, step=0.1, key="gen_duration")

    st.info("Klik **Generate Audio** untuk membuat sinyal, lalu atur Durasi tampilan tanpa membuat data hilang.")

    # initialize session_state storage for generator
    if "gen_wave" not in st.session_state:
        st.session_state.gen_wave = None
        st.session_state.gen_fs = None
        st.session_state.gen_freq = None
        st.session_state.gen_type = None

    if st.button("🎵 Generate Audio", key="gen_button"):
        # generate and store in session state
        w = generate_waveform(wave_type, freq, duration, fs)
        st.session_state.gen_wave = w
        st.session_state.gen_fs = fs
        st.session_state.gen_freq = freq
        st.session_state.gen_type = wave_type

    # if stored signal exists, show controls/plots
    if st.session_state.get("gen_wave") is not None:
        wave = st.session_state.gen_wave
        gen_fs = st.session_state.gen_fs
        gen_freq = st.session_state.gen_freq
        gen_type = st.session_state.gen_type

        # audio preview & download
        stereo = np.vstack((wave, wave)).T
        buf = io.BytesIO()
        sf.write(buf, stereo, gen_fs, format='wav')
        buf.seek(0)

        st.subheader("▶️ Preview Suara")
        st.audio(buf, format='audio/wav')
        st.download_button(label="💾 Download File .WAV", data=buf,
                           file_name=f"{gen_type}_{gen_freq}Hz_{len(wave)/gen_fs:.2f}s.wav",
                           mime="audio/wav")

        st.markdown("🕒 **Durasi tampilan (detik)**: atur berapa panjang potongan sinyal yang ditampilkan (zoom).")
        zoom_dur = st.slider("Durasi tampilan (detik)", 0.001, 0.2, 0.02, step=0.001, key="zoom_gen")

        visualize_waveform_plotly(wave, gen_fs, f"{gen_type} Wave - {gen_freq} Hz", duration_display=zoom_dur)
        visualize_spectrum_plotly(wave, gen_fs, f"Spektrum {gen_type} {gen_freq} Hz")

