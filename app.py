import streamlit as st
import numpy as np
import soundfile as sf
import io
from scipy.signal import butter, lfilter, sawtooth, square
import plotly.graph_objects as go

# ==================================================
# 🔧 Helper Functions
# ==================================================
def db_to_gain(db):
    return 10 ** (db / 20)

def apply_filter(data, fs, filter_type, cutoff_low=None, cutoff_high=None, order=4):
    """Apply LPF, HPF, atau BPF"""
    nyq = 0.5 * fs
    if filter_type == 'low':
        btype = 'low'
        cutoff = cutoff_low / nyq
    elif filter_type == 'high':
        btype = 'high'
        cutoff = cutoff_high / nyq
    elif filter_type == 'band':
        btype = 'band'
        cutoff = [cutoff_low / nyq, cutoff_high / nyq]
    else:
        return data
    b, a = butter(order, cutoff, btype=btype)
    return lfilter(b, a, data)

def mix_audio(audio1, audio2, gain1, gain2, pan1, pan2):
    """Mix dua sinyal mono dengan gain (dB) dan pan (-1..1)"""
    audio1 *= db_to_gain(gain1)
    audio2 *= db_to_gain(gain2)
    left = audio1 * (1 - pan1) + audio2 * (1 - pan2)
    right = audio1 * (1 + pan1) + audio2 * (1 + pan2)
    return np.vstack((left, right)).T

def generate_waveform(wave_type, freq, duration, fs):
    """Generate sinyal dasar"""
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

# ==================================================
# 📊 VISUALISASI INTERAKTIF (Plotly)
# ==================================================
def visualize_waveform(wave, fs, title="Waveform", duration_display=0.02):
    samples_to_show = int(fs * duration_display)
    samples_to_show = min(samples_to_show, len(wave))
    t = np.linspace(0, samples_to_show / fs, samples_to_show, endpoint=False)
    wave_segment = wave[:samples_to_show]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=wave_segment,
        mode='lines',
        line=dict(color='#1E88E5', width=1.5),
        name="Amplitudo"
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#222831')),
        xaxis_title="Waktu (detik)",
        yaxis_title="Amplitudo",
        template="plotly_white",
        height=350,
        hovermode="x unified",
        margin=dict(l=50, r=30, t=50, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

def visualize_spectrum(wave, fs, title="Spektrum Frekuensi", show_cutoff=True):
    N = len(wave)
    f = np.fft.rfftfreq(N, 1/fs)
    mag = np.abs(np.fft.rfft(wave))
    fft_db = 20 * np.log10(mag / np.max(mag) + 1e-10)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=f, y=fft_db,
        mode='lines',
        line=dict(color='#E64A19', width=1.6),
        name="Spektrum (dB)"
    ))

    if show_cutoff:
        fig.add_vrect(x0=0, x1=250, fillcolor="skyblue", opacity=0.1, layer="below", line_width=0)
        fig.add_vrect(x0=500, x1=4000, fillcolor="lightgreen", opacity=0.1, layer="below", line_width=0)
        fig.add_vrect(x0=5000, x1=fs/2, fillcolor="lightcoral", opacity=0.1, layer="below", line_width=0)
        fig.add_vline(x=250, line_dash="dash", line_color="blue", opacity=0.5)
        fig.add_vline(x=4000, line_dash="dash", line_color="green", opacity=0.5)
        fig.add_vline(x=5000, line_dash="dash", line_color="red", opacity=0.5)

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#222831')),
        xaxis_title="Frekuensi (Hz)",
        yaxis_title="Magnitudo (dB)",
        template="plotly_white",
        height=350,
        hovermode="x unified",
        margin=dict(l=50, r=30, t=50, b=40),
        xaxis=dict(range=[0, fs/2])
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# 🧠 Streamlit App
# ==================================================
st.set_page_config(
    page_title="Aplikasi Audio Mixer Kelompok 2",
    page_icon="https://cdn-icons-png.flaticon.com/512/168/168821.png",
    layout="centered"
)

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
            st.error(" Sample rate kedua file harus sama!")
        else:
            if len(data1.shape) > 1: data1 = np.mean(data1, axis=1)
            if len(data2.shape) > 1: data2 = np.mean(data2, axis=1)

            st.subheader(" Channel Control")
            col1, col2 = st.columns(2)
            with col1:
                gain1 = st.slider("Gain Ch1 (dB)", -20, 12, 0)
                pan1 = st.slider("Pan Ch1 (-1=L, +1=R)", -1.0, 1.0, 0.0)
            with col2:
                gain2 = st.slider("Gain Ch2 (dB)", -20, 12, 0)
                pan2 = st.slider("Pan Ch2 (-1=L, +1=R)", -1.0, 1.0, 0.0)

            mixed = mix_audio(data1, data2, gain1, gain2, pan1, pan2)
            temp_mix = io.BytesIO()
            sf.write(temp_mix, mixed, fs1, format='wav')
            temp_mix.seek(0)
            st.audio(temp_mix, format='audio/wav')

            # ==== Equalizer ====
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

            temp_eq = io.BytesIO()
            sf.write(temp_eq, eq_stereo, fs1, format='wav')
            temp_eq.seek(0)
            st.audio(temp_eq, format='audio/wav')

            # ==== Analisis ====
            st.subheader("📈 Analisis Domain Waktu")
            zoom_dur = st.slider("Durasi tampilan (detik)", 0.001, 0.1, 0.02, step=0.001, key="zoom_eq")
            st.caption(
                f"Durasi tampilan (detik), "
                f"Saat ini: {zoom_dur:.3f} detik."
            )
            visualize_waveform(left, fs1, "Sebelum EQ (Left Channel)", duration_display=zoom_dur)
            visualize_waveform(eq, fs1, "Sesudah EQ (Left Channel)", duration_display=zoom_dur)

            st.subheader("📊 Analisis Spektrum Frekuensi")
            visualize_spectrum(left, fs1, "Spektrum Sebelum EQ")
            visualize_spectrum(eq, fs1, "Spektrum Sesudah EQ")

            # ==== Download ====
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
        freq = st.number_input("Frekuensi (Hz)", min_value=1, max_value=20000, value=440, step=1)
        duration = st.number_input("Durasi (detik)", min_value=0.1, max_value=60.0, value=3.0, step=0.1)


    # Simpan hasil generator di session_state agar tidak hilang saat UI berubah
    if "wave_data" not in st.session_state:
        st.session_state.wave_data = None
        st.session_state.fs = None
        st.session_state.freq = None
        st.session_state.wave_type = None

    # Tombol generate
    if st.button("🎵 Generate Audio"):
        wave = generate_waveform(wave_type, freq, duration, fs)
        st.session_state.wave_data = wave
        st.session_state.fs = fs
        st.session_state.freq = freq
        st.session_state.wave_type = wave_type

    # Jika ada data tersimpan, tampilkan
    if st.session_state.wave_data is not None:
        wave = st.session_state.wave_data
        fs = st.session_state.fs
        freq = st.session_state.freq
        wave_type = st.session_state.wave_type

        stereo = np.vstack((wave, wave)).T
        buf = io.BytesIO()
        sf.write(buf, stereo, fs, format='wav')
        buf.seek(0)

        st.subheader("▶️ Preview Suara")
        st.audio(buf, format="audio/wav")

        st.download_button(
            label="💾 Download File .WAV",
            data=buf,
            file_name=f"{wave_type}_{freq}Hz.wav",
            mime="audio/wav"
        )

        # Slider durasi tampilan
        zoom_dur = st.slider(
            "Durasi tampilan (detik)",
            0.001, 0.1, 0.02, step=0.001,
            key="zoom_gen"
        )
        st.caption(
            f"Durasi tampilan (detik), "
            f"Saat ini: {zoom_dur:.3f} detik."
        )

        visualize_waveform(wave, fs, f"{wave_type} Wave - {freq} Hz", duration_display=zoom_dur)
        visualize_spectrum(wave, fs, f"Spektrum {wave_type} {freq} Hz")


