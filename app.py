import streamlit as st
import numpy as np
import soundfile as sf
import io
from scipy.signal import sawtooth, square
import tempfile

# ==================================================
# 🎶 Fungsi Pembuat Gelombang
# ==================================================
def generate_waveform(wave_type, freq, duration, fs):
    """Generate audio waveform: sine, square, triangle, or noise."""
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
    return y / np.max(np.abs(y))  # Normalisasi agar tidak pecah

# ==================================================
# 🎛️ Tampilan Streamlit
# ==================================================
st.set_page_config(page_title="Audio Signal Generator", page_icon="🎚️", layout="centered")
st.title("🎚️ Audio Signal Generator")
st.write("Buat dan dengarkan suara **Sine**, **Triangle**, **Square**, atau **Noise** dengan frekuensi dan durasi yang bisa kamu atur sendiri!")

# ==================================================
# ⚙️ Input Parameter
# ==================================================
col1, col2 = st.columns(2)
with col1:
    wave_type = st.selectbox("Jenis Gelombang", ["Sine", "Square", "Triangle", "Noise"])
    fs = st.number_input("Sample Rate (Hz)", 8000, 48000, 44100)
with col2:
    freq = st.slider("Frekuensi (Hz)", 20, 5000, 440)
    duration = st.slider("Durasi (detik)", 1, 10, 3)

# ==================================================
# 🔊 Generate Button
# ==================================================
if st.button("🎵 Generate Audio"):
    wave = generate_waveform(wave_type, freq, duration, fs)
    stereo = np.vstack((wave, wave)).T  # stereo kiri-kanan sama

    # Simpan ke buffer
    buf = io.BytesIO()
    sf.write(buf, stereo, fs, format='wav')
    buf.seek(0)

    st.subheader("▶️ Hasil Audio")
    st.audio(buf, format="audio/wav")

    # ==================================================
    # 💾 Download Button
    # ==================================================
    st.download_button(
        label="💾 Download File .WAV",
        data=buf,
        file_name=f"{wave_type}_{freq}Hz_{duration}s.wav",
        mime="audio/wav"
    )

    # ==================================================
    # 📈 Visualisasi Gelombang
    # ==================================================
    import matplotlib.pyplot as plt

    st.subheader("📊 Visualisasi Gelombang")
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    plt.figure(figsize=(8, 3))
    plt.plot(t[:1000], wave[:1000])  # tampilkan 1000 sample pertama
    plt.title(f"{wave_type} Wave - {freq} Hz")
    plt.xlabel("Waktu (detik)")
    plt.ylabel("Amplitudo")
    st.pyplot(plt)
