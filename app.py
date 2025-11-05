with tab2:
    st.header("🎛️ Signal Generator")

    col1, col2 = st.columns(2)
    with col1:
        wave_type = st.selectbox("Jenis Gelombang", ["Sine", "Square", "Triangle", "Noise"])
        fs = st.number_input("Sample Rate (Hz)", min_value=8000, max_value=96000, value=44100, step=1000)
    with col2:
        freq = st.number_input("Frekuensi (Hz)", min_value=1, max_value=20000, value=440, step=1)
        duration = st.number_input("Durasi (detik)", min_value=0.1, max_value=60.0, value=3.0, step=0.1)

    st.info("💡 Kamu bisa ketik nilai manual di kolom atas untuk hasil yang lebih presisi.")

    # Tombol generate
    if st.button("🎵 Generate Audio"):
        wave = generate_waveform(wave_type, freq, duration, fs)
        stereo = np.vstack((wave, wave)).T

        # Simpan ke buffer
        buf = io.BytesIO()
        sf.write(buf, stereo, fs, format='wav')
        buf.seek(0)

        # Preview & Download
        st.subheader("▶️ Preview Suara")
        st.audio(buf, format="audio/wav")

        st.download_button(
            label="💾 Download File .WAV",
            data=buf,
            file_name=f"{wave_type}_{freq}Hz_{duration}s.wav",
            mime="audio/wav"
        )

        # Pilihan durasi tampilan gelombang
        zoom_dur = st.slider("Durasi tampilan gelombang (detik)", 0.001, 0.05, 0.01, step=0.001)
        visualize_waveform(wave, fs, f"{wave_type} Wave - {freq} Hz", duration_display=zoom_dur)


# ==================================================
# FUNGSI VISUALISASI (versi jelas)
# ==================================================
def visualize_waveform(wave, fs, title="Waveform", duration_display=0.01):
    """Visualisasi gelombang waktu-domain (zoomable & jelas)."""
    import matplotlib.pyplot as plt

    samples_to_show = int(fs * duration_display)
    t = np.linspace(0, duration_display, samples_to_show, endpoint=False)
    wave_segment = wave[:samples_to_show]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, wave_segment, color='royalblue', linewidth=1.5)

    ax.set_title(f"{title}\n(Sample Rate: {fs} Hz, Durasi ditampilkan: {duration_display*1000:.1f} ms)",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("Waktu (detik)", fontsize=11)
    ax.set_ylabel("Amplitudo", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(-1.1, 1.1)

    st.pyplot(fig)
