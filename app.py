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

    st.info("💡 Kamu bisa ketik nilai manual di kolom atas untuk hasil yang lebih presisi.")

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
        st.markdown("🕒 **Durasi tampilan (detik)** menentukan seberapa panjang potongan sinyal yang akan ditampilkan.")
        zoom_dur = st.slider(
            "Durasi tampilan (detik)",
            0.001, 0.1, 0.02, step=0.001,
            key="zoom_gen"
        )

        visualize_waveform(wave, fs, f"{wave_type} Wave - {freq} Hz", duration_display=zoom_dur)
        visualize_spectrum(wave, fs, f"Spektrum {wave_type} {freq} Hz")
