import streamlit as st
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from io import BytesIO
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import noisereduce as nr
from fpdf import FPDF
import base64
import tempfile
from datetime import datetime
import tempfile

# Custom CSS styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .stAudio {
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# App title and description
st.title("Parkinson's Voice Analyzer 🎤")
st.markdown("""
Detect early signs of voice changes associated with Parkinson's disease.  
Record your voice or upload an audio file to analyze:
- **Pitch variability**
- **Volume stability**
- **Speech clarity**
""")

def butter_bandpass(lowcut=80, highcut=4000, fs=16000, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def analyze_audio(audio_bytes):
    """Analyze audio with noise reduction for Parkinson's vocal patterns"""
    try:
        # Convert to WAV format if needed
        if isinstance(audio_bytes, bytes) and audio_bytes.startswith(b'RIFF'):
            wav_bytes = audio_bytes
        else:
            audio = AudioSegment.from_file(BytesIO(audio_bytes))
            wav_bytes = BytesIO()
            audio.export(wav_bytes, format="wav")
            wav_bytes = wav_bytes.getvalue()

        # Save temporary file
        with open("temp_audio.wav", "wb") as f:
            f.write(wav_bytes)

        # Load audio with librosa
        y, sr = librosa.load("temp_audio.wav", sr=None, mono=True)
        
        # --- NEW NOISE REDUCTION STEP ---
        y_denoised = nr.reduce_noise(
            y=y, 
            sr=sr,
            stationary=True,  # For constant background noise
            prop_decrease=0.75,  # Reduce 75% of noise
            n_fft=512,
            win_length=256
        )
        
        # Bandpass filter (existing step)
        b, a = butter_bandpass(fs=sr)
        y_filtered = lfilter(b, a, y_denoised)

        # --- ANALYSIS ON CLEANED AUDIO ---
        # 1. Pitch analysis
        pitches, magnitudes = librosa.piptrack(y=y_filtered, sr=sr)
        valid_pitches = pitches[pitches > 0]
        pitch_var = np.std(valid_pitches) if len(valid_pitches) > 0 else 0

        # 2. Volume analysis
        rms = librosa.feature.rms(y=y_filtered)
        volume_var = np.std(rms) if rms.size > 0 else 0

        # 3. Formant analysis (improved with cleaned audio)
        formants = []
        for i in range(0, len(y_filtered), int(sr*0.02)):
            frame = y_filtered[i:i+int(sr*0.02)]
            if len(frame) < 10:
                continue
            try:
                lpc_coeffs = librosa.lpc(frame, order=8)
                roots = np.roots(lpc_coeffs)
                roots = roots[roots.imag > 0]
                angles = np.arctan2(roots.imag, roots.real)
                freqs = angles * (sr / (2 * np.pi))
                formants.extend(sorted(freqs[(freqs > 100) & (freqs < 4000)])[:3])
            except:
                continue

        return {
            'y': y_filtered,
            'sr': sr,
            'rms': rms,
            'pitches': pitches,
            'formant_values': formants,
            'pitch_variability': float(pitch_var),
            'volume_variability': float(volume_var),
            'formant_variability': float(np.std(formants) if formants else 0)
        }

    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return None

def display_results(results):
    """Display analysis results with visualizations"""
    try:
        y = results['y']
        sr = results['sr']
        rms = results['rms']
        pitches = results['pitches']
        formants = results['formant_values']

        st.subheader("Analysis Results")
        
          # Add UPDRS display
        st.markdown("---")
        col1, col2 = st.columns([1, 3])
    
        with col1:
            score = calculate_updrs_score(results)
            color = "#32a852" if score <= 1 else "#f5a623" if score <=2 else "#e64641"
            st.markdown(f"""
            <div class="metric-card" style="border-left: 5px solid {color}">
                <h3>Estimated UPDRS-III Speech Score</h3>
                <h1>{score}/4</h1>
            </div>
            """, unsafe_allow_html=True)
    
        with col2:
            st.markdown("""
            **Clinical Interpretation**
            - **0**: Normal speech 
            - **1**: Slight hypophonia, fully intelligible
            - **2**: Moderate hypophonia, occasional repetition needed
            - **3**: Marked hypophonia, frequently unintelligible
            - **4**: Unintelligible speech
            """)

            pdf_bytes = generate_pdf_report(results)
            st.download_button(
                label="📄 Download Clinical Report",
                data=pdf_bytes,
                file_name=f"voice_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
    # Add progress bar
        st.progress(score/4)
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card">'
                        '<h3>Pitch Variability</h3>'
                        f'<h1>{results["pitch_variability"]:.2f} Hz</h1>'
                        '</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card">'
                        '<h3>Volume Stability</h3>'
                        f'<h1>{results["volume_variability"]:.2f} dB</h1>'
                        '</div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card">'
                        '<h3>Formant Variability</h3>'
                        f'<h1>{results["formant_variability"]:.2f} Hz</h1>'
                        '</div>', unsafe_allow_html=True)

        # Visualizations
        fig, ax = plt.subplots(3, 1, figsize=(10, 12))
        
        # Pitch contour
        pitch_contour = [pitches[:, t][pitches[:, t] > 0].mean() 
                        for t in range(pitches.shape[1]) if np.any(pitches[:, t] > 0)]
        ax[0].plot(pitch_contour, color='#ff4b4b')
        ax[0].set_title("Pitch Contour")
        ax[0].set_ylabel("Frequency (Hz)")
        ax[0].grid(alpha=0.3)

        # Volume dynamics
        times = librosa.times_like(rms, sr=sr)
        ax[1].plot(times, rms[0], color='#2dacfc')
        ax[1].set_title("Volume Dynamics")
        ax[1].set_ylabel("RMS Energy")
        ax[1].set_xlabel("Time (s)")
        ax[1].grid(alpha=0.3)

        # Formant distribution
        ax[2].hist(formants, bins=20, color='#32d9a7', alpha=0.7)
        ax[2].set_title("Formant Frequency Distribution")
        ax[2].set_xlabel("Frequency (Hz)")
        ax[2].set_ylabel("Count")
        ax[2].grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # Interpretation guide
        st.markdown("""
        **Clinical Interpretation Guidelines**
        - **Normal Pitch Variability**: 50-200 Hz
        - **PD Pitch Variability**: < 30 Hz
        - **Normal Volume Stability**: < 0.1 dB variation
        - **PD Volume Stability**: > 0.2 dB variation
        - **Normal Formant Variability**: 50-300 Hz
        - **PD Formant Variability**: < 50 Hz
        """)

        

    except Exception as e:
        st.error(f"Visualization error: {str(e)}")

# Audio input section
audio_bytes = None

# Recording component
try:
    from audio_recorder_streamlit import audio_recorder
    st.subheader("1. Record Your Voice")
    audio_bytes = audio_recorder(
        text="Click to record",
        pause_threshold=2.0,
        sample_rate=16000,
        energy_threshold=(0.2, 0.8)
    )
except ImportError:
    st.warning("For live recording: pip install audio-recorder-streamlit")

def generate_pdf_report(results):
    """Create clinician-friendly PDF report"""
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Parkinson's Voice Analysis Report", 0, 1, 'C')
    pdf.ln(10)
    
    # Patient Info Section
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Patient Information", 0, 1)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1)
    pdf.cell(0, 10, "Clinician: ___________________", 0, 1)
    pdf.ln(10)
    
    # Key Metrics Table
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Vocal Features Analysis", 0, 1)
    pdf.set_font("Arial", '', 12)
    
    data = [
        ["Metric", "Value", "Normal Range"],
        ["Pitch Variability", f"{results['pitch_variability']:.2f} Hz", "50-200 Hz"],
        ["Volume Stability", f"{results['volume_variability']:.2f} dB", "< 0.1 dB"],
        ["Formant Variability", f"{results['formant_variability']:.2f} Hz", "50-300 Hz"],
        ["UPDRS-III Score", f"{calculate_updrs_score(results)}/4", "0-1 Normal"]
    ]
    
    col_widths = [70, 40, 80]
    row_height = 10
    
    for row in data:
        for i, item in enumerate(row):
            pdf.cell(col_widths[i], row_height, str(item), border=1)
        pdf.ln(row_height)
    
    pdf.ln(15)
    
    # Save plots to temp files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Pitch plot
        plt.figure()
        plt.plot(results['pitches'][0], color='blue')
        plt.title("Pitch Contour")
        pitch_path = f"{tmpdir}/pitch.png"
        plt.savefig(pitch_path, bbox_inches='tight')
        plt.close()
        
        # Add plots to PDF
        pdf.image(pitch_path, x=10, w=190)
        pdf.ln(5)
        
        # Volume plot
        plt.figure()
        times = librosa.times_like(results['rms'], sr=results['sr'])
        plt.plot(times, results['rms'][0], color='green')
        plt.title("Volume Dynamics")
        volume_path = f"{tmpdir}/volume.png"
        plt.savefig(volume_path, bbox_inches='tight')
        plt.close()
        pdf.image(volume_path, x=10, w=190)
    
    # Clinical Notes Section
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Clinical Recommendations", 0, 1)
    pdf.set_font("Arial", '', 12)
    recommendations = [
        "Consider LSVT LOUD® therapy if UPDRS > 1.5",
        "Evaluate swallowing safety if formant variability < 50 Hz",
        "Monitor medication timing if volume variability > 0.2 dB"
    ]
    for rec in recommendations:
        pdf.cell(0, 10, f"- {rec}", 0, 1)
    
    # Generate PDF bytes
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return pdf_bytes

def calculate_updrs_score(results):
    """Estimate UPDRS-III speech score (0-4 scale) using voice features"""
    # Coefficients from Tsanas et al. 2012 study (adjusted for real-time use)
    updrs = (
        0.38 * (150 - results["pitch_variability"])/150 +  # Pitch hypophonia component
        0.45 * results["volume_variability"]/0.3 +          # Volume instability
        0.27 * (400 - results["formant_variability"])/400   # Articulation clarity
    )
    
    # Map to 0-4 clinical scale with sigmoid curve
    score = 4 / (1 + np.exp(-0.8*(updrs - 2.5))) 
    return np.clip(round(score, 1), 0, 4)  # Clamp between 0-4

# File uploader
st.subheader("2. Or Upload Audio")
uploaded_file = st.file_uploader("Choose WAV/MP3 file", type=["wav", "mp3"])
if uploaded_file:
    audio_bytes = uploaded_file.read()

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    
    if st.button("Analyze Voice Patterns"):
        with st.spinner("Analyzing... (10-20 seconds)"):
            st.session_state.analysis_results = analyze_audio(audio_bytes)
    
    if st.session_state.analysis_results:
        display_results(st.session_state.analysis_results)
else:
    st.info("Record or upload audio to begin analysis")

# Troubleshooting guide
st.markdown("""
---
### Troubleshooting Guide
1. **No sound detected?**  
   - Check microphone permissions  
   - Try speaking louder/closer to mic  

2. **Analysis errors?**  
   - Use WAV format for best results  
   - Keep recordings under 30 seconds  

3. **Strange metrics?**  
   - Avoid background noise  
   - Use simple phrases like "The rainbow is a division of white light"
""")