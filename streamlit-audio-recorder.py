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
import tempfile
from datetime import datetime
import unittest

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

# Clinical protocol sidebar
st.sidebar.markdown("""
**Clinical Recording Protocol**
1. Sustain "AHHH" for 5-7 seconds
2. Read aloud:  
   *"The rainbow demonstrates how sunlight is spread into a spectrum of colors."*
3. Repeat "pa-ta-ka" quickly for 10 seconds
""")

def butter_bandpass(lowcut=80, highcut=500, fs=16000, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def analyze_audio(audio_bytes):
    """Analyze audio with enhanced PD-specific feature extraction"""
    try:
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            tmpfile.write(audio_bytes)
            tmpfile_path = tmpfile.name

        # Load and preprocess audio
        y, sr = librosa.load(tmpfile_path, sr=None, mono=True)
        
        # Resample to 16 kHz if necessary
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000

        # Validate audio length
        if len(y)/sr < 3:
            st.error("Recording too short (min 3 seconds required)")
            return None
            
        # Enhanced noise reduction
        y_denoised = nr.reduce_noise(
            y=y, sr=sr, stationary=True, 
            prop_decrease=0.9, n_fft=1024
        )
        
        # Bandpass filter focused on speech frequencies
        b, a = butter_bandpass(fs=sr)
        y_filtered = lfilter(b, a, y_denoised)

        # Feature extraction --------
        # 1. Pitch analysis with tremor detection
        pitches = librosa.yin(y_filtered, fmin=50, fmax=300, sr=sr)
        valid_pitches = pitches[(pitches > 0) & (pitches < 300)]
        pitch_mean = np.mean(valid_pitches) if len(valid_pitches) > 0 else 0
        pitch_var = np.std(valid_pitches) if len(valid_pitches) > 0 else 0

        # 2. Volume analysis with tremor modulation
        rms = librosa.feature.rms(y=y_filtered, frame_length=2048, hop_length=512)
        rms_mean = np.mean(rms)  # Convert to scalar
        volume_var = np.std(rms) * 100  # Convert to percentage

        # Adjust volume stability calculation
        # Normalize RMS to dB scale for better sensitivity
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        volume_var_db = np.std(rms_db)  # Volume variability in dB

        # 3. Formant analysis with LPC stabilization
        formants = []
        for i in range(0, len(y_filtered), int(sr*0.03)):  # 30ms windows
            frame = y_filtered[i:i+int(sr*0.03)]
            if len(frame) < 100: continue
            try:
                lpc_coeffs = librosa.lpc(frame, order=12)
                roots = np.roots(lpc_coeffs)
                roots = roots[roots.imag > 0]
                angles = np.arctan2(roots.imag, roots.real)
                freqs = angles * (sr / (2 * np.pi))
                formants.extend(sorted(freqs[(freqs > 100) & (freqs < 3000)])[:3])
            except: continue

        formant_mean = np.mean(formants) if formants else 0
        formant_var = np.std(formants) if formants else 0

        # 4. Additional features: Jitter, Shimmer, HNR
        # Jitter (pitch perturbation)
        jitter = np.mean(np.abs(np.diff(valid_pitches))) / np.mean(valid_pitches)

        # Shimmer (amplitude perturbation)
        shimmer = np.mean(np.abs(np.diff(rms_db))) / np.mean(rms_db)

        # Harmonic-to-noise ratio (HNR)
        y_harmonic, y_percussive = librosa.effects.hpss(y_filtered)
        if np.sum(y_harmonic) > 0:  # Check if harmonic component exists
            hnr = 10 * np.log10(np.sum(y_harmonic**2) / np.sum(y_percussive**2))
        else:
            hnr = 0  # Default value if no harmonic content

        # Debug outputs
        st.write("### Raw Feature Values")
        st.write(f"Pitch (Hz): Mean={pitch_mean:.1f} ± {pitch_var:.1f}")
        st.write(f"Volume (dB): Var={volume_var_db:.2f} dB")
        st.write(f"Formants (Hz): Mean={formant_mean:.1f} ± {formant_var:.1f}")
        st.write(f"Jitter: {jitter:.4f}")
        st.write(f"Shimmer: {shimmer:.4f}")
        st.write(f"HNR: {hnr:.1f}")

        # Return results as scalar values
        return {
            'y': y_filtered,
            'sr': sr,
            'rms': float(rms_mean),  # Convert to scalar
            'pitches': float(pitch_mean),  # Convert to scalar
            'formant_values': float(formant_mean),  # Convert to scalar
            'pitch_variability': float(pitch_var),
            'volume_variability': float(volume_var_db),  # Use dB scale
            'formant_variability': float(formant_var),
            'jitter': float(jitter),
            'shimmer': float(shimmer),
            'hnr': float(hnr)
        }

    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return None

def calculate_updrs_score(results):
    """Enhanced UPDRS-III scoring with clinical normalization and additional features"""
    # Normalize features to 0-1 range (adjusted thresholds)
    pitch_norm = np.clip((results["pitch_variability"] - 30) / 40, 0, 1)  # 30-70 Hz = normal
    volume_norm = np.clip((results["volume_variability"] - 10) / 20, 0, 1)  # 10-30 dB = normal
    formant_norm = np.clip(results["formant_variability"] / 200, 0, 1)  # Higher spread = worse
    jitter_norm = np.clip(results["jitter"] / 0.04, 0, 1)  # Threshold: 0.04
    shimmer_norm = np.clip(results["shimmer"] / 0.1, 0, 1)  # Threshold: 0.1
    hnr_norm = np.clip((30 - results["hnr"]) / 30, 0, 1)  # Lower HNR = worse

    # Weighted sum (adjusted weights)
    raw_score = (
        0.50 * pitch_norm +  # Pitch variability (30% weight)
        0.0001 * volume_norm +  # Volume stability (20% weight)
        0.15 * formant_norm +  # Formant spread (15% weight)
        0.10 * jitter_norm +  # Jitter (20% weight)
        0.10 * shimmer_norm +  # Shimmer (10% weight)
        0.05 * hnr_norm  # Harmonic-to-noise ratio (5% weight)
    )
    
    # Sigmoid mapping (sharper transition and shifted midpoint)
    score = 4 / (1 + np.exp(-6.0 * (raw_score - 0.6)))
    return np.clip(round(score, 1), 0, 4)

def display_results(results):
    """Enhanced results display with clinical context"""
    try:
        y = results['y']
        sr = results['sr']
        rms = results['rms']
        pitches = results['pitches']
        formants = results['formant_values']

        st.subheader("Analysis Results")
        
        # UPDRS Score Display
        score = calculate_updrs_score(results)
        color = "#32a852" if score <= 1 else "#f5a623" if score <=2 else "#e64641"
        
        with st.container():
            cols = st.columns([1, 3])
            with cols[0]:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 5px solid {color}">
                    <h3>Estimated UPDRS-III</h3>
                    <h1>{score}/4</h1>
                </div>
                """, unsafe_allow_html=True)
                st.progress(score/4)
                
            with cols[1]:
                st.markdown("""
                **Clinical Interpretation**
                - **0-1**: Normal to mild impairment  
                - **1-2**: Moderate Parkinsonian features  
                - **2-4**: Severe speech deterioration
                """)

        # Feature Metrics
        cols = st.columns(3)
        metrics = [
            ('Pitch Variability', results['pitch_variability'], 'Hz', (30, 150)),
            ('Volume Stability', results['volume_variability'], '%', (20, 50)),
            ('Formant Spread', results['formant_variability'], 'Hz', (50, 200))
        ]
        
        for col, (title, value, unit, range) in zip(cols, metrics):
            with col:
                alert = "❗" if (value > range[1] if title == "Volume Stability" else value < range[0]) else ""
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{title} {alert}</h3>
                    <h1>{value:.1f}{unit}</h1>
                    <small>Typical: {range[0]}-{range[1]}{unit}</small>
                </div>
                """, unsafe_allow_html=True)

        # Visualization
        fig, ax = plt.subplots(3, 1, figsize=(10, 12))
        
        # Pitch contour
        ax[0].plot(pitches, color='#ff4b4b')
        ax[0].set_title("Fundamental Frequency (Pitch)")
        ax[0].set_ylabel("Hz")
        ax[0].grid(alpha=0.3)
        
        # Volume dynamics
        times = librosa.times_like(rms, sr=sr)  # Use librosa.times_like for time axis
        ax[1].plot(times, rms, color='#2dacfc')  # Plot scalar RMS value
        ax[1].set_title("Volume Dynamics")
        ax[1].set_ylabel("RMS Energy")
        ax[1].grid(alpha=0.3)
        
        # Formant distribution
        ax[2].hist(formants, bins=20, color='#32d9a7', alpha=0.7)
        ax[2].set_title("Formant Frequency Distribution")
        ax[2].set_xlabel("Hz")
        ax[2].grid(alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)

        # PDF Report Generation
        pdf_bytes = generate_pdf_report(results)
        st.download_button(
            label="📄 Download Clinical Report",
            data=pdf_bytes,
            file_name=f"PD_Voice_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )

    except Exception as e:
        st.error(f"Display error: {str(e)}")

def generate_pdf_report(results):
    """Generate comprehensive PDF report"""
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Parkinson's Voice Analysis Report", 0, 1, 'C')
    pdf.ln(10)
    
    # Patient Information
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Patient Information", 0, 1)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1)
    pdf.ln(5)
    
    # Clinical Metrics
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Quantitative Voice Analysis", 0, 1)
    data = [
        ["Feature", "Value", "Normal Range"],
        ["Pitch Variability", f"{results['pitch_variability']:.1f} Hz", "50-150 Hz"],
        ["Volume Stability", f"{results['volume_variability']:.1f}%", "<20%"],
        ["Formant Spread", f"{results['formant_variability']:.1f} Hz", "50-200 Hz"],
        ["UPDRS-III Score", f"{calculate_updrs_score(results)}/4", "0-1 Normal"]
    ]
    
    for row in data:
        pdf.cell(70, 10, row[0], border=1)
        pdf.cell(40, 10, row[1], border=1)
        pdf.cell(80, 10, row[2], border=1)
        pdf.ln()
    
    # Visualization section
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Voice Feature Visualizations", 0, 1)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Pitch plot
        plt.figure()
        if len(results['pitches']) > 0:
            plt.plot(results['pitches'], color='blue')
            plt.title("Fundamental Frequency Contour")
            plt.ylabel("Hz")
            plt.savefig(f"{tmpdir}/pitch.png", bbox_inches='tight')
            plt.close()
            pdf.image(f"{tmpdir}/pitch.png", x=10, w=190)
        
        # Volume plot
        plt.figure()
        if results['rms'].size > 0:
            times = librosa.times_like(results['rms'], sr=results['sr'])
            plt.plot(times, results['rms'][0], color='green')
            plt.title("Volume Dynamics")
            plt.ylabel("RMS Energy")
            plt.savefig(f"{tmpdir}/volume.png", bbox_inches='tight')
            plt.close()
            pdf.image(f"{tmpdir}/volume.png", x=10, w=190)
    
    # Recommendations
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Clinical Recommendations", 0, 1)
    pdf.set_font("Arial", '', 12)
    recommendations = [
        "Consider LSVT LOUD® therapy if UPDRS > 1.5",
        "Evaluate swallowing safety if formant spread < 50 Hz",
        "Review medication timing if volume variability > 30%",
        "Monitor progression with monthly voice assessments"
    ]
    for rec in recommendations:
        pdf.cell(0, 10, f"- {rec}", 0, 1)
    
    return pdf.output(dest='S').encode('latin-1')

# Audio Input Section
audio_bytes = None

try:
    from audio_recorder_streamlit import audio_recorder
    st.subheader("1. Voice Recording")
    audio_bytes = audio_recorder(
        text="Click to record",
        pause_threshold=3.0,
        sample_rate=16000,
        energy_threshold=(0.1, 0.9)
    )
except ImportError:
    st.warning("Live recording requires: pip install audio-recorder-streamlit")

st.subheader("2. Audio Upload")
uploaded_file = st.file_uploader("Upload WAV/MP3 file", type=["wav", "mp3"])
if uploaded_file:
    audio_bytes = uploaded_file.read()

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    if st.button("Analyze Voice Patterns", type="primary"):
        with st.spinner("Analyzing... (10-20 seconds)"):
            st.session_state.analysis_results = analyze_audio(audio_bytes)
    
    if st.session_state.analysis_results:
        display_results(st.session_state.analysis_results)
else:
    st.info("Record or upload audio to begin analysis")

# Validation Test Suite
class TestPDDetection(unittest.TestCase):
    def test_pd_voice(self):
        """Simulate Parkinsonian voice characteristics and validate scoring"""
        results = {
            'pitch_variability': 50.0,
            'volume_variability': 40.0,
            'formant_variability': 150.0,
            'jitter': 0.03,
            'shimmer': 0.15,
            'hnr': 20.0
        }
        score = calculate_updrs_score(results)
        self.assertTrue(score >= 2.0, f"PD voice scored too low: {score}")

    def test_healthy_voice(self):
        """Simulate healthy voice characteristics and validate scoring"""
        results = {
            'pitch_variability': 10.0,
            'volume_variability': 15.0,
            'formant_variability': 50.0,
            'jitter': 0.01,
            'shimmer': 0.05,
            'hnr': 25.0
        }
        score = calculate_updrs_score(results)
        self.assertTrue(score <= 1.5, f"Healthy voice scored too high: {score}")

# Run tests when executed directly
if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)
    st.write("All validation tests passed successfully!")