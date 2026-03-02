import streamlit as st
from streamlit_mic_recorder import mic_recorder
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import io

st.set_page_config(page_title="English Pronunciation Visualizer", layout="wide")

st.title("🗣️ AI-Mediated Linguistic Analysis")
st.subheader("Analyze your Voice Onset Time (VOT) and Pitch Patterns")

st.write("""
This tool helps you visualize your pronunciation patterns. 
Record your voice (e.g., saying words like 'Pat' vs 'Bat') to see the acoustic differences.
""")

# 레이아웃 분할
col1, col2 = st.columns([1, 2])

with col1:
    st.info("📱 Mobile Users: Tap 'Start Recording' and allow microphone access.")
    
    # 오디오 레코더 설정 (아이폰/안드로이드 호환을 위해 고안됨)
    audio = mic_recorder(
        start_prompt="⏺️ Start Recording",
        stop_prompt="⏹️ Stop Recording",
        key='recorder'
    )

if audio:
    # 녹음된 데이터를 바이너리 형태로 읽기
    audio_bio = io.BytesIO(audio['bytes'])
    
    with col2:
        st.success("Audio captured successfully!")
        st.audio(audio_bio)
        
        # Librosa를 이용한 음성 분석
        try:
            # 오디오 로드 (Librosa는 다양한 포맷을 자동으로 처리)
            y, sr = librosa.load(audio_bio, sr=None)
            
            # 1. 파형(Waveform) 시각화
            fig, ax = plt.subplots(2, 1, figsize=(10, 8))
            
            librosa.display.waveshow(y, sr=sr, ax=ax[0], color="blue")
            ax[0].set_title("Waveform (Check for Aspiration & VOT)")
            ax[0].set_xlabel("Time (s)")
            
            # 2. 스펙트로그램(Spectrogram) 시각화
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax[1])
            ax[1].set_title("Spectrogram (Voice Energy Patterns)")
            fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # 분석 팁 제공
            st.markdown("""
            **🔍 How to analyze:**
            - **VOT (Voice Onset Time):** Look for the gap between the release of the consonant (burst) and the start of the periodic vocal fold vibration.
            - **Intensity:** Higher peaks represent stronger aspiration or stress.
            """)
            
        except Exception as e:
            st.error(f"Error processing audio: {e}")
else:
    with col2:
        st.warning("Awaiting recording... Please record your voice in the left panel.")

# 🏛️ Theoretical Connection (Scaffolding)
with st.expander("Theoretical Background"):
    st.write("""
    This app serves as a **scaffolding tool** for linguistic analysis. 
    By visualizing abstract acoustic data, students can bridge the gap between 
    **Difficulty** (perceiving subtle sound differences) and **Depth** (understanding phonetic features).
    """)
