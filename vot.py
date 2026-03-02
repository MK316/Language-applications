import streamlit as st
from streamlit_mic_recorder import mic_recorder
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import io
from pydub import AudioSegment  # 포맷 변환을 위해 추가

st.set_page_config(page_title="Linguistic Visualizer", layout="wide")

st.title("🗣️ AI-Mediated Linguistic Analysis")
st.write("Record your voice to analyze VOT and Pitch patterns. (iOS/Android Compatible)")

col1, col2 = st.columns([1, 2])

with col1:
    # 아이폰 호환성을 위해 mic_recorder를 사용
    audio = mic_recorder(
        start_prompt="⏺️ Start Recording",
        stop_prompt="⏹️ Stop Recording",
        key='recorder'
    )

if audio:
    try:
        # 1. 아이폰의 특수 포맷 데이터를 읽기 위해 BytesIO 생성
        audio_bytes = audio['bytes']
        audio_bio = io.BytesIO(audio_bytes)
        
        # 2. pydub를 사용하여 데이터를 범용적인 wav로 변환 (아이폰 .webm 대응)
        # 이 과정에서 오디오 포맷을 자동으로 인식하여 처리합니다.
        audio_segment = AudioSegment.from_file(audio_bio)
        
        # 분석을 위해 다시 BytesIO에 wav로 저장
        wav_bio = io.BytesIO()
        audio_segment.export(wav_bio, format="wav")
        wav_bio.seek(0)

        with col2:
            st.success("Analysis Complete!")
            st.audio(audio_bytes) # 원본 오디오 재생
            
            # 3. Librosa로 로드 (wav 포맷이므로 이제 안전하게 읽힙니다)
            y, sr = librosa.load(wav_bio, sr=None)
            
            # 시각화 코드 (이전과 동일)
            fig, ax = plt.subplots(2, 1, figsize=(10, 8))
            librosa.display.waveshow(y, sr=sr, ax=ax[0], color="teal")
            ax[0].set_title("Waveform (Check for Aspiration & VOT)")
            
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax[1])
            ax[1].set_title("Spectrogram")
            
            plt.tight_layout()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing audio: {e}")
        st.info("Tip: If you're on iPhone, ensure your silent mode is off and try again.")
