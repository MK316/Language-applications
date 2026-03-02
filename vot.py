import streamlit as st
from streamlit_mic_recorder import mic_recorder
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import io
from pydub import AudioSegment

# 페이지 설정
st.set_page_config(page_title="Linguistic Pronunciation Visualizer", layout="wide")

# 스타일 설정
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stAlert {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🗣️ AI-Mediated Linguistic Analysis")
st.subheader("Analyze Voice Onset Time (VOT) & Acoustic Patterns")

st.markdown("""
이 도구는 인문학적 난해함(**Difficulty**)을 분석적 깊이(**Depth**)로 전환하여 학습자의 능동적 참여(**Engagement**)를 돕는 도구입니다. 
단어를 녹음하고 파형을 분석하며 영어 음성학적 특징을 탐구해 보세요.
""")

# 레이아웃 구성
col1, col2 = st.columns([1, 2])

with col1:
    st.info("📱 **Mobile Tip**: 아이폰 사용자는 기기 옆면의 **무음 스위치(Silent Mode)**를 해제해야 소리가 들립니다.")
    
    # 오디오 레코더: 아이폰/안드로이드 호환성 확보
    audio = mic_recorder(
        start_prompt="⏺️ 녹음 시작 (Start)",
        stop_prompt="⏹️ 녹음 중지 (Stop)",
        key='recorder'
    )

if audio:
    try:
        # 1. 녹음 데이터 처리
        audio_bytes = audio['bytes']
        audio_bio = io.BytesIO(audio_bytes)
        
        # 2. 아이폰 .webm/.ogg 포맷을 표준 .wav로 변환 (호환성 핵심)
        audio_segment = AudioSegment.from_file(audio_bio)
        
        # 분석 및 재생용 WAV 바이너리 생성
        wav_bio = io.BytesIO()
        audio_segment.export(wav_bio, format="wav")
        wav_data = wav_bio.getvalue()
        wav_bio.seek(0)

        with col2:
            st.success("녹음이 완료되었습니다. 아래에서 분석 결과를 확인하세요!")
            
            # [아이폰 재생 해결] 변환된 표준 WAV 데이터를 재생
            st.audio(wav_data, format="audio/wav")
            
            # 3. Librosa를 이용한 음성 분석 데이터 로드
            y, sr = librosa.load(wav_bio, sr=None)
            
            # 4. 시각화 (Waveform & Spectrogram)
            fig, ax = plt.subplots(2, 1, figsize=(10, 8), dpi=100)
            plt.subplots_adjust(hspace=0.4)
            
            # Waveform
            librosa.display.waveshow(y, sr=sr, ax=ax[0], color="#2c3e50")
            ax[0].set_title("Waveform: 시각적 비계(Scaffolding)를 통한 VOT 분석", fontsize=12)
            ax[0].set_xlabel("Time (seconds)")
            ax[0].set_ylabel("Amplitude")
            
            # Spectrogram
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax[1], cmap='magma')
            ax[1].set_title("Spectrogram: 다중 양식(Multimodal) 에너지 분포 확인", fontsize=12)
            fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
            
            st.pyplot(fig)
            
            # 5. 분석 가이드 (Pedagogical Implication)
            with st.expander("📝 분석 가이드: 어떻게 관찰하나요?"):
                st.write("""
                - **VOT 분석**: 파형의 시작 부분에서 자음의 파열(Burst)과 모음의 진동 시작 사이의 간격을 확인하세요. 
                - **무성음 vs 유성음**: 'Pat'과 'Bat'을 녹음하여 파형의 모양과 기성(Aspiration)의 차이를 비교해 보세요.
                - **깊이 있는 학습**: 시각화된 데이터는 추상적인 언어 지식을 구체적인 경험으로 전환시켜 줍니다.
                """)

    except Exception as e:
        st.error(f"오디오 처리 중 오류가 발생했습니다: {e}")
        st.info("FFmpeg가 서버에 설치되어 있는지 확인해 주세요.")

else:
    with col2:
        st.write("---")
        st.write("왼쪽의 버튼을 눌러 녹음을 시작하면 이곳에 분석 도표가 나타납니다.")
        st.image("https://via.placeholder.com/800x400.png?text=Waiting+for+Voice+Input...", use_column_width=True)

# 이론적 배경 섹션
st.divider()
st.markdown("""
### 🏛️ Theoretical Background
이 애플리케이션은 **Vygotsky의 ZPD** 내에서 학습자가 스스로 언어적 특징을 발견할 수 있도록 **Scaffolding**을 제공합니다. 
복잡한 음성 신호를 **Multimodal Representation**으로 시각화하여 학습의 깊이를 더합니다.
""")
