import streamlit as st
from streamlit_mic_recorder import mic_recorder
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import io
from pydub import AudioSegment

# 페이지 설정 (넓은 화면 사용)
st.set_page_config(page_title="Linguistic Pronunciation Visualizer", layout="wide")

st.title("🗣️ AI-Mediated Linguistic Analysis")
st.subheader("VOT: Voice Onset Time")

# 1. 상단: 안내 및 녹음 컨트롤 (Center Aligned)
top_col1, top_col2, top_col3 = st.columns([1, 2, 1])

with top_col2:
    st.info("📱 **Mobile Tip**: 아이폰 사용자는 반드시 **Silent Mode** 해제")
    
    # 녹음 컨트롤러를 중앙에 배치
    audio = mic_recorder(
        start_prompt="⏺️ 녹음 시작 (Start Recording)",
        stop_prompt="⏹️ 녹음 중지 (Stop Recording)",
        key='recorder'
    )

st.divider()

# 2. 하단: 분석 섹션 (녹음이 있을 때만 가로 전체 공간 사용)
if audio:
    try:
        # 데이터 처리 및 변환
        audio_bytes = audio['bytes']
        audio_bio = io.BytesIO(audio_bytes)
        audio_segment = AudioSegment.from_file(audio_bio)
        
        wav_bio = io.BytesIO()
        audio_segment.export(wav_bio, format="wav")
        wav_data = wav_bio.getvalue()
        wav_bio.seek(0)

        # 오디오 로드 및 전체 시간 계산
        y, sr = librosa.load(wav_bio, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        # 분석 결과 헤더
        st.success(f"✅ 녹음 완료! (총 길이: {duration:.2f}초)")
        st.audio(wav_data, format="audio/wav")
        
        # 가로 공간을 가득 채우는 구간 선택 슬라이더
        st.write("🔍 **분석할 구간을 선택하세요 (Select Range for Detailed Analysis)**")
        start_time, end_time = st.slider(
            "시간 범위 (초)",
            0.0, float(duration), (0.0, float(duration)),
            step=0.01
        )

        # 선택된 구간 크롭
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        y_cropped = y[start_sample:end_sample]
        
        if len(y_cropped) > 0:
            # 🎨 시각화: 가로 폭을 넓게 설정 (figsize 조정)
            fig, ax = plt.subplots(2, 1, figsize=(15, 10), dpi=100)
            plt.subplots_adjust(hspace=0.4)
            
            # Waveform
            librosa.display.waveshow(y_cropped, sr=sr, ax=ax[0], color="#2c3e50")
            ax[0].set_title(f"Detailed Waveform ({start_time:.2f}s ~ {end_time:.2f}s)", fontsize=14)
            ax[0].set_xlabel("Time (s)")
            
            # Spectrogram
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y_cropped)), ref=np.max)
            img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax[1], 
                                         cmap='magma', x_coords=np.linspace(start_time, end_time, D.shape[1]))
            ax[1].set_title("Detailed Spectrogram: Frequency & Energy", fontsize=14)
            fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
            
            # Streamlit 화면에 출력 (use_container_width=True로 가로 꽉 채움)
            st.pyplot(fig, use_container_width=True)
            
            st.info(f"💡 현재 {end_time - start_time:.2f}초 구간을 집중 분석 중입니다. 가로 폭이 넓어져 VOT 분석이 훨씬 수월합니다.")
        else:
            st.warning("구간을 다시 선택해 주세요.")

    except Exception as e:
        st.error(f"오디오 처리 중 오류가 발생했습니다: {e}")

else:
    # 녹음 전 대기 화면 (상하 배치이므로 아래쪽에 크게 안내)
    st.write("---")
    st.markdown("""
    <div style="text-align: center; color: grey; padding: 50px;">
        <h3>상단의 버튼을 눌러 녹음을 시작하세요.</h3>
        <p>녹음이 완료되면 가로로 넓은 정밀 분석 도표가 이곳에 나타납니다.</p>
    </div>
    """, unsafe_allow_html=True)

# 🏛️ 이론적 배경 섹션
st.divider()
st.markdown("""
### 🏛️ Theoretical Background
* **Accessibility**: 가로로 확장된 시각화는 복잡한 음향 정보를 더 쉽게 파악하게 함으로써 **Scaffolding** 역할을 수행합니다.
* **Depth**: 특정 구간을 확대(Zoom-in)하는 기능은 학습자가 표면적인 이해를 넘어 데이터의 이면을 탐구하는 **Interpretive Inquiry**를 가능하게 합니다.
""")
