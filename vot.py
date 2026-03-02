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

st.title("🗣️ AI-Mediated Linguistic Analysis")
st.subheader("Target: Voice Onset Time (VOT)")

# 1. 상단: 안내 및 녹음 컨트롤
top_col1, top_col2, top_col3 = st.columns([1, 2, 1])

with top_col2:
    st.info("📱 **Mobile Tip**: 아이폰 사용자는 반드시 **Silent Mode** 해제")
    audio = mic_recorder(
        start_prompt="⏺️ 녹음 시작 (Start Recording)",
        stop_prompt="⏹️ 녹음 중지 (Stop Recording)",
        key='recorder'
    )

st.divider()

if audio:
    try:
        audio_bytes = audio['bytes']
        audio_bio = io.BytesIO(audio_bytes)
        audio_segment = AudioSegment.from_file(audio_bio)
        
        wav_bio = io.BytesIO()
        audio_segment.export(wav_bio, format="wav")
        wav_data = wav_bio.getvalue()
        wav_bio.seek(0)

        y, sr = librosa.load(wav_bio, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        max_sr_half = int(sr / 2)  # 나이퀴스트 주파수 (표현 가능한 최대 주파수)

        st.success(f"✅ 녹음 완료! (총 길이: {duration:.2f}초 / 샘플링 레이트: {sr}Hz)")
        st.audio(wav_data, format="audio/wav")
        
        # --- 분석 설정 레이아웃 (시간 및 주파수 조절) ---
        st.write("### ⚙️ 분석 설정 (Analysis Settings)")
        config_col1, config_col2 = st.columns(2)

        with config_col1:
            st.write("🔍 **시간 구간 선택 (Time Range)**")
            start_time, end_time = st.slider(
                "범위 (초)",
                0.0, float(duration), (0.0, float(duration)),
                step=0.01
            )

        with config_col2:
            st.write("📈 **주파수 범위 제한 (Frequency Range)**")
            # 인간의 목소리는 보통 8000Hz 이내에 주요 정보가 밀집해 있습니다.
            max_freq = st.slider(
                "최대 주파수 표시 (Hz)",
                2000, max_sr_half, min(8000, max_sr_half),
                step=500,
                help="표시할 주파수의 상한선을 조절합니다. 정보가 없는 고주파 영역을 제외하면 분석이 용이합니다."
            )

        # 선택된 구간 크롭
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        y_cropped = y[start_sample:end_sample]
        
        if len(y_cropped) > 0:
            fig, ax = plt.subplots(2, 1, figsize=(15, 10), dpi=100)
            plt.subplots_adjust(hspace=0.4)
            
            # 1. Waveform
            librosa.display.waveshow(y_cropped, sr=sr, ax=ax[0], color="#2c3e50")
            ax[0].set_title(f"Detailed Waveform ({start_time:.2f}s ~ {end_time:.2f}s)", fontsize=14)
            ax[0].set_xlabel("Time (s)")
            
            # 2. Spectrogram
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y_cropped)), ref=np.max)
            img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax[1], 
                                         cmap='magma', x_coords=np.linspace(start_time, end_time, D.shape[1]))
            
            # 세로축(주파수) 범위 제한 적용
            ax[1].set_ylim(0, max_freq) 
            ax[1].set_title(f"Detailed Spectrogram (Up to {max_freq}Hz)", fontsize=14)
            fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
            
            st.pyplot(fig, use_container_width=True)
            
            st.info(f"💡 주파수 축을 {max_freq}Hz로 제한하여 성도의 공명(Formants)과 마찰음의 에너지를 더 정밀하게 관찰할 수 있습니다.")
        else:
            st.warning("구간을 다시 선택해 주세요.")

    except Exception as e:
        st.error(f"오디오 처리 중 오류가 발생했습니다: {e}")

else:
    st.write("---")
    st.markdown("""
    <div style="text-align: center; color: grey; padding: 50px;">
        <h3>상단의 버튼을 눌러 녹음을 시작하세요.</h3>
        <p>녹음 후 시간과 주파수 범위를 조절하여 정밀하게 분석할 수 있습니다.</p>
    </div>
    """, unsafe_allow_html=True)

# 🏛️ 이론적 배경 섹션
st.divider()
st.markdown("""
### 🏛️ Theoretical Background
* **Cognitive Load Management**: 불필요한 고주파 정보를 제거함으로써 학습자가 핵심 데이터(VOT, Formants)에 집중할 수 있도록 돕습니다.
* **Conceptual Chunking**: 시간과 주파수를 학습자가 직접 조절하는 과정은 음성 데이터를 의미 있는 단위로 분절하여 이해하는 능력을 길러줍니다.
""")
