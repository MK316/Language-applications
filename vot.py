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
st.caption("Say 'pie, bye, spy'")

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
        # 데이터 처리 및 WAV 변환
        audio_bytes = audio['bytes']
        audio_bio = io.BytesIO(audio_bytes)
        audio_segment = AudioSegment.from_file(audio_bio)
        
        wav_bio = io.BytesIO()
        audio_segment.export(wav_bio, format="wav")
        wav_data = wav_bio.getvalue()
        wav_bio.seek(0)

        # 오디오 로드 및 기본 정보 추출
        y, sr = librosa.load(wav_bio, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        max_sr_half = int(sr / 2)

        st.success(f"✅ 녹음 완료! (총 길이: {duration:.2f}초 / 샘플링 레이트: {sr}Hz)")
        st.audio(wav_data, format="audio/wav")
        
        # --- 분석 설정 레이아웃 ---
        st.write("### ⚙️ 분석 설정 (Analysis Settings)")
        config_col1, config_col2 = st.columns(2)

        with config_col1:
            st.write("🔍 **시간 구간 선택 (Time Range)**")
            start_time, end_time = st.slider(
                "범위 (초)",
                0.0, float(duration), (0.0, float(duration)),
                step=0.01,
                key="time_slider"
            )

        with config_col2:
            st.write("📈 **주파수 범위 제한 (Frequency Range)**")
            max_freq = st.slider(
                "최대 주파수 표시 (Hz)",
                2000, max_sr_half, min(8000, max_sr_half),
                step=500,
                key="freq_slider"
            )

        # 분석 시작 버튼 (사용자 개입 강조)
        if st.button("🚀 분석 시작 (Start Analysis)", type="primary", use_container_width=True):
            # 선택된 구간 크롭
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            y_cropped = y[start_sample:end_sample]
            
            if len(y_cropped) > 0:
                fig, ax = plt.subplots(2, 1, figsize=(15, 10), dpi=100)
                plt.subplots_adjust(hspace=0.4)
                
                # 1. Waveform (offset 설정을 통해 시간축 정렬 문제 해결)
                librosa.display.waveshow(y_cropped, sr=sr, ax=ax[0], color="#2c3e50", offset=start_time)
                ax[0].set_title(f"Detailed Waveform ({start_time:.2f}s ~ {end_time:.2f}s)", fontsize=14)
                ax[0].set_xlabel("Time (s)")
                
                # 2. Spectrogram (x_coords 설정을 통해 시간축 일치)
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y_cropped)), ref=np.max)
                img = librosa.display.specshow(
                    D, sr=sr, x_axis='time', y_axis='hz', ax=ax[1], 
                    cmap='magma', 
                    x_coords=np.linspace(start_time, end_time, D.shape[1])
                )
                
                ax[1].set_ylim(0, max_freq) 
                ax[1].set_title(f"Detailed Spectrogram (Up to {max_freq}Hz)", fontsize=14)
                fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
                
                st.pyplot(fig, use_container_width=True)
                
                st.info(f"💡 현재 시각화된 구간은 전체 녹음 중 {start_time:.2f}초부터 {end_time:.2f}초까지입니다. 두 그래프의 시간축이 정렬되었습니다.")
            else:
                st.warning("선택된 구간이 너무 짧습니다. 범위를 다시 조정해 주세요.")
        else:
            st.write("---")
            st.info("설정을 조정한 후 위의 '분석 시작' 버튼을 눌러주세요.")

    except Exception as e:
        st.error(f"오디오 처리 중 오류가 발생했습니다: {e}")

else:
    st.write("---")
    st.markdown("""
    <div style="text-align: center; color: grey; padding: 50px;">
        <h3>상단의 버튼을 눌러 녹음을 시작하세요.</h3>
        <p>녹음 완료 후 '분석 시작' 버튼을 통해 정밀 도표를 확인할 수 있습니다.</p>
    </div>
    """, unsafe_allow_html=True)

# 🏛️ 이론적 배경 섹션
st.divider()
st.markdown("""
### 🏛️ Theoretical Background
* **Temporal Alignment**: 파형과 스펙트로그램의 시간축을 일치시키는 것은 데이터의 **신뢰성(Consistency)**을 확보하는 중요한 시각적 비계입니다.
* **Learner Agency**: '분석 시작' 버튼을 추가함으로써 학습자는 단순히 결과를 기다리는 수동적 관찰자에서, 데이터를 직접 처리하고 분석을 결정하는 **능동적 주체(Agent)**로 전환됩니다.
""")
