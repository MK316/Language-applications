import streamlit as st
from streamlit_mic_recorder import mic_recorder
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import io
from pydub import AudioSegment

# 1. 페이지 설정
st.set_page_config(page_title="Linguistic Pronunciation Visualizer", layout="wide")

st.title("🗣️ AI-Mediated Linguistic Analysis")
st.subheader("Precision Analysis for Voice Onset Time (VOT)")
st.caption("Instructions: Record words like 'pie, bye, spy' and measure the aspiration gap.")

# 2. 상단: 안내 및 녹음 컨트롤
top_col1, top_col2, top_col3 = st.columns([1, 2, 1])

with top_col2:
    st.info("📱 **Mobile Tip**: 아이폰 사용자는 기기 옆면의 **Silent Mode**를 해제해 주세요.")
    audio = mic_recorder(
        start_prompt="⏺️ 녹음 시작 (Start Recording)",
        stop_prompt="⏹️ 녹음 중지 (Stop Recording)",
        key='recorder'
    )

st.divider()

if audio:
    try:
        # 데이터 처리 및 WAV 변환 (아이폰 호환성 해결)
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
        
        # --- 3. 분석 설정 레이아웃 (시간 및 주파수 조절) ---
        st.write("### ⚙️ 분석 설정 (Analysis Settings)")
        config_col1, config_col2 = st.columns(2)

        with config_col1:
            st.write("🔍 **시간 구간 선택 (Time Range Selection)**")
            start_time, end_time = st.slider(
                "분석할 구간을 선택하세요 (단어 또는 음절 단위)",
                0.0, float(duration), (0.0, float(duration)),
                step=0.001, # 정밀 측정을 위해 1ms 단위 설정
                key="time_slider"
            )
            # VOT 측정값 계산 (Engagement: 학습자가 직접 수치 확인)
            selected_duration_ms = (end_time - start_time) * 1000
            st.metric(label="선택된 구간의 길이 (Measured Duration)", value=f"{selected_duration_ms:.2f} ms")

        with config_col2:
            st.write("📈 **주파수 범위 제한 (Frequency Range)**")
            max_freq = st.slider(
                "최대 주파수 표시 (Hz)",
                2000, max_sr_half, min(8000, max_sr_half),
                step=500,
                key="freq_slider",
                help="일반적으로 인간의 음성은 8000Hz 이내에 핵심 정보가 있습니다."
            )

        # 4. 분석 시작 버튼 (사용자가 모든 설정을 마친 후 실행)
        if st.button("🚀 분석 시작 (Start Detailed Analysis)", type="primary", use_container_width=True):
            
            # 선택된 구간 크롭
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            y_cropped = y[start_sample:end_sample]
            
            if len(y_cropped) > 0:
                # 시각화 (상하 배치로 가로 공간 최대 활용)
                fig, ax = plt.subplots(2, 1, figsize=(15, 10), dpi=100)
                plt.subplots_adjust(hspace=0.4)
                
                # 1) Waveform (offset 설정을 통해 시간축 정렬 문제 해결)
                librosa.display.waveshow(y_cropped, sr=sr, ax=ax[0], color="#2c3e50", offset=start_time)
                ax[0].set_title(f"Waveform: {start_time:.3f}s ~ {end_time:.3f}s", fontsize=14)
                ax[0].set_xlabel("Time (s)")
                ax[0].grid(True, axis='x', linestyle='--', alpha=0.5)
                
                # 2) Spectrogram (x_coords 설정을 통해 시간축 일치)
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y_cropped)), ref=np.max)
                img = librosa.display.specshow(
                    D, sr=sr, x_axis='time', y_axis='hz', ax=ax[1], 
                    cmap='magma', 
                    x_coords=np.linspace(start_time, end_time, D.shape[1])
                )
                
                ax[1].set_ylim(0, max_freq) 
                ax[1].set_title(f"Spectrogram: Frequency Distribution (up to {max_freq}Hz)", fontsize=14)
                fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
                
                # 결과 출력
                st.pyplot(fig, use_container_width=True)
                
                st.info(f"💡 분석 팁: Waveform에서 소리의 폭발 지점(Burst)과 Spectrogram에서 에너지가 수직으로 나타나는 지점을 비교해 보세요.")
            else:
                st.warning("선택된 구간이 너무 짧습니다. 슬라이더 범위를 다시 조정해 주세요.")
        else:
            st.write("---")
            st.info("👆 위에서 분석하고 싶은 구간을 정한 뒤 '분석 시작' 버튼을 누르면 정밀 도표가 나타납니다.")

    except Exception as e:
        st.error(f"오디오 처리 중 오류가 발생했습니다: {e}")

else:
    st.write("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 50px; border: 2px dashed #bdc3c7; border-radius: 15px;">
        <h3>🎙️ Waiting for Recording...</h3>
        <p>상단의 버튼을 눌러 자신의 목소리를 녹음해 보세요.</p>
        <p>녹음이 완료되면 언어학적 분석을 위한 슬라이더와 도구가 나타납니다.</p>
    </div>
    """, unsafe_allow_html=True)

# 5. 이론적 배경 (Pedagogical Scaffolding)
st.divider()
st.markdown("""
### 🏛️ Pedagogical Framework
* **Difficulty to Depth**: 음성 신호라는 복잡한(Difficulty) 데이터를 시각화하고 구간을 직접 제어함으로써, 추상적인 음성학 지식을 깊이 있는(Depth) 분석적 이해로 전환합니다.
* **Temporal Alignment**: 파형과 스펙트로그램의 시간축을 일치시켜 데이터의 **시각적 신뢰성**을 확보하였습니다.
* **Engagement**: 학습자가 직접 슬라이더를 움직여 VOT를 측정하는 행위는 지식의 수동적 수용이 아닌 **능동적 탐구(Agency)를** 유도합니다.
""")
