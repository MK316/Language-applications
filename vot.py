import streamlit as st
from streamlit_mic_recorder import mic_recorder
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import io
from pydub import AudioSegment

st.set_page_config(page_title="Linguistic Pronunciation Visualizer", layout="wide")

st.title("🗣️ AI-Mediated Linguistic Analysis")
st.subheader("Precision Analysis: Select a segment to explore in depth")

# 레이아웃 구성
col1, col2 = st.columns([1, 2])

with col1:
    st.info("📱 **Mobile Tip**: 아이폰 사용자는 반드시 **무음 스위치**를 해제해 주세요.")
    
    # 오디오 레코더
    audio = mic_recorder(
        start_prompt="⏺️ 녹음 시작 (Start)",
        stop_prompt="⏹️ 녹음 중지 (Stop)",
        key='recorder'
    )

if audio:
    try:
        # 1. 녹음 데이터 처리 및 변환
        audio_bytes = audio['bytes']
        audio_bio = io.BytesIO(audio_bytes)
        audio_segment = AudioSegment.from_file(audio_bio)
        
        # 분석을 위해 WAV로 변환
        wav_bio = io.BytesIO()
        audio_segment.export(wav_bio, format="wav")
        wav_data = wav_bio.getvalue()
        wav_bio.seek(0)

        # 2. 오디오 로드 및 전체 시간 계산
        y, sr = librosa.load(wav_bio, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        with col2:
            st.success(f"녹음 완료! (총 길이: {duration:.2f}초)")
            st.audio(wav_data, format="audio/wav")
            
            st.divider()
            
            # 3. 구간 선택 슬라이더 (사용자 개입: Engagement)
            st.write("🔍 **분석할 구간을 선택하세요 (Select Range)**")
            start_time, end_time = st.slider(
                "시간 범위 (초)",
                0.0, float(duration), (0.0, float(duration)),
                step=0.01,
                help="슬라이더를 조절하여 특정 단어나 음절 구간만 확대해서 볼 수 있습니다."
            )

            # 4. 선택된 구간만 크롭
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            y_cropped = y[start_sample:end_sample]
            
            if len(y_cropped) > 0:
                # 5. 시각화 (선택된 구간만 반영)
                fig, ax = plt.subplots(2, 1, figsize=(10, 8), dpi=100)
                plt.subplots_adjust(hspace=0.5)
                
                # Waveform (선택 구간)
                librosa.display.waveshow(y_cropped, sr=sr, ax=ax[0], color="#2c3e50")
                ax[0].set_title(f"Selected Waveform: {start_time:.2e}s ~ {end_time:.2e}s", fontsize=12)
                ax[0].set_xlabel("Time (s)")
                
                # Spectrogram (선택 구간)
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y_cropped)), ref=np.max)
                img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax[1], cmap='magma', x_coords=np.linspace(start_time, end_time, D.shape[1]))
                ax[1].set_title("Selected Spectrogram: Detailed Energy Patterns", fontsize=12)
                fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
                
                st.pyplot(fig)
                
                # 분석 팁
                st.info(f"💡 현재 {end_time - start_time:.2f}초 구간을 집중 분석 중입니다. 파열음(Plosives)의 경우 아주 짧은 구간(약 0.1~0.2초)을 확대하면 VOT를 더 잘 관찰할 수 있습니다.")
            else:
                st.warning("구간을 다시 선택해 주세요.")

    except Exception as e:
        st.error(f"오디오 처리 중 오류가 발생했습니다: {e}")

else:
    with col2:
        st.image("https://via.placeholder.com/800x400.png?text=Waiting+for+Voice+Input...", use_column_width=True)
        st.write("왼쪽에서 녹음을 마치면 구간 선택 슬라이더와 분석 도표가 나타납니다.")
