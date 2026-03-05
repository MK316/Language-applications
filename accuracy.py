import streamlit as st
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import io
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from difflib import SequenceMatcher

# 페이지 설정
st.set_page_config(page_title="AI 발음 비교 분석기", layout="wide")

st.title("🎙️ AI-Native 발음 비교 분석기 (v2.0)")
st.write("원어민의 파형과 자신의 발음을 동일한 시간 선상에서 비교해보세요.")

# 1. 목표 문장 설정
target_text = "The quick brown fox jumps over the lazy dog."
st.info(f"🎯 **Target:** {target_text}")

# 2. 녹음 섹션
audio = mic_recorder(
    start_prompt="🎤 녹음 시작",
    stop_prompt="🛑 녹음 완료",
    key="recorder"
)

def get_tts_waveform(text, target_duration, target_sr):
    """gTTS로 생성된 음성을 목표 길이에 맞춰 numpy 배열로 반환"""
    tts = gTTS(text=text, lang='en')
    tts_fp = io.BytesIO()
    tts.save(tts_fp)
    tts_fp.seek(0)
    
    # librosa로 직접 로드 (pydub 미사용)
    y_tts, _ = librosa.load(tts_fp, sr=target_sr)
    
    # 시간 맞추기 (Padding)
    target_samples = int(target_duration * target_sr)
    if len(y_tts) < target_samples:
        padding = (target_samples - len(y_tts)) // 2
        y_tts = np.pad(y_tts, (padding, target_samples - len(y_tts) - padding), 'constant')
    else:
        y_tts = y_tts[:target_samples]
    return y_tts

if audio:
    # 3. 데이터 로드 및 STT 분석
    audio_bytes = audio['bytes']
    audio_fp = io.BytesIO(audio_bytes)
    
    # 학습자 데이터 로드
    y_learner, sr_rate = librosa.load(audio_fp, sr=22050)
    duration_sec = len(y_learner) / sr_rate
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 인식 결과")
        r = sr.Recognizer()
        # STT를 위해 wav 형식으로 임시 변환 (speech_recognition용)
        try:
            with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
                audio_data = r.record(source)
                transcript = r.recognize_google(audio_data, language='en-US')
                
                score = SequenceMatcher(None, target_text.lower(), transcript.lower()).ratio()
                st.metric("발음 정확도", f"{int(score * 100)}%")
                st.write(f"**AI 인식:** {transcript}")
        except:
            st.warning("텍스트 변환에 실패했습니다. 하지만 파형 비교는 가능합니다.")

    with col2:
        st.subheader("📊 시각화 도구")
        if st.button("원어민 파형과 내 발음 대조하기"):
            with st.spinner("데이터 동기화 중..."):
                try:
                    # 원어민 TTS 파형 생성 (학습자 시간에 맞춤)
                    y_native = get_tts_waveform(target_text, duration_sec, sr_rate)
                    
                    # 그래프 그리기
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
                    
                    # Native (상단)
                    librosa.display.waveshow(y_native, sr=sr_rate, ax=ax1, color='lightgray')
                    ax1.set_title("Native Speaker (Standard)")
                    ax1.set_ylabel("Amp")
                    
                    # Learner (하단)
                    librosa.display.waveshow(y_learner, sr=sr_rate, ax=ax2, color='skyblue')
                    ax2.set_title("Your Pronunciation")
                    ax2.set_ylabel("Amp")
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.caption(f"💡 총 {duration_sec:.2f}초의 구간을 동일하게 정렬했습니다. 파형의 굴곡(Peak)이 일치하는지 확인하세요.")
                except Exception as e:
                    st.error(f"시각화 에러: {e}")

# 사이드바 가이드
st.sidebar.title("📚 학습 가이드")
st.sidebar.write("""
1. **정확도(%)**: 단어의 일치도를 나타냅니다.
2. **파형(Waveform)**: 
    - **가로축**: 시간의 흐름
    - **세로축**: 소리의 강도(Stress)
- 원어민 파형과 자신의 파형이 비슷한 위치에서 솟아오르는지 확인해보세요!
""")
