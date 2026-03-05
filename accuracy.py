import streamlit as st
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import io
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from difflib import SequenceMatcher

st.set_page_config(page_title="AI 발음 분석기", layout="wide")
st.title("🎙️ AI-Native 발음 비교 분석기 (Audio Playback)")

# 1. 목표 문장 설정
target_text = "The quick brown fox jumps over the lazy dog."
st.info(f"🎯 **Target:** {target_text}")

# 2. 녹음 섹션
audio = mic_recorder(
    start_prompt="🎤 녹음 시작",
    stop_prompt="🛑 녹음 완료",
    key="recorder"
)

if audio:
    try:
        # 학습자 데이터를 WAV로 강제 변환 및 저장
        learner_segment = AudioSegment.from_file(io.BytesIO(audio['bytes']))
        learner_segment.export("temp_learner.wav", format="wav")
        
        # 3. 분석 결과 및 오디오 재생 (상단 배치)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 인식 결과")
            r = sr.Recognizer()
            with sr.AudioFile("temp_learner.wav") as source:
                audio_data = r.record(source)
                transcript = r.recognize_google(audio_data, language='en-US')
                score = SequenceMatcher(None, target_text.lower(), transcript.lower()).ratio()
                st.metric("발음 정확도", f"{int(score * 100)}%")
                st.write(f"**AI 인식:** {transcript}")
        
        with col2:
            st.subheader("🔊 다시 듣기")
            # 내 목소리 재생
            st.write("나의 발음:")
            st.audio("temp_learner.wav")
            
            # 원어민 음성 생성 및 재생
            tts = gTTS(text=target_text, lang='en')
            tts.save("temp_native.mp3")
            st.write("원어민 가이드:")
            st.audio("temp_native.mp3")

        # 4. 시각화 섹션 (버튼 클릭 시)
        st.divider()
        if st.button("📊 파형으로 정밀 비교하기"):
            with st.spinner("데이터 동기화 중..."):
                # TTS를 WAV로 변환 (시각화용)
                native_segment = AudioSegment.from_file("temp_native.mp3", format="mp3")
                native_segment.export("temp_native.wav", format="wav")
                
                # librosa로 두 WAV 파일 로드
                y_learner, sr_l = librosa.load("temp_learner.wav", sr=22050)
                y_native, _ = librosa.load("temp_native.wav", sr=sr_l)
                
                # 시간 축 맞추기 (Padding)
                target_samples = len(y_learner)
                if len(y_native) < target_samples:
                    padding = (target_samples - len(y_native)) // 2
                    y_native = np.pad(y_native, (padding, target_samples - len(y_native) - padding), 'constant')
                else:
                    y_native = y_native[:target_samples]

                # 그래프 출력
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
                librosa.display.waveshow(y_native, sr=sr_l, ax=ax1, color='lightgray')
                ax1.set_title("Native Speaker (Standard)")
                librosa.display.waveshow(y_learner, sr=sr_l, ax=ax2, color='skyblue')
                ax2.set_title("Your Pronunciation")
                plt.tight_layout()
                st.pyplot(fig)
                
                # 시각화용 임시 파일 정리
                if os.path.exists("temp_native.wav"):
                    os.remove("temp_native.wav")

    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
    finally:
        # 파일 정리 (재생을 위해 mp3는 유지하거나 나중에 삭제하도록 설계)
        pass

# 사이드바: 교육적 가이드
st.sidebar.markdown("### 💡 학습 팁")
st.sidebar.write("""
1. 먼저 **원어민 가이드**를 듣고 리듬을 익히세요.
2. 자신의 녹음을 듣고 **어색한 부분**을 찾아보세요.
3. **파형 비교**를 통해 원어민과 나의 '강세(Peak)' 위치가 일치하는지 확인하세요.
""")
