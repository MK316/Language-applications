import streamlit as st
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import io, os, librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from difflib import SequenceMatcher

# --- 유틸리티 함수 ---
def get_net_speaking_time(audio_path):
    audio = AudioSegment.from_file(audio_path)
    nonsilent_chunks = detect_nonsilent(audio, min_silence_len=100, silence_thresh=-45)
    if not nonsilent_chunks: return 0
    return sum([end - start for start, end in nonsilent_chunks]) / 1000.0

# --- 스트림릿 설정 ---
st.set_page_config(page_title="AI 발음 분석기", layout="wide")

# [수정] 유성음 비율 80% 이상, 자연스러운 문장 10단계
sample_sentences = {
    "Level 1: 모음의 흐름": "All in line.",
    "Level 2: 비음의 울림": "Morning is near.",
    "Level 3: 유음과 비음": "Learning is mainly online.",
    "Level 4: 부드러운 연결": "Rainy morning on Monday.",
    "Level 5: 의문문 억양": "Where are you roaming now?",
    "Level 6: 감정의 고조": "I am really moving on.",
    "Level 7: 유성 마찰음": "Love always wins over all.",
    "Level 8: 긴 호흡 연결": "Early morning running is narrow.",
    "Level 9: 복합 유성음": "Millions are zooming in on me.",
    "Level 10: 최종 도전": "Normal morning reveals a yellow moon."
}

st.title("🎙️ AI-Native 발음 & 유창성 클리닉")

# --- Step 1: 문장 선택 ---
st.subheader("1단계: 연습할 문장 선택하기")
selected_level = st.selectbox("난이도를 선택하세요 (1~10):", list(sample_sentences.keys()))
target_text = sample_sentences[selected_level]

# [수정] 문장 박스와 아래 요소 사이의 간격(margin-bottom)을 넓힘
st.markdown(f"""
    <div style="border: 2px solid #1f77b4; border-radius: 10px; padding: 25px; 
                background-color: #f0f2f6; text-align: center; margin-bottom: 50px;">
        <h2 style="color: #1f77b4; margin: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
            "{target_text}"
        </h2>
    </div>
    """, unsafe_allow_html=True)

# --- Step 2: 녹음 ---
# 간격이 확보된 상태에서 녹음 버튼이 나타남
audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 녹음 완료", key="recorder")

if audio:
    try:
        # 오디오 처리
        learner_raw = AudioSegment.from_file(io.BytesIO(audio['bytes']))
        learner_raw.export("temp_learner.wav", format="wav")
        
        tts = gTTS(text=target_text, lang='en')
        tts.save("temp_native.mp3")
        native_raw = AudioSegment.from_file("temp_native.mp3", format="mp3")
        native_raw.export("temp_native.wav", format="wav")

        learner_net_time = get_net_speaking_time("temp_learner.wav")
        native_net_time = get_net_speaking_time("temp_native.wav")

        y_learner, sr_l = librosa.load("temp_learner.wav", sr=22050)
        y_native, _ = librosa.load("temp_native.wav", sr=sr_l)

        tab1, tab2, tab3, tab4 = st.tabs(["🎯 인식 결과", "⏱️ 유창성 분석", "🔊 음파 대조", "📈 피치 분석"])

        with tab1:
            st.subheader("AI 피드백")
            r = sr.Recognizer()
            with sr.AudioFile("temp_learner.wav") as source:
                audio_data = r.record(source)
                try:
                    transcript = r.recognize_google(audio_data, language='en-US')
                    score = SequenceMatcher(None, target_text.lower().replace('.', ''), transcript.lower()).ratio()
                    c1, c2 = st.columns([1, 2])
                    c1.metric("정확도 점수", f"{int(score * 100)}점")
                    c2.success(f"**인식 결과:** {transcript}")
                    st.info("💡 옆의 **[⏱️ 유창성 분석]** 탭에서 속도를 확인해보세요!")
                except: st.error("인식 실패")

        with tab2:
            st.subheader("발화 속도 분석")
            ratio = (learner_net_time / native_net_time) * 100 if native_net_time > 0 else 0
            c1, c2, c3 = st.columns(3)
            c1.metric("내 발화 시간", f"{learner_net_time:.2f}초")
            c2.metric("원어민 시간", f"{native_net_time:.2f}초")
            c3.metric("속도 비율", f"{int(ratio)}%")
            st.caption("※ 무음 구간을 제외한 순수 스피치 구간만 측정되었습니다.")

        with tab3:
            st.audio("temp_learner.wav"); st.audio("temp_native.mp3")
            fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
            librosa.display.waveshow(y_native, sr=sr_l, ax=ax1, color='lightgray')
            librosa.display.waveshow(y_learner, sr=sr_l, ax=ax2, color='skyblue')
            st.pyplot(fig1)

        with tab4:
            st.subheader("피치 점선 비교 (나열형)")
            st.audio("temp_learner.wav"); st.audio("temp_native.mp3")
            f0_l, voiced_l, _ = librosa.pyin(y_learner, fmin=70, fmax=400)
            f0_n, voiced_n, _ = librosa.pyin(y_native, fmin=70, fmax=400)
            fig2, ax = plt.subplots(figsize=(12, 5))
            ax.plot(librosa.times_like(f0_n)[voiced_n], f0_n[voiced_n], 'o--', label='Native', color='lightgray', markersize=3, alpha=0.6)
            ax.plot(librosa.times_like(f0_l)[voiced_l], f0_l[voiced_l], 'o--', label='You', color='#1f77b4', markersize=4)
            ax.set_ylim([50, 400]); ax.legend(); st.pyplot(fig2)

    except Exception as e: st.error(f"오류: {e}")
    finally:
        for f in ["temp_native.mp3", "temp_native.wav", "temp_learner.wav"]:
            if os.path.exists(f): os.remove(f)
