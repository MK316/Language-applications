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
    try:
        audio = AudioSegment.from_file(audio_path)
        nonsilent_chunks = detect_nonsilent(audio, min_silence_len=100, silence_thresh=-40)
        if not nonsilent_chunks: return 0
        return sum([end - start for start, end in nonsilent_chunks]) / 1000.0
    except: return 0

# --- 스트림릿 설정 ---
st.set_page_config(page_title="AI 발음 분석기", layout="wide")

# [20단계] 실용적 표현 리스트
sample_sentences = {
    "Level 01: (인사/기초)": "I am on my way.",
    "Level 02: (일상/기초)": "Nice room you have.",
    "Level 03: (일상/기초)": "Dinner is ready now.",
    "Level 04: (일상/기초)": "Leave a message online.",
    "Level 05: (캠퍼스/기초)": "Our classroom is really warm.",
    "Level 06: (캠퍼스/기초)": "No one knows my name here.",
    "Level 07: (일상/중급)": "Running alone is always fine.",
    "Level 08: (비즈니스/중급)": "Email me any minor news.",
    "Level 09: (비즈니스/중급)": "My main revenue is moving up.",
    "Level 10: (일상/주어 확장)": "Millions of men are moving online.",
    "Level 11: (캠퍼스/주어 확장)": "Learning a new language is normal now.",
    "Level 12: (비즈니스/주어 확장)": "All our managers are in a long meeting.",
    "Level 13: (일상/연결 확장)": "Early morning jogging is my main manner.",
    "Level 14: (캠퍼스/연결 확장)": "Our long moonlit journey remains in my mind.",
    "Level 15: (대학/학술)": "Online learning remains a main avenue in our era.",
    "Level 16: (대학/학술)": "Modern laws remain relevant in our human memory.",
    "Level 17: (비즈니스/심화)": "Managing a small loan is always a main worry.",
    "Level 18: (대학/심화)": "Meaningful rumors are blooming on the rainy river.",
    "Level 19: (고급/실무)": "Maintaining a warm memory lowers our lonely alarm.",
    "Level 20: (대학/고급)": "Enormous animal roaming remains a normal human alarm."
}

st.title("🎙️ AI-Native 실용 발음 & 유창성 클리닉")

selected_level = st.selectbox("학습 단계를 선택하세요:", list(sample_sentences.keys()))
target_text = sample_sentences[selected_level]

st.markdown(f"""
    <div style="border: 2px solid #1f77b4; border-radius: 12px; padding: 35px; 
                background-color: #f8f9fb; text-align: center; margin-bottom: 70px;">
        <h2 style="color: #1f77b4; margin: 0; font-family: 'Segoe UI', sans-serif;">"{target_text}"</h2>
    </div>
    """, unsafe_allow_html=True)

audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 녹음 완료", key="recorder")

if audio:
    try:
        audio_stream = io.BytesIO(audio['bytes'])
        learner_raw = AudioSegment.from_file(audio_stream)
        learner_proc = learner_raw.strip_silence(silence_thresh=-40, padding=150)
        learner_proc.export("temp_learner.wav", format="wav")
        
        tts = gTTS(text=target_text, lang='en')
        tts.save("temp_native.mp3")
        native_raw = AudioSegment.from_file("temp_native.mp3", format="mp3")
        native_raw = native_raw.strip_silence(silence_thresh=-40, padding=150)
        native_raw.export("temp_native.wav", format="wav")

        learner_net_time = get_net_speaking_time("temp_learner.wav")
        native_net_time = get_net_speaking_time("temp_native.wav")
        y_learner, sr_l = librosa.load("temp_learner.wav", sr=22050)
        y_native, _ = librosa.load("temp_native.wav", sr=sr_l)

        tab1, tab2, tab3, tab4 = st.tabs(["🎯 AI 점수", "⏱️ 유창성 분석", "🔊 음파 대조", "📈 피치 분석"])

        with tab1:
            r = sr.Recognizer()
            with sr.AudioFile("temp_learner.wav") as source:
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = r.record(source)
                try:
                    transcript = r.recognize_google(audio_data, language='en-US')
                    clean_target = target_text.lower().replace('.', '').replace(',', '').replace('?', '')
                    score = SequenceMatcher(None, clean_target, transcript.lower()).ratio()
                    st.metric("정확도 점수", f"{int(score * 100)}점")
                    st.success(f"**인식 결과:** {transcript}")
                except: st.error("인식 실패: 마이크와 입 사이의 거리를 조절해 보세요.")

        with tab2:
            ratio = (learner_net_time / native_net_time) * 100 if native_net_time > 0 else 0
            c1, c2, c3 = st.columns(3)
            c1.metric("내 발화 시간", f"{learner_net_time:.2f}초")
            c2.metric("원어민 시간", f"{native_net_time:.2f}초")
            c3.metric("속도 비율", f"{int(ratio)}%")

        with tab3:
            fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4))
            librosa.display.waveshow(y_native, sr=sr_l, ax=ax1, color='lightgray')
            librosa.display.waveshow(y_learner, sr=sr_l, ax=ax2, color='skyblue')
            st.pyplot(fig1)

        with tab4:
            st.subheader("억양 멜로디 (좌: 원어민, 우: 나)")
            st.audio("temp_learner.wav"); st.audio("temp_native.mp3")
            
            # [핵심 수정] hop_length를 512에서 128로 줄여 해상도 4배 향상 (직선형 방지)
            # win_length를 조절하여 부드러운 추출 유도
            f0_l, v_l, p_l = librosa.pyin(y_learner, fmin=80, fmax=400, hop_length=128)
            f0_n, v_n, p_n = librosa.pyin(y_native, fmin=60, fmax=400, hop_length=128)
            
            # 필터링 임계값 완화 (0.6 -> 0.4) 및 저역대 필터 조정 (95 -> 85)
            valid_l = v_l & (p_l > 0.4) & (f0_l > 85)
            valid_n = v_n & (p_n > 0.2) 

            fig2, (ax_n, ax_l) = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
            ax_n.plot(librosa.times_like(f0_n, hop_length=128)[valid_n], f0_n[valid_n], 'o', 
                      color='lightgray', markersize=2, alpha=0.5)
            # 점들을 선으로 잇지 않고 '점'으로만 표현하여 실제 데이터를 정직하게 보여줌
            
            ax_l.plot(librosa.times_like(f0_l, hop_length=128)[valid_l], f0_l[valid_l], 'o', 
                      color='#1f77b4', markersize=3)
            
            ax_n.set_title("Native Speaker Intonation"); ax_n.set_ylim([80, 400])
            ax_l.set_title("Your Intonation"); ax_l.set_ylim([80, 400])
            plt.tight_layout(); st.pyplot(fig2)
            st.caption("※ 분석 해상도를 높여 억양의 미세한 굴곡을 더 정확하게 표현합니다.")

    except Exception as e: st.error(f"오류: {e}")
    finally:
        for f in ["temp_native.mp3", "temp_native.wav", "temp_learner.wav"]:
            if os.path.exists(f): os.remove(f)
