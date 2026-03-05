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
        nonsilent_chunks = detect_nonsilent(audio, min_silence_len=100, silence_thresh=-45)
        if not nonsilent_chunks: return 0
        return sum([end - start for start, end in nonsilent_chunks]) / 1000.0
    except: return 0

# --- 스트림릿 설정 ---
st.set_page_config(page_title="AI 발음 분석기", layout="wide")

if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

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

selected_level = st.selectbox("학습 단계를 선택하세요:", list(sample_sentences.keys()), 
                              on_change=lambda: st.session_state.update({"analysis_done": False}))
target_text = sample_sentences[selected_level]

# 텍스트 박스 (절반 사이즈 및 margin-bottom: 30px 설정)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(f"""
        <div style="border: 2px solid #1f77b4; border-radius: 10px; padding: 20px; 
                    background-color: #f8f9fb; text-align: center; margin-bottom: 30px;">
            <h3 style="color: #1f77b4; margin: 0; font-weight: 600;">"{target_text}"</h3>
        </div>
        """, unsafe_allow_html=True)

# 녹음 인터페이스
c_rec1, c_rec2, c_rec3 = st.columns([1, 1, 1])
with c_rec2:
    audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 녹음 완료", key="recorder")

if audio:
    st.write("✅ 녹음이 완료되었습니다.")
    if st.button("📊 결과 분석하기", use_container_width=True):
        st.session_state.analysis_done = True
        st.session_state.audio_bytes = audio['bytes']

if st.session_state.get('analysis_done'):
    try:
        audio_stream = io.BytesIO(st.session_state.audio_bytes)
        full_audio = AudioSegment.from_file(audio_stream)
        full_audio.export("temp_stt.wav", format="wav")
        
        learner_proc = full_audio.strip_silence(silence_thresh=-45, padding=300)
        learner_proc.export("temp_learner.wav", format="wav")
        
        tts = gTTS(text=target_text, lang='en')
        tts.save("temp_native.mp3")
        native_raw = AudioSegment.from_file("temp_native.mp3", format="mp3")
        native_raw = native_raw.strip_silence(silence_thresh=-45, padding=300)
        native_raw.export("temp_native.wav", format="wav")

        y_learner, sr_l = librosa.load("temp_learner.wav", sr=22050)
        y_native, _ = librosa.load("temp_native.wav", sr=sr_l)
        
        learner_net_time = get_net_speaking_time("temp_learner.wav")
        native_net_time = get_net_speaking_time("temp_native.wav")

        tab1, tab2, tab3, tab4 = st.tabs(["🎯 AI 점수", "⏱️ 유창성 분석", "🔊 음파 대조", "📈 피치 분석"])

        with tab1:
            r = sr.Recognizer()
            with sr.AudioFile("temp_stt.wav") as source:
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = r.record(source)
                try:
                    transcript = r.recognize_google(audio_data, language='en-US')
                    clean_target = target_text.lower().replace('.', '').replace(',', '').replace('?', '')
                    score = SequenceMatcher(None, clean_target, transcript.lower()).ratio()
                    final_score = int(score * 100)

                    # [수정] 점수와 인식 결과를 Info 박스 형태로 처리
                    st.markdown(f"""
                        <div style="display: flex; gap: 10px; margin-top: 20px;">
                            <div style="flex: 1; background-color: #e8f4f8; border-left: 5px solid #1f77b4; padding: 15px; border-radius: 5px;">
                                <small style="color: #1f77b4; font-weight: bold;">정확도 점수</small>
                                <h2 style="margin: 0; color: #1f77b4;">{final_score}점</h2>
                            </div>
                            <div style="flex: 2; background-color: #eafaf1; border-left: 5px solid #2ecc71; padding: 15px; border-radius: 5px;">
                                <small style="color: #27ae60; font-weight: bold;">AI 인식 결과</small>
                                <p style="margin: 0; font-size: 1.2rem; color: #1e8449;">{transcript}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                except: st.error("인식 실패: 마이크와 입 사이의 거리를 조절하여 다시 녹음해 주세요.")

        # --- 나머지 Tab 로직은 기존 최적화 버전 유지 ---
        with tab2:
            ratio = (learner_net_time / native_net_time) * 100 if native_net_time > 0 else 0
            c1, c2, c3 = st.columns(3)
            c1.metric("내 발화 시간", f"{learner_net_time:.2f}초")
            c2.metric("원어민 시간", f"{native_net_time:.2f}초")
            c3.metric("속도 비율", f"{int(ratio)}%")

        with tab3:
            st.audio("temp_learner.wav"); st.audio("temp_native.mp3")
            fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4))
            librosa.display.waveshow(y_native, sr=sr_l, ax=ax1, color='lightgray')
            librosa.display.waveshow(y_learner, sr=sr_l, ax=ax2, color='skyblue')
            plt.tight_layout(); st.pyplot(fig1)

        with tab4:
            st.audio("temp_learner.wav"); st.audio("temp_native.mp3")
            f0_l, v_l, p_l = librosa.pyin(y_learner, fmin=75, fmax=400, hop_length=128)
            f0_n, v_n, p_n = librosa.pyin(y_native, fmin=60, fmax=400, hop_length=128)
            f0_l_filtered = np.where(v_l & (p_l > 0.25) & (f0_l > 80), f0_l, np.nan)
            f0_n_filtered = np.where(v_n & (p_n > 0.05), f0_n, np.nan)
            fig2, (ax_n, ax_l) = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
            ax_n.plot(librosa.times_like(f0_n, hop_length=128), f0_n_filtered, color='lightgray', linewidth=3)
            ax_l.plot(librosa.times_like(f0_l, hop_length=128), f0_l_filtered, color='#1f77b4', linewidth=2.5)
            st.pyplot(fig2)

    except Exception as e: st.error(f"오류: {e}")
    finally:
        for f in ["temp_native.mp3", "temp_native.wav", "temp_learner.wav", "temp_stt.wav"]:
            if os.path.exists(f): os.remove(f)
