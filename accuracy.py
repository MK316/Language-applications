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
def get_speech_bounds(audio_segment, silence_thresh=-40, min_silence_len=100, buffer_ms=100):
    nonsilent_intervals = detect_nonsilent(audio_segment, 
                                           min_silence_len=min_silence_len, 
                                           silence_thresh=silence_thresh)
    if not nonsilent_intervals: return 0, len(audio_segment)
    start_trim = max(0, nonsilent_intervals[0][0] - buffer_ms)
    end_trim = min(len(audio_segment), nonsilent_intervals[-1][1] + 50)
    return start_trim, end_trim

# --- 스트림릿 설정 ---
st.set_page_config(page_title="AI 발음 분석기", layout="wide")

if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'start_time' not in st.session_state:
    st.session_state.start_time = 0.0
if 'end_time' not in st.session_state:
    st.session_state.end_time = 0.0

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

st.markdown("### 🎙️ AI 활용 발음 연습")

selected_level = st.selectbox("Step 1: 학습 단계를 선택하세요:", list(sample_sentences.keys()), 
                              on_change=lambda: st.session_state.update({"analysis_done": False}))
target_text = sample_sentences[selected_level]

col2 = st.columns([1, 2, 1])[1]
with col2:
    st.markdown(f"""<div style="border: 2px solid #1f77b4; border-radius: 12px; padding: 15px; background-color: #f8f9fb; text-align: center; margin-bottom: 20px;"><h3 style="color: #1f77b4; margin: 0; font-weight: 700;">"{target_text}"</h3></div>""", unsafe_allow_html=True)
    rec_col2 = st.columns([1, 1, 1])[1]
    with rec_col2:
        audio = mic_recorder(start_prompt="🎤 Step 2: 녹음 시작", stop_prompt="🛑 녹음 완료", key="recorder")

if audio:
    st.divider()
    audio_bytes = audio['bytes']
    audio_stream = io.BytesIO(audio_bytes)
    full_audio = AudioSegment.from_file(audio_stream)
    full_audio.export("temp_preview.wav", format="wav")
    duration_sec = len(full_audio) / 1000.0
    y_full, sr_f = librosa.load("temp_preview.wav", sr=22050)
    
    st.subheader("✂️ 발화 구간 조정")
    fig_preview, ax = plt.subplots(figsize=(12, 2.5))
    librosa.display.waveshow(y_full, sr=sr_f, ax=ax, color='skyblue', alpha=0.8)
    st.pyplot(fig_preview)
    
    c_play, c_slide = st.columns([1, 2])
    with c_play: st.audio(audio_bytes)
    with c_slide:
        v_start, v_end = get_speech_bounds(full_audio)
        time_range = st.slider("분석 구간 선택:", 0.0, duration_sec, (float(v_start/1000), float(v_end/1000)), step=0.1)
        
    if st.button("📊 Step 3: 결과 분석하기", use_container_width=True):
        st.session_state.analysis_done = True
        st.session_state.audio_bytes = audio_bytes
        st.session_state.start_time = time_range[0]
        st.session_state.end_time = time_range[1]

if st.session_state.analysis_done:
    try:
        audio_stream = io.BytesIO(st.session_state.audio_bytes)
        full_audio = AudioSegment.from_file(audio_stream)
        
        # 1. 학습자 오디오 크롭 및 저장
        start_ms, end_ms = st.session_state.start_time * 1000, st.session_state.end_time * 1000
        cropped_audio = full_audio[start_ms:end_ms]
        l_s, l_e = get_speech_bounds(cropped_audio, buffer_ms=100)
        final_learner = cropped_audio[l_s:l_e]
        final_learner.export("temp_learner.wav", format="wav")
        full_audio.export("temp_stt.wav", format="wav")
        
        # 2. 원어민 오디오 생성 및 저장
        tts = gTTS(text=target_text, lang='en')
        tts.save("temp_native.mp3")
        native_raw = AudioSegment.from_file("temp_native.mp3", format="mp3")
        n_start, n_end = get_speech_bounds(native_raw, silence_thresh=-35, buffer_ms=0)
        final_native = native_raw[n_start:n_end]
        final_native.export("temp_native.wav", format="wav")

        y_learner, sr_l = librosa.load("temp_learner.wav", sr=22050)
        y_native, _ = librosa.load("temp_native.wav", sr=sr_l)

        learner_speech_dur = len(final_learner) / 1000.0
        native_speech_dur = len(final_native) / 1000.0

        # --- [추가] 탭 상단 공통 오디오 플레이어 ---
        st.divider()
        audio_col1, audio_col2 = st.columns(2)
        with audio_col1:
            st.write("🎙️ **나의 발음 (조정된 구간)**")
            st.audio("temp_learner.wav")
        with audio_col2:
            st.write("🔊 **원어민 발음**")
            st.audio("temp_native.wav")
        st.write("")

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
                    res_col1, res_col2 = st.columns([1, 2])
                    with res_col1:
                        st.markdown(f"""<div style="background-color: #e8f4f8; border-left: 5px solid #1f77b4; padding: 20px; border-radius: 8px; height: 120px;"><div style="color: #1f77b4; font-weight: bold;">정확도 점수</div><h1 style="margin: 0; color: #1f77b4;">{int(score * 100)}점</h1></div>""", unsafe_allow_html=True)
                    with res_col2:
                        st.markdown(f"""<div style="background-color: #eafaf1; border-left: 5px solid #2ecc71; padding: 20px; border-radius: 8px; height: 120px;"><div style="color: #27ae60; font-weight: bold;">AI 인식 결과</div><div style="font-size: 1.4rem; color: #1e8449; font-weight: 500;">{transcript}</div></div>""", unsafe_allow_html=True)
                except: st.error("인식 실패")

        with tab2:
            st.subheader("순수 발화 구간 분석 (Detected Pure Speech)")
            fig_dur, (ax_n, ax_l) = plt.subplots(2, 1, figsize=(12, 5))
            librosa.display.waveshow(y_native, sr=sr_l, ax=ax_n, color='lightgray', alpha=0.5)
            ax_n.axvline(x=0, color='red', linestyle='--'); ax_n.axvline(x=native_speech_dur, color='red', linestyle='--')
            ax_n.set_title(f"Native Speaker (Pure: {native_speech_dur:.2f}s)")
            librosa.display.waveshow(y_learner, sr=sr_l, ax=ax_l, color='skyblue', alpha=0.7)
            ax_l.axvline(x=0, color='blue', linestyle='--'); ax_l.axvline(x=learner_speech_dur, color='blue', linestyle='--')
            ax_l.set_title(f"Learner (Pure: {learner_speech_dur:.2f}s)")
            plt.tight_layout(); st.pyplot(fig_dur)
            
            diff_ratio = ((learner_speech_dur / native_speech_dur) - 1) * 100
            diff_text = f"{'+' if diff_ratio >= 0 else ''}{int(diff_ratio)}%"
            
            if abs(diff_ratio) <= 10:
                st.success(f"✅ **Optimal: 원어민과 거의 유사한 속도입니다. ({diff_text})**")
            elif 10 < diff_ratio <= 25:
                st.info(f"🟢 **Acceptable: 명확하고 자연스러운 속도입니다. ({diff_text})**")
            elif diff_ratio > 25:
                st.warning(f"🟠 **Slow: 조금 더 연음(Linking)을 활용해 빠르게 읽어보세요. ({diff_text})**")
            else:
                st.error(f"🔴 **Too Fast: 발음이 뭉개질 수 있으니 조금만 천천히 읽어보세요. ({diff_text})**")

            with st.expander("📚 발화 속도 해석 근거 및 참고문헌"):
                st.markdown("""
                **1. 해석 가이드라인:**
                * **±10% 이내:** 원어민 수준의 유창성 (Native-like Fluency).
                * **+10% ~ +25%:** 이해 가능한 수준의 안정적인 발화 (Intelligible).
                * **+25% 초과:** 과도한 휴지(Pause) 혹은 음소 연장으로 인한 유창성 저하.
                
                **2. 참고 문헌 및 표준 지침:**
                * **American Council on the Teaching of Foreign Languages (2012).** *ACTFL Proficiency Guidelines 2012*.
                * **Munro, M. J., & Derwing, T. M. (1995).** Foreign accent, comprehensibility, and intelligibility: Evidence from L2 learners. *Language Learning*.
                * **Derwing, T. M., & Munro, M. J. (2001).** What makes accent-free speakers? *Journal of Phonetics*.
                """)

        with tab3:
            fig_wave, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4))
            librosa.display.waveshow(y_native, sr=sr_l, ax=ax1, color='lightgray')
            librosa.display.waveshow(y_learner, sr=sr_l, ax=ax2, color='skyblue')
            plt.tight_layout(); st.pyplot(fig_wave)

        with tab4:
            f0_l, v_l, p_l = librosa.pyin(y_learner, fmin=75, fmax=400, hop_length=128)
            f0_n, v_n, p_n = librosa.pyin(y_native, fmin=60, fmax=400, hop_length=128)
            f0_l_filtered = np.where(v_l & (p_l > 0.25) & (f0_l > 80), f0_l, np.nan)
            f0_n_filtered = np.where(v_n & (p_n > 0.05), f0_n, np.nan)
            fig_pitch, (ax_n, ax_l) = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
            ax_n.plot(librosa.times_like(f0_n, hop_length=128), f0_n_filtered, color='lightgray', linewidth=3)
            ax_l.plot(librosa.times_like(f0_l, hop_length=128), f0_l_filtered, color='#1f77b4', linewidth=2.5)
            st.pyplot(fig_pitch)

    except Exception as e: st.error(f"오류: {e}")
    finally:
        for f in ["temp_native.mp3", "temp_native.wav", "temp_learner.wav", "temp_stt.wav", "temp_preview.wav"]:
            if os.path.exists(f): os.remove(f)
