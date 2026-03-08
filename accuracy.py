import streamlit as st
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import io, os, librosa, librosa.display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from difflib import SequenceMatcher

# --- 유틸리티 함수 ---
def get_speech_bounds(audio_segment, silence_thresh=-40, min_silence_len=100, buffer_ms=100):
    nonsilent_intervals = detect_nonsilent(audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    if not nonsilent_intervals: return 0, len(audio_segment)
    start_trim = max(0, nonsilent_intervals[0][0] - buffer_ms)
    end_trim = min(len(audio_segment), nonsilent_intervals[-1][1] + 50)
    return start_trim, end_trim

def normalize_pitch(f0):
    mu = np.nanmean(f0)
    sigma = np.nanstd(f0)
    return (f0 - mu) / sigma if sigma != 0 and not np.isnan(sigma) else f0 - mu

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

st.markdown("### 🎙️ AI 활용 발음 연습")

selected_level = st.selectbox("Step 1: 학습 단계를 선택하세요:", list(sample_sentences.keys()), 
                              on_change=lambda: st.session_state.update({"analysis_done": False}))
target_text = sample_sentences[selected_level]

col_box = st.columns([1, 2, 1])[1]
with col_box:
    st.markdown(f"""<div style="border: 2px solid #1f77b4; border-radius: 12px; padding: 15px; background-color: #f8f9fb; text-align: center; margin-bottom: 20px;"><h3 style="color: #1f77b4; margin: 0; font-weight: 700;">"{target_text}"</h3></div>""", unsafe_allow_html=True)
    rec_btn = st.columns([1, 1, 1])[1]
    with rec_btn:
        audio = mic_recorder(start_prompt="🎤 Step 2: 녹음 시작", stop_prompt="🛑 녹음 완료", key="recorder")

if audio:
    st.divider()
    audio_bytes = audio['bytes']
    audio_stream = io.BytesIO(audio_bytes)
    full_audio = AudioSegment.from_file(audio_stream)
    full_audio.export("temp_preview.wav", format="wav")
    duration_sec = len(full_audio) / 1000.0
    y_full, sr_f = librosa.load("temp_preview.wav", sr=22050)
    
    st.subheader("✂️ 발화 구간 및 줌 설정")
    
    c_zoom, c_input = st.columns([1, 1])
    with c_zoom:
        zoom_range = st.slider("🔍 파형 확대 범위 (Zoom Window):", 0.0, duration_sec, (0.0, duration_sec), step=0.01)
    
    with c_input:
        in_col1, in_col2 = st.columns(2)
        v_start_init, v_end_init = get_speech_bounds(full_audio)
        start_val = in_col1.number_input("시작 시간 (sec):", 0.0, duration_sec, float(v_start_init/1000), step=0.01, format="%.2f")
        end_val = in_col2.number_input("종료 시간 (sec):", 0.0, duration_sec, float(v_end_init/1000), step=0.01, format="%.2f")

    fig_p, ax = plt.subplots(figsize=(12, 3))
    librosa.display.waveshow(y_full, sr=sr_f, ax=ax, color='skyblue', alpha=0.6)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1)) # 0.1초 단위 눈금
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.axvline(x=start_val, color='red', linewidth=2); ax.axvline(x=end_val, color='red', linewidth=2)
    ax.set_xlim(zoom_range)
    st.pyplot(fig_p)
    st.audio(audio_bytes)
        
    if st.button("📊 Step 3: 설정된 구간으로 분석하기", use_container_width=True):
        st.session_state.analysis_done = True
        st.session_state.audio_bytes = audio_bytes
        st.session_state.start_time = start_val
        st.session_state.end_time = end_val

if st.session_state.analysis_done:
    try:
        audio_stream = io.BytesIO(st.session_state.audio_bytes)
        full_audio = AudioSegment.from_file(audio_stream)
        s_ms, e_ms = st.session_state.start_time * 1000, st.session_state.end_time * 1000
        cropped_audio = full_audio[s_ms:e_ms]
        l_s, l_e = get_speech_bounds(cropped_audio, buffer_ms=100)
        final_learner = cropped_audio[l_s:l_e]
        final_learner.export("temp_learner.wav", format="wav")
        full_audio.export("temp_stt.wav", format="wav")
        
        tts = gTTS(text=target_text, lang='en')
        tts.save("temp_native.mp3")
        native_raw = AudioSegment.from_file("temp_native.mp3", format="mp3")
        n_start, n_end = get_speech_bounds(native_raw, silence_thresh=-35, buffer_ms=0)
        final_native = native_raw[n_start:n_end]
        final_native.export("temp_native.wav", format="wav")

        y_learner, sr_l = librosa.load("temp_learner.wav", sr=22050)
        y_native, _ = librosa.load("temp_native.wav", sr=sr_l)
        l_dur, n_dur = len(final_learner)/1000.0, len(final_native)/1000.0

        st.divider()
        ac1, ac2 = st.columns(2)
        with ac1: st.write("🎙️ **나의 발음**"); st.audio("temp_learner.wav")
        with ac2: st.write("🔊 **원어민 발음**"); st.audio("temp_native.wav")

        tab1, tab2, tab3, tab4 = st.tabs(["🎯 AI 점수", "⏱️ 유창성 분석", "🔊 음파 대조", "📈 피치 분석"])

        with tab1:
            r = sr.Recognizer()
            with sr.AudioFile("temp_stt.wav") as source:
                r.adjust_for_ambient_noise(source, duration=0.5); data = r.record(source)
                try:
                    transcript = r.recognize_google(data, language='en-US')
                    score = SequenceMatcher(None, target_text.lower().replace('.',''), transcript.lower()).ratio()
                    c1, c2 = st.columns([1, 2])
                    with c1: st.markdown(f"""<div style="background-color: #e8f4f8; border-left: 5px solid #1f77b4; padding: 20px; border-radius: 8px; height: 120px;"><b>정확도</b><h1 style="color: #1f77b4;">{int(score*100)}점</h1></div>""", unsafe_allow_html=True)
                    with c2: st.markdown(f"""<div style="background-color: #eafaf1; border-left: 5px solid #2ecc71; padding: 20px; border-radius: 8px; height: 120px;"><b>인식 결과</b><p style="font-size: 1.2rem; color: #27ae60;">{transcript}</p></div>""", unsafe_allow_html=True)
                except: st.error("인식 실패")

        with tab2:
            fig_dur, (ax_n, ax_l) = plt.subplots(2, 1, figsize=(12, 5))
            librosa.display.waveshow(y_native, sr=sr_l, ax=ax_n, color='lightgray', alpha=0.5)
            ax_n.axvline(x=0, color='red', linestyle='--'); ax_n.axvline(x=n_dur, color='red', linestyle='--')
            librosa.display.waveshow(y_learner, sr=sr_l, ax=ax_l, color='skyblue', alpha=0.7)
            ax_l.axvline(x=0, color='blue', linestyle='--'); ax_l.axvline(x=l_dur, color='blue', linestyle='--')
            plt.tight_layout(); st.pyplot(fig_dur)
            diff = ((l_dur / n_dur) - 1) * 100
            st.info(f"💡 원어민 대비 발화 속도 편차: **{'+' if diff>=0 else ''}{int(diff)}%**")
            with st.expander("📚 참고문헌"):
                st.markdown("* Munro & Derwing (1995), ACTFL Proficiency Guidelines 2012.")

        with tab3:
            fig_w, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4))
            librosa.display.waveshow(y_native, sr=sr_l, ax=ax1, color='lightgray')
            librosa.display.waveshow(y_learner, sr=sr_l, ax=ax2, color='skyblue')
            plt.tight_layout(); st.pyplot(fig_w)

        with tab4:
            st.subheader("억양 멜로디 분석 (Pitch Contour)")
            # [핵심 수정] 피치 복구를 위해 임계값 완화 (p_l: 0.25 -> 0.15, p_n: 0.05 -> 0.01)
            f0_l, v_l, p_l = librosa.pyin(y_learner, fmin=75, fmax=400, hop_length=128)
            f0_n, v_n, p_n = librosa.pyin(y_native, fmin=60, fmax=400, hop_length=128)
            f0_l_f = np.where(v_l & (p_l > 0.15) & (f0_l > 80), f0_l, np.nan)
            f0_n_f = np.where(v_n & (p_n > 0.01), f0_n, np.nan)
            
            fig_p, (ax_n1, ax_l1) = plt.subplots(1, 2, figsize=(15, 4), sharey=True)
            ax_n1.plot(librosa.times_like(f0_n, hop_length=128), f0_n_f, color='lightgray', linewidth=3)
            ax_l1.plot(librosa.times_like(f0_l, hop_length=128), f0_l_f, color='#1f77b4', linewidth=2.5)
            ax_n1.set_title("Native Speaker"); ax_l1.set_title("Your Pitch")
            st.pyplot(fig_p)
            
            if st.checkbox("📈 패턴 대조(Normalized)"):
                fn_norm = normalize_pitch(f0_n_f); fl_norm = normalize_pitch(f0_l_f)
                fig_nm, axn = plt.subplots(figsize=(12, 4))
                axn.plot(librosa.times_like(f0_n, hop_length=128), fn_norm, color='lightgray', label='Native')
                axn.plot(librosa.times_like(f0_l, hop_length=128), fl_norm, color='#1f77b4', label='You')
                axn.legend(); plt.tight_layout(); st.pyplot(fig_nm)

    except Exception as e: st.error(f"오류: {e}")
    finally:
        for f in ["temp_native.mp3", "temp_native.wav", "temp_learner.wav", "temp_stt.wav", "temp_preview.wav"]:
            if os.path.exists(f): os.remove(f)
