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
from scipy.stats import pearsonr

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
    return (f0 - mu) / sigma if (sigma != 0 and not np.isnan(sigma)) else np.zeros_like(f0)

def calculate_intonation_score(f0_n, f0_l):
    min_len = min(len(f0_n), len(f0_l))
    if min_len < 5: return 0
    vec_n = np.nan_to_num(f0_n[:min_len])
    vec_l = np.nan_to_num(f0_l[:min_len])
    with np.errstate(divide='ignore', invalid='ignore'):
        corr, _ = pearsonr(vec_n, vec_l)
    return int(max(0, corr) * 100) if not np.isnan(corr) else 0

# --- 위젯 연동 콜백 ---
def update_slider():
    st.session_state.zoom_range = (st.session_state.start_val, st.session_state.end_val)

def update_num_input():
    st.session_state.start_val = st.session_state.zoom_range[0]
    st.session_state.end_val = st.session_state.zoom_range[1]

# --- 스트림릿 설정 ---
st.set_page_config(page_title="AI 발음 분석기", layout="wide")

if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
if 'start_val' not in st.session_state: st.session_state.start_val = 0.0
if 'end_val' not in st.session_state: st.session_state.end_val = 1.0

sample_sentences = {
    "Level 01: (인사/기초)": "I am on my way.",
    "Level 02: (일상/기초)": "Nice room you have.",
    "Level 03: (일상/기초)": "Dinner is ready now.",
}

st.markdown("### 🎙️ AI 활용 발음 연습")

selected_level = st.selectbox("Step 1: 학습 단계를 선택하세요:", list(sample_sentences.keys()))
target_text = sample_sentences.get(selected_level, "I am on my way.")

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
    
    if 'v_detected' not in st.session_state:
        v_s, v_e = get_speech_bounds(full_audio)
        st.session_state.start_val = float(v_s/1000)
        st.session_state.end_val = float(v_e/1000)
        st.session_state.zoom_range = (0.0, duration_sec)
        st.session_state.v_detected = True

    st.subheader("✂️ 발화 구간 및 줌 설정")
    c_zoom, c_input = st.columns([1, 1])
    with c_zoom:
        st.slider("🔍 파형 확대 범위 (Zoom):", 0.0, duration_sec, key="zoom_range", on_change=update_num_input, step=0.01)
    with c_input:
        in_col1, in_col2 = st.columns(2)
        in_col1.number_input("시작 (sec):", 0.0, duration_sec, key="start_val", on_change=update_slider, step=0.01, format="%.2f")
        in_col2.number_input("종료 (sec):", 0.0, duration_sec, key="end_val", on_change=update_slider, step=0.01, format="%.2f")

    fig_p, ax = plt.subplots(figsize=(12, 3.5))
    librosa.display.waveshow(y_full, sr=sr_f, ax=ax, color='skyblue', alpha=0.6)
    ax.set_xlim(st.session_state.zoom_range)
    ax.axvline(x=st.session_state.start_val, color='red', linewidth=2)
    ax.axvline(x=st.session_state.end_val, color='red', linewidth=2)
    st.pyplot(fig_p)
    st.audio(audio_bytes)
        
    if st.button("📊 Step 3: 설정된 구간으로 분석하기", use_container_width=True):
        st.session_state.analysis_done = True
        st.session_state.final_audio_bytes = audio_bytes
        st.session_state.final_start = st.session_state.start_val
        st.session_state.final_end = st.session_state.end_val

if st.session_state.analysis_done:
    try:
        audio_stream = io.BytesIO(st.session_state.final_audio_bytes)
        full_audio = AudioSegment.from_file(audio_stream)
        s_ms, e_ms = st.session_state.final_start * 1000, st.session_state.final_end * 1000
        cropped_audio = full_audio[s_ms:e_ms]
        l_s, l_e = get_speech_bounds(cropped_audio, buffer_ms=50)
        final_learner = cropped_audio[l_s:l_e]
        final_learner.export("temp_learner.wav", format="wav")
        full_audio.export("temp_stt.wav", format="wav")
        
        tts = gTTS(text=target_text, lang='en'); tts.save("temp_native.mp3")
        native_raw = AudioSegment.from_file("temp_native.mp3", format="mp3")
        n_s, n_e = get_speech_bounds(native_raw, silence_thresh=-35, buffer_ms=0)
        final_native = native_raw[n_s:n_e]; final_native.export("temp_native.wav", format="wav")

        y_l, sr_l = librosa.load("temp_learner.wav", sr=22050); y_n, _ = librosa.load("temp_native.wav", sr=sr_l)
        l_dur, n_dur = len(final_learner)/1000.0, len(final_native)/1000.0

        st.divider()
        ac1, ac2 = st.columns(2)
        with ac1: st.write("🎙️ **나의 발음**"); st.audio("temp_learner.wav")
        with ac2: st.write("🔊 **원어민 발음**"); st.audio("temp_native.wav")

        tab1, tab2, tab3, tab4 = st.tabs(["🎯 AI 점수", "⏱️ 유창성 분석", "🔊 음파 대조", "📈 피치 분석"])

        with tab1:
            # [해결] r 대신 구체적인 변수명 recognizer 사용
            recognizer = sr.Recognizer()
            with sr.AudioFile("temp_stt.wav") as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = recognizer.record(source)
                try:
                    transcript = recognizer.recognize_google(audio_data, language='en-US')
                    score_ratio = SequenceMatcher(None, target_text.lower().replace('.',''), transcript.lower()).ratio()
                    c1, c2 = st.columns([1, 2])
                    with c1: st.markdown(f"""<div style="background-color: #e8f4f8; border-left: 5px solid #1f77b4; padding: 20px; border-radius: 8px; height: 120px;"><b>정확도</b><h1 style="color: #1f77b4;">{int(score_ratio*100)}점</h1></div>""", unsafe_allow_html=True)
                    with c2: st.markdown(f"""<div style="background-color: #eafaf1; border-left: 5px solid #2ecc71; padding: 20px; border-radius: 8px; height: 120px;"><b>인식 결과</b><p style="font-size: 1.2rem; color: #27ae60;">{transcript}</p></div>""", unsafe_allow_html=True)
                except: st.error("인식 실패")

        with tab2:
            fig_dur, (ax_l, ax_n) = plt.subplots(2, 1, figsize=(12, 5))
            librosa.display.waveshow(y_l, sr=sr_l, ax=ax_l, color='skyblue', alpha=0.7)
            librosa.display.waveshow(y_n, sr=sr_l, ax=ax_n, color='lightgray', alpha=0.5)
            plt.tight_layout(); st.pyplot(fig_dur)
            diff = ((l_dur / n_dur) - 1) * 100
            st.info(f"💡 원어민 대비 발화 속도 편차: **{'+' if diff>=0 else ''}{int(diff)}%**")

        with tab4:
            st.subheader("억양 및 멜로디 분석")
            f0_l, v_l, p_l = librosa.pyin(y_l, fmin=75, fmax=400, hop_length=64)
            f0_n, v_n, p_n = librosa.pyin(y_n, fmin=60, fmax=400, hop_length=64)
            f0_l_f = np.where(v_l & (p_l > 0.15) & (f0_l > 80), f0_l, np.nan)
            f0_n_f = np.where(v_n & (p_n > 0.01), f0_n, np.nan)
            
            fig_p, (ax_l1, ax_n1) = plt.subplots(1, 2, figsize=(15, 4), sharey=True)
            t_l = librosa.times_like(f0_l, sr=sr_l, hop_length=64); t_n = librosa.times_like(f0_n, sr=sr_l, hop_length=64)
            ax_l1.plot(t_l, f0_l_f, color='#1f77b4', linestyle=':', marker='o', markersize=2)
            ax_n1.plot(t_n, f0_n_f, color='lightgray', linestyle=':', marker='o', markersize=2)
            st.pyplot(fig_p)

            fn_norm = normalize_pitch(f0_n_f); fl_norm = normalize_pitch(f0_l_f)
            into_score = calculate_intonation_score(fn_norm, fl_norm)
            st.write("---")
            st.markdown(f"#### 📊 억양 유사도 점수: **{into_score}점**")
            
            if into_score >= 65: st.success("👍 Good 억양입니다.")
            else: st.warning("🧐 억양 연습이 더 필요합니다.")

    except Exception as e: st.error(f"오류: {e}")
    finally:
        for f in ["temp_native.mp3", "temp_native.wav", "temp_learner.wav", "temp_stt.wav", "temp_preview.wav"]:
            if os.path.exists(f): os.remove(f)
