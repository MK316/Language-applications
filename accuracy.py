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

# --- 1. 유틸리티 함수 ---
def get_speech_bounds(audio_segment, silence_thresh=-40, min_silence_len=100, buffer_ms=100):
    nonsilent_intervals = detect_nonsilent(audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    if not nonsilent_intervals: return 0, len(audio_segment)
    start_trim = max(0, nonsilent_intervals[0][0] - buffer_ms)
    end_trim = min(len(audio_segment), nonsilent_intervals[-1][1] + 50)
    return start_trim, end_trim

def normalize_pitch(f0):
    """Z-score 정규화: 개인별 음역대 차이를 제거하여 패턴만 비교"""
    mu = np.nanmean(f0)
    sigma = np.nanstd(f0)
    if sigma == 0 or np.isnan(sigma): return np.zeros_like(f0)
    return (f0 - mu) / sigma

def calculate_intonation_score(f0_n, f0_l):
    """정규화된 곡선 간의 상관계수 계산"""
    min_len = min(len(f0_n), len(f0_l))
    if min_len < 5: return 0
    vec_n = np.nan_to_num(f0_n[:min_len])
    vec_l = np.nan_to_num(f0_l[:min_len])
    with np.errstate(divide='ignore', invalid='ignore'):
        corr, _ = pearsonr(vec_n, vec_l)
    return int(max(0, corr) * 100) if not np.isnan(corr) else 0

# --- 2. 세션 상태 초기화 ---
if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
if 'v_detected' not in st.session_state: st.session_state.v_detected = False
if 'start_val' not in st.session_state: st.session_state.start_val = 0.0
if 'end_val' not in st.session_state: st.session_state.end_val = 1.0

# --- 3. UI 설정 ---
st.set_page_config(page_title="AI 발음 분석기", layout="wide")
st.markdown("### 🎙️ AI 활용 발음 연습 (학습자 중심 분석)")

sample_sentences = {
    "Level 01: (인사/기초)": "I am on my way.",
    "Level 02: (일상/기초)": "Nice room you have.",
    "Level 03: (일상/기초)": "Dinner is ready now.",
}

selected_level = st.selectbox("Step 1: 학습 단계를 선택하세요:", list(sample_sentences.keys()))
target_text = sample_sentences.get(selected_level)

col_box = st.columns([1, 2, 1])[1]
with col_box:
    st.markdown(f"""<div style="border: 2px solid #1f77b4; border-radius: 12px; padding: 15px; background-color: #f8f9fb; text-align: center; margin-bottom: 20px;">
                <h3 style="color: #1f77b4; margin: 0; font-weight: 700;">"{target_text}"</h3></div>""", unsafe_allow_html=True)
    rec_btn = st.columns([1, 1, 1])[1]
    with rec_btn:
        audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 녹음 완료", key="recorder")

# --- 4. 녹음 후 즉시 실행 로직 ---
if audio:
    audio_bytes = audio['bytes']
    audio_stream = io.BytesIO(audio_bytes)
    full_audio = AudioSegment.from_file(audio_stream)
    full_audio.export("temp_preview.wav", format="wav")
    duration_sec = len(full_audio) / 1000.0
    y_full, sr_f = librosa.load("temp_preview.wav", sr=22050)
    
    # 음성 구간 자동 감지 (최초 1회)
    if not st.session_state.v_detected:
        v_s, v_e = get_speech_bounds(full_audio)
        st.session_state.start_val = float(v_s/1000)
        st.session_state.end_val = float(v_e/1000)
        st.session_state.zoom_range = (0.0, duration_sec)
        st.session_state.v_detected = True

    st.subheader("✂️ 발화 구간 설정")
    c1, c2 = st.columns([2, 1])
    with c1:
        # 슬라이더 조작 시 세션 상태 업데이트
        st.session_state.zoom_range = st.slider("🔍 확대 범위:", 0.0, duration_sec, value=st.session_state.zoom_range, step=0.01)
    with c2:
        st.session_state.start_val = st.number_input("시작(s):", 0.0, duration_sec, value=st.session_state.start_val, step=0.01)
        st.session_state.end_val = st.number_input("종료(s):", 0.0, duration_sec, value=st.session_state.end_val, step=0.01)

    # 파형 자동 렌더링
    fig_prev, ax = plt.subplots(figsize=(12, 3))
    librosa.display.waveshow(y_full, sr=sr_f, ax=ax, color='skyblue', alpha=0.6)
    ax.axvline(x=st.session_state.start_val, color='red', lw=2)
    ax.axvline(x=st.session_state.end_val, color='red', lw=2)
    ax.set_xlim(st.session_state.zoom_range)
    st.pyplot(fig_prev)
    st.audio(audio_bytes)
        
    if st.button("📊 분석 시작하기", use_container_width=True):
        st.session_state.analysis_done = True
        st.session_state.final_audio_bytes = audio_bytes
        st.session_state.final_start = st.session_state.start_val
        st.session_state.final_end = st.session_state.end_val

# --- 5. 상세 분석 탭 ---
if st.session_state.analysis_done:
    try:
        # 파일 준비
        audio_stream = io.BytesIO(st.session_state.final_audio_bytes)
        full_audio = AudioSegment.from_file(audio_stream)
        s_ms, e_ms = st.session_state.final_start * 1000, st.session_state.final_end * 1000
        cropped = full_audio[s_ms:e_ms]
        l_s, l_e = get_speech_bounds(cropped, buffer_ms=50)
        final_learner = cropped[l_s:l_e]
        final_learner.export("temp_l.wav", format="wav")
        full_audio.export("temp_stt.wav", format="wav")
        
        tts = gTTS(text=target_text, lang='en'); tts.save("temp_n.mp3")
        native_raw = AudioSegment.from_file("temp_n.mp3")
        n_s, n_e = get_speech_bounds(native_raw, silence_thresh=-35)
        final_native = native_raw[n_s:n_e]; final_native.export("temp_n.wav", format="wav")

        y_l, sr = librosa.load("temp_l.wav", sr=22050); y_n, _ = librosa.load("temp_n.wav", sr=sr)
        l_dur, n_dur = len(final_learner)/1000.0, len(final_native)/1000.0

        st.divider()
        ac1, ac2 = st.columns(2)
        with ac1: st.write("🎙️ **나의 발음**"); st.audio("temp_l.wav")
        with ac2: st.write("🔊 **원어민 발음**"); st.audio("temp_n.wav")

        tab1, tab2, tab3, tab4 = st.tabs(["🎯 정확도", "⏱️ 유창성", "🔊 음파", "📈 억양 분석"])

        with tab1:
            recognizer = sr.Recognizer()
            with sr.AudioFile("temp_stt.wav") as source:
                audio_data = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio_data, language='en-US')
                    sim = SequenceMatcher(None, target_text.lower(), text.lower()).ratio()
                    st.metric("발음 정확도", f"{int(sim*100)}점")
                    st.success(f"인식된 문장: {text}")
                except: st.warning("인식 실패")

        with tab2:
            fig_dur, (ax_l, ax_n) = plt.subplots(2, 1, figsize=(10, 4))
            librosa.display.waveshow(y_l, sr=sr, ax=ax_l, color='skyblue')
            librosa.display.waveshow(y_n, sr=sr, ax=ax_n, color='lightgray')
            ax_l.set_title("Your Voice"); ax_n.set_title("Native Speaker")
            plt.tight_layout(); st.pyplot(fig_dur)
            st.info(f"발화 속도: 나({l_dur:.2f}s) vs 원어민({n_dur:.2f}s)")

        with tab3:
            fig_w, axw = plt.subplots(figsize=(10, 3))
            librosa.display.waveshow(y_l, sr=sr, ax=axw, color='skyblue', alpha=0.7, label='You')
            librosa.display.waveshow(y_n, sr=sr, ax=axw, color='lightgray', alpha=0.5, label='Native')
            axw.legend(); st.pyplot(fig_w)

        with tab4:
            st.subheader("억양(Pitch) 패턴 정밀 진단")
            f0_l, v_l, p_l = librosa.pyin(y_l, fmin=75, fmax=400, hop_length=64)
            f0_n, v_n, p_n = librosa.pyin(y_n, fmin=60, fmax=400, hop_length=64)
            f0_l_f = np.where(v_l & (p_l > 0.1), f0_l, np.nan)
            f0_n_f = np.where(v_n & (p_n > 0.1), f0_n, np.nan)
            
            t_l = librosa.times_like(f0_l, sr=sr, hop_length=64); t_n = librosa.times_like(f0_n, sr=sr, hop_length=64)
            fig_p, (ax_l1, ax_n1) = plt.subplots(1, 2, figsize=(12, 3), sharey=True)
            ax_l1.plot(t_l, f0_l_f, color='#1f77b4', marker='o', markersize=2, ls=':'); ax_l1.set_title("Your Pitch")
            ax_n1.plot(t_n, f0_n_f, color='gray', marker='o', markersize=2, ls=':'); ax_n1.set_title("Native Pitch")
            st.pyplot(fig_p)

            st.write("---")
            if st.button("🚀 정규화 분석 및 피드백 생성", use_container_width=True):
                fl_norm = normalize_pitch(f0_l_f); fn_norm = normalize_pitch(f0_n_f)
                score = calculate_intonation_score(fn_norm, fl_norm)
                
                fig_ov, axo = plt.subplots(figsize=(12, 4))
                axo.plot(t_n[:len(fn_norm)], fn_norm, color='lightgray', lw=3, label='Native', alpha=0.7)
                axo.plot(t_l[:len(fl_norm)], fl_norm, color='#1f77b4', lw=2, label='You')
                axo.set_title("Normalized Melody Overlay"); axo.legend(); st.pyplot(fig_ov)
                
                st.markdown(f"#### 🎯 억양 유사도: **{score}점**")
                if score >= 80: st.success("🌟 완벽한 억양입니다!")
                elif score >= 50: st.info("👍 리듬감이 좋습니다. 강조점 위주로 다듬어보세요.")
                else: st.warning("🧐 억양의 높낮이 변화를 더 크게 주어 연습해보세요.")

    except Exception as e: st.error(f"오류 발생: {e}")
    finally:
        for f in ["temp_n.mp3", "temp_n.wav", "temp_l.wav", "temp_stt.wav", "temp_preview.wav"]:
            if os.path.exists(f): os.remove(f)
