import streamlit as st
from streamlit_mic_recorder import mic_recorder
import speech_recognition as speech_rec
import io, os, librosa, librosa.display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from difflib import SequenceMatcher
from scipy.stats import pearsonr
import re # 정규표현식 추가

# --- 1. 유틸리티 함수 ---
def get_speech_bounds(audio_segment, silence_thresh=-40, min_silence_len=100, buffer_ms=100):
    nonsilent_intervals = detect_nonsilent(audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    if not nonsilent_intervals: return 0, len(audio_segment)
    start_trim = max(0, nonsilent_intervals[0][0] - buffer_ms)
    end_trim = min(len(audio_segment), nonsilent_intervals[-1][1] + 50)
    return start_trim, end_trim

def normalize_pitch(f0):
    mu = np.nanmean(f0)
    sigma = np.nanstd(f0)
    if sigma == 0 or np.isnan(sigma): return np.zeros_like(f0)
    return (f0 - mu) / sigma

def calculate_intonation_score(f0_n, f0_l):
    """
    시간 축 불일치 문제를 해결하기 위해 두 곡선의 길이를 
    동일한 구간으로 리샘플링하여 비교합니다.
    """
    # 결측치(NaN)를 0으로 채움
    vec_n = np.nan_to_num(f0_n)
    vec_l = np.nan_to_num(f0_l)
    
    # 두 벡터 중 유효한 피치 데이터가 있는 구간만 추출
    vec_n = vec_n[np.where(vec_n != 0)]
    vec_l = vec_l[np.where(vec_l != 0)]
    
    if len(vec_n) < 10 or len(vec_l) < 10:
        return 0

    # [핵심] 두 곡선의 길이를 강제로 맞춤 (Interpolation)
    from scipy.interpolate import interp1d
    x_new = np.linspace(0, 1, 100) # 100개 포인트로 표준화
    
    f_n = interp1d(np.linspace(0, 1, len(vec_n)), vec_n, kind='linear')
    f_l = interp1d(np.linspace(0, 1, len(vec_l)), vec_l, kind='linear')
    
    norm_vec_n = f_n(x_new)
    norm_vec_l = f_l(x_new)
    
    # 상관계수 계산
    with np.errstate(divide='ignore', invalid='ignore'):
        corr, _ = pearsonr(norm_vec_n, norm_vec_l)
        
    # 단순히 0 이하를 0점으로 처리하기보다, 절대적인 패턴 유사도를 위해 보정
    if np.isnan(corr): return 0
    
    # 억양의 흐름(올라가고 내려감)이 비슷하면 점수를 부여
    score = int(max(0, corr) * 100)
    return score

# --- 2. 설정 및 세션 초기화 ---
st.set_page_config(page_title="AI 발음 분석기", layout="wide")
if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False

sample_sentences = {
    "Level 01: (인사/기초)": "I am on my way.",
    "Level 02: (일상/기초)": "Nice room you have.",
    "Level 03: (일상/기초)": "Dinner is ready now.",
}

st.markdown("### 🎙️ AI 활용 발음 연습")
selected_level = st.selectbox("Step 1: 학습 단계를 선택하세요:", list(sample_sentences.keys()))
target_text = sample_sentences.get(selected_level)

col_box = st.columns([1, 2, 1])[1]
with col_box:
    st.markdown(f"""<div style="border: 2px solid #1f77b4; border-radius: 12px; padding: 15px; background-color: #f8f9fb; text-align: center; margin-bottom: 20px;">
                <h3 style="color: #1f77b4; margin: 0; font-weight: 700;">"{target_text}"</h3></div>""", unsafe_allow_html=True)
    rec_btn = st.columns([1, 1, 1])[1]
    with rec_btn:
        audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 녹음 완료", key="recorder")

# --- 3. 녹음 후 구간 설정 ---
if audio:
    st.divider()
    audio_bytes = audio['bytes']
    audio_stream = io.BytesIO(audio_bytes)
    full_audio = AudioSegment.from_file(audio_stream)
    full_audio.export("temp_preview.wav", format="wav")
    duration_sec = len(full_audio) / 1000.0
    y_full, sr_f = librosa.load("temp_preview.wav", sr=22050)
    
    st.subheader("✂️ 발화 구간 설정")
    v_s, v_e = get_speech_bounds(full_audio)
    trim_range = st.slider("분석할 목소리 구간을 선택하세요 (초):", 
                           0.0, duration_sec, (float(v_s/1000), float(v_e/1000)), step=0.01)

    fig_prev, ax = plt.subplots(figsize=(12, 3))
    librosa.display.waveshow(y_full, sr=sr_f, ax=ax, color='skyblue', alpha=0.6)
    ax.axvline(x=trim_range[0], color='red', lw=2)
    ax.axvline(x=trim_range[1], color='red', lw=2)
    ax.set_xlim(max(0, trim_range[0] - 0.2), min(duration_sec, trim_range[1] + 0.2))
    st.pyplot(fig_prev)
    st.audio(audio_bytes)
        
    if st.button("📊 설정된 구간으로 분석 시작하기", use_container_width=True):
        st.session_state.analysis_done = True
        st.session_state.final_audio_bytes = audio_bytes
        st.session_state.final_start = trim_range[0]
        st.session_state.final_end = trim_range[1]

# --- 4. 상세 분석 결과 ---
if st.session_state.analysis_done:
    try:
        audio_stream = io.BytesIO(st.session_state.final_audio_bytes)
        full_audio = AudioSegment.from_file(audio_stream)
        s_ms, e_ms = st.session_state.final_start * 1000, st.session_state.final_end * 1000
        cropped = full_audio[s_ms:e_ms]
        
        l_s, l_e = get_speech_bounds(cropped, buffer_ms=50)
        final_l = cropped[l_s:l_e]; final_l.export("temp_l.wav", format="wav")
        full_audio.export("temp_stt.wav", format="wav")
        
        tts = gTTS(text=target_text, lang='en'); tts.save("temp_n.mp3")
        native_raw = AudioSegment.from_file("temp_n.mp3")
        n_s, n_e = get_speech_bounds(native_raw, silence_thresh=-35)
        final_n = native_raw[n_s:n_e]; final_n.export("temp_n.wav", format="wav")

        y_l, sr_curr = librosa.load("temp_l.wav", sr=22050); y_n, _ = librosa.load("temp_n.wav", sr=sr_curr)
        l_dur, n_dur = len(final_l)/1000.0, len(final_n)/1000.0

        st.divider()
        ac1, ac2 = st.columns(2)
        with ac1: st.write("🎙️ **나의 발음**"); st.audio("temp_l.wav")
        with ac2: st.write("🔊 **원어민 발음**"); st.audio("temp_n.wav")

        tab1, tab2, tab3, tab4 = st.tabs(["🎯 AI 점수", "⏱️ 유창성 분석", "🔊 음파 대조", "📈 피치 분석"])

        with tab1:
            recognizer_obj = speech_rec.Recognizer()
            with speech_rec.AudioFile("temp_stt.wav") as source:
                audio_data = recognizer_obj.record(source)
                try:
                    text_res = recognizer_obj.recognize_google(audio_data, language='en-US')
                    
                    # [핵심 추가] 전처리 로직: 대소문자 통합 및 문장부호 제거
                    def clean_string(s):
                        return re.sub(r'[^\w\s]', '', s).lower().strip()
                    
                    clean_target = clean_string(target_text)
                    clean_result = clean_string(text_res)
                    
                    # 전처리된 텍스트로 유사도 측정
                    sim_val = SequenceMatcher(None, clean_target, clean_result).ratio()
                    
                    # 98점 이상이면 사실상 일치하므로 100점으로 보정
                    final_acc_score = 100 if sim_val > 0.98 else int(sim_val * 100)
                    
                    c1, c2 = st.columns([1, 2])
                    with c1: 
                        st.markdown(f"""<div style="background-color: #e8f4f8; border-left: 5px solid #1f77b4; padding: 20px; border-radius: 8px; height: 120px;">
                                    <b>정확도 점수</b><h1 style="color: #1f77b4; margin:0;">{final_acc_score}점</h1></div>""", unsafe_allow_html=True)
                    with c2: 
                        st.markdown(f"""<div style="background-color: #eafaf1; border-left: 5px solid #2ecc71; padding: 20px; border-radius: 8px; height: 120px;">
                                    <b>인식 결과</b><p style="font-size: 1.2rem; color: #27ae60; margin:0;">{text_res}</p></div>""", unsafe_allow_html=True)
                except: st.error("인식 실패")

        with tab2:
            fig_dur, (ax_l, ax_n) = plt.subplots(2, 1, figsize=(12, 5))
            librosa.display.waveshow(y_l, sr=sr_curr, ax=ax_l, color='skyblue')
            librosa.display.waveshow(y_n, sr=sr_curr, ax=ax_n, color='lightgray')
            plt.tight_layout(); st.pyplot(fig_dur)
            diff = ((l_dur / n_dur) - 1) * 100
            st.info(f"💡 발화 속도 편차: **{'+' if diff>=0 else ''}{int(diff)}%**")

        with tab3:
            fig_w, (axw1, axw2) = plt.subplots(2, 1, figsize=(12, 6))
            librosa.display.waveshow(y_l, sr=sr_curr, ax=axw1, color='skyblue')
            librosa.display.waveshow(y_n, sr=sr_curr, ax=axw2, color='lightgray')
            plt.tight_layout(); st.pyplot(fig_w)

        with tab4:
            st.subheader("억양 멜로디 분석 (Pitch Contour)")
            f0_l, v_l, p_l = librosa.pyin(y_l, fmin=75, fmax=400, hop_length=64)
            f0_n, v_n, p_n = librosa.pyin(y_n, fmin=60, fmax=400, hop_length=64)
            f0_l_f = np.where(v_l & (p_l > 0.1), f0_l, np.nan)
            f0_n_f = np.where(v_n & (p_n > 0.01), f0_n, np.nan)
            
            t_l = librosa.times_like(f0_l, sr=sr_curr, hop_length=64); t_n = librosa.times_like(f0_n, sr=sr_curr, hop_length=64)
            fig_p, (ax_l1, ax_n1) = plt.subplots(1, 2, figsize=(15, 4), sharey=True)
            ax_l1.plot(t_l, f0_l_f, color='#1f77b4', ls=':', marker='o', markersize=2)
            ax_n1.plot(t_n, f0_n_f, color='gray', ls=':', marker='o', markersize=2)
            st.pyplot(fig_p)

            st.write("---")
            if st.button("🚀 정규화 분석 및 피드백 실행", use_container_width=True):
                fl_norm = normalize_pitch(f0_l_f); fn_norm = normalize_pitch(f0_n_f)
                score = calculate_intonation_score(fn_norm, fl_norm)
                fig_ov, axo = plt.subplots(figsize=(12, 4))
                axo.plot(t_n[:len(fn_norm)], fn_norm, color='lightgray', lw=3, label='Native', alpha=0.7)
                axo.plot(t_l[:len(fl_norm)], fl_norm, color='#1f77b4', lw=2, label='Learner')
                axo.set_title("Melody Pattern Overlay"); axo.legend(); st.pyplot(fig_ov)
                st.metric("억양 유사도 점수", f"{score}점")

    except Exception as e: st.error(f"분석 중 오류 발생: {e}")
    finally:
        for f in ["temp_n.mp3", "temp_n.wav", "temp_l.wav", "temp_stt.wav", "temp_preview.wav"]:
            if os.path.exists(f): os.remove(f)
