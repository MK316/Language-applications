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
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

# --- 1. 유틸리티 및 분석 함수 ---
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
    """시간축 정규화 및 피크 일치도 기반 세분화 점수 산출"""
    vec_n = f0_n[~np.isnan(f0_n)]
    vec_l = f0_l[~np.isnan(f0_l)]
    
    if len(vec_n) < 10 or len(vec_l) < 10: return 0

    # [Step 1] 시간축 정규화 (100 포인트로 통일)
    target_pts = 100
    x_n = np.linspace(0, 1, len(vec_n))
    x_l = np.linspace(0, 1, len(vec_l))
    x_new = np.linspace(0, 1, target_pts)
    
    norm_n = interp1d(x_n, vec_n, kind='linear')(x_new)
    norm_l = interp1d(x_l, vec_l, kind='linear')(x_new)
    
    # [Step 2] 값 정규화 (Z-score)
    z_n = (norm_n - np.mean(norm_n)) / (np.std(norm_n) + 1e-6)
    z_l = (norm_l - np.mean(norm_l)) / (np.std(norm_l) + 1e-6)

    # [Step 3] 피크(강세) 감지 (가중치 40%)
    peaks_n, _ = find_peaks(z_n, distance=10, prominence=0.5)
    peaks_l, _ = find_peaks(z_l, distance=10, prominence=0.5)
    
    peak_score = 0
    if len(peaks_n) > 0:
        match_count = sum(1 for p_n in peaks_n if any(abs(p_n - p_l) <= 10 for p_l in peaks_l))
        peak_score = (match_count / len(peaks_n)) * 100
    
    # [Step 4] 전체 패턴 상관관계 (가중치 60%)
    corr, _ = pearsonr(z_n, z_l)
    pattern_score = max(0, corr) * 100 if not np.isnan(corr) else 0
    
    return int((pattern_score * 0.6) + (peak_score * 0.4))

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

# --- 3. 녹음 및 구간 설정 ---
col_box = st.columns([1, 2, 1])[1]
with col_box:
    st.markdown(f"""<div style="border: 2px solid #1f77b4; border-radius: 12px; padding: 15px; background-color: #f8f9fb; text-align: center; margin-bottom: 20px;">
                <h3 style="color: #1f77b4; margin: 0; font-weight: 700;">"{target_text}"</h3></div>""", unsafe_allow_html=True)
    audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 녹음 완료", key="recorder")

if audio:
    st.divider()
    audio_bytes = audio['bytes']
    y_full, sr_f = librosa.load(io.BytesIO(audio_bytes), sr=22050)
    duration_sec = len(y_full) / sr_f
    
    st.subheader("✂️ 발화 구간 설정")
    v_s_idx, v_e_idx = get_speech_bounds(AudioSegment.from_file(io.BytesIO(audio_bytes)))
    trim_range = st.slider("분석할 목소리 구간:", 0.0, duration_sec, (float(v_s_idx/1000), float(v_e_idx/1000)), step=0.01)

    fig_prev, ax = plt.subplots(figsize=(12, 3))
    librosa.display.waveshow(y_full, sr=sr_f, ax=ax, color='skyblue', alpha=0.6)
    ax.axvline(x=trim_range[0], color='red', lw=2); ax.axvline(x=trim_range[1], color='red', lw=2)
    ax.set_xlim(max(0, trim_range[0] - 0.2), min(duration_sec, trim_range[1] + 0.2))
    st.pyplot(fig_prev)
    
    if st.button("📊 설정된 구간으로 분석 시작하기", use_container_width=True):
        st.session_state.analysis_done = True
        st.session_state.final_audio_bytes = audio_bytes
        st.session_state.final_start = trim_range[0]
        st.session_state.final_end = trim_range[1]

# --- 4. 상세 분석 결과 ---
if st.session_state.analysis_done:
    try:
        # 파일 경로 안전하게 관리
        path_l = "temp_l.wav"; path_n_mp3 = "temp_n.mp3"; path_n_wav = "temp_n.wav"; path_stt = "temp_stt.wav"
        
        full_audio = AudioSegment.from_file(io.BytesIO(st.session_state.final_audio_bytes))
        s_ms, e_ms = st.session_state.final_start * 1000, st.session_state.final_end * 1000
        cropped = full_audio[s_ms:e_ms]
        
        # 학습자 파일 저장
        l_s, l_e = get_speech_bounds(cropped, buffer_ms=50)
        final_l = cropped[l_s:l_e]; final_l.export(path_l, format="wav")
        full_audio.export(path_stt, format="wav")
        
        # 원어민 파일 생성 (오류 방지 로직)
        tts = gTTS(text=target_text, lang='en')
        tts.save(path_n_mp3)
        if os.path.exists(path_n_mp3):
            native_raw = AudioSegment.from_file(path_n_mp3)
            n_s, n_e = get_speech_bounds(native_raw, silence_thresh=-35)
            final_n = native_raw[n_s:n_e]; final_n.export(path_n_wav, format="wav")
        else:
            st.error("원어민 음성 생성 실패"); st.stop()

        y_l, sr_curr = librosa.load(path_l, sr=22050); y_n, _ = librosa.load(path_n_wav, sr=sr_curr)

        st.divider()
        c_a1, c_a2 = st.columns(2)
        with c_a1: st.write("🎙️ **나의 발음**"); st.audio(path_l)
        with c_a2: st.write("🔊 **원어민 발음**"); st.audio(path_n_wav)

        tab1, tab2, tab3, tab4 = st.tabs(["🎯 AI 점수", "⏱️ 유창성", "🔊 음파 대조", "📈 피치 분석"])

        with tab4:
            st.subheader("억양 멜로디 분석 (Time-Aligned)")
            f0_l, v_l, p_l = librosa.pyin(y_l, fmin=75, fmax=400, hop_length=64)
            f0_n, v_n, p_n = librosa.pyin(y_n, fmin=60, fmax=400, hop_length=64)
            f0_l_f = np.where(v_l & (p_l > 0.15), f0_l, np.nan)
            f0_n_f = np.where(v_n & (p_n > 0.01), f0_n, np.nan)
            
            fig_p, (ax_l, ax_n) = plt.subplots(1, 2, figsize=(15, 4), sharey=True)
            ax_l.plot(librosa.times_like(f0_l, sr=sr_curr, hop_length=64), f0_l_f, color='#1f77b4', ls=':', marker='o', markersize=2)
            ax_n.plot(librosa.times_like(f0_n, sr=sr_curr, hop_length=64), f0_n_f, color='gray', ls=':', marker='o', markersize=2)
            st.pyplot(fig_p)

            st.write("---")
            if st.button("🚀 정규화 및 피크 분석 실행", use_container_width=True):
                # 세분화된 점수 계산
                score = calculate_intonation_score(f0_n_f, f0_l_f)
                
                # 시각화를 위한 시간축 정렬 곡선 생성
                vec_n = f0_n_f[~np.isnan(f0_n_f)]; vec_l = f0_l_f[~np.isnan(f0_l_f)]
                norm_n = interp1d(np.linspace(0,1,len(vec_n)), normalize_pitch(vec_n))(np.linspace(0,1,100))
                norm_l = interp1d(np.linspace(0,1,len(vec_l)), normalize_pitch(vec_l))(np.linspace(0,1,100))
                
                fig_ov, axo = plt.subplots(figsize=(12, 4))
                axo.plot(np.linspace(0,1,100), norm_n, color='lightgray', lw=3, label='Native', alpha=0.7)
                axo.plot(np.linspace(0,1,100), norm_l, color='#1f77b4', lw=2, label='Learner')
                axo.set_title("Time-Aligned Melody Pattern"); axo.legend(); st.pyplot(fig_ov)
                
                st.metric("억양 유사도 점수 (피크 가중치 적용)", f"{score}점")
                if score >= 75: st.success("🌟 강세 위치와 멜로디가 매우 자연스럽습니다!")
                elif score >= 45: st.info("👍 흐름은 비슷합니다. 주요 단어의 강세(Peak)를 더 명확히 해보세요.")
                else: st.warning("🧐 억양의 높낮이 변화가 원어민과 많이 다릅니다. 노래하듯 리듬을 타보세요.")

    except Exception as e: st.error(f"분석 중 오류 발생: {e}")
    finally:
        for f in [path_l, path_n_mp3, path_n_wav, path_stt, "temp_preview.wav"]:
            if os.path.exists(f): os.remove(f)
