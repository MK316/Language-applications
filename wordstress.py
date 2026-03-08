import streamlit as st
from streamlit_mic_recorder import mic_recorder
import io, os, librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from scipy.interpolate import interp1d

# --- [1] 레이아웃 설정 ---
st.set_page_config(page_title="Word Stress Master", layout="centered")
st.markdown("<style>.stSlider { padding-left: 0px; padding-right: 0px; }</style>", unsafe_allow_html=True)

# 세션 상태 초기화
if 'reset_key' not in st.session_state: st.session_state.reset_key = 0
if 'last_audio_id' not in st.session_state: st.session_state.last_audio_id = None
if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
if 'final_y_l' not in st.session_state: st.session_state.final_y_l = None

# --- [2] 분석 엔진 ---
def get_rms_envelope(y, hop_length=256):
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    return np.convolve(rms, np.ones(5)/5, mode='same')

def detect_syllable_stress(y, sr):
    env = get_rms_envelope(y)
    weighted_env = (env / (np.max(env) + 1e-6)) * 0.8 + (env / (np.sum(env) + 1e-6)) * 0.2
    return np.argmax(weighted_env), env

def calculate_normalized_score(env_n, env_l):
    if len(env_n) < 2 or len(env_l) < 2: return 0, np.zeros(100), np.zeros(100)
    standard_x = np.linspace(0, 1, 100)
    f_n = interp1d(np.linspace(0, 1, len(env_n)), env_n, fill_value="extrapolate")
    f_l = interp1d(np.linspace(0, 1, len(env_l)), env_l, fill_value="extrapolate")
    norm_n, norm_l = f_n(standard_x), f_l(standard_x)
    shape_corr = np.corrcoef(norm_n, norm_l)[0, 1]
    return int(max(0, shape_corr) * 100), norm_n, norm_l

# --- [3] 메인 UI ---
st.title("🎙️ Word Stress Master")
word_db = {"Photograph": "photograph", "Photographer": "photographer", "Education": "education"}
target_word = word_db[st.selectbox("학습할 단어 선택:", list(word_db.keys()))]

if st.button("🔊 원어민 발음 듣기"):
    tts = gTTS(text=target_word, lang='en')
    mp3_buf = io.BytesIO(); tts.write_to_fp(mp3_buf); st.audio(mp3_buf.getvalue())

st.divider()

# 녹음 컴포넌트
audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 완료", key=f"rec_{st.session_state.reset_key}")

if audio:
    if audio['id'] != st.session_state.last_audio_id:
        st.session_state.last_audio_id, st.session_state.analysis_done, st.session_state.final_y_l = audio['id'], False, None

    # [중요] 오디오 로드 및 시간축 계산 정밀화
    l_raw = AudioSegment.from_file(io.BytesIO(audio['bytes']))
    sr_f = l_raw.frame_rate
    y_full = np.array(l_raw.get_array_of_samples(), dtype=np.float32) / (2**15)
    if l_raw.channels > 1: y_full = y_full.reshape((-1, l_raw.channels)).mean(axis=1)
    
    # 전체 시간(초) 계산
    duration_sec = len(y_full) / sr_f
    
    st.markdown(f"#### ✂️ 분석 구간 설정 (전체: {duration_sec:.2f}초)")
    
    # 무음 탐지 기반 초기 슬라이더 범위
    auto_b = detect_nonsilent(l_raw, min_silence_len=100, silence_thresh=-45)
    start_init, end_init = (auto_b[0][0]/1000, auto_b[0][1]/1000) if auto_b else (0.0, duration_sec)
    
    # 슬라이더 (X축 시간과 직결됨)
    trim_range = st.slider("구간 선택 (초):", 0.0, float(duration_sec), (float(start_init), float(end_init)), step=0.01)

    # [핵심 수정] 그래프의 X축 눈금을 슬라이더의 시간과 강제로 일치시킴
    [Image of audio waveform amplitude envelope and stress peaks]
    fig_p, axp = plt.subplots(figsize=(10, 2.5))
    times = np.linspace(0, duration_sec, len(y_full)) # 물리적 시간축 생성
    axp.plot(times, y_full, color='skyblue', alpha=0.7, lw=1)
    
    # 슬라이더 위치 표시 (빨간 점선)
    axp.axvline(x=trim_range[0], color='red', lw=2, ls='--')
    axp.axvline(x=trim_range[1], color='red', lw=2, ls='--')
    
    # 그래프 스타일 설정
    axp.set_xlim(0, duration_sec)
    axp.set_xlabel("Time (seconds)")
    axp.set_yticks([])
    axp.grid(True, axis='x', alpha=0.3) # 눈금 보조선 추가
    st.pyplot(fig_p)

    st.write("🔈 **선택 구간 미리듣기:**")
    trimmed_audio = l_raw[int(trim_range[0]*1000):int(trim_range[1]*1000)]
    st.audio(trimmed_audio.export(io.BytesIO(), format="wav").getvalue())

    c1, c2 = st.columns(2)
    with c1:
        if st.button("📊 정밀 분석 실행", type="primary"):
            st.session_state.analysis_done = True
            st.session_state.final_y_l = y_full[int(trim_range[0]*sr_f):int(trim_range[1]*sr_f)]
            st.session_state.current_sr = sr_f
            st.session_state.final_audio_l = trimmed_audio
    with c2:
        if st.button("🔄 연습 리셋"):
            st.session_state.reset_key += 1
            st.session_state.last_audio_id, st.session_state.analysis_done, st.session_state.final_y_l = None, False, None
            st.rerun()

# --- [4] 결과 분석 섹션 ---
if st.session_state.analysis_done and st.session_state.final_y_l is not None:
    y_l, sr = st.session_state.final_y_l, st.session_state.current_sr
    try:
        tts = gTTS(text=target_word, lang='en'); n_mp3_io = io.BytesIO(); tts.write_to_fp(n_mp3_io); n_mp3_io.seek(0)
        n_seg = AudioSegment.from_file(n_mp3_io)
        sr_n = n_seg.frame_rate
        y_n = np.array(n_seg.get_array_of_samples(), dtype=np.float32) / (2**15)
        if n_seg.channels > 1: y_n = y_n.reshape((-1, n_seg.channels)).mean(axis=1)
        
        y_l_norm, y_n_norm = librosa.util.normalize(y_l), librosa.util.normalize(y_n)
        p_idx_l, env_l = detect_syllable_stress(y_l_norm, sr)
        p_idx_n, env_n = detect_syllable_stress(y_n_norm, sr_n)
        score, norm_n, norm_l = calculate_normalized_score(env_n, env_l)

        st.divider(); st.metric("종합 리듬 점수 (속도 보정)", f"{score}점")
        
        # 절대 시간축 비교 (여기서도 X축 동기화 적용)
        st.write("### 📏 절대 시간 기반 비교")
        fig_abs, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        mt = max(len(y_l)/sr, len(y_n)/sr_n)
        
        for ax, y, env, s, p, t, col in [(ax1, y_l_norm, env_l, sr, p_idx_l, "My Rhythm", "skyblue"), 
                                         (ax2, y_n_norm, env_n, sr_n, p_idx_n, "Native Standard", "lightgray")]:
            ts = np.linspace(0, len(y)/s, len(env))
            ax.plot(ts, env, color='#1f77b4' if col=="skyblue" else "gray", lw=2)
            ax.axvline(x=ts[p], color='red', lw=3)
            ax.set_title(f"{t} ({len(y)/s:.2f}s)")
            ax.set_xlim(0, mt)
        plt.tight_layout(); st.pyplot(fig_abs)

        # 시간 정규화 오버레이
        st.write("### 🔄 시간 정규화 리듬 패턴 대조")
        [Image of signal overlay for pattern matching]
        fig_norm, axn = plt.subplots(figsize=(10, 4))
        x_range = np.linspace(0, 100, 100)
        axn.fill_between(x_range, 0, norm_n, color='gray', alpha=0.2, label='Native Guide')
        axn.plot(x_range, norm_l, color='#ff4b4b', lw=3, label='My Rhythm')
        axn.set_xlabel("Progression (%)"); axn.legend(); st.pyplot(fig_norm)

    except Exception as e: st.error(f"분석 오류: {e}")
