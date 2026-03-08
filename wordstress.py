import streamlit as st
from streamlit_mic_recorder import mic_recorder
import io, librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

# --- [1] 레이아웃 및 디자인 설정 ---
st.set_page_config(page_title="Word Stress Master", layout="centered")

st.markdown("""
    <style>
    .stSlider { padding-left: 0px; padding-right: 0px; }
    .main .block-container { padding-top: 1rem; }
    button { height: 3.5em !important; font-weight: bold !important; border-radius: 10px !important; }
    .stButton > button[kind="primary"] { background-color: #ff4b4b !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# 세션 상태 초기화
if 'reset_key' not in st.session_state: st.session_state.reset_key = 0
if 'last_audio_id' not in st.session_state: st.session_state.last_audio_id = None
if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
if 'final_y_l' not in st.session_state: st.session_state.final_y_l = None

# --- [2] 분석 엔진: 피크 탐지 및 스무딩 ---
def get_smoothed_envelope(y, sr):
    # RMS 에너지를 구한 뒤 이동평균(Convolution)으로 미세 떨림 제거
    rms = librosa.feature.rms(y=y, hop_length=256)[0]
    smoothed = np.convolve(rms, np.ones(10)/10, mode='same')
    return smoothed

def analyze_rhythm_peaks(env):
    # 음절 봉우리(Peak) 탐지: 최소 높이 15% 기준
    peaks, _ = find_peaks(env, height=np.max(env)*0.15, distance=5)
    primary_stress_idx = np.argmax(env) if len(env) > 0 else 0
    return peaks, primary_stress_idx

def calculate_pedagogical_score(env_n, env_l):
    if len(env_n) < 2 or len(env_l) < 2: return 0, np.zeros(100), np.zeros(100), []
    
    # 시간 정규화 (패턴 비교용)
    standard_x = np.linspace(0, 1, 100)
    f_n = interp1d(np.linspace(0, 1, len(env_n)), env_n, fill_value="extrapolate")
    f_l = interp1d(np.linspace(0, 1, len(env_l)), env_l, fill_value="extrapolate")
    n_norm, l_norm = f_n(standard_x), f_l(standard_x)
    
    # 피크 추출
    peaks_n, stress_n = analyze_rhythm_peaks(n_norm)
    peaks_l, stress_l = analyze_rhythm_peaks(l_norm)
    
    # 교육적 점수 산정
    stress_dist = abs(stress_n - stress_l) / 100
    stress_score = max(0, 1 - stress_dist) * 50 # 강세 위치 가중치 50%
    
    pattern_corr = np.corrcoef(n_norm, l_norm)[0, 1]
    pattern_score = max(0, pattern_corr) * 50   # 패턴 일치도 가중치 50%
    
    total = int(stress_score + pattern_score)
    return total, n_norm, l_norm, peaks_l

# --- [3] 메인 UI ---
st.title("🎙️ Word Stress Master")
word_db = {"Photograph": "photograph", "Photographer": "photographer", "Education": "education"}
target_word = word_db[st.selectbox("학습할 단어 선택:", list(word_db.keys()))]

if st.button("🔊 원어민 표준 발음 듣기"):
    tts = gTTS(text=target_word, lang='en')
    mp3_buf = io.BytesIO(); tts.write_to_fp(mp3_buf); st.audio(mp3_buf.getvalue())

st.divider()

audio = mic_recorder(
    start_prompt="🎤 녹음 시작", 
    stop_prompt="🛑 완료", 
    key=f"rec_{st.session_state.reset_key}"
)

if audio:
    if audio['id'] != st.session_state.last_audio_id:
        st.session_state.last_audio_id, st.session_state.analysis_done, st.session_state.final_y_l = audio['id'], False, None

    # [에러 해결 핵심] pydub을 사용하여 파일을 열고 직접 numpy 배열로 변환 (LibsndfileError 우회)
    l_raw = AudioSegment.from_file(io.BytesIO(audio['bytes']))
    sr_f = l_raw.frame_rate
    y_full = np.array(l_raw.get_array_of_samples(), dtype=np.float32) / (2**15)
    if l_raw.channels > 1:
        y_full = y_full.reshape((-1, l_raw.channels)).mean(axis=1)
    
    duration_sec = len(y_full) / sr_f
    
    st.markdown(f"#### ✂️ 분석 구간 설정 (물리적 실제 길이: {duration_sec:.2f}s)")
    
    # 구간 조절 슬라이더 및 파형 시각화
    auto_b = detect_nonsilent(l_raw, min_silence_len=100, silence_thresh=-45)
    s_init, e_init = (auto_b[0][0]/1000, auto_b[0][1]/1000) if auto_b else (0.0, duration_sec)
    trim_range = st.slider("구간 선택 (초):", 0.0, float(duration_sec), (float(s_init), float(e_init)), step=0.01)

    fig_p, axp = plt.subplots(figsize=(10, 2.5))
    times = np.linspace(0, duration_sec, len(y_full))
    axp.plot(times, y_full, color='skyblue', alpha=0.7, lw=1)
    axp.axvline(x=trim_range[0], color='red', lw=2, ls='--')
    axp.axvline(x=trim_range[1], color='red', lw=2, ls='--')
    axp.set_xlim(0, duration_sec); axp.set_yticks([]); st.pyplot(fig_p)

    st.write("🔈 **선택된 구간 미리듣기:**")
    trimmed_audio_seg = l_raw[int(trim_range[0]*1000):int(trim_range[1]*1000)]
    st.audio(trimmed_audio_seg.export(io.BytesIO(), format="wav").getvalue())

    c1, c2 = st.columns(2)
    with c1:
        if st.button("📊 교육적 관점 분석 실행", type="primary"):
            st.session_state.analysis_done = True
            st.session_state.final_y_l = y_full[int(trim_range[0]*sr_f):int(trim_range[1]*sr_f)]
            st.session_state.current_sr = sr_f
            st.session_state.final_audio_l = trimmed_audio_seg
    with c2:
        if st.button("🔄 연습 리셋"):
            st.session_state.reset_key += 1
            st.rerun()

# --- [4] 결과 분석 섹션 ---
if st.session_state.analysis_done and st.session_state.final_y_l is not None:
    y_l, sr = st.session_state.final_y_l, st.session_state.current_sr
    
    try:
        # 원어민 데이터 처리 (pydub 활용)
        tts = gTTS(text=target_word, lang='en'); n_mp3_io = io.BytesIO(); tts.write_to_fp(n_mp3_io); n_mp3_io.seek(0)
        n_seg = AudioSegment.from_file(n_mp3_io)
        y_n = np.array(n_seg.get_array_of_samples(), dtype=np.float32) / (2**15)
        if n_seg.channels > 1: y_n = y_n.reshape((-1, n_seg.channels)).mean(axis=1)
        sr_n = n_seg.frame_rate
        
        # 에너지 포형 스무딩 처리 (미세 떨림 제거)
        env_l = get_smoothed_envelope(librosa.util.normalize(y_l), sr)
        env_n = get_smoothed_envelope(librosa.util.normalize(y_n), sr_n)
        
        # 교육적 점수 및 피크 계산
        score, norm_n, norm_l, peaks_l = calculate_pedagogical_score(env_n, env_l)

        st.divider()
        st.metric("종합 리듬 점수", f"{score}점")
        st.write(f"💡 분석 결과, 단어에서 **{len(peaks_l)}개**의 주요 음절 봉우리가 확인되었습니다.")

        # 절대 시간 비교
        st.write("### 📏 절대 시간 기반 비교")
        fig_abs, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        mt = max(len(y_l)/sr, len(y_n)/sr_n)
        for ax, y, env, s, t, col in [(ax1, y_l, env_l, sr, "My Rhythm", "skyblue"), (ax2, y_n, env_n, sr_n, "Native Standard", "lightgray")]:
            ts = np.linspace(0, len(y)/s, len(env))
            ax.plot(ts, env, color='#1f77b4' if col=="skyblue" else "gray", lw=2)
            ax.axvline(x=ts[np.argmax(env)], color='red', lw=3, ls='--')
            ax.set_title(f"{t} ({len(y)/s:.2f}s)"); ax.set_xlim(0, mt)
        plt.tight_layout(); st.pyplot(fig_abs)

        # 시간 정규화 오버레이 (피크 강조)
        st.write("### 🔄 시간 정규화 리듬 패턴 대조 (피크 분석)")
        fig_norm, axn = plt.subplots(figsize=(10, 4))
        x_range = np.linspace(0, 100, 100)
        axn.fill_between(x_range, 0, norm_n, color='gray', alpha=0.15, label='Native Guide')
        axn.plot(x_range, norm_l, color='#ff4b4b', lw=2.5, label='My Rhythm (Smoothed)')
        # 학습자의 피크 지점 표시
        axn.scatter(x_range[peaks_l], norm_l[peaks_l], color='red', s=40, zorder=5, label='Detected Syllables')
        axn.legend(); st.pyplot(fig_norm)

    except Exception as e:
        st.error(f"분석 중 오류 발생: {e}")
