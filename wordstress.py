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

# --- [2] 분석 엔진: 조절 가능한 음절 탐지 ---
def get_smoothed_envelope(y, sr):
    rms = librosa.feature.rms(y=y, hop_length=256)[0]
    smoothed = np.convolve(rms, np.ones(7)/7, mode='same')
    return smoothed

def analyze_rhythm_peaks(env, sensitivity):
    # 슬라이더 값에 따라 prominence(돌출도)를 조절하여 민감도 제어
    # sensitivity가 높을수록 작은 봉우리도 잡고, 낮을수록 큰 봉우리만 잡음
    peaks, _ = find_peaks(env, 
                          height=np.max(env) * 0.15, 
                          distance=10,             # 너무 가까운 봉우리는 하나로 합침
                          prominence=sensitivity)  # 슬라이더로 조절되는 핵심 값
    primary_stress_idx = np.argmax(env) if len(env) > 0 else 0
    return peaks, primary_stress_idx

def calculate_pedagogical_score(env_n, env_l, sensitivity):
    if len(env_n) < 2 or len(env_l) < 2: return 0, np.zeros(100), np.zeros(100), [], []
    standard_x = np.linspace(0, 1, 100)
    f_n = interp1d(np.linspace(0, 1, len(env_n)), env_n, fill_value="extrapolate")
    f_l = interp1d(np.linspace(0, 1, len(env_l)), env_l, fill_value="extrapolate")
    n_norm, l_norm = f_n(standard_x), f_l(standard_x)
    
    # 설정된 민감도로 피크 추출
    peaks_n, stress_n = analyze_rhythm_peaks(n_norm, sensitivity)
    peaks_l, stress_l = analyze_rhythm_peaks(l_norm, sensitivity)
    
    stress_dist = abs(stress_n - stress_l) / 100
    stress_score = max(0, 1 - stress_dist) * 50
    pattern_corr = np.corrcoef(n_norm, l_norm)[0, 1]
    pattern_score = max(0, pattern_corr) * 50
    
    total = int(stress_score + pattern_score)
    return total, n_norm, l_norm, peaks_l, peaks_n

# --- [3] 메인 UI ---
st.title("🎙️ Word Stress Master")

# 사이드바 혹은 상단에 민감도 조절기 추가
st.sidebar.header("⚙️ 분석 설정")
s_val = st.sidebar.slider("음절 탐지 민감도", 0.01, 0.20, 0.05, step=0.01, 
                         help="값이 클수록 확실한 음절만 잡고, 작을수록 미세한 떨림도 음절로 인식합니다.")

word_db = {"Photograph": "photograph", "Photographer": "photographer", "Education": "education"}
target_word = word_db[st.selectbox("학습할 단어 선택:", list(word_db.keys()))]

if st.button("🔊 원어민 표준 발음 듣기"):
    tts = gTTS(text=target_word, lang='en')
    mp3_buf = io.BytesIO(); tts.write_to_fp(mp3_buf); st.audio(mp3_buf.getvalue())

st.divider()

audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 완료", key=f"rec_{st.session_state.reset_key}")

if audio:
    if audio['id'] != st.session_state.last_audio_id:
        st.session_state.last_audio_id, st.session_state.analysis_done, st.session_state.final_y_l = audio['id'], False, None

    l_raw = AudioSegment.from_file(io.BytesIO(audio['bytes']))
    sr_f = l_raw.frame_rate
    y_full = np.array(l_raw.get_array_of_samples(), dtype=np.float32) / (2**15)
    if l_raw.channels > 1: y_full = y_full.reshape((-1, l_raw.channels)).mean(axis=1)
    
    duration_sec = len(y_full) / sr_f
    st.markdown(f"#### ✂️ 분석 구간 설정 ({duration_sec:.2f}s)")
    
    auto_b = detect_nonsilent(l_raw, min_silence_len=100, silence_thresh=-45)
    s_init, e_init = (auto_b[0][0]/1000, auto_b[0][1]/1000) if auto_b else (0.0, duration_sec)
    trim_range = st.slider("구간 선택 (초):", 0.0, float(duration_sec), (float(s_init), float(e_init)), step=0.01)

    fig_p, axp = plt.subplots(figsize=(10, 2.0))
    times = np.linspace(0, duration_sec, len(y_full))
    axp.plot(times, y_full, color='skyblue', alpha=0.7, lw=1)
    axp.axvline(x=trim_range[0], color='red', lw=2, ls='--')
    axp.axvline(x=trim_range[1], color='red', lw=2, ls='--')
    axp.set_xlim(0, duration_sec); axp.set_yticks([]); st.pyplot(fig_p)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("📊 정밀 분석 실행", type="primary"):
            st.session_state.analysis_done = True
            st.session_state.final_y_l = y_full[int(trim_range[0]*sr_f):int(trim_range[1]*sr_f)]
            st.session_state.current_sr = sr_f
            st.session_state.final_audio_l = l_raw[int(trim_range[0]*1000):int(trim_range[1]*1000)]
    with c2:
        if st.button("🔄 연습 리셋"):
            st.session_state.reset_key += 1
            st.rerun()

# --- [4] 결과 분석 섹션 ---
if st.session_state.analysis_done and st.session_state.final_y_l is not None:
    y_l, sr = st.session_state.final_y_l, st.session_state.current_sr
    try:
        tts = gTTS(text=target_word, lang='en'); n_mp3_io = io.BytesIO(); tts.write_to_fp(n_mp3_io); n_mp3_io.seek(0)
        n_seg = AudioSegment.from_file(n_mp3_io)
        y_n = np.array(n_seg.get_array_of_samples(), dtype=np.float32) / (2**15)
        if n_seg.channels > 1: y_n = y_n.reshape((-1, n_seg.channels)).mean(axis=1)
        sr_n = n_seg.frame_rate
        
        env_l = get_smoothed_envelope(librosa.util.normalize(y_l), sr)
        env_n = get_smoothed_envelope(librosa.util.normalize(y_n), sr_n)
        
        # 슬라이더에서 받은 s_val을 민감도로 사용
        score, norm_n, norm_l, peaks_l, peaks_n = calculate_pedagogical_score(env_n, env_l, s_val)

        st.divider()
        st.metric("종합 리듬 점수", f"{score}점")
        
        if len(peaks_n) != len(peaks_l):
            st.warning(f"⚠️ 음절 수 불일치 (원어민: {len(peaks_n)} / 나: {len(peaks_l)})")
        else:
            st.success(f"✅ 음절 수 일치 ({len(peaks_n)}음절)")

        st.write("### 🔄 리듬 패턴 및 음절 위치 대조")
        fig_norm, axn = plt.subplots(figsize=(10, 4))
        x_range = np.linspace(0, 100, 100)
        
        axn.fill_between(x_range, 0, norm_n, color='gray', alpha=0.15, label='Native Guide')
        axn.scatter(x_range[peaks_n], norm_n[peaks_n], color='gray', s=70, edgecolors='white', zorder=4, label='Native Syllables')
        
        axn.plot(x_range, norm_l, color='#ff4b4b', lw=2.5, label='My Rhythm')
        axn.scatter(x_range[peaks_l], norm_l[peaks_l], color='red', s=70, zorder=5, label='My Syllables')
        
        axn.set_title(f"Syllable Detection (Sensitivity: {s_val})")
        axn.legend(); st.pyplot(fig_norm)

    except Exception as e: st.error(f"분석 중 오류 발생: {e}")
