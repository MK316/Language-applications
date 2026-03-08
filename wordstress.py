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

# --- [1] 디자인 설정 ---
st.set_page_config(page_title="Word Stress Master", layout="centered")
st.markdown("""
    <style>
    .stSlider { padding-left: 0px; padding-right: 0px; }
    button { height: 3.5em !important; font-weight: bold !important; border-radius: 10px !important; }
    .stButton > button[kind="primary"] { background-color: #ff4b4b !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# --- [2] 분석 엔진: 지능형 음절 탐지 ---
def get_adaptive_envelope(y, sr, is_native=True):
    rms = librosa.feature.rms(y=y, hop_length=256)[0]
    # 학습자 데이터는 조음 노이즈가 많으므로 더 강력하게 스무딩 (15), 원어민은 (7)
    window_size = 7 if is_native else 15
    smoothed = np.convolve(rms, np.ones(window_size)/window_size, mode='same')
    return smoothed

def detect_syllables_smart(env, is_native=True):
    """
    원어민과 학습자의 특성에 맞춰 피크 탐지 파라미터를 다르게 적용
    """
    if is_native:
        # 원어민: 약한 음절도 잡아야 하므로 prominence를 낮춤
        peaks, _ = find_peaks(env, height=np.max(env)*0.08, distance=5, prominence=0.01)
    else:
        # 학습자: 노이즈로 인한 가짜 피크를 방지하기 위해 높이와 돌출도 기준을 강화
        peaks, _ = find_peaks(env, height=np.max(env)*0.15, distance=8, prominence=0.03)
    
    primary_idx = np.argmax(env) if len(env) > 0 else 0
    return peaks, primary_idx

def calculate_pedagogical_score(env_n, env_l):
    if len(env_n) < 2 or len(env_l) < 2: return 0, np.zeros(100), np.zeros(100), [], []
    standard_x = np.linspace(0, 1, 100)
    f_n = interp1d(np.linspace(0, 1, len(env_n)), env_n, fill_value="extrapolate")
    f_l = interp1d(np.linspace(0, 1, len(env_l)), env_l, fill_value="extrapolate")
    n_norm, l_norm = f_n(standard_x), f_l(standard_x)
    
    # 각각 최적화된 로직으로 피크 추출
    peaks_n, stress_n = detect_syllables_smart(n_norm, is_native=True)
    peaks_l, stress_l = detect_syllables_smart(l_norm, is_native=False)
    
    # 점수: 강세 위치(50%) + 패턴 유사도(50%)
    stress_score = max(0, 1 - abs(stress_n - stress_l)/100) * 50
    pattern_score = max(0, np.corrcoef(n_norm, l_norm)[0, 1]) * 50
    
    return int(stress_score + pattern_score), n_norm, l_norm, peaks_l, peaks_n

# --- [3] 메인 UI ---
st.title("🎙️ Word Stress Master")
word_db = {"Photograph": "photograph", "Photographer": "photographer", "Education": "education"}
target_word = word_db[st.selectbox("단어 선택:", list(word_db.keys()))]

if st.button("🔊 원어민 표준 발음 듣기"):
    tts = gTTS(text=target_word, lang='en')
    mp3_buf = io.BytesIO(); tts.write_to_fp(mp3_buf); st.audio(mp3_buf.getvalue())

st.divider()

if 'reset_key' not in st.session_state: st.session_state.reset_key = 0
audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 완료", key=f"rec_{st.session_state.reset_key}")

if audio:
    l_raw = AudioSegment.from_file(io.BytesIO(audio['bytes']))
    sr_f = l_raw.frame_rate
    y_full = np.array(l_raw.get_array_of_samples(), dtype=np.float32) / (2**15)
    if l_raw.channels > 1: y_full = y_full.reshape((-1, l_raw.channels)).mean(axis=1)
    
    duration = len(y_full) / sr_f
    st.markdown(f"#### ✂️ 분석 구간 설정 ({duration:.2f}s)")
    auto_b = detect_nonsilent(l_raw, min_silence_len=100, silence_thresh=-45)
    s_init, e_init = (auto_b[0][0]/1000, auto_b[0][1]/1000) if auto_b else (0.0, duration)
    trim_range = st.slider("구간 선택:", 0.0, float(duration), (float(s_init), float(e_init)), step=0.01)

    fig_p, axp = plt.subplots(figsize=(10, 2.2))
    times = np.linspace(0, duration, len(y_full))
    axp.plot(times, y_full, color='skyblue', alpha=0.7)
    axp.axvline(x=trim_range[0], color='red', ls='--'); axp.axvline(x=trim_range[1], color='red', ls='--')
    axp.set_xlim(0, duration); axp.set_yticks([]); st.pyplot(fig_p)

    if st.button("📊 정밀 분석 실행", type="primary"):
        st.session_state.analysis_done = True
        st.session_state.final_y_l = y_full[int(trim_range[0]*sr_f):int(trim_range[1]*sr_f)]
        st.session_state.current_sr = sr_f
        st.session_state.final_audio_l = l_raw[int(trim_range[0]*1000):int(trim_range[1]*1000)]

# --- [4] 결과 출력 ---
if st.session_state.get('analysis_done') and st.session_state.final_y_l is not None:
    y_l, sr = st.session_state.final_y_l, st.session_state.current_sr
    try:
        tts = gTTS(text=target_word, lang='en'); n_mp3 = io.BytesIO(); tts.write_to_fp(n_mp3); n_mp3.seek(0)
        n_seg = AudioSegment.from_file(n_mp3)
        y_n = np.array(n_seg.get_array_of_samples(), dtype=np.float32) / (2**15)
        if n_seg.channels > 1: y_n = y_n.reshape((-1, n_seg.channels)).mean(axis=1)
        sr_n = n_seg.frame_rate
        
        # [핵심] 각각 다른 스무딩 강도 적용
        env_l = get_adaptive_envelope(librosa.util.normalize(y_l), sr, is_native=False)
        env_n = get_adaptive_envelope(librosa.util.normalize(y_n), sr_n, is_native=True)
        
        score, norm_n, norm_l, peaks_l, peaks_n = calculate_pedagogical_score(env_n, env_l)

        st.divider(); st.metric("종합 리듬 점수", f"{score}점")
        if len(peaks_n) != len(peaks_l):
            st.warning(f"⚠️ 음절 수 차이 (원어민: {len(peaks_n)} / 나: {len(peaks_l)})")
        else: st.success(f"✅ 음절 수 일치 ({len(peaks_n)}음절)")

        st.write("### 🔄 리듬 패턴 대조")
        fig_norm, axn = plt.subplots(figsize=(10, 4))
        x = np.linspace(0, 100, 100)
        axn.fill_between(x, 0, norm_n, color='gray', alpha=0.15, label='Native Guide')
        axn.scatter(x[peaks_n], norm_n[peaks_n], color='gray', s=60, edgecolors='white', zorder=4)
        axn.plot(x, norm_l, color='#ff4b4b', lw=2.5, label='My Rhythm')
        axn.scatter(x[peaks_l], norm_l[peaks_l], color='red', s=60, zorder=5)
        axn.set_xlabel("Progression (%)"); axn.legend(); st.pyplot(fig_norm)

    except Exception as e: st.error(f"오류: {e}")

if st.button("🔄 리셋"):
    st.session_state.reset_key += 1
    st.session_state.analysis_done = False
    st.rerun()
