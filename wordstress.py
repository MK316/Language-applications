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
    # RMS 에너지를 구한 뒤 이동평균으로 미세 떨림 제거 (언어적 포형 추출)
    rms = librosa.feature.rms(y=y, hop_length=256)[0]
    smoothed = np.convolve(rms, np.ones(10)/10, mode='same')
    return smoothed

def analyze_rhythm_peaks(env):
    # 음절 봉우리 탐지: 최소 높이 15% 기준, 음절 간 거리 확보
    peaks, _ = find_peaks(env, height=np.max(env)*0.15, distance=5)
    primary_stress_idx = np.argmax(env) if len(env) > 0 else 0
    return peaks, primary_stress_idx

def calculate_pedagogical_score(env_n, env_l):
    if len(env_n) < 2 or len(env_l) < 2: return 0, np.zeros(100), np.zeros(100), [], []
    
    # 시간 정규화 (패턴 비교용)
    standard_x = np.linspace(0, 1, 100)
    f_n = interp1d(np.linspace(0, 1, len(env_n)), env_n, fill_value="extrapolate")
    f_l = interp1d(np.linspace(0, 1, len(env_l)), env_l, fill_value="extrapolate")
    n_norm, l_norm = f_n(standard_x), f_l(standard_x)
    
    # 양측 피크 추출
    peaks_n, stress_n = analyze_rhythm_peaks(n_norm)
    peaks_l, stress_l = analyze_rhythm_peaks(l_norm)
    
    # 교육적 점수 산정 (강세 위치 50% + 패턴 상관도 50%)
    stress_dist = abs(stress_n - stress_l) / 100
    stress_score = max(0, 1 - stress_dist) * 50
    
    pattern_corr = np.corrcoef(n_norm, l_norm)[0, 1]
    pattern_score = max(0, pattern_corr) * 50
    
    total = int(stress_score + pattern_score)
    return total, n_norm, l_norm, peaks_l, peaks_n

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

    # pydub으로 로드하여 LibsndfileError 방지 및 정확한 샘플링 레이트 확보
    l_raw = AudioSegment.from_file(io.BytesIO(audio['bytes']))
    sr_f = l_raw.frame_rate
    y_full = np.array(l_raw.get_array_of_samples(), dtype=np.float32) / (2**15)
    if l_raw.channels > 1:
        y_full = y_full.reshape((-1, l_raw.channels)).mean(axis=1)
    
    duration_sec = len(y_full) / sr_f
    st.markdown(f"#### ✂️ 분석 구간 설정 (전체 길이: {duration_sec:.2f}s)")
    
    auto_b = detect_nonsilent(l_raw, min_silence_len=100, silence_thresh=-45)
    s_init, e_init = (auto_b[0][0]/1000, auto_b[0][1]/1000) if auto_b else (0.0, duration_sec)
    trim_range = st.slider("구간 선택 (초):", 0.0, float(duration_sec), (float(s_init), float(e_init)), step=0.01)

    # 파형 시각화 (X축 정밀 동기화)
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
        if st.button("📊 정밀 분석 실행", type="primary"):
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
        # 원어민 데이터 처리
        tts = gTTS(text=target_word, lang='en'); n_mp3_io = io.BytesIO(); tts.write_to_fp(n_mp3_io); n_mp3_io.seek(0)
        n_seg = AudioSegment.from_file(n_mp3_io)
        y_n = np.array(n_seg.get_array_of_samples(), dtype=np.float32) / (2**15)
        if n_seg.channels > 1: y_n = y_n.reshape((-1, n_seg.channels)).mean(axis=1)
        sr_n = n_seg.frame_rate
        
        # 포형 스무딩 및 교육적 점수 계산
        env_l = get_smoothed_envelope(librosa.util.normalize(y_l), sr)
        env_n = get_smoothed_envelope(librosa.util.normalize(y_n), sr_n)
        score, norm_n, norm_l, peaks_l, peaks_n = calculate_pedagogical_score(env_n, env_l)

        st.divider()
        st.metric("종합 리듬 점수", f"{score}점")
        
        # 음절 수 대조 피드백
        if len(peaks_n) != len(peaks_l):
            st.warning(f"⚠️ 음절 수가 다릅니다. 원어민: {len(peaks_n)}음절 / 나: {len(peaks_l)}음절")
        else:
            st.success(f"✅ 원어민과 동일한 {len(peaks_n)}음절 리듬 패턴을 유지하고 있습니다.")

        # 오디오 대조
        a1, a2 = st.columns(2)
        with a1: st.write("🙋 나의 발음"); st.audio(st.session_state.final_audio_l.export(io.BytesIO(), format="wav").getvalue())
        with a2: st.write("🎙️ 원어민 표준"); st.audio(n_seg.export(io.BytesIO(), format="wav").getvalue())

        # 시간 정규화 오버레이 (양측 피크 분석)
        st.write("### 🔄 리듬 패턴 및 음절 위치 대조")
        [Image of audio waveform amplitude envelope and stress peaks]
        fig_norm, axn = plt.subplots(figsize=(10, 4))
        x_range = np.linspace(0, 100, 100)
        
        # 원어민 가이드: 회색 영역 + 회색 점(피크)
        axn.fill_between(x_range, 0, norm_n, color='gray', alpha=0.15, label='Native Guide')
        axn.scatter(x_range[peaks_n], norm_n[peaks_n], color='gray', s=60, edgecolors='white', zorder=4, label='Native Syllables')
        
        # 학습자 리듬: 빨간 선 + 빨간 점(피크)
        axn.plot(x_range, norm_l, color='#ff4b4b', lw=2.5, label='My Rhythm (Smoothed)')
        axn.scatter(x_range[peaks_l], norm_l[peaks_l], color='red', s=60, zorder=5, label='My Syllables')
        
        axn.set_title(f"Rhythm Pattern Matching (Native {len(peaks_n)} vs Me {len(peaks_l)})")
        axn.set_xlabel("Progression (%)"); axn.legend()
        st.pyplot(fig_norm)

        # 절대 시간 비교
        st.write("### 📏 절대 시간 기반 에너지 프로필")
        [Image of energy envelope comparison for word stress]
        fig_abs, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        mt = max(len(y_l)/sr, len(y_n)/sr_n)
        for ax, y, env, s, t, col in [(ax1, y_l, env_l, sr, "My Rhythm", "skyblue"), (ax2, y_n, env_n, sr_n, "Native Standard", "lightgray")]:
            ts = np.linspace(0, len(y)/s, len(env))
            ax.plot(ts, env, color='#1f77b4' if col=="skyblue" else "gray", lw=2)
            ax.axvline(x=ts[np.argmax(env)], color='red', lw=3, ls='--')
            ax.set_title(f"{t} ({len(y)/s:.2f}s)"); ax.set_xlim(0, mt)
        plt.tight_layout(); st.pyplot(fig_abs)

    except Exception as e:
        st.error(f"분석 중 오류 발생: {e}")
