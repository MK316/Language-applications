import streamlit as st
from streamlit_mic_recorder import mic_recorder
import io, os, librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from scipy.interpolate import interp1d

# --- [1] 모바일 레이아웃 및 세그먼트 버튼 CSS ---
st.set_page_config(page_title="Word Stress Master", layout="centered")

st.markdown("""
    <style>
    .stSlider { padding-left: 0px; padding-right: 0px; }
    .main .block-container { padding-top: 1rem; }
    
    /* [핵심] 버튼 두 개를 하나로 합치는 세그먼트 디자인 */
    div[data-testid="stHorizontalBlock"] {
        gap: 0px !important; /* 버튼 사이 간격 제거 */
        border: 1px solid #dcdcdc; /* 전체 테두리 */
        border-radius: 12px;
        overflow: hidden;
        background-color: #f0f2f6;
    }

    div[data-testid="stHorizontalBlock"] div[data-testid="stVerticalBlock"] > div button {
        width: 100% !important;
        height: 3.8em !important;
        border: none !important; /* 개별 테두리 제거 */
        border-radius: 0px !important; /* 둥근 모서리 제거 */
        margin: 0px !important;
        font-size: 15px !important;
        font-weight: bold !important;
        transition: 0.3s;
    }

    /* 왼쪽 녹음 버튼 스타일 */
    div[data-testid="stHorizontalBlock"] > div:nth-child(1) button {
        background-color: #ffefef !important; /* 아주 연한 빨강 배경 */
        color: #ff4b4b !important; /* 빨간 글씨 */
        border-right: 1px solid #dcdcdc !important; /* 중간 구분선 */
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(1) button:hover {
        background-color: #ff4b4b !important;
        color: white !important;
    }

    /* 오른쪽 리셋 버튼 스타일 */
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) button {
        background-color: white !important;
        color: #31333F !important;
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) button:hover {
        background-color: #f0f2f6 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 세션 초기화
if 'reset_key' not in st.session_state: st.session_state.reset_key = 0
if 'last_audio_id' not in st.session_state: st.session_state.last_audio_id = None
if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
if 'final_y_l' not in st.session_state: st.session_state.final_y_l = None

# --- [2] 분석 로직 (기존과 동일) ---
def get_rms_envelope(y, hop_length=256):
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    return np.convolve(rms, np.ones(5)/5, mode='same')

def safe_trim(audio_segment):
    bounds = detect_nonsilent(audio_segment, min_silence_len=100, silence_thresh=-50)
    if not bounds: return audio_segment
    start, end = bounds[0]
    return audio_segment[start:end] if (end - start) > 200 else audio_segment

def detect_syllable_stress(y, sr):
    env = get_rms_envelope(y)
    weighted_env = (env / (np.max(env) + 1e-6)) * 0.8 + (env / (np.sum(env) + 1e-6)) * 0.2
    return np.argmax(weighted_env), env

def calculate_score(env_n, env_l):
    if len(env_n) < 2 or len(env_l) < 2: return 0, 0, 0
    f_n = interp1d(np.linspace(0, 1, len(env_n)), env_n, fill_value="extrapolate")
    f_l = interp1d(np.linspace(0, 1, len(env_l)), env_l, fill_value="extrapolate")
    new_x = np.linspace(0, 1, 100)
    shape_corr = np.corrcoef(f_n(new_x), f_l(new_x))[0, 1]
    dur_n = np.sum(env_n > np.max(env_n)*0.2) / len(env_n)
    dur_l = np.sum(env_l > np.max(env_l)*0.2) / len(env_l)
    return int(max(0, shape_corr) * 100), dur_l * 100, dur_n * 100

# --- [3] 앱 UI ---
st.title("🎙️ Word Stress Master")
word_db = {"Photograph (1음절 강세)": "photograph", "Photographer (2음절 강세)": "photographer", "Education (3음절 강세)": "education"}
selected_label = st.selectbox("학습할 단어 선택:", list(word_db.keys()))
target_word = word_db[selected_label]

if st.button("🔊 원어민 표준 발음 듣기"):
    tts = gTTS(text=target_word, lang='en'); b = io.BytesIO(); tts.write_to_fp(b); st.audio(b.getvalue())

st.divider()
st.subheader(f"🎯 연습: {target_word.upper()}")

# --- [4] 핵심 수정: 세그먼트 스타일 가로 배치 ---
col_l, col_r = st.columns(2)

with col_l:
    audio = mic_recorder(
        start_prompt="🎤 녹음 시작",
        stop_prompt="🛑 완료",
        key=f"recorder_{st.session_state.reset_key}"
    )

with col_r:
    if st.button("🔄 리셋"):
        st.session_state.reset_key += 1
        st.session_state.last_audio_id = None
        st.session_state.analysis_done = False
        st.session_state.final_y_l = None
        st.rerun()

# --- [5] 구간 설정 및 결과 분석 (기존 로직) ---
if audio:
    if audio['id'] != st.session_state.last_audio_id:
        st.session_state.last_audio_id, st.session_state.analysis_done = audio['id'], False
    
    l_raw = AudioSegment.from_file(io.BytesIO(audio['bytes']))
    y_full = np.array(l_raw.get_array_of_samples(), dtype=np.float32) / (2**15)
    if l_raw.channels > 1: y_full = y_full.reshape((-1, l_raw.channels)).mean(axis=1)
    sr_f = l_raw.frame_rate
    
    st.markdown("#### ✂️ 분석 구간 설정")
    b = detect_nonsilent(l_raw, min_silence_len=100, silence_thresh=-45)
    as_ms, ae_ms = b[0] if b else (0, len(l_raw))
    trim_range = st.slider("단어 구간 조절:", 0.0, float(len(y_full)/sr_f), (float(as_ms/1000), float(ae_ms/1000)), step=0.01)

    fig_p = plt.figure(figsize=(10, 2.2)); axp = fig_p.add_axes([0, 0.2, 1, 0.8])
    librosa.display.waveshow(y_full, sr=sr_f, ax=axp, color='skyblue', alpha=0.5)
    axp.axvline(x=trim_range[0], color='red', lw=2, ls='--'); axp.axvline(x=trim_range[1], color='red', lw=2, ls='--')
    axp.set_xlim(0, len(y_full)/sr_f); axp.set_yticks([]); st.pyplot(fig_p)

    st.audio(l_raw[int(trim_range[0]*1000):int(trim_range[1]*1000)].export(io.BytesIO(), format="wav").getvalue())
    
    if st.button("📊 정밀 분석 실행", type="primary"):
        st.session_state.analysis_done = True
        st.session_state.final_y_l = y_full[int(trim_range[0]*sr_f):int(trim_range[1]*sr_f)]
        st.session_state.current_sr = sr_f

if st.session_state.get('analysis_done') and st.session_state.final_y_l is not None:
    y_l, sr = st.session_state.final_y_l, st.session_state.current_sr
    try:
        tts = gTTS(text=target_word, lang='en'); n_mp3 = io.BytesIO(); tts.write_to_fp(n_mp3); n_mp3.seek(0)
        n_seg = safe_trim(AudioSegment.from_file(n_mp3))
        y_n = np.array(n_seg.get_array_of_samples(), dtype=np.float32) / (2**15)
        if n_seg.channels > 1: y_n = y_n.reshape((-1, n_seg.channels)).mean(axis=1)
        y_l, y_n = librosa.util.normalize(y_l), librosa.util.normalize(y_n)
        p_idx_l, env_l = detect_syllable_stress(y_l, sr)
        p_idx_n, env_n = detect_syllable_stress(y_n, sr)
        score, dur_l, dur_n = calculate_score(env_n, env_l)

        st.divider(); c1, c2 = st.columns(2)
        c1.metric("종합 점수", f"{score}점"); c2.metric("강세 구간 비중", f"{dur_l:.1f}%", f"{dur_l-dur_n:+.1f}%")

        fig_res, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6)); mt = max(len(y_l), len(y_n)) / sr
        for ax, y, env, p, t, col in [(ax1, y_l, env_l, p_idx_l, "My Rhythm", "skyblue"), (ax2, y_n, env_n, p_idx_n, "Native Standard", "lightgray")]:
            times = np.linspace(0, len(y)/sr, len(env))
            librosa.display.waveshow(y, sr=sr, ax=ax, color=col, alpha=0.3)
            ax.plot(times, env, color='#1f77b4' if col=="skyblue" else "gray", lw=2.5)
            ax.fill_between(times, 0, env, where=(env > np.max(env)*0.2), color='#1f77b4' if col=="skyblue" else "gray", alpha=0.3)
            ax.axvline(x=times[p], color='red', lw=3); ax.set_title(t); ax.set_xlim(0, mt)
        plt.tight_layout(); st.pyplot(fig_res)
    except Exception as e: st.error(f"오류: {e}")
