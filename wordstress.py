import streamlit as st
from streamlit_mic_recorder import mic_recorder
import io, os, librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from scipy.interpolate import interp1d

# --- [1] 모바일 정밀 레이아웃 및 버튼 통합 CSS ---
st.set_page_config(page_title="Word Stress Master", layout="centered")

st.markdown("""
    <style>
    /* 전체 여백 조정 */
    .main .block-container { padding-top: 1rem; }
    
    /* [핵심] 컬럼 간격 제거하여 버튼을 하나로 붙임 */
    div[data-testid="stHorizontalBlock"] {
        gap: 0px !important;
    }

    /* 녹음 위젯과 일반 버튼의 높이, 글자 크기, 테두리를 동일하게 강제 고정 */
    div[data-testid="stHorizontalBlock"] .stButton button, 
    div[data-testid="stHorizontalBlock"] .stMicrophone button {
        height: 50px !important;      /* 높이 50px 고정 */
        font-size: 16px !important;    /* 글자 크기 16px 고정 */
        font-weight: bold !important;
        margin: 0px !important;
        width: 100% !important;
        border-radius: 0px !important; /* 일단 모서리 제거 */
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    /* 왼쪽 녹음 버튼 스타일: 왼쪽 모서리만 둥글게 */
    div[data-testid="stHorizontalBlock"] > div:nth-child(1) button {
        background-color: #ff4b4b !important;
        color: white !important;
        border: 1px solid #ff4b4b !important;
        border-top-left-radius: 12px !important;
        border-bottom-left-radius: 12px !important;
    }

    /* 오른쪽 리셋 버튼 스타일: 오른쪽 모서리만 둥글게 및 경계선 처리 */
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) button {
        background-color: #f0f2f6 !important;
        color: #31333F !important;
        border: 1px solid #dcdcdc !important;
        border-left: none !important; /* 버튼 사이 선 중복 제거 */
        border-top-right-radius: 12px !important;
        border-bottom-right-radius: 12px !important;
    }
    
    /* 슬라이더 패딩 제거 */
    .stSlider { padding: 0px !important; }
    </style>
    """, unsafe_allow_html=True)

# 세션 상태 초기화
if 'reset_key' not in st.session_state: st.session_state.reset_key = 0
if 'last_audio_id' not in st.session_state: st.session_state.last_audio_id = None
if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
if 'final_y_l' not in st.session_state: st.session_state.final_y_l = None

# --- [2] 분석 유틸리티 함수 ---
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

# --- [3] 메인 UI 섹션 ---
st.title("🎙️ Word Stress Master")

word_db = {
    "Photograph (1음절 강세)": "photograph",
    "Photographer (2음절 강세)": "photographer",
    "Education (3음절 강세)": "education",
    "Record (Noun - 1음절)": "record",
    "Record (Verb - 2음절)": "record"
}

selected_label = st.selectbox("학습할 단어 선택:", list(word_db.keys()))
target_word = word_db[selected_label]

if st.button("🔊 원어민 발음 듣기"):
    tts = gTTS(text=target_word, lang='en')
    mp3_buf = io.BytesIO()
    tts.write_to_fp(mp3_buf)
    st.audio(mp3_buf.getvalue())

st.divider()

# --- [4] 핵심 수정: 버튼 세그먼트 배치 (녹음/리셋) ---
st.subheader(f"🎯 연습: {target_word.upper()}")

col_l, col_r = st.columns(2)

with col_l:
    # 녹음 위젯
    audio = mic_recorder(
        start_prompt="🎤 녹음 시작",
        stop_prompt="🛑 완료",
        key=f"rec_{st.session_state.reset_key}"
    )

with col_r:
    # 리셋 버튼
    if st.button("🔄 리셋"):
        st.session_state.reset_key += 1 # Key를 바꿔서 녹음 위젯 완전 초기화
        st.session_state.last_audio_id = None
        st.session_state.analysis_done = False
        st.session_state.final_y_l = None
        st.rerun()

# --- [5] 구간 설정 및 결과 분석 ---
if audio:
    if audio['id'] != st.session_state.last_audio_id:
        st.session_state.last_audio_id = audio['id']
        st.session_state.analysis_done = False
        st.session_state.final_y_l = None

    l_raw = AudioSegment.from_file(io.BytesIO(audio['bytes']))
    y_full = np.array(l_raw.get_array_of_samples(), dtype=np.float32) / (2**15)
    if l_raw.channels > 1: y_full = y_full.reshape((-1, l_raw.channels)).mean(axis=1)
    sr_f = l_raw.frame_rate
    
    st.markdown("#### ✂️ 분석 구간 설정")
    auto_bounds = detect_nonsilent(l_raw, min_silence_len=100, silence_thresh=-45)
    as_ms, ae_ms = auto_bounds[0] if auto_bounds else (0, len(l_raw))
    
    trim_range = st.slider("단어 구간 조절:", 
                           0.0, float(len(y_full)/sr_f), 
                           (float(as_ms/1000), float(ae_ms/1000)), step=0.01)

    fig_p = plt.figure(figsize=(10, 2.2))
    axp = fig_p.add_axes([0, 0.2, 1, 0.8])
    librosa.display.waveshow(y_full, sr=sr_f, ax=axp, color='skyblue', alpha=0.5)
    axp.axvline(x=trim_range[0], color='red', lw=2, ls='--')
    axp.axvline(x=trim_range[1], color='red', lw=2, ls='--')
    axp.set_xlim(0, len(y_full)/sr_f); axp.set_yticks([]); st.pyplot(fig_p)

    trimmed_audio = l_raw[int(trim_range[0]*1000):int(trim_range[1]*1000)]
    st.audio(trimmed_audio.export(io.BytesIO(), format="wav").getvalue())
    
    if st.button("📊 정밀 분석 실행", type="primary"):
        st.session_state.analysis_done = True
        st.session_state.final_y_l = y_full[int(trim_range[0]*sr_f):int(trim_range[1]*sr_f)]
        st.session_state.current_sr = sr_f

# 결과 분석 출력 섹션 (이하 동일...)
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
        
        st.divider(); c1, c2 = st.columns(2)
        c1.metric("강세 정확도", "분석 완료"); c2.metric("강세 비중", f"{(np.sum(env_l > np.max(env_l)*0.2)/len(env_l))*100:.1f}%")

        fig_res, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6)); mt = max(len(y_l), len(y_n)) / sr
        for ax, y, env, p, t, col in [(ax1, y_l, env_l, p_idx_l, "My Rhythm", "skyblue"), (ax2, y_n, env_n, p_idx_n, "Native Standard", "lightgray")]:
            times = np.linspace(0, len(y)/sr, len(env))
            librosa.display.waveshow(y, sr=sr, ax=ax, color=col, alpha=0.3)
            ax.plot(times, env, color='#1f77b4' if col=="skyblue" else "gray", lw=2.5)
            ax.axvline(x=times[p], color='red', lw=3); ax.set_xlim(0, mt)
        st.pyplot(fig_res)
    except Exception as e: st.error(f"오류: {e}")
