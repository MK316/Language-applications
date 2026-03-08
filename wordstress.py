import streamlit as st
from streamlit_mic_recorder import mic_recorder
import io, os, librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from scipy.interpolate import interp1d

# --- [1] 모바일 레이아웃 및 CSS (디자인 핵심) ---
st.set_page_config(page_title="Word Stress Master", layout="centered")

st.markdown("""
    <style>
    /* 슬라이더와 그래프 너비 동기화 */
    .stSlider { padding-left: 0px; padding-right: 0px; }
    .main .block-container { padding-top: 1rem; }
    
    /* [핵심 수정] 리셋 및 녹음 버튼 스타일 통일 및 모바일 최적화 */
    div.stButton > button, div[data-testid="stVerticalBlock"] > div button {
        width: 100% !important;
        height: 3.8em !important; /* 버튼 높이를 키워 터치 편의성 향상 */
        font-weight: bold !important;
        font-size: 16px !important;
        border-radius: 12px !important;
        margin-bottom: 10px !important; /* 버튼 간 간격 확보 */
    }
    
    /* 리셋 버튼 색상 (회색 계열) */
    div[data-testid="stVerticalBlock"] > div:nth-child(2) button {
        background-color: #f0f2f6 !important;
        color: #31333F !important;
        border: 1px solid #dcdcdc !important;
    }
    
    /* 녹음 버튼 색상 (주요 강조색) */
    div[data-testid="stVerticalBlock"] > div:nth-child(3) button {
        background-color: #ff4b4b !important;
        color: white !important;
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 세션 상태 초기화
for key in ['last_audio_id', 'analysis_done', 'final_y_l']:
    if key not in st.session_state: st.session_state[key] = None if key != 'analysis_done' else False

# --- [2] 분석 로직 (이전과 동일) ---
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

# --- [3] 앱 UI: 단어 선택 ---
st.title("🎙️ Word Stress Master")

word_db = {
    "Photograph (1음절 강세)": "photograph",
    "Photographer (2음절 강세)": "photographer",
    "Education (3음절 강세)": "education",
    "Record (Noun - 1음절)": "record",
    "Record (Verb - 2음절)": "record"
}

selected_label = st.selectbox("학습할 단어를 선택하세요:", list(word_db.keys()))
target_word = word_db[selected_label]

if st.button("🔊 원어민 표준 발음 듣기"):
    tts = gTTS(text=target_word, lang='en')
    mp3_buf = io.BytesIO()
    tts.write_to_fp(mp3_buf)
    st.audio(mp3_buf.getvalue())

st.divider()

# --- [4] 핵심 수정: 리셋 및 녹음 컨트롤 (세로 배치 및 크기 통일) ---
st.subheader(f"🎯 연습: {target_word.upper()}")

# [변경] 리셋 버튼을 녹음 버튼 위에 배치하고, 크기를 통일
if st.button("🔄 리셋 (다시 하기)"):
    st.session_state.last_audio_id = None
    st.session_state.analysis_done = False
    st.session_state.final_y_l = None
    st.rerun()

# 녹음 버튼 (위 리셋 버튼과 크기/디자인 통일)
audio = mic_recorder(
    start_prompt="🎤 녹음 시작",
    stop_prompt="🛑 녹음 완료",
    key="word_recorder"
)

# --- [5] 녹음 이후 단계 (이전과 동일) ---
if audio:
    # (이하 구간 설정 및 분석 로직은 이전과 동일하므로 생략 - 전체 코드 참조)
    # ...
