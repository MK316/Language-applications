import streamlit as st
from streamlit_mic_recorder import mic_recorder
import io, os, librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from scipy.interpolate import interp1d

# --- [1] 모바일 정밀 디자인: 강제 규격 통일 ---
st.set_page_config(page_title="Word Stress Master", layout="centered")

st.markdown("""
    <style>
    .main .block-container { padding-top: 1rem; }

    /* [핵심] 모든 버튼의 높이와 폰트를 절대 수치로 강제 고정 */
    button, .stMicrophone button {
        height: 54px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: none !important;
        margin: 0 !important;
    }

    /* 버튼을 감싸는 컬럼의 정렬을 강제로 일치시킴 */
    [data-testid="stHorizontalBlock"] {
        align-items: center !important;
        gap: 10px !important;
    }

    /* 녹음 버튼 (빨간색) */
    [data-testid="stHorizontalBlock"] > div:nth-child(1) button {
        background-color: #ff4b4b !important;
        color: white !important;
        border: none !important;
    }

    /* 리셋 버튼 (회색) */
    [data-testid="stHorizontalBlock"] > div:nth-child(2) button {
        background-color: #f0f2f6 !important;
        color: #31333F !important;
        border: 1px solid #dcdcdc !important;
    }
    
    /* 슬라이더 패딩 제거 */
    .stSlider { padding: 0 !important; }
    </style>
    """, unsafe_allow_html=True)

# 세션 상태 초기화
if 'reset_key' not in st.session_state: st.session_state.reset_key = 0
if 'last_audio_id' not in st.session_state: st.session_state.last_audio_id = None
if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
if 'final_y_l' not in st.session_state: st.session_state.final_y_l = None

# --- [2] 분석 엔진 함수 ---
def get_rms_envelope(y, hop_length=256):
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    return np.convolve(rms, np.ones(5)/5, mode='same')

def safe_trim(audio_segment):
    bounds = detect_nonsilent(audio_segment, min_silence_len=100, silence_thresh=-50)
    if not bounds: return audio_segment
    start, end = bounds[0]
    return audio_segment[start:end] if (end - start) > 200 else audio_segment

# --- [3] 메인 UI ---
st.title("🎙️ Word Stress Master")
word_db = {"Photograph (1음절 강세)": "photograph", "Photographer (2음절 강세)": "photographer", "Education (3음절 강세)": "education"}
target_word = word_db[st.selectbox("단어 선택:", list(word_db.keys()))]

if st.button("🔊 원어민 표준 발음 듣기"):
    tts = gTTS(text=target_word, lang='en'); mp3 = io.BytesIO(); tts.write_to_fp(mp3); st.audio(mp3.getvalue())

st.divider()
st.subheader(f"🎯 연습: {target_word.upper()}")

# --- [4] 핵심 수정: 버튼 수직/수평 정렬 강제 고정 ---
c1, c2 = st.columns(2)
with c1:
    audio = mic_recorder(start_prompt="녹음 시작", stop_prompt="완료", key=f"rec_{st.session_state.reset_key}")
with c2:
    if st.button("리셋"):
        st.session_state.reset_key += 1
        st.session_state.last_audio_id, st.session_state.analysis_done, st.session_state.final_y_l = None, False, None
        st.rerun()

# --- [5] 구간 설정 및 분석 로직 ---
if audio:
    if audio['id'] != st.session_state.last_audio_id:
        st.session_state.last_audio_id, st.session_state.analysis_done = audio['id'], False

    l_raw = AudioSegment.from_file(io.BytesIO(audio['bytes']))
    y_full = np.array(l_raw.get_array_of_samples(), dtype=np.float32) / (2**15)
    if l_raw.channels > 1: y_full = y_full.reshape((-1, l_raw.channels)).mean(axis=1)
    sr_f = l_raw.frame_rate
    
    st.markdown("#### ✂️ 분석 구간 설정")
    auto_b = detect_nonsilent(l_raw, min_silence_len=100, silence_thresh=-45)
    as_ms, ae_ms = auto_b[0] if auto_b else (0, len(l_raw))
    trim_range = st.slider("구간 조절:", 0.0, float(len(y_full)/sr_f), (float(as_ms/1000), float(ae_ms/1000)), step=0.01)

    fig_p = plt.figure(figsize=(10, 2.2)); axp = fig_p.add_axes([0, 0.2, 1, 0.8])
    librosa.display.waveshow(y_full, sr=sr_f, ax=axp, color='skyblue', alpha=0.5)
    axp.axvline(x=trim_range[0], color='red', lw=2, ls='--'); axp.axvline(x=trim_range[1], color='red', lw=2, ls='--')
    axp.set_xlim(0, len(y_full)/sr_f); axp.set_yticks([]); st.pyplot(fig_p)

    if st.button("📊 정밀 분석 실행", type="primary"):
        st.session_state.analysis_done = True
        st.session_state.final_y_l = y_full[int(trim_range[0]*sr_f):int(trim_range[1]*sr_f)]
        st.session_state.current_sr = sr_f

# --- [6] 분석 결과 출력 ---
if st.session_state.get('analysis_done') and st.session_state.final_y_l is not None:
    y_l, sr = st.session_state.final_y_l, st.session_state.current_sr
    # (결과 시각화 로직 - 이전과 동일)
    st.success("분석이 완료되었습니다.")
