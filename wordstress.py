import streamlit as st
from streamlit_mic_recorder import mic_recorder
import io, os, librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from scipy.interpolate import interp1d

# --- [추가] 슬라이더 너비 조정을 위한 CSS ---
st.markdown("""
    <style>
    .stSlider {
        padding-left: 25px;
        padding-right: 25px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. 분석 엔진 함수 ---
def get_speech_bounds(audio_segment, silence_thresh=-45, min_silence_len=50):
    intervals = detect_nonsilent(audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    if not intervals: return 0, len(audio_segment)
    return intervals[0][0], intervals[-1][1]

# --- 2. 설정 및 데이터 ---
st.set_page_config(page_title="Word Stress Master", layout="wide")
if 'analysis_ready' not in st.session_state: st.session_state.analysis_ready = False

word_db = {"Photograph (1음절 강세)": "photograph", "Photographer (2음절 강세)": "photographer", "Education (3음절 강세)": "education"}
selected_label = st.selectbox("학습할 단어 선택:", list(word_db.keys()))
target_word = word_db[selected_label]

# --- 3. 녹음 및 구간 설정 (시각적 일치 핵심) ---
audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 녹음 완료", key="word_recorder")

if audio:
    audio_bytes = audio['bytes']
    with open("temp_raw.wav", "wb") as f: f.write(audio_bytes)
    
    y_full, sr_f = librosa.load("temp_raw.wav", sr=22050)
    l_raw_seg = AudioSegment.from_file("temp_raw.wav")
    full_dur = len(y_full) / sr_f

    st.divider()
    st.markdown("#### ✂️ Step 3: 분석 구간 설정")
    
    auto_s, auto_e = get_speech_bounds(l_raw_seg)
    
    # 1. 슬라이더 배치
    trim_range = st.slider("파형과 슬라이더 위치를 맞춰 구간을 조절하세요:", 
                           0.0, float(full_dur), 
                           (float(auto_s/1000), float(auto_e/1000)), step=0.01)

    # 2. 그래프 여백 제거 (여백을 0으로 설정하여 슬라이더와 위치 동기화)
    fig_prev = plt.figure(figsize=(12, 2.5))
    axp = fig_prev.add_axes([0, 0.1, 1, 0.8]) # [left, bottom, width, height] 여백 최소화
    
    librosa.display.waveshow(y_full, sr=sr_f, ax=axp, color='skyblue', alpha=0.5)
    axp.axvline(x=trim_range[0], color='red', linestyle='--', linewidth=2)
    axp.axvline(x=trim_range[1], color='red', linestyle='--', linewidth=2)
    
    # X축 범위를 실제 오디오 길이와 정확히 일치시킴
    axp.set_xlim(0, full_dur)
    axp.axis('off') # 눈금을 숨겨서 슬라이더와 더 일직선으로 보이게 함 (필요시 제거 가능)
    st.pyplot(fig_prev)

    # 미리 듣기 및 분석 버튼
    start_ms, end_ms = int(trim_range[0] * 1000), int(trim_range[1] * 1000)
    trimmed_audio = l_raw_seg[start_ms:end_ms]
    
    cp, cb = st.columns([2, 1])
    with cp:
        buf = io.BytesIO(); trimmed_audio.export(buf, format="wav")
        st.audio(buf)
    with cb:
        if st.button("📊 이 구간으로 최종 분석", use_container_width=True):
            st.session_state.analysis_ready = True
            st.session_state.trimmed_wav = buf.getvalue()

# --- 4. 분석 결과 (생략 - 이전 로직과 동일) ---
if st.session_state.analysis_ready:
    st.success("분석 결과가 하단에 노출됩니다.")
    # (이전 분석 코드 조각들...)
