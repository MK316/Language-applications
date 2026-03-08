import streamlit as st
from streamlit_mic_recorder import mic_recorder
import io, os, librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from scipy.interpolate import interp1d

# --- 1. 분석 엔진 함수 ---
def get_speech_bounds(audio_segment, silence_thresh=-45, min_silence_len=50):
    intervals = detect_nonsilent(audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    if not intervals: return 0, len(audio_segment)
    return intervals[0][0], intervals[-1][1]

def normalize_audio(y):
    if len(y) == 0: return y
    return librosa.util.normalize(y)

def calculate_stress_score(y_n, y_l, sr):
    hop_length = 512
    rms_n = librosa.feature.rms(y=y_n, hop_length=hop_length)[0]
    rms_l = librosa.feature.rms(y=y_l, hop_length=hop_length)[0]
    if len(rms_n) < 2 or len(rms_l) < 2: return 0
    f_n = interp1d(np.linspace(0, 1, len(rms_n)), rms_n, fill_value="extrapolate")
    f_l = interp1d(np.linspace(0, 1, len(rms_l)), rms_l, fill_value="extrapolate")
    new_x = np.linspace(0, 1, 100)
    correlation = np.corrcoef(f_n(new_x), f_l(new_x))[0, 1]
    return int(max(0, correlation) * 100) if not np.isnan(correlation) else 0

# --- 2. 앱 설정 및 세션 초기화 ---
st.set_page_config(page_title="Word Stress Master", layout="wide")
st.title("🎙️ Word Stress & Amplitude Master")

if 'analysis_ready' not in st.session_state: st.session_state.analysis_ready = False
if 'prev_audio_id' not in st.session_state: st.session_state.prev_audio_id = None

word_db = {
    "Photograph (1음절 강세)": "photograph",
    "Photographer (2음절 강세)": "photographer",
    "Record (Noun - 1음절)": "record",
    "Record (Verb - 2음절)": "record",
    "Education (3음절 강세)": "education",
}

# --- 3. Step 1: 단어 선택 ---
with st.sidebar:
    st.header("📍 Step 1: 단어 선택")
    selected_label = st.selectbox("학습할 단어를 선택하세요:", list(word_db.keys()))
    target_word = word_db[selected_label]
    if st.button("🔊 원어민 표준 발음 듣기"):
        tts = gTTS(text=target_word, lang='en')
        mp3_fp = io.BytesIO(); tts.write_to_fp(mp3_fp); mp3_fp.seek(0)
        st.audio(mp3_fp)

# --- 4. Step 2: 녹음 및 구간 설정 ---
st.subheader(f"🎯 도전 단어: **{target_word.upper()}**")
audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 녹음 완료", key="word_recorder")

if audio:
    if audio['id'] != st.session_state.prev_audio_id:
        st.session_state.analysis_ready = False
        st.session_state.prev_audio_id = audio['id']

    audio_bytes = audio['bytes']
    # 전체 녹음 데이터 로드 (파형 표시용)
    y_full, sr_full = librosa.load(io.BytesIO(audio_bytes), sr=22050)
    l_raw_seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
    full_duration = len(y_full) / sr_full

    st.divider()
    st.markdown("#### ✂️ Step 3: 분석 구간 설정")
    
    # 1. AI 자동 감지 구간 계산
    auto_s, auto_e = get_speech_bounds(l_raw_seg)
    
    # 2. 슬라이더로 구간 선택
    trim_range = st.slider("파형을 보고 분석할 음성 구간을 조절하세요 (초):", 
                           0.0, full_duration, 
                           (float(auto_s/1000), float(auto_e/1000)), step=0.01)

    # 3. 실시간 파형 프리뷰 (슬라이더 값에 따라 빨간 선 표시)
    fig_prev, axp = plt.subplots(figsize=(12, 3))
    librosa.display.waveshow(y_full, sr=sr_full, ax=axp, color='skyblue', alpha=0.5)
    axp.axvline(x=trim_range[0], color='red', linestyle='--', linewidth=2)
    axp.axvline(x=trim_range[1], color='red', linestyle='--', linewidth=2)
    axp.set_title("Full Recording Waveform (Red lines: Selection)")
    st.pyplot(fig_prev)

    # 4. 선택된 구간 미리 듣기
    start_ms, end_ms = int(trim_range[0] * 1000), int(trim_range[1] * 1000)
    trimmed_audio = l_raw_seg[start_ms:end_ms]
    
    col_play, col_btn = st.columns([2, 1])
    with col_play:
        st.write("🔈 선택된 구간 소리 확인:")
        playback_buffer = io.BytesIO()
        trimmed_audio.export(playback_buffer, format="wav")
        st.audio(playback_buffer)
    
    with col_btn:
        st.write(" ") 
        if st.button("📊 이 구간으로 최종 분석 수행", use_container_width=True):
            st.session_state.analysis_ready = True
            st.session_state.trimmed_wav = playback_buffer.getvalue()

# --- 5. Step 4: 상세 분석 결과 ---
if st.session_state.get('analysis_ready'):
    with st.spinner("🎯 강세 패턴 분석 중..."):
        try:
            y_l, sr = librosa.load(io.BytesIO(st.session_state.trimmed_wav), sr=22050)
            
            # 원어민 데이터 생성
            tts = gTTS(text=target_word, lang='en')
            n_fp = io.BytesIO(); tts.write_to_fp(n_fp); n_fp.seek(0)
            n_seg_full = AudioSegment.from_file(n_fp)
            ns, ne = get_speech_bounds(n_seg_full)
            final_n_seg = n_seg_full[ns:ne]
            
            n_buf = io.BytesIO(); final_n_seg.export(n_buf, format="wav"); n_buf.seek(0)
            y_n, _ = librosa.load(n_buf, sr=sr)
            
            y_l = normalize_audio(y_l); y_n = normalize_audio(y_n)

            st.success("🎉 분석 완료! 원어민의 강세 에너지 패턴과 비교해 보세요.")
            
            tab1, tab2 = st.tabs(["📊 상세 강세 대조", "✍️ 분석 노트"])
            
            with tab1:
                # 분석용 그래프 (나란히 배치)
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
                librosa.display.waveshow(y_l, sr=sr, ax=ax1, color='#1f77b4')
                librosa.display.waveshow(y_n, sr=sr, ax=ax2, color='#A9A9A9')
                
                # 강세 피크(최대 진폭) 자동 감지 및 표시
                if len(y_l) > 0 and len(y_n) > 0:
                    ax1.axvline(x=librosa.samples_to_time(np.argmax(np.abs(y_l)), sr=sr), color='red', lw=2)
                    ax2.axvline(x=librosa.samples_to_time(np.argmax(np.abs(y_n)), sr=sr), color='red', lw=2)
                
                ax1.set_title("My Stress (Adjusted Selection)"); ax2.set_title("Native Standard Stress")
                plt.tight_layout(); st.pyplot(fig)
                
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("구간 길이", f"{len(y_l)/sr:.2f}s")
                with c2: st.metric("에너지 패턴 일치도", f"{calculate_stress_score(y_n, y_l, sr)}점")
                with c3:
                    tip = selected_label.split("(")[1].replace(")", "") if "(" in selected_label else "강세 확인"
                    st.info(f"💡 **강세 위치:** {tip}")

        except Exception as e:
            st.error(f"오류: {e}")
