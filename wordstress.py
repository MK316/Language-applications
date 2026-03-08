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
        tts.save("native_voice.mp3")
        st.audio("native_voice.mp3")

# --- 4. Step 2: 녹음 및 구간 설정 ---
st.subheader(f"🎯 도전 단어: **{target_word.upper()}**")
audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 녹음 완료", key="word_recorder")

if audio:
    # 새로운 녹음 발생 시 상태 초기화
    if audio['id'] != st.session_state.prev_audio_id:
        st.session_state.analysis_ready = False
        st.session_state.prev_audio_id = audio['id']

    audio_bytes = audio['bytes']
    
    # [해결 핵심] 임시 파일에 명시적으로 저장 후 librosa로 로드
    with open("temp_raw.wav", "wb") as f:
        f.write(audio_bytes)
    
    l_raw_seg = AudioSegment.from_file("temp_raw.wav")
    y_full, sr_full = librosa.load("temp_raw.wav", sr=22050)
    full_duration = len(y_full) / sr_full

    st.divider()
    st.markdown("#### ✂️ Step 3: 분석 구간 설정")
    
    # AI 자동 구간 추천
    auto_s, auto_e = get_speech_bounds(l_raw_seg)
    
    # 파형 시각화와 슬라이더 결합
    trim_range = st.slider("파형을 보고 분석할 음성 구간을 조절하세요 (초):", 
                           0.0, full_duration, 
                           (float(auto_s/1000), float(auto_e/1000)), step=0.01)

    fig_prev, axp = plt.subplots(figsize=(12, 3))
    librosa.display.waveshow(y_full, sr=sr_full, ax=axp, color='skyblue', alpha=0.5)
    axp.axvline(x=trim_range[0], color='red', linestyle='--', linewidth=2)
    axp.axvline(x=trim_range[1], color='red', linestyle='--', linewidth=2)
    axp.set_title("Recording Waveform (Drag slider to adjust red lines)")
    st.pyplot(fig_prev)

    # 선택된 구간 미리 듣기
    start_ms, end_ms = int(trim_range[0] * 1000), int(trim_range[1] * 1000)
    trimmed_audio = l_raw_seg[start_ms:end_ms]
    
    col_play, col_btn = st.columns([2, 1])
    with col_play:
        st.write("🔈 선택된 구간 소리 확인:")
        trimmed_audio.export("temp_trimmed.wav", format="wav")
        st.audio("temp_trimmed.wav")
    
    with col_btn:
        st.write(" ") 
        if st.button("📊 이 구간으로 최종 분석 수행", use_container_width=True):
            st.session_state.analysis_ready = True

# --- 5. Step 4: 상세 분석 결과 ---
if st.session_state.get('analysis_ready'):
    with st.spinner("🎯 강세 에너지 분석 중..."):
        try:
            # 최종 확정된 트리밍 파일 로드
            y_l, sr = librosa.load("temp_trimmed.wav", sr=22050)
            
            # 원어민 데이터 생성
            tts = gTTS(text=target_text if 'target_text' in locals() else target_word, lang='en')
            tts.save("native_temp.mp3")
            n_seg_full = AudioSegment.from_file("native_temp.mp3")
            ns, ne = get_speech_bounds(n_seg_full)
            final_n_seg = n_seg_full[ns:ne]
            final_n_seg.export("native_trimmed.wav", format="wav")
            
            y_n, _ = librosa.load("native_trimmed.wav", sr=sr)
            
            y_l = normalize_audio(y_l); y_n = normalize_audio(y_n)

            st.success("🎉 분석 완료! 아래 탭에서 강세 위치를 대조해 보세요.")
            
            tab1, tab2 = st.tabs(["📊 상세 강세 대조", "✍️ 분석 노트"])
            
            with tab1:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
                librosa.display.waveshow(y_l, sr=sr, ax=ax1, color='#1f77b4')
                librosa.display.waveshow(y_n, sr=sr, ax=ax2, color='#A9A9A9')
                
                # 강세 피크 시각화
                if len(y_l) > 0 and len(y_n) > 0:
                    ax1.axvline(x=librosa.samples_to_time(np.argmax(np.abs(y_l)), sr=sr), color='red', lw=2)
                    ax2.axvline(x=librosa.samples_to_time(np.argmax(np.abs(y_n)), sr=sr), color='red', lw=2)
                
                ax1.set_title("My Adjusted Stress Pattern"); ax2.set_title("Native Standard Pattern")
                plt.tight_layout(); st.pyplot(fig)
                
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("구간 길이", f"{len(y_l)/sr:.2f}s")
                with c2: st.metric("에너지 일치도", f"{calculate_stress_score(y_n, y_l, sr)}점")
                with c3:
                    tip = selected_label.split("(")[1].replace(")", "") if "(" in selected_label else "강세 확인"
                    st.info(f"💡 **강세 위치:** {tip}")

        except Exception as e:
            st.error(f"오류: {e}")
        finally:
            # 임시 파일 정리
            for f in ["temp_raw.wav", "temp_trimmed.wav", "native_temp.mp3", "native_trimmed.wav", "native_voice.mp3"]:
                if os.path.exists(f): os.remove(f)
