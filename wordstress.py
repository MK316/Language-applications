import streamlit as st
from streamlit_mic_recorder import mic_recorder
import io, os, librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from scipy.interpolate import interp1d

# --- [CSS] 슬라이더와 그래프의 시각적 동기화 ---
st.markdown("""
    <style>
    .stSlider { padding-left: 25px; padding-right: 25px; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. 분석 엔진 함수 ---
def get_speech_bounds(audio_segment, silence_thresh=-45, min_silence_len=50):
    intervals = detect_nonsilent(audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    if not intervals: return 0, len(audio_segment)
    return intervals[0][0], intervals[-1][1]

def normalize_audio(y):
    if len(y) == 0: return y
    return librosa.util.normalize(y)

def calculate_stress_metrics(y_n, y_l, sr):
    """강세 일치도 점수 및 타이밍 편차 계산"""
    hop_length = 512
    rms_n = librosa.feature.rms(y=y_n, hop_length=hop_length)[0]
    rms_l = librosa.feature.rms(y=y_l, hop_length=hop_length)[0]
    
    if len(rms_n) < 2 or len(rms_l) < 2: return 0, 0
    
    # 1. 상관계수 기반 일치도 점수
    f_n = interp1d(np.linspace(0, 1, len(rms_n)), rms_n, fill_value="extrapolate")
    f_l = interp1d(np.linspace(0, 1, len(rms_l)), rms_l, fill_value="extrapolate")
    new_x = np.linspace(0, 1, 100)
    correlation = np.corrcoef(f_n(new_x), f_l(new_x))[0, 1]
    score = int(max(0, correlation) * 100) if not np.isnan(correlation) else 0
    
    # 2. 강세 타이밍 편차 (상대적 위치 % 비교)
    peak_n_idx = np.argmax(rms_n)
    peak_l_idx = np.argmax(rms_l)
    peak_n_pos = (peak_n_idx / len(rms_n)) * 100
    peak_l_pos = (peak_l_idx / len(rms_l)) * 100
    timing_diff = peak_l_pos - peak_n_pos # 양수면 늦게, 음수면 빠르게 강세를 줌
    
    return score, timing_diff

# --- 2. 앱 설정 및 데이터 ---
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
    if audio['id'] != st.session_state.prev_audio_id:
        st.session_state.analysis_ready = False
        st.session_state.prev_audio_id = audio['id']

    audio_bytes = audio['bytes']
    with open("temp_raw.wav", "wb") as f: f.write(audio_bytes)
    
    y_full, sr_full = librosa.load("temp_raw.wav", sr=22050)
    l_raw_seg = AudioSegment.from_file("temp_raw.wav")
    full_duration = len(y_full) / sr_full

    st.divider()
    st.markdown("#### ✂️ Step 3: 분석 구간 설정")
    auto_s, auto_e = get_speech_bounds(l_raw_seg)
    
    trim_range = st.slider("파형을 보고 실제 단어 발음 구간을 조절하세요 (초):", 
                           0.0, float(full_duration), 
                           (float(auto_s/1000), float(auto_e/1000)), step=0.01)

    # 파형 프리뷰 (슬라이더 동기화)
    fig_prev = plt.figure(figsize=(12, 2.5))
    axp = fig_prev.add_axes([0.02, 0.2, 0.96, 0.7]) 
    librosa.display.waveshow(y_full, sr=sr_full, ax=axp, color='skyblue', alpha=0.5)
    axp.axvline(x=trim_range[0], color='red', linestyle='--', linewidth=2)
    axp.axvline(x=trim_range[1], color='red', linestyle='--', linewidth=2)
    axp.set_xlim(0, full_duration); axp.set_yticks([]); axp.tick_params(labelsize=8)
    st.pyplot(fig_prev)

    start_ms, end_ms = int(trim_range[0] * 1000), int(trim_range[1] * 1000)
    trimmed_audio = l_raw_seg[start_ms:end_ms]
    
    col_play, col_btn = st.columns([2, 1])
    with col_play:
        trimmed_audio.export("temp_trimmed.wav", format="wav")
        st.audio("temp_trimmed.wav")
    with col_btn:
        st.write(" ") 
        if st.button("📊 이 구간으로 최종 분석 수행", use_container_width=True):
            st.session_state.analysis_ready = True

# --- 5. Step 4: 상세 분석 결과 ---
if st.session_state.get('analysis_ready'):
    with st.spinner("🎯 정밀 분석 중..."):
        try:
            y_l, sr = librosa.load("temp_trimmed.wav", sr=22050)
            tts = gTTS(text=target_word, lang='en')
            tts.save("native_temp.mp3")
            n_seg_f = AudioSegment.from_file("native_temp.mp3")
            ns, ne = get_speech_bounds(n_seg_f)
            n_seg_f[ns:ne].export("native_trimmed.wav", format="wav")
            y_n, _ = librosa.load("native_trimmed.wav", sr=sr)
            
            y_l = normalize_audio(y_l); y_n = normalize_audio(y_n)
            score, timing_diff = calculate_stress_metrics(y_n, y_l, sr)

            st.success("🎉 분석 완료! 나의 강세 리듬을 확인해 보세요.")
            
            c1, c2, c3 = st.columns(3)
            with c1: 
                st.metric("에너지 일치도", f"{score}점")
            with c2:
                # 타이밍 편차 출력 (해석 가이드 포함)
                label = "강세 타이밍 편차"
                val = f"{timing_diff:+.1f}%"
                delta_color = "inverse" if abs(timing_diff) > 10 else "normal"
                st.metric(label, val, help="원어민 대비 강세 위치(%). (+)면 늦게, (-)면 빠르게 강세를 준 것입니다.")
            with c3:
                tip = selected_label.split("(")[1].replace(")", "") if "(" in selected_label else "강세 확인"
                st.info(f"💡 **강세 위치:** {tip}")

            tab1, tab2 = st.tabs(["📊 파형 비교", "✍️ 성찰 노트"])
            with tab1:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
                librosa.display.waveshow(y_l, sr=sr, ax=ax1, color='#1f77b4')
                librosa.display.waveshow(y_n, sr=sr, ax=ax2, color='#A9A9A9')
                
                # 강세 피크(최대 에너지 지점) 시각화
                p_l = librosa.samples_to_time(np.argmax(np.abs(y_l)), sr=sr)
                p_n = librosa.samples_to_time(np.argmax(np.abs(y_n)), sr=sr)
                ax1.axvline(x=p_l, color='red', lw=2, label='My Stress Peak')
                ax2.axvline(x=p_n, color='red', lw=2, label='Native Stress Peak')
                
                ax1.set_title("My Voice Stress Pattern"); ax2.set_title("Native Standard Pattern")
                plt.tight_layout(); st.pyplot(fig)
                
                if abs(timing_diff) > 15:
                    st.warning(f"⚠️ 강세 타이밍이 원어민과 많이 다릅니다 ({'늦음' if timing_diff > 0 else '빠름'}). 리듬에 주의하며 다시 시도해 보세요.")

            with tab2:
                reflection = st.text_area("분석 결과를 바탕으로 개선할 점을 기록하세요.")
                if st.button("마크다운 복사용 텍스트 생성"):
                    st.code(f"### Word Stress Analysis: {target_word}\n- Score: {score}\n- Timing Deviation: {timing_diff:+.1f}%\n- Reflection: {reflection}")

        except Exception as e:
            st.error(f"오류: {e}")
        finally:
            for f in ["temp_raw.wav", "temp_trimmed.wav", "native_temp.mp3", "native_trimmed.wav", "native_voice.mp3"]:
                if os.path.exists(f): os.remove(f)
