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

def get_rms_envelope(y, hop_length=256):
    """단순 피크가 아닌 에너지 덩어리(RMS)를 추출하여 매끄럽게 만듭니다."""
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    # 이동 평균을 사용하여 자잘한 노이즈 제거 (Smooth)
    return np.convolve(rms, np.ones(5)/5, mode='same')

def calculate_stress_metrics(y_n, y_l, sr):
    env_n = get_rms_envelope(y_n)
    env_l = get_rms_envelope(y_l)
    
    if len(env_n) < 2 or len(env_l) < 2: return 0, 0
    
    # 1. 상관계수 점수
    f_n = interp1d(np.linspace(0, 1, len(env_n)), env_n, fill_value="extrapolate")
    f_l = interp1d(np.linspace(0, 1, len(env_l)), env_l, fill_value="extrapolate")
    new_x = np.linspace(0, 1, 100)
    correlation = np.corrcoef(f_n(new_x), f_l(new_x))[0, 1]
    score = int(max(0, correlation) * 100) if not np.isnan(correlation) else 0
    
    # 2. 강세 위치 (에너지 덩어리가 가장 큰 지점)
    peak_n_pos = (np.argmax(env_n) / len(env_n)) * 100
    peak_l_pos = (np.argmax(env_l) / len(env_l)) * 100
    timing_diff = peak_l_pos - peak_n_pos
    
    return score, timing_diff, env_n, env_l

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
    with st.spinner("🎯 에너지 포괄선 분석 중..."):
        try:
            y_l, sr = librosa.load("temp_trimmed.wav", sr=22050)
            tts = gTTS(text=target_word, lang='en')
            tts.save("native_temp.mp3")
            n_seg_f = AudioSegment.from_file("native_temp.mp3")
            ns, ne = get_speech_bounds(n_seg_f)
            n_seg_f[ns:ne].export("native_trimmed.wav", format="wav")
            y_n, _ = librosa.load("native_trimmed.wav", sr=sr)
            
            y_l = normalize_audio(y_l); y_n = normalize_audio(y_n)
            score, timing_diff, env_n, env_l = calculate_stress_metrics(y_n, y_l, sr)

            st.success("🎉 분석 완료! 에너지 덩어리가 가장 큰 지점을 확인하세요.")
            
            tab1, tab2 = st.tabs(["📊 에너지 패턴 대조", "✍️ 성찰 노트"])
            with tab1:
                # [개선] 파형 위에 에너지 포괄선(Envelope)을 함께 그립니다.
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
                
                # 학습자
                librosa.display.waveshow(y_l, sr=sr, ax=ax1, color='skyblue', alpha=0.4)
                times_l = np.linspace(0, len(y_l)/sr, len(env_l))
                ax1.plot(times_l, env_l, color='#1f77b4', lw=2, label='Energy Envelope')
                peak_l_time = times_l[np.argmax(env_l)]
                ax1.axvline(x=peak_l_time, color='red', lw=3, label='Stress Peak')
                
                # 원어민
                librosa.display.waveshow(y_n, sr=sr, ax=ax2, color='lightgray', alpha=0.4)
                times_n = np.linspace(0, len(y_n)/sr, len(env_n))
                ax2.plot(times_n, env_n, color='gray', lw=2, label='Energy Envelope')
                peak_n_time = times_n[np.argmax(env_n)]
                ax2.axvline(x=peak_n_time, color='red', lw=3, label='Stress Peak')
                
                ax1.set_title("My Energy Pattern (Blue line: RMS Envelope)"); ax2.set_title("Native Energy Pattern")
                ax1.legend(); ax2.legend()
                plt.tight_layout(); st.pyplot(fig)
                
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("에너지 일치도", f"{score}점")
                with c2: st.metric("강세 타이밍 편차", f"{timing_diff:+.1f}%")
                with c3:
                    tip = selected_label.split("(")[1].replace(")", "") if "(" in selected_label else "강세 확인"
                    st.info(f"💡 **강세 위치:** {tip}")

        except Exception as e:
            st.error(f"오류: {e}")
        finally:
            for f in ["temp_raw.wav", "temp_trimmed.wav", "native_temp.mp3", "native_trimmed.wav", "native_voice.mp3"]:
                if os.path.exists(f): os.remove(f)
