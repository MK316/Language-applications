import streamlit as st
from streamlit_mic_recorder import mic_recorder
import io, os, librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from scipy.interpolate import interp1d

# --- [CSS] UI 정밀 조정 ---
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

def get_rms_envelope(y, hop_length=256):
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    return np.convolve(rms, np.ones(5)/5, mode='same')

def calculate_stress_metrics(y_n, y_l, sr):
    env_n = get_rms_envelope(y_n)
    env_l = get_rms_envelope(y_l)
    
    if len(env_n) < 2 or len(env_l) < 2: return 0, 0, 0, 0
    
    # 1. 에너지 패턴 일치도 (상관계수)
    f_n = interp1d(np.linspace(0, 1, len(env_n)), env_n, fill_value="extrapolate")
    f_l = interp1d(np.linspace(0, 1, len(env_l)), env_l, fill_value="extrapolate")
    new_x = np.linspace(0, 1, 100)
    correlation = np.corrcoef(f_n(new_x), f_l(new_x))[0, 1]
    score = int(max(0, correlation) * 100) if not np.isnan(correlation) else 0
    
    # 2. 강세 음절 길이 비중 (Duration Ratio)
    # 에너지가 임계값(최대치의 20%) 이상인 구간을 음절 발화 구간으로 간주
    thresh_n = np.max(env_n) * 0.2
    thresh_l = np.max(env_l) * 0.2
    duration_n = np.sum(env_n > thresh_n) / len(env_n) * 100
    duration_l = np.sum(env_l > thresh_l) / len(env_l) * 100
    
    # 3. 강세 타이밍 (Peak Position %)
    peak_n_pos = (np.argmax(env_n) / len(env_n)) * 100
    peak_l_pos = (np.argmax(env_l) / len(env_l)) * 100
    timing_diff = peak_l_pos - peak_n_pos
    
    return score, timing_diff, duration_n, duration_l, env_n, env_l

# --- 2. 앱 설정 및 데이터 ---
st.set_page_config(page_title="Word Stress Master", layout="wide")
st.title("🎙️ Word Stress & Duration Master")

word_db = {
    "Photograph (1음절 강세)": "photograph",
    "Photographer (2음절 강세)": "photographer",
    "Record (Noun - 1음절)": "record",
    "Record (Verb - 2음절)": "record",
    "Education (3음절 강세)": "education",
}

if 'analysis_ready' not in st.session_state: st.session_state.analysis_ready = False
if 'prev_audio_id' not in st.session_state: st.session_state.prev_audio_id = None

# --- 3. Step 1: 단어 선택 ---
with st.sidebar:
    st.header("📍 Step 1: 단어 선택")
    selected_label = st.selectbox("학습할 단어를 선택하세요:", list(word_db.keys()))
    target_word = word_db[selected_label]
    if st.button("🔊 원어민 발음 듣기"):
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
    trim_range = st.slider("파형을 보고 실제 단어 구간을 조절하세요:", 
                           0.0, float(full_duration), 
                           (float(auto_s/1000), float(auto_e/1000)), step=0.01)

    fig_prev = plt.figure(figsize=(12, 2.5))
    axp = fig_prev.add_axes([0.02, 0.2, 0.96, 0.7]) 
    librosa.display.waveshow(y_full, sr=sr_full, ax=axp, color='skyblue', alpha=0.5)
    axp.axvline(x=trim_range[0], color='red', lw=2, ls='--')
    axp.axvline(x=trim_range[1], color='red', lw=2, ls='--')
    axp.set_xlim(0, full_duration); axp.set_yticks([]); axp.tick_params(labelsize=8)
    st.pyplot(fig_prev)

    start_ms, end_ms = int(trim_range[0] * 1000), int(trim_range[1] * 1000)
    trimmed_audio = l_raw_seg[start_ms:end_ms]
    
    cp, cb = st.columns([2, 1])
    with cp:
        trimmed_audio.export("temp_trimmed.wav", format="wav")
        st.audio("temp_trimmed.wav")
    with cb:
        if st.button("📊 이 구간으로 상세 분석 수행", use_container_width=True):
            st.session_state.analysis_ready = True

# --- 5. Step 4: 상세 분석 결과 ---
if st.session_state.get('analysis_ready'):
    with st.spinner("🎯 리듬 및 에너지 비율 분석 중..."):
        try:
            y_l, sr = librosa.load("temp_trimmed.wav", sr=22050)
            tts = gTTS(text=target_word, lang='en')
            tts.save("native_temp.mp3")
            n_seg_f = AudioSegment.from_file("native_temp.mp3")
            ns, ne = get_speech_bounds(n_seg_f)
            n_seg_f[ns:ne].export("native_trimmed.wav", format="wav")
            y_n, _ = librosa.load("native_trimmed.wav", sr=sr)
            
            y_l = normalize_audio(y_l); y_n = normalize_audio(y_n)
            score, timing_diff, dur_n, dur_l, env_n, env_l = calculate_stress_metrics(y_n, y_l, sr)

            st.success("🎉 분석 완료! 강세 음절의 '길이' 비중을 확인하세요.")
            
            # 메트릭 표시
            m1, m2, m3 = st.columns(3)
            with m1: st.metric("에너지 패턴 점수", f"{score}점")
            with m2: 
                # 길이 비율 비교
                diff_dur = dur_l - dur_n
                st.metric("강세 구간 길이 비중", f"{dur_l:.1f}%", 
                          delta=f"{diff_dur:+.1f}% (원어민 {dur_n:.1f}%)",
                          help="단어 전체에서 강세 에너지가 유지된 시간의 비중입니다.")
            with m3:
                st.metric("강세 타이밍 편차", f"{timing_diff:+.1f}%")

            tab1, tab2 = st.tabs(["📊 리듬 및 에너지 분석", "✍️ 분석 가이드"])
            with tab1:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
                
                # 학습자 그래프
                times_l = np.linspace(0, len(y_l)/sr, len(env_l))
                librosa.display.waveshow(y_l, sr=sr, ax=ax1, color='skyblue', alpha=0.3)
                ax1.plot(times_l, env_l, color='#1f77b4', lw=2)
                ax1.axvline(x=times_l[np.argmax(env_l)], color='red', lw=3)
                ax1.fill_between(times_l, 0, env_l, where=(env_l > np.max(env_l)*0.2), color='blue', alpha=0.1, label='Stressed Area')
                
                # 원어민 그래프
                times_n = np.linspace(0, len(y_n)/sr, len(env_n))
                librosa.display.waveshow(y_n, sr=sr, ax=ax2, color='lightgray', alpha=0.3)
                ax2.plot(times_n, env_n, color='gray', lw=2)
                ax2.axvline(x=times_n[np.argmax(env_n)], color='red', lw=3)
                ax2.fill_between(times_n, 0, env_n, where=(env_n > np.max(env_n)*0.2), color='black', alpha=0.1)
                
                ax1.set_title(f"My Rhythm (Stress Duration: {dur_l:.1f}%)")
                ax2.set_title(f"Native Rhythm (Stress Duration: {dur_n:.1f}%)")
                plt.tight_layout(); st.pyplot(fig)

            with tab2:
                st.info(f"💡 **분석 결과 해석:**\n\n1. **길이 비중(Duration Ratio):** 원어민({dur_n:.1f}%)보다 수치가 낮다면, 강세 모음을 충분히 길게 끌어주지 못한 것입니다.\n2. **에너지 패턴:** 파란색 실선(Envelope)의 면적이 넓고 높을수록 원어민다운 리듬감이 생깁니다.")

        except Exception as e:
            st.error(f"오류 발생: {e}")
        finally:
            for f in ["temp_raw.wav", "temp_trimmed.wav", "native_temp.mp3", "native_trimmed.wav", "native_voice.mp3"]:
                if os.path.exists(f): os.remove(f)
