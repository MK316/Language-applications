import streamlit as st
from streamlit_mic_recorder import mic_recorder
import io, os, librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from scipy.interpolate import interp1d

# --- [1] 모바일 최적화 레이아웃 및 CSS 설정 ---
st.set_page_config(page_title="Word Stress & Duration Master", layout="centered") # 모바일은 중앙 집중형이 유리

st.markdown("""
    <style>
    /* 슬라이더 좌우 여백을 제거하여 아래 그래프와 위치 동기화 */
    .stSlider { padding-left: 0px; padding-right: 0px; }
    /* 메인 컨테이너 패딩 조절 */
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    /* 버튼 스타일 통일 */
    div.stButton > button { width: 100%; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- [2] 분석 엔진 함수 ---
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
    env_n, env_l = get_rms_envelope(y_n), get_rms_envelope(y_l)
    if len(env_n) < 2 or len(env_l) < 2: return 0, 0, 0, 0, env_n, env_l
    
    f_n = interp1d(np.linspace(0, 1, len(env_n)), env_n, fill_value="extrapolate")
    f_l = interp1d(np.linspace(0, 1, len(env_l)), env_l, fill_value="extrapolate")
    new_x = np.linspace(0, 1, 100)
    score = int(max(0, np.corrcoef(f_n(new_x), f_l(new_x))[0, 1]) * 100)
    
    dur_n = np.sum(env_n > np.max(env_n)*0.2) / len(env_n) * 100
    dur_l = np.sum(env_l > np.max(env_l)*0.2) / len(env_l) * 100
    timing_diff = (np.argmax(env_l)/len(env_l)*100) - (np.argmax(env_n)/len(env_n)*100)
    
    return score, timing_diff, dur_n, dur_l, env_n, env_l

# --- [3] 앱 UI 시작 ---
st.title("🎙️ Word Stress & Duration")

if 'analysis_ready' not in st.session_state: st.session_state.analysis_ready = False
if 'prev_audio_id' not in st.session_state: st.session_state.prev_audio_id = None

# [변경] 사이드바 대신 메인 상단에 배치 (모바일 접근성 향상)
word_db = {
    "Photograph (1음절 강세)": "photograph",
    "Photographer (2음절 강세)": "photographer",
    "Record (Noun - 1음절)": "record",
    "Record (Verb - 2음절)": "record",
    "Education (3음절 강세)": "education",
}

st.info("📍 **Step 1: 학습할 단어 선택**")
selected_label = st.selectbox("단어를 선택하면 원어민 발음을 들을 수 있습니다.", list(word_db.keys()), label_visibility="collapsed")
target_word = word_db[selected_label]

c_tts1, c_tts2 = st.columns([1, 1])
with c_tts1:
    if st.button("🔊 원어민 발음 생성"):
        tts = gTTS(text=target_word, lang='en')
        tts.save("native_voice.mp3")
        st.audio("native_voice.mp3")

st.divider()

# --- [4] 녹음 및 정밀 구간 설정 ---
st.subheader(f"🎯 도전: **{target_word.upper()}**")
audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 녹음 완료", key="word_recorder")

if audio:
    if audio['id'] != st.session_state.prev_audio_id:
        st.session_state.analysis_ready = False
        st.session_state.prev_audio_id = audio['id']

    audio_bytes = audio['bytes']
    with open("temp_raw.wav", "wb") as f: f.write(audio_bytes)
    
    y_full, sr_f = librosa.load("temp_raw.wav", sr=22050)
    l_seg = AudioSegment.from_file("temp_raw.wav")
    full_dur = len(y_full) / sr_f

    st.markdown("#### ✂️ Step 3: 분석 구간 설정")
    auto_s, auto_e = get_speech_bounds(l_seg)
    
    # [핵심] 슬라이더와 아래 파형의 너비 및 시작점 동기화
    trim_range = st.slider("파형의 빨간 선을 보고 단어 구간을 맞추세요:", 
                           0.0, float(full_dur), 
                           (float(auto_s/1000), float(auto_e/1000)), step=0.01)

    # [핵심] Matplotlib 여백(Margin)을 완전히 제거하여 슬라이더와 일대일 대응
    fig_prev = plt.figure(figsize=(10, 2.2))
    # [left, bottom, width, height] -> left를 0으로 설정하여 슬라이더 시작점과 맞춤
    axp = fig_prev.add_axes([0, 0.2, 1, 0.8]) 
    
    librosa.display.waveshow(y_full, sr=sr_f, ax=axp, color='skyblue', alpha=0.5)
    axp.axvline(x=trim_range[0], color='red', lw=2.5, ls='--')
    axp.axvline(x=trim_range[1], color='red', lw=2.5, ls='--')
    
    axp.set_xlim(0, full_dur)
    axp.set_yticks([]) # Y축 숨김
    axp.tick_params(axis='x', labelsize=9, pad=0)
    st.pyplot(fig_prev)

    start_ms, end_ms = int(trim_range[0] * 1000), int(trim_range[1] * 1000)
    trimmed_audio = l_seg[start_ms:end_ms]
    
    st.write("🔈 **구간 확인:**")
    buf = io.BytesIO(); trimmed_audio.export(buf, format="wav")
    st.audio(buf)
    
    if st.button("📊 이 구간으로 상세 분석 수행", type="primary"):
        st.session_state.analysis_ready = True
        st.session_state.trimmed_wav = buf.getvalue()

# --- [5] 상세 분석 결과 ---
if st.session_state.get('analysis_ready'):
    with st.spinner("AI 분석 중..."):
        try:
            y_l, sr = librosa.load(io.BytesIO(st.session_state.trimmed_wav), sr=22050)
            tts = gTTS(text=target_word, lang='en')
            n_fp = io.BytesIO(); tts.write_to_fp(n_fp); n_fp.seek(0)
            n_seg = AudioSegment.from_file(n_fp)
            ns, ne = get_speech_bounds(n_seg)
            n_buf = io.BytesIO(); n_seg[ns:ne].export(n_buf, format="wav"); n_buf.seek(0)
            y_n, _ = librosa.load(n_buf, sr=sr)
            
            y_l, y_n = normalize_audio(y_l), normalize_audio(y_n)
            score, t_diff, dur_n, dur_l, env_n, env_l = calculate_stress_metrics(y_n, y_l, sr)

            st.success("분석 완료! 아래 탭에서 리듬을 대조하세요.")
            
            m1, m2 = st.columns(2)
            m1.metric("에너지 일치도", f"{score}점")
            m2.metric("강세 구간 비중", f"{dur_l:.1f}%", f"{dur_l - dur_n:+.1f}% vs 원어민")

            tab1, tab2 = st.tabs(["📊 리듬 대조", "✍️ 분석 노트"])
            with tab1:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
                t_l = np.linspace(0, len(y_l)/sr, len(env_l))
                ax1.plot(t_l, env_l, color='#1f77b4', lw=2)
                ax1.fill_between(t_l, 0, env_l, where=(env_l > np.max(env_l)*0.2), color='blue', alpha=0.1)
                ax1.axvline(x=t_l[np.argmax(env_l)], color='red', lw=2)
                
                t_n = np.linspace(0, len(y_n)/sr, len(env_n))
                ax2.plot(t_n, env_n, color='gray', lw=2)
                ax2.fill_between(t_n, 0, env_n, where=(env_n > np.max(env_n)*0.2), color='black', alpha=0.1)
                ax2.axvline(x=t_n[np.argmax(env_n)], color='red', lw=2)
                
                ax1.set_title("My Stress Rhythm"); ax2.set_title("Native Rhythm")
                plt.tight_layout(); st.pyplot(fig)

        except Exception as e: st.error(f"오류: {e}")
        finally:
            for f in ["temp_raw.wav", "native_voice.mp3"]:
                if os.path.exists(f): os.remove(f)
