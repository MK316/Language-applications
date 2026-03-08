import streamlit as st
from streamlit_mic_recorder import mic_recorder
import io, os, librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from scipy.interpolate import interp1d

# --- [1] 레이아웃 및 CSS ---
st.set_page_config(page_title="Word Stress Master", layout="centered")

st.markdown("""
    <style>
    .stSlider { padding-left: 0px; padding-right: 0px; }
    .main .block-container { padding-top: 1rem; }
    div.stButton > button { height: 3.5em; font-weight: bold; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- [2] 세션 상태 관리 ---
if 'last_audio_id' not in st.session_state: st.session_state.last_audio_id = None
if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False

# --- [3] 분석 엔진 함수 (Numpy 버전 대응 수정) ---
def get_speech_bounds(audio_segment, silence_thresh=-45, min_silence_len=50):
    intervals = detect_nonsilent(audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    if not intervals: return 0, len(audio_segment)
    return intervals[0][0], intervals[-1][1]

def get_rms_envelope(y, hop_length=256):
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    return np.convolve(rms, np.ones(5)/5, mode='same')

def calculate_advanced_score(env_n, env_l):
    if len(env_n) < 2 or len(env_l) < 2: return 0, 0, 0
    
    # 1. 모양 점수 (상관계수)
    f_n = interp1d(np.linspace(0, 1, len(env_n)), env_n, fill_value="extrapolate")
    f_l = interp1d(np.linspace(0, 1, len(env_l)), env_l, fill_value="extrapolate")
    new_x = np.linspace(0, 1, 100)
    norm_n, norm_l = f_n(new_x), f_l(new_x)
    shape_corr = np.corrcoef(norm_n, norm_l)[0, 1]
    
    # 2. 길이 점수 (유효 발화 구간 비중 비교)
    thresh_n, thresh_l = np.max(env_n) * 0.2, np.max(env_l) * 0.2
    dur_n = np.sum(env_n > thresh_n) / len(env_n)
    dur_l = np.sum(env_l > thresh_l) / len(env_l)
    duration_score = 1 - abs(dur_n - dur_l)
    
    # 3. 면적 점수 (np.trapz 대신 np.sum으로 범용성 확보)
    area_n = np.sum(env_n) / len(env_n)
    area_l = np.sum(env_l) / len(env_l)
    area_score = 1 - abs(area_n - area_l) / (area_n + 1e-6)

    final_score = (max(0, shape_corr) * 0.5 + max(0, duration_score) * 0.3 + max(0, area_score) * 0.2) * 100
    return int(final_score), dur_l * 100, dur_n * 100

# --- [4] 앱 UI ---
st.title("🎙️ Word Stress Master")

word_db = {
    "Photograph (1음절)": "photograph",
    "Photographer (2음절)": "photographer",
    "Record (Noun-1음절)": "record",
    "Record (Verb-2음절)": "record",
    "Education (3음절)": "education",
}

selected_label = st.selectbox("학습할 단어 선택:", list(word_db.keys()))
target_word = word_db[selected_label]

if st.button("🔊 원어민 표준 발음 듣기"):
    tts = gTTS(text=target_word, lang='en')
    tts.save("native_voice.mp3")
    st.audio("native_voice.mp3")

st.divider()

# --- [5] 녹음 및 정밀 구간 설정 ---
st.subheader(f"🎯 연습: {target_word.upper()}")
audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 녹음 완료", key="word_recorder")

if audio:
    if audio['id'] != st.session_state.last_audio_id:
        st.session_state.last_audio_id = audio['id']
        st.session_state.analysis_done = False

    audio_bytes = audio['bytes']
    with open("temp_raw.wav", "wb") as f: f.write(audio_bytes)
    
    l_seg = AudioSegment.from_file("temp_raw.wav")
    y_full, sr_f = librosa.load("temp_raw.wav", sr=22050)
    full_dur = len(y_full) / sr_f

    st.markdown("#### ✂️ 분석 구간 설정")
    auto_s, auto_e = get_speech_bounds(l_seg)
    trim_range = st.slider("파형을 보고 빨간 선을 조절하세요:", 
                           0.0, float(full_dur), 
                           (float(auto_s/1000), float(auto_e/1000)), 
                           step=0.01, label_visibility="collapsed")

    fig_prev = plt.figure(figsize=(10, 2.5))
    axp = fig_prev.add_axes([0, 0.2, 1, 0.8]) 
    librosa.display.waveshow(y_full, sr=sr_f, ax=axp, color='skyblue', alpha=0.5)
    axp.axvline(x=trim_range[0], color='red', lw=2, ls='--')
    axp.axvline(x=trim_range[1], color='red', lw=2, ls='--')
    axp.set_xlim(0, full_dur); axp.set_yticks([]); axp.tick_params(axis='x', labelsize=9)
    st.pyplot(fig_prev)

    start_ms, end_ms = int(trim_range[0] * 1000), int(trim_range[1] * 1000)
    trimmed_audio = l_seg[start_ms:end_ms]
    
    st.write("🔈 **녹음 확인:**")
    trimmed_buf = io.BytesIO()
    trimmed_audio.export(trimmed_buf, format="wav")
    st.audio(trimmed_buf.getvalue())
    
    if st.button("📊 면적 및 길이 기반 AI 분석 수행", type="primary"):
        st.session_state.analysis_done = True
        st.session_state.final_trimmed_wav = trimmed_buf.getvalue()

# --- [6] 최종 분석 결과 ---
if st.session_state.get('analysis_done'):
    with st.spinner("리듬 및 에너지 분석 중..."):
        try:
            y_l, sr = librosa.load(io.BytesIO(st.session_state.final_trimmed_wav), sr=22050)
            
            tts = gTTS(text=target_word, lang='en')
            n_fp = io.BytesIO(); tts.write_to_fp(n_fp); n_fp.seek(0)
            n_seg = AudioSegment.from_file(n_fp)
            ns, ne = get_speech_bounds(n_seg)
            n_buf = io.BytesIO(); n_seg[ns:ne].export(n_buf, format="wav"); n_buf.seek(0)
            y_n, _ = librosa.load(n_buf, sr=sr)
            
            y_l, y_n = librosa.util.normalize(y_l), librosa.util.normalize(y_n)
            env_l, env_n = get_rms_envelope(y_l), get_rms_envelope(y_n)
            
            final_score, dur_l, dur_n = calculate_advanced_score(env_n, env_l)
            
            st.divider()
            col1, col2 = st.columns(2)
            col1.metric("종합 점수", f"{final_score}점")
            col2.metric("강세 길이 비중", f"{dur_l:.1f}%", f"{dur_l - dur_n:+.1f}% vs 원어민")

            tab1, tab2 = st.tabs(["📊 에너지 대조", "✍️ 성찰 노트"])
            with tab1:
                fig_res, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
                t_l = np.linspace(0, len(y_l)/sr, len(env_l))
                ax1.plot(t_l, env_l, color='#1f77b4', lw=2)
                ax1.fill_between(t_l, 0, env_l, where=(env_l > np.max(env_l)*0.2), color='blue', alpha=0.15)
                ax1.axvline(x=t_l[np.argmax(env_l)], color='red', lw=2)
                
                t_n = np.linspace(0, len(y_n)/sr, len(env_n))
                ax2.plot(t_n, env_n, color='gray', lw=2)
                ax2.fill_between(t_n, 0, env_n, where=(env_n > np.max(env_n)*0.2), color='black', alpha=0.15)
                ax2.axvline(x=t_n[np.argmax(env_n)], color='red', lw=2)
                
                ax1.set_title(f"My Energy Profile"); ax2.set_title("Native Standard Profile")
                plt.tight_layout(); st.pyplot(fig_res)
                st.info("💡 **Tip:** 색칠된 '에너지 면적'의 부피가 원어민과 비슷해야 자연스럽습니다.")
                
        except Exception as e:
            st.error(f"분석 오류: {e}")

if os.path.exists("temp_raw.wav"): os.remove("temp_raw.wav")
