import streamlit as st
from streamlit_mic_recorder import mic_recorder
import io, os, librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from scipy.interpolate import interp1d

# --- [1] 모바일 레이아웃 및 CSS 설정 ---
st.set_page_config(page_title="Word Stress Master", layout="centered")

st.markdown("""
    <style>
    /* 슬라이더와 그래프 너비 동기화 */
    .stSlider { padding-left: 0px; padding-right: 0px; }
    .main .block-container { padding-top: 1rem; }
    
    /* 리셋 및 녹음 버튼 스타일 통일 */
    div.stButton > button, div[data-testid="stVerticalBlock"] > div button {
        width: 100% !important;
        height: 3.5em !important;
        font-weight: bold !important;
        font-size: 16px !important;
        border-radius: 12px !important;
        margin-bottom: 5px !important;
    }
    
    /* 리셋 버튼 색상 (회색) */
    div.stButton > button:contains("리셋") {
        background-color: #f0f2f6 !important;
        color: #31333F !important;
        border: 1px solid #dcdcdc !important;
    }
    
    /* 녹음 버튼 색상 (빨간색 강조) */
    div[data-testid="stVerticalBlock"] > div button:contains("녹음") {
        background-color: #ff4b4b !important;
        color: white !important;
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 세션 상태 초기화
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

def detect_syllable_stress(y, sr):
    env = get_rms_envelope(y)
    # 진폭 가중치를 높여 시각적 피크 일치
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

selected_label = st.selectbox("학습할 단어 선택:", list(word_db.keys()))
target_word = word_db[selected_label]

if st.button("🔊 원어민 표준 발음 듣기"):
    tts = gTTS(text=target_word, lang='en')
    mp3_buf = io.BytesIO()
    tts.write_to_fp(mp3_buf)
    st.audio(mp3_buf.getvalue())

st.divider()

# --- [4] 리셋 및 녹음 컨트롤 (세로 배치) ---
st.subheader(f"🎯 연습: {target_word.upper()}")

if st.button("🔄 리셋 (다시 하기)"):
    st.session_state.last_audio_id = None
    st.session_state.analysis_done = False
    st.session_state.final_y_l = None
    st.rerun()

audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 녹음 완료", key="word_recorder")

# --- [5] 구간 설정 및 분석 로직 ---
if audio:
    if audio['id'] != st.session_state.last_audio_id:
        st.session_state.last_audio_id = audio['id']
        st.session_state.analysis_done = False
        st.session_state.final_y_l = None

    # 오디오 데이터 처리
    l_raw = AudioSegment.from_file(io.BytesIO(audio['bytes']))
    y_full = np.array(l_raw.get_array_of_samples(), dtype=np.float32) / (2**15)
    if l_raw.channels > 1: y_full = y_full.reshape((-1, l_raw.channels)).mean(axis=1)
    sr_f = l_raw.frame_rate
    
    st.markdown("#### ✂️ 분석 구간 설정")
    auto_bounds = detect_nonsilent(l_raw, min_silence_len=100, silence_thresh=-45)
    as_ms, ae_ms = auto_bounds[0] if auto_bounds else (0, len(l_raw))
    
    trim_range = st.slider("단어의 시작과 끝을 맞추세요:", 
                           0.0, float(len(y_full)/sr_f), 
                           (float(as_ms/1000), float(ae_ms/1000)), step=0.01)

    # 파형 프리뷰 (슬라이더와 수직 동기화)
    fig_p = plt.figure(figsize=(10, 2.2))
    axp = fig_p.add_axes([0, 0.2, 1, 0.8])
    librosa.display.waveshow(y_full, sr=sr_f, ax=axp, color='skyblue', alpha=0.5)
    axp.axvline(x=trim_range[0], color='red', lw=2, ls='--')
    axp.axvline(x=trim_range[1], color='red', lw=2, ls='--')
    axp.set_xlim(0, len(y_full)/sr_f); axp.set_yticks([]); st.pyplot(fig_p)

    # 구간 확인 미리듣기
    trimmed_audio = l_raw[int(trim_range[0]*1000):int(trim_range[1]*1000)]
    st.audio(trimmed_audio.export(io.BytesIO(), format="wav").getvalue())
    
    if st.button("📊 정밀 에너지 분석 시작", type="primary"):
        st.session_state.analysis_done = True
        st.session_state.final_y_l = y_full[int(trim_range[0]*sr_f):int(trim_range[1]*sr_f)]
        st.session_state.current_sr = sr_f

# --- [6] 결과 분석 출력 ---
if st.session_state.get('analysis_done') and st.session_state.final_y_l is not None:
    y_l, sr = st.session_state.final_y_l, st.session_state.current_sr
    try:
        with st.spinner("AI 분석 중..."):
            tts = gTTS(text=target_word, lang='en')
            n_mp3 = io.BytesIO(); tts.write_to_fp(n_mp3); n_mp3.seek(0)
            n_seg = safe_trim(AudioSegment.from_file(n_mp3))
            y_n = np.array(n_seg.get_array_of_samples(), dtype=np.float32) / (2**15)
            if n_seg.channels > 1: y_n = y_n.reshape((-1, n_seg.channels)).mean(axis=1)
            
            y_l, y_n = librosa.util.normalize(y_l), librosa.util.normalize(y_n)
            p_idx_l, env_l = detect_syllable_stress(y_l, sr)
            p_idx_n, env_n = detect_syllable_stress(y_n, sr)
            score, dur_l, dur_n = calculate_score(env_n, env_l)

            st.divider()
            c1, c2 = st.columns(2)
            c1.metric("종합 발음 점수", f"{score}점")
            c2.metric("강세 구간 비중", f"{dur_l:.1f}%", f"{dur_l-dur_n:+.1f}% vs 원어민")

            fig_res, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
            max_t = max(len(y_l), len(y_n)) / sr
            for ax, y, env, p_idx, title, color in [(ax1, y_l, env_l, p_idx_l, "My Rhythm", "skyblue"), 
                                                    (ax2, y_n, env_n, p_idx_n, "Native Standard", "lightgray")]:
                t = np.linspace(0, len(y)/sr, len(env))
                librosa.display.waveshow(y, sr=sr, ax=ax, color=color, alpha=0.3)
                ax.plot(t, env, color='#1f77b4' if color=="skyblue" else "gray", lw=2.5)
                ax.fill_between(t, 0, env, where=(env > np.max(env)*0.2), color='#1f77b4' if color=="skyblue" else "gray", alpha=0.3)
                ax.axvline(x=t[p_idx], color='red', lw=3)
                ax.set_title(title); ax.set_xlim(0, max_t)
            plt.tight_layout(); st.pyplot(fig_res)
    except Exception as e:
        st.error(f"분석 중 오류 발생: {e}")
