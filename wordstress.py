import streamlit as st
from streamlit_mic_recorder import mic_recorder
import io, os, librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from scipy.interpolate import interp1d

# --- [1] 레이아웃 및 CSS (모바일 최적화) ---
st.set_page_config(page_title="Word Stress Master", layout="centered")

st.markdown("""
    <style>
    /* 슬라이더와 그래프 너비 동기화 */
    .stSlider { padding-left: 0px; padding-right: 0px; }
    .main .block-container { padding-top: 1rem; }
    /* 녹음 버튼 강조 */
    div.stButton > button { height: 3.5em; font-weight: bold; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# 세션 상태 초기화 (데이터 휘발 방지)
if 'last_audio_id' not in st.session_state: st.session_state.last_audio_id = None
if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
if 'final_y_l' not in st.session_state: st.session_state.final_y_l = None

# --- [2] 정밀 분석 엔진 함수 ---
def get_rms_envelope(y, hop_length=256):
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    return np.convolve(rms, np.ones(5)/5, mode='same')

def safe_trim(audio_segment):
    """원어민 음성이 너무 짧게 잘리는 것 방지"""
    bounds = detect_nonsilent(audio_segment, min_silence_len=100, silence_thresh=-50)
    if not bounds: return audio_segment
    start, end = bounds[0]
    return audio_segment[start:end] if (end - start) > 200 else audio_segment

def detect_syllable_stress(y, sr):
    """진폭 80% + 면적 20% 가중치로 실제 피크 탐지"""
    env = get_rms_envelope(y)
    weighted_env = (env / (np.max(env) + 1e-6)) * 0.8 + (env / (np.sum(env) + 1e-6)) * 0.2
    return np.argmax(weighted_env), env

def calculate_score(env_n, env_l):
    """모양 일치도 및 길이 비중 계산"""
    if len(env_n) < 2 or len(env_l) < 2: return 0, 0, 0
    f_n = interp1d(np.linspace(0, 1, len(env_n)), env_n, fill_value="extrapolate")
    f_l = interp1d(np.linspace(0, 1, len(env_l)), env_l, fill_value="extrapolate")
    new_x = np.linspace(0, 1, 100)
    shape_corr = np.corrcoef(f_n(new_x), f_l(new_x))[0, 1]
    dur_n = np.sum(env_n > np.max(env_n)*0.2) / len(env_n)
    dur_l = np.sum(env_l > np.max(env_l)*0.2) / len(env_l)
    return int(max(0, shape_corr) * 100), dur_l * 100, dur_n * 100

# --- [3] 앱 UI 시작 ---
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

# --- [4] 녹음 단계 (버튼 노출 보장) ---
st.subheader(f"🎯 연습: {target_word.upper()}")
audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 녹음 완료", key="word_recorder")

if audio:
    # 새로운 녹음 시 상태 초기화
    if audio['id'] != st.session_state.last_audio_id:
        st.session_state.last_audio_id = audio['id']
        st.session_state.analysis_done = False
        st.session_state.final_y_l = None

    # 오디오 로드 (pydub 사용으로 LibsndfileError 방지)
    l_raw = AudioSegment.from_file(io.BytesIO(audio['bytes']))
    y_full = np.array(l_raw.get_array_of_samples(), dtype=np.float32) / (2**15)
    if l_raw.channels > 1: y_full = y_full.reshape((-1, l_raw.channels)).mean(axis=1)
    sr_f = l_raw.frame_rate
    
    st.markdown("#### ✂️ 분석 구간 설정")
    # 구간 자동 추천
    auto_bounds = detect_nonsilent(l_raw, min_silence_len=100, silence_thresh=-45)
    as_ms, ae_ms = auto_bounds[0] if auto_bounds else (0, len(l_raw))
    
    # [핵심] 슬라이더와 파형의 시각적 일치
    trim_range = st.slider("파형을 보고 단어 구간을 조절하세요:", 
                           0.0, float(len(y_full)/sr_f), 
                           (float(as_ms/1000), float(ae_ms/1000)), step=0.01)

    fig_p = plt.figure(figsize=(10, 2.2))
    axp = fig_p.add_axes([0, 0.2, 1, 0.8]) # 여백 제거
    librosa.display.waveshow(y_full, sr=sr_f, ax=axp, color='skyblue', alpha=0.5)
    axp.axvline(x=trim_range[0], color='red', lw=2, ls='--')
    axp.axvline(x=trim_range[1], color='red', lw=2, ls='--')
    axp.set_xlim(0, len(y_full)/sr_f); axp.set_yticks([]); st.pyplot(fig_p)

    # 구간 확인 미리듣기
    trimmed_audio_segment = l_raw[int(trim_range[0]*1000):int(trim_range[1]*1000)]
    st.audio(trimmed_audio_segment.export(io.BytesIO(), format="wav").getvalue())
    
    if st.button("📊 정밀 리듬 분석 시작", type="primary"):
        st.session_state.analysis_done = True
        st.session_state.final_y_l = y_full[int(trim_range[0]*sr_f):int(trim_range[1]*sr_f)]
        st.session_state.current_sr = sr_f

# --- [5] 결과 분석 (웨이브폼 + 에너지 면적) ---
if st.session_state.analysis_done and st.session_state.final_y_l is not None:
    y_l, sr = st.session_state.final_y_l, st.session_state.current_sr
    
    try:
        # 원어민 데이터 생성 및 정밀 트리밍
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
        c1.metric("종합 점수", f"{score}점")
        c2.metric("강세 구간 비중", f"{dur_l:.1f}%", f"{dur_l-dur_n:+.1f}% vs 원어민")

        fig_res, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        max_t = max(len(y_l), len(y_n)) / sr
        
        # 학습자 시각화
        t_l = np.linspace(0, len(y_l)/sr, len(env_l))
        librosa.display.waveshow(y_l, sr=sr, ax=ax1, color='skyblue', alpha=0.3)
        ax1.plot(t_l, env_l, color='#1f77b4', lw=2.5)
        ax1.fill_between(t_l, 0, env_l, where=(env_l > np.max(env_l)*0.2), color='#1f77b4', alpha=0.3)
        ax1.axvline(x=t_l[p_idx_l], color='red', lw=3)
        ax1.set_title("My Rhythm Profile"); ax1.set_xlim(0, max_t)
        
        # 원어민 시각화
        t_n = np.linspace(0, len(y_n)/sr, len(env_n))
        librosa.display.waveshow(y_n, sr=sr, ax=ax2, color='lightgray', alpha=0.3)
        ax2.plot(t_n, env_n, color='gray', lw=2.5)
        ax2.fill_between(t_n, 0, env_n, where=(env_n > np.max(env_n)*0.2), color='gray', alpha=0.3)
        ax2.axvline(x=t_n[p_idx_n], color='red', lw=3)
        ax2.set_title("Native Standard Profile"); ax2.set_xlim(0, max_t)
        
        plt.tight_layout(); st.pyplot(fig_res)

    except Exception as e:
        st.error(f"분석 중 오류: {e}")
