import streamlit as st
from streamlit_mic_recorder import mic_recorder
import io, os, librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from scipy.interpolate import interp1d

# --- [1] 레이아웃 및 디자인 설정 ---
st.set_page_config(page_title="Word Stress Master", layout="centered")

st.markdown("""
    <style>
    .stSlider { padding-left: 0px; padding-right: 0px; }
    .main .block-container { padding-top: 1rem; }
    
    /* 모든 버튼의 높이와 폰트를 통일 */
    button {
        height: 3.5em !important;
        font-weight: bold !important;
        border-radius: 10px !important;
    }
    
    /* 분석 실행 버튼 강조 */
    .stButton > button[kind="primary"] {
        background-color: #ff4b4b !important;
        color: white !important;
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 세션 상태 초기화
if 'reset_key' not in st.session_state: st.session_state.reset_key = 0
if 'last_audio_id' not in st.session_state: st.session_state.last_audio_id = None
if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
if 'final_y_l' not in st.session_state: st.session_state.final_y_l = None
if 'current_sr' not in st.session_state: st.session_state.current_sr = None

# --- [2] 정밀 분석 엔진 ---
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
    # 진폭 80% + 면적 20% 가중치로 실제 강세 피크 탐지
    weighted_env = (env / (np.max(env) + 1e-6)) * 0.8 + (env / (np.sum(env) + 1e-6)) * 0.2
    return np.argmax(weighted_env), env

def calculate_score(env_n, env_l):
    if len(env_n) < 2 or len(env_l) < 2: return 0, 0, 0
    # 타임 워핑 없이 1:1 리듬 일치도 계산 (상관계수)
    f_n = interp1d(np.linspace(0, 1, len(env_n)), env_n, fill_value="extrapolate")
    f_l = interp1d(np.linspace(0, 1, len(env_l)), env_l, fill_value="extrapolate")
    new_x = np.linspace(0, 1, 100)
    shape_corr = np.corrcoef(f_n(new_x), f_l(new_x))[0, 1]
    
    dur_n = np.sum(env_n > np.max(env_n)*0.2) / len(env_n)
    dur_l = np.sum(env_l > np.max(env_l)*0.2) / len(env_l)
    return int(max(0, shape_corr) * 100), dur_l * 100, dur_n * 100

# --- [3] 메인 UI: 입력 섹션 ---
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

# --- [4] 녹음 및 구간 설정 ---
st.subheader(f"🎯 연습: {target_word.upper()}")

# 녹음 버튼 (상단 유지)
audio = mic_recorder(
    start_prompt="🎤 녹음 시작",
    stop_prompt="🛑 완료",
    key=f"rec_{st.session_state.reset_key}"
)

if audio:
    if audio['id'] != st.session_state.last_audio_id:
        st.session_state.last_audio_id = audio['id']
        st.session_state.analysis_done = False
        st.session_state.final_y_l = None

    l_raw = AudioSegment.from_file(io.BytesIO(audio['bytes']))
    y_full = np.array(l_raw.get_array_of_samples(), dtype=np.float32) / (2**15)
    if l_raw.channels > 1: y_full = y_full.reshape((-1, l_raw.channels)).mean(axis=1)
    sr_f = l_raw.frame_rate
    
    st.markdown("#### ✂️ 분석 구간 설정")
    auto_bounds = detect_nonsilent(l_raw, min_silence_len=100, silence_thresh=-45)
    as_ms, ae_ms = auto_bounds[0] if auto_bounds else (0, len(l_raw))
    
    trim_range = st.slider("단어의 시작과 끝을 조절하세요:", 
                           0.0, float(len(y_full)/sr_f), 
                           (float(as_ms/1000), float(ae_ms/1000)), step=0.01)

    # 구간 설정용 파형 프리뷰
    fig_p = plt.figure(figsize=(10, 2.2))
    axp = fig_p.add_axes([0, 0.2, 1, 0.8])
    librosa.display.waveshow(y_full, sr=sr_f, ax=axp, color='skyblue', alpha=0.5)
    axp.axvline(x=trim_range[0], color='red', lw=2, ls='--')
    axp.axvline(x=trim_range[1], color='red', lw=2, ls='--')
    axp.set_xlim(0, len(y_full)/sr_f); axp.set_yticks([]); st.pyplot(fig_p)

    # 미리보기 재생
    trimmed_audio = l_raw[int(trim_range[0]*1000):int(trim_range[1]*1000)]
    st.audio(trimmed_audio.export(io.BytesIO(), format="wav").getvalue())
    
    # [핵심] 분석 및 리셋 버튼을 구간 설정 바로 아래에 배치
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📊 정밀 분석 실행", type="primary"):
            st.session_state.analysis_done = True
            st.session_state.final_y_l = y_full[int(trim_range[0]*sr_f):int(trim_range[1]*sr_f)]
            st.session_state.current_sr = sr_f
    with c2:
        if st.button("🔄 연습 리셋"):
            st.session_state.reset_key += 1
            st.session_state.last_audio_id, st.session_state.analysis_done, st.session_state.final_y_l = None, False, None
            st.rerun()

# --- [5] 결과 분석 및 시각화 (복구된 코드) ---
if st.session_state.analysis_done and st.session_state.final_y_l is not None:
    y_l, sr = st.session_state.final_y_l, st.session_state.current_sr
    
    with st.spinner("원어민 데이터와 대조 중..."):
        try:
            # 원어민 데이터 생성 (안전 트리밍 적용)
            tts = gTTS(text=target_word, lang='en')
            n_mp3 = io.BytesIO(); tts.write_to_fp(n_mp3); n_mp3.seek(0)
            n_seg = safe_trim(AudioSegment.from_file(n_mp3))
            y_n = np.array(n_seg.get_array_of_samples(), dtype=np.float32) / (2**15)
            if n_seg.channels > 1: y_n = y_n.reshape((-1, n_seg.channels)).mean(axis=1)
            
            # 정규화 및 강세 분석
            y_l, y_n = librosa.util.normalize(y_l), librosa.util.normalize(y_n)
            p_idx_l, env_l = detect_syllable_stress(y_l, sr)
            p_idx_n, env_n = detect_syllable_stress(y_n, sr)
            score, dur_l, dur_n = calculate_score(env_n, env_l)

            st.divider()
            col1, col2 = st.columns(2)
            col1.metric("종합 리듬 점수", f"{score}점")
            col2.metric("강세 구간 비중", f"{dur_l:.1f}%", f"{dur_l-dur_n:+.1f}% vs 원어민")

            # 결과 그래프 시각화
            [Image of audio waveform amplitude envelope and stress peaks]
            fig_res, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
            max_t = max(len(y_l), len(y_n)) / sr
            
            # 학습자 프로필
            t_l = np.linspace(0, len(y_l)/sr, len(env_l))
            librosa.display.waveshow(y_l, sr=sr, ax=ax1, color='skyblue', alpha=0.3)
            ax1.plot(t_l, env_l, color='#1f77b4', lw=2.5)
            ax1.fill_between(t_l, 0, env_l, where=(env_l > np.max(env_l)*0.2), color='#1f77b4', alpha=0.3)
            ax1.axvline(x=t_l[p_idx_l], color='red', lw=3)
            ax1.set_title("My Rhythm Profile"); ax1.set_xlim(0, max_t)
            
            # 원어민 프로필
            t_n = np.linspace(0, len(y_n)/sr, len(env_n))
            librosa.display.waveshow(y_n, sr=sr, ax=ax2, color='lightgray', alpha=0.3)
            ax2.plot(t_n, env_n, color='gray', lw=2.5)
            ax2.fill_between(t_n, 0, env_n, where=(env_n > np.max(env_n)*0.2), color='gray', alpha=0.3)
            ax2.axvline(x=t_n[p_idx_n], color='red', lw=3)
            ax2.set_title("Native Standard Profile"); ax2.set_xlim(0, max_t)
            
            plt.tight_layout(); st.pyplot(fig_res)
            st.info("💡 빨간 선은 소리의 에너지가 가장 날카롭게 집중된 '진짜 강세' 지점입니다.")

        except Exception as e:
            st.error(f"분석 중 오류가 발생했습니다: {e}")
