import streamlit as st
from streamlit_mic_recorder import mic_recorder
import io, os, librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from scipy.interpolate import interp1d

# --- [1] 모바일 및 레이아웃 최적화 ---
st.set_page_config(page_title="Word Stress Master", layout="centered")

st.markdown("""
    <style>
    .stSlider { padding-left: 0px; padding-right: 0px; }
    .main .block-container { padding-top: 1rem; }
    div.stButton > button { height: 3.5em; font-weight: bold; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# 세션 상태 초기화
for key in ['last_audio_id', 'analysis_done', 'final_y_l', 'current_sr']:
    if key not in st.session_state: 
        st.session_state[key] = None if key != 'analysis_done' else False

# --- [2] 정밀 분석 엔진 함수 ---
def get_rms_envelope(y, hop_length=256):
    """소리의 에너지 덩어리(RMS)를 추출하여 매끄럽게 만듭니다."""
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    return np.convolve(rms, np.ones(5)/5, mode='same')

def safe_trim(audio_segment):
    """원어민 음성이 잡음 제거 과정에서 너무 짧게 잘리는 것을 방지 (최소 200ms 보장)"""
    bounds = detect_nonsilent(audio_segment, min_silence_len=100, silence_thresh=-50)
    if not bounds: return audio_segment
    start, end = bounds[0]
    # 실제 발화 구간이 너무 짧으면 원본을 사용하거나 최소 길이를 확보
    return audio_segment[start:end] if (end - start) > 200 else audio_segment

def detect_syllable_stress(y, sr):
    """음절 내 진폭 가중치를 높여 실제 힘이 실린 피크를 탐지합니다."""
    env = get_rms_envelope(y)
    # 진폭(Height)에 80%, 면적(Area)에 20% 가중치를 주어 시각적 피크와 일치시킴
    weighted_env = (env / (np.max(env) + 1e-6)) * 0.8 + (env / (np.sum(env) + 1e-6)) * 0.2
    return np.argmax(weighted_env), env

def calculate_score(env_n, env_l):
    """모양 일치도 및 길이 비중을 계산합니다."""
    if len(env_n) < 2 or len(env_l) < 2: return 0, 0, 0
    f_n = interp1d(np.linspace(0, 1, len(env_n)), env_n, fill_value="extrapolate")
    f_l = interp1d(np.linspace(0, 1, len(env_l)), env_l, fill_value="extrapolate")
    new_x = np.linspace(0, 1, 100)
    shape_corr = np.corrcoef(f_n(new_x), f_l(new_x))[0, 1]
    
    dur_n = np.sum(env_n > np.max(env_n)*0.2) / len(env_n)
    dur_l = np.sum(env_l > np.max(env_l)*0.2) / len(env_l)
    
    return int(max(0, shape_corr) * 100), dur_l * 100, dur_n * 100

# --- [3] 앱 UI: 단어 선택 및 녹음 ---
st.title("🎙️ Word Stress Master")

word_db = {
    "Photograph (1음절 강세)": "photograph",
    "Photographer (2음절 강세)": "photographer",
    "Education (3음절 강세)": "education",
    "Record (Noun - 1음절)": "record",
    "Record (Verb - 2음절)": "record"
}

selected_label = st.selectbox("학습할 단어를 선택하세요:", list(word_db.keys()))
target_word = word_db[selected_label]

if st.button("🔊 원어민 표준 발음 듣기"):
    tts = gTTS(text=target_word, lang='en')
    mp3_buf = io.BytesIO()
    tts.write_to_fp(mp3_buf)
    st.audio(mp3_buf.getvalue(), format="audio/mp3")

st.divider()
st.subheader(f"🎯 연습 단어: {target_word.upper()}")

# 녹음 버튼
audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 녹음 완료", key="word_recorder")

if audio:
    if audio['id'] != st.session_state.last_audio_id:
        st.session_state.last_audio_id, st.session_state.analysis_done = audio['id'], False

    # 오디오 데이터 로드 및 수치화 (pydub 사용으로 LibsndfileError 방지)
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

    # 설정 구간 파형 프리뷰
    fig_p, axp = plt.subplots(figsize=(10, 2.2))
    plt.subplots_adjust(left=0, right=1, bottom=0.2, top=0.8)
    librosa.display.waveshow(y_full, sr=sr_f, ax=axp, color='skyblue', alpha=0.5)
    axp.axvline(x=trim_range[0], color='red', lw=2, ls='--')
    axp.axvline(x=trim_range[1], color='red', lw=2, ls='--')
    axp.set_xlim(0, len(y_full)/sr_f); axp.set_yticks([]); st.pyplot(fig_p)

    if st.button("📊 정밀 에너지 대조 분석 시작", type="primary"):
        st.session_state.analysis_done = True
        st.session_state.final_y_l = y_full[int(trim_range[0]*sr_f):int(trim_range[1]*sr_f)]
        st.session_state.current_sr = sr_f

# --- [4] 결과 분석 및 시각화 ---
if st.session_state.analysis_done and st.session_state.final_y_l is not None:
    y_l, sr = st.session_state.final_y_l, st.session_state.current_sr
    
    with st.spinner("AI 분석 중..."):
        try:
            # 1. 원어민 TTS 생성 및 안전한 트리밍
            tts = gTTS(text=target_word, lang='en')
            n_mp3 = io.BytesIO(); tts.write_to_fp(n_mp3); n_mp3.seek(0)
            n_seg = safe_trim(AudioSegment.from_file(n_mp3))
            y_n = np.array(n_seg.get_array_of_samples(), dtype=np.float32) / (2**15)
            if n_seg.channels > 1: y_n = y_n.reshape((-1, n_seg.channels)).mean(axis=1)
            
            # 2. 정규화 및 강세 피크 탐지
            y_l, y_n = librosa.util.normalize(y_l), librosa.util.normalize(y_n)
            p_idx_l, env_l = detect_syllable_stress(y_l, sr)
            p_idx_n, env_n = detect_syllable_stress(y_n, sr)
            
            # 3. 점수 계산
            score, dur_l, dur_n = calculate_score(env_n, env_l)

            st.divider()
            c1, c2 = st.columns(2)
            c1.metric("종합 발음 점수", f"{score}점")
            c2.metric("강세 음절 길이 비중", f"{dur_l:.1f}%", f"{dur_l-dur_n:+.1f}% vs 원어민")

            # 4. 결과 그래프 (시간축 동기화 오버레이)
            fig_res, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
            max_time = max(len(y_l), len(y_n)) / sr
            
            # 학습자 그래프
            t_l = np.linspace(0, len(y_l)/sr, len(env_l))
            librosa.display.waveshow(y_l, sr=sr, ax=ax1, color='skyblue', alpha=0.3)
            ax1.plot(t_l, env_l, color='#1f77b4', lw=2.5)
            ax1.fill_between(t_l, 0, env_l, where=(env_l > np.max(env_l)*0.2), color='#1f77b4', alpha=0.3)
            ax1.axvline(x=t_l[p_idx_l], color='red', lw=3)
            ax1.set_title("My Energy Profile"); ax1.set_xlim(0, max_time)
            
            # 원어민 그래프
            t_n = np.linspace(0, len(y_n)/sr, len(env_n))
            librosa.display.waveshow(y_n, sr=sr, ax=ax2, color='lightgray', alpha=0.3)
            ax2.plot(t_n, env_n, color='gray', lw=2.5)
            ax2.fill_between(t_n, 0, env_n, where=(env_n > np.max(env_n)*0.2), color='gray', alpha=0.3)
            ax2.axvline(x=t_n[p_idx_n], color='red', lw=3)
            ax2.set_title("Native Standard Profile"); ax2.set_xlim(0, max_time)
            
            plt.tight_layout(); st.pyplot(fig_res)
            st.info(f"💡 **분석 결과:** 빨간 선은 강세가 실린 시점입니다. 파란색 영역의 '폭(길이)'이 원어민과 비슷할수록 자연스럽습니다.")

        except Exception as e:
            st.error(f"분석 오류: {e}")
