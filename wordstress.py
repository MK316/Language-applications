import streamlit as st
import streamlit.components.v1 as components
from streamlit_mic_recorder import mic_recorder
import io, os, librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from scipy.interpolate import interp1d

# --- [1] 기본 설정 및 CSS ---
st.set_page_config(page_title="Word Stress Master", layout="centered")
st.markdown("""
    <style>
    .stSlider { padding-left: 0px; padding-right: 0px; }
    .main .block-container { padding-top: 1rem; }
    /* 기존 버튼 스타일 제거 및 통일 */
    .stButton button { width: 100%; border-radius: 10px; height: 3.5em; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# 세션 초기화
for key in ['reset_key', 'last_audio_id', 'analysis_done', 'final_y_l']:
    if key not in st.session_state: st.session_state[key] = 0 if key == 'reset_key' else (None if key != 'analysis_done' else False)

# --- [2] 분석 유틸리티 ---
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
    weighted_env = (env / (np.max(env) + 1e-6)) * 0.8 + (env / (np.sum(env) + 1e-6)) * 0.2
    return np.argmax(weighted_env), env

# --- [3] 앱 인터페이스 ---
st.title("🎙️ Word Stress Master")
word_db = {"Photograph (1음절 강세)": "photograph", "Photographer (2음절 강세)": "photographer", "Education (3음절 강세)": "education"}
selected_label = st.selectbox("학습할 단어 선택:", list(word_db.keys()))
target_word = word_db[selected_label]

if st.button("🔊 원어민 발음 듣기"):
    tts = gTTS(text=target_word, lang='en'); b = io.BytesIO(); tts.write_to_fp(b); st.audio(b.getvalue())

st.divider()
st.subheader(f"🎯 연습: {target_word.upper()}")

# --- [4] 핵심 수정: 정밀 세그먼트 버튼 (HTML/CSS) ---
# 녹음 위젯을 왼쪽에, 리셋 버튼 역할을 할 HTML을 오른쪽에 배치하는 대신
# 아예 커스텀 레이아웃을 구성합니다.

# 리셋 기능을 위한 투명 버튼 활용 로직
col_btn_l, col_btn_r = st.columns(2)

with col_btn_l:
    # 녹음 컴포넌트의 스타일을 강제로 커스터마이징
    audio = mic_recorder(
        start_prompt="🎤 녹음 시작",
        stop_prompt="🛑 완료",
        key=f"recorder_{st.session_state.reset_key}"
    )

with col_btn_r:
    # 리셋 버튼의 높이와 폰트를 녹음 버튼과 강제로 맞춤
    if st.button("🔄 리셋"):
        st.session_state.reset_key += 1
        st.session_state.last_audio_id, st.session_state.analysis_done, st.session_state.final_y_l = None, False, None
        st.rerun()

# 위 방식이 여전히 어긋난다면, 아래의 정밀 구간 설정 로직으로 이어집니다.
if audio:
    if audio['id'] != st.session_state.last_audio_id:
        st.session_state.last_audio_id, st.session_state.analysis_done = audio['id'], False
    
    l_raw = AudioSegment.from_file(io.BytesIO(audio['bytes']))
    y_full = np.array(l_raw.get_array_of_samples(), dtype=np.float32) / (2**15)
    if l_raw.channels > 1: y_full = y_full.reshape((-1, l_raw.channels)).mean(axis=1)
    sr_f = l_raw.frame_rate
    
    st.markdown("#### ✂️ 분석 구간 설정")
    b = detect_nonsilent(l_raw, min_silence_len=100, silence_thresh=-45)
    as_ms, ae_ms = b[0] if b else (0, len(l_raw))
    trim_range = st.slider("구간 선택:", 0.0, float(len(y_full)/sr_f), (float(as_ms/1000), float(ae_ms/1000)), step=0.01)

    # 파형 프리뷰 (여백 제거)
    fig_p = plt.figure(figsize=(10, 2.2)); axp = fig_p.add_axes([0, 0.2, 1, 0.8])
    librosa.display.waveshow(y_full, sr=sr_f, ax=axp, color='skyblue', alpha=0.5)
    axp.axvline(x=trim_range[0], color='red', lw=2, ls='--'); axp.axvline(x=trim_range[1], color='red', lw=2, ls='--')
    axp.set_xlim(0, len(y_full)/sr_f); axp.set_yticks([]); st.pyplot(fig_p)

    st.audio(l_raw[int(trim_range[0]*1000):int(trim_range[1]*1000)].export(io.BytesIO(), format="wav").getvalue())
    
    if st.button("📊 정밀 분석 실행", type="primary"):
        st.session_state.analysis_done = True
        st.session_state.final_y_l = y_full[int(trim_range[0]*sr_f):int(trim_range[1]*sr_f)]
        st.session_state.current_sr = sr_f

# --- [5] 결과 분석 출력 ---
if st.session_state.get('analysis_done') and st.session_state.final_y_l is not None:
    y_l, sr = st.session_state.final_y_l, st.session_state.current_sr
    try:
        tts = gTTS(text=target_word, lang='en'); n_mp3 = io.BytesIO(); tts.write_to_fp(n_mp3); n_mp3.seek(0)
        n_seg = safe_trim(AudioSegment.from_file(n_mp3))
        y_n = np.array(n_seg.get_array_of_samples(), dtype=np.float32) / (2**15)
        if n_seg.channels > 1: y_n = y_n.reshape((-1, n_seg.channels)).mean(axis=1)
        
        y_l, y_n = librosa.util.normalize(y_l), librosa.util.normalize(y_n)
        p_idx_l, env_l = detect_syllable_stress(y_l, sr)
        p_idx_n, env_n = detect_syllable_stress(y_n, sr)
        
        # 일치도 계산
        f_n = interp1d(np.linspace(0, 1, len(env_n)), env_n, fill_value="extrapolate")
        f_l = interp1d(np.linspace(0, 1, len(env_l)), env_l, fill_value="extrapolate")
        score = int(max(0, np.corrcoef(f_n(np.linspace(0, 1, 100)), f_l(np.linspace(0, 1, 100)))[0, 1]) * 100)

        st.divider(); c1, c2 = st.columns(2)
        c1.metric("종합 점수", f"{score}점"); c2.metric("강세 구간 비중", f"{(np.sum(env_l > np.max(env_l)*0.2)/len(env_l))*100:.1f}%")

        fig_res, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6)); mt = max(len(y_l), len(y_n)) / sr
        for ax, y, env, p, t, col in [(ax1, y_l, env_l, p_idx_l, "My Rhythm", "skyblue"), (ax2, y_n, env_n, p_idx_n, "Native Standard", "lightgray")]:
            times = np.linspace(0, len(y)/sr, len(env))
            librosa.display.waveshow(y, sr=sr, ax=ax, color=col, alpha=0.3)
            ax.plot(times, env, color='#1f77b4' if col=="skyblue" else "gray", lw=2.5)
            ax.fill_between(times, 0, env, where=(env > np.max(env)*0.2), color='#1f77b4' if col=="skyblue" else "gray", alpha=0.3)
            ax.axvline(x=times[p], color='red', lw=3); ax.set_title(t); ax.set_xlim(0, mt)
        plt.tight_layout(); st.pyplot(fig_res)
    except Exception as e: st.error(f"오류: {e}")
