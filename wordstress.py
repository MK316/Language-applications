import streamlit as st
from streamlit_mic_recorder import mic_recorder
import io, os, librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from scipy.interpolate import interp1d

# --- [1] 디자인 설정 ---
st.set_page_config(page_title="Word Stress Master", layout="centered")
st.markdown("""
    <style>
    .stSlider { padding-left: 0px; padding-right: 0px; }
    button { height: 3.5em !important; font-weight: bold !important; border-radius: 10px !important; }
    .stButton > button[kind="primary"] { background-color: #ff4b4b !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

if 'reset_key' not in st.session_state: st.session_state.reset_key = 0
if 'last_audio_id' not in st.session_state: st.session_state.last_audio_id = None
if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
if 'final_y_l' not in st.session_state: st.session_state.final_y_l = None

# --- [2] 분석 엔진 ---
def get_rms_envelope(y, hop_length=256):
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    return np.convolve(rms, np.ones(5)/5, mode='same')

def detect_syllable_stress(y, sr):
    env = get_rms_envelope(y)
    weighted_env = (env / (np.max(env) + 1e-6)) * 0.8 + (env / (np.sum(env) + 1e-6)) * 0.2
    return np.argmax(weighted_env), env

def calculate_normalized_score(env_n, env_l):
    """시간 정규화로 리듬 패턴만 비교"""
    if len(env_n) < 2 or len(env_l) < 2: return 0, np.zeros(100), np.zeros(100)
    standard_x = np.linspace(0, 1, 100)
    f_n = interp1d(np.linspace(0, 1, len(env_n)), env_n, fill_value="extrapolate")
    f_l = interp1d(np.linspace(0, 1, len(env_l)), env_l, fill_value="extrapolate")
    norm_n, norm_l = f_n(standard_x), f_l(standard_x)
    shape_corr = np.corrcoef(norm_n, norm_l)[0, 1]
    return int(max(0, shape_corr) * 100), norm_n, norm_l

# --- [3] 메인 UI ---
st.title("🎙️ Word Stress Master")
word_db = {"Photograph": "photograph", "Photographer": "photographer", "Education": "education"}
target_word = word_db[st.selectbox("단어 선택:", list(word_db.keys()))]

if st.button("🔊 원어민 표준 발음 듣기"):
    tts = gTTS(text=target_word, lang='en'); b = io.BytesIO(); tts.write_to_fp(b); st.audio(b.getvalue())

st.divider()
audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 완료", key=f"rec_{st.session_state.reset_key}")

if audio:
    if audio['id'] != st.session_state.last_audio_id:
        st.session_state.last_audio_id, st.session_state.analysis_done, st.session_state.final_y_l = audio['id'], False, None

    l_raw = AudioSegment.from_file(io.BytesIO(audio['bytes']))
    y_full = np.array(l_raw.get_array_of_samples(), dtype=np.float32) / (2**15)
    sr_f = l_raw.frame_rate
    
    st.markdown("#### ✂️ 분석 구간 설정")
    auto_b = detect_nonsilent(l_raw, min_silence_len=100, silence_thresh=-45)
    as_ms, ae_ms = auto_b[0] if auto_b else (0, len(l_raw))
    trim_range = st.slider("구간 조절:", 0.0, float(len(y_full)/sr_f), (float(as_ms/1000), float(ae_ms/1000)), step=0.01)

    fig_p, axp = plt.subplots(figsize=(10, 2.2)); plt.subplots_adjust(left=0, right=1)
    librosa.display.waveshow(y_full, sr=sr_f, ax=axp, color='skyblue', alpha=0.5)
    axp.axvline(x=trim_range[0], color='red', lw=2); axp.axvline(x=trim_range[1], color='red', lw=2)
    axp.set_xlim(0, len(y_full)/sr_f); axp.set_yticks([]); st.pyplot(fig_p)

    trimmed_audio = l_raw[int(trim_range[0]*1000):int(trim_range[1]*1000)]
    st.audio(trimmed_audio.export(io.BytesIO(), format="wav").getvalue())
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📊 정밀 분석 실행", type="primary"):
            st.session_state.analysis_done = True
            st.session_state.final_y_l = y_full[int(trim_range[0]*sr_f):int(trim_range[1]*sr_f)]
            st.session_state.current_sr = sr_f
            st.session_state.final_audio_l = trimmed_audio
    with c2:
        if st.button("🔄 연습 리셋"):
            st.session_state.reset_key += 1; st.rerun()

# --- [4] 결과 분석 (트리밍 제거 버전) ---
if st.session_state.analysis_done and st.session_state.final_y_l is not None:
    y_l, sr = st.session_state.final_y_l, st.session_state.current_sr
    try:
        # [핵심] 원어민 음성을 트리밍 없이 원본 그대로 사용 (Praat 일치)
        tts = gTTS(text=target_word, lang='en'); n_mp3 = io.BytesIO(); tts.write_to_fp(n_mp3); n_mp3.seek(0)
        n_seg = AudioSegment.from_file(n_mp3) # safe_trim_v2 제거
        y_n = np.array(n_seg.get_array_of_samples(), dtype=np.float32) / (2**15)
        if n_seg.channels > 1: y_n = y_n.reshape((-1, n_seg.channels)).mean(axis=1)
        
        y_l_norm, y_n_norm = librosa.util.normalize(y_l), librosa.util.normalize(y_n)
        p_idx_l, env_l = detect_syllable_stress(y_l_norm, sr)
        p_idx_n, env_n = detect_syllable_stress(y_n_norm, sr)
        score, norm_n, norm_l = calculate_normalized_score(env_n, env_l)

        st.divider(); st.metric("종합 리듬 점수 (속도 보정)", f"{score}점")
        
        a1, a2 = st.columns(2)
        with a1: st.write("🙋 나의 발음"); st.audio(st.session_state.final_audio_l.export(io.BytesIO(), format="wav").getvalue())
        with a2: st.write("🎙️ 원어민 표준"); st.audio(n_seg.export(io.BytesIO(), format="wav").getvalue())

        # 절대 시간축 비교 (이제 Praat과 같은 0.8초 내외로 보임)
        st.write("### 📏 절대 시간 기반 비교 (Original Length)")
        fig_abs, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
        mt = max(len(y_l), len(y_n)) / sr
        for ax, y, env, p, t, col in [(ax1, y_l_norm, env_l, p_idx_l, "My Rhythm", "skyblue"), (ax2, y_n_norm, env_n, p_idx_n, "Native Standard", "lightgray")]:
            ts = np.linspace(0, len(y)/sr, len(env))
            librosa.display.waveshow(y, sr=sr, ax=ax, color=col, alpha=0.3)
            ax.plot(ts, env, color='#1f77b4' if col=="skyblue" else "gray", lw=2)
            ax.axvline(x=ts[p], color='red', lw=3); ax.set_title(t); ax.set_xlim(0, mt)
        plt.tight_layout(); st.pyplot(fig_abs)

        # 시간 정규화 오버레이 (리듬 패턴 대조)
        st.write("### 🔄 시간 정규화 리듬 패턴 대조 (Pattern Only)")
        
        fig_norm, axn = plt.subplots(figsize=(10, 4)); x_range = np.linspace(0, 100, 100)
        axn.fill_between(x_range, 0, norm_n, color='gray', alpha=0.2, label='Native Guide')
        axn.plot(x_range, norm_l, color='#ff4b4b', lw=3, label='My Rhythm'); axn.legend(); st.pyplot(fig_norm)

    except Exception as e: st.error(f"분석 오류: {e}")
