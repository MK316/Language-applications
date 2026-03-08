import streamlit as st
from streamlit_mic_recorder import mic_recorder
import io, os, librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from scipy.interpolate import interp1d

# --- [1] 모바일 최적화 및 레이아웃 설정 ---
st.set_page_config(page_title="Word Stress Master", layout="centered")

st.markdown("""
    <style>
    /* 슬라이더와 그래프 너비 동기화 */
    .stSlider { padding-left: 0px; padding-right: 0px; }
    .main .block-container { padding-top: 1.5rem; }
    div.stButton > button { width: 100%; height: 3em; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- [2] 분석 유틸리티 함수 ---
def get_speech_bounds(audio_segment, silence_thresh=-45, min_silence_len=50):
    intervals = detect_nonsilent(audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    if not intervals: return 0, len(audio_segment)
    return intervals[0][0], intervals[-1][1]

def normalize_audio(y):
    return librosa.util.normalize(y) if len(y) > 0 else y

def get_rms_envelope(y, hop_length=256):
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    return np.convolve(rms, np.ones(5)/5, mode='same')

def calculate_stress_metrics(y_n, y_l, sr):
    env_n, env_l = get_rms_envelope(y_n), get_rms_envelope(y_l)
    if len(env_n) < 2 or len(env_l) < 2: return 0, 0, 0, 0, env_n, env_l
    
    # 에너지 패턴 일치도
    f_n = interp1d(np.linspace(0, 1, len(env_n)), env_n, fill_value="extrapolate")
    f_l = interp1d(np.linspace(0, 1, len(env_l)), env_l, fill_value="extrapolate")
    new_x = np.linspace(0, 1, 100)
    score = int(max(0, np.corrcoef(f_n(new_x), f_l(new_x))[0, 1]) * 100)
    
    # 길이 비중 및 타이밍
    dur_n = np.sum(env_n > np.max(env_n)*0.2) / len(env_n) * 100
    dur_l = np.sum(env_l > np.max(env_l)*0.2) / len(env_l) * 100
    timing_diff = (np.argmax(env_l)/len(env_l)*100) - (np.argmax(env_n)/len(env_n)*100)
    
    return score, timing_diff, dur_n, dur_l, env_n, env_l

# --- [3] 앱 UI: 단어 선택 ---
st.title("🎙️ Word Stress Master")

word_db = {
    "Photograph (1음절 강세)": "photograph",
    "Photographer (2음절 강세)": "photographer",
    "Record (Noun - 1음절)": "record",
    "Record (Verb - 2음절)": "record",
    "Education (3음절 강세)": "education",
}

selected_label = st.selectbox("학습할 단어를 선택하세요:", list(word_db.keys()))
target_word = word_db[selected_label]

if st.button("🔊 원어민 발음 듣기 (모델)"):
    tts = gTTS(text=target_word, lang='en')
    tts.save("native_voice.mp3")
    st.audio("native_voice.mp3")

st.divider()

# --- [4] Step 2: 녹음 단계 (추가됨) ---
st.subheader(f"🎯 연습 단어: {target_word.upper()}")
audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 녹음 완료", key="word_recorder")

# --- [5] Step 3: 구간 설정 및 파형 동기화 ---
if audio:
    audio_bytes = audio['bytes']
    # librosa 로딩을 위한 임시 저장
    with open("temp_raw.wav", "wb") as f: f.write(audio_bytes)
    
    y_full, sr_f = librosa.load("temp_raw.wav", sr=22050)
    l_seg = AudioSegment.from_file("temp_raw.wav")
    full_dur = len(y_full) / sr_f

    st.markdown("#### ✂️ 분석 구간 설정")
    st.caption("파형의 빨간 선이 단어의 시작과 끝에 오도록 조절하세요.")
    
    auto_s, auto_e = get_speech_bounds(l_seg)
    # 슬라이더 (CSS로 그래프와 너비 동기화)
    trim_range = st.slider("구간 조절 (초):", 
                           0.0, float(full_dur), 
                           (float(auto_s/1000), float(auto_e/1000)), step=0.01, label_visibility="collapsed")

    # 그래프 여백 제거하여 슬라이더와 수직 일치
    fig_prev = plt.figure(figsize=(10, 2.5))
    axp = fig_prev.add_axes([0, 0.2, 1, 0.8]) # [left, bottom, width, height]
    librosa.display.waveshow(y_full, sr=sr_f, ax=axp, color='skyblue', alpha=0.5)
    axp.axvline(x=trim_range[0], color='red', lw=2, ls='--')
    axp.axvline(x=trim_range[1], color='red', lw=2, ls='--')
    axp.set_xlim(0, full_dur)
    axp.set_yticks([]); axp.tick_params(axis='x', labelsize=9)
    st.pyplot(fig_prev)

    # 구간 확인 및 분석 실행
    start_ms, end_ms = int(trim_range[0] * 1000), int(trim_range[1] * 1000)
    trimmed_audio = l_seg[start_ms:end_ms]
    
    st.audio(audio_bytes) # 전체 다시 듣기
    st.info("💡 위 슬라이더로 편집된 구간만 분석에 사용됩니다.")
    
    if st.button("📊 설정된 구간으로 AI 분석 시작", type="primary"):
        with st.spinner("에너지 패턴 분석 중..."):
            try:
                # 분석용 데이터 준비
                buf = io.BytesIO(); trimmed_audio.export(buf, format="wav")
                y_l, sr = librosa.load(io.BytesIO(buf.getvalue()), sr=22050)
                
                tts = gTTS(text=target_word, lang='en')
                n_fp = io.BytesIO(); tts.write_to_fp(n_fp); n_fp.seek(0)
                n_seg = AudioSegment.from_file(n_fp)
                ns, ne = get_speech_bounds(n_seg)
                n_buf = io.BytesIO(); n_seg[ns:ne].export(n_buf, format="wav"); n_buf.seek(0)
                y_n, _ = librosa.load(n_buf, sr=sr)
                
                y_l, y_n = normalize_audio(y_l), normalize_audio(y_n)
                score, t_diff, dur_n, dur_l, env_n, env_l = calculate_stress_metrics(y_n, y_l, sr)

                st.success(f"분석 완료! 일치도: {score}점")
                
                # 결과 표시 (탭 구성)
                tab1, tab2 = st.tabs(["🎯 분석 결과", "✍️ 성찰"])
                with tab1:
                    c1, c2 = st.columns(2)
                    c1.metric("강세 구간 비중", f"{dur_l:.1f}%", f"{dur_l - dur_n:+.1f}%")
                    c2.metric("타이밍 편차", f"{t_diff:+.1f}%")
                    
                    fig_res, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
                    t_l = np.linspace(0, len(y_l)/sr, len(env_l))
                    ax1.plot(t_l, env_l, color='#1f77b4', lw=2)
                    ax1.fill_between(t_l, 0, env_l, where=(env_l > np.max(env_l)*0.2), color='blue', alpha=0.1)
                    ax1.axvline(x=t_l[np.argmax(env_l)], color='red', lw=2)
                    
                    t_n = np.linspace(0, len(y_n)/sr, len(env_n))
                    ax2.plot(t_n, env_n, color='gray', lw=2)
                    ax2.fill_between(t_n, 0, env_n, where=(env_n > np.max(env_n)*0.2), color='black', alpha=0.1)
                    ax2.axvline(x=t_n[np.argmax(env_n)], color='red', lw=2)
                    
                    ax1.set_title("My Energy Envelope"); ax2.set_title("Native Energy Envelope")
                    plt.tight_layout(); st.pyplot(fig_res)

            except Exception as e: st.error(f"오류: {e}")

# 파일 정리
if os.path.exists("temp_raw.wav"): os.remove("temp_raw.wav")
if os.path.exists("native_voice.mp3"): os.remove("native_voice.mp3")
