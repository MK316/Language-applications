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
    .stSlider { padding-left: 0px; padding-right: 0px; }
    .main .block-container { padding-top: 1rem; }
    /* 녹음 버튼을 크게 강조 */
    div[data-testid="stVerticalBlock"] > div:nth-child(7) button {
        height: 4em !important;
        background-color: #ff4b4b !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- [2] 유틸리티 함수 ---
def get_speech_bounds(audio_segment, silence_thresh=-45, min_silence_len=50):
    intervals = detect_nonsilent(audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    if not intervals: return 0, len(audio_segment)
    return intervals[0][0], intervals[-1][1]

def get_rms_envelope(y, hop_length=256):
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    return np.convolve(rms, np.ones(5)/5, mode='same')

# --- [3] 앱 인터페이스 ---
st.title("🎙️ Word Stress Master")

word_db = {
    "Photograph (1음절 강세)": "photograph",
    "Photographer (2음절 강세)": "photographer",
    "Record (Noun - 1음절)": "record",
    "Record (Verb - 2음절)": "record",
    "Education (3음절 강세)": "education",
}

# 단어 선택
selected_label = st.selectbox("학습할 단어를 선택하세요:", list(word_db.keys()))
target_word = word_db[selected_label]

# 원어민 소리 듣기
if st.button("🔊 원어민 모델 보이스 듣기"):
    tts = gTTS(text=target_word, lang='en')
    tts.save("native_voice.mp3")
    st.audio("native_voice.mp3")

st.divider()

# --- [4] 핵심: 녹음 단계 (버튼 강제 노출) ---
st.subheader(f"🎯 연습: {target_word.upper()}")
st.info("아래 버튼을 눌러 발음하세요. (마이크 권한 허용 필요)")

# mic_recorder는 조건문 밖에서 항상 호출되어야 버튼이 안정적으로 보입니다.
audio = mic_recorder(
    start_prompt="🎤 녹음 시작 (START)",
    stop_prompt="🛑 녹음 완료 (STOP)",
    key="word_recorder"
)

# --- [5] 녹음 이후 단계: 구간 설정 및 분석 ---
if audio:
    audio_bytes = audio['bytes']
    with open("temp_raw.wav", "wb") as f:
        f.write(audio_bytes)
    
    # 데이터 로드
    l_seg = AudioSegment.from_file("temp_raw.wav")
    y_full, sr_f = librosa.load("temp_raw.wav", sr=22050)
    full_dur = len(y_full) / sr_f

    st.success("✅ 녹음 성공! 이제 아래 파형을 보고 구간을 조절하세요.")
    
    # 구간 설정 슬라이더
    auto_s, auto_e = get_speech_bounds(l_seg)
    trim_range = st.slider("분석 구간 설정 (초):", 
                           0.0, float(full_dur), 
                           (float(auto_s/1000), float(auto_e/1000)), 
                           step=0.01, label_visibility="collapsed")

    # 파형 프리뷰 (슬라이더와 수직 동기화)
    fig_prev = plt.figure(figsize=(10, 2.5))
    axp = fig_prev.add_axes([0, 0.2, 1, 0.8]) # 여백 제거
    librosa.display.waveshow(y_full, sr=sr_f, ax=axp, color='skyblue', alpha=0.6)
    axp.axvline(x=trim_range[0], color='red', lw=2, ls='--')
    axp.axvline(x=trim_range[1], color='red', lw=2, ls='--')
    axp.set_xlim(0, full_dur)
    axp.set_yticks([]); axp.tick_params(axis='x', labelsize=9)
    st.pyplot(fig_prev)

    # 구간 확인 및 분석 실행
    start_ms, end_ms = int(trim_range[0] * 1000), int(trim_range[1] * 1000)
    trimmed_audio = l_seg[start_ms:end_ms]
    
    st.write("🔈 **편집된 소리 확인:**")
    buf = io.BytesIO()
    trimmed_audio.export(buf, format="wav")
    st.audio(buf.getvalue())
    
    if st.button("📊 AI 정밀 분석 시작", type="primary"):
        with st.spinner("에너지 패턴 계산 중..."):
            try:
                y_l, sr = librosa.load(io.BytesIO(buf.getvalue()), sr=22050)
                
                # 원어민 비교 데이터 생성
                tts = gTTS(text=target_word, lang='en')
                n_fp = io.BytesIO(); tts.write_to_fp(n_fp); n_fp.seek(0)
                n_seg = AudioSegment.from_file(n_fp)
                ns, ne = get_speech_bounds(n_seg)
                n_buf = io.BytesIO(); n_seg[ns:ne].export(n_buf, format="wav"); n_buf.seek(0)
                y_n, _ = librosa.load(n_buf, sr=sr)
                
                # 정규화 및 엔벨롭 추출
                y_l = librosa.util.normalize(y_l)
                y_n = librosa.util.normalize(y_n)
                env_l, env_n = get_rms_envelope(y_l), get_rms_envelope(y_n)
                
                # 점수 계산
                f_n = interp1d(np.linspace(0, 1, len(env_n)), env_n, fill_value="extrapolate")
                f_l = interp1d(np.linspace(0, 1, len(env_l)), env_l, fill_value="extrapolate")
                score = int(max(0, np.corrcoef(f_n(np.linspace(0, 1, 100)), f_l(np.linspace(0, 1, 100)))[0, 1]) * 100)
                
                # 결과 탭
                st.divider()
                st.subheader(f"분석 점수: {score}점")
                
                tab1, tab2 = st.tabs(["📊 리듬 대조", "✍️ 분석 노트"])
                with tab1:
                    fig_res, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
                    t_l = np.linspace(0, len(y_l)/sr, len(env_l))
                    ax1.plot(t_l, env_l, color='#1f77b4', lw=2)
                    ax1.axvline(x=t_l[np.argmax(env_l)], color='red', lw=2)
                    
                    t_n = np.linspace(0, len(y_n)/sr, len(env_n))
                    ax2.plot(t_n, env_n, color='gray', lw=2)
                    ax2.axvline(x=t_n[np.argmax(env_n)], color='red', lw=2)
                    
                    ax1.set_title("My Energy Envelope"); ax2.set_title("Native Energy Envelope")
                    plt.tight_layout(); st.pyplot(fig_res)
                    
            except Exception as e:
                st.error(f"분석 오류: {e}")

# 파일 정리
if os.path.exists("temp_raw.wav"): os.remove("temp_raw.wav")
if os.path.exists("native_voice.mp3"): os.remove("native_voice.mp3")
