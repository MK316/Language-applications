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
    div.stButton > button { height: 3.5em; font-weight: bold; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- [2] 세션 상태 초기화 (오류 방지 핵심) ---
if 'last_audio_id' not in st.session_state: st.session_state.last_audio_id = None
if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
if 'final_wav' not in st.session_state: st.session_state.final_wav = None

# --- [3] 분석 엔진 함수 ---
def get_speech_bounds(audio_segment, silence_thresh=-45, min_silence_len=50):
    intervals = detect_nonsilent(audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    if not intervals: return 0, len(audio_segment)
    return intervals[0][0], intervals[-1][1]

def get_rms_envelope(y, hop_length=256):
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    return np.convolve(rms, np.ones(5)/5, mode='same')

def calculate_advanced_score(env_n, env_l):
    if len(env_n) < 2 or len(env_l) < 2: return 0, 0, 0
    f_n = interp1d(np.linspace(0, 1, len(env_n)), env_n, fill_value="extrapolate")
    f_l = interp1d(np.linspace(0, 1, len(env_l)), env_l, fill_value="extrapolate")
    new_x = np.linspace(0, 1, 100)
    shape_corr = np.corrcoef(f_n(new_x), f_l(new_x))[0, 1]
    
    thresh_n, thresh_l = np.max(env_n) * 0.2, np.max(env_l) * 0.2
    dur_n, dur_l = np.sum(env_n > thresh_n) / len(env_n), np.sum(env_l > thresh_l) / len(env_l)
    
    area_n, area_l = np.sum(env_n) / len(env_n), np.sum(env_l) / len(env_l)
    area_score = 1 - abs(area_n - area_l) / (area_n + 1e-6)

    final_score = (max(0, shape_corr) * 0.5 + (1 - abs(dur_n - dur_l)) * 0.3 + max(0, area_score) * 0.2) * 100
    return int(final_score), dur_l * 100, dur_n * 100

# --- [4] 앱 UI ---
st.title("🎙️ Word Stress Master")
word_db = {
    "Photograph (1음절)": "photograph", 
    "Photographer (2음절)": "photographer", 
    "Education (3음절)": "education",
    "Record (Noun-1음절)": "record",
    "Record (Verb-2음절)": "record"
}
selected_label = st.selectbox("단어 선택:", list(word_db.keys()))
target_word = word_db[selected_label]

if st.button("🔊 원어민 표준 발음 듣기"):
    tts = gTTS(text=target_word, lang='en')
    tts.save("native_voice.mp3")
    st.audio("native_voice.mp3")

st.divider()
st.subheader(f"🎯 연습: {target_word.upper()}")

# 녹음 컴포넌트
audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 녹음 완료", key="word_recorder")

if audio:
    # 새로운 녹음 시 세션 리셋
    if audio['id'] != st.session_state.last_audio_id:
        st.session_state.last_audio_id = audio['id']
        st.session_state.analysis_done = False
        st.session_state.final_wav = None

    audio_bytes = audio['bytes']
    l_seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
    y_full, sr_f = librosa.load(io.BytesIO(audio_bytes), sr=22050)
    full_dur = len(y_full) / sr_f

    st.markdown("#### ✂️ 분석 구간 설정")
    auto_s, auto_e = get_speech_bounds(l_seg)
    trim_range = st.slider("구간 선택:", 0.0, float(full_dur), (float(auto_s/1000), float(auto_e/1000)), step=0.01)

    # 구간 설정용 파형 프리뷰
    fig_prev = plt.figure(figsize=(10, 2.2))
    axp = fig_prev.add_axes([0, 0.2, 1, 0.8]) 
    librosa.display.waveshow(y_full, sr=sr_f, ax=axp, color='skyblue', alpha=0.5)
    axp.axvline(x=trim_range[0], color='red', lw=2, ls='--')
    axp.axvline(x=trim_range[1], color='red', lw=2, ls='--')
    axp.set_xlim(0, full_dur); axp.set_yticks([]); st.pyplot(fig_prev)

    # 구간 잘라내기
    start_ms, end_ms = int(trim_range[0] * 1000), int(trim_range[1] * 1000)
    trimmed_buf = io.BytesIO()
    l_seg[start_ms:end_ms].export(trimmed_buf, format="wav")
    
    st.write("🔈 **선택된 구간 미리듣기:**")
    st.audio(trimmed_buf.getvalue())
    
    if st.button("📊 색상 기반 에너지 대조 분석 시작", type="primary"):
        st.session_state.analysis_done = True
        st.session_state.final_wav = trimmed_buf.getvalue()

# --- [5] 결과 분석 (세션 데이터 체크 추가) ---
if st.session_state.analysis_done and st.session_state.final_wav is not None:
    with st.spinner("리듬 및 에너지 대조 중..."):
        try:
            y_l, sr = librosa.load(io.BytesIO(st.session_state.final_wav), sr=22050)
            
            # 원어민 데이터 생성
            tts = gTTS(text=target_word, lang='en')
            n_fp = io.BytesIO(); tts.write_to_fp(n_fp); n_fp.seek(0)
            n_seg = AudioSegment.from_file(n_fp)
            ns, ne = get_speech_bounds(n_seg)
            n_buf = io.BytesIO(); n_seg[ns:ne].export(n_buf, format="wav"); n_buf.seek(0)
            y_n, _ = librosa.load(n_buf, sr=sr)
            
            y_l, y_n = librosa.util.normalize(y_l), librosa.util.normalize(y_n)
            env_l, env_n = get_rms_envelope(y_l), get_rms_envelope(y_n)
            score, dur_l, dur_n = calculate_advanced_score(env_n, env_l)
            
            st.divider()
            c1, c2 = st.columns(2)
            c1.metric("종합 점수", f"{score}점")
            c2.metric("강세 구간 비중", f"{dur_l:.1f}%", f"{dur_l - dur_n:+.1f}%")

            tab1, tab2 = st.tabs(["📊 시각적 에너지 대조", "✍️ 분석 가이드"])
            with tab1:
                fig_res, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
                
                # 학습자 시각화 (웨이브폼 + 엔벨롭 + 색상 면적)
                t_l = np.linspace(0, len(y_l)/sr, len(env_l))
                librosa.display.waveshow(y_l, sr=sr, ax=ax1, color='skyblue', alpha=0.3)
                ax1.plot(t_l, env_l, color='#1f77b4', lw=2.5, label='Energy Envelope')
                ax1.fill_between(t_l, 0, env_l, where=(env_l > np.max(env_l)*0.2), color='#1f77b4', alpha=0.3)
                ax1.axvline(x=t_l[np.argmax(env_l)], color='red', lw=2)
                
                # 원어민 시각화
                t_n = np.linspace(0, len(y_n)/sr, len(env_n))
                librosa.display.waveshow(y_n, sr=sr, ax=ax2, color='lightgray', alpha=0.3)
                ax2.plot(t_n, env_n, color='gray', lw=2.5)
                ax2.fill_between(t_n, 0, env_n, where=(env_n > np.max(env_n)*0.2), color='gray', alpha=0.3)
                ax2.axvline(x=t_n[np.argmax(env_n)], color='red', lw=2)
                
                ax1.set_title("My Energy Profile (Waveform Overlay)"); ax2.set_title("Native Standard Profile")
                plt.tight_layout(); st.pyplot(fig_res)
                st.info("💡 파란색 영역의 '넓이(길이)'와 '높이(강도)'가 원어민의 회색 영역과 비슷해야 합니다.")

        except Exception as e:
            st.error(f"분석 중 오류가 발생했습니다: {e}")
