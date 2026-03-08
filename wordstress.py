import streamlit as st
from streamlit_mic_recorder import mic_recorder
import io, os, librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from scipy.interpolate import interp1d

# --- [CSS] 슬라이더와 그래프의 좌우 간격을 맞추기 위한 여백 조정 ---
st.markdown("""
    <style>
    .stSlider {
        padding-left: 25px;
        padding-right: 25px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. 분석 엔진 함수 ---
def get_speech_bounds(audio_segment, silence_thresh=-45, min_silence_len=50):
    intervals = detect_nonsilent(audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    if not intervals: return 0, len(audio_segment)
    return intervals[0][0], intervals[-1][1]

def normalize_audio(y):
    if len(y) == 0: return y
    return librosa.util.normalize(y)

def calculate_stress_score(y_n, y_l, sr):
    hop_length = 512
    rms_n = librosa.feature.rms(y=y_n, hop_length=hop_length)[0]
    rms_l = librosa.feature.rms(y=y_l, hop_length=hop_length)[0]
    if len(rms_n) < 2 or len(rms_l) < 2: return 0
    f_n = interp1d(np.linspace(0, 1, len(rms_n)), rms_n, fill_value="extrapolate")
    f_l = interp1d(np.linspace(0, 1, len(rms_l)), rms_l, fill_value="extrapolate")
    new_x = np.linspace(0, 1, 100)
    correlation = np.corrcoef(f_n(new_x), f_l(new_x))[0, 1]
    return int(max(0, correlation) * 100) if not np.isnan(correlation) else 0

# --- 2. 앱 설정 및 세션 초기화 ---
st.set_page_config(page_title="Word Stress Master", layout="wide")
st.title("🎙️ Word Stress & Amplitude Master")

if 'analysis_ready' not in st.session_state: st.session_state.analysis_ready = False
if 'prev_audio_id' not in st.session_state: st.session_state.prev_audio_id = None

word_db = {
    "Photograph (1음절 강세)": "photograph",
    "Photographer (2음절 강세)": "photographer",
    "Record (Noun - 1음절)": "record",
    "Record (Verb - 2음절)": "record",
    "Education (3음절 강세)": "education",
}

# --- 3. Step 1: 단어 선택 ---
with st.sidebar:
    st.header("📍 Step 1: 단어 선택")
    selected_label = st.selectbox("학습할 단어를 선택하세요:", list(word_db.keys()))
    target_word = word_db[selected_label]
    if st.button("🔊 원어민 표준 발음 듣기"):
        tts = gTTS(text=target_word, lang='en')
        tts.save("native_voice.mp3")
        st.audio("native_voice.mp3")

# --- 4. Step 2: 녹음 및 구간 설정 ---
st.subheader(f"🎯 도전 단어: **{target_word.upper()}**")
audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 녹음 완료", key="word_recorder")

if audio:
    if audio['id'] != st.session_state.prev_audio_id:
        st.session_state.analysis_ready = False
        st.session_state.prev_audio_id = audio['id']

    audio_bytes = audio['bytes']
    with open("temp_raw.wav", "wb") as f:
        f.write(audio_bytes)
    
    y_full, sr_full = librosa.load("temp_raw.wav", sr=22050)
    l_raw_seg = AudioSegment.from_file("temp_raw.wav")
    full_duration = len(y_full) / sr_full

    st.divider()
    st.markdown("#### ✂️ Step 3: 분석 구간 설정 (파형과 슬라이더 위치 동기화)")
    
    auto_s, auto_e = get_speech_bounds(l_raw_seg)
    
    # 1. 슬라이더 (CSS 여백 적용됨)
    trim_range = st.slider("아래 파형의 시간축을 확인하며 구간을 조절하세요 (단위: 초):", 
                           0.0, float(full_duration), 
                           (float(auto_s/1000), float(auto_e/1000)), step=0.01)

    # 2. 파형 프리뷰 (여백 제어로 슬라이더와 일치시킴)
    fig_prev = plt.figure(figsize=(12, 3))
    # [left, bottom, width, height] 여백을 최소화하여 슬라이더 폭과 일치시킴
    axp = fig_prev.add_axes([0.02, 0.2, 0.96, 0.7]) 
    
    librosa.display.waveshow(y_full, sr=sr_full, ax=axp, color='skyblue', alpha=0.5)
    axp.axvline(x=trim_range[0], color='red', linestyle='--', linewidth=2)
    axp.axvline(x=trim_range[1], color='red', linestyle='--', linewidth=2)
    
    axp.set_xlim(0, full_duration) # 시간축 범위 고정
    axp.set_xlabel("Time (seconds)", fontsize=9)
    axp.tick_params(axis='x', labelsize=8)
    axp.set_yticks([]) # Y축 진폭 숫자는 생략하여 깔끔하게 표시
    st.pyplot(fig_prev)

    # 선택 구간 미리 듣기
    start_ms, end_ms = int(trim_range[0] * 1000), int(trim_range[1] * 1000)
    trimmed_audio = l_raw_seg[start_ms:end_ms]
    
    col_play, col_btn = st.columns([2, 1])
    with col_play:
        st.write("🔈 선택된 구간 소리 확인:")
        trimmed_audio.export("temp_trimmed.wav", format="wav")
        st.audio("temp_trimmed.wav")
    
    with col_btn:
        st.write(" ") 
        if st.button("📊 이 구간으로 최종 분석 수행", use_container_width=True):
            st.session_state.analysis_ready = True

# --- 5. Step 4: 상세 분석 결과 ---
if st.session_state.get('analysis_ready'):
    with st.spinner("🎯 강세 에너지 패턴 분석 중..."):
        try:
            y_l, sr = librosa.load("temp_trimmed.wav", sr=22050)
            
            # 원어민 데이터 생성
            tts = gTTS(text=target_word, lang='en')
            tts.save("native_temp.mp3")
            n_seg_full = AudioSegment.from_file("native_temp.mp3")
            ns, ne = get_speech_bounds(n_seg_full)
            final_n_seg = n_seg_full[ns:ne]
            final_n_seg.export("native_trimmed.wav", format="wav")
            
            y_n, _ = librosa.load("native_trimmed.wav", sr=sr)
            y_l = normalize_audio(y_l); y_n = normalize_audio(y_n)

            st.success("🎉 분석 완료! 원어민의 강세 위치와 비교해 보세요.")
            
            tab1, tab2 = st.tabs(["📊 상세 강세 대조", "✍️ 분석 노트"])
            
            with tab1:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
                librosa.display.waveshow(y_l, sr=sr, ax=ax1, color='#1f77b4')
                librosa.display.waveshow(y_n, sr=sr, ax=ax2, color='#A9A9A9')
                
                # 강세 피크 시각화
                if len(y_l) > 0 and len(y_n) > 0:
                    ax1.axvline(x=librosa.samples_to_time(np.argmax(np.abs(y_l)), sr=sr), color='red', lw=2)
                    ax2.axvline(x=librosa.samples_to_time(np.argmax(np.abs(y_n)), sr=sr), color='red', lw=2)
                
                ax1.set_title("My Adjusted Stress Pattern"); ax2.set_title("Native Standard Pattern")
                plt.tight_layout(); st.pyplot(fig)
                
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("구간 길이", f"{len(y_l)/sr:.2f}s")
                with c2: st.metric("에너지 일치도", f"{calculate_stress_score(y_n, y_l, sr)}점")
                with c3:
                    tip = selected_label.split("(")[1].replace(")", "") if "(" in selected_label else "강세 확인"
                    st.info(f"💡 **강세 팁:** {tip}")

            with tab2:
                st.write("### ✍️ 예비 교사 분석 노트")
                reflection = st.text_area("파형을 보고 느낀 원어민과의 차이점과 개선 방안을 적어보세요.")
                if st.button("마크다운 복사용 텍스트 생성"):
                    st.code(f"### Analysis: {target_word}\n- Score: {calculate_stress_score(y_n, y_l, sr)}\n- Reflection: {reflection}")

        except Exception as e:
            st.error(f"분석 오류: {e}")
        finally:
            # 파일 정리
            for f in ["temp_raw.wav", "temp_trimmed.wav", "native_temp.mp3", "native_trimmed.wav", "native_voice.mp3"]:
                if os.path.exists(f): os.remove(f)
