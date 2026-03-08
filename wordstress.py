import streamlit as st
from streamlit_mic_recorder import mic_recorder
import speech_recognition as speech_rec
import io, os, librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

# --- 1. 분석 엔진 함수 ---
def get_speech_bounds(audio_segment, silence_thresh=-45, min_silence_len=50):
    """음성 시작과 끝을 정밀하게 탐지"""
    intervals = detect_nonsilent(audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    if not intervals: return 0, len(audio_segment)
    return intervals[0][0], intervals[-1][1]

def normalize_audio(y):
    """진폭 정규화 (비교를 위해 최대 크기를 1로 맞춤)"""
    return librosa.util.normalize(y)

def calculate_stress_score(y_n, y_l, sr):
    """
    진폭 패턴(Envelope)의 상관관계를 통해 강세 일치도 계산
    강세는 에너지(RMS)의 변화이므로 전체적인 에너지 흐름을 비교합니다.
    """
    # 0.1초 단위의 에너지(RMS) 추출
    hop_length = 512
    rms_n = librosa.feature.rms(y=y_n, hop_length=hop_length)[0]
    rms_l = librosa.feature.rms(y=y_l, hop_length=hop_length)[0]
    
    # 두 배열의 길이를 동일하게 맞춤 (리샘플링)
    from scipy.interpolate import interp1d
    f_n = interp1d(np.linspace(0, 1, len(rms_n)), rms_n)
    f_l = interp1d(np.linspace(0, 1, len(rms_l)), rms_l)
    
    new_x = np.linspace(0, 1, 100)
    norm_rms_n = f_n(new_x)
    norm_rms_l = f_l(new_x)
    
    # 상관계수 계산
    correlation = np.corrcoef(norm_rms_n, norm_rms_l)[0, 1]
    return int(max(0, correlation) * 100) if not np.isnan(correlation) else 0

# --- 2. 앱 설정 및 데이터 ---
st.set_page_config(page_title="Word Stress Master", layout="wide")
st.title("🎙️ Word Stress & Amplitude Master")
st.markdown("예비 영어교사를 위한 **단어 강세 및 시각적 파형 분석** 학습 도구입니다.")

# 학습용 단어 DB (강세 위치 정보 포함)
word_db = {
    "Photograph (1음절 강세)": "photograph",
    "Photographer (2음절 강세)": "photographer",
    "Record (Noun - 1음절)": "record",
    "Record (Verb - 2음절)": "record",
    "Education (3음절 강세)": "education",
}

# --- 3. Step 1: 단어 선택 및 원어민 청취 ---
with st.sidebar:
    st.header("📍 Step 1: 단어 선택")
    selected_label = st.selectbox("학습할 단어를 선택하세요:", list(word_db.keys()))
    target_word = word_db[selected_label]
    
    if st.button("🔊 원어민 발음 듣기"):
        tts = gTTS(text=target_word, lang='en')
        tts.save("native.mp3")
        st.audio("native.mp3")

# --- 4. Step 2: 나의 발음 녹음 ---
st.subheader(f"🎯 도전 단어: **{target_word.upper()}**")
col_rec, col_info = st.columns([1, 2])

with col_rec:
    st.write("아래 버튼을 눌러 단어를 발음하세요.")
    audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 녹음 완료", key="word_recorder")

# --- 5. Step 3: 시각적 대조 및 분석 (핵심 단계) ---
if audio:
    audio_bytes = audio['bytes']
    with open("learner_raw.wav", "wb") as f:
        f.write(audio_bytes)
    
    # 오디오 로드 및 정규화
    y_l_raw, sr = librosa.load("learner_raw.wav", sr=22050)
    
    # TTS로 원어민 오디오 생성 및 로드
    tts = gTTS(text=target_word, lang='en')
    tts.save("native_target.mp3")
    audio_n_seg = AudioSegment.from_file("native_target.mp3")
    
    # 무음 제거 및 구간 추출 (Trim)
    l_seg = AudioSegment.from_file("learner_raw.wav")
    l_start, l_end = get_speech_bounds(l_seg)
    n_start, n_end = get_speech_bounds(audio_n_seg)
    
    final_l = l_seg[l_start:l_e] if 'l_e' in locals() else l_seg[l_start:l_end]
    final_n = audio_n_seg[n_start:n_end]
    
    final_l.export("l_trimmed.wav", format="wav")
    final_n.export("n_trimmed.wav", format="wav")
    
    y_l, _ = librosa.load("l_trimmed.wav", sr=sr)
    y_n, _ = librosa.load("n_trimmed.wav", sr=sr)
    
    # 정규화
    y_l = normalize_audio(y_l)
    y_n = normalize_audio(y_n)

    # UI 출력
    st.divider()
    st.success("✅ 분석 준비 완료! 원어민의 강세 위치와 비교해 보세요.")
    
    tab_wave, tab_analysis = st.tabs(["📊 파형 대조 분석", "📝 학습 성찰 기록"])
    
    with tab_wave:
        # 파형 대조 그래프
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
        
        # 학습자 파형
        librosa.display.waveshow(y_l, sr=sr, ax=ax1, color='#1f77b4', alpha=0.7)
        ax1.set_title(f"My Voice: {target_word}", fontsize=12)
        ax1.set_ylabel("Amplitude")
        
        # 원어민 파형
        librosa.display.waveshow(y_n, sr=sr, ax=ax2, color='#A9A9A9', alpha=0.6)
        ax2.set_title(f"Native Voice: {target_word}", fontsize=12)
        ax2.set_ylabel("Amplitude")
        
        # 강세 Peak 표시 (최대 진폭 지점)
        peak_l = np.argmax(np.abs(y_l))
        peak_n = np.argmax(np.abs(y_n))
        ax1.axvline(x=librosa.samples_to_time(peak_l, sr=sr), color='red', linestyle='--', label='My Stress')
        ax2.axvline(x=librosa.samples_to_time(peak_n, sr=sr), color='red', linestyle='--', label='Target Stress')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # 상세 수치 비교
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("단어 길이 (초)", f"{len(y_l)/sr:.2f}s", delta=f"{len(y_l)/sr - len(y_n)/sr:.2f}s")
        with c2:
            # 강세 일치도 점수
            stress_score = calculate_stress_score(y_n, y_l, sr)
            st.metric("강세 패턴 일치도", f"{stress_score}점")
        with c3:
            st.write("**강세 가이드:**")
            st.info(selected_level.split("(")[1].replace(")", ""))

    with tab_analysis:
        st.write("### ✍️ 예비 교사 분석 노트 (Github 제출용)")
        analysis_text = st.text_area("파형을 보고 느낀 원어민과의 차이점과 개선 방안을 적어보세요:", 
                                     placeholder="예: 원어민은 첫 음절에서 진폭이 크게 솟구치는데, 나는 두 번째 음절이 더 높게 나타남. 강세 위치 수정 필요.")
        
        if st.button("복사용 마크다운 생성"):
            md_output = f"### Word Stress Analysis: {target_word}\n- **Stress Score:** {stress_score}점\n- **Duration:** {len(y_l)/sr:.2f}s\n- **Reflection:** {analysis_text}"
            st.code(md_output, language='markdown')

# --- 6. 마무리 및 파일 정리 ---
finally_paths = ["temp_entry.wav", "native.mp3", "learner_raw.wav", "native_target.mp3", "l_trimmed.wav", "n_trimmed.wav"]
for p in finally_paths:
    if os.path.exists(p):
        try: os.remove(p)
        except: pass
