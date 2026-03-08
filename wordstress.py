import streamlit as st
from streamlit_mic_recorder import mic_recorder
import speech_recognition as speech_rec
import io, os, librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from scipy.interpolate import interp1d

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
    norm_rms_n = f_n(new_x)
    norm_rms_l = f_l(new_x)
    
    correlation = np.corrcoef(norm_rms_n, norm_rms_l)[0, 1]
    return int(max(0, correlation) * 100) if not np.isnan(correlation) else 0

# --- 2. 앱 설정 및 데이터 ---
st.set_page_config(page_title="Word Stress Master", layout="wide")
st.title("🎙️ Word Stress & Amplitude Master")

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
    
    if st.button("🔊 원어민 발음 듣기"):
        tts = gTTS(text=target_word, lang='en')
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        st.audio(mp3_fp)

# --- 4. Step 2: 나의 발음 녹음 ---
st.subheader(f"🎯 도전 단어: **{target_word.upper()}**")
col_rec, _ = st.columns([1, 2])
with col_rec:
    audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 녹음 완료", key="word_recorder")

# --- 5. Step 3: 분석 (메모리 직접 참조 방식) ---
if audio:
    audio_bytes = audio['bytes']
    
    with st.spinner("강세 패턴 분석 중..."):
        try:
            # [해결] 파일 대신 BytesIO 사용
            audio_stream = io.BytesIO(audio_bytes)
            l_seg = AudioSegment.from_file(audio_stream)
            
            # 원어민 TTS 생성 (메모리 상에서 처리)
            tts = gTTS(text=target_word, lang='en')
            n_mp3_fp = io.BytesIO()
            tts.write_to_fp(n_mp3_fp)
            n_mp3_fp.seek(0)
            n_seg = AudioSegment.from_file(n_mp3_fp)
            
            # 무음 제거 및 구간 추출
            l_start, l_end = get_speech_bounds(l_seg)
            n_start, n_end = get_speech_bounds(n_seg)
            
            final_l_seg = l_seg[l_start:l_end]
            final_n_seg = n_seg[n_start:n_end]
            
            # librosa 로드를 위해 임시 메모리 버퍼 활용
            l_buffer = io.BytesIO()
            final_l_seg.export(l_buffer, format="wav")
            l_buffer.seek(0)
            y_l, sr = librosa.load(l_buffer, sr=22050)
            
            n_buffer = io.BytesIO()
            final_n_seg.export(n_buffer, format="wav")
            n_buffer.seek(0)
            y_n, _ = librosa.load(n_buffer, sr=sr)
            
            y_l = normalize_audio(y_l)
            y_n = normalize_audio(y_n)

            st.divider()
            st.success("✅ 분석 완료!")
            
            tab_wave, tab_analysis = st.tabs(["📊 파형 대조 분석", "📝 학습 성찰 기록"])
            
            with tab_wave:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
                librosa.display.waveshow(y_l, sr=sr, ax=ax1, color='#1f77b4', alpha=0.7)
                librosa.display.waveshow(y_n, sr=sr, ax=ax2, color='#A9A9A9', alpha=0.6)
                
                # 강세 Peak (최대 진폭 지점) 시각화
                if len(y_l) > 0 and len(y_n) > 0:
                    ax1.axvline(x=librosa.samples_to_time(np.argmax(np.abs(y_l)), sr=sr), color='red', linestyle='--')
                    ax2.axvline(x=librosa.samples_to_time(np.argmax(np.abs(y_n)), sr=sr), color='red', linestyle='--')
                
                ax1.set_title("My Stress Pattern"); ax2.set_title("Native Stress Pattern")
                plt.tight_layout()
                st.pyplot(fig)
                
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("길이 편차", f"{len(y_l)/sr:.2f}s", delta=f"{len(y_l)/sr - len(y_n)/sr:.2f}s")
                with c2: st.metric("강세 일치도", f"{calculate_stress_score(y_n, y_l, sr)}점")
                with c3:
                    guide = selected_label.split("(")[1].replace(")", "") if "(" in selected_label else "강세를 확인하세요."
                    st.info(f"💡 **팁:** {guide}")

        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")
