import streamlit as st
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import io, os, librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from difflib import SequenceMatcher

# --- 유틸리티 함수 ---

def smooth_and_interpolate(f0, window_size=7):
    mask = np.isnan(f0)
    if np.any(~mask):
        f0[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), f0[~mask])
    if len(f0) < window_size: return f0
    return np.convolve(f0, np.ones(window_size)/window_size, mode='same')

def get_aligned_pitch(y, sr_rate, common_size=200):
    f0, voiced_flag, _ = librosa.pyin(y, fmin=70, fmax=400)
    f0[~voiced_flag] = np.nan
    f0_smooth = smooth_and_interpolate(f0)
    resampled = np.interp(np.linspace(0, 1, common_size), np.linspace(0, 1, len(f0_smooth)), f0_smooth)
    return resampled

# --- 스트림릿 설정 ---
st.set_page_config(page_title="AI 발음 분석기", layout="wide")

# 샘플 문장 리스트 (난이도 및 유성음 비율 고려)
sample_sentences = {
    "Level 1: 기본 모음": "Rain rain go away.",
    "Level 2: 비음 연습": "Mummy may marry Miller.",
    "Level 3: 유음 연습": "Lily loves yellow lemons.",
    "Level 4: 문장 연결": "Online learning is normal now.",
    "Level 5: 의문문 억양": "Where are you roaming?",
    "Level 6: 감정 표현": "I am really happy now.",
    "Level 7: 긴 문장 연결": "My mama makes money mainly on Mondays.",
    "Level 8: 복합 발음": "The blue moon is looming over the river.",
    "Level 9: 강조와 억양": "No one knows why the lion is running.",
    "Level 10: 최종 도전": "Learning a new language opens a new world."
}

st.title("🎙️ AI-Native 발음 클리닉: 10단계 챌린지")

# --- Step 1: 문장 선택 (Dropdown) ---
st.subheader("1단계: 연습할 문장 선택하기")
selected_level = st.selectbox("난이도를 선택하세요 (1~10):", list(sample_sentences.keys()))
target_text = sample_sentences[selected_level]

# 문장 강조 박스 (CSS 사용)
st.markdown(f"""
    <div style="border: 2px solid #1f77b4; border-radius: 10px; padding: 20px; background-color: #f0f2f6; text-align: center;">
        <h2 style="color: #1f77b4; margin: 0;">"{target_text}"</h2>
    </div>
    """, unsafe_allow_html=True)

st.write("") # 간격 조절
st.write("위 문장을 충분히 익힌 후, 준비가 되면 아래 버튼을 눌러 녹음을 시작하세요.")

# --- Step 2: 녹음 및 분석 ---
audio = mic_recorder(
    start_prompt="🎤 녹음 시작",
    stop_prompt="🛑 녹음 완료",
    key="recorder"
)

if audio:
    try:
        # 오디오 처리 및 무음 제거
        learner_segment = AudioSegment.from_file(io.BytesIO(audio['bytes']))
        learner_segment = learner_segment.strip_silence(silence_thresh=-40)
        learner_segment.export("temp_learner.wav", format="wav")
        
        tts = gTTS(text=target_text, lang='en')
        tts.save("temp_native.mp3")
        native_segment = AudioSegment.from_file("temp_native.mp3", format="mp3")
        native_segment = native_segment.strip_silence(silence_thresh=-40)
        native_segment.export("temp_native.wav", format="wav")

        y_learner, sr_l = librosa.load("temp_learner.wav", sr=22050)
        y_native, _ = librosa.load("temp_native.wav", sr=sr_l)

        # 탭 구성
        tab1, tab2, tab3 = st.tabs(["🎯 인식 결과 & 점수", "🔊 음파 대조", "📈 피치(억양) 분석"])

        # --- Tab 1: 인식 결과 ---
        with tab1:
            st.subheader("AI 피드백")
            r = sr.Recognizer()
            with sr.AudioFile("temp_learner.wav") as source:
                audio_data = r.record(source)
                try:
                    transcript = r.recognize_google(audio_data, language='en-US')
                    score = SequenceMatcher(None, target_text.lower().replace('.', ''), transcript.lower()).ratio()
                    
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.metric("나의 정확도 점수", f"{int(score * 100)}점")
                    with c2:
                        st.success(f"**AI 인식 결과:** {transcript}")
                    
                    st.divider()
                    st.info("💡 **안내:** 점수 확인이 끝나셨나요? 상단의 **[🔊 음파 대조]**와 **[📈 피치 분석]** 탭을 눌러 원어민과 나의 발음을 더 자세히 비교해 보세요!")
                
                except:
                    st.error("발음을 인식하지 못했습니다. 배경 소음을 줄이고 다시 한번 녹음해 보세요.")

        # --- Tab 2: 음파 대조 ---
        with tab2:
            st.subheader("리듬과 강세 비교")
            col_a, col_b = st.columns(2)
            with col_a: st.audio("temp_learner.wav"); st.caption("나의 목소리")
            with col_b: st.audio("temp_native.mp3"); st.caption("원어민 가이드")
            
            max_len = max(len(y_learner), len(y_native))
            y_l_pad = librosa.util.fix_length(y_learner, size=max_len)
            y_n_pad = librosa.util.fix_length(y_native, size=max_len)

            fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
            librosa.display.waveshow(y_n_pad, sr=sr_l, ax=ax1, color='lightgray')
            ax1.set_title("Native Speaker Waveform")
            librosa.display.waveshow(y_l_pad, sr=sr_l, ax=ax2, color='skyblue')
            ax2.set_title("Your Waveform")
            plt.tight_layout()
            st.pyplot(fig1)

        # --- Tab 3: 피치 분석 ---
        with tab3:
            st.subheader("억양(Intonation) 흐름 비교")
            st.audio("temp_learner.wav")
            st.audio("temp_native.mp3")

            f0_l_aligned = get_aligned_pitch(y_learner, sr_l)
            f0_n_aligned = get_aligned_pitch(y_native, sr_l)

            fig2, ax = plt.subplots(figsize=(12, 5))
            time_axis = np.linspace(0, 100, 200)

            ax.plot(time_axis, f0_n_aligned, label='Native', color='lightgray', linewidth=4, alpha=0.6)
            ax.plot(time_axis, f0_l_aligned, label='You', color='#1f77b4', linewidth=2.5)

            ax.set_title("Intonation Contour Comparison (%)")
            ax.set_xlabel("Sentence Progress (%)")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_ylim([50, 400])
            ax.legend()
            st.pyplot(fig2)
            st.caption("회색 선(원어민)의 굴곡에 맞춰 파란색 선(나)이 움직이는지 확인해 보세요.")

    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
    finally:
        # 임시 파일 정리
        for f in ["temp_native.mp3", "temp_native.wav", "temp_learner.wav"]:
            if os.path.exists(f): os.remove(f)

# 사이드바 가이드
st.sidebar.markdown("""
### 🏛️ 발음 클리닉 안내
본 앱은 학생의 자기주도적 발음 교정을 위해 설계되었습니다.
1. **1단계**: 문장을 고르고 큰 소리로 읽어보세요.
2. **2단계**: AI 점수로 현재 수준을 파악하세요.
3. **3단계**: 시각적 데이터를 통해 원어민과의 미세한 차이를 발견하세요.
""")
