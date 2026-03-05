import streamlit as st
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import io, os, librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from difflib import SequenceMatcher

# --- 스트림릿 설정 ---
st.set_page_config(page_title="AI 발음 분석기", layout="wide")

# 샘플 문장 리스트
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

st.title("🎙️ AI-Native 발음 클리닉 (Pitch Optimized)")

# --- Step 1: 문장 선택 ---
st.subheader("1단계: 연습할 문장 선택하기")
selected_level = st.selectbox("난이도를 선택하세요 (1~10):", list(sample_sentences.keys()))
target_text = sample_sentences[selected_level]

st.markdown(f"""
    <div style="border: 2px solid #1f77b4; border-radius: 10px; padding: 20px; background-color: #f0f2f6; text-align: center;">
        <h2 style="color: #1f77b4; margin: 0;">"{target_text}"</h2>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("위 문장을 충분히 익힌 후, 준비가 되면 아래 버튼을 눌러 녹음을 시작하세요.")

# --- Step 2: 녹음 및 분석 ---
audio = mic_recorder(
    start_prompt="🎤 녹음 시작",
    stop_prompt="🛑 녹음 완료",
    key="recorder"
)

if audio:
    try:
        # [핵심 수정 1] 앞뒤 무음을 매우 타이트하게 제거하여 그래프 시작/끝 최적화
        learner_segment = AudioSegment.from_file(io.BytesIO(audio['bytes']))
        learner_segment = learner_segment.strip_silence(silence_thresh=-45, padding=100) # 더 민감하게 조절
        learner_segment.export("temp_learner.wav", format="wav")
        
        tts = gTTS(text=target_text, lang='en')
        tts.save("temp_native.mp3")
        native_segment = AudioSegment.from_file("temp_native.mp3", format="mp3")
        native_segment = native_segment.strip_silence(silence_thresh=-45, padding=100) # 무음 제거
        native_segment.export("temp_native.wav", format="wav")

        y_learner, sr_l = librosa.load("temp_learner.wav", sr=22050)
        y_native, _ = librosa.load("temp_native.wav", sr=sr_l)

        # 탭 구성
        tab1, tab2, tab3 = st.tabs(["🎯 인식 결과 & 점수", "🔊 음파 대조", "📈 피치(억양) 분석"])

        # --- Tab 1: 인식 결과 --- (기존 유지)
        with tab1:
            st.subheader("AI 피드백")
            r = sr.Recognizer()
            with sr.AudioFile("temp_learner.wav") as source:
                audio_data = r.record(source)
                try:
                    transcript = r.recognize_google(audio_data, language='en-US')
                    score = SequenceMatcher(None, target_text.lower().replace('.', ''), transcript.lower()).ratio()
                    
                    c1, c2 = st.columns([1, 2])
                    with c1: st.metric("나의 정확도 점수", f"{int(score * 100)}점")
                    with c2: st.success(f"**AI 인식 결과:** {transcript}")
                    
                    st.divider()
                    st.info("💡 **안내:** 상단의 **[📈 피치 분석]** 탭을 눌러 정교한 억양 흐름을 비교해 보세요!")
                
                except: st.error("인식에 실패했습니다. 배경 소음을 줄이고 다시 녹음해 보세요.")

        # --- Tab 2: 음파 대조 --- (기존 유지)
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

        # --- Tab 3: 피치 분석 --- (스타일 및 타임라인 최적화)
        with tab3:
            st.subheader("억양(Intonation) 윤곽 비교")
            st.audio("temp_learner.wav")
            st.audio("temp_native.mp3")

            # [핵심 수정 2] 피치 추출 및 유성음 구간 점선 시각화
            f0_l, voiced_l, _ = librosa.pyin(y_learner, fmin=70, fmax=400)
            f0_n, voiced_n, _ = librosa.pyin(y_native, fmin=70, fmax=400)

            # 유성음 구간만 점선으로 표시 (image_0.png 스타일 적용)
            fig2, ax = plt.subplots(figsize=(12, 5))
            
            # 원어민 (회색 점선)
            ax.plot(librosa.times_like(f0_n)[voiced_n], f0_n[voiced_n], 'o--', 
                    label='Native', color='lightgray', markersize=3, alpha=0.6)
            
            # 학습자 (파란색 점선)
            ax.plot(librosa.times_like(f0_l)[voiced_l], f0_l[voiced_l], 'o--', 
                    label='You', color='#1f77b4', markersize=4)

            ax.set_title("Pitch Tracking (Voiced Segments Only)")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_ylim([50, 400])
            ax.legend()
            st.pyplot(fig2)
            
            # [핵심 수정 3] 교육적 안내 멘트 수정
            st.info("💡 **최적화 완료:** 앞뒤 무음을 제거하여 억양 본연의 흐름에 집중했습니다. 점으로 표시된 유성음 구간의 억양 멜로디를 원어민과 비교해보세요.")

    except Exception as e: st.error(f"오류: {e}")
    finally:
        for f in ["temp_native.mp3", "temp_native.wav", "temp_learner.wav"]:
            if os.path.exists(f): os.remove(f)

# 사이드바 가이드 (기존 유지)
st.sidebar.markdown("""
### 🏛️ 발음 클리닉 안내
1. **1단계**: 문장을 고르고 익히세요.
2. **2단계**: AI 점수로 정확도를 체크하세요.
3. **3단계**: 시각적 데이터로 미세한 차이를 발견하세요.
""")
