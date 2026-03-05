import streamlit as st
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import io, os, librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from difflib import SequenceMatcher

# --- 유틸리티 함수 ---

def get_net_speaking_time(audio_path):
    """무음 구간을 제외한 실제 발화 시간(ms)을 계산"""
    audio = AudioSegment.from_file(audio_path)
    # -45dB 이하를 무음으로 간주, 최소 무음 길이를 100ms로 설정하여 발화 구간 추출
    nonsilent_chunks = detect_nonsilent(audio, min_silence_len=100, silence_thresh=-45)
    
    if not nonsilent_chunks:
        return 0
    
    # 각 발화 구간의 길이를 합산
    net_time_ms = sum([end - start for start, end in nonsilent_chunks])
    return net_time_ms / 1000.0  # 초 단위 변환

# --- 스트림릿 설정 ---
st.set_page_config(page_title="AI 발음 분석기", layout="wide")

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

st.title("🎙️ AI-Native 발음 & 유창성 클리닉")

# --- Step 1: 문장 선택 ---
selected_level = st.selectbox("난이도를 선택하세요 (1~10):", list(sample_sentences.keys()))
target_text = sample_sentences[selected_level]

st.markdown(f"""
    <div style="border: 2px solid #1f77b4; border-radius: 10px; padding: 20px; background-color: #f0f2f6; text-align: center;">
        <h2 style="color: #1f77b4; margin: 0;">"{target_text}"</h2>
    </div>
    """, unsafe_allow_html=True)

# --- Step 2: 녹음 ---
audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 녹음 완료", key="recorder")

if audio:
    try:
        # 오디오 저장 (순수 분석용)
        learner_raw = AudioSegment.from_file(io.BytesIO(audio['bytes']))
        learner_raw.export("temp_learner.wav", format="wav")
        
        tts = gTTS(text=target_text, lang='en')
        tts.save("temp_native.mp3")
        native_raw = AudioSegment.from_file("temp_native.mp3", format="mp3")
        native_raw.export("temp_native.wav", format="wav")

        # 실제 발화 시간 측정 (순수 스피치 구간만)
        learner_net_time = get_net_speaking_time("temp_learner.wav")
        native_net_time = get_net_speaking_time("temp_native.wav")

        # 분석용 데이터 로드 (librosa)
        y_learner, sr_l = librosa.load("temp_learner.wav", sr=22050)
        y_native, _ = librosa.load("temp_native.wav", sr=sr_l)

        # 탭 구성 (유창성 분석 탭 추가)
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 인식 결과", "⏱️ 유창성(속도) 분석", "🔊 음파 대조", "📈 피치(억양) 분석"])

        # --- Tab 1: 인식 결과 ---
        with tab1:
            st.subheader("AI 피드백")
            r = sr.Recognizer()
            with sr.AudioFile("temp_learner.wav") as source:
                audio_data = r.record(source)
                try:
                    transcript = r.recognize_google(audio_data, language='en-US')
                    score = SequenceMatcher(None, target_text.lower().replace('.', ''), transcript.lower()).ratio()
                    st.metric("발음 정확도", f"{int(score * 100)}점")
                    st.success(f"**AI 인식 결과:** {transcript}")
                except: st.error("인식 실패")
            st.info("💡 점수 확인 후, 옆의 **[⏱️ 유창성 분석]** 탭으로 이동해 보세요!")

        # --- Tab 2: 유창성(속도) 분석 (신규 추가) ---
        with tab2:
            st.subheader("발화 속도(Speech Rate) 분석")
            
            # 속도 비율 계산
            ratio = (learner_net_time / native_net_time) * 100 if native_net_time > 0 else 0
            
            c1, c2, c3 = st.columns(3)
            c1.metric("나의 실제 발화 시간", f"{learner_net_time:.2f}초")
            c2.metric("원어민 발화 시간", f"{native_net_time:.2f}초")
            c3.metric("원어민 대비 속도", f"{int(ratio)}%")

            # 시각적 피드백 바
            st.write("### 유창성 가이드")
            if 90 <= ratio <= 120:
                st.success("✅ **훌륭합니다!** 원어민과 거의 유사한 속도로 자연스럽게 발화하고 있습니다.")
            elif ratio < 90:
                st.warning("🚀 **조금 빨라요!** 너무 서두르기보다 단어 사이의 연음과 강세를 조금 더 살려보세요.")
            else:
                st.info("🐢 **조금 느려요!** 단어들을 더 매끄럽게 연결(Linking)해서 읽는 연습이 필요합니다.")
            
            st.caption("※ 침묵(Silence) 구간을 제외한 순수 말하기 시간만 측정되었습니다.")

        # --- Tab 3: 음파 대조 ---
        with tab3:
            st.audio("temp_learner.wav"); st.audio("temp_native.mp3")
            fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
            librosa.display.waveshow(y_native, sr=sr_l, ax=ax1, color='lightgray')
            librosa.display.waveshow(y_learner, sr=sr_l, ax=ax2, color='skyblue')
            st.pyplot(fig1)

        # --- Tab 4: 피치 분석 ---
        with tab4:
            f0_l, voiced_l, _ = librosa.pyin(y_learner, fmin=70, fmax=400)
            f0_n, voiced_n, _ = librosa.pyin(y_native, fmin=70, fmax=400)

            fig2, ax = plt.subplots(figsize=(12, 5))
            ax.plot(librosa.times_like(f0_n)[voiced_n], f0_n[voiced_n], 'o--', label='Native', color='lightgray', markersize=3, alpha=0.6)
            ax.plot(librosa.times_like(f0_l)[voiced_l], f0_l[voiced_l], 'o--', label='You', color='#1f77b4', markersize=4)
            ax.set_ylim([50, 400]); ax.legend()
            st.pyplot(fig2)

    except Exception as e: st.error(f"오류: {e}")
    finally:
        for f in ["temp_native.mp3", "temp_native.wav", "temp_learner.wav"]:
            if os.path.exists(f): os.remove(f)
