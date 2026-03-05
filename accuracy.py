import streamlit as st
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import io
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from difflib import SequenceMatcher

st.set_page_config(page_title="AI 발음 정밀 분석기", layout="wide")
st.title("🎙️ AI-Native 발음 클리닉 (Pitch & Waveform)")

# 1. 목표 문장 설정
target_text = "The quick brown fox jumps over the lazy dog."
st.info(f"🎯 **오늘의 도전 문장:** {target_text}")

# 2. 녹음 섹션
audio = mic_recorder(
    start_prompt="🎤 녹음 시작 (문장을 읽어주세요)",
    stop_prompt="🛑 녹음 완료",
    key="recorder"
)

if audio:
    try:
        # 학습자 데이터를 WAV로 강제 변환 및 저장
        learner_segment = AudioSegment.from_file(io.BytesIO(audio['bytes']))
        learner_segment.export("temp_learner.wav", format="wav")
        
        # 원어민 가이드 생성 및 저장
        tts = gTTS(text=target_text, lang='en')
        tts.save("temp_native.mp3")
        native_segment = AudioSegment.from_file("temp_native.mp3", format="mp3")
        native_segment.export("temp_native.wav", format="wav")

        # 데이터 로드 (분석용)
        y_learner, sr_l = librosa.load("temp_learner.wav", sr=22050)
        y_native, _ = librosa.load("temp_native.wav", sr=sr_l)

        # ---------------------------------------------------------
        # 탭 구성 (Step 1, 2, 3)
        # ---------------------------------------------------------
        tab1, tab2, tab3 = st.tabs(["🎯 AI 점수 확인", "🔊 음파 & 다시듣기", "📈 피치(억양) 분석"])

        # --- Tab 1: AI 점수 확인 ---
        with tab1:
            st.subheader("인식 결과 및 정확도")
            r = sr.Recognizer()
            with sr.AudioFile("temp_learner.wav") as source:
                audio_data = r.record(source)
                try:
                    transcript = r.recognize_google(audio_data, language='en-US')
                    score = SequenceMatcher(None, target_text.lower(), transcript.lower()).ratio()
                    st.metric("발음 정확도 점수", f"{int(score * 100)}점")
                    st.success(f"**AI 인식 결과:** {transcript}")
                except:
                    st.error("발음을 인식하지 못했습니다. 다시 녹음해보세요.")

        # --- Tab 2: 음파 & 다시듣기 ---
        with tab2:
            st.subheader("소리 비교 및 음파 대조")
            c1, c2 = st.columns(2)
            with c1:
                st.write("나의 발음:")
                st.audio("temp_learner.wav")
            with c2:
                st.write("원어민 가이드:")
                st.audio("temp_native.mp3")
            
            st.divider()
            st.write("📊 **음파(Waveform) 비교 (박자와 강세)**")
            
            # 시간 정렬 (Padding)
            max_len = max(len(y_learner), len(y_native))
            y_l_pad = librosa.util.fix_length(y_learner, size=max_len)
            y_n_pad = librosa.util.fix_length(y_native, size=max_len)

            fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
            librosa.display.waveshow(y_n_pad, sr=sr_l, ax=ax1, color='lightgray')
            ax1.set_title("Native Speaker")
            librosa.display.waveshow(y_l_pad, sr=sr_l, ax=ax2, color='skyblue')
            ax2.set_title("Learner")
            plt.tight_layout()
            st.pyplot(fig1)

        # --- Tab 3: 피치(억양) 분석 ---
        with tab3:
            st.subheader("억양(Intonation) 정밀 분석")
            st.write("파란색 곡선이 나의 목소리 높낮이 변화(피치)입니다. 원어민의 곡선 흐름과 비교해보세요.")
            
            # 피치 추출 (Yin 알고리즘 사용)
            f0_l, voiced_flag_l, voiced_probs_l = librosa.pyin(y_learner, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            f0_n, voiced_flag_n, voiced_probs_n = librosa.pyin(y_native, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

            fig2, ax = plt.subplots(figsize=(12, 4))
            # 원어민 피치 (회색)
            times_n = librosa.times_like(f0_n)
            ax.plot(times_n, f0_n, label='Native', color='lightgray', linewidth=3, alpha=0.6)
            # 학습자 피치 (파란색)
            times_l = librosa.times_like(f0_l)
            ax.plot(times_l, f0_l, label='You', color='#1f77b4', linewidth=2)
            
            ax.set_title("Pitch Curve (Intonation) Comparison")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_xlabel("Time (s)")
            ax.legend()
            st.pyplot(fig2)
            
            st.info("💡 **팁:** 문장의 끝에서 원어민처럼 피치가 내려가는지(내림조) 혹은 올라가는지 확인해보세요.")

    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
    finally:
        # 임시 파일 정리 (선택 사항)
        if os.path.exists("temp_native.mp3"): os.remove("temp_native.mp3")
        if os.path.exists("temp_native.wav"): os.remove("temp_native.wav")

# 사이드바 가이드
st.sidebar.title("📚 학습 단계")
st.sidebar.write("""
1. **점수 확인**: 전체적인 정확도를 체크합니다.
2. **음파 비교**: 단어 사이의 '리듬'과 '강세'를 봅니다.
3. **피치 분석**: 목소리의 '높낮이(억양)'를 비교합니다.
""")
