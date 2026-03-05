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

st.set_page_config(page_title="AI 발음 분석기", layout="wide")
st.title("🎙️ AI-Native 발음 정밀 분석기")

# 1. 목표 문장 설정
target_text = "The quick brown fox jumps over the lazy dog."
st.info(f"🎯 **Target:** {target_text}")

# 2. 녹음 섹션
audio = mic_recorder(
    start_prompt="🎤 녹음 시작",
    stop_prompt="🛑 녹음 완료",
    key="recorder"
)

if audio:
    try:
        # 파일 변환 및 저장
        learner_segment = AudioSegment.from_file(io.BytesIO(audio['bytes']))
        learner_segment.export("temp_learner.wav", format="wav")
        
        tts = gTTS(text=target_text, lang='en')
        tts.save("temp_native.mp3")
        native_segment = AudioSegment.from_file("temp_native.mp3", format="mp3")
        native_segment.export("temp_native.wav", format="wav")

        # 데이터 로드
        y_learner, sr_l = librosa.load("temp_learner.wav", sr=22050)
        y_native, _ = librosa.load("temp_native.wav", sr=sr_l)

        # 탭 구성
        tab1, tab2, tab3 = st.tabs(["🎯 인식 결과", "🔊 음파 비교", "📈 피치(억양) 분석"])

        # --- Tab 1: 인식 결과 ---
        with tab1:
            r = sr.Recognizer()
            with sr.AudioFile("temp_learner.wav") as source:
                audio_data = r.record(source)
                try:
                    transcript = r.recognize_google(audio_data, language='en-US')
                    score = SequenceMatcher(None, target_text.lower(), transcript.lower()).ratio()
                    st.metric("정확도", f"{int(score * 100)}%")
                    st.success(f"**인식된 문장:** {transcript}")
                except:
                    st.error("인식에 실패했습니다.")

        # --- Tab 2: 음파 비교 ---
        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                st.write("나의 발음:")
                st.audio("temp_learner.wav")
            with c2:
                st.write("원어민 가이드:")
                st.audio("temp_native.mp3")
            
            # 음파 시각화
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

        # --- Tab 3: 피치 분석 ---
        with tab3:
            st.subheader("억양 및 높낮이 분석")
            
            # 다시듣기 추가
            c3, c4 = st.columns(2)
            with c3:
                st.write("내 목소리 확인:")
                st.audio("temp_learner.wav")
            with c4:
                st.write("원어민 가이드 확인:")
                st.audio("temp_native.mp3")

            # 피치 추출 (fmin, fmax 조절로 Halving/Doubling 억제)
            f0_l, voiced_flag_l, voiced_probs_l = librosa.pyin(y_learner, 
                                                              fmin=librosa.note_to_hz('C2'), 
                                                              fmax=librosa.note_to_hz('C6'))
            f0_n, voiced_flag_n, voiced_probs_n = librosa.pyin(y_native, 
                                                              fmin=librosa.note_to_hz('C2'), 
                                                              fmax=librosa.note_to_hz('C6'))

            fig2, ax = plt.subplots(figsize=(12, 5))
            
            # 유성음 구간(Voiced)만 추출하여 점선으로 표시
            times_n = librosa.times_like(f0_n)
            times_l = librosa.times_like(f0_l)

            # 원어민 피치 (회색 점선)
            ax.plot(times_n[voiced_flag_n], f0_n[voiced_flag_n], 'o--', 
                    label='Native', color='lightgray', markersize=3, alpha=0.7)
            
            # 학습자 피치 (파란색 점선)
            ax.plot(times_l[voiced_flag_l], f0_l[voiced_flag_l], 'o--', 
                    label='You', color='#1f77b4', markersize=4)

            ax.set_ylim([50, 400]) # 일반적인 음성 주파수 대역 고정
            ax.set_title("Pitch Tracking (Voiced Segments Only)")
            ax.set_ylabel("Frequency (Hz)")
            ax.legend()
            st.pyplot(fig2)
            
            st.info("💡 **그래프 보는 법:** 점으로 표시된 부분이 실제 목소리가 나온 유성음 구간입니다. 억양의 흐름(곡선)이 원어민과 비슷한지 확인하세요.")

    except Exception as e:
        st.error(f"분석 중 오류 발생: {e}")
    finally:
        # 임시 파일 정리 (선택사항)
        if os.path.exists("temp_native.mp3"): os.remove("temp_native.mp3")
        if os.path.exists("temp_native.wav"): os.remove("temp_native.wav")
