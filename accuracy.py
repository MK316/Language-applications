import streamlit as st
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import io, os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from difflib import SequenceMatcher

st.title("🎙️ AI-Native 발음 비교 분석기 (Stable)")

target_text = "The quick brown fox jumps over the lazy dog."
st.info(f"🎯 **Target:** {target_text}")

audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 녹음 완료", key="recorder")

if audio:
    # 1. 임시 파일로 저장 (BytesIO의 인식 문제를 피하기 위함)
    with open("temp_audio.wav", "wb") as f:
        f.write(audio['bytes'])
    
    try:
        # 2. librosa로 물리 파일 로드
        # sr=None으로 설정하여 원본 샘플링 레이트 유지
        y_learner, sr_rate = librosa.load("temp_audio.wav", sr=22050)
        duration_sec = len(y_learner) / sr_rate
        
        # 3. STT 분석
        r = sr.Recognizer()
        with sr.AudioFile("temp_audio.wav") as source:
            audio_data = r.record(source)
            transcript = r.recognize_google(audio_data, language='en-US')
            score = SequenceMatcher(None, target_text.lower(), transcript.lower()).ratio()
            st.metric("발음 정확도", f"{int(score * 100)}%")
            st.write(f"**AI 인식:** {transcript}")

        # 4. 시각화 섹션
        if st.button("📊 원어민 파형과 내 발음 대조하기"):
            # TTS 생성 및 임시 저장
            tts = gTTS(text=target_text, lang='en')
            tts.save("temp_tts.mp3")
            
            # TTS 로드 (librosa는 mp3 파일을 물리적으로 읽을 때 더 안정적입니다)
            y_native, _ = librosa.load("temp_tts.mp3", sr=sr_rate)
            
            # 시간 맞추기 (Padding)
            target_samples = len(y_learner)
            if len(y_native) < target_samples:
                padding = (target_samples - len(y_native)) // 2
                y_native = np.pad(y_native, (padding, target_samples - len(y_native) - padding), 'constant')
            else:
                y_native = y_native[:target_samples]

            # 그래프 출력
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
            librosa.display.waveshow(y_native, sr=sr_rate, ax=ax1, color='lightgray')
            ax1.set_title("Native Speaker (Standard)")
            librosa.display.waveshow(y_learner, sr=sr_rate, ax=ax2, color='skyblue')
            ax2.set_title("Your Pronunciation")
            st.pyplot(fig)
            
            # 임시 파일 삭제 (정리)
            os.remove("temp_tts.mp3")

    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {e}")
    finally:
        # 임시 파일 삭제
        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")
