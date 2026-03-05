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

st.title("🎙️ AI-Native 발음 분석기 (Final Stable)")

target_text = "The quick brown fox jumps over the lazy dog."
st.info(f"🎯 **Target:** {target_text}")

audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 녹음 완료", key="recorder")

if audio:
    # 1. pydub을 사용하여 어떤 포맷이든 WAV로 강제 변환
    try:
        # 브라우저의 webm/ogg 데이터를 읽어서 WAV로 변환
        audio_segment = AudioSegment.from_file(io.BytesIO(audio['bytes']))
        audio_segment.export("temp_learner.wav", format="wav")
        
        # 2. STT 분석 (WAV 파일로 수행)
        r = sr.Recognizer()
        with sr.AudioFile("temp_learner.wav") as source:
            audio_data = r.record(source)
            # Google STT는 WAV 포맷을 받아야 에러가 나지 않습니다.
            transcript = r.recognize_google(audio_data, language='en-US')
            score = SequenceMatcher(None, target_text.lower(), transcript.lower()).ratio()
            st.metric("발음 정확도", f"{int(score * 100)}%")
            st.write(f"**AI 인식:** {transcript}")

        # 3. 시각화 섹션 (버튼 클릭 시)
        if st.button("📊 원어민 파형과 내 발음 대조하기"):
            with st.spinner("원어민 음성 생성 및 동기화 중..."):
                # TTS 생성 및 WAV 변환
                tts = gTTS(text=target_text, lang='en')
                tts.save("temp_native.mp3")
                native_segment = AudioSegment.from_file("temp_native.mp3", format="mp3")
                native_segment.export("temp_native.wav", format="wav")
                
                # librosa로 두 WAV 파일 로드
                y_learner, sr_l = librosa.load("temp_learner.wav", sr=22050)
                y_native, _ = librosa.load("temp_native.wav", sr=sr_l)
                
                # 시간 축 맞추기 (Padding)
                target_samples = len(y_learner)
                if len(y_native) < target_samples:
                    padding = (target_samples - len(y_native)) // 2
                    y_native = np.pad(y_native, (padding, target_samples - len(y_native) - padding), 'constant')
                else:
                    y_native = y_native[:target_samples]

                # 그래프 출력
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
                librosa.display.waveshow(y_native, sr=sr_l, ax=ax1, color='lightgray')
                ax1.set_title("Native Speaker (Standard)")
                librosa.display.waveshow(y_learner, sr=sr_l, ax=ax2, color='skyblue')
                ax2.set_title("Your Pronunciation")
                st.pyplot(fig)
                
                # 임시 파일 정리
                os.remove("temp_native.mp3")
                os.remove("temp_native.wav")

    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {e}")
        st.info("Tip: FFmpeg 설치 여부와 파일 형식을 확인해주세요.")
    finally:
        # 학습자 임시 파일 정리
        if os.path.exists("temp_learner.wav"):
            os.remove("temp_learner.wav")
