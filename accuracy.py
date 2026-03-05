import streamlit as st
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import io
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from difflib import SequenceMatcher

st.title("🎙️ AI-Native 발음 비교 분석기")

target_text = "The quick brown fox jumps over the lazy dog."
st.info(f"🎯 **Target:** {target_text}")

audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 녹음 완료", key="recorder")

if audio:
    audio_bytes = audio['bytes']
    
    # 1. 학습자 오디오 로드 (librosa는 webm/ogg를 직접 읽으려 시도함)
    # 만약 여기서 에러가 나면 io.BytesIO(audio_bytes)를 사용
    try:
        y_learner, sr_rate = librosa.load(io.BytesIO(audio_bytes), sr=22050)
        duration_sec = len(y_learner) / sr_rate
        
        # 2. STT 분석 (Google 무료 엔진)
        r = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = r.record(source)
            transcript = r.recognize_google(audio_data, language='en-US')
            score = SequenceMatcher(None, target_text.lower(), transcript.lower()).ratio()
            st.metric("발음 정확도", f"{int(score * 100)}%")
            st.write(f"**AI 인식:** {transcript}")

        # 3. 시각화 섹션
        if st.button("📊 원어민 파형과 내 발음 대조하기"):
            # TTS 생성
            tts = gTTS(text=target_text, lang='en')
            tts_fp = io.BytesIO()
            tts.save(tts_fp)
            tts_fp.seek(0)
            
            # [핵심] pydub 없이 librosa로 MP3 직접 로드
            y_native, _ = librosa.load(tts_fp, sr=sr_rate)
            
            # 시간 맞추기 (중앙 정렬)
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

    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다. (오디오 형식 문제): {e}")
