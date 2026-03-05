import streamlit as st
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import io
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment # 포맷 변환용
from difflib import SequenceMatcher

st.set_page_config(page_title="AI 발음 비교 분석기", layout="wide")
st.title("🎙️ AI-Native 발음 비교 분석기 (Final)")

target_text = "The quick brown fox jumps over the lazy dog."
st.info(f"🎯 **Target:** {target_text}")

audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 녹음 완료", key="recorder")

if audio:
    # [핵심] pydub을 사용하여 어떤 포맷이든 WAV로 먼저 변환합니다.
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio['bytes']))
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)
        
        # 이제 librosa는 표준 wav를 읽으므로 에러가 나지 않습니다.
        y_learner, sr_rate = librosa.load(wav_io, sr=22050)
        duration_sec = len(y_learner) / sr_rate
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📝 인식 결과")
            r = sr.Recognizer()
            wav_io.seek(0) # 스트림 리셋
            with sr.AudioFile(wav_io) as source:
                audio_data = r.record(source)
                transcript = r.recognize_google(audio_data, language='en-US')
                score = SequenceMatcher(None, target_text.lower(), transcript.lower()).ratio()
                st.metric("발음 정확도", f"{int(score * 100)}%")
                st.write(f"**AI 인식:** {transcript}")

        with col2:
            st.subheader("📊 시각화 도구")
            if st.button("원어민 파형과 내 발음 대조하기"):
                # TTS 생성 및 정렬 (기존 로직 유지)
                tts = gTTS(text=target_text, lang='en')
                tts_fp = io.BytesIO()
                tts.save(tts_fp)
                tts_fp.seek(0)
                
                # gTTS(mp3)도 pydub으로 wav 변환하여 librosa로 전달
                tts_segment = AudioSegment.from_file(tts_fp, format="mp3")
                tts_wav = io.BytesIO()
                tts_segment.export(tts_wav, format="wav")
                tts_wav.seek(0)
                
                y_native, _ = librosa.load(tts_wav, sr=sr_rate)
                
                # 시간 맞추기 (Padding)
                target_samples = len(y_learner)
                if len(y_native) < target_samples:
                    padding = (target_samples - len(y_native)) // 2
                    y_native = np.pad(y_native, (padding, target_samples - len(y_native) - padding), 'constant')
                else:
                    y_native = y_native[:target_samples]

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
                librosa.display.waveshow(y_native, sr=sr_rate, ax=ax1, color='lightgray')
                ax1.set_title("Native Speaker (Standard)")
                librosa.display.waveshow(y_learner, sr=sr_rate, ax=ax2, color='skyblue')
                ax2.set_title("Your Pronunciation")
                st.pyplot(fig)

    except Exception as e:
        st.error(f"오디오 처리 중 에러가 발생했습니다: {e}")
        st.write("Tip: 마이크 권한을 확인하거나 다시 녹음해보세요.")
