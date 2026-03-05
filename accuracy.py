import streamlit as st
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import io
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from gtts import gTTS

# 1. TTS 생성 함수 (원어민 가이드 음성)
def generate_tts_wav(text, duration_sec):
    tts = gTTS(text=text, lang='en')
    tts_fp = io.BytesIO()
    tts.save(tts_fp)
    tts_fp.seek(0)
    
    # [수정] format="mp3"를 명시하고, pydub이 데이터 스트림을 잘 읽도록 설정
    try:
        tts_audio = AudioSegment.from_mp3(tts_fp)
    except:
        # mp3 직접 읽기 실패 시 일반 from_file 시도
        tts_fp.seek(0)
        tts_audio = AudioSegment.from_file(tts_fp, format="mp3")
    
    # 학습자 녹음 시간에 맞춰 중앙 정렬 (Padding)
    # 기존 코드와 동일...
    padding_duration = (duration_sec * 1000) - len(tts_audio)
    if padding_duration > 0:
        silence = AudioSegment.silent(duration=padding_duration / 2)
        tts_audio = silence + tts_audio + silence
    
    tts_audio = tts_audio[:int(duration_sec * 1000)]
    
    wav_io = io.BytesIO()
    tts_audio.export(wav_io, format="wav")
    wav_io.seek(0)
    return wav_io

# --- UI 부분 ---
st.title("🎙️ AI-Native 발음 비교 분석기")
target_text = "The quick brown fox jumps over the lazy dog."
st.info(f"🎯 **Target:** {target_text}")

audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 녹음 완료", key="recorder")

if audio:
    # 학습자 오디오 처리
    audio_bytes = audio['bytes']
    learner_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    duration_sec = len(learner_segment) / 1000.0 # 녹음 시간 계산
    
    # 학습자 WAV 변환
    learner_wav = io.BytesIO()
    learner_segment.export(learner_wav, format="wav")
    learner_wav.seek(0)

    # TTS 가이드 생성 (학습자 시간과 동일하게)
    tts_wav = generate_tts_wav(target_text, duration_sec)

    # 시각화 버튼
    if st.button("📊 원어민 파형과 내 발음 비교하기"):
        # 데이터 로드
        y_learner, sr_l = librosa.load(learner_wav)
        y_tts, sr_t = librosa.load(tts_wav)

        # 그래프 생성 (2행 1열)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        
        # 상단: 원어민 (TTS)
        librosa.display.waveshow(y_tts, sr=sr_t, ax=ax1, color='lightgray')
        ax1.set_title("Native Speaker (Guide)")
        ax1.set_ylabel("Amplitude")
        
        # 하단: 학습자
        librosa.display.waveshow(y_learner, sr=sr_l, ax=ax2, color='skyblue')
        ax2.set_title("Your Pronunciation")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")

        plt.tight_layout()
        st.pyplot(fig)

        st.success(f"총 {duration_sec:.2f}초 동안 녹음되었습니다. 원어민 파형의 '봉우리' 위치와 본인의 강세를 비교해 보세요!")
