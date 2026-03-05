import streamlit as st
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import io
from pydub import AudioSegment  # 추가: 포맷 변환용
from difflib import SequenceMatcher

st.title("AI 발음 평가 도우미")

target_text = "The quick brown fox jumps over the lazy dog."
st.info(f"🎯 Target: {target_text}")

audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 녹음 완료", key="recorder")

if audio:
    # 1. 수신된 오디오 바이트 읽기
    audio_bytes = audio['bytes']
    
    # 2. pydub을 사용하여 WebM/Ogg를 WAV로 변환
    try:
        # 메모리 상의 바이트 데이터를 오디오 세그먼트로 로드
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        
        # WAV 형식으로 변환하여 메모리에 저장
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0) # 스트림 처음으로 이동

        # 3. SpeechRecognition 작업 수행
        r = sr.Recognizer()
        with sr.AudioFile(wav_io) as source:
            audio_data = r.record(source)
            transcript = r.recognize_google(audio_data, language='en-US')
            
            # 결과 출력 및 유사도 계산
            score = SequenceMatcher(None, target_text.lower(), transcript.lower()).ratio()
            st.subheader(f"평가 결과: {int(score * 100)}점")
            st.write(f"**AI 인식 결과:** {transcript}")

    except Exception as e:
        st.error(f"오디오 변환 중 오류가 발생했습니다: {e}")
