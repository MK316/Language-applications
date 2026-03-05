import streamlit as st
from streamlit_mic_recorder import mic_recorder
from openai import OpenAI
from difflib import SequenceMatcher

# 1. 목표 문장 설정
target_text = "The quick brown fox jumps over the lazy dog."
st.write(f"**Target:** {target_text}")

# 2. 녹음 버튼
audio = mic_recorder(start_prompt="Click to Record", stop_prompt="Stop Recording")

if audio:
    # 3. Whisper API를 이용한 음성 인식
    client = OpenAI(api_key="YOUR_API_KEY")
    transcript = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio['bytes']
    ).text
    
    # 4. 유사도 점수 계산
    score = SequenceMatcher(None, target_text.lower(), transcript.lower()).ratio()
    
    # 5. 결과 출력
    st.subheader(f"Your Score: {int(score * 100)}%")
    st.write(f"AI recognized: {transcript}")
