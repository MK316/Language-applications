import streamlit as st
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import io
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from difflib import SequenceMatcher

# 페이지 설정
st.set_page_config(page_title="AI 발음 분석기", layout="wide")

st.title("🎙️ AI 발음 분석 및 시각화 도우미")
st.write("문장을 읽고 자신의 발음을 데이터로 확인해보세요.")

# 1. 목표 설정
target_text = "The quick brown fox jumps over the lazy dog."
st.info(f"🎯 **Target:** {target_text}")

# 2. 녹음 섹션
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1단계: 녹음하기")
    audio = mic_recorder(
        start_prompt="🎤 녹음 시작",
        stop_prompt="🛑 녹음 완료",
        key="recorder"
    )

if audio:
    with col2:
        st.subheader("2단계: 분석 결과")
        
        # 오디오 변환 (WebM -> WAV)
        audio_bytes = audio['bytes']
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)

        # STT 엔진 가동
        r = sr.Recognizer()
        try:
            with sr.AudioFile(wav_io) as source:
                audio_data = r.record(source)
                # 신뢰도 확인을 위해 show_all=True 설정
                response = r.recognize_google(audio_data, language='en-US', show_all=True)
                
                if not response or 'alternative' not in response:
                    st.error("발음을 인식하지 못했습니다. 조금 더 크고 명확하게 말씀해주세요.")
                else:
                    # 가장 가능성 높은 결과 선택
                    best_alt = response['alternative'][0]
                    transcript = best_alt['transcript']
                    confidence = best_alt.get('confidence', 0) # 구글 무료 버전은 0으로 올 수 있음

                    # 유사도 점수
                    score = SequenceMatcher(None, target_text.lower(), transcript.lower()).ratio()
                    accuracy_pct = int(score * 100)

                    # 점수 표시
                    st.metric("발음 정확도", f"{accuracy_pct}%")

                    # 3. 단어별 하이라이트 (신뢰도 기반 시뮬레이션)
                    # 실제 신뢰도는 단어별로 뽑기 어려우므로 target 문장과 비교하여 시각화
                    target_words = target_text.lower().replace('.', '').split()
                    recognized_words = transcript.lower().split()

                    display_html = "<div>"
                    for word in recognized_words:
                        if word in target_words:
                            color = "#2ecc71" # 성공 (초록)
                        else:
                            color = "#e74c3c" # 실패 (빨강)
                        display_html += f"<span style='color:{color}; font-size:20px; font-weight:bold; margin-right:5px;'>{word}</span>"
                    display_html += "</div>"
                    
                    st.markdown("**AI 인식 텍스트:**", unsafe_allow_html=True)
                    st.markdown(display_html, unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")

    # 4. 음성학적 시각화 (Waveform)
    st.divider()
    st.subheader("3단계: 음성 파형(Waveform) 분석")
    
    # 파형 그래프 생성
    y, sr_rate = librosa.load(io.BytesIO(wav_io.getvalue()))
    
    fig, ax = plt.subplots(figsize=(12, 3))
    librosa.display.waveshow(y, sr=sr_rate, ax=ax, color='skyblue')
    ax.set_title("Your Voice Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)
    
    st.caption("💡 Tip: 파형의 진폭(높이)이 일정하고 단어 사이의 끊어읽기가 명확한지 확인해보세요.")

# 5. 음성학적 시각화 (버튼 클릭 시 노출)
    st.divider()
    st.subheader("3단계: 데이터 심층 분석")
    
    if st.button("📊 나의 발음 파형(Waveform) 분석하기"):
        with st.spinner("파형 데이터를 생성 중입니다..."):
            # 파형 그래프 생성 로직
            y, sr_rate = librosa.load(io.BytesIO(wav_io.getvalue()))
            
            fig, ax = plt.subplots(figsize=(12, 3))
            librosa.display.waveshow(y, sr=sr_rate, ax=ax, color='skyblue')
            ax.set_title("Your Voice Waveform Analysis")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            
            # 그래프 출력
            st.pyplot(fig)
            
            # 추가적인 음성학적 가이드
            st.info("""
            **💡 파형 읽는 법:**
            - **진폭(높이):** 목소리의 크기와 강세를 나타냅니다. 중요한 단어에서 파형이 커졌는지 확인하세요.
            - **간격:** 단어와 단어 사이의 휴지(Pause)를 나타냅니다. 너무 길거나 짧지 않은지 체크해보세요.
            """)
            
# 5. 하단 안내 (선생님의 교육 철학 반영)
st.sidebar.markdown("### 🏛️ 교육적 가이드")
st.sidebar.write("""
**점수가 낮아지는 주요 원인:**
1. **생략(Omission):** 단어를 너무 약하게 발음함.
2. **대체(Substitution):** 유사한 다른 발음으로 인식됨.
3. **배경 소음:** AI가 목소리와 소음을 구분하지 못함.
""")
