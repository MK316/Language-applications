import streamlit as st
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import io
from difflib import SequenceMatcher

st.title("AI 발음 평가 도우미")

# 1. 목표 문장 설정 (수업 내용에 맞춰 변경 가능)
target_text = "The quick brown fox jumps over the lazy dog."
st.markdown(f"### 🎯 Target Sentence")
st.info(target_text)

# 2. 녹음 버튼 (브라우저 마이크 사용)
st.write("아래 버튼을 눌러 문장을 읽어주세요.")
audio = mic_recorder(
    start_prompt="🎤 녹음 시작",
    stop_prompt="🛑 녹음 완료",
    key="recorder"
)

if audio:
    # 3. 오디오 데이터 처리
    audio_bio = io.BytesIO(audio['bytes'])
    
    # 4. SpeechRecognition을 이용한 무료 STT (Google 엔진)
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_bio) as source:
            audio_data = r.record(source)
            # Google 무료 API 사용 (API 키 불필요)
            transcript = r.recognize_google(audio_data, language='en-US')
            
            # 5. 유사도 점수 계산
            score = SequenceMatcher(None, target_text.lower(), transcript.lower()).ratio()
            accuracy_pct = int(score * 100)

            # 6. 결과 시각화 (단어별 비교)
            st.divider()
            st.subheader(f"평가 결과: {accuracy_pct}점")
            
            # 단어별 하이라이트 로직
            target_words = target_text.lower().replace('.', '').split()
            recognized_words = transcript.lower().split()
            
            display_html = ""
            for word in recognized_words:
                if word in target_words:
                    display_html += f"<span style='color:green;'>{word}</span> "
                else:
                    display_html += f"<span style='color:red; text-decoration:underline;'>{word}</span> "
            
            st.markdown("---")
            st.write("**AI가 인식한 내용:**")
            st.markdown(display_html, unsafe_allow_html=True)
            st.caption("Tip: 초록색은 정확한 발음, 빨간색은 인식이 안 된 발음입니다.")

    except sr.UnknownValueError:
        st.error("죄송합니다. 목소리를 인식하지 못했습니다. 다시 시도해 주세요.")
    except sr.RequestError:
        st.error("인터넷 연결 문제로 서버에 접근할 수 없습니다.")
