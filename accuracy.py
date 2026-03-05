import streamlit as st
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import io, os, librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from difflib import SequenceMatcher

# 이동 평균 필터 함수 (뾰족한 변화 제거용)
def smooth_curve(data, window_size=5):
    if len(data) < window_size: return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

st.set_page_config(page_title="AI 발음 분석기", layout="wide")
st.title("🎙️ AI-Native 발음 정밀 분석기 (Alignment optimized)")

target_text = "The quick brown fox jumps over the lazy dog."
st.info(f"🎯 **Target:** {target_text}")

audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 녹음 완료", key="recorder")

if audio:
    try:
        # 파일 변환 및 저장
        learner_segment = AudioSegment.from_file(io.BytesIO(audio['bytes']))
        # 앞뒤 무음 제거 (시간 정렬을 위한 핵심 단계)
        learner_segment = learner_segment.strip_silence(silence_thresh=-40)
        learner_segment.export("temp_learner.wav", format="wav")
        
        tts = gTTS(text=target_text, lang='en')
        tts.save("temp_native.mp3")
        native_segment = AudioSegment.from_file("temp_native.mp3", format="mp3")
        native_segment = native_segment.strip_silence(silence_thresh=-40) # 무음 제거
        native_segment.export("temp_native.wav", format="wav")

        y_learner, sr_l = librosa.load("temp_learner.wav", sr=22050)
        y_native, _ = librosa.load("temp_native.wav", sr=sr_l)

        tab1, tab2, tab3 = st.tabs(["🎯 인식 결과", "🔊 음파 비교", "📈 피치(억양) 분석"])

        # --- Tab 1 & 2는 기존과 동일하게 유지 ---
        with tab1:
            r = sr.Recognizer()
            with sr.AudioFile("temp_learner.wav") as source:
                audio_data = r.record(source)
                try:
                    transcript = r.recognize_google(audio_data, language='en-US')
                    score = SequenceMatcher(None, target_text.lower(), transcript.lower()).ratio()
                    st.metric("정확도", f"{int(score * 100)}%")
                    st.success(f"**인식된 문장:** {transcript}")
                except: st.error("인식 실패")

        with tab2:
            st.audio("temp_learner.wav"); st.audio("temp_native.mp3")
            max_len = max(len(y_learner), len(y_native))
            y_l_pad = librosa.util.fix_length(y_learner, size=max_len)
            y_n_pad = librosa.util.fix_length(y_native, size=max_len)
            fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
            librosa.display.waveshow(y_n_pad, sr=sr_l, ax=ax1, color='lightgray')
            librosa.display.waveshow(y_l_pad, sr=sr_l, ax=ax2, color='skyblue')
            st.pyplot(fig1)

        # --- Tab 3: 피치 분석 최적화 (Alignment & Smoothing) ---
        with tab3:
            st.audio("temp_learner.wav"); st.audio("temp_native.mp3")
            
            # 피치 추출
            f0_l, v_flag_l, v_prob_l = librosa.pyin(y_learner, fmin=70, fmax=400)
            f0_n, v_flag_n, v_prob_n = librosa.pyin(y_native, fmin=70, fmax=400)

            # 유성음 구간만 남기고 나머지는 NaN 처리
            f0_l[~v_flag_l] = np.nan
            f0_n[~v_flag_n] = np.nan

            # 곡선 부드럽게 만들기 (Moving Average)
            f0_l_smooth = smooth_curve(f0_l, window_size=7)
            f0_n_smooth = smooth_curve(f0_n, window_size=7)

            # 시간 축 강제 동기화 (두 데이터를 동일한 타임스탬프 수로 리샘플링)
            common_size = 200
            f0_l_resampled = np.interp(np.linspace(0, 1, common_size), np.linspace(0, 1, len(f0_l_smooth)), f0_l_smooth)
            f0_n_resampled = np.interp(np.linspace(0, 1, common_size), np.linspace(0, 1, len(f0_n_smooth)), f0_n_smooth)

            fig2, ax = plt.subplots(figsize=(12, 5))
            time_axis = np.linspace(0, 100, common_size) # 0~100% 진행률로 표시

            # 원어민 (배경선)
            ax.plot(time_axis, f0_n_resampled, label='Native', color='lightgray', linewidth=4, alpha=0.6)
            # 학습자 (강조선)
            ax.plot(time_axis, f0_l_resampled, label='You', color='#1f77b4', linewidth=2.5)

            ax.set_title("Aligned Intonation Curve (%)")
            ax.set_xlabel("Sentence Progress (%)")
            ax.set_ylabel("Pitch (Hz)")
            ax.legend()
            st.pyplot(fig2)
            
            st.info("💡 **최적화 완료:** 앞뒤 무음을 제거하고 문장 진행률(0~100%)에 맞춰 정렬했습니다. 전반적인 곡선 흐름을 비교해 보세요.")

    except Exception as e: st.error(f"오류: {e}")
    finally:
        for f in ["temp_native.mp3", "temp_native.wav"]:
            if os.path.exists(f): os.remove(f)
