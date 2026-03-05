import streamlit as st
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import io, os, librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from difflib import SequenceMatcher

# --- 유틸리티 함수 ---

def smooth_and_interpolate(f0, window_size=7):
    """피치 곡선의 노이즈를 제거하고 끊긴 구간(무성음)을 보간함"""
    # 1. 선형 보간 (무성음 구간 NaN 채우기)
    mask = np.isnan(f0)
    if np.any(~mask):
        f0[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), f0[~mask])
    
    # 2. 이동 평균 필터 (뾰족한 떨림 제거)
    if len(f0) < window_size: return f0
    return np.convolve(f0, np.ones(window_size)/window_size, mode='same')

def get_aligned_pitch(y, sr_rate, common_size=200):
    """피치를 추출하고 정해진 크기로 리샘플링하여 시간 축을 맞춤"""
    f0, voiced_flag, _ = librosa.pyin(y, fmin=70, fmax=400)
    f0[~voiced_flag] = np.nan  # 무성음 구간 NaN 처리
    
    # 부드럽게 만들기 및 보간
    f0_smooth = smooth_and_interpolate(f0)
    
    # 시간 축 강제 동기화 (0~100% 진행률로 리샘플링)
    resampled = np.interp(
        np.linspace(0, 1, common_size), 
        np.linspace(0, 1, len(f0_smooth)), 
        f0_smooth
    )
    return resampled

# --- 스트림릿 UI ---

st.set_page_config(page_title="AI 발음 분석기", layout="wide")
st.title("🎙️ AI-Native 발음 정밀 분석기 (Expert Edition)")

# 추천 문장 (유성음 중심)
target_text = "Online learning is normal now." 
st.info(f"🎯 **Target:** {target_text}")

audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 녹음 완료", key="recorder")

if audio:
    try:
        # 1. 학습자 오디오 처리 (무음 제거 포함)
        learner_segment = AudioSegment.from_file(io.BytesIO(audio['bytes']))
        learner_segment = learner_segment.strip_silence(silence_thresh=-40)
        learner_segment.export("temp_learner.wav", format="wav")
        
        # 2. 원어민 TTS 생성 (무음 제거 포함)
        tts = gTTS(text=target_text, lang='en')
        tts.save("temp_native.mp3")
        native_segment = AudioSegment.from_file("temp_native.mp3", format="mp3")
        native_segment = native_segment.strip_silence(silence_thresh=-40)
        native_segment.export("temp_native.wav", format="wav")

        # 분석용 데이터 로드
        y_learner, sr_l = librosa.load("temp_learner.wav", sr=22050)
        y_native, _ = librosa.load("temp_native.wav", sr=sr_l)

        # --- 탭 구성 ---
        tab1, tab2, tab3 = st.tabs(["🎯 인식 결과", "🔊 음파 대조", "📈 피치(억양) 분석"])

        with tab1:
            st.subheader("인식 결과 및 정확도")
            r = sr.Recognizer()
            with sr.AudioFile("temp_learner.wav") as source:
                audio_data = r.record(source)
                try:
                    transcript = r.recognize_google(audio_data, language='en-US')
                    score = SequenceMatcher(None, target_text.lower(), transcript.lower()).ratio()
                    st.metric("발음 정확도", f"{int(score * 100)}%")
                    st.success(f"**AI 인식 결과:** {transcript}")
                except:
                    st.error("인식에 실패했습니다. 더 명확하게 읽어주세요.")

        with tab2:
            st.subheader("소리 비교 및 파형 분석")
            c1, c2 = st.columns(2)
            with c1: st.audio("temp_learner.wav", format="audio/wav", start_time=0); st.caption("나의 발음")
            with c2: st.audio("temp_native.mp3"); st.caption("원어민 가이드")
            
            # 음파 정렬 시각화
            max_len = max(len(y_learner), len(y_native))
            y_l_pad = librosa.util.fix_length(y_learner, size=max_len)
            y_n_pad = librosa.util.fix_length(y_native, size=max_len)

            fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
            librosa.display.waveshow(y_n_pad, sr=sr_l, ax=ax1, color='lightgray')
            ax1.set_title("Native Speaker Waveform")
            librosa.display.waveshow(y_l_pad, sr=sr_l, ax=ax2, color='skyblue')
            ax2.set_title("Your Waveform")
            plt.tight_layout()
            st.pyplot(fig1)

        with tab3:
            st.subheader("억양(Intonation) 정밀 분석")
            c3, c4 = st.columns(2)
            with c3: st.audio("temp_learner.wav")
            with c4: st.audio("temp_native.mp3")

            # 피치 추출 및 시간 정렬 (0~100%)
            f0_l_aligned = get_aligned_pitch(y_learner, sr_l)
            f0_n_aligned = get_aligned_pitch(y_native, sr_l)

            # 시각화
            fig2, ax = plt.subplots(figsize=(12, 5))
            time_axis = np.linspace(0, 100, 200) # 진행률 표시

            ax.plot(time_axis, f0_n_aligned, label='Native', color='lightgray', linewidth=4, alpha=0.6)
            ax.plot(time_axis, f0_l_aligned, label='You', color='#1f77b4', linewidth=2.5)

            ax.set_title("Aligned Pitch Contour Comparison (%)")
            ax.set_xlabel("Sentence Progress (%)")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_ylim([50, 400])
            ax.legend()
            st.pyplot(fig2)
            
            st.info("💡 **Tip:** 이 문장은 유성음이 많아 곡선이 부드럽게 나타납니다. 원어민의 오르내림과 나의 곡선을 대조해보세요.")

    except Exception as e:
        st.error(f"분석 중 오류 발생: {e}")
    finally:
        # 파일 정리
        for f in ["temp_native.mp3", "temp_native.wav", "temp_learner.wav"]:
            if os.path.exists(f): os.remove(f)

# 사이드바 가이드
st.sidebar.markdown("""
### 🏛️ 교육적 설계 원리
1. **Difficulty (어려움)**: 점수를 통해 자신의 발음 한계를 인지합니다.
2. **Depth (깊이)**: 파형과 피치 곡선을 통해 음성학적 원인을 스스로 분석합니다.
3. **Strategic Engagement**: 시각적 피드백을 바탕으로 전략적으로 재도전합니다.
""")
