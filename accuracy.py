import streamlit as st
from streamlit_mic_recorder import mic_recorder
import speech_recognition as speech_rec
import io, os, librosa, librosa.display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from difflib import SequenceMatcher
import re

# --- 1. 유틸리티 함수 ---
def get_speech_bounds(audio_segment, silence_thresh=-40, min_silence_len=100, buffer_ms=100):
    nonsilent_intervals = detect_nonsilent(audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    if not nonsilent_intervals: return 0, len(audio_segment)
    start_trim = max(0, nonsilent_intervals[0][0] - buffer_ms)
    end_trim = min(len(audio_segment), nonsilent_intervals[-1][1] + 50)
    return start_trim, end_trim

# --- 2. 설정 및 세션 초기화 ---
st.set_page_config(page_title="AI 발음 분석기", layout="wide")
if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False

sample_sentences = {
    "Level 01: (인사/기초)": "I am on my way.",
    "Level 02: (일상/기초)": "Nice room you have.",
    "Level 03: (일상/기초)": "Dinner is ready now.",
}

st.markdown("### 🎙️ AI 활용 발음 연습")
selected_level = st.selectbox("Step 1: 학습 단계를 선택하세요:", list(sample_sentences.keys()))
target_text = sample_sentences.get(selected_level)

col_box = st.columns([1, 2, 1])[1]
with col_box:
    st.markdown(f"""<div style="border: 2px solid #1f77b4; border-radius: 12px; padding: 15px; background-color: #f8f9fb; text-align: center; margin-bottom: 20px;">
                <h3 style="color: #1f77b4; margin: 0; font-weight: 700;">"{target_text}"</h3></div>""", unsafe_allow_html=True)
    audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 녹음 완료", key="recorder")

# --- 3. 녹음 후 구간 설정 ---
if audio:
    st.divider()
    audio_bytes = audio['bytes']
    with open("temp_entry.wav", "wb") as f:
        f.write(audio_bytes)
    
    y_full, sr_f = librosa.load("temp_entry.wav", sr=22050)
    duration_sec = len(y_full) / sr_f
    
    st.info("✅ 녹음 완료! 아래 슬라이더로 분석할 구간을 선택하세요.")
    
    full_audio_seg = AudioSegment.from_file("temp_entry.wav")
    v_s_idx, v_e_idx = get_speech_bounds(full_audio_seg)
    trim_range = st.slider("구간 선택 (초):", 0.0, duration_sec, (float(v_s_idx/1000), float(v_e_idx/1000)), step=0.01)

    fig_prev, ax = plt.subplots(figsize=(12, 3))
    librosa.display.waveshow(y_full, sr=sr_f, ax=ax, color='skyblue', alpha=0.6)
    ax.axvline(x=trim_range[0], color='red', lw=2); ax.axvline(x=trim_range[1], color='red', lw=2)
    ax.set_xlim(max(0, trim_range[0] - 0.2), min(duration_sec, trim_range[1] + 0.2))
    st.pyplot(fig_prev)
    st.audio(audio_bytes)
        
    if st.button("📊 분석 시작하기", use_container_width=True):
        st.session_state.analysis_done = True
        st.session_state.final_audio_bytes = audio_bytes
        st.session_state.final_start = trim_range[0]
        st.session_state.final_end = trim_range[1]

# --- 4. 상세 분석 결과 ---
if st.session_state.analysis_done:
    p_l = p_n_m = p_n_w = p_stt = None
    with st.spinner("🚀 AI가 발음과 억양을 분석 중입니다..."):
        try:
            full_audio = AudioSegment.from_file(io.BytesIO(st.session_state.final_audio_bytes))
            s_ms, e_ms = st.session_state.final_start * 1000, st.session_state.final_end * 1000
            cropped = full_audio[s_ms:e_ms]
            l_s, l_e = get_speech_bounds(cropped, buffer_ms=50)
            final_l_seg = cropped[l_s:l_e]
            
            p_l, p_n_m, p_n_w, p_stt = "temp_l.wav", "temp_n.mp3", "temp_n.wav", "temp_stt.wav"
            final_l_seg.export(p_l, format="wav")
            full_audio.export(p_stt, format="wav")
            
            tts = gTTS(text=target_text, lang='en'); tts.save(p_n_m)
            n_raw = AudioSegment.from_file(p_n_m)
            n_s, n_e = get_speech_bounds(n_raw, silence_thresh=-35)
            final_n_seg = n_raw[n_s:n_e]; final_n_seg.export(p_n_w, format="wav")

            y_l, sr = librosa.load(p_l, sr=22050); y_n, _ = librosa.load(p_n_w, sr=sr)
            l_dur, n_dur = len(final_l_seg)/1000.0, len(final_n_seg)/1000.0

            st.success("🎉 분석 완료! 결과를 확인하세요.")

            ac1, ac2 = st.columns(2)
            with ac1: st.write("🎙️ **나의 발음**"); st.audio(p_l)
            with ac2: st.write("🔊 **원어민 발음**"); st.audio(p_n_w)

            tab1, tab2, tab3, tab4 = st.tabs(["🎯 AI 점수", "⏱️ 유창성", "🔊 음파 대조", "📈 피치 분석"])

            with tab1:
                recognizer = speech_rec.Recognizer()
                with speech_rec.AudioFile(p_stt) as source:
                    audio_data = recognizer.record(source)
                    try:
                        text_res = recognizer.recognize_google(audio_data, language='en-US')
                        clean = lambda s: re.sub(r'[^\w\s]', '', s).lower().strip()
                        sim = SequenceMatcher(None, clean(target_text), clean(text_res)).ratio()
                        acc = 100 if sim > 0.98 else int(sim * 100)
                        c1, c2 = st.columns([1, 2])
                        with c1: st.metric("정확도", f"{acc}점")
                        with c2: st.success(f"인식 결과: {text_res}")
                    except: st.error("인식 실패")

            with tab2:
                fig_dur, (axl, axn) = plt.subplots(2, 1, figsize=(12, 5))
                librosa.display.waveshow(y_l, sr=sr, ax=axl, color='skyblue')
                librosa.display.waveshow(y_n, sr=sr, ax=axn, color='lightgray')
                axl.set_title(f"Learner ({l_dur:.2f}s)"); axn.set_title(f"Native ({n_dur:.2f}s)")
                plt.tight_layout(); st.pyplot(fig_dur)
                st.info(f"💡 속도 차이: {int(((l_dur/n_dur)-1)*100)}%")

            with tab3:
                fig_w, (axw1, axw2) = plt.subplots(2, 1, figsize=(12, 6))
                librosa.display.waveshow(y_l, sr=sr, ax=axw1, color='skyblue')
                librosa.display.waveshow(y_n, sr=sr, ax=axw2, color='lightgray')
                plt.tight_layout(); st.pyplot(fig_w)

            with tab4:
                st.subheader("억양 멜로디 분석 (Pitch Contour)")
                f0_l, v_l, p_l_v = librosa.pyin(y_l, fmin=75, fmax=400, hop_length=64)
                f0_n, v_n, p_n_v = librosa.pyin(y_n, fmin=60, fmax=400, hop_length=64)
                f0_l_f = np.where(v_l & (p_l_v > 0.15), f0_l, np.nan)
                f0_n_f = np.where(v_n & (p_n_v > 0.01), f0_n, np.nan)
                
                # [복구] 점선 스타일(':')과 마커('o') 적용
                fig_p, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4), sharey=True)
                ax1.plot(librosa.times_like(f0_l, sr=sr, hop_length=64), f0_l_f, color='#1f77b4', linestyle=':', marker='o', markersize=2, label='Learner')
                ax2.plot(librosa.times_like(f0_n, sr=sr, hop_length=64), f0_n_f, color='gray', linestyle=':', marker='o', markersize=2, label='Native')
                
                ax1.set_title("Your Pitch (Dotted)"); ax2.set_title("Native Pitch (Dotted)")
                ax1.grid(axis='y', alpha=0.3); ax2.grid(axis='y', alpha=0.3)
                st.pyplot(fig_p)
                st.info("💡 점으로 표시된 피치 변화를 보며 원어민과 억양 높낮이를 비교해 보세요.")

        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")
        finally:
            for f in [p_l, p_n_m, p_n_w, p_stt, "temp_entry.wav"]:
                if isinstance(f, str) and os.path.exists(f):
                    os.remove(f)
