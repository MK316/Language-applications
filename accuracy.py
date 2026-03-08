import streamlit as st
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import io, os, librosa, librosa.display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from difflib import SequenceMatcher
from scipy.stats import pearsonr

# --- 유틸리티 함수 ---
def get_speech_bounds(audio_segment, silence_thresh=-40, min_silence_len=100, buffer_ms=100):
    nonsilent_intervals = detect_nonsilent(audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    if not nonsilent_intervals: return 0, len(audio_segment)
    start_trim = max(0, nonsilent_intervals[0][0] - buffer_ms)
    end_trim = min(len(audio_segment), nonsilent_intervals[-1][1] + 50)
    return start_trim, end_trim

def normalize_pitch(f0):
    """Z-score Normalization: 화자 간 피치 대역 차이를 제거"""
    mu = np.nanmean(f0)
    sigma = np.nanstd(f0)
    if sigma == 0 or np.isnan(sigma): return np.zeros_like(f0)
    return (f0 - mu) / sigma

def calculate_intonation_score(f0_n, f0_l):
    """정규화된 곡선 간의 피어슨 상관계수를 통한 점수 산출"""
    min_len = min(len(f0_n), len(f0_l))
    if min_len < 10: return 0
    
    vec_n = np.nan_to_num(f0_n[:min_len])
    vec_l = np.nan_to_num(f0_l[:min_len])
    
    # 두 벡터 간의 상관계수 계산
    with np.errstate(divide='ignore', invalid='ignore'):
        corr, _ = pearsonr(vec_n, vec_l)
    
    # 음수 상관계수는 0으로 처리, 양수 상관계수를 100점 만점으로 환산
    final_score = int(max(0, corr) * 100) if not np.isnan(corr) else 0
    return final_score

# --- 스트림릿 설정 및 세션 초기화 ---
st.set_page_config(page_title="AI 발음 분석기", layout="wide")
if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
if 'start_val' not in st.session_state: st.session_state.start_val = 0.0
if 'end_val' not in st.session_state: st.session_state.end_val = 1.0

sample_sentences = {
    "Level 01: (인사/기초)": "I am on my way.",
    "Level 02: (일상/기초)": "Nice room you have.",
    "Level 03: (일상/기초)": "Dinner is ready now.",
}

st.markdown("### 🎙️ AI 활용 발음 연습")

# [Step 1-2 로직 생략: 이전 대화와 동일하게 유지]
selected_level = st.selectbox("Step 1: 학습 단계를 선택하세요:", list(sample_sentences.keys()))
target_text = sample_sentences.get(selected_level, "I am on my way.")
col_box = st.columns([1, 2, 1])[1]
with col_box:
    st.markdown(f"""<div style="border: 2px solid #1f77b4; border-radius: 12px; padding: 15px; background-color: #f8f9fb; text-align: center; margin-bottom: 20px;"><h3 style="color: #1f77b4; margin: 0; font-weight: 700;">"{target_text}"</h3></div>""", unsafe_allow_html=True)
    rec_btn = st.columns([1, 1, 1])[1]
    with rec_btn:
        audio = mic_recorder(start_prompt="🎤 Step 2: 녹음 시작", stop_prompt="🛑 녹음 완료", key="recorder")

if audio:
    # [구간 조정 로직 생략: 이전 대화와 동일]
    st.divider()
    audio_bytes = audio['bytes']
    audio_stream = io.BytesIO(audio_bytes)
    full_audio = AudioSegment.from_file(audio_stream)
    full_audio.export("temp_preview.wav", format="wav")
    duration_sec = len(full_audio) / 1000.0
    y_full, sr_f = librosa.load("temp_preview.wav", sr=22050)
    
    if 'v_detected' not in st.session_state:
        v_s, v_e = get_speech_bounds(full_audio)
        st.session_state.start_val = float(v_s/1000); st.session_state.end_val = float(v_e/1000)
        st.session_state.zoom_range = (0.0, duration_sec); st.session_state.v_detected = True

    c_zoom, c_input = st.columns([1, 1])
    with c_zoom: st.slider("🔍 파형 확대 범위 (Zoom Window):", 0.0, duration_sec, key="zoom_range", step=0.01)
    with c_input:
        in_col1, in_col2 = st.columns(2)
        in_col1.number_input("시작 (sec):", 0.0, duration_sec, key="start_val", step=0.01, format="%.2f")
        in_col2.number_input("종료 (sec):", 0.0, duration_sec, key="end_val", step=0.01, format="%.2f")

    if st.button("📊 Step 3: 설정된 구간으로 분석하기", use_container_width=True):
        st.session_state.analysis_done = True
        st.session_state.final_audio_bytes = audio_bytes
        st.session_state.final_start = st.session_state.start_val
        st.session_state.final_end = st.session_state.end_val

if st.session_state.analysis_done:
    try:
        audio_stream = io.BytesIO(st.session_state.final_audio_bytes)
        full_audio = AudioSegment.from_file(audio_stream)
        s_ms, e_ms = st.session_state.final_start * 1000, st.session_state.final_end * 1000
        cropped_audio = full_audio[s_ms:e_ms]
        l_s, l_e = get_speech_bounds(cropped_audio, buffer_ms=50)
        final_learner = cropped_audio[l_s:l_e]
        final_learner.export("temp_learner.wav", format="wav")
        full_audio.export("temp_stt.wav", format="wav")
        
        tts = gTTS(text=target_text, lang='en'); tts.save("temp_native.mp3")
        native_raw = AudioSegment.from_file("temp_native.mp3", format="mp3")
        n_s, n_e = get_speech_bounds(native_raw, silence_thresh=-35, buffer_ms=0)
        final_native = native_raw[n_s:n_e]; final_native.export("temp_native.wav", format="wav")

        y_l, sr_l = librosa.load("temp_learner.wav", sr=22050); y_n, _ = librosa.load("temp_native.wav", sr=sr_l)

        # 공통 오디오 플레이어 (학습자 우선)
        st.divider()
        ac1, ac2 = st.columns(2)
        with ac1: st.write("🎙️ **나의 발음**"); st.audio("temp_learner.wav")
        with ac2: st.write("🔊 **원어민 발음**"); st.audio("temp_native.wav")

        tab1, tab2, tab3, tab4 = st.tabs(["🎯 AI 점수", "⏱️ 유창성 분석", "🔊 음파 대조", "📈 피치 분석"])

        # [Tab 1-3 생략: 이전 대화와 동일]

        with tab4:
            st.subheader("억양 및 멜로디 분석 (Pitch Contour)")
            # 피치 추출
            f0_l, v_l, p_l = librosa.pyin(y_l, fmin=75, fmax=400, hop_length=64)
            f0_n, v_n, p_n = librosa.pyin(y_n, fmin=60, fmax=400, hop_length=64)
            f0_l_f = np.where(v_l & (p_l > 0.15) & (f0_l > 80), f0_l, np.nan)
            f0_n_f = np.where(v_n & (p_n > 0.01), f0_n, np.nan)
            
            t_l = librosa.times_like(f0_l, sr=sr_l, hop_length=64); t_n = librosa.times_like(f0_n, sr=sr_l, hop_length=64)
            
            # 절대 피치 그래프 (기본 노출)
            fig_p, (ax_l1, ax_n1) = plt.subplots(1, 2, figsize=(15, 4), sharey=True)
            ax_l1.plot(t_l, f0_l_f, color='#1f77b4', linestyle=':', marker='o', markersize=2)
            ax_n1.plot(t_n, f0_n_f, color='lightgray', linestyle=':', marker='o', markersize=2)
            ax_l1.set_title("Your Absolute Pitch (Hz)"); ax_n1.set_title("Native Absolute Pitch (Hz)")
            st.pyplot(fig_p)

            # [수정] 정규화 및 피드백 실행 버튼
            st.markdown("---")
            if st.button("📈 정규화 분석 실행 (Melody Pattern Analysis)", use_container_width=True):
                # 정규화 수행
                fn_norm = normalize_pitch(f0_n_f)
                fl_norm = normalize_pitch(f0_l_f)
                
                # 유사도 점수 산출
                into_score = calculate_intonation_score(fn_norm, fl_norm)
                
                # 1. 오버레이 그래프 출력
                fig_nm, axn = plt.subplots(figsize=(12, 4))
                axn.plot(t_n[:len(fn_norm)], fn_norm, color='lightgray', linestyle=':', linewidth=3, label='Native', alpha=0.7)
                axn.plot(t_l[:len(fl_norm)], fl_norm, color='#1f77b4', linestyle=':', linewidth=3, label='You')
                axn.set_title("Intonation Pattern Comparison (Normalized)"); axn.legend(); st.pyplot(fig_nm)
                
                # 2. 결과 점수 및 전문가 피드백
                st.markdown(f"#### 📊 억양 유사도 점수: **{into_score}점**")
                
                if into_score >= 80:
                    st.success("🌟 **Excellent!** 원어민과 억양의 흐름이 매우 흡사합니다. 강조점(Nuclear Stress)과 문장 종결 어미 처리가 완벽합니다.")
                elif into_score >= 50:
                    st.info("👍 **Good.** 전체적인 리듬은 형성되어 있습니다. 다만, 그래프가 어긋나는 부분에서 강조를 더 주거나 빼는 연습을 해보세요.")
                else:
                    st.warning("🧐 **Needs Practice.** 억양이 다소 평평하거나 원어민과 반대되는 패턴이 감지됩니다. 곡선의 굴곡을 더 역동적으로 표현해 보세요.")
                
                with st.expander("📚 분석 원리 및 참고문헌"):
                    st.markdown("""
                    * **Z-score 정규화**: 화자 간의 신체적 차이(성대 길이 등)로 인한 절대적 주파수 차이를 제거하고 순수 멜로디 패턴만 대조합니다.                     * **피어슨 상관분석**: 시간 흐름에 따른 두 곡선의 수학적 상관관계(Correlation)를 통해 점수를 산출합니다.
                    * **참고 문헌**: Chun, D. M. (2002). *Discourse Intonation in L2*. 시각적 피드백이 학습자의 인토네이션 교정에 미치는 효과 연구.
                    """)
            else:
                st.caption("위 버튼을 누르면 화자 간 피치 차이를 보정한 정밀 패턴 분석 결과가 나타납니다.")

    except Exception as e: st.error(f"오류: {e}")
    finally:
        for f in ["temp_native.mp3", "temp_native.wav", "temp_learner.wav", "temp_stt.wav", "temp_preview.wav"]:
            if os.path.exists(f): os.remove(f)
