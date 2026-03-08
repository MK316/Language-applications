import streamlit as st
from streamlit_mic_recorder import mic_recorder
import io, librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

# --- [1] 디자인 설정 ---
st.set_page_config(page_title="Word Stress Master", layout="centered")

# --- [2] 피크 중심 분석 엔진 ---
def get_smoothed_envelope(y, sr, lowpass_hz=10):
    """미세한 떨림을 제거하고 부드러운 에너지 포형만 추출 (Low-pass Filter 효과)"""
    rms = librosa.feature.rms(y=y, hop_length=256)[0]
    # 더 넓은 윈도우로 이동평균을 적용하여 울퉁불퉁함을 제거
    smoothed = np.convolve(rms, np.ones(10)/10, mode='same')
    return smoothed

def analyze_peaks(env):
    """에너지 봉우리의 위치와 개수를 파악"""
    # 최소 높이와 거리를 설정하여 의미 있는 음절 피크만 추출
    peaks, _ = find_peaks(env, height=np.max(env)*0.15, distance=5)
    primary_stress_idx = np.argmax(env) if len(env) > 0 else 0
    return peaks, primary_stress_idx

def calculate_pedagogical_score(env_n, env_l):
    """언어 교육적 관점의 점수 산정 (피크 위치 및 패턴 중점)"""
    if len(env_n) < 2 or len(env_l) < 2: return 0
    
    # 1. 시간 정규화 (패턴 비교용)
    standard_x = np.linspace(0, 1, 100)
    f_n = interp1d(np.linspace(0, 1, len(env_n)), env_n, fill_value="extrapolate")
    f_l = interp1d(np.linspace(0, 1, len(env_l)), env_l, fill_value="extrapolate")
    n_norm, l_norm = f_n(standard_x), f_l(standard_x)
    
    # 2. 피크 분석
    peaks_n, stress_n = analyze_peaks(n_norm)
    peaks_l, stress_l = analyze_peaks(l_norm)
    
    # 3. 점수 가중치 계산
    # A. 강세 위치 일치도 (40%) - 빨간 선의 근접성
    stress_dist = abs(stress_n - stress_l) / 100
    stress_score = max(0, 1 - stress_dist) * 40
    
    # B. 전체적 흐름 일치도 (40%) - 상관계수 (부드러운 곡선 기반)
    pattern_corr = np.corrcoef(n_norm, l_norm)[0, 1]
    pattern_score = max(0, pattern_corr) * 40
    
    # C. 음절 수(봉우리 수) 일치도 (20%)
    peak_diff = abs(len(peaks_n) - len(peaks_l))
    peak_score = max(0, (1 - peak_diff/len(peaks_n))) * 20 if len(peaks_n) > 0 else 20
    
    total = int(stress_score + pattern_score + peak_score)
    return total, n_norm, l_norm, peaks_l

# --- [3] 메인 UI (앞선 코드의 구간 설정 부분 유지) ---
st.title("🎙️ Word Stress Master (Pedagogical View)")
word_db = {"Photograph": "photograph", "Photographer": "photographer", "Education": "education"}
target_word = word_db[st.selectbox("단어 선택:", list(word_db.keys()))]

if 'reset_key' not in st.session_state: st.session_state.reset_key = 0
audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 완료", key=f"rec_{st.session_state.reset_key}")

if audio:
    audio_bio = io.BytesIO(audio['bytes'])
    y_full, sr_f = librosa.load(audio_bio, sr=None)
    l_raw = AudioSegment.from_file(io.BytesIO(audio['bytes']))
    duration = len(y_full) / sr_f
    
    trim_range = st.slider("구간 선택:", 0.0, float(duration), (0.0, float(duration)), step=0.01)
    
    if st.button("📊 교육적 관점 분석 실행", type="primary"):
        y_l = y_full[int(trim_range[0]*sr_f):int(trim_range[1]*sr_f)]
        
        # 원어민 데이터 생성
        tts = gTTS(text=target_word, lang='en'); n_mp3 = io.BytesIO(); tts.write_to_fp(n_mp3); n_mp3.seek(0)
        y_n, sr_n = librosa.load(n_mp3, sr=None)
        
        # 부드러운 에너지 포형 추출 (울퉁불퉁함 제거)
        env_l = get_smoothed_envelope(librosa.util.normalize(y_l), sr_f)
        env_n = get_smoothed_envelope(librosa.util.normalize(y_n), sr_n)
        
        # [핵심] 교육용 점수 계산
        score, norm_n, norm_l, peaks_l = calculate_pedagogical_score(env_n, env_l)
        
        st.divider()
        st.metric("종합 리듬 점수", f"{score}점")
        st.info(f"학습자의 발음에서 총 {len(peaks_l)}개의 음절 패턴이 탐지되었습니다.")

        # 시각화 (피크 포인트 표시)
        fig_res, ax = plt.subplots(figsize=(10, 4))
        x = np.linspace(0, 100, 100)
        ax.fill_between(x, 0, norm_n, color='gray', alpha=0.15, label='Native Guide')
        ax.plot(x, norm_l, color='#ff4b4b', lw=2.5, label='My Rhythm (Smoothed)')
        
        # 피크 지점에 점 찍기
        ax.scatter(x[peaks_l], norm_l[peaks_l], color='red', s=50, zorder=5, label='Detected Syllables')
        ax.axvline(x=np.argmax(norm_l), color='red', lw=2, ls='--', label='My Stress Peak')
        ax.axvline(x=np.argmax(norm_n), color='gray', lw=2, ls=':', label='Native Stress Peak')
        
        ax.set_title("Rhythm Pattern Comparison (Syllable Based)")
        ax.legend(); st.pyplot(fig_res)

if st.button("🔄 리셋"):
    st.session_state.reset_key += 1
    st.rerun()
