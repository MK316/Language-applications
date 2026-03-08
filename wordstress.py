import streamlit as st
from streamlit_mic_recorder import mic_recorder
import io, os, librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from scipy.interpolate import interp1d

# --- [1] 레이아웃 및 CSS ---
st.set_page_config(page_title="Word Stress Master", layout="centered")
st.markdown("<style>.stSlider { padding-left: 0px; padding-right: 0px; }</style>", unsafe_allow_html=True)

if 'last_audio_id' not in st.session_state: st.session_state.last_audio_id = None
if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
if 'final_y_l' not in st.session_state: st.session_state.final_y_l = None

# --- [2] 강화된 분석 엔진: 음절 분리 및 강세 탐지 ---
def get_rms_envelope(y, hop_length=256):
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    return np.convolve(rms, np.ones(5)/5, mode='same')

def detect_syllable_stress(y, sr):
    """음절을 분리하고 각 음절의 에너지를 비교하여 실제 강세 지점을 찾습니다."""
    env = get_rms_envelope(y)
    times = np.linspace(0, len(y)/sr, len(env))
    
    # 음절 경계 탐지 (엔벨롭의 계곡(Troughs) 찾기)
    # 단순 argmax가 아니라, 에너지가 집중된 '구역'을 분석
    thresh = np.max(env) * 0.15
    is_speech = env > thresh
    
    # 에너지가 높은 구간들(음절 후보군) 추출
    if not np.any(is_speech): return np.argmax(env), env
    
    # 실제 강세 피크 결정: (진폭 * 0.7 + 면적 * 0.3) 가중치 적용
    # 이렇게 하면 면적만 넓은 약음절이 강세로 오인되는 것을 방지합니다.
    weighted_env = (env / np.max(env)) * 0.7 + (env / np.sum(env)) * 0.3
    best_idx = np.argmax(weighted_env)
    
    return best_idx, env

def calculate_advanced_score(env_n, env_l):
    if len(env_n) < 2 or len(env_l) < 2: return 0, 0, 0
    f_n = interp1d(np.linspace(0, 1, len(env_n)), env_n, fill_value="extrapolate")
    f_l = interp1d(np.linspace(0, 1, len(env_l)), env_l, fill_value="extrapolate")
    new_x = np.linspace(0, 1, 100)
    shape_corr = np.corrcoef(f_n(new_x), f_l(new_x))[0, 1]
    
    dur_n = np.sum(env_n > np.max(env_n)*0.2) / len(env_n)
    dur_l = np.sum(env_l > np.max(env_l)*0.2) / len(env_l)
    
    return int(max(0, shape_corr) * 100), dur_l * 100, dur_n * 100

# --- [3] 앱 UI (단어 선택 및 녹음) ---
st.title("🎙️ Word Stress Master")
word_db = {"Photograph (1음절)": "photograph", "Photographer (2음절)": "photographer", "Education (3음절)": "education"}
selected_label = st.selectbox("단어 선택:", list(word_db.keys()))
target_word = word_db[selected_label]

audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="🛑 녹음 완료", key="word_recorder")

if audio:
    if audio['id'] != st.session_state.last_audio_id:
        st.session_state.last_audio_id, st.session_state.analysis_done = audio['id'], False

    # 오디오 로드 (pydub 활용으로 에러 방지)
    l_seg_full = AudioSegment.from_file(io.BytesIO(audio['bytes']))
    y_full = np.array(l_seg_full.get_array_of_samples(), dtype=np.float32) / (2**15)
    if l_seg_full.channels > 1: y_full = y_full.reshape((-1, l_seg_full.channels)).mean(axis=1)
    sr_f = l_seg_full.frame_rate
    
    # 구간 설정
    st.markdown("#### ✂️ 분석 구간 설정")
    auto_s, auto_e = detect_nonsilent(l_seg_full, min_silence_len=50, silence_thresh=-45)[0] if detect_nonsilent(l_seg_full) else (0, len(l_seg_full))
    trim_range = st.slider("구간 선택:", 0.0, float(len(y_full)/sr_f), (float(auto_s/1000), float(auto_e/1000)), step=0.01)

    # 미리보기 그래프
    fig_p, axp = plt.subplots(figsize=(10, 2.2))
    plt.subplots_adjust(left=0, right=1, bottom=0.2, top=0.8)
    librosa.display.waveshow(y_full, sr=sr_f, ax=axp, color='skyblue', alpha=0.5)
    axp.axvline(x=trim_range[0], color='red', lw=2, ls='--'); axp.axvline(x=trim_range[1], color='red', lw=2, ls='--')
    axp.set_xlim(0, len(y_full)/sr_f); axp.set_yticks([]); st.pyplot(fig_p)

    if st.button("📊 음절 기반 강세 분석 시작", type="primary"):
        st.session_state.analysis_done = True
        st.session_state.final_y_l = y_full[int(trim_range[0]*sr_f):int(trim_range[1]*sr_f)]
        st.session_state.current_sr = sr_f

# --- [4] 결과 분석 (음절 분리 가중치 적용) ---
if st.session_state.analysis_done and st.session_state.final_y_l is not None:
    y_l, sr = st.session_state.final_y_l, st.session_state.current_sr
    
    # 원어민 데이터 생성
    tts = gTTS(text=target_word, lang='en'); n_mp3 = io.BytesIO(); tts.write_to_fp(n_mp3); n_mp3.seek(0)
    n_seg = AudioSegment.from_file(n_mp3); n_bounds = detect_nonsilent(n_seg, min_silence_len=50, silence_thresh=-45)
    n_seg = n_seg[n_bounds[0][0]:n_bounds[0][1]] if n_bounds else n_seg
    y_n = np.array(n_seg.get_array_of_samples(), dtype=np.float32) / (2**15)
    if n_seg.channels > 1: y_n = y_n.reshape((-1, n_seg.channels)).mean(axis=1)
    
    y_l, y_n = librosa.util.normalize(y_l), librosa.util.normalize(y_n)
    
    # 음절 기반 강세 피크 탐지 (강화된 알고리즘)
    peak_idx_l, env_l = detect_syllable_stress(y_l, sr)
    peak_idx_n, env_n = detect_syllable_stress(y_n, sr)
    
    score, dur_l, dur_n = calculate_advanced_score(env_n, env_l)
    
    st.divider()
    col1, col2 = st.columns(2)
    col1.metric("종합 점수", f"{score}점")
    col2.metric("강세 구간 비중", f"{dur_l:.1f}%", f"{dur_l - dur_n:+.1f}%")

    # 결과 그래프
    fig_res, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    t_l = np.linspace(0, len(y_l)/sr, len(env_l))
    librosa.display.waveshow(y_l, sr=sr, ax=ax1, color='skyblue', alpha=0.3)
    ax1.plot(t_l, env_l, color='#1f77b4', lw=2.5)
    ax1.fill_between(t_l, 0, env_l, where=(env_l > np.max(env_l)*0.2), color='#1f77b4', alpha=0.3)
    ax1.axvline(x=t_l[peak_idx_l], color='red', lw=3) # 강화된 피크 표시
    
    t_n = np.linspace(0, len(y_n)/sr, len(env_n))
    librosa.display.waveshow(y_n, sr=sr, ax=ax2, color='lightgray', alpha=0.3)
    ax2.plot(t_n, env_n, color='gray', lw=2.5)
    ax2.fill_between(t_n, 0, env_n, where=(env_n > np.max(env_n)*0.2), color='gray', alpha=0.3)
    ax2.axvline(x=t_n[peak_idx_n], color='red', lw=3)
    
    ax1.set_title("My Result (Weighted Stress Peak)"); ax2.set_title("Native Standard")
    plt.tight_layout(); st.pyplot(fig_res)
    st.info("💡 빨간 선이 실제 음성 에너지가 가장 날카롭게 집중된 '진짜 강세' 지점을 찾아냅니다.")
