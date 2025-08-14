import os
import io
import json
import math
import shlex
import shutil
import subprocess
from typing import List, Tuple, Dict

import streamlit as st
from openai import OpenAI

# =========================
# KONFIG STRONY / API KEY
# =========================
st.set_page_config(page_title="Transkryptor – FFmpeg + OpenAI", layout="centered")
st.title("🎙️ Transkryptor (PL) – FFmpeg + OpenAI")

if "OPENAI_API_KEY" not in st.secrets:
    st.error("Brak klucza API. Dodaj OPENAI_API_KEY do .streamlit/secrets.toml (lub Streamlit Cloud → Secrets).")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# =========================
# NARZĘDZIA FFmpeg/FFprobe
# =========================
def check_binaries() -> Tuple[bool, str, str]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    return (bool(ffmpeg and ffprobe), ffmpeg or "?", ffprobe or "?")

def ffprobe_duration_seconds(path: str) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    return float(data["format"]["duration"])

def detect_silences(path: str, silence_thresh_db: int = -36, min_silence_ms: int = 1200) -> List[Tuple[float, float]]:
    """
    Zwraca listę wykrytych odcinków ciszy (start, end) w sekundach.
    Używa filtra 'silencedetect' z FFmpeg.
    """
    # Uwaga: 'silencedetect' wypisuje logi na STDERR.
    # Komenda bez tworzenia pliku wyjściowego:
    cmd = [
        "ffmpeg", "-i", path, "-af",
        f"silencedetect=noise={silence_thresh_db}dB:d={min_silence_ms/1000.0}",
        "-f", "null", "-"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    silences = []
    start = None
    for line in proc.stderr.splitlines():
        line = line.strip()
        if "silencedetect" in line and "silence_start" in line:
            # ... silence_start: 12.345
            try:
                start = float(line.split("silence_start:")[1].strip())
            except:
                start = None
        if "silencedetect" in line and "silence_end" in line and "silence_duration" in line:
            # ... silence_end: 23.456 | silence_duration: 11.111
            try:
                parts = line.split("silence_end:")[1].strip().split("|")[0].strip()
                end = float(parts)
                if start is not None:
                    silences.append((start, end))
                start = None
            except:
                pass
    return silences

def cut_to_memory(path: str, start_s: float, end_s: float, fmt: str = "mp3") -> bytes:
    """
    Wycinanie fragmentu audio do pamięci (BytesIO) przy użyciu ffmpeg.
    """
    duration = max(0.0, end_s - start_s)
    # Uwaga: kolejność -ss/-to/-t ma znaczenie; -ss przed -i = szybsze cięcie klatkowe
    cmd = [
        "ffmpeg",
        "-ss", f"{start_s:.3f}",
        "-i", path,
        "-t", f"{duration:.3f}",
        "-vn",
        "-acodec", "libmp3lame",
        "-b:a", "160k",
        "-f", fmt,
        "pipe:1"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.stdout

# =========================
# LOGIKA DZIELENIA
# =========================
def group_by_silence(total_dur: float,
                     silences: List[Tuple[float, float]],
                     max_chunk_s: int) -> List[Tuple[float, float]]:
    """
    Buduje listę (start, end) dla fragmentów max <= max_chunk_s,
    starając się rozcinać w punktach ciszy.
    """
    # Wstaw "sztuczne" cisze na początku i końcu, żeby uprościć logikę
    points = [0.0] + [end for (_, end) in silences]  # końce ciszy = dobre miejsca startu mowy
    points = sorted(set([p for p in points if 0 <= p <= total_dur]))
    chunks = []
    start = 0.0
    cursor = 0.0

    def add_chunk(s, e):
        if e - s > 0.05:
            chunks.append((max(0.0, s), min(total_dur, e)))

    # Idziemy po czasie i jeśli przekraczamy max_chunk_s,
    # szukamy najbliższego punktu "ciszy" do cięcia.
    while start < total_dur:
        end_target = min(total_dur, start + max_chunk_s)
        # znajdź najbliższy punkt <= end_target i > start
        cut_candidates = [p for p in points if start < p <= end_target]
        if cut_candidates:
            cut = cut_candidates[-1]
            add_chunk(start, cut)
            start = cut
        else:
            # nie ma dobrej ciszy w obrębie limitu — tnij "na sztywno"
            add_chunk(start, end_target)
            start = end_target

    # scal drobne fragmenty, jeżeli wyszły minimalne „ogonki”
    merged = []
    for s, e in chunks:
        if not merged:
            merged.append([s, e])
            continue
        # jeśli kolejny start równa się poprzedniemu end → sklej
        if abs(s - merged[-1][1]) < 0.05 and (e - merged[-1][0]) <= max_chunk_s + 1:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    return [(round(a, 3), round(b, 3)) for a, b in merged]

def split_fixed(total_dur: float, max_chunk_s: int) -> List[Tuple[float, float]]:
    """
    Proste cięcie co 'max_chunk_s' sekund.
    """
    chunks = []
    start = 0.0
    while start < total_dur:
        end = min(total_dur, start + max_chunk_s)
        chunks.append((round(start, 3), round(end, 3)))
        start = end
    return chunks

# =========================
# TRANSKRYPCJA
# =========================
def transcribe_bytes(blob: bytes, name: str, language: str, model: str) -> str:
    resp = client.audio.transcriptions.create(
        model=model,
        file=(name, blob),
        language=language
    )
    return getattr(resp, "text", str(resp)).strip()

def format_ts(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

def remove_fillers(text: str) -> str:
    import re
    fillers = [
        r"\b(e+|y+|ee+|yyy+)\b",
        r"\bno\b",
        r"\bwięc\b",
        r"\bw sensie\b",
        r"\bgeneralnie\b",
        r"\btak jakby\b",
        r"\bjakby\b",
        r"\bże tak powiem\b",
        r"\bpo prostu\b",
    ]
    cleaned = text
    for f in fillers:
        cleaned = re.sub(f, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
    return cleaned.strip()

def summarize_md(text: str, model: str, temperature: float = 0.2) -> str:
    system = (
        "Jesteś redaktorem podcastu (styl: dociekliwość, klarowność, naturalny polski). "
        "Na podstawie transkrypcji po polsku przygotuj:\n"
        "1) Krótkie podsumowanie (3–6 zdań),\n"
        "2) Listę głównych wątków (5–10 punktów),\n"
        "3) 6–10 cytatów do promocji (zwięzłe, naturalne, bez wypełniaczy). "
        "Nie wymyślaj treści — trzymaj się dosłownie tego, co jest w tekście."
    )
    chat = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"TRANSKRYPCJA:\n{text}"}
        ],
        temperature=temperature,
    )
    return chat.choices[0].message.content

# =========================
# SIDEBAR – USTAWIENIA
# =========================
ok, ffmpeg, ffprobe = check_binaries()
with st.sidebar:
    st.header("Ustawienia")
    st.write(f"FFmpeg: `{ffmpeg}`")
    st.write(f"FFprobe: `{ffprobe}`")
    if not ok:
        st.error("Nie znaleziono ffmpeg/ffprobe. Upewnij się, że w `packages.txt` jest `ffmpeg`.")

    model_transcribe = st.selectbox(
        "Model transkrypcji",
        ["gpt-4o-mini-transcribe", "gpt-4o-transcribe"],
        index=0
    )
    model_summarize = st.selectbox(
        "Model podsumowania",
        ["gpt-4o-mini", "gpt-4o"],
        index=0
    )
    language = st.text_input("Język nagrania (ISO)", value="pl")

    st.markdown("**Długie pliki – dzielenie**")
    use_silence = st.checkbox("Dzielenie po ciszy (bardziej naturalne)", value=True)
    max_chunk_s = st.slider("Maks. długość kawałka (s)", 300, 1200, 900, step=60)
    silence_thresh_db = st.slider("Próg ciszy (dBFS)", -60, -10, -36, step=1)
    min_silence_ms = st.slider("Min. długość ciszy (ms)", 300, 3000, 1200, step=100)

    st.markdown("---")
    rm_fill = st.checkbox("Usuń wypełniacze (eee/yyy/no/po prostu…)", value=True)
    creativity = st.slider("Kreatywność podsumowania", 0.0, 1.0, 0.2, 0.1)

st.write("Wgraj plik audio (MP3/WAV/M4A/AAC/MP4/OGG/WEBM), zrób transkrypcję po polsku, a potem wygeneruj skrót, wątki i cytaty.")

# =========================
# UI
# =========================
uploaded = st.file_uploader("Plik audio", type=["mp3","wav","m4a","aac","mp4","ogg","webm"])

transcribed_text = st.session_state.get("transcribed_text", "")
clean_text = st.session_state.get("clean_text", "")
summary_md = st.session_state.get("summary_md", "")

col1, col2 = st.columns(2)
with col1:
    run_transcribe = st.button("🔁 Zrób transkrypcję", use_container_width=True, disabled=not uploaded)
with col2:
    run_summarize = st.button("✨ Stwórz skrót i cytaty", use_container_width=True, disabled=not bool(transcribed_text))

# =========================
# GŁÓWNA LOGIKA
# =========================
if run_transcribe and uploaded:
    try:
        with st.spinner("Analiza pliku…"):
            # Zapisz upload do pliku tymczasowego (ffmpeg/ffprobe potrzebują ścieżki)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(uploaded.name)[1], delete=False) as tmp:
                tmp.write(uploaded.read())
                src_path = tmp.name

            total = ffprobe_duration_seconds(src_path)

        # 1) Spróbuj „na raz” (przy krótkich plikach)
        try:
            with st.spinner("Transkrypcja (pojedynczy plik)…"):
                with open(src_path, "rb") as f:
                    b = f.read()
                text = transcribe_bytes(b, uploaded.name, language, model_transcribe)
        except Exception as e_single:
            # 2) Dzielenie (po ciszy lub co X sekund)
            with st.spinner("Transkrypcja kawałkami…"):
                if use_silence:
                    silences = detect_silences(src_path, silence_thresh_db, min_silence_ms)
                    chunks = group_by_silence(total, silences, max_chunk_s)
                else:
                    chunks = split_fixed(total, max_chunk_s)

                combined = []
                prog = st.progress(0.0, text="Postęp…")
                for idx, (a, b) in enumerate(chunks, start=1):
                    blob = cut_to_memory(src_path, a, b, fmt="mp3")
                    name = f"chunk_{int(a):06d}-{int(b):06d}.mp3"
                    t = transcribe_bytes(blob, name, language, model_transcribe)
                    label = f"[{format_ts(int(a))}–{format_ts(int(b))}]"
                    combined.append(f"{label} {t}")
                    prog.progress(idx / len(chunks), text=f"Kawałek {idx}/{len(chunks)}")

                prog.empty()
                text = "\n\n".join(combined)

        # opcjonalne czyszczenie
        text_clean = remove_fillers(text) if rm_fill else text

        st.session_state["transcribed_text"] = text
        st.session_state["clean_text"] = text_clean
        transcribed_text = text
        clean_text = text_clean
        st.success("Transkrypcja gotowa ✅")

    except Exception as e:
        st.error(f"Błąd transkrypcji: {e}")

# Podgląd
if transcribed_text:
    st.subheader("📄 Transkrypcja (surowa)")
    st.text_area("Tekst", transcribed_text, height=280)

    st.subheader("🧹 Po czyszczeniu (opcjonalnie)")
    st.caption("Wersja z usuniętymi wypełniaczami i poprawioną interpunkcją.")
    st.text_area("Tekst (clean)", clean_text, height=280)

# Podsumowanie
if run_summarize and transcribed_text:
    try:
        with st.spinner("Generuję skrót i cytaty…"):
            out = summarize_md(clean_text if rm_fill else transcribed_text, model_summarize, creativity)
            st.session_state["summary_md"] = out
            summary_md = out
        st.success("Gotowe ✅")
    except Exception as e:
        st.error(f"Błąd generowania podsumowania: {e}")

if summary_md:
    st.subheader("🧭 Skrót, wątki i cytaty (Markdown)")
    st.markdown(summary_md)
    st.download_button("⬇️ Pobierz wynik (MD)", data=summary_md, file_name="transkryptor_wynik.md")

st.caption("Wskazówki: dla trudnych nagrań zwiększ max długość kawałka lub obniż próg ciszy (np. −40 dBFS).")
