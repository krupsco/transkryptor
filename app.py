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
            try:
                start = float(line.split("silence_start:")[1].strip())
            except:
                start = None
        if "silencedetect" in line and "silence_end" in line and "silence_duration" in line:
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
# LOGIKA DZIELENIA (1-ścieżka)
# =========================
def group_by_silence(total_dur: float,
                     silences: List[Tuple[float, float]],
                     max_chunk_s: int) -> List[Tuple[float, float]]:
    points = [0.0] + [end for (_, end) in silences]
    points = sorted(set([p for p in points if 0 <= p <= total_dur]))
    chunks = []
    start = 0.0

    def add_chunk(s, e):
        if e - s > 0.05:
            chunks.append((max(0.0, s), min(total_dur, e)))

    while start < total_dur:
        end_target = min(total_dur, start + max_chunk_s)
        cut_candidates = [p for p in points if start < p <= end_target]
        if cut_candidates:
            cut = cut_candidates[-1]
            add_chunk(start, cut)
            start = cut
        else:
            add_chunk(start, end_target)
            start = end_target

    merged = []
    for s, e in chunks:
        if not merged:
            merged.append([s, e])
            continue
        if abs(s - merged[-1][1]) < 0.05 and (e - merged[-1][0]) <= max_chunk_s + 1:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    return [(round(a, 3), round(b, 3)) for a, b in merged]

def split_fixed(total_dur: float, max_chunk_s: int) -> List[Tuple[float, float]]:
    chunks = []
    start = 0.0
    while start < total_dur:
        end = min(total_dur, start + max_chunk_s)
        chunks.append((round(start, 3), round(end, 3)))
        start = end
    return chunks

# =========================
# TRANSKRYPCJA (wspólne)
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

    st.markdown("**Długie pliki – dzielenie (1 ścieżka)**")
    use_silence = st.checkbox("Dzielenie po ciszy (bardziej naturalne)", value=True)
    max_chunk_s = st.slider("Maks. długość kawałka (s)", 300, 1200, 900, step=60)
    silence_thresh_db = st.slider("Próg ciszy (dBFS)", -60, -10, -36, step=1)
    min_silence_ms = st.slider("Min. długość ciszy (ms)", 300, 3000, 1200, step=100)

    st.markdown("---")
    rm_fill = st.checkbox("Usuń wypełniacze (eee/yyy/no/po prostu…)", value=True)
    creativity = st.slider("Kreatywność podsumowania", 0.0, 1.0, 0.2, 0.1)

st.write("Wgraj plik audio (MP3/WAV/M4A/AAC/MP4/OGG/WEBM), zrób transkrypcję po polsku, a potem wygeneruj skrót, wątki i cytaty.")

# =========================
# UI – TRYB 1 ŚCIEŻKA
# =========================
uploaded = st.file_uploader("Plik audio (1 ścieżka)", type=["mp3","wav","m4a","aac","mp4","ogg","webm"])

transcribed_text = st.session_state.get("transcribed_text", "")
clean_text = st.session_state.get("clean_text", "")
summary_md = st.session_state.get("summary_md", "")

col1, col2 = st.columns(2)
with col1:
    run_transcribe = st.button("🔁 Zrób transkrypcję", use_container_width=True, disabled=not uploaded)
with col2:
    run_summarize = st.button("✨ Stwórz skrót i cytaty", use_container_width=True, disabled=not bool(transcribed_text))

# =========================
# GŁÓWNA LOGIKA – 1 ŚCIEŻKA
# =========================
if run_transcribe and uploaded:
    try:
        with st.spinner("Analiza pliku…"):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(uploaded.name)[1], delete=False) as tmp:
                tmp.write(uploaded.read())
                src_path = tmp.name
            total = ffprobe_duration_seconds(src_path)

        try:
            with st.spinner("Transkrypcja (pojedynczy plik)…"):
                with open(src_path, "rb") as f:
                    b = f.read()
                text = transcribe_bytes(b, uploaded.name, language, model_transcribe)
        except Exception:
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

        text_clean = remove_fillers(text) if rm_fill else text
        st.session_state["transcribed_text"] = text
        st.session_state["clean_text"] = text_clean
        transcribed_text = text
        clean_text = text_clean
        st.success("Transkrypcja gotowa ✅")

    except Exception as e:
        st.error(f"Błąd transkrypcji: {e}")

# Podgląd (1 ścieżka)
if transcribed_text:
    st.subheader("📄 Transkrypcja (surowa)")
    st.text_area("Tekst", transcribed_text, height=280)

    st.subheader("🧹 Po czyszczeniu (opcjonalnie)")
    st.caption("Wersja z usuniętymi wypełniaczami i poprawioną interpunkcją.")
    st.text_area("Tekst (clean)", clean_text, height=280)

# Podsumowanie (wspólne)
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

# =========================================================
# ✨ NOWOŚĆ: TRYB WYWIADU (DWIE ŚCIEŻKI = DWOJE ROZMÓWCÓW)
# =========================================================
st.markdown("---")
st.header("🎤 Wywiad (2 ścieżki) – automatyczny podział na rozmówców")

colA, colB = st.columns(2)
with colA:
    file_A = st.file_uploader("Ścieżka A (rozmówca 1)", type=["mp3","wav","m4a","aac","mp4","ogg","webm"], key="uA")
with colB:
    file_B = st.file_uploader("Ścieżka B (rozmówca 2)", type=["mp3","wav","m4a","aac","mp4","ogg","webm"], key="uB")

name_A = st.text_input("Imię rozmówcy 1 (dla ścieżki A)", value="Rozmówca A")
name_B = st.text_input("Imię rozmówcy 2 (dla ścieżki B)", value="Rozmówca B")

st.caption("Wskazówka: to powinny być dwie równoległe ścieżki z tego samego nagrania (np. mikrofony lav A/B).")

# Ustawienia dla trybu 2-ścieżkowego
with st.expander("Ustawienia (2 ścieżki)"):
    chunk_win_s = st.slider("Długość okna transkrypcji (s)", 10, 120, 30, step=5,
                            help="Mniejsze okno = lepsze „przeplatanie” kwestii, większy koszt zapytań.")

run_dual = st.button("🔁 Zrób transkrypcję wywiadu (2 ścieżki)", use_container_width=True,
                     disabled=not (file_A and file_B))

def split_fixed_windows(total_dur: float, win_s: int) -> List[Tuple[float, float]]:
    out = []
    start = 0.0
    while start < total_dur:
        end = min(total_dur, start + win_s)
        out.append((round(start,3), round(end,3)))
        start = end
    return out

if run_dual and file_A and file_B:
    try:
        with st.spinner("Wczytywanie i weryfikacja długości…"):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file_A.name)[1], delete=False) as tmpA:
                tmpA.write(file_A.read())
                pathA = tmpA.name
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file_B.name)[1], delete=False) as tmpB:
                tmpB.write(file_B.read())
                pathB = tmpB.name

            durA = ffprobe_duration_seconds(pathA)
            durB = ffprobe_duration_seconds(pathB)
            tol = 2.0  # sekundy
            if abs(durA - durB) > tol:
                st.error(f"Długości ścieżek różnią się o więcej niż {tol}s: A={durA:.2f}s, B={durB:.2f}s. "
                         "Upewnij się, że to równoległe nagrania z tego samego wywiadu.")
                st.stop()

            total = min(durA, durB)

        # dzielimy na równe okna czasowe
        windows = split_fixed_windows(total, chunk_win_s)
        combined_lines: List[str] = []
        prog = st.progress(0.0, text="Transkrypcja okien…")

        for idx, (a, b) in enumerate(windows, start=1):
            # wytnij okno z A i B
            blobA = cut_to_memory(pathA, a, b, fmt="mp3")
            blobB = cut_to_memory(pathB, a, b, fmt="mp3")

            # transkrybuj każde niezależnie
            textA = transcribe_bytes(blobA, f"A_{int(a):06d}-{int(b):06d}.mp3", language, model_transcribe).strip()
            textB = transcribe_bytes(blobB, f"B_{int(a):06d}-{int(b):06d}.mp3", language, model_transcribe).strip()

            label = f"[{format_ts(int(a))}–{format_ts(int(b))}]"
            # heurystyka: wyświetl w kolejności – kto „mówi więcej” w tym oknie idzie pierwszy
            lenA, lenB = len(textA), len(textB)
            if lenA == 0 and lenB == 0:
                # nic się nie dzieje w tym oknie – pomiń
                pass
            elif lenA >= lenB:
                if lenA > 0:
                    combined_lines.append(f"{label} {name_A}: {textA}")
                if lenB > 0:
                    combined_lines.append(f"{label} {name_B}: {textB}")
            else:
                if lenB > 0:
                    combined_lines.append(f"{label} {name_B}: {textB}")
                if lenA > 0:
                    combined_lines.append(f"{label} {name_A}: {textA}")

            prog.progress(idx/len(windows), text=f"Okno {idx}/{len(windows)}")

        prog.empty()

        dialog_text = "\n\n".join(combined_lines)
        dialog_clean = remove_fillers(dialog_text) if rm_fill else dialog_text

        # pokaż i zapisz w tych samych polach (żeby działał przycisk „Stwórz skrót i cytaty”)
        st.session_state["transcribed_text"] = dialog_text
        st.session_state["clean_text"] = dialog_clean

        st.success("Transkrypcja wywiadu gotowa ✅ (2 ścieżki)")
        st.subheader("📄 Wywiad – transkrypcja (surowa)")
        st.text_area("Tekst", dialog_text, height=320)

        st.subheader("🧹 Wywiad – po czyszczeniu (opcjonalnie)")
        st.caption("Usunięte wypełniacze i korekta interpunkcji.")
        st.text_area("Tekst (clean)", dialog_clean, height=320)

    except Exception as e:
        st.error(f"Błąd transkrypcji wywiadu: {e}")

st.caption("Wskazówki: w trybie 2 ścieżek skracaj okno (np. 20–30 s), aby lepiej „przeplatać” kwestie rozmówców.")
