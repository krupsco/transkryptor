import io
import re
import math
import tempfile
from typing import List, Dict

import streamlit as st
from openai import OpenAI

from pydub import AudioSegment
from pydub.silence import split_on_silence

# =========================
# KONFIGURACJA STRONY
# =========================
st.set_page_config(page_title="Transkryptor – Streamlit + OpenAI", layout="centered")
st.title("🎙️ Transkryptor (PL) – Streamlit + OpenAI")

# Klucz API z Secrets
if "OPENAI_API_KEY" not in st.secrets:
    st.error("Brak klucza API. Dodaj OPENAI_API_KEY do .streamlit/secrets.toml lub w Streamlit Cloud → Secrets.")
    st.stop()

# Klient OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# =========================
# USTAWIENIA (SIDEBAR)
# =========================
with st.sidebar:
    st.header("Ustawienia")
    model_transcribe = st.selectbox(
        "Model transkrypcji",
        ["gpt-4o-mini-transcribe", "gpt-4o-transcribe"],
        index=0,
        help="mini = szybszy/tańszy; transcribe = dokładniejszy"
    )
    model_summarize = st.selectbox(
        "Model podsumowania",
        ["gpt-4o-mini", "gpt-4o"],
        index=0
    )
    language = st.text_input("Język nagrania (ISO)", value="pl", help="Np. pl, en, de…")

    st.markdown("**Długie pliki – cięcie na kawałki**")
    use_silence_split = st.checkbox("Dzielenie po ciszy (bardziej naturalne)", value=True)
    chunk_seconds = st.slider("Max długość kawałka (s)", 300, 1200, 900, step=60,
                              help="Górny limit długości pojedynczego żądania do API.")
    if use_silence_split:
        min_silence_len = st.slider("Min. długość ciszy (ms)", 300, 3000, 1200, step=100,
                                    help="Cisza dłuższa od tej wartości może być miejscem cięcia.")
        silence_thresh_db = st.slider("Próg ciszy (dBFS)", -60, -10, -36, step=1,
                                      help="Im bliżej 0, tym mniej wrażliwy na ciszę.")
        keep_silence_ms = st.slider("Zachowaj ciszę na brzegach (ms)", 0, 1500, 250, step=50)
    else:
        min_silence_len, silence_thresh_db, keep_silence_ms = 1200, -36, 250

    st.markdown("---")
    rm_fillers = st.checkbox("Usuń wypełniacze (eee, yyy, no więc…)", value=True)
    temperature = st.slider("Kreatywność podsumowania", 0.0, 1.0, 0.2, 0.1)

st.write(
    "Wgraj plik audio (MP3/WAV/M4A/AAC/MP4/OGG/WEBM), zrób transkrypcję po polsku, "
    "a następnie wygeneruj skrót, listę wątków i cytaty."
)

# =========================
# POMOCNICZE
# =========================
def format_ts(seconds: int) -> str:
    # Zwraca znacznik mm:ss lub hh:mm:ss dla czytelności
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def remove_fillers(text: str) -> str:
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

def do_transcribe(file_bytes: bytes, filename: str) -> str:
    """
    Szybka próba transkrypcji jako pojedynczy plik.
    Dobra dla krótkich nagrań (poniżej limitu modelu).
    """
    resp = client.audio.transcriptions.create(
        model=model_transcribe,
        file=(filename, file_bytes),
        language=language,
    )
    return getattr(resp, "text", str(resp))

def export_segment_to_bytes(segment: AudioSegment, name: str, fmt="mp3") -> Dict:
    buf = io.BytesIO()
    segment.export(buf, format=fmt)
    buf.seek(0)
    return {"bytes": buf.read(), "name": f"{name}.{fmt}"}

def split_by_duration(audio: AudioSegment, max_seconds: int) -> List[Dict]:
    """
    Proste cięcie co 'max_seconds', bez analizy ciszy.
    """
    total_ms = len(audio)
    step_ms = max_seconds * 1000
    chunks = []
    for i in range(0, total_ms, step_ms):
        segment = audio[i:i + step_ms]
        start_s = i // 1000
        end_s = min((i + step_ms) // 1000, math.ceil(total_ms / 1000))
        chunks.append({
            "segment": segment,
            "start_s": start_s,
            "end_s": end_s,
            "label": f"{format_ts(start_s)}–{format_ts(end_s)}"
        })
    return chunks

def split_by_silence_capped(audio: AudioSegment,
                            max_seconds: int,
                            min_silence_len: int,
                            silence_thresh_db: int,
                            keep_silence_ms: int) -> List[Dict]:
    """
    Dzielenie po ciszy z limitem długości każdego kawałka (soft cap).
    Tworzy „mikro-kawałki” po ciszy, a następnie składa je w większe fragmenty <= max_seconds.
    """
    # 1) Cięcie w punktach ciszy
    micro = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh_db,
        keep_silence=keep_silence_ms
    )
    if not micro:
        # Jeśli nie udało się znaleźć ciszy – fallback do cięcia co czas
        return split_by_duration(audio, max_seconds)

    # 2) Składanie mikro-kawałków do „kapowanych” segmentów
    max_ms = max_seconds * 1000
    chunks = []
    current = AudioSegment.empty()
    current_start_s = 0
    acc_ms = 0
    global_cursor = 0  # w sekundach, idziemy przez audio

    for seg in micro:
        if len(current) == 0:
            current_start_s = global_cursor
        if len(current) + len(seg) <= max_ms:
            current += seg
            acc_ms += len(seg)
            global_cursor += len(seg) // 1000
        else:
            # zamknij obecny
            start_s = current_start_s
            end_s = current_start_s + (len(current) // 1000)
            chunks.append({
                "segment": current,
                "start_s": start_s,
                "end_s": end_s,
                "label": f"{format_ts(start_s)}–{format_ts(end_s)}"
            })
            # zacznij nowy
            current = seg
            current_start_s = global_cursor
            acc_ms = len(seg)
            global_cursor += len(seg) // 1000

    if len(current) > 0:
        start_s = current_start_s
        end_s = current_start_s + (len(current) // 1000)
        chunks.append({
            "segment": current,
            "start_s": start_s,
            "end_s": end_s,
            "label": f"{format_ts(start_s)}–{format_ts(end_s)}"
        })

    return chunks

def transcribe_chunked(file_bytes: bytes,
                       filename: str,
                       chunk_seconds: int,
                       use_silence: bool,
                       min_silence_len: int,
                       silence_thresh_db: int,
                       keep_silence_ms: int) -> str:
    """
    Transkrypcja długiego pliku: cięcie -> transkrypcja kawałków -> sklejenie.
    """
    # Zapis do pliku tymczasowego (pydub czyta z pliku)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    audio = AudioSegment.from_file(tmp_path)

    # Wybór strategii dzielenia
    if use_silence:
        chunks = split_by_silence_capped(
            audio,
            max_seconds=chunk_seconds,
            min_silence_len=min_silence_len,
            silence_thresh_db=silence_thresh_db,
            keep_silence_ms=keep_silence_ms
        )
    else:
        chunks = split_by_duration(audio, max_seconds=chunk_seconds)

    combined_text = []
    prog = st.progress(0, text="Transkrypcja kawałków…")
    total = len(chunks)

    for idx, ch in enumerate(chunks, start=1):
        payload = export_segment_to_bytes(ch["segment"], f"chunk_{ch['start_s']:06d}-{ch['end_s']:06d}")
        resp = client.audio.transcriptions.create(
            model=model_transcribe,
            file=(payload["name"], payload["bytes"]),
            language=language
        )
        text = getattr(resp, "text", str(resp)).strip()
        combined_text.append(f"[{ch['label']}] {text}")
        prog.progress(idx / total, text=f"Kawałek {idx}/{total} ({ch['label']})")

    prog.empty()
    return "\n\n".join(combined_text)

def build_summary_md(text: str) -> str:
    """
    Generuje skrót, listę wątków i pakiet cytatów na podstawie transkrypcji.
    """
    system = (
        "Jesteś redaktorem podcastu (styl: dociekliwość, klarowność, naturalny polski). "
        "Na podstawie transkrypcji po polsku przygotuj:\n"
        "1) Krótkie podsumowanie (3–6 zdań),\n"
        "2) Listę głównych wątków (5–10 punktów),\n"
        "3) 6–10 cytatów do promocji (zwięzłe, naturalne, bez wypełniaczy). "
        "Nie wymyślaj treści — trzymaj się dosłownie tego, co jest w tekście."
    )
    user = f"TRANSKRYPCJA (PL):\n{text}"
    chat = client.chat.completions.create(
        model=model_summarize,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=temperature,
    )
    return chat.choices[0].message.content

# =========================
# UI – UPLOAD & AKCJE
# =========================
uploaded = st.file_uploader("Plik audio", type=["mp3", "wav", "m4a", "aac", "mp4", "ogg", "webm"])

transcribed_text = st.session_state.get("transcribed_text", "")
clean_text = st.session_state.get("clean_text", "")
summary_md = st.session_state.get("summary_md", "")

col1, col2 = st.columns(2)
with col1:
    run_transcribe = st.button("🔁 Zrób transkrypcję", use_container_width=True, disabled=not uploaded)
with col2:
    run_summarize = st.button("✨ Stwórz skrót i cytaty", use_container_width=True, disabled=not bool(transcribed_text))

# =========================
# LOGIKA – TRANSKRYPCJA
# =========================
if run_transcribe and uploaded:
    try:
        with st.spinner("Transkrypcja w toku…"):
            file_bytes = uploaded.read()
            # 1) Szybka próba dla krótkich plików
            try:
                text = do_transcribe(file_bytes, uploaded.name)
            except Exception as e_single:
                # 2) Jeśli za długi lub inny błąd – transkrypcja kawałkami
                text = transcribe_chunked(
                    file_bytes=file_bytes,
                    filename=uploaded.name,
                    chunk_seconds=chunk_seconds,
                    use_silence=use_silence_split,
                    min_silence_len=min_silence_len,
                    silence_thresh_db=silence_thresh_db,
                    keep_silence_ms=keep_silence_ms
                )

            # Czyszczenie wypełniaczy (opcjonalnie)
            text_clean = remove_fillers(text) if rm_fillers else text

            st.session_state["transcribed_text"] = text
            st.session_state["clean_text"] = text_clean
            transcribed_text = text
            clean_text = text_clean

        st.success("Transkrypcja gotowa ✅")

    except Exception as e:
        st.error(f"Błąd transkrypcji: {e}")

# =========================
# PODGLĄD TEKSTU
# =========================
if transcribed_text:
    st.subheader("📄 Transkrypcja (surowa)")
    st.text_area("Tekst", transcribed_text, height=280)

    st.subheader("🧹 Po czyszczeniu (opcjonalnie)")
    st.caption("Wersja z usuniętymi wypełniaczami i poprawioną interpunkcją.")
    st.text_area("Tekst (clean)", clean_text, height=280)

# =========================
# PODSUMOWANIE / CYTATY
# =========================
if run_summarize and transcribed_text:
    try:
        with st.spinner("Generuję skrót i cytaty…"):
            summary = build_summary_md(clean_text if rm_fillers else transcribed_text)
            st.session_state["summary_md"] = summary
            summary_md = summary
        st.success("Gotowe ✅")
    except Exception as e:
        st.error(f"Błąd generowania podsumowania: {e}")

if summary_md:
    st.subheader("🧭 Skrót, wątki i cytaty (Markdown)")
    st.markdown(summary_md)
    st.download_button("⬇️ Pobierz wynik (MD)", data=summary_md, file_name="transkryptor_wynik.md")

st.caption("Wskazówka: dla trudniejszych nagrań wybierz dokładniejszy model transkrypcji lub zwiększ czułość cięcia po ciszy.")
