import io
import re
import time
import streamlit as st
from openai import OpenAI

# --- KONFIG ---
st.set_page_config(page_title="Transkryptor – Streamlit + OpenAI", layout="centered")
st.title("🎙️ Transkryptor (PL) – Streamlit + OpenAI")

# --- KLIENT OPENAI Z SECRETS ---
if "OPENAI_API_KEY" not in st.secrets:
    st.error("Brak klucza API. Dodaj OPENAI_API_KEY do .streamlit/secrets.toml lub w Streamlit Cloud → Secrets.")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- SIDEBAR USTAWIENIA ---
with st.sidebar:
    st.header("Ustawienia")
    model_transcribe = st.selectbox(
        "Model transkrypcji",
        ["gpt-4o-mini-transcribe", "gpt-4o-transcribe"],
        index=0,
        help="mini = szybszy i tańszy; transcribe = wyższa jakość"
    )
    model_summarize = st.selectbox(
        "Model podsumowania",
        ["gpt-4o-mini", "gpt-4o"],
        index=0
    )
    language = st.text_input("Język nagrania (ISO)", value="pl", help="Np. pl, en, de…")
    rm_fillers = st.checkbox("Usuń wypełniacze (eee, yyy, no więc…)", value=True)
    temperature = st.slider("Kreatywność podsumowania", 0.0, 1.0, 0.2, 0.1)

st.write("Wgraj plik audio (MP3/WAV/M4A/AAC/MP4/OGG/WEBM) i zrób transkrypcję po polsku, a następnie wygeneruj skrót, wątki i cytaty.")

# --- UPLOAD ---
uploaded = st.file_uploader("Plik audio", type=["mp3","wav","m4a","aac","mp4","ogg","webm"])
transcribed_text = st.session_state.get("transcribed_text", "")
clean_text = st.session_state.get("clean_text", "")
summary_md = st.session_state.get("summary_md", "")

def remove_fillers(text: str) -> str:
    # Prosta filtracja polskich wypełniaczy; możesz rozwinąć listę
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
    # Usuń wielokrotne spacje i popraw interpunkcję
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
    return cleaned.strip()

def do_transcribe(file_bytes: bytes, filename: str) -> str:
    # Przesyłamy jako tuple (name, bytes) — openai-python v1.x
    resp = client.audio.transcriptions.create(
        model=model_transcribe,
        file=(filename, file_bytes),
        language=language
        # Możesz dodać: response_format="verbose_json", timestamp_granularities=["word"]
    )
    # SDK zwraca obiekt z polem .text
    return getattr(resp, "text", str(resp))

def build_summary_md(text: str) -> str:
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
        messages=[{"role":"system","content":system},
                  {"role":"user","content":user}],
        temperature=temperature,
    )
    return chat.choices[0].message.content

# --- AKCJE ---
col1, col2 = st.columns(2)
with col1:
    run_transcribe = st.button("🔁 Zrób transkrypcję", use_container_width=True, disabled=not uploaded)
with col2:
    run_summarize = st.button("✨ Stwórz skrót i cytaty", use_container_width=True, disabled=not bool(transcribed_text))

# --- TRANSKRYPCJA ---
if run_transcribe and uploaded:
    try:
        with st.spinner("Transkrypcja w toku…"):
            file_bytes = uploaded.read()
            text = do_transcribe(file_bytes, uploaded.name)
            if rm_fillers:
                text_clean = remove_fillers(text)
            else:
                text_clean = text
            st.session_state["transcribed_text"] = text
            st.session_state["clean_text"] = text_clean
            transcribed_text = text
            clean_text = text_clean
        st.success("Gotowe ✅")
    except Exception as e:
        st.error(f"Błąd transkrypcji: {e}")

# --- PODGLĄD TEKSTU ---
if transcribed_text:
    st.subheader("📄 Transkrypcja (surowa)")
    st.text_area("Tekst", transcribed_text, height=280)

    st.subheader("🧹 Po czyszczeniu (opcjonalnie)")
    st.caption("Wersja z usuniętymi wypełniaczami i poprawioną interpunkcją.")
    st.text_area("Tekst (clean)", clean_text, height=280)

# --- PODSUMOWANIE / CYTATY ---
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

st.caption("Wskazówka: dla gorszych nagrań wybierz dokładniejszy model transkrypcji w panelu po lewej.")
