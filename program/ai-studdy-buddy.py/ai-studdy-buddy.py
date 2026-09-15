import io
import re

import ollama
import streamlit as st
from pypdf import PdfReader


st.set_page_config(
    page_title="AI Study Buddy",
    page_icon="📚",
    layout="wide",
)

MODEL = "gemma3:1b"


def extract_pdf_text(uploaded_file):
    reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
    pages = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)
    return "\n".join(pages)


def ask_ollama(prompt):
    response = ollama.chat(
        model=MODEL,
        options={"num_predict": 700, "temperature": 0.2},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful college study assistant. "
                    "Give simple, accurate, exam-friendly answers."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response["message"]["content"]


def split_study_pack(result):
    marker = re.search(r"(?im)^\s*(?:FLASHCARDS|##?\s*FLASHCARDS)\s*:?\s*$", result)
    if not marker:
        return result, result
    return result[: marker.start()].strip(), result[marker.end() :].strip()


def make_pictograph(text):
    words = re.findall(r"\b[A-Za-z]{4,}\b", text.lower())
    stop_words = {
        "this", "that", "with", "from", "have", "which", "their",
        "there", "about", "these", "those", "using", "into", "also",
        "more", "than", "when", "where", "what", "will", "been",
        "were", "they", "them", "then", "such", "each",
    }
    counts = {}
    for word in words:
        if word not in stop_words:
            counts[word] = counts.get(word, 0) + 1

    top_words = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:8]
    if not top_words:
        return [], "No keywords found."

    pictograph = "\n\n".join(
        f"**{word.title()}**  {'📘' * min(max(1, count // 2), 10)}"
        for word, count in top_words
    )
    return top_words, pictograph


def create_study_pack(summary, flashcards, pictograph, filename):
    return f"""AI STUDY BUDDY - STUDY PACK
================================

SOURCE PDF
{filename}

SHORT SUMMARY
-------------
{summary}

PICTOGRAPH / KEYWORDS
---------------------
{pictograph}

FLASHCARDS
----------
{flashcards}

================================
Created using a local Ollama model.
No OpenAI API credits are required.
"""


st.title("📚 AI Study Buddy")
st.write(
    "Upload your PDF and get a short summary, pictograph, and flashcards "
    "using a local AI model with no OpenAI API credits."
)

with st.sidebar:
    st.header("⚙️ Settings")
    st.info("AI runs locally through Ollama.")
    st.write(f"Model: `{MODEL}`")
    st.write("No OpenAI API key is required.")

uploaded_file = st.file_uploader("📄 Upload your study PDF", type=["pdf"])

if not uploaded_file:
    st.info("👆 Upload a PDF to start.")
else:
    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("🚀 Generate Study Pack", type="primary"):
        try:
            with st.spinner("Reading your PDF..."):
                text = extract_pdf_text(uploaded_file)

            if not text.strip():
                st.error("No readable text was found in this PDF.")
                st.stop()

            # A shorter excerpt keeps CPU-only generation responsive.
            study_text = text[:14000]

            with st.spinner("Creating your study pack..."):
                generated_pack = ask_ollama(
                    f"""Create a short study pack from the following study material.

Rules:
- Use simple English.
- Keep it short and exam-friendly.
- Use these exact headings: SHORT SUMMARY and FLASHCARDS.
- Under SHORT SUMMARY, write 5-8 bullet points.
- Under FLASHCARDS, create 5 question-and-answer pairs.
- Include important definitions, concepts, formulas, and facts.
- Do not add information that is not present in the material.

STUDY MATERIAL:
{study_text}"""
                )
            summary, flashcards = split_study_pack(generated_pack)

            with st.spinner("Creating pictograph..."):
                keyword_counts, pictograph = make_pictograph(text)

            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📝 Short Summary")
                st.markdown(summary)
            with col2:
                st.subheader("📊 Pictograph / Important Keywords")
                st.markdown(pictograph)

            st.divider()
            st.subheader("🧠 Flashcards")
            st.markdown(flashcards)

            study_pack = create_study_pack(
                summary, flashcards, pictograph, uploaded_file.name
            )
            st.divider()
            st.subheader("📥 Download Study Pack")
            st.download_button(
                label="⬇️ Download Study Pack",
                data=study_pack,
                file_name="AI_Study_Buddy_Study_Pack.txt",
                mime="text/plain",
            )
        except Exception as error:
            error_text = str(error)
            if "CUDA" in error_text or "cuda" in error_text:
                st.error("Ollama could not initialize CUDA/GPU acceleration.")
                st.info(
                    "Close Ollama, then start it in CPU mode from PowerShell with: "
                    "`$env:OLLAMA_LLM_LIBRARY='cpu_avx2'; ollama serve`"
                )
                st.warning(
                    "If `ollama` is not in PATH, use the full Ollama executable path "
                    "instead of `ollama serve`."
                )
            elif "Connection" in error_text or "11434" in error_text:
                st.error("Ollama is not running.")
                st.info(
                    "Open PowerShell and run `ollama serve`, then install the "
                    "model with `ollama pull gemma3`."
                )
            elif "not found" in error_text.lower() and "model" in error_text.lower():
                st.error(f"The model '{MODEL}' is not installed.")
                st.info(f"Run this in PowerShell: `ollama pull {MODEL}`")
            else:
                st.error("Something went wrong.")
                st.code(error_text)
