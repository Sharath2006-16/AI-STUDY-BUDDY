import streamlit as st
import ollama
import io
import re
from pypdf import PdfReader

st.set_page_config(
    page_title="AI Study Buddy",
    page_icon="📚",
    layout="wide"
)

MODEL = "gemma3"

st.title("📚 AI Study Buddy")
st.write("Upload your PDF and get a short summary, pictograph, and flashcards — using a local AI model with no OpenAI API credits.")

def extract_pdf_text(uploaded_file):
    reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)

def ask_ollama(prompt):
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful college study assistant. Give simple, accurate, exam-friendly answers."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.message.content

def make_pictograph(text):
    words = re.findall(r"\b[A-Za-z]{4,}\b", text.lower())
    stop_words = {
        "this", "that", "with", "from", "have", "which", "their",
        "there", "about", "these", "those", "using", "into",
        "also", "more", "than", "when", "where", "what", "will",
        "been", "were", "they", "them", "then", "such", "each"
    }

    counts = {}
    for word in words:
        if word not in stop_words:
            counts[word] = counts.get(word, 0) + 1

    top_words = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:8]

    if not top_words:
        return "No keywords found."

    result = []
    for word, count in top_words:
        icons = "📘" * min(max(1, count // 2), 10)
        result.append(f"**{word.title()}**  {icons}")

    return "\n\n".join(result)

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

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.info("AI runs locally through Ollama.")
    st.write(f"Model: `{MODEL}`")
    st.write("No OpenAI API key is required.")

uploaded_file = st.file_uploader(
    "📄 Upload your study PDF",
    type=["pdf"]
)

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("🚀 Generate Study Pack", type="primary"):
        try:
            with st.spinner("Reading your PDF..."):
                text = extract_pdf_text(uploaded_file)

            if not text.strip():
                st.error("No readable text was found in this PDF.")
                st.stop()

            # Keep very large PDFs manageable for a small local model.
            max_chars = 30000
            study_text = text[:max_chars]

            with st.spinner("Creating short summary..."):
                summary = ask_ollama(
                    f"""Summarize the following study material.

Rules:
- Use simple English.
- Keep it short and exam-friendly.
- Use clear headings and bullet points.
- Include important definitions, concepts, formulas, and facts.
- Do not add information that is not present in the material.

STUDY MATERIAL:
{study_text}"""
                )

            with st.spinner("Creating flashcards..."):
                flashcards = ask_ollama(
                    f"""Create 10 useful exam flashcards from this study material.

Format exactly like:
Q1. Question
A1. Short answer

Q2. Question
A2. Short answer

Use simple language and focus on important concepts.

STUDY MATERIAL:
{study_text}"""
                )

            with st.spinner("Creating pictograph..."):
                pictograph = make_pictograph(text)

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
                summary,
                flashcards,
                pictograph,
                uploaded_file.name
            )

            st.divider()
            st.subheader("📥 Download Study Pack")

            st.download_button(
                label="⬇️ Download Study Pack",
                data=study_pack,
                file_name="AI_Study_Buddy_Study_Pack.txt",
                mime="text/plain"
            )

        except Exception as e:
            error_text = str(e)

            if "Connection" in error_text or "11434" in error_text:
                st.error("Ollama is not running.")
                st.info(
                    "Open PowerShell and run: ollama serve\n\n"
                    "Then make sure the model is installed with: ollama pull gemma3"
                )
            elif "model" in error_text.lower() and "not found" in error_text.lower():
                st.error(f"The model '{MODEL}' is not installed.")
                st.info(f"Run this in PowerShell: ollama pull {MODEL}")
            else:
                st.error("Something went wrong.")
                st.code(error_text)

else:
    st.info("👆 Upload a PDF to start.")

    st.markdown(
        )
