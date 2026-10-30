
"""ResearchLens: a simple web interface for evidence-grounded PDF research."""

import hashlib
import io
import os

import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

from research import (
    OPENROUTER_BASE_URL,
    PageText,
    clean_text,
    make_chunks,
    retrieve_evidence,
    write_brief,
)

st.set_page_config(
    page_title="ResearchLens",
    page_icon="🔎",
    layout="wide",
)

st.title("ResearchLens")
st.write(
    "Turn PDF reports into research briefs with traceable page citations."
)

st.info(
    "ResearchLens retrieves relevant passages from your documents. "
    "Always check the cited passages before relying on an answer."
)

with st.sidebar:
    st.header("About ResearchLens")
    st.write("1. Upload PDF reports.")
    st.write("2. Ask a specific research question.")
    st.write("3. Review the answer and its supporting evidence.")
    st.caption(
        "Document text sent for analysis is processed by the configured "
        "AI provider. Avoid uploading confidential documents."
    )

uploaded_files = st.file_uploader(
    "Upload your PDF reports",
    type=["pdf"],
    accept_multiple_files=True,
)

question = st.text_area(
    "Your research question",
    placeholder=(
        "Example: What drove MTN Nigeria's FY2025 revenue growth, "
        "and how did profitability change?"
    ),
    height=100,
)

if uploaded_files:
    st.caption(
        f"{len(uploaded_files)} PDF file(s) selected."
    )

if st.button("Generate research brief", type="primary"):
    if not uploaded_files:
        st.error("Upload at least one PDF report first.")
        st.stop()

    if not question.strip():
        st.error("Enter a research question first.")
        st.stop()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        st.error(
            "OPENROUTER_API_KEY is not set in this terminal environment. "
            "Set it locally, then restart the app."
        )
        st.stop()

    try:
        with st.spinner("Reading your reports..."):
            pages = []
            file_summaries = []

            for uploaded_file in uploaded_files:
                pdf_bytes = uploaded_file.getvalue()
                reader = PdfReader(io.BytesIO(pdf_bytes))

                extracted_count = 0

                for page_number, page in enumerate(reader.pages, start=1):
                    raw_text = page.extract_text() or ""
                    text = clean_text(raw_text)

                    if text.strip():
                        pages.append(
                            PageText(
                                filename=uploaded_file.name,
                                page_number=page_number,
                                text=text,
                            )
                        )
                        extracted_count += 1

                file_summaries.append(
                    {
                        "filename": uploaded_file.name,
                        "pages": len(reader.pages),
                        "pages_with_text": extracted_count,
                    }
                )

            if not pages:
                st.error(
                    "No readable text was extracted. The PDFs may be "
                    "scanned images and require OCR."
                )
                st.stop()

            chunks = make_chunks(pages)

            if not chunks:
                st.error("No searchable text chunks could be created.")
                st.stop()

        st.write("### Reports processed")
        for item in file_summaries:
            st.write(
                f"- **{item['filename']}** — "
                f"{item['pages']} PDF pages; "
                f"{item['pages_with_text']} pages with readable text"
            )

        with st.spinner(
            f"Searching {len(chunks)} text chunks and writing your brief..."
        ):
            client = OpenAI(
                api_key=api_key,
                base_url=OPENROUTER_BASE_URL,
            )

            evidence = retrieve_evidence(
                client,
                question.strip(),
                chunks,
            )

            if not evidence:
                st.warning(
                    "No relevant evidence was retrieved. Try a more "
                    "specific question or a different report."
                )
                st.stop()

            brief = write_brief(
                client,
                question.strip(),
                evidence,
            )

        st.divider()
        st.subheader("Research brief")
        st.markdown(brief)

        st.download_button(
            "Download brief as Markdown",
            data=brief,
            file_name="researchlens_brief.md",
            mime="text/markdown",
        )

        with st.expander(
            f"Inspect supporting evidence ({len(evidence)} passages)"
        ):
            st.caption(
                "These are the passages retrieved for your question. "
                "Check that each passage actually supports the claim "
                "for which it is cited."
            )

            for index, chunk in enumerate(evidence, start=1):
                st.markdown(
                    f"**Passage {index} — {chunk.filename}, "
                    f"page {chunk.page_number}**"
                )
                st.write(chunk.text)
                st.divider()

    except Exception as exc:
        st.error(
            "ResearchLens could not complete the request. "
            "Check the terminal for diagnostic details."
        )
        print(f"ResearchLens error: {type(exc).__name__}: {exc}")
