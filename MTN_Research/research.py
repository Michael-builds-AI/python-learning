"""MTN Research Brief: a small, evidence-grounded PDF research assistant.

Place public MTN Nigeria PDF reports in a local ``documents`` folder, then run:
    python research.py "What did the reports say about subscriber growth?"

The program sends report text only for the most relevant chunks to the LLM.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from openai import OpenAI
from pypdf import PdfReader

DOCUMENTS_DIR = Path("documents")
EMBEDDING_MODEL = os.getenv("MTN_EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("MTN_CHAT_MODEL", "openai/gpt-4.1-mini")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CHUNK_SIZE = 1_200
CHUNK_OVERLAP = 200
TOP_RESULTS = 6
INSUFFICIENT_EVIDENCE_MESSAGE = "The available evidence is insufficient to answer this question."


@dataclass
class PageText:
    """Text extracted from one PDF page, with source information."""

    filename: str
    page_number: int
    text: str


@dataclass
class Chunk:
    """A searchable piece of report text that retains its source information."""

    filename: str
    page_number: int
    text: str
    embedding: list[float] | None = None
    similarity_score: float | None = None


def clean_text(text: str) -> str:
    """Make extracted PDF text easier to search without changing its meaning."""
    text = text.replace("\u00ad", "")  # Soft hyphens commonly appear in PDFs.
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)  # Join broken words.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_pages(documents_dir: Path) -> list[PageText]:
    """Extract non-empty text from every page in every PDF in a directory."""
    pdf_paths = sorted(documents_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in '{documents_dir}'. Add PDF reports there and try again."
        )

    pages: list[PageText] = []
    for pdf_path in pdf_paths:
        try:
            reader = PdfReader(pdf_path)
        except Exception as error:
            print(f"Warning: could not open {pdf_path.name}: {error}", file=sys.stderr)
            continue

        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = clean_text(page.extract_text() or "")
            except Exception as error:
                print(
                    f"Warning: could not extract {pdf_path.name}, page {page_number}: {error}",
                    file=sys.stderr,
                )
                continue
            if text:
                pages.append(PageText(pdf_path.name, page_number, text))

    if not pages:
        raise ValueError("No readable text was extracted from the PDF files.")
    return pages


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping, readable chunks, preferring sentence breaks."""
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            break_at = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if break_at > start + size // 2:
                end = break_at + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def make_chunks(pages: Iterable[PageText]) -> list[Chunk]:
    """Create chunks without mixing content from different pages."""
    return [
        Chunk(page.filename, page.page_number, text)
        for page in pages
        for text in split_text(page.text)
    ]


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Create embeddings in batches so larger reports remain practical."""
    embeddings: list[list[float]] = []
    batch_size = 100
    for start in range(0, len(texts), batch_size):
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts[start : start + batch_size],
        )
        embeddings.extend(item.embedding for item in response.data)
    return embeddings


def cosine_similarity(first: list[float], second: list[float]) -> float:
    """Return cosine similarity between two embedding vectors."""
    numerator = sum(a * b for a, b in zip(first, second))
    first_length = math.sqrt(sum(a * a for a in first))
    second_length = math.sqrt(sum(b * b for b in second))
    if not first_length or not second_length:
        return 0.0
    return numerator / (first_length * second_length)


def retrieve_evidence(client: OpenAI, question: str, chunks: list[Chunk]) -> list[Chunk]:
    """Return the chunks whose embeddings are closest to the question embedding."""
    chunk_embeddings = embed_texts(client, [chunk.text for chunk in chunks])
    for chunk, embedding in zip(chunks, chunk_embeddings):
        chunk.embedding = embedding

    question_embedding = embed_texts(client, [question])[0]
    for chunk in chunks:
        chunk.similarity_score = cosine_similarity(question_embedding, chunk.embedding or [])
    return sorted(
        chunks,
        key=lambda chunk: chunk.similarity_score or 0.0,
        reverse=True,
    )[: min(TOP_RESULTS, len(chunks))]


def format_evidence(chunks: list[Chunk]) -> str:
    """Label evidence so the LLM can use exact source citations."""
    return "\n\n".join(
        f"[{chunk.filename}, p. {chunk.page_number}]\n{chunk.text}"
        for chunk in chunks
    )


def citations_are_valid(brief: str, evidence: list[Chunk]) -> bool:
    """Accept only the filename/page citations that were supplied as evidence."""
    allowed_citations = {
        f"[{chunk.filename}, p. {chunk.page_number}]" for chunk in evidence
    }

    bracketed_text = set(re.findall(r"\[[^\]]+\]", brief))

    citations_in_brief = {
        citation
        for citation in bracketed_text
        if re.search(
            r"\.(?:pdf|docx|txt)\b|\b(?:pages?|pp?\.?)\s*\d",
            citation,
            re.IGNORECASE,
        )
    }

    if not citations_in_brief:
        return brief.strip() == INSUFFICIENT_EVIDENCE_MESSAGE

    return all(
        citation_is_valid(citation, allowed_citations)
        for citation in citations_in_brief
    )



def citation_is_valid(citation: str, allowed_citations: set[str]) -> bool:
    """Validate individual or grouped citations inside one pair of brackets."""
    citation_parts = re.split(r";\s*", citation[1:-1])

    if not citation_parts:
        return False

    filename = None
    validated_parts = []

    for part in citation_parts:
        part = part.strip()

        # A citation may omit the filename only when it follows
        # a citation that already supplied the filename.
        if filename is not None and re.fullmatch(r"p{1,2}\.\s*\d+", part):
            part = f"{filename}, {part}"
        else:
            match = re.match(r"^(.+?\.(?:pdf|docx|txt)),\s*(.+)$", part, re.IGNORECASE)
            if not match:
                return False
            filename, page_reference = match.groups()
            part = f"{filename}, {page_reference}"

        if part.startswith(filename + ", pp. "):
            if not citation_part_is_valid(f"[{part}]", allowed_citations):
                return False
        else:
            match = re.fullmatch(r"(.+),\s*p\.\s*(\d+)", part)
            if not match:
                return False
            file_name, page_number = match.groups()
            if f"[{file_name}, p. {page_number}]" not in allowed_citations:
                return False

        validated_parts.append(part)

    return bool(validated_parts)



def citation_part_is_valid(citation: str, allowed_citations: set[str]) -> bool:
    """Validate individual or grouped citations against retrieved evidence."""
    if citation in allowed_citations:
        return True

    grouped_match = re.fullmatch(r"\[(.+), pp\. (.+)\]", citation)
    if not grouped_match:
        return False

    filename, page_references = grouped_match.groups()
    cited_pages: list[int] = []

    for page_reference in page_references.split(", "):
        page_range = re.fullmatch(r"(\d+)(?:\s*[–-]\s*(\d+))?", page_reference)
        if not page_range:
            return False

        first_page = int(page_range.group(1))
        last_page = int(page_range.group(2) or first_page)

        if last_page < first_page:
            return False

        cited_pages.extend(range(first_page, last_page + 1))

    if len(cited_pages) < 2:
        return False

    return all(
        f"[{filename}, p. {page_number}]" in allowed_citations
        for page_number in cited_pages
    )


def write_brief(
    client: OpenAI, question: str, evidence: list[Chunk], debug: bool = False
) -> str:
    """Ask the LLM for a concise answer based solely on retrieved report text."""
    instructions = """You write concise research briefs about MTN Nigeria reports.

Use ONLY the evidence provided below. Do not use outside knowledge or make inferences
that are not directly supported by the evidence.

CITATIONS:
Cite every factual claim using the exact format [filename, p. N], matching a supplied
source label. Every citation MUST include the complete filename exactly as supplied in
the retrieved evidence. Never use shorthand citations such as [p. 5], [p. 3], [pp. 3, 5],
or [p. 3; p. 5]. Never assume that a filename from an earlier citation carries forward
to a later citation.

Never invent a citation, filename, page number, figure, or fact. Build citations only
from the exact filename/page labels in the retrieved evidence.

You may use a grouped citation such as [filename, pp. 3, 5, 7] when all cited pages are
supplied. You may use a page range such as [filename, pp. 7–9] only when every page in
the range (p. 7, p. 8, and p. 9) is supplied. Never use a range that includes an
unavailable page. For example, if p. 7 and p. 9 are supplied but p. 8 is not, cite
[filename, pp. 7, 9] instead of [filename, pp. 7–9].

RESEARCH DISCIPLINE:
When a question asks about revenue drivers, distinguish carefully between:
1. Direct revenue drivers — factors that the supplied evidence explicitly connects to
   revenue or service-revenue growth.
2. Margin or profitability factors — factors that the supplied evidence explicitly
   connects to costs, margins, EBITDA, profitability, or efficiency.
3. Contextual factors — broader economic or operating conditions that provide context
   but are not explicitly established as direct revenue drivers.
4. Risks or uncertainties — issues identified in the evidence whose future impact is
   uncertain.

Do not classify a factor as a direct revenue driver merely because management describes
it as supporting overall performance. For example, if the evidence says that FX
stability, lower inflation, or currency appreciation reduced cost pressure or supported
margins, classify those as margin/profitability or contextual factors unless the
evidence explicitly connects them to revenue growth.

Do not turn a management statement about overall performance into a stronger causal
claim than the evidence supports.

TEMPORAL AND CROSS-PAGE DISCIPLINE:
When evidence from different pages refers to different time periods, events, or stages,
do not combine them into one statement unless the evidence explicitly connects them.
Preserve the exact timing stated in each source. In particular, distinguish between the
timing of an event's impact and the later timing of a base effect, comparison effect,
absorption, or normalization.

REVENUE-DRIVER EVIDENCE DISCIPLINE:
Do not describe a factor as driving revenue merely because it increased alongside revenue,
was associated with business activity, or was described as significant to the business.

For fintech and payment services in particular, distinguish between:
- transaction volume or transaction value,
- customer/activity growth,
- and actual reported revenue.

If the evidence explicitly says that transaction activity does not translate proportionally
into revenue, do not describe transaction volume or value as a direct revenue driver unless
the evidence separately establishes that connection.

When the evidence establishes that a factor affected costs, margins, EBITDA, or profitability
but does not explicitly connect it to revenue, classify it under margin/profitability or
context rather than revenue growth.

PARTIAL-ANSWER RULE:
Do not return the insufficient-evidence message merely because the evidence does not
establish every requested causal relationship.

If the evidence supports some parts of the question, answer those parts and clearly
label uncertain or indirect factors as contextual rather than presenting them as proven
drivers.

For example, if the evidence shows that fintech revenue increased and separately explains
that transaction values do not necessarily translate proportionally into revenue, report
the fintech revenue growth as an observed result and explain the transaction-activity
evidence as context, rather than claiming that transaction value caused the revenue growth.

Use the insufficient-evidence message only when the retrieved evidence genuinely cannot
support a useful answer to the question at all.

CAUSALITY DISCIPLINE:
If the evidence describes an operational activity, investment, capability, or business
condition but does not explicitly connect it to revenue, margin, EBITDA, cost reduction,
or profitability, do not present it as a financial driver.

You may mention such information as operational context when relevant, but do not infer
that it supported profitability merely because it may plausibly have done so.

Use phrases such as "the evidence shows" or "the company reported" when describing an
observed fact. Reserve causal language such as "drove", "supported", "enabled", or
"resulted in" for relationships explicitly established by the evidence.

If the evidence does not answer the question, clearly state:
'The available evidence is insufficient to answer this question.'
You may still summarize what the evidence does show.

Do not give investment, buy/sell, or price predictions.

FINAL ANSWER STRUCTURE:
For questions about revenue growth and profitability, use these four short sections:
1. Direct revenue drivers explicitly supported by the evidence.
2. Profitability and margin factors explicitly supported by the evidence.
3. Reported financial results and relevant context.
4. Limitations or relationships the evidence does not establish.

LENGTH LIMIT:
Keep the complete answer under 300 words, including the conclusion.
Use four short numbered sections with no more than two bullet points
per section. Prioritize the strongest findings and combine related
facts where appropriate. Include complete, valid citations immediately
after supported claims. Do not repeat findings or add a separate
summary after the four sections.

Keep the entire answer under 650 words, preferably 400–500 words.
Use concise bullet points and avoid repeating the same finding in multiple sections.
Prioritize the strongest evidence and include citations immediately after the
claims they support. Finish every section and the conclusion before stopping.
Do not begin a new claim if there is insufficient space to complete the answer
and provide its citation.

A passage containing an analyst's question is not evidence that the question's
assumption is true or that a proposed explanation is correct. Use management's
answer or another substantive source passage to support the claim. Do not cite
a question-only passage as evidence for the answer.

Do not claim that fintech revenue grew, or that a particular factor caused
fintech revenue growth, unless the retrieved evidence explicitly establishes it.
Distinguish an observed revenue contribution from an explanation of its cause.

Do not list macroeconomic conditions as direct revenue drivers unless the
evidence explicitly connects them to revenue growth.

A passage containing an analyst's question is not evidence that the question's
assumption is true or that a proposed explanation is correct. Use management's
answer or another substantive source passage to support the claim. Do not cite
a question-only passage as evidence for the answer.

Do not claim that fintech revenue grew, or that a particular factor caused
fintech revenue growth, unless the retrieved evidence explicitly establishes it.
Distinguish an observed revenue contribution from an explanation of its cause."""
    evidence_text = format_evidence(evidence)
    if debug:
        print("\n--- Debug: evidence sent to model ---")
        print(evidence_text)

    response = client.responses.create(
        model=CHAT_MODEL,
        max_output_tokens=1000,
        instructions=instructions,
        input=(
            f"Research question: {question}\n\n"
            f"Retrieved evidence:\n{evidence_text}"
        ),
    )

    if response.status == "incomplete":
        details = response.incomplete_details
        reason = details.reason if details else "unknown"
        raise RuntimeError(
            f"The research brief was incomplete (reason: {reason}). "
            "Try a shorter question or reduce the requested answer length."
        )

    raw_response = response.output_text
    print("Response status:", response.status)
    print("Output details:", response.incomplete_details)
    brief = raw_response.strip()

    if debug:
        print("\n--- Debug: raw model response ---")
        print(raw_response)
        print("--- Debug: citations extracted by re.findall() ---")
        print(re.findall(r"\[[^\]]+\]", brief))
        print("--- Debug: allowed citations set ---")
        print({f"[{chunk.filename}, p. {chunk.page_number}]" for chunk in evidence})
    validation_result = citations_are_valid(brief, evidence)
    if debug:
        print("--- Debug: citation validation result ---")
        print(validation_result)
    if not validation_result:
        return (
            "The available evidence is insufficient to produce a safely cited answer. "
            "Please review the retrieved report pages directly."
        )
    return brief


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Answer a research question from PDF reports in the documents folder."
    )
    parser.add_argument("question", nargs="*", help="Research question to answer.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print retrieved evidence and citation-validation details.",
    )
    return parser.parse_args()


def current_revision() -> str:
    """Return the current Git revision when the program is run from a repository."""
    try:
        result = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return "unavailable"
    return result.stdout.strip() or "unavailable"


def main() -> int:
    args = parse_arguments()
    question = " ".join(args.question).strip() or input("Research question: ").strip()
    if args.debug:
        print("--- Debug: running version ---")
        print(f"research.py path: {Path(__file__).resolve()}")
        print(f"Git revision: {current_revision()}")
    if not question:
        print("Please provide a research question.", file=sys.stderr)
        return 1
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY is not set. Set it before running this program.", file=sys.stderr)
        return 1

    try:
        pages = extract_pdf_pages(DOCUMENTS_DIR)
        chunks = make_chunks(pages)
        print(f"Read {len(pages)} pages and created {len(chunks)} searchable chunks.")
        client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
        evidence = retrieve_evidence(client, question, chunks)
        if args.debug:
            print("\n--- Debug: retrieved chunks ---")
            for number, chunk in enumerate(evidence, start=1):
                print(
                    f"Chunk {number}: {chunk.filename}, page {chunk.page_number}, "
                    f"cosine similarity {chunk.similarity_score:.6f}"
                )
                print(chunk.text)
        print("\n--- Research Brief ---\n")
        returned_brief = write_brief(client, question, evidence, debug=args.debug)
        if args.debug:
            print("--- Debug: write_brief return value, first 100 characters ---")
            print(repr(returned_brief[:100]))
        print(returned_brief)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Unexpected error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
