
import re
from dataclasses import dataclass

import pytest

from research import (
    citation_is_valid,
    citation_part_is_valid,
    citations_are_valid,
)


@dataclass
class Chunk:
    filename: str
    page_number: int


@pytest.fixture
def evidence():
    return [
        Chunk("report.pdf", 3),
        Chunk("report.pdf", 5),
        Chunk("report.pdf", 7),
        Chunk("report.pdf", 8),
        Chunk("report.pdf", 9),
        Chunk("other.pdf", 2),
    ]


def test_single_page_citation_is_valid(evidence):
    assert citations_are_valid(
        "Revenue increased [report.pdf, p. 3].", evidence
    )


def test_grouped_pages_are_valid(evidence):
    assert citations_are_valid(
        "The report confirms this [report.pdf, pp. 3, 5].", evidence
    )


def test_complete_page_range_is_valid(evidence):
    assert citations_are_valid(
        "The result appears here [report.pdf, pp. 7–9].", evidence
    )


def test_range_with_missing_page_is_rejected(evidence):
    incomplete_evidence = [
        Chunk("report.pdf", 7),
        Chunk("report.pdf", 9),
    ]

    assert not citations_are_valid(
        "The result appears here [report.pdf, pp. 7–9].",
        incomplete_evidence,
    )


def test_multiple_files_are_valid(evidence):
    assert citations_are_valid(
        "Both sources support this [report.pdf, p. 3; other.pdf, p. 2].",
        evidence,
    )


def test_bracketed_non_citation_text_does_not_break_validation(evidence):
    brief = "Revenue increased [Table 1]. [report.pdf, p. 3]"
    assert citations_are_valid(brief, evidence)

def test_citation_to_unretrieved_page_is_rejected(evidence):
    brief = "Revenue increased [report.pdf, p. 99]."

    assert not citations_are_valid(brief, evidence)

def test_hallucinated_nonstandard_citation_is_rejected(evidence):
    brief = (
        "Revenue increased [report.pdf, p. 3]. "
        "The annual report confirms this [Annual Report 2023, page 15]."
    )

    assert not citations_are_valid(brief, evidence)


def test_multiple_citations_without_space_after_semicolon(evidence):
    brief = (
        "Both sources support this "
        "[report.pdf, p. 3;other.pdf, p. 2]."
    )

    assert citations_are_valid(brief, evidence)


def test_page_range_with_spaces_around_dash(evidence):
    brief = "The result appears here [report.pdf, pp. 7 – 9]."

    assert citations_are_valid(brief, evidence)

def test_same_file_multiple_pages_with_semicolon(evidence):
    brief = "The report confirms this [report.pdf, p. 3; p. 5]."

    assert citations_are_valid(brief, evidence)


def test_grouped_citation_with_normal_hyphen(evidence):
    brief = "The result appears here [report.pdf, pp. 7-9]."

    assert citations_are_valid(brief, evidence)