"""Offline unit tests for the PDF RAG CLI's retrieval and prompt construction."""
from pdf_search import build_augmented_prompt, load_pages, retrieve_chunks


def test_retrieval_ranks_relevant_page_first():
    pages = [
        (1, "A recipe explains how to bake bread with flour and yeast."),
        (2, "Recursion is a programming technique where a function calls itself."),
        (3, "The solar system contains planets orbiting the Sun."),
    ]
    chunks = retrieve_chunks(pages, "What is recursion in programming?", top_k=2)
    assert chunks
    assert chunks[0]["page"] == 2
    assert "function calls itself" in chunks[0]["text"]


def test_augmented_prompt_contains_query_context_and_page():
    chunks = [{"page": 66, "text": "A recursive function calls itself.", "score": 0.42}]
    prompt = build_augmented_prompt("What is recursion?", chunks)
    assert "What is recursion?" in prompt
    assert "[Page 66]" in prompt
    assert "A recursive function calls itself." in prompt
    assert "cite relevant page numbers" in prompt.lower()


def test_load_pages_from_included_pdf():
    pages = load_pages("book.pdf")
    assert pages
    assert all(isinstance(number, int) and text.strip() for number, text in pages)
