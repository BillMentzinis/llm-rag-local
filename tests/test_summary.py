import re

import pytest

from config import SUMMARY_CONFIG
from conftest import make_chunks
from document_processor import DocumentProcessor
from llm_backends import LLMBackend
from rag_pipeline import RAGPipeline, join_chunks, split_into_parts


class SummaryBackend(LLMBackend):
    """Answers every prompt with a fixed text, and records the prompts and closed streams."""

    def __init__(self, window=4096, reply="- a point"):
        self.window = window
        self.reply = reply
        self.prompts = []
        self.closed = 0

    def context_window(self):
        return self.window

    def stream_chat(self, messages, generation_config):
        self.prompts.append(messages[-1]["content"])
        try:
            for word in self.reply.split(" "):
                yield word + " "
        finally:
            self.closed += 1


def pipeline_with(store, backend):
    return RAGPipeline(backend, vector_store=store, doc_processor=DocumentProcessor())


def index(store, filename, texts, pages=None):
    chunks = make_chunks(filename, texts)
    for chunk, page in zip(chunks, pages or []):
        chunk["page_start"] = chunk["page_end"] = page
    store.add_documents(chunks)


def paragraphs(count, words=60):
    return [f"Paragraph {i}: " + " ".join(f"word{i}x{j}" for j in range(words)) for i in range(count)]


# --- joining chunks back up -------------------------------------------------------

def test_joining_chunks_drops_the_repeated_overlap():
    first = "The backup runs every night at two. Snapshots go to the USB drive."
    second = "Snapshots go to the USB drive. On Sundays they are copied off-site."
    assert join_chunks([first, second]) == (
        "The backup runs every night at two. Snapshots go to the USB drive. On Sundays they are copied off-site.")


def test_chunks_without_overlap_are_joined_as_paragraphs():
    assert join_chunks(["First part.", "Second part."]) == "First part.\n\nSecond part."
    assert join_chunks([]) == ""


def test_joined_chunks_hold_the_whole_text_once():
    text = "\n\n".join(paragraphs(12, words=30))
    chunks = DocumentProcessor(chunk_size=100, chunk_overlap=20).chunk_text(text)
    joined = join_chunks([c["text"] for c in chunks])

    assert len(chunks) > 5
    for i in range(12):
        for j in range(30):
            assert len(re.findall(rf"\bword{i}x{j}\b", joined)) == 1


def test_parts_stay_within_the_token_budget_and_know_their_pages():
    chunks = [{"text": "x" * 400, "metadata": {"page_start": 1 + i // 2, "page_end": 1 + i // 2}} for i in range(7)]

    parts = split_into_parts(chunks, max_tokens=300, count_tokens=lambda t: len(t) // 4)

    assert [p["pages"] for p in parts] == [(1, 2), (2, 3), (4, 4)]
    assert all(len(p["text"]) // 4 <= 300 + 10 for p in parts)


# --- summaries -------------------------------------------------------------------

def test_a_short_document_is_summarized_in_one_go(store):
    index(store, "notes.txt", ["Cats purr when content.", "Dogs bark at strangers."])
    backend = SummaryBackend(reply="Notes about cats and dogs.")

    result = pipeline_with(store, backend).stream_summary("notes.txt", {"max_new_tokens": 200})

    assert result["parts"] == 1 and result["pages"] is None
    assert "".join(result["stream"]).strip() == "Notes about cats and dogs."
    [prompt] = backend.prompts
    assert "full text of notes.txt" in prompt
    assert "Cats purr when content." in prompt and "Dogs bark at strangers." in prompt


def test_a_long_document_is_summarized_part_by_part_then_combined(store):
    texts = paragraphs(40)  # ~40 * 120 tokens, more than one part
    index(store, "report.pdf", texts, pages=[1 + i // 4 for i in range(40)])
    backend = SummaryBackend()
    progress = []

    result = pipeline_with(store, backend).stream_summary(
        "report.pdf", {"max_new_tokens": 300}, on_progress=lambda done, label: progress.append((done, label)))
    summary = "".join(result["stream"])

    parts = result["parts"]
    assert parts > 1 and result["pages"] == (1, 10)
    assert summary.strip() == "- a point"
    part_prompts, final = backend.prompts[:-1], backend.prompts[-1]
    assert len(part_prompts) == parts
    assert part_prompts[0].startswith(f"Here is part 1 of {parts} of report.pdf (pages 1-")
    # Every paragraph is read, in order, exactly once
    read = "\n".join(part_prompts)
    assert [read.index(f"Paragraph {i}:") for i in range(40)] == sorted(read.index(f"Paragraph {i}:") for i in range(40))
    assert all(read.count(f"Paragraph {i}:") == 1 for i in range(40))
    assert "summaries of consecutive parts of report.pdf" in final and final.count("Part ") == parts
    # Progress only moves forward and ends with the summary being written
    fractions = [done for done, _ in progress]
    assert fractions == sorted(fractions) and progress[-1] == (1.0, "Writing the summary")
    assert progress[0][1].startswith(f"Reading part 1 of {parts}")


def test_part_summaries_that_dont_fit_together_are_condensed_first(store):
    index(store, "long.txt", paragraphs(60))
    # Long part summaries, and a window that only fits a few of them at once
    backend = SummaryBackend(window=1600, reply=" ".join(["detail"] * 150))

    result = pipeline_with(store, backend).stream_summary("long.txt", {"max_new_tokens": 200})
    "".join(result["stream"])

    condense = [p for p in backend.prompts if "one shorter list of bullet points" in p]
    assert condense, "expected the part summaries to be condensed"
    final = backend.prompts[-1]
    assert "Combine them into one summary of the whole document" in final
    assert len(final) // 4 <= 1600


def test_a_context_window_too_small_to_summarize_is_reported(store):
    index(store, "notes.txt", ["Some text."])
    with pytest.raises(RuntimeError, match="context window is too small"):
        pipeline_with(store, SummaryBackend(window=900)).stream_summary("notes.txt", {"max_new_tokens": 512})


def test_summarizing_a_document_that_isnt_indexed_fails(store):
    with pytest.raises(ValueError, match="missing.pdf isn't in the document index"):
        pipeline_with(store, SummaryBackend()).stream_summary("missing.pdf")


def test_interrupting_while_reading_the_parts_stops_generation(store):
    index(store, "report.txt", paragraphs(40))
    backend = SummaryBackend(reply=" ".join(["word"] * 50))

    class Interrupted(Exception):
        pass

    def on_progress(done, label):
        if done > 0.1:  # e.g. the Stop button, which Streamlit raises in the script's next UI update
            raise Interrupted

    result = pipeline_with(store, backend).stream_summary("report.txt", {"max_new_tokens": 200}, on_progress)
    with pytest.raises(Interrupted):
        next(result["stream"])

    assert backend.closed == len(backend.prompts) >= 1  # the model was told to stop


def test_summary_settings_leave_room_for_the_text():
    assert SUMMARY_CONFIG["part_tokens"] > 4 * SUMMARY_CONFIG["part_summary_tokens"]
