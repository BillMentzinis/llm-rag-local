import json
import os

import pytest

import evaluate_retrieval as ev
from document_processor import DocumentProcessor
from fake_embedder import FakeEmbedder
from llm_backends import LLMBackend

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(REPO, "examples", "eval", "questions.json")

CATS = ("Cats purr when they are content. A cat sleeps for around fifteen hours a day, "
        "mostly in warm spots near a window.")
BOATS = ("Sailing boats tack into the wind by zigzagging. The keel stops the boat from "
         "drifting sideways while the sails pull it forward.")


@pytest.fixture
def folder(tmp_path):
    (tmp_path / "cats.txt").write_text(CATS, encoding="utf-8")
    (tmp_path / "boats.md").write_text(BOATS, encoding="utf-8")
    (tmp_path / "notes.json").write_text("{}", encoding="utf-8")  # not a supported document
    return tmp_path


def write_questions(folder, questions):
    path = folder / "questions.json"
    path.write_text(json.dumps({"questions": questions}), encoding="utf-8")
    return str(path)


def test_normalizing_ignores_case_punctuation_and_line_breaks():
    assert ev.normalize("Copied to an Off-\nsite  bucket.") == ev.normalize("copied to an off-site bucket")
    assert ev.normalize("ﬁle") == "file"  # PDF ligatures


def test_the_example_questions_all_quote_their_documents():
    folder = os.path.dirname(EXAMPLE)
    documents = ev.load_documents(folder, DocumentProcessor())
    with open(EXAMPLE, encoding="utf-8") as f:
        questions = json.load(f)["questions"]

    assert len(questions) >= 20
    for q in questions:
        assert documents[q["document"]].find(q["quote"]), q


def document_of_words(count, chunk_size):
    text = " ".join(f"word{i}" for i in range(count))
    return ev.Document("doc.txt", text, DocumentProcessor(chunk_size=chunk_size, chunk_overlap=0).chunk_text(text))


def texts_holding(document, quote):
    return [document.chunks[i]["text"] for i in document.chunks_holding(document.find(quote))]


def test_the_chunk_with_most_of_a_quote_holds_it():
    document = document_of_words(60, 10)
    assert document.chunks[1]["text"] == "word6 word7 word8 word9 word10 word11"

    assert texts_holding(document, "word7 word8 word9") == ["word6 word7 word8 word9 word10 word11"]
    assert texts_holding(document, "word9 word10 word11 word12") == ["word6 word7 word8 word9 word10 word11"]


def test_a_quote_split_evenly_is_held_by_both_chunks():
    document = document_of_words(60, 10)
    assert texts_holding(document, "word10 word11 word12 word13") == [
        "word6 word7 word8 word9 word10 word11", "word12 word13 word14 word15 word16"]


def test_a_long_quote_is_held_by_chunks_that_are_mostly_quote():
    document = document_of_words(60, 10)
    assert texts_holding(document, " ".join(f"word{i}" for i in range(5, 30))) == [
        c["text"] for c in document.chunks[1:5]] + ["word27 word28 word29 word30 word31"]


def test_evaluation_ranks_the_right_passage(folder):
    path = write_questions(folder, [
        {"question": "how many hours does a cat sleep", "document": "cats.txt", "quote": "sleeps for around fifteen hours"},
        {"question": "how do sailing boats tack into the wind", "document": "boats.md", "quote": "The keel stops the boat"},
        {"question": "what do dogs eat", "document": "cats.txt", "quote": "dogs eat biscuits"},
        {"question": "where is the map", "document": "map.pdf", "quote": "the map"},
    ])

    report = ev.evaluate(path, DocumentProcessor(), embedding_model=FakeEmbedder())

    assert report["documents"] == 2
    assert [(r["rank"], r["found"]) for r in report["results"]] == [(1, True), (1, True)]
    assert report["summary"]["found"] == 2 and report["summary"]["mrr"] == 1.0
    assert any("the quote isn't in cats.txt" in p for p in report["problems"])
    assert any("no document called map.pdf" in p for p in report["problems"])
    text = ev.format_report(report)
    assert "**Found by the app: 2/2 (100%)**" in text and "### Questions skipped" in text


def test_a_passage_under_the_similarity_threshold_isnt_found(folder):
    path = write_questions(folder, [
        {"question": "cats purr", "document": "cats.txt", "quote": "Cats purr when they are content"},
    ])

    report = ev.evaluate(path, DocumentProcessor(), embedding_model=FakeEmbedder(), min_similarity=0.99)

    [result] = report["results"]
    assert result["rank"] == 1 and not result["found"]
    assert report["summary"]["below_threshold"] == 1
    assert "### Not found by the app" in ev.format_report(report)


def test_unanswerable_questions_and_the_threshold_table(folder):
    path = folder / "questions.json"
    path.write_text(json.dumps({
        "questions": [{"question": "how many hours does a cat sleep", "document": "cats.txt",
                       "quote": "sleeps for around fifteen hours"}],
        "unanswerable": ["cats purr content", "zebra quantum"],
    }), encoding="utf-8")

    report = ev.evaluate(str(path), DocumentProcessor(), embedding_model=FakeEmbedder(), min_similarity=0.3)

    near, far = report["unanswerable"]
    assert near["best_similarity"] > 0.3 and near["given_excerpts"]  # shares words with cats.txt
    assert far["best_similarity"] < near["best_similarity"]
    rows = {row["threshold"]: row for row in report["thresholds"]}
    # The stand-in embedder's vectors are word counts, so no similarity is below 0
    assert rows[0.0]["found"] == 1 and rows[0.0]["unanswerable_given_excerpts"] == 2
    assert rows[0.3]["unanswerable_given_excerpts"] == 1 + (far["best_similarity"] >= 0.3)
    for low, high in zip(report["thresholds"], report["thresholds"][1:]):
        assert high["found"] <= low["found"]
        assert high["unanswerable_given_excerpts"] <= low["unanswerable_given_excerpts"]

    text = ev.format_report(report)
    assert "### Questions the documents can't answer" in text
    given = 1 + far["given_excerpts"]
    assert f"The app would pass excerpts to the model for {given} of 2" in text
    assert "| 0.30 (current) |" in text


def test_the_example_set_has_unanswerable_questions():
    with open(EXAMPLE, encoding="utf-8") as f:
        assert len(json.load(f)["unanswerable"]) >= 5


class ScriptedBackend(LLMBackend):
    """Answers with the given replies in turn, and records the prompts."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def stream_chat(self, messages, generation_config):
        self.prompts.append(messages[-1]["content"])
        yield self.replies.pop(0)


def test_drafts_keep_only_questions_that_quote_their_excerpt(folder):
    # Each document is one chunk; they're taken in turn, boats.md first
    llm = ScriptedBackend([
        'Sure! {"question": "Why do boats zigzag?", "quote": "Sailing boats tack into the wind by zigzagging."}',
        '{"question": "How long do cats sleep?", "quote": "Cats sleep all day long."}',  # not in the excerpt
    ])

    drafts, rejected = ev.draft_questions(str(folder), llm, count=2)

    assert drafts == [{"question": "Why do boats zigzag?", "document": "boats.md",
                       "quote": "Sailing boats tack into the wind by zigzagging."}]
    assert rejected == 1
    assert "document called boats.md" in llm.prompts[0] and "Cats purr" in llm.prompts[1]


def test_replies_that_arent_json_are_dropped(folder):
    llm = ScriptedBackend(["I can't help with that.", '{"question": "Do cats purr?", "quote": "cats PURR when they are content"}'])

    drafts, rejected = ev.draft_questions(str(folder), llm, count=5)

    assert [d["document"] for d in drafts] == ["cats.txt"] and rejected == 1


def test_drafting_stops_when_the_chunks_run_out(folder):
    drafts, rejected = ev.draft_questions(str(folder), ScriptedBackend(["no"] * 10), count=5)
    assert drafts == [] and rejected == 2  # one chunk per document


def test_command_line_run_and_draft(folder, monkeypatch, capsys):
    import llm_backends
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", lambda name: FakeEmbedder())
    path = write_questions(folder, [
        {"question": "how many hours does a cat sleep", "document": "cats.txt", "quote": "fifteen hours a day"},
    ])
    out = folder / "report.json"

    assert ev.main(["run", path, "--chunk-size", "100", "--chunk-overlap", "10", "--json", str(out)]) == 0
    assert "chunks 100/10" in capsys.readouterr().out
    assert json.loads(out.read_text(encoding="utf-8"))["summary"]["found"] == 1

    reply = '{"question": "Do cats purr?", "quote": "Cats purr when they are content."}'
    loaded = []
    monkeypatch.setattr(llm_backends, "list_ollama_models", lambda host=None: ["llama3.1:8b"])
    monkeypatch.setattr(llm_backends, "load_backend", lambda key: loaded.append(key) or ScriptedBackend([reply] * 5))
    assert ev.main(["draft", str(folder), "--count", "1", "--model", "llama3.1:8b"]) == 0
    draft = json.loads((folder / "questions-draft.json").read_text(encoding="utf-8"))
    assert loaded == ["ollama:llama3.1:8b"] and len(draft["questions"]) == 1

    with pytest.raises(SystemExit):  # won't overwrite an earlier draft
        ev.main(["draft", str(folder), "--count", "1"])
