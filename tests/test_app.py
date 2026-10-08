"""
Tests that run the real streamlit_app.py headlessly (Streamlit's AppTest)
against a fake Ollama server and the stand-in embedder. No browser needed.
"""

import os

import pytest
from streamlit.testing.v1 import AppTest

from fake_embedder import FakeEmbedder
from fake_ollama import FakeOllama

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(REPO, "streamlit_app.py")

CATS = b"cats purr and chase mice around the barn"
DOGS = b"dogs bark loudly and chase cats in the yard"


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """A fake Ollama the app starts on, with its data in a temp directory."""
    import chat_manager
    import config
    import sentence_transformers
    import streamlit as st

    server = FakeOllama(words=["An", "answer."]).start()
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", lambda name: FakeEmbedder())
    monkeypatch.setitem(config.PATHS, "chroma_db", str(tmp_path / "chroma_db"))
    monkeypatch.setitem(config.PATHS, "chats", str(tmp_path / "chats"))
    monkeypatch.setattr(chat_manager, "CHATS_DIR", str(tmp_path / "chats"))
    monkeypatch.setitem(config.OLLAMA_CONFIG, "host", server.url)
    monkeypatch.setattr(config, "DEFAULT_MODEL", "ollama:llama3.1:8b")
    st.cache_resource.clear()
    st.cache_data.clear()
    yield server
    st.cache_resource.clear()
    st.cache_data.clear()
    server.stop()


def _upload(files):
    """AppTest script: send (name, bytes) pairs through the app's real upload handler."""
    import streamlit as st
    import streamlit_app as app

    class Upload:
        def __init__(self, name, data):
            self.name, self._data = name, data

        def getbuffer(self):
            return memoryview(self._data)

    if not st.session_state.get("done"):
        st.session_state.done = True
        app.initialize_session_state()
        app.process_uploaded_files([Upload(name, data) for name, data in files], app.get_pipeline())


def upload(files):
    at = AppTest.from_function(_upload, args=(files,), default_timeout=60).run()
    assert not at.exception, at.exception
    return at


def start_app():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    return at


def ask(at, question):
    at.chat_input[0].set_value(question).run()
    assert not at.exception, at.exception
    return at


def indexed_documents():
    from vector_store_manager import VectorStoreManager
    store = VectorStoreManager(embedding_model=FakeEmbedder())
    return {d["filename"]: store.collection.get(where={"filename": d["filename"]})["documents"]
            for d in store.list_documents()}


def test_the_model_gets_system_prompt_complete_turns_and_the_question(fake):
    import config

    at = start_app()
    assert at.session_state["selected_model"] == "ollama:llama3.1:8b"
    for i in range(1, 6):
        ask(at, f"question {i}")

    request = fake.requests[-1]
    assert request["messages"][0] == {"role": "system", "content": config.SYSTEM_PROMPT}
    assert [m["content"] for m in request["messages"][1:]] == [
        "question 2", "An answer.", "question 3", "An answer.", "question 4", "An answer.", "question 5"]
    assert request["options"]["num_ctx"] == config.OLLAMA_CONFIG["num_ctx"]
    assert request["options"]["num_predict"] == at.session_state["max_tokens"]


def test_a_failed_answer_is_left_out_of_later_history(fake):
    at = start_app()
    for i in range(1, 4):
        ask(at, f"question {i}")
    fake.fail_after = 0
    ask(at, "question 4")
    assert at.session_state["messages"][-1]["content"] == ""
    assert "error" in at.session_state["messages"][-1]

    fake.fail_after = None
    ask(at, "question 5")

    contents = [m["content"] for m in fake.requests[-1]["messages"][1:]]
    assert contents == ["question 1", "An answer.", "question 2", "An answer.",
                        "question 3", "An answer.", "question 5"]


def test_regenerate_resends_the_same_prompt_and_replaces_the_answer(fake):
    at = start_app()
    ask(at, "question 1")
    ask(at, "question 2")
    before = len(fake.requests)
    count = len(at.session_state["messages"])

    at.button(key="regenerate").click().run()

    assert len(fake.requests) == before + 1
    assert fake.requests[-1]["messages"] == fake.requests[-2]["messages"]
    assert len(at.session_state["messages"]) == count


def test_uploads_are_indexed_under_their_base_name_without_leaving_files(fake, tmp_path, monkeypatch):
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    upload([("cats.txt", CATS), ("../../dogs.txt", DOGS)])

    assert sorted(indexed_documents()) == ["cats.txt", "dogs.txt"]
    assert list(workdir.iterdir()) == []


def test_reuploading_replaces_changed_files_and_skips_unchanged_ones(fake):
    upload([("cats.txt", CATS), ("dogs.txt", DOGS)])
    at = upload([("cats.txt", CATS), ("dogs.txt", b"dogs now nap all day")])

    results = {r["filename"]: r for r in at.session_state["ingest_results"]}
    assert results["cats.txt"]["skipped"] and "unchanged" in results["cats.txt"]["message"]
    assert results["dogs.txt"]["replaced"]
    documents = indexed_documents()
    assert documents["dogs.txt"] == ["dogs now nap all day"]
    assert documents["cats.txt"] == [CATS.decode()]


def test_focus_limits_retrieval_to_one_document(fake):
    upload([("cats.txt", CATS), ("dogs.txt", DOGS)])
    at = start_app()
    at.sidebar.toggle[0].set_value(True).run()
    [focus] = [s for s in at.sidebar.selectbox if s.label == "Focus on document"]
    focus.select("dogs.txt").run()

    ask(at, "which animals chase cats")

    question = fake.requests[-1]["messages"][-1]["content"]
    assert "[Source: dogs.txt" in question and "cats.txt" not in question
    assert [s["metadata"]["filename"] for s in at.session_state["messages"][-1]["sources"]] == ["dogs.txt"]


def test_an_answer_without_relevant_context_says_so(fake):
    upload([("cats.txt", CATS), ("dogs.txt", DOGS)])
    at = start_app()
    at.sidebar.toggle[0].set_value(True).run()

    ask(at, "telescope")  # shares no words with either document

    assert fake.requests[-1]["messages"][-1]["content"] == "telescope"
    note = "No relevant document context found"
    assert any(note in c.value for c in at.caption)
    at.run()  # the note stays on the saved message
    assert any(note in c.value for c in at.caption)


def test_an_interrupted_answer_is_kept_and_marked_stopped(fake):
    at = start_app()
    at.session_state["messages"].append({"role": "user", "content": "interrupted question"})
    at.session_state["pending_response"] = {"role": "assistant", "content": "partial ans", "sources": []}

    at.run()

    last = at.session_state["messages"][-1]
    assert last["content"] == "partial ans" and last["stopped"]
    assert "pending_response" not in at.session_state
    assert any("Stopped before" in c.value for c in at.caption)


def test_chats_with_the_same_first_question_are_saved_separately(fake):
    import chat_manager

    def new_chat():  # saves the current chat first
        next(b for b in at.sidebar.button if b.label == "New chat").click().run()

    at = start_app()
    ask(at, "hello")
    new_chat()
    ask(at, "hello")
    ask(at, "and again")
    new_chat()

    chats = {c["name"]: c["message_count"] for c in chat_manager.list_chats()}
    assert chats == {"hello": 2, "hello (2)": 4}


def test_a_summary_request_summarizes_the_whole_document_and_regenerates_as_one(fake):
    upload([("cats.txt", CATS), ("dogs.txt", DOGS)])
    at = start_app()
    # As the Documents page's Summarize button leaves it
    at.session_state["messages"] = [{"role": "user", "content": "Summarize dogs.txt", "summarize": "dogs.txt"}]
    at.session_state["summary_requested"] = True
    at.run()
    assert not at.exception, at.exception

    prompt = fake.requests[-1]["messages"][-1]["content"]
    assert "full text of dogs.txt" in prompt and DOGS.decode() in prompt and CATS.decode() not in prompt
    summary = at.session_state["messages"][-1]
    assert summary["content"] == "An answer." and summary["summary_of"] == "dogs.txt"
    assert any("Summary of all of dogs.txt" in c.value for c in at.caption)

    at.button(key="regenerate").click().run()
    assert len(fake.requests) == 2 and fake.requests[-1]["messages"] == fake.requests[-2]["messages"]
    assert at.session_state["messages"][-1]["summary_of"] == "dogs.txt"
