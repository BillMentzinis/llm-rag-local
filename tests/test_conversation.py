import pytest

from config import SYSTEM_PROMPT
from llm_backends import LLMBackend, TransformersBackend
from rag_pipeline import RAGPipeline, select_history


def user(text):
    return {"role": "user", "content": text}


def assistant(text):
    return {"role": "assistant", "content": text}


class RecordingBackend(LLMBackend):
    """Records the messages it's asked to answer; 1 token per 4 characters."""

    def __init__(self, window=4096):
        self.window = window
        self.messages = None

    def context_window(self):
        return self.window

    def stream_chat(self, messages, generation_config):
        self.messages = messages
        yield "ok"


def pipeline(window=4096):
    return RAGPipeline(RecordingBackend(window), vector_store=object(), doc_processor=object())


def chunk(i, size=400):
    return {"text": f"excerpt {i} " + "x" * size, "metadata": {"filename": f"doc{i}.txt", "chunk_index": 0}}


# --- history selection ------------------------------------------------------

def test_history_keeps_the_last_complete_pairs():
    history = [user("q1"), assistant("a1"), user("q2"), assistant("a2"), user("q3"), assistant("a3")]
    assert select_history(history, 2) == [[user("q2"), assistant("a2")], [user("q3"), assistant("a3")]]


def test_history_skips_answers_without_text_and_their_questions():
    # e.g. an answer that failed, or was stopped before any output
    history = [user("q1"), assistant("a1"), user("q2"), assistant(""), user("q3"), assistant("a3")]
    assert select_history(history, 3) == [[user("q1"), assistant("a1")], [user("q3"), assistant("a3")]]


def test_history_drops_unpaired_messages():
    history = [assistant("orphan answer"), user("q1"), user("q2"), assistant("a2")]
    assert select_history(history, 3) == [[user("q2"), assistant("a2")]]


def test_history_strips_extra_fields_and_can_be_empty():
    history = [dict(user("q"), sources=[1]), dict(assistant("a"), stopped=True)]
    assert select_history(history, 3) == [[user("q"), assistant("a")]]
    assert select_history(history, 0) == []
    assert select_history(None, 3) == []


# --- messages sent to the model --------------------------------------------

def test_system_prompt_first_then_alternating_turns_then_the_question():
    p = pipeline()
    history = [assistant("stray"), user("q1"), assistant("a1"), user("q2"), assistant("a2")]

    list(p.stream_response("q3", history=history)["stream"])

    roles = [m["role"] for m in p.llm.messages]
    assert p.llm.messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    assert p.llm.messages[-1]["content"] == "q3"


def test_at_most_the_configured_number_of_turns_is_sent():
    p = pipeline()
    history = [m for i in range(10) for m in (user(f"q{i}"), assistant(f"a{i}"))]

    result = p.stream_response("next", history=history)
    list(result["stream"])

    assert result["history_turns"] == 3
    assert [m["content"] for m in p.llm.messages[1:-1]] == ["q7", "a7", "q8", "a8", "q9", "a9"]


# --- fitting the context window ---------------------------------------------

def test_nothing_is_trimmed_when_it_fits():
    result = pipeline().stream_response("q", context_chunks=[chunk(1)], history=[user("q0"), assistant("a0")],
                                        generation_config={"max_new_tokens": 512})
    assert not result["trimmed_to_fit"] and result["history_turns"] == 1 and result["num_sources"] == 1


def test_oldest_turns_are_dropped_first_to_fit():
    # Each turn is ~250 tokens; the window leaves room for about two
    history = [m for i in range(3) for m in (user(f"q{i} " + "y" * 500), assistant(f"a{i} " + "z" * 500))]
    p = pipeline(window=512 + 64 + 700)

    result = p.stream_response("latest", context_chunks=[chunk(1)], history=history,
                               generation_config={"max_new_tokens": 512})
    list(result["stream"])

    assert result["trimmed_to_fit"]
    assert result["num_sources"] == 1  # excerpts are kept while turns can still go
    kept = [m["content"][:2] for m in p.llm.messages if m["role"] == "user"]
    assert kept[-1] == "Re"  # the question, with its excerpts
    assert "q0" not in kept and len(kept) - 1 == result["history_turns"] < 3


def test_least_relevant_excerpts_go_after_the_history():
    chunks = [chunk(i) for i in range(1, 6)]  # ~100 tokens each, ranked best first
    p = pipeline(window=512 + 64 + 330)

    result = p.stream_response("q", context_chunks=chunks, history=[user("q0"), assistant("a0")],
                               generation_config={"max_new_tokens": 512})
    list(result["stream"])

    assert result["trimmed_to_fit"] and result["history_turns"] == 0
    assert 0 < result["num_sources"] < 5
    assert [c["metadata"]["filename"] for c in result["sources"]] == \
        [f"doc{i}.txt" for i in range(1, result["num_sources"] + 1)]
    assert "doc5.txt" not in p.llm.messages[-1]["content"]


def test_the_question_is_sent_even_if_nothing_else_fits():
    p = pipeline(window=512 + 64 + 10)
    result = p.stream_response("a long question " * 20, context_chunks=[chunk(1)],
                               history=[user("q0"), assistant("a0")], generation_config={"max_new_tokens": 512})
    list(result["stream"])

    assert result["num_sources"] == 0 and result["history_turns"] == 0
    assert p.llm.messages[-1]["content"] == "a long question " * 20


def test_the_answer_reserve_follows_max_new_tokens():
    history = [user("q0 " + "y" * 400), assistant("a0 " + "z" * 400)]
    small = pipeline(window=1000).stream_response("q", history=history, generation_config={"max_new_tokens": 100})
    large = pipeline(window=1000).stream_response("q", history=history, generation_config={"max_new_tokens": 900})
    assert small["history_turns"] == 1 and large["history_turns"] == 0


# --- chat templates without a system role -----------------------------------

class NoSystemTokenizer:
    """Mimics a chat template that raises on a system message (e.g. Gemma's)."""

    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        if any(m["role"] == "system" for m in messages):
            raise ValueError("System role not supported")
        self.rendered = messages
        return "prompt"


def test_system_prompt_is_folded_into_the_first_question_when_unsupported():
    tokenizer = NoSystemTokenizer()
    backend = TransformersBackend("meta-llama/Llama-3.1-8B-Instruct", model=object(), tokenizer=tokenizer)

    backend._format_prompt([{"role": "system", "content": "Be brief."}, user("q1"), assistant("a1"), user("q2")])

    assert tokenizer.rendered == [user("Be brief.\n\nq1"), assistant("a1"), user("q2")]


def test_other_template_errors_are_not_hidden():
    backend = TransformersBackend("meta-llama/Llama-3.1-8B-Instruct", model=object(), tokenizer=NoSystemTokenizer())
    with pytest.raises(ValueError):
        backend._format_prompt([{"role": "system", "content": "x"}])  # nothing to fold it into


def test_transformers_context_window_uses_the_model_limit():
    class Config:
        max_position_embeddings = 131072

    class Model:
        config = Config()

    class Tokenizer:
        model_max_length = int(1e30)  # "no limit" placeholder, ignored

    backend = TransformersBackend("meta-llama/Llama-3.1-8B-Instruct", model=Model(), tokenizer=Tokenizer())
    assert backend.context_window() == 131072
