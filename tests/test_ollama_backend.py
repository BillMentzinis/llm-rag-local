import socket
import time

import pytest

import llm_backends
from fake_ollama import GIB, FakeOllama
from llm_backends import OllamaBackend, default_model_key, list_ollama_models, model_label, short_label
from rag_pipeline import RAGPipeline

MESSAGES = [{"role": "user", "content": "Hi"}]


@pytest.fixture
def fake():
    server = FakeOllama().start()
    yield server
    server.stop()


def _unused_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_streams_the_answer_in_pieces(fake):
    pieces = list(OllamaBackend("llama3.1:8b", host=fake.url).stream_chat(MESSAGES, {}))

    assert pieces == ["Hello", " from", " the", " fake", " Ollama", " server."]
    assert fake.requests[0]["stream"] is True
    assert fake.requests[0]["messages"] == MESSAGES


def test_generation_settings_become_ollama_options(fake):
    config = {"max_new_tokens": 7, "temperature": 0.3, "top_p": 0.8, "top_k": 20,
              "repetition_penalty": 1.2, "do_sample": True}

    list(OllamaBackend("llama3.1:8b", host=fake.url).stream_chat(MESSAGES, config))

    assert fake.requests[0]["options"] == {"num_predict": 7, "temperature": 0.3, "top_p": 0.8,
                                           "top_k": 20, "repeat_penalty": 1.2}


def test_greedy_decoding_means_temperature_zero():
    assert llm_backends.ollama_options({"do_sample": False, "temperature": 0.7})["temperature"] == 0


def test_closing_the_stream_disconnects_so_ollama_stops(fake):
    fake.words = [f"w{i}" for i in range(200)]
    fake.delay = 0.01
    stream = OllamaBackend("llama3.1:8b", host=fake.url).stream_chat(MESSAGES, {})

    assert next(stream) == "w0"
    stream.close()

    assert fake.disconnected.wait(timeout=5)
    assert fake.chunks_sent < 200


def test_nothing_is_requested_until_the_stream_is_read(fake):
    OllamaBackend("llama3.1:8b", host=fake.url).stream_chat(MESSAGES, {})
    assert fake.requests == []


def test_missing_model_says_how_to_pull_it(fake):
    with pytest.raises(RuntimeError, match="ollama pull mistral:7b"):
        list(OllamaBackend("mistral:7b", host=fake.url).stream_chat(MESSAGES, {}))


def test_error_during_the_stream_is_reported(fake):
    fake.fail_after = 2
    with pytest.raises(RuntimeError, match="Ollama error: model runner has unexpectedly stopped"):
        list(OllamaBackend("llama3.1:8b", host=fake.url).stream_chat(MESSAGES, {}))


def test_unreachable_server_says_where_it_looked():
    host = f"http://127.0.0.1:{_unused_port()}"
    with pytest.raises(RuntimeError, match=f"Can't reach Ollama at {host}"):
        list(OllamaBackend("llama3.1:8b", host=host).stream_chat(MESSAGES, {}))


def test_lists_installed_models(fake):
    fake.models = ["qwen2.5:7b", "llama3.1:8b"]
    assert list_ollama_models(fake.url) == ["llama3.1:8b", "qwen2.5:7b"]


def test_listing_models_without_ollama_is_empty_and_quick():
    start = time.time()
    assert list_ollama_models(f"http://127.0.0.1:{_unused_port()}") == []
    assert time.time() - start < 2


def test_status_shows_vram_once_the_model_is_loaded(fake):
    backend = OllamaBackend("llama3.1:8b", host=fake.url)
    assert backend.status() == "Ollama"  # not loaded yet

    list(backend.stream_chat(MESSAGES, {}))
    assert backend.status() == "Ollama · 4.5 GB VRAM"

    fake.size_vram = 0
    assert backend.status() == "Ollama · CPU"


def test_rag_prompt_reaches_ollama(fake):
    pipeline = RAGPipeline(OllamaBackend("llama3.1:8b", host=fake.url), vector_store=object(), doc_processor=object())
    chunk = {"text": "cats purr", "metadata": {"filename": "cats.txt", "chunk_index": 0}}

    result = pipeline.generate_response("why purr?", context_chunks=[chunk], generation_config={})

    assert result["response"] == "Hello from the fake Ollama server."
    prompt = fake.requests[0]["messages"][-1]["content"]
    assert "[Source: cats.txt, Chunk 1]" in prompt and "why purr?" in prompt


@pytest.mark.parametrize("configured, ollama_models, cuda, expected", [
    ("transformers:meta-llama/Llama-3.1-8B-Instruct", ["llama3.1:8b"], False, "ollama:llama3.1:8b"),
    ("transformers:meta-llama/Llama-3.1-8B-Instruct", ["llama3.1:8b"], True,
     "transformers:meta-llama/Llama-3.1-8B-Instruct"),
    ("transformers:meta-llama/Llama-3.1-8B-Instruct", [], False, "transformers:meta-llama/Llama-3.1-8B-Instruct"),
    ("ollama:qwen2.5:7b", ["llama3.1:8b"], False, "ollama:qwen2.5:7b"),
    # Without a prefix: a known Hugging Face ID means that model, anything else Ollama
    ("llama3.1:8b", [], True, "ollama:llama3.1:8b"),
    ("meta-llama/Llama-3.2-3B-Instruct", [], True, "transformers:meta-llama/Llama-3.2-3B-Instruct"),
])
def test_default_model_prefers_ollama_without_a_gpu(monkeypatch, configured, ollama_models, cuda, expected):
    monkeypatch.setattr(llm_backends, "cuda_available", lambda: cuda)
    assert default_model_key(configured, ollama_models) == expected


def test_labels():
    assert model_label("ollama:llama3.1:8b") == "llama3.1:8b · Ollama"
    assert model_label("transformers:meta-llama/Llama-3.1-8B-Instruct") == "Llama 3.1 8B (~4GB) · HF"
    assert short_label("transformers:meta-llama/Llama-3.1-8B-Instruct") == "Llama 3.1 8B"
    assert short_label("ollama:llama3.1:8b") == "llama3.1:8b"


def test_hugging_face_model_without_a_gpu_fails_before_downloading(monkeypatch):
    import sys
    monkeypatch.setattr(llm_backends, "cuda_available", lambda: False)
    monkeypatch.setitem(sys.modules, "transformers", None)  # any import attempt would raise ImportError

    with pytest.raises(RuntimeError, match="need an NVIDIA GPU"):
        llm_backends.TransformersBackend("meta-llama/Llama-3.1-8B-Instruct")
