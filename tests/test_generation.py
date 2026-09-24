import threading
import time

import pytest
import torch

from rag_pipeline import RAGPipeline

WORDS = [f"w{i}" for i in range(50)]


class FakeTokenizer:
    """Token id i decodes to WORDS[i]; the prompt is a fixed run of ids."""

    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.last_messages = messages
        return messages[-1]["content"]

    def __call__(self, text, return_tensors="pt"):
        from transformers import BatchEncoding
        return BatchEncoding({"input_ids": torch.zeros((1, 4), dtype=torch.long)})

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=True, **kwargs):
        return "".join(WORDS[i] + " " for i in ids)


class FakeModel:
    """Mimics generate(): feeds the streamer one token at a time and honours stopping criteria."""

    device = "cpu"

    def __init__(self, delay=0.0, fail_after=None):
        self.delay = delay
        self.fail_after = fail_after
        self.generated = 0
        self.finished = threading.Event()

    def generate(self, input_ids, streamer, stopping_criteria, max_new_tokens=20, **kwargs):
        try:
            streamer.put(input_ids)  # the prompt, which the streamer skips
            ids = input_ids
            for i in range(1, max_new_tokens + 1):
                if self.fail_after is not None and i > self.fail_after:
                    raise RuntimeError("CUDA out of memory")
                time.sleep(self.delay)
                token = torch.tensor([[i]])
                ids = torch.cat([ids, token], dim=1)
                self.generated += 1
                streamer.put(token[0])
                if stopping_criteria(ids, None).all():
                    break
            streamer.end()
            return ids
        finally:
            self.finished.set()


def _pipeline(model):
    return RAGPipeline(model=model, tokenizer=FakeTokenizer(), vector_store=object(), doc_processor=object())


def test_stream_yields_text_incrementally():
    result = _pipeline(FakeModel()).stream_response("hi", generation_config={"max_new_tokens": 5})

    pieces = list(result["stream"])

    assert len(pieces) == 5
    assert "".join(pieces) == "w1 w2 w3 w4 w5 "
    assert result["prompt_tokens"] == 4


def test_generate_response_returns_the_full_text():
    result = _pipeline(FakeModel()).generate_response("hi", generation_config={"max_new_tokens": 3})

    assert result["response"] == "w1 w2 w3 "
    assert "stream" not in result


def test_closing_the_stream_stops_generation():
    model = FakeModel(delay=0.01)
    stream = _pipeline(model).stream_response("hi", generation_config={"max_new_tokens": 40})["stream"]

    assert next(stream) == "w1 "
    stream.close()

    assert model.finished.is_set()  # close() waits for the generation thread
    assert model.generated < 40


def test_generation_errors_reach_the_reader():
    stream = _pipeline(FakeModel(fail_after=2)).stream_response(
        "hi", generation_config={"max_new_tokens": 10})["stream"]

    with pytest.raises(RuntimeError, match="out of memory"):
        list(stream)


def test_generation_starts_only_when_the_stream_is_read():
    model = FakeModel()
    _pipeline(model).stream_response("hi", generation_config={"max_new_tokens": 3})

    assert model.generated == 0


def test_rag_context_is_placed_in_the_prompt():
    tokenizer_pipeline = _pipeline(FakeModel())
    chunk = {"text": "cats purr", "metadata": {"filename": "cats.txt", "chunk_index": 0}}

    result = tokenizer_pipeline.stream_response("why purr?", context_chunks=[chunk],
                                                generation_config={"max_new_tokens": 1})
    list(result["stream"])

    prompt = tokenizer_pipeline.tokenizer.last_messages[-1]["content"]
    assert "[Source: cats.txt, Chunk 1]" in prompt and "why purr?" in prompt
    assert result["rag_enabled"] and result["num_sources"] == 1
