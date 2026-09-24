"""
LLM backends: where answers are generated.

- TransformersBackend runs a Hugging Face model in-process, 4-bit quantized.
  It needs an NVIDIA GPU (CUDA and bitsandbytes).
- OllamaBackend talks to a local Ollama server (https://ollama.com), which
  runs on macOS, Windows and Linux, on CPU or on NVIDIA, AMD and Apple GPUs,
  and manages model memory itself.

A model key picks the backend: "transformers:<Hugging Face id>" or
"ollama:<model name>". torch, transformers and ollama are imported only by the
backend that needs them.
"""

import gc
from threading import Event, Thread
from typing import Any, Dict, Iterator, List, Optional

from config import AVAILABLE_MODELS, OLLAMA_CONFIG

TRANSFORMERS_PREFIX = "transformers:"
OLLAMA_PREFIX = "ollama:"


class LLMBackend:
    """Common interface for generating chat responses."""

    display_name = ""

    def stream_chat(self, messages: List[Dict], generation_config: Dict) -> Iterator[str]:
        """
        Generate a response to a conversation, yielding text as it's produced.

        Generation starts when the iterator is first read. Closing it early
        stops generation.

        Args:
            messages: Conversation [{"role": "user/assistant", "content": "..."}]
            generation_config: Generation parameters (max_new_tokens, temperature, ...)

        Yields:
            Pieces of response text
        """
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        """Count (or estimate) the tokens in a text."""
        return len(text) // 4

    def context_window(self) -> int:
        """How many tokens the model can take in at once, prompt and answer together."""
        return 4096

    def status(self) -> str:
        """Short description of where the model is running, for the status bar."""
        return ""

    def close(self):
        """Release the model's memory."""


class TransformersBackend(LLMBackend):
    """A Hugging Face model run in-process with transformers (4-bit, NVIDIA GPU)."""

    def __init__(self, model_id: str, model: Any = None, tokenizer: Any = None):
        """
        Load a model from AVAILABLE_MODELS.

        Args:
            model_id: Hugging Face model ID (a key of AVAILABLE_MODELS)
            model: Preloaded model, skipping loading (used by tests)
            tokenizer: Preloaded tokenizer, skipping loading (used by tests)
        """
        cfg = AVAILABLE_MODELS[model_id]
        self.model_id = model_id
        self.display_name = cfg["display_name"]

        if model is None or tokenizer is None:
            # Fail before downloading several GB of weights that can't be used here
            if not cuda_available():
                raise RuntimeError("Hugging Face models need an NVIDIA GPU (CUDA), and none was found.")

            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=cfg["trust_remote_code"])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=BitsAndBytesConfig(**cfg["quantization"]),
                device_map=cfg["device_map"],
                trust_remote_code=cfg["trust_remote_code"]
            )

        self.model = model
        self.tokenizer = tokenizer

    def stream_chat(self, messages: List[Dict], generation_config: Dict) -> Iterator[str]:
        """
        Run model.generate in a background thread and yield decoded text as it arrives.

        Closing the iterator stops generation at the next token and waits for the
        thread, so the GPU is free before anything else runs.
        """
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        class StopOnEvent(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return torch.full((input_ids.shape[0],), stop.is_set(),
                                  dtype=torch.bool, device=input_ids.device)

        inputs = self.tokenizer(self._format_prompt(messages), return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        stop = Event()
        errors = []

        def generate():
            try:
                with torch.no_grad():  # grad mode is per thread
                    self.model.generate(
                        **inputs,
                        **generation_config,
                        pad_token_id=self.tokenizer.eos_token_id,
                        streamer=streamer,
                        stopping_criteria=StoppingCriteriaList([StopOnEvent()])
                    )
            except Exception as e:
                errors.append(e)
                streamer.end()  # unblock the reader

        thread = Thread(target=generate, daemon=True)
        thread.start()
        try:
            for text in streamer:
                if text:
                    yield text
            if errors:
                raise errors[0]
        finally:
            # Stop early if the reader went away, and wait so the GPU is free
            # before anything else is generated
            stop.set()
            thread.join()

    def _format_prompt(self, messages: List[Dict]) -> str:
        """
        Apply the model's chat template.

        Some templates reject a system message; the system prompt is then put
        at the start of the first user message instead.
        """
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            if len(messages) < 2 or messages[0]["role"] != "system":
                raise
            system, first, rest = messages[0], messages[1], messages[2:]
            merged = {"role": first["role"], "content": f"{system['content']}\n\n{first['content']}"}
            return self.tokenizer.apply_chat_template([merged] + rest, tokenize=False, add_generation_prompt=True)

    def context_window(self) -> int:
        """The model's maximum sequence length (from its config or tokenizer)."""
        limits = [
            getattr(getattr(self.model, "config", None), "max_position_embeddings", None),
            getattr(self.tokenizer, "model_max_length", None),
        ]
        # Tokenizers without a limit report a huge placeholder value
        limits = [n for n in limits if isinstance(n, int) and 0 < n < 10 ** 7]
        return min(limits) if limits else super().context_window()

    def count_tokens(self, text: str) -> int:
        """Count tokens with the model's tokenizer."""
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return super().count_tokens(text)

    def status(self) -> str:
        """GPU memory in use, or "CPU"."""
        return gpu_status() or "CPU"

    def close(self):
        """Drop the model and return its GPU memory."""
        self.model = None
        self.tokenizer = None
        gc.collect()
        if cuda_available():
            import torch
            torch.cuda.empty_cache()


class OllamaBackend(LLMBackend):
    """A model served by a local Ollama server."""

    def __init__(self, model: str, host: Optional[str] = None):
        """
        Args:
            model: Ollama model name, e.g. "llama3.1:8b"
            host: Ollama server URL (default from OLLAMA_CONFIG)
        """
        import ollama

        self.model = model
        self.display_name = model
        self.host = host or OLLAMA_CONFIG["host"]
        self.client = ollama.Client(host=self.host)
        self._context_window = None

    def stream_chat(self, messages: List[Dict], generation_config: Dict) -> Iterator[str]:
        """
        Stream a chat response from Ollama.

        Closing the iterator closes the HTTP response, and Ollama stops
        generating when its client disconnects.
        """
        import httpx
        import ollama

        stream = None
        try:
            stream = self.client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={**ollama_options(generation_config), "num_ctx": self.context_window()}
            )
            for part in stream:
                if part.message.content:
                    yield part.message.content
        except ollama.ResponseError as e:
            if e.status_code == 404:
                raise RuntimeError(
                    f"Ollama doesn't have the model “{self.model}”. "
                    f"Download it with: ollama pull {self.model}"
                ) from None
            raise RuntimeError(f"Ollama error: {e.error}") from None
        # A refused connection surfaces as httpx.ConnectError when streaming
        except (ConnectionError, httpx.ConnectError):
            raise RuntimeError(
                f"Can't reach Ollama at {self.host}. Start it (the Ollama app, or `ollama serve`), "
                "or set OLLAMA_HOST to where it's running."
            ) from None
        finally:
            if stream is not None:
                stream.close()

    def context_window(self) -> int:
        """
        The context window requested from Ollama: OLLAMA_CONFIG's num_ctx, capped
        at the model's own limit when Ollama reports it.
        """
        if self._context_window is not None:
            return self._context_window
        try:
            info = self.client.show(self.model).modelinfo or {}
        except Exception:
            return OLLAMA_CONFIG["num_ctx"]  # not cached, so it's asked again later
        native = next((v for k, v in info.items() if k.endswith(".context_length")), None)
        self._context_window = min(OLLAMA_CONFIG["num_ctx"], int(native)) if native else OLLAMA_CONFIG["num_ctx"]
        return self._context_window

    def status(self) -> str:
        """Where Ollama has the model loaded, e.g. "Ollama · 4.9 GB VRAM"."""
        try:
            running = {m.model: m for m in self.client.ps().models}
        except Exception:
            return "Ollama"
        loaded = running.get(self.model) or running.get(f"{self.model}:latest")
        if loaded is None:
            return "Ollama"
        if loaded.size_vram:
            return f"Ollama · {loaded.size_vram / 1024 ** 3:.1f} GB VRAM"
        return "Ollama · CPU"


def ollama_options(generation_config: Dict) -> Dict:
    """Translate transformers-style generation settings into Ollama options."""
    names = {
        "max_new_tokens": "num_predict",
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "repetition_penalty": "repeat_penalty",
    }
    options = {names[k]: v for k, v in generation_config.items() if k in names}
    if generation_config.get("do_sample") is False:
        options["temperature"] = 0
    return options


def list_ollama_models(host: Optional[str] = None) -> List[str]:
    """
    List the models a local Ollama server has.

    Returns:
        Model names, or an empty list if Ollama isn't installed or isn't running
    """
    try:
        import ollama

        client = ollama.Client(host=host or OLLAMA_CONFIG["host"], timeout=OLLAMA_CONFIG["list_timeout"])
        return sorted(m.model for m in client.list().models if m.model)
    except Exception:
        return []


_loaded: List[LLMBackend] = []


def load_backend(model_key: str) -> LLMBackend:
    """
    Load the model a key names.

    Args:
        model_key: "transformers:<Hugging Face id>" or "ollama:<model name>"

    Returns:
        The backend, ready to generate
    """
    if model_key.startswith(OLLAMA_PREFIX):
        backend = OllamaBackend(model_key[len(OLLAMA_PREFIX):])
    elif model_key.startswith(TRANSFORMERS_PREFIX):
        backend = TransformersBackend(model_key[len(TRANSFORMERS_PREFIX):])
    else:
        raise ValueError(f"Unknown model key: {model_key}")
    _loaded.append(backend)
    return backend


def release_all():
    """Release every loaded model, e.g. before switching to another one."""
    while _loaded:
        _loaded.pop().close()
    gc.collect()


def model_label(model_key: str) -> str:
    """Name shown in the model picker, e.g. "llama3.1:8b · Ollama" or "Llama 3.1 8B (~4GB) · HF"."""
    if model_key.startswith(OLLAMA_PREFIX):
        return f"{model_key[len(OLLAMA_PREFIX):]} · Ollama"
    model_id = model_key[len(TRANSFORMERS_PREFIX):]
    name = AVAILABLE_MODELS[model_id]["display_name"] if model_id in AVAILABLE_MODELS else model_id
    return f"{name} · HF"


def short_label(model_key: str) -> str:
    """Compact model name for the status bar, e.g. "Llama 3.1 8B"."""
    if model_key.startswith(OLLAMA_PREFIX):
        return model_key[len(OLLAMA_PREFIX):]
    model_id = model_key[len(TRANSFORMERS_PREFIX):]
    return AVAILABLE_MODELS.get(model_id, {}).get("display_name", model_id).split(" (")[0]


def cuda_available() -> bool:
    """Whether an NVIDIA GPU is usable (False if torch isn't installed)."""
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def default_model_key(configured: str, ollama_models: List[str]) -> str:
    """
    Choose the model to start with.

    A Hugging Face model needs an NVIDIA GPU, so without one, start on the
    first Ollama model if Ollama has any.

    Args:
        configured: DEFAULT_MODEL from config. Without a prefix, a Hugging Face
                    ID from AVAILABLE_MODELS means that model, anything else Ollama.
        ollama_models: Models Ollama has (empty if it isn't running)
    """
    if not configured.startswith((TRANSFORMERS_PREFIX, OLLAMA_PREFIX)):
        prefix = TRANSFORMERS_PREFIX if configured in AVAILABLE_MODELS else OLLAMA_PREFIX
        configured = prefix + configured
    if configured.startswith(TRANSFORMERS_PREFIX) and ollama_models and not cuda_available():
        return OLLAMA_PREFIX + ollama_models[0]
    return configured


def gpu_status() -> Optional[str]:
    """
    Describe GPU memory in use, e.g. "RTX 3060 · 5.1 / 12.0 GB".

    Returns:
        The description, or None when no CUDA GPU is available
    """
    if not cuda_available():
        return None
    import torch
    try:
        count = torch.cuda.device_count()
        memory = [torch.cuda.mem_get_info(i) for i in range(count)]  # (free, total) bytes
        name = torch.cuda.get_device_name(0).replace("NVIDIA ", "").replace("GeForce ", "")
    except Exception:  # display only: never let a driver hiccup break the page
        return None
    used_gb = sum(total - free for free, total in memory) / 1024 ** 3
    total_gb = sum(total for _, total in memory) / 1024 ** 3
    label = name if count == 1 else f"{count} GPUs"
    return f"{label} · {used_gb:.1f} / {total_gb:.1f} GB"
