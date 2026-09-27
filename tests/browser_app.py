"""
Runs the real streamlit_app.py for the browser tests (tests/test_browser.py),
started with `streamlit run tests/browser_app.py` from the repository folder.

It swaps in the stand-in embedder so nothing is downloaded, and keeps the
document index and saved chats in APP_TEST_DATA. Answers come from whatever
OLLAMA_HOST points at (the tests start a fake Ollama). With APP_TEST_NO_GPU
set, the app behaves as on a machine without an NVIDIA GPU.
"""

import os
import runpy
import sys

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TESTS)
sys.path[:0] = [REPO, TESTS]

import sentence_transformers  # noqa: E402

from fake_embedder import FakeEmbedder  # noqa: E402

sentence_transformers.SentenceTransformer = lambda name: FakeEmbedder()

import chat_manager  # noqa: E402
import config  # noqa: E402

data = os.environ["APP_TEST_DATA"]
config.PATHS["chroma_db"] = os.path.join(data, "chroma_db")
config.PATHS["chats"] = chat_manager.CHATS_DIR = os.path.join(data, "chats")

if os.environ.get("APP_TEST_NO_GPU"):
    import llm_backends
    llm_backends.cuda_available = lambda: False

runpy.run_path(os.path.join(REPO, "streamlit_app.py"), run_name="__main__")
