"""
A small stand-in for an Ollama server, for tests.

It serves the endpoints the app uses: /api/tags (installed models), /api/ps
(loaded models), /api/show (model details) and /api/chat (streamed as
newline-delimited JSON, like Ollama). It records each chat request and notices when a client disconnects
mid-stream, which is how Ollama learns to stop generating.

Run it on its own to point the app at it:
    python tests/fake_ollama.py --port 11500
    OLLAMA_HOST=http://127.0.0.1:11500 streamlit run streamlit_app.py
"""

import argparse
import json
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GIB = 1024 ** 3


class FakeOllama:
    """Fake Ollama server running in a background thread."""

    def __init__(self, models=("llama3.1:8b", "qwen2.5:7b"), words=None, delay=0.0,
                 size_vram=4.5 * GIB, fail_after=None, port=0, log=False, context_length=131072):
        """
        Args:
            models: Model names it reports as installed
            words: Words each answer is streamed as, one per chunk
            delay: Seconds between chunks
            size_vram: VRAM /api/ps reports for a loaded model (0 means CPU)
            fail_after: Send an error line after this many chunks
            port: Port to listen on (0 picks a free one)
            log: Print chat requests and disconnects
            context_length: Context window /api/show reports for every model
        """
        self.models = list(models)
        self.words = words or ["Hello", "from", "the", "fake", "Ollama", "server."]
        self.delay = delay
        self.size_vram = size_vram
        self.fail_after = fail_after
        self.log = log
        self.context_length = context_length
        self.requests = []           # JSON bodies of /api/chat requests
        self.loaded = set()          # models that have answered a chat (shown by /api/ps)
        self.chunks_sent = 0         # chunks of the last answer that reached the client
        self.disconnected = threading.Event()
        self.server = ThreadingHTTPServer(("127.0.0.1", port), self._handler())
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def start(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def stop(self):
        self.server.shutdown()
        self.server.server_close()

    def _handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _json(self, status, payload):
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/api/tags":
                    self._json(200, {"models": [
                        {"name": m, "model": m, "size": 4 * GIB, "digest": "0" * 12} for m in fake.models
                    ]})
                elif self.path == "/api/ps":
                    self._json(200, {"models": [
                        {"name": m, "model": m, "size": 5 * GIB, "size_vram": fake.size_vram}
                        for m in sorted(fake.loaded)
                    ]})
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/api/show":
                    if request["model"] not in fake.models:
                        self._json(404, {"error": f"model '{request['model']}' not found"})
                    else:
                        self._json(200, {"model_info": {"general.architecture": "llama",
                                                        "llama.context_length": fake.context_length}})
                    return
                if self.path != "/api/chat":
                    self._json(404, {"error": "not found"})
                    return
                fake.requests.append(request)
                model = request["model"]
                if fake.log:
                    print(f"chat request for {model}", flush=True)
                if model not in fake.models:
                    self._json(404, {"error": f"model \"{model}\" not found, try pulling it first"})
                    return
                fake.loaded.add(model)
                fake.chunks_sent = 0
                fake.disconnected.clear()

                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()  # no length: the body ends when the connection closes

                def chunk(payload):
                    self.wfile.write((json.dumps(payload) + "\n").encode())
                    self.wfile.flush()

                now = datetime.now(timezone.utc).isoformat()
                try:
                    for i, word in enumerate(fake.words):
                        if fake.fail_after is not None and i == fake.fail_after:
                            chunk({"error": "model runner has unexpectedly stopped"})
                            return
                        time.sleep(fake.delay)
                        chunk({"model": model, "created_at": now, "done": False,
                               "message": {"role": "assistant", "content": (" " if i else "") + word}})
                        fake.chunks_sent += 1
                    chunk({"model": model, "created_at": now, "done": True, "done_reason": "stop",
                           "message": {"role": "assistant", "content": ""},
                           "eval_count": len(fake.words)})
                except (BrokenPipeError, ConnectionResetError):
                    fake.disconnected.set()
                    if fake.log:
                        print(f"client disconnected after {fake.chunks_sent} of {len(fake.words)} chunks", flush=True)
                self.close_connection = True

        return Handler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=11500)
    parser.add_argument("--delay", type=float, default=0.1, help="seconds between streamed words")
    parser.add_argument("--models", nargs="+", default=["llama3.1:8b", "qwen2.5:7b"])
    args = parser.parse_args()
    words = ("Ollama streams this answer one word at a time, so it appears as it is generated "
             "rather than all at once when the model has finished. ").split() * 8
    server = FakeOllama(models=args.models, words=words, delay=args.delay, port=args.port, log=True).start()
    print(f"Fake Ollama listening on {server.url} with models {args.models}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.stop()
