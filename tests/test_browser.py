"""
Browser tests: the real app served by Streamlit, driven in Chromium, with
answers from a fake Ollama server.

Needs Playwright:  pip install playwright && playwright install chromium
They're skipped when Playwright isn't installed. To use an existing Chromium
instead of Playwright's, set CHROMIUM_PATH.
"""

import os
import re
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import expect, sync_playwright  # noqa: E402

from fake_ollama import FakeOllama  # noqa: E402

pytestmark = pytest.mark.browser

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TESTS)
LONG_ANSWER = ("Streaming makes a local model feel responsive because each word appears as soon as "
               "it is generated instead of after the whole answer is done.").split() * 12
TRICKY_ANSWER = ["Use", "<b>bold</b>", "and", '"double"', "or", "'single'", "quotes", "&", "&amp;",
                 "</script>\nNew", "line."]
ACCENT = {"light": "rgb(79, 70, 229)", "dark": "rgb(97, 100, 241)"}


# --- fixtures ----------------------------------------------------------------

@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH") or None)
        yield browser
        browser.close()


@pytest.fixture
def ollama():
    server = FakeOllama(words=LONG_ANSWER, delay=0.03).start()
    yield server
    server.stop()


@pytest.fixture
def launch_app(tmp_path):
    """Start the app with its own data folder; returns its URL. Stopped after the test."""
    started = []

    def launch(ollama_url, **env):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        log = open(tmp_path / f"streamlit-{port}.log", "w")
        full_env = {**os.environ, "OLLAMA_HOST": ollama_url, "APP_TEST_DATA": str(tmp_path / "data"),
                    "LLM_MODEL": "ollama:llama3.1:8b", "PYTHONUNBUFFERED": "1"}
        for key, value in env.items():
            if value is None:
                full_env.pop(key, None)
            else:
                full_env[key] = value
        proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", os.path.join(TESTS, "browser_app.py"),
             "--server.headless", "true", "--server.address", "127.0.0.1", "--server.port", str(port),
             "--browser.gatherUsageStats", "false", "--server.fileWatcherType", "none"],
            cwd=REPO, env=full_env, stdout=log, stderr=subprocess.STDOUT)
        started.append((proc, log))
        url = f"http://127.0.0.1:{port}"
        deadline = time.time() + 90
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                urllib.request.urlopen(url + "/_stcore/health", timeout=1)
                return url
            except OSError:
                time.sleep(0.3)
        log.flush()
        pytest.fail("The app didn't start:\n" + open(log.name).read()[-3000:])

    yield launch
    for proc, log in started:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


def open_page(browser, url, scheme="light", **context_options):
    context = browser.new_context(viewport={"width": 1280, "height": 900}, color_scheme=scheme, **context_options)
    page = context.new_page()
    page.goto(url, wait_until="networkidle")
    page.get_by_placeholder("Ask me anything...").wait_for(timeout=90000)
    return page


# --- helpers -------------------------------------------------------------------

def messages(page):
    return page.locator('[data-testid="stChatMessage"]')


def sidebar(page):
    return page.locator('[data-testid="stSidebar"]')


def ask(page, question, wait=True):
    """Send a question; with wait, return once its answer has finished streaming."""
    before = messages(page).count()
    box = page.get_by_placeholder("Ask me anything...")
    box.fill(question)
    box.press("Enter")
    page.wait_for_function(
        f"document.querySelectorAll('[data-testid=stChatMessage]').length >= {before + 2}", timeout=30000)
    if wait:
        page.get_by_role("button", name="Stop generating").wait_for(state="detached", timeout=60000)
        page.wait_for_timeout(300)


def badges(page):
    row = page.locator('[data-testid="stMainBlockContainer"] [data-testid="stHorizontalBlock"]').first
    return [t.split(" ", 1)[-1].strip() for t in row.locator('[data-testid="stMarkdown"]').all_inner_texts()]


def pick_model(page, label):
    sidebar(page).get_by_role("combobox", name="Model").click()
    page.get_by_role("option", name=label).click()
    page.wait_for_timeout(1000)


def copy_button(page):
    return page.frame_locator("iframe").last.get_by_role("button", name="Copy")


# --- chat ----------------------------------------------------------------------

def test_answers_stream_in_and_stop_halts_generation(browser, ollama, launch_app):
    page = open_page(browser, launch_app(ollama.url))

    ask(page, "How does streaming help?", wait=False)
    answer = messages(page).nth(1)
    page.get_by_role("button", name="Stop generating").wait_for(timeout=10000)
    page.wait_for_timeout(800)
    early = len(answer.inner_text())
    page.wait_for_timeout(800)
    assert len(answer.inner_text()) > early > 0  # text arrives while it's generated

    page.get_by_role("button", name="Stop generating").click()
    page.get_by_text("Stopped before the answer was complete.").wait_for(timeout=10000)
    assert ollama.disconnected.wait(timeout=10)  # the model was told to stop
    assert ollama.chunks_sent < len(LONG_ANSWER)
    assert page.get_by_role("button", name="Stop generating").count() == 0

    ollama.words = ["A", "follow-up."]
    ask(page, "And a follow-up?")
    assert "A follow-up." in messages(page).nth(3).inner_text()


def test_new_chat_mid_answer_keeps_the_partial_answer_with_its_chat(browser, ollama, launch_app):
    page = open_page(browser, launch_app(ollama.url))

    ask(page, "Interrupted question", wait=False)
    page.get_by_role("button", name="Stop generating").wait_for(timeout=10000)
    page.wait_for_timeout(800)
    sidebar(page).get_by_role("button", name=re.compile("New chat")).click()
    page.wait_for_timeout(1500)
    assert messages(page).count() == 0  # the new chat starts empty

    sidebar(page).get_by_role("button", name=re.compile("^Interrupted question")).locator("visible=true").first.click()
    page.get_by_text("Stopped before the answer was complete.").wait_for(timeout=10000)
    assert messages(page).count() == 2
    assert "Streaming makes" in messages(page).nth(1).inner_text()


def test_copy_puts_the_exact_answer_on_the_clipboard(browser, ollama, launch_app):
    ollama.words = TRICKY_ANSWER
    expected = " ".join(TRICKY_ANSWER)
    url = launch_app(ollama.url)

    page = open_page(browser, url, permissions=["clipboard-read", "clipboard-write"])
    ask(page, "Show me some formatting")
    copy_button(page).click()
    page.frame_locator("iframe").last.get_by_text("Copied").wait_for(timeout=5000)
    assert page.evaluate("navigator.clipboard.readText()") == expected

    # Plain http on a LAN address has no navigator.clipboard: the fallback must copy the same text
    fallback = browser.new_context(viewport={"width": 1280, "height": 900})
    fallback.add_init_script("Object.defineProperty(Navigator.prototype, 'clipboard', {get() { return undefined; }});")
    lan_page = fallback.new_page()
    lan_page.goto(url, wait_until="networkidle")
    lan_page.get_by_placeholder("Ask me anything...").wait_for(timeout=60000)
    ask(lan_page, "Fallback check")
    copy_button(lan_page).click()
    lan_page.frame_locator("iframe").last.get_by_text("Copied").wait_for(timeout=5000)
    assert page.evaluate("navigator.clipboard.readText()") == expected


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_copy_button_follows_the_theme(browser, ollama, launch_app, scheme):
    ollama.words = ["Short", "answer."]
    page = open_page(browser, launch_app(ollama.url), scheme=scheme)
    ask(page, "Theme check")

    button = copy_button(page)
    body_colour = page.evaluate("getComputedStyle(document.body).color")
    assert button.evaluate("e => getComputedStyle(e).color") == body_colour
    button.hover()
    page.wait_for_timeout(300)
    assert button.evaluate("e => getComputedStyle(e).color") == ACCENT[scheme]


def test_regenerate_replaces_the_latest_answer(browser, ollama, launch_app):
    ollama.words = ["First", "answer."]
    page = open_page(browser, launch_app(ollama.url))
    ask(page, "Question one")
    ask(page, "Question two")
    page.wait_for_function("document.querySelectorAll('iframe').length === 2", timeout=10000)
    assert page.get_by_role("button", name="Regenerate").count() == 1  # only on the latest answer

    ollama.words = ["Second", "answer."]
    requests = len(ollama.requests)
    page.get_by_role("button", name="Regenerate").click()
    page.get_by_text("Second answer.").wait_for(timeout=30000)
    assert len(ollama.requests) == requests + 1
    assert messages(page).count() == 4


# --- documents and saved chats --------------------------------------------------

def test_documents_page(browser, ollama, launch_app, tmp_path):
    files = []
    for name, text in [("cats.txt", "Cats purr and chase mice."), ("dogs.md", "# Dogs\n\nDogs bark loudly.")]:
        (tmp_path / name).write_text(text)
        files.append(str(tmp_path / name))
    ollama.words = ["A", "summary."]
    page = open_page(browser, launch_app(ollama.url))
    main = page.locator('[data-testid="stMainBlockContainer"]')

    page.get_by_role("link", name=re.compile("Documents")).first.click()
    page.get_by_role("heading", name="Documents").wait_for(timeout=10000)
    page.locator('input[type="file"]').set_input_files(files)
    page.get_by_role("button", name=re.compile("Process documents")).click()
    page.get_by_text("Processed cats.txt").wait_for(timeout=30000)
    page.get_by_text("Processed dogs.md").wait_for()
    assert page.locator('[data-testid="stFileUploaderFile"]').count() == 0  # the uploader cleared itself
    expect(page.locator('[data-testid="stMetricValue"]').first).to_have_text("2")

    page.locator('input[type="file"]').set_input_files(files[:1])
    page.get_by_role("button", name=re.compile("Process documents")).click()
    page.get_by_text("cats.txt is already indexed and unchanged").wait_for(timeout=30000)

    # Two quick deletes each show their own message
    page.get_by_role("button", name=re.compile("^.*Delete$")).click()
    main.get_by_text(re.compile(r"Deleted cats\.txt")).wait_for(timeout=10000)
    page.get_by_role("button", name=re.compile("^.*Delete$")).click()
    main.get_by_text(re.compile(r"Deleted dogs\.md")).wait_for(timeout=10000)
    page.get_by_text("No documents indexed yet").wait_for(timeout=10000)

    # Clear all asks first
    page.locator('input[type="file"]').set_input_files(files)
    page.get_by_role("button", name=re.compile("Process documents")).click()
    page.get_by_text("Processed dogs.md").wait_for(timeout=30000)
    page.get_by_role("button", name=re.compile("Clear all")).click()
    page.get_by_text("This can't be undone").wait_for(timeout=5000)
    page.get_by_role("button", name="Clear all documents").click()
    page.get_by_text("No documents indexed yet").wait_for(timeout=10000)

    # Summarize switches to the chat, with the summary in it
    page.locator('input[type="file"]').set_input_files(files[:1])
    page.get_by_role("button", name=re.compile("Process documents")).click()
    page.get_by_text("Processed cats.txt").wait_for(timeout=30000)
    page.get_by_role("button", name=re.compile("Summarize")).click()
    page.get_by_placeholder("Ask me anything...").wait_for(timeout=30000)
    expect(messages(page).nth(1)).to_contain_text("A summary.", timeout=30000)
    assert "Summarize cats.txt" in messages(page).nth(0).inner_text()


def test_saved_chats(browser, ollama, launch_app):
    ollama.words = ["An", "answer."]
    page = open_page(browser, launch_app(ollama.url))
    side = sidebar(page)
    chat_named = lambda start: side.get_by_role("button", name=re.compile("^" + start)).locator("visible=true")

    ask(page, "First chat question")
    side.get_by_role("button", name=re.compile("Save")).click()
    side.get_by_text(re.compile("Saved .First chat question")).wait_for(timeout=10000)
    assert chat_named("First chat").first.get_attribute("data-testid") == "stBaseButton-secondary"  # current

    side.get_by_role("button", name=re.compile("New chat")).click()
    page.wait_for_timeout(1000)
    assert messages(page).count() == 0
    ask(page, "Second chat question")

    chat_named("First chat").first.click()  # switching saves the current chat first
    page.wait_for_timeout(1000)
    assert "First chat question" in messages(page).first.inner_text()
    expect(chat_named("Second chat")).to_have_count(1)

    # Delete the second chat: chat names and delete icons alternate in list order
    names = [t for t in side.locator('[data-testid="stBaseButton-tertiary"]').locator("visible=true")
             .all_inner_texts() if t != "delete"]
    index = next(i for i, n in enumerate(names) if n.startswith("Second chat"))
    side.get_by_role("button", name="delete icon", exact=True).locator("visible=true").nth(index).click()
    side.get_by_text(re.compile("Deleted .Second chat")).wait_for(timeout=10000)
    expect(chat_named("Second chat")).to_have_count(0)


# --- models, status bar ------------------------------------------------------------

def test_status_bar_and_welcome(browser, ollama, launch_app, tmp_path):
    (tmp_path / "cats.txt").write_text("Cats purr.")
    ollama.words = ["An", "answer."]
    page = open_page(browser, launch_app(ollama.url))
    welcome = page.get_by_role("heading", name="What can I help with?")

    assert welcome.is_visible()
    assert badges(page) == ["llama3.1:8b", "Documents off", "Ollama"]
    sidebar(page).get_by_text("Answer from documents").click()
    page.wait_for_timeout(800)
    assert "No documents indexed" in badges(page)

    sidebar(page).get_by_role("link", name=re.compile("Add documents")).click()
    page.get_by_role("heading", name="Documents").wait_for(timeout=10000)
    page.locator('input[type="file"]').set_input_files([str(tmp_path / "cats.txt")])
    page.get_by_role("button", name=re.compile("Process documents")).click()
    page.get_by_text("Processed cats.txt").wait_for(timeout=30000)
    page.get_by_role("link", name=re.compile("Chat")).first.click()
    page.get_by_placeholder("Ask me anything...").wait_for(timeout=10000)
    page.wait_for_timeout(800)
    assert "Documents on · 1 indexed" in badges(page)

    sidebar(page).get_by_role("combobox", name="Focus on document").click()
    page.get_by_role("option", name="cats.txt").click()
    page.wait_for_timeout(800)
    assert "Focused on cats.txt" in badges(page)

    ask(page, "Tell me about cats")
    assert welcome.count() == 0
    ask(page, "More?")  # the status bar reads Ollama's VRAM once the model is loaded
    assert "Ollama · 4.5 GB VRAM" in badges(page)


def test_without_a_gpu_ollama_is_used_and_hugging_face_models_explain_why_not(browser, ollama, launch_app, tmp_path):
    (tmp_path / "cats.txt").write_text("Cats purr.")
    ollama.words = ["An", "answer."]
    page = open_page(browser, launch_app(ollama.url, LLM_MODEL=None, APP_TEST_NO_GPU="1"))

    # The configured default is a Hugging Face model; with no GPU the app starts on Ollama's
    assert badges(page)[0] == "llama3.1:8b"
    sidebar(page).get_by_role("combobox", name="Model").click()
    options = page.get_by_role("option").all_inner_texts()
    assert options[:2] == ["llama3.1:8b · Ollama", "qwen2.5:7b · Ollama"] and "Llama 3.1 8B (~4GB) · HF" in options
    page.keyboard.press("Escape")

    ask(page, "Hello")
    pick_model(page, "Llama 3.1 8B (~4GB) · HF")
    expect(page.get_by_text(re.compile("Couldn't load Llama 3.1 8B"))).to_be_visible(timeout=30000)
    expect(page.get_by_text(re.compile("need an NVIDIA GPU"))).to_be_in_viewport()
    assert page.get_by_placeholder("Ask me anything...").is_disabled()
    assert "Model not loaded" in badges(page)
    assert page.get_by_role("button", name="Regenerate").count() == 0

    # The Documents page doesn't need a model; Summarize explains that it does
    page.get_by_role("link", name=re.compile("Documents")).first.click()
    page.get_by_role("heading", name="Documents").wait_for(timeout=10000)
    page.locator('input[type="file"]').set_input_files([str(tmp_path / "cats.txt")])
    page.get_by_role("button", name=re.compile("Process documents")).click()
    page.get_by_text("Processed cats.txt").wait_for(timeout=30000)
    page.get_by_role("button", name=re.compile("Summarize")).click()
    page.get_by_text(re.compile("Can't summarize")).wait_for(timeout=10000)

    page.get_by_role("link", name=re.compile("Chat")).first.click()
    page.get_by_placeholder("Ask me anything...").wait_for(timeout=10000)
    pick_model(page, "qwen2.5:7b · Ollama")
    ask(page, "Back on Ollama?")
    assert badges(page)[0] == "qwen2.5:7b"


def test_an_unreachable_ollama_is_reported_on_the_answer(browser, ollama, launch_app):
    ollama.words = ["An", "answer."]
    page = open_page(browser, launch_app(ollama.url))
    ask(page, "Hello")

    ollama.stop()
    ask(page, "Is anyone there?")

    expect(page.get_by_text(re.compile(f"Can't reach Ollama at {re.escape(ollama.url)}"))).to_be_visible()
    assert page.get_by_role("button", name="Regenerate").count() == 1  # try again once it's back


def test_no_deprecation_warnings_while_chatting(browser, ollama, launch_app, tmp_path):
    ollama.words = ["An", "answer."]
    page = open_page(browser, launch_app(ollama.url))
    ask(page, "Hello")
    ask(page, "Again")

    log = "".join(p.read_text() for p in tmp_path.glob("streamlit-*.log"))
    # How Streamlit words its deprecation warnings
    assert "will be removed" not in log and "Please replace" not in log, log[-2000:]
