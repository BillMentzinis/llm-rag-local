"""
Smoke test for the Docker image: drives the app running in a container.

    python tests/docker_smoke.py http://127.0.0.1:8501            # upload a document, ask about it
    python tests/docker_smoke.py http://127.0.0.1:8501 --persisted  # after a restart: is it still there?

The container must reach an Ollama (the fake one in tests/fake_ollama.py will
do). CI runs this against the image it builds; see .github/workflows/docker.yml.
"""

import argparse
import os
import re
import sys
import tempfile

from playwright.sync_api import expect, sync_playwright

DOCUMENT = "smoke-test-notes.txt"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url")
    parser.add_argument("--persisted", action="store_true", help="check the document from an earlier run is kept")
    args = parser.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH") or None)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(args.url, wait_until="networkidle")
        page.get_by_placeholder("Ask me anything...").wait_for(timeout=120000)

        page.get_by_role("link", name=re.compile("Documents")).first.click()
        page.get_by_role("heading", name="Documents").wait_for(timeout=10000)
        if args.persisted:
            expect(page.get_by_role("combobox", name="Document")).to_have_value(DOCUMENT, timeout=10000)
            print(f"{DOCUMENT} is still indexed after the restart")
            return 0

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, DOCUMENT)
            with open(path, "w", encoding="utf-8") as f:
                f.write("The office plants are watered every Tuesday morning by whoever arrives first.")
            page.locator('input[type="file"]').set_input_files(path)
            page.get_by_role("button", name=re.compile("Process documents")).click()
            page.get_by_text(f"Processed {DOCUMENT}").wait_for(timeout=120000)

        page.get_by_role("link", name=re.compile("Chat")).first.click()
        box = page.get_by_placeholder("Ask me anything...")
        box.wait_for(timeout=10000)
        page.locator('[data-testid="stSidebar"]').get_by_text("Answer from documents").click()
        page.wait_for_timeout(1000)
        box.fill("When are the office plants watered?")
        box.press("Enter")
        page.get_by_role("button", name="Stop generating").wait_for(state="detached", timeout=120000)
        answer = page.locator('[data-testid="stChatMessage"]').nth(1)
        expect(answer).to_contain_text("Ollama", timeout=10000)  # the fake Ollama's answer
        expect(answer).to_contain_text("Sources (1 chunks used)")
        print("Indexed a document and answered from it")
        return 0


if __name__ == "__main__":
    sys.exit(main())
