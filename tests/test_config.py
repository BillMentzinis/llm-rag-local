import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def paths_with(**env):
    """config.PATHS as a fresh interpreter sees it, with DATA_DIR only if given."""
    environment = {k: v for k, v in os.environ.items() if k != "DATA_DIR"} | env
    out = subprocess.run([sys.executable, "-c", "import json, config; print(json.dumps(config.PATHS))"],
                         cwd=REPO, env=environment, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_data_is_kept_next_to_the_app_by_default():
    assert paths_with() == {"chroma_db": os.path.join(REPO, "chroma_db"), "chats": os.path.join(REPO, "chats")}


def test_data_dir_moves_the_index_and_chats(tmp_path):
    assert paths_with(DATA_DIR=str(tmp_path)) == {
        "chroma_db": str(tmp_path / "chroma_db"), "chats": str(tmp_path / "chats")}
