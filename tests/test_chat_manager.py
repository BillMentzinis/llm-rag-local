import json

import pytest

import chat_manager


@pytest.fixture(autouse=True)
def chats_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(chat_manager, "CHATS_DIR", str(tmp_path / "chats"))
    return tmp_path / "chats"


MESSAGES = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]


def test_new_chats_with_the_same_name_get_their_own_files():
    first = chat_manager.save_chat(chat_manager.unique_chat_name("hello"), MESSAGES)
    second_name = chat_manager.unique_chat_name("hello")
    second = chat_manager.save_chat(second_name, MESSAGES[:1])

    assert second_name == "hello (2)" and first != second
    assert chat_manager.unique_chat_name("hello") == "hello (3)"
    assert sorted(c["name"] for c in chat_manager.list_chats()) == ["hello", "hello (2)"]
    assert chat_manager.load_chat(first) == MESSAGES


def test_names_that_make_the_same_filename_do_not_collide():
    chat_manager.save_chat(chat_manager.unique_chat_name("what?"), MESSAGES)
    assert chat_manager.unique_chat_name("what*") == "what* (2)"


def test_resaving_writes_to_the_chats_own_file(chats_dir):
    path = chat_manager.save_chat("hello", MESSAGES[:1])
    assert chat_manager.save_chat("hello", MESSAGES, path) == path

    assert [p.name for p in chats_dir.iterdir()] == ["hello.json"]
    assert json.loads((chats_dir / "hello.json").read_text(encoding="utf-8"))["messages"] == MESSAGES
