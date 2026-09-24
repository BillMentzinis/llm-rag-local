import html

from streamlit.testing.v1 import AppTest

# Characters that must survive the trip into the copy button's HTML
TRICKY = 'Use <b>bold</b> and "double" or \'single\' quotes & &amp; </script>\nNew line. /etc/passwd'


def _render_copy_button(text):
    import streamlit_app
    streamlit_app.render_message_actions({"role": "assistant", "content": text}, can_regenerate=False)


def _copy_iframe(text):
    at = AppTest.from_function(_render_copy_button, args=(text,), default_timeout=30).run()
    assert not at.exception, at.exception
    iframes = at.get("iframe")
    assert len(iframes) == 1
    return iframes[0].proto


def test_copy_button_is_embedded_as_html_not_a_url_or_file():
    proto = _copy_iframe(TRICKY)
    assert proto.srcdoc and not proto.src


def test_answer_text_is_escaped_into_the_data_attribute():
    srcdoc = _copy_iframe(TRICKY).srcdoc
    assert f'data-text="{html.escape(TRICKY, quote=True)}"' in srcdoc
    assert "</b> and" not in srcdoc  # never interpolated raw


def test_answer_that_looks_like_a_path_or_url_is_still_just_text():
    for text in ["/app/static/report.html", "https://example.com", "README.md"]:
        proto = _copy_iframe(text)
        assert not proto.src and f'data-text="{html.escape(text, quote=True)}"' in proto.srcdoc


def test_rendering_logs_no_deprecation_warning():
    import logging

    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logger = logging.getLogger("streamlit.deprecation_util")
    logger.addHandler(handler)
    try:
        _copy_iframe(TRICKY)
    finally:
        logger.removeHandler(handler)

    assert [r.getMessage() for r in records] == []
