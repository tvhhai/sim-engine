"""The Studio "Hướng dẫn" screen serves docs/*.md and docs/image/* only."""

from sim_engine.studio.api import docs_list, handle_docs


class FakeHandler:
    def __init__(self):
        self.sent, self.error = None, None

    def _send(self, body, ctype="application/json"):
        self.sent = (body, ctype)

    def send_error(self, code):
        self.error = code


def test_lists_guide_with_title():
    docs = {d["name"]: d["title"] for d in docs_list()}
    assert docs["guide_vi"].startswith("Hiểu SimEngine")


def test_serves_markdown_and_images():
    h = FakeHandler()
    handle_docs(h, "/studio/docs/guide_vi.md")
    assert h.sent[1] == "text/markdown" and h.sent[0].startswith("# ")
    h = FakeHandler()
    handle_docs(h, "/studio/docs/image/guide_vi_1.svg")
    assert h.sent[1] == "image/svg+xml" and h.sent[0].startswith(b"<svg")


def test_rejects_paths_outside_docs():
    for path in ("/studio/docs/../config.py", "/studio/docs/image/../../config.py", "/studio/docs/missing.md"):
        h = FakeHandler()
        handle_docs(h, path)
        assert h.error == 404 and h.sent is None, path
