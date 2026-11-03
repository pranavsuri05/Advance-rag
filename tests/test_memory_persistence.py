import pytest

from src.memory.session_manager import SessionManager


def test_corrupt_session_file_does_not_crash_and_is_quarantined(tmp_path):
    m = SessionManager(tmp_path)
    sid = m.create()
    (tmp_path / f"{sid}.json").write_text("{ this is not json")
    assert m.messages(sid) == []                                   # chat page must still render
    m.add_message(sid, "user", "fresh start")                      # and the session is usable again
    assert [x["content"] for x in m.messages(sid)] == ["fresh start"]
    assert list(tmp_path.glob(f"{sid}.json.corrupt*"))             # the damaged data is kept for inspection, not destroyed


@pytest.mark.parametrize("payload", ["[]", '{"messages": "nope"}', '{"session_id": 1}', "null", '"str"'])
def test_structurally_invalid_session_json_recovers(tmp_path, payload):
    m = SessionManager(tmp_path)
    sid = m.create()
    (tmp_path / f"{sid}.json").write_text(payload)
    assert m.messages(sid) == [] and m.window(sid, 3) == []


@pytest.mark.parametrize("n", [0, -1])
def test_non_positive_window_sends_no_history(tmp_path, n):
    m = SessionManager(tmp_path)
    sid = m.create()
    for i in range(3):
        m.add_message(sid, "user", f"m{i}")
    assert m.window(sid, n) == []                                  # window of 0 must not leak the whole conversation


def test_persistence_across_restart_multiple_turns_unicode(tmp_path):
    sid = SessionManager(tmp_path).create()
    a = SessionManager(tmp_path)
    for i in range(5):
        a.add_message(sid, "user" if i % 2 == 0 else "assistant", f"turn {i} – café 日本語 🚀", sources=[{"chunk_id": f"c{i}"}] if i % 2 else None)
    b = SessionManager(tmp_path)                                    # simulated process restart
    msgs = b.messages(sid)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant", "user"]
    assert msgs[1]["sources"] == [{"chunk_id": "c1"}] and "日本語 🚀" in msgs[4]["content"]
    assert all("timestamp" in m for m in msgs)


def test_isolation_between_many_sessions_and_unknown_session(tmp_path):
    m = SessionManager(tmp_path)
    sids = [m.create() for _ in range(4)]
    for i, s in enumerate(sids):
        m.add_message(s, "user", f"only-{i}")
    for i, s in enumerate(sids):
        assert [x["content"] for x in m.messages(s)] == [f"only-{i}"]
    assert m.messages("neverwritten123") == [] and not m.exists("neverwritten123")
    assert len(set(sids)) == 4


@pytest.mark.parametrize("bad", ["../x", "a/b", "", "a b", "..", "x\\y"])
def test_session_id_validation(tmp_path, bad):
    with pytest.raises(ValueError):
        SessionManager(tmp_path).load(bad)


def test_reset_clears_messages_and_context_but_keeps_session(tmp_path):
    m = SessionManager(tmp_path)
    sid = m.create()
    m.add_message(sid, "user", "x"); m.add_context(sid, {"question": "q", "rewritten": "r", "chunks": []})
    m.reset(sid)
    data = m.load(sid)
    assert data["messages"] == [] and data["context_history"] == [] and "reset_at" in data and m.exists(sid)


def test_export_markdown_contains_all_turns_and_sources(tmp_path):
    m = SessionManager(tmp_path)
    sid = m.create()
    m.add_message(sid, "user", "Q?"); m.add_message(sid, "assistant", "A!", sources=[{"source": "f.md", "chunk_id": "f_chunk_000"}])
    md = m.export_markdown(sid)
    assert "Q?" in md and "A!" in md and "f.md" in md and "f_chunk_000" in md and sid in md


def test_empty_session_export_and_window(tmp_path):
    m = SessionManager(tmp_path)
    sid = m.create()
    assert m.messages(sid) == [] and m.window(sid, 6) == [] and sid in m.export_markdown(sid)
