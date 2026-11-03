from src.memory.session_manager import SessionManager


def test_create_store_retrieve(tmp_path):
    m = SessionManager(tmp_path)
    sid = m.create()
    m.add_message(sid, "user", "hi")
    m.add_message(sid, "assistant", "hello", sources=[{"chunk_id": "c"}], metadata={"k": 1})
    msgs = m.messages(sid)
    assert [x["role"] for x in msgs] == ["user", "assistant"]
    assert msgs[1]["sources"][0]["chunk_id"] == "c" and "timestamp" in msgs[0]


def test_persists_across_simulated_rerun(tmp_path):
    sid = SessionManager(tmp_path).create()
    SessionManager(tmp_path).add_message(sid, "user", "first")
    assert SessionManager(tmp_path).messages(sid)[0]["content"] == "first"


def test_session_isolation(tmp_path):
    m = SessionManager(tmp_path)
    a, b = m.create(), m.create()
    m.add_message(a, "user", "only in a")
    assert a != b and m.messages(b) == []


def test_window_bounded_and_reset(tmp_path):
    m = SessionManager(tmp_path)
    sid = m.create()
    for i in range(10):
        m.add_message(sid, "user", f"m{i}")
    w = m.window(sid, 4)
    assert [x["content"] for x in w] == ["m6", "m7", "m8", "m9"] and set(w[0]) == {"role", "content"}
    m.reset(sid)
    assert m.messages(sid) == [] and m.exists(sid)


def test_rejects_path_traversal(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        SessionManager(tmp_path).load("../evil")


def test_export(tmp_path):
    m = SessionManager(tmp_path)
    sid = m.create()
    m.add_message(sid, "user", "question?")
    assert "question?" in m.export_markdown(sid)
