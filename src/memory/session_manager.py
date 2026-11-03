"""Per-session conversation memory, persisted as one JSON file per session."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SessionManager:
    def __init__(self, sessions_dir: Path):
        self.dir = Path(sessions_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        if not session_id.replace("-", "").isalnum():
            raise ValueError("invalid session id")
        return self.dir / f"{session_id}.json"

    def create(self) -> str:
        sid = uuid.uuid4().hex[:12]
        self._write(sid, {"session_id": sid, "created_at": _now(), "messages": [], "context_history": []})
        return sid

    def exists(self, session_id: str) -> bool:
        return self._path(session_id).exists()

    def _fresh(self, session_id: str) -> dict:
        return {"session_id": session_id, "created_at": _now(), "messages": [], "context_history": []}

    def load(self, session_id: str) -> dict:
        """Load a session. A damaged/invalid file is quarantined (renamed *.corrupt) and replaced by an empty session
        so one bad file can never crash the UI."""
        p = self._path(session_id)
        if not p.exists():
            return self._fresh(session_id)
        try:
            data = json.loads(p.read_text())
            if isinstance(data, dict) and isinstance(data.get("messages"), list):
                data.setdefault("context_history", [])
                data.setdefault("created_at", _now())
                return data
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            pass
        try:
            p.replace(p.with_name(f"{p.name}.corrupt-{_now().replace(':', '-')}"))
        except OSError:
            pass
        return self._fresh(session_id)

    def _write(self, session_id: str, data: dict) -> None:
        self._path(session_id).write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def add_message(self, session_id: str, role: str, content: str, sources: list[dict] | None = None,
                    metadata: dict | None = None) -> dict:
        data = self.load(session_id)
        msg = {"timestamp": _now(), "role": role, "content": content, "sources": sources or [],
               "metadata": metadata or {}}
        data["messages"].append(msg)
        self._write(session_id, data)
        return msg

    def add_context(self, session_id: str, entry: dict) -> None:
        data = self.load(session_id)
        data.setdefault("context_history", []).append({"timestamp": _now(), **entry})
        self._write(session_id, data)

    def messages(self, session_id: str) -> list[dict]:
        return self.load(session_id)["messages"]

    def window(self, session_id: str, n: int = 6) -> list[dict]:
        """Bounded recent history (role + content only) for follow-up rewriting."""
        if n <= 0:
            return []
        return [{"role": m["role"], "content": m["content"]} for m in self.messages(session_id)[-n:]]

    def reset(self, session_id: str) -> None:
        data = self.load(session_id)
        data["messages"], data["context_history"] = [], []
        data["reset_at"] = _now()
        self._write(session_id, data)

    def export_markdown(self, session_id: str) -> str:
        data = self.load(session_id)
        lines = [f"# Session {session_id}", f"_Created {data.get('created_at')}_", ""]
        for m in data["messages"]:
            lines.append(f"**{m['role'].title()}** · {m['timestamp']}\n\n{m['content']}\n")
            for s in m.get("sources", []):
                lines.append(f"- source: {s.get('source')} / {s.get('chunk_id')}")
            lines.append("")
        return "\n".join(lines)
