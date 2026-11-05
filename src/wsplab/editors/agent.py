"""A coding agent as the editor. The normal run happens in a background process; this Editor hands each message to whoever
connects to its socket and waits for their reply, so an agent gets the same counters, rules and evidence as any other editor.

An agent also has a shell, so unlike an API editor it is not tool-free. Patches record that: rules like "do not run the model
yourself" rest on the agent's honesty here.
"""

from __future__ import annotations

import json
import os
import queue
import socket
import threading
from pathlib import Path

from .base import Reply

SOCKET = "session.sock"


class AgentEditor:
    name = "agent-cli"
    tool_free = False

    def __init__(self, folder: str | os.PathLike):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / SOCKET
        self.path.unlink(missing_ok=True)
        self.replies: queue.Queue[str] = queue.Queue()
        self.messages: queue.Queue[dict] = queue.Queue()
        self.last: dict | None = None
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(str(self.path))
        self.server.listen()
        threading.Thread(target=self._serve, daemon=True).start()

    def reply(self, system: str, messages: list[dict]) -> Reply:
        self.messages.put({"message": messages[-1]["content"]})
        return Reply(self.replies.get())

    def finished(self, report: dict) -> None:
        self.messages.put({"done": True, "report": report})

    def _next(self) -> dict:
        self.last = self.messages.get()
        return self.last

    def _serve(self) -> None:
        while True:
            connection, _ = self.server.accept()
            with connection, connection.makefile("rw") as stream:
                request = json.loads(stream.readline())
                if "reply" in request:
                    self.replies.put(request["reply"])
                    answer = self._next()
                else:
                    answer = self.last or self._next()
                stream.write(json.dumps(answer) + "\n")


def talk(folder: str | os.PathLike, request: dict, timeout: float | None = None) -> dict:
    """Send one request to a running session and wait for its next message."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(str(Path(folder) / SOCKET))
        with s.makefile("rw") as stream:
            stream.write(json.dumps(request) + "\n")
            stream.flush()
            return json.loads(stream.readline())
