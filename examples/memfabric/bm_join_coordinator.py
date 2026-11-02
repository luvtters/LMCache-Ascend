#!/usr/bin/env python3
"""Small CPU-only HTTP coordinator for BMSession after-mmap JOIN events.

Implements only the protocol used by ``BMSession.wait_fixed_group_ready``:
POST /runs/<group_id>/events/<event_name> and GET
/runs/<group_id>/events. Duplicate identical events are idempotent; conflicting
events return HTTP 409.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
import os
import re
import signal
import threading
from urllib.parse import unquote, urlsplit


_RUN_PATH = re.compile(r"^/runs/([^/]+)/events(?:/([^/]+))?$")


class _Coordinator(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address):
        super().__init__(address, _Handler)
        self._events: dict[str, dict[str, dict]] = {}
        self._events_lock = threading.Lock()


class _Handler(BaseHTTPRequestHandler):
    server: _Coordinator

    def log_message(self, _format, *_args):
        return

    def _json(self, status: int, body: dict):
        payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _route(self):
        path = unquote(urlsplit(self.path).path)
        match = _RUN_PATH.fullmatch(path)
        if match is None:
            return None
        return match.group(1), match.group(2)

    def do_GET(self):
        route = self._route()
        if route is None or route[1] is not None:
            self._json(404, {"error": "unknown route"})
            return
        group_id, _ = route
        with self.server._events_lock:
            events = dict(self.server._events.get(group_id, {}))
        self._json(200, {"events": events})

    def do_POST(self):
        route = self._route()
        if route is None or route[1] is None:
            self._json(404, {"error": "unknown route"})
            return
        group_id, event_name = route
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 65536:
                raise ValueError("invalid body length")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("event must be an object")
            rank = payload.get("rank")
            if (payload.get("name") != event_name or type(rank) is not int
                    or event_name != f"bm-joined-{rank}"
                    or payload.get("protocol") != "bm-after-mmap-join-v1"
                    or not isinstance(payload.get("nonce"), str)
                    or not payload["nonce"]
                    or not isinstance(payload.get("contract"), dict)):
                raise ValueError("event does not match the BM JOIN contract")
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
            return

        with self.server._events_lock:
            events = self.server._events.setdefault(group_id, {})
            prior = events.get(event_name)
            if prior is None:
                events[event_name] = payload
                status = 201
            elif prior == payload:
                status = 200
            else:
                self._json(409, {"error": "conflicting JOIN event"})
                return
        self._json(status, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    # Background launchers may pass SIGINT=SIG_IGN to Python. Restore our
    # explicit graceful-stop contract before publishing the ready event.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    server = _Coordinator((args.host, args.port))
    print(json.dumps({
        "event": "BM_JOIN_COORDINATOR_READY",
        "host": args.host,
        "port": args.port,
        "pid": os.getpid(),
    }), flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
