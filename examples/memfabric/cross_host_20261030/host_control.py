"""Private-container child supervisor, raw wait records, and 06:00 yield policy.

This CPU-only helper never imports torch/ACL/BM and never kills foreign jobs.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


class Supervisor:
    def __init__(self, root, exclusive_until):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.exclusive_until = exclusive_until
        self.lock = threading.RLock()
        self.tasks = {}
        self.halted = (self.root / "halted.json").exists()

    def policy(self):
        return dict(time=time.time(), exclusive_until=self.exclusive_until,
                    phase="exclusive" if time.time() < self.exclusive_until else "yield",
                    halted=self.halted, supervisor_pid=os.getpid())

    def start(self, request):
        ident = request["id"]
        if not isinstance(ident, str) or not ident or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in ident):
            raise ValueError("Invalid task id")
        argv = request["argv"]
        if not isinstance(argv, list) or not argv or not all(isinstance(v, str) for v in argv):
            raise ValueError("argv must be a nonempty string list")
        cwd = Path(request.get("cwd", self.root)).resolve()
        if not cwd.is_relative_to(self.root) or not cwd.is_dir():
            raise ValueError("cwd must exist inside this personal run root")
        with self.lock:
            if self.halted:
                raise RuntimeError("Campaign halted; no new task may be started")
            if ident in self.tasks or (self.root / "tasks" / ident).exists():
                raise ValueError("Task id/attempt must be fresh")
            folder = self.root / "tasks" / ident
            folder.mkdir(parents=True)
            env = os.environ.copy()
            env.update(request.get("env", {}))
            log = open(folder / "process.raw.log", "wb")
            try:
                proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
            except BaseException:
                log.close()
                raise
            record = dict(id=ident, pid=proc.pid, started=time.time(), argv=argv,
                          cwd=str(cwd), planned_stop=False, state="RUNNING",
                          persistent=bool(request.get("persistent", False)),
                          log=str(folder / "process.raw.log"))
            item = dict(proc=proc, log=log, record=record, folder=folder)
            self.tasks[ident] = item
            dump(folder / "started.json", record)
            threading.Thread(target=self._wait, args=(ident,), daemon=True).start()
            return dict(record)

    def _wait(self, ident):
        item = self.tasks[ident]
        rc = item["proc"].wait()
        item["log"].close()
        with self.lock:
            record = item["record"]
            record.update(state="EXITED", returncode=rc, signal=-rc if rc < 0 else None,
                          ended=time.time())
            dump(item["folder"] / "wait.raw.json", record)
            if (rc != 0 or record["persistent"]) and not record["planned_stop"] and time.time() >= self.exclusive_until:
                self.halt("unplanned_child_exit_after_exclusive_window", ident)

    def halt(self, reason, ident=None):
        with self.lock:
            self.halted = True
            dump(self.root / "halted.json", dict(time=time.time(), reason=reason, task=ident))

    def status(self, ident=None):
        with self.lock:
            if ident is not None:
                return dict(self.tasks[ident]["record"])
            return dict(policy=self.policy(), tasks={k: dict(v["record"]) for k, v in self.tasks.items()})

    def stop(self, ident, grace=30, reason="controlled_case_shutdown"):
        with self.lock:
            item = self.tasks[ident]
            proc = item["proc"]
            if proc.poll() is not None:
                return dict(item["record"])
            item["record"]["planned_stop"] = True
            item["record"]["stop_reason"] = reason
            dump(item["folder"] / "stop_requested.json", dict(time=time.time(), reason=reason, pid=proc.pid))
        for sig, delay in ((signal.SIGINT, grace), (signal.SIGTERM, 8), (signal.SIGKILL, 5)):
            if proc.poll() is not None:
                break
            with open(item["folder"] / "signals.raw.jsonl", "a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(time=time.time(), pid=proc.pid, signal=int(sig), reason=reason)) + "\n")
            try:
                if os.name == "posix":
                    os.killpg(proc.pid, sig)
                else:
                    proc.terminate()
            except ProcessLookupError:
                break
            try:
                proc.wait(timeout=delay)
            except subprocess.TimeoutExpired:
                continue
        time.sleep(.05)
        return self.status(ident)


def handler(supervisor):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body):
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            try:
                if self.path == "/policy":
                    return self.send(200, supervisor.policy())
                if self.path == "/tasks":
                    return self.send(200, supervisor.status())
                if self.path.startswith("/tasks/"):
                    return self.send(200, supervisor.status(self.path.rsplit("/", 1)[1]))
                self.send(404, {"error": "unknown path"})
            except KeyError as exc:
                self.send(404, {"error": str(exc)})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 131072:
                    raise ValueError("Invalid body size")
                request = json.loads(self.rfile.read(length))
                if self.path == "/start":
                    return self.send(200, supervisor.start(request))
                if self.path == "/stop":
                    return self.send(200, supervisor.stop(request["id"], min(120, max(0, request.get("grace", 30))), request.get("reason", "controlled_case_shutdown")))
                if self.path == "/halt":
                    supervisor.halt(request["reason"])
                    return self.send(200, supervisor.policy())
                self.send(404, {"error": "unknown path"})
            except (ValueError, RuntimeError, KeyError) as exc:
                self.send(409, {"error": str(exc)})

        def log_message(self, fmt, *args):
            if self.command == "POST":
                super().log_message(fmt, *args)
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29870)
    parser.add_argument("--exclusive-until", required=True)
    args = parser.parse_args()
    supervisor = Supervisor(args.root, datetime.fromisoformat(args.exclusive_until).timestamp())
    server = ThreadingHTTPServer((args.bind, args.port), handler(supervisor))
    dump(Path(args.root) / "host_control.ready.json", dict(supervisor.policy(), bind=args.bind, port=args.port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for ident in list(supervisor.tasks):
            supervisor.stop(ident, 15, "supervisor_shutdown")
        server.server_close()


if __name__ == "__main__":
    main()
