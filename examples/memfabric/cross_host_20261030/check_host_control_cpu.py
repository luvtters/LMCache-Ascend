"""Real CPU subprocess checks; no accelerator imports."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import time

spec = importlib.util.spec_from_file_location("host_control", Path(__file__).with_name("host_control.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def wait(s, ident):
    until = time.time() + 5
    while time.time() < until:
        r = s.status(ident)
        if r["state"] == "EXITED":
            return r
        time.sleep(.02)
    raise AssertionError("child did not exit")


with tempfile.TemporaryDirectory() as d:
    s = module.Supervisor(d, time.time() + 60)
    s.start(dict(id="ok", argv=[sys.executable, "-c", "print('RAW_CHILD_OUTPUT')"]))
    assert wait(s, "ok")["returncode"] == 0
    assert "RAW_CHILD_OUTPUT" in (Path(d) / "tasks/ok/process.raw.log").read_text()
    s.start(dict(id="failed", argv=[sys.executable, "-c", "raise SystemExit(7)"]))
    assert wait(s, "failed")["returncode"] == 7 and not s.halted
    try:
        s.start(dict(id="ok", argv=[sys.executable, "-c", "pass"]))
    except ValueError:
        pass
    else:
        raise AssertionError("reused attempt accepted")
    s.exclusive_until = time.time() - 1
    s.start(dict(id="interrupted", argv=[sys.executable, "-c", "raise SystemExit(9)"]))
    assert wait(s, "interrupted")["returncode"] == 9 and s.halted
    try:
        s.start(dict(id="must_not_restart", argv=[sys.executable, "-c", "pass"]))
    except RuntimeError:
        pass
    else:
        raise AssertionError("post-window restart accepted")
    assert (Path(d) / "halted.json").exists()
print("CPU_ONLY: raw output, real wait, fresh attempts, pre-window failure, post-window halt verified")
