# SPDX-License-Identifier: Apache-2.0
"""Container-local child supervision; never connects to a host or escalates signals.

Run:  python supervisor.py run --spec /private/spec.json
Stop: python supervisor.py stop --case /private/case --role service

The spec contains case, role, argv, and optional env, env_script,
pythonpath_prefix. Signal the recorded child PGID, not the supervisor's group.
Only a completed child.wait() produces ROLE.exit.json. If the supervisor is
interrupted, ROLE.supervisor-exit.json explicitly leaves child exit unknown.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def stamp():
    return dict(time=time.time(), utc=datetime.now(timezone.utc).isoformat(),
                monotonic_ns=time.monotonic_ns())


def write_json(path, record):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def event(case, role, name, **details):
    record = dict(stamp(), event=name, role=role, sender_pid=os.getpid(), **details)
    data = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    descriptor = os.open(case / "signals.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, data)
    finally:
        os.close(descriptor)


def proc_state(pid):
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return dict(state=fields[0], proc_start_ticks=fields[19])


def valid_role(role):
    if not isinstance(role, str) or re.fullmatch(r"[A-Za-z0-9_-]+", role) is None:
        raise ValueError("role must contain only letters, digits, '_' or '-'")
    return role


def child_command(spec):
    argv = spec["argv"]
    environment = spec.get("env", {})
    prefix = spec.get("pythonpath_prefix", "")
    env_script = spec.get("env_script")
    if not isinstance(argv, list) or not argv or any(not isinstance(v, str) for v in argv):
        raise ValueError("argv must be a nonempty list of strings")
    if not isinstance(environment, dict) or any(
        re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None or not isinstance(value, str)
        for key, value in environment.items()
    ):
        raise ValueError("env must map shell variable names to strings")
    if not isinstance(prefix, str) or (env_script is not None and not isinstance(env_script, str)):
        raise ValueError("env_script and pythonpath_prefix must be strings")
    inherited = os.environ.copy()
    inherited.update(environment)
    if env_script:
        # Positional arguments preserve literal spaces, dollars and quotes.
        shell = '''source "$1" || exit $?
shift
_mf_prefix="$1"; shift
_mf_count="$1"; shift
while [ "$_mf_count" -gt 0 ]; do
    export "$1"; shift; _mf_count=$((_mf_count - 1))
done
if [ -n "$_mf_prefix" ]; then
    export PYTHONPATH="$_mf_prefix${PYTHONPATH:+:$PYTHONPATH}"
fi
exec "$@"
'''
        assignments = [key + "=" + value for key, value in environment.items()]
        return ["bash", "-c", shell, "memfabric-supervisor", env_script,
                prefix, str(len(assignments)), *assignments, *argv], inherited
    if prefix:
        inherited["PYTHONPATH"] = prefix + (
            ":" + inherited["PYTHONPATH"] if inherited.get("PYTHONPATH") else "")
    return argv, inherited


class SupervisorInterrupted(BaseException):
    def __init__(self, signum):
        self.signum = signum


def run(spec_path):
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    case, role = Path(spec["case"]).resolve(), valid_role(spec["role"])
    command, environment = child_command(spec)
    case.mkdir(parents=True, exist_ok=True)
    for suffix in ("log", "process.json", "exit.json", "supervisor-exit.json"):
        if (case / (role + "." + suffix)).exists():
            raise FileExistsError("Use a fresh case/role; refusing to overwrite " + suffix)

    def interrupted(signum, _frame):
        event(case, role, "supervisor_signal", signal=signum)
        raise SupervisorInterrupted(signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    child = None
    try:
        with (case / (role + ".log")).open("xb") as log:
            child = subprocess.Popen(command, env=environment, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            state = proc_state(child.pid)
            process = dict(stamp(), role=role, case=str(case),
                           supervisor_pid=os.getpid(), child_pid=child.pid, pid=child.pid,
                           pgid=os.getpgid(child.pid), proc_start_ticks=state["proc_start_ticks"],
                           argv=spec["argv"], env=spec.get("env", {}),
                           env_script=spec.get("env_script"),
                           pythonpath_prefix=spec.get("pythonpath_prefix", ""))
            write_json(case / (role + ".process.json"), process)
            print(json.dumps(process), flush=True)
            returncode = child.wait()
        result = dict(stamp(), role=role, child_pid=child.pid,
                      supervisor_pid=os.getpid(), child_exit_observed=True,
                      returncode=returncode, signal=-returncode if returncode < 0 else None,
                      normal_exit=returncode == 0)
        write_json(case / (role + ".exit.json"), result)
        print(json.dumps(result), flush=True)
        return returncode if returncode >= 0 else 128 - returncode
    except SupervisorInterrupted as error:
        result = dict(stamp(), role=role, supervisor_pid=os.getpid(),
                      supervisor_signal=error.signum, child_pid=child.pid if child else None,
                      child_exit_observed=False, returncode=None, normal_exit=False)
        write_json(case / (role + ".supervisor-exit.json"), result)
        print(json.dumps(result), flush=True)
        return 128 + error.signum
    except Exception as error:
        write_json(case / (role + ".supervisor-exit.json"),
                   dict(stamp(), role=role, supervisor_pid=os.getpid(),
                        child_pid=child.pid if child else None, child_exit_observed=False,
                        returncode=None, normal_exit=False, error=repr(error)))
        raise


def stop(case_path, role):
    case, role = Path(case_path).resolve(), valid_role(role)
    record = json.loads((case / (role + ".process.json")).read_text(encoding="utf-8"))
    pid, pgid = record["child_pid"], record["pgid"]
    if type(pid) is not int or pid <= 1 or pgid != pid:
        raise ValueError("Expected an owned child with an independent process group")
    state = proc_state(pid)
    if state is None:
        event(case, role, "already_exited", child_pid=pid)
        return 0
    if state["proc_start_ticks"] != record["proc_start_ticks"]:
        raise RuntimeError("PID was reused; refusing to signal")
    if state["state"] == "Z":
        event(case, role, "already_exited", child_pid=pid, state="Z")
        return 0
    if os.getpgid(pid) != pgid:
        raise RuntimeError("Child PGID changed; refusing to signal")
    event(case, role, "signal_attempt", child_pid=pid, target_pgid=pgid, signal=signal.SIGINT)
    try:
        os.killpg(pgid, signal.SIGINT)
    except ProcessLookupError:
        event(case, role, "already_exited", child_pid=pid)
    else:
        event(case, role, "signal_sent", child_pid=pid, target_pgid=pgid, signal=signal.SIGINT)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    launch = actions.add_parser("run")
    launch.add_argument("--spec", required=True)
    halt = actions.add_parser("stop")
    halt.add_argument("--case", required=True)
    halt.add_argument("--role", required=True)
    args = parser.parse_args()
    return run(args.spec) if args.action == "run" else stop(args.case, args.role)


if __name__ == "__main__":
    sys.exit(main())
