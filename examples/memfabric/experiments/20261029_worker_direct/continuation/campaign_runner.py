"""Run a prevalidated command list without a chat turn; stdlib, PLAN_ONLY by default."""
import argparse
from collections import deque
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


def stamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Window timestamps require an explicit timezone")
    return parsed.timestamp()


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def command(argv, cwd):
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x for x in argv):
        raise ValueError("Commands must be nonempty argv arrays")
    executable = Path(argv[0])
    if not executable.is_absolute():
        executable = cwd / executable
    if not executable.is_file() and not shutil.which(argv[0]):
        raise ValueError("Missing executable: " + argv[0])
    for token in argv[1:]:
        if token.endswith(".py"):
            if "{attempt" in token:
                raise ValueError("Script paths must exist before starting; use attempt placeholders for output/group names")
            script = Path(token)
            if not script.is_absolute():
                script = cwd / script
            if not script.is_file():
                raise ValueError("Missing script: " + token)
    return [str(executable.resolve()) if executable.is_file() else shutil.which(argv[0]), *argv[1:]]


def validate(path):
    plan = json.loads(path.read_text(encoding="utf-8"))
    window = [stamp(plan[key]) for key in ("start_at", "stop_new_at", "stop_at")]
    if not window[0] < window[1] < window[2]:
        raise ValueError("Require start_at < stop_new_at < stop_at")
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("The entire nonempty step list must be supplied before starting")
    ids = set()
    for index, step in enumerate(steps):
        key = step.get("id", "")
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", key) or key in ids or key == "_quality":
            raise ValueError("Each step needs a unique simple id")
        ids.add(key)
        step["argv"] = command(step.get("argv"), path.parent)
        if not positive(step.get("timeout_seconds")):
            raise ValueError("Each step needs a positive timeout_seconds")
        handoff = step.get("handoff_exit_codes", [])
        if (not isinstance(handoff, list) or any(type(x) is not int or not 1 <= x <= 255 for x in handoff)
                or (handoff and index == len(steps) - 1)):
            raise ValueError("Handoff codes require a validated successor; zero is not a handoff")
        if "cleanup_argv" in step:
            step["cleanup_argv"] = command(step["cleanup_argv"], path.parent)
            if not positive(step.get("cleanup_timeout_seconds", 5)):
                raise ValueError("Cleanup requires a positive reserved timeout")
        if (plan.get("retry_failed", False) or "report_gate_argv" in plan) and not any(
                "{attempt}" in x or "{attempt_dir}" in x for x in step["argv"]):
            raise ValueError("Retry requires a fresh-attempt placeholder in every command")
    if type(plan.get("retry_failed", False)) is not bool:
        raise ValueError("retry_failed must be boolean")
    if "report_gate_argv" in plan:
        plan["report_gate_argv"] = command(plan["report_gate_argv"], path.parent)
        if not positive(plan.get("report_gate_timeout_seconds", 5)):
            raise ValueError("Report gate requires a positive timeout")
    return plan, window


def write(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def expanded(argv, attempt, folder):
    return [x.replace("{attempt}", str(attempt)).replace("{attempt_dir}", str(folder)) for x in argv]


def execute(argv, cwd, folder, label, timeout, hard_stop, kill_stop=None, latest_start=None, child_env=None):
    record = dict(argv=argv, started=time.time(), pid=None, returncode=None, timed_out=False)
    remaining = min(timeout, hard_stop - time.time())
    kill_stop = hard_stop if kill_stop is None else kill_stop
    if remaining <= 0:
        record.update(not_started="deadline", finished=time.time())
        return record
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with (folder / (label + ".log")).open("xb") as log:
        # Recheck after filesystem work: it may have crossed the start deadline.
        remaining = min(timeout, hard_stop - time.time())
        if remaining <= 0 or (latest_start is not None and time.time() >= latest_start):
            record.update(not_started="deadline", finished=time.time())
            write(folder / (label + ".json"), record)
            return record
        child = subprocess.Popen(argv, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=os.name != "nt", creationflags=flags, env=child_env)
        record["pid"] = child.pid
        write(folder / (label + ".json"), record)
        try:
            record["returncode"] = child.wait(timeout=max(.001, min(remaining, hard_stop - time.time())))
        except subprocess.TimeoutExpired:
            record["timed_out"] = True
            child.terminate()
            try:
                child.wait(timeout=max(0, min(.25, kill_stop - time.time())))
            except subprocess.TimeoutExpired:
                child.kill()
                try:
                    child.wait(timeout=max(0, min(.25, kill_stop - time.time())))
                except subprocess.TimeoutExpired:
                    pass
            record["returncode"] = child.poll()
        record["child_reaped"] = child.poll() is not None
    record["finished"] = time.time()
    write(folder / (label + ".json"), record)
    return record


def run(plan, window, plan_path, output):
    start, stop_new, stop_at = window
    child_env = os.environ.copy()
    child_env.update(MF_AUTHORIZED_START_AT=plan["start_at"], MF_AUTHORIZED_STOP_NEW_AT=plan["stop_new_at"])
    status = dict(state="READY", executor_pid=os.getpid(), started=time.time(), attempts=[], pending=[],
                  report_ready=False, report_gate_attempts=[])
    write(output / "ready.json", dict(state="READY", executor_pid=os.getpid(),
                                     validated_steps=[s["id"] for s in plan["steps"]], window=window))
    write(output / "status.json", status)
    if not start <= time.time() < stop_new:
        status.update(state="OUTSIDE_START_WINDOW", finished=time.time())
        write(output / "status.json", status)
        return 2
    queue = deque((step, 1) for step in plan["steps"])
    attempts = {}
    failures = False
    while queue:
        step, number = queue.popleft()
        reserve = step.get("cleanup_timeout_seconds", 5) + .6 if "cleanup_argv" in step else .6
        if time.time() >= stop_new or time.time() + reserve >= stop_at:
            queue.appendleft((step, number))
            status["state"] = "DEADLINE_REACHED"
            break
        folder = output / step["id"] / ("attempt" + str(number))
        folder.mkdir(parents=True, exist_ok=False)
        attempts[step["id"]] = number
        item = dict(id=step["id"], attempt=number, state="RUNNING", output_dir=str(folder))
        status["attempts"].append(item)
        status.update(state="RUNNING", pending=[dict(id=s["id"], attempt=n) for s, n in queue])
        write(output / "status.json", status)
        try:
            result = execute(expanded(step["argv"], number, folder), plan_path.parent, folder, "command",
                             step["timeout_seconds"], stop_at - reserve, stop_at - .1,
                             latest_start=stop_new, child_env=child_env)
            item["command"] = result
            if result.get("not_started") == "deadline":
                item["state"] = "NOT_STARTED_DEADLINE"
                queue.appendleft((step, number))
                status["state"] = "DEADLINE_REACHED"
                break
            code = result["returncode"]
            if not result["timed_out"] and code == 0:
                item["state"] = "PASSED"
            elif not result["timed_out"] and code in step.get("handoff_exit_codes", []):
                item["state"] = "HANDOFF"
                if "cleanup_argv" in step:
                    cleanup = execute(expanded(step["cleanup_argv"], number, folder), plan_path.parent, folder,
                                      "cleanup", step.get("cleanup_timeout_seconds", 5), stop_at - .1,
                                      child_env=child_env)
                    item["cleanup"] = cleanup
                    item["cleanup_verified"] = cleanup["returncode"] == 0 and not cleanup["timed_out"]
                    if not item["cleanup_verified"]:
                        item["state"] = "HANDOFF_BLOCKED"
                        status["state"] = "NEEDS_ATTENTION"
                        break
            else:
                failures = True
                item["state"] = "FAILED"
                if "cleanup_argv" not in step:
                    item["cleanup_verified"] = False
                else:
                    cleanup = execute(expanded(step["cleanup_argv"], number, folder), plan_path.parent, folder,
                                      "cleanup", step.get("cleanup_timeout_seconds", 5), stop_at - .1,
                                      child_env=child_env)
                    item["cleanup"] = cleanup
                    item["cleanup_verified"] = (cleanup["returncode"] == 0 and not cleanup["timed_out"]
                                                and result.get("child_reaped") is True)
                if not item["cleanup_verified"]:
                    status["state"] = "NEEDS_ATTENTION"
                    break
                if plan.get("retry_failed", False):
                    queue.append((step, number + 1))
        except Exception as error:
            item.update(state="FAILED", error=repr(error))
            status["state"] = "NEEDS_ATTENTION"
            break
        finally:
            status["updated"] = time.time()
            write(output / "status.json", status)
        if not queue and "report_gate_argv" in plan:
            status["state"] = "CHECKING_REPORT"
            write(output / "status.json", status)
            gate_folder = output / "_quality" / ("attempt" + str(len(status["report_gate_attempts"]) + 1))
            gate_folder.mkdir(parents=True, exist_ok=False)
            try:
                gate = execute(plan["report_gate_argv"], plan_path.parent, gate_folder, "gate",
                               plan.get("report_gate_timeout_seconds", 5), stop_at - .1, child_env=child_env)
            except Exception as error:
                status.update(state="NEEDS_ATTENTION", report_gate_error=repr(error))
                break
            status["report_gate_attempts"].append(gate)
            status["report_ready"] = gate["returncode"] == 0 and not gate["timed_out"]
            if not status["report_ready"]:
                queue.extend((next_step, attempts[next_step["id"]] + 1) for next_step in plan["steps"])
                if time.time() >= stop_new:
                    status["state"] = "DEADLINE_REACHED"
                    break
    else:
        status["state"] = "FINISHED_WITH_FAILURES" if failures else "COMPLETE"
    status.update(finished=time.time(), pending=[dict(id=s["id"], attempt=n) for s, n in queue])
    write(output / "status.json", status)
    return 0 if status["state"] == "COMPLETE" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    path = args.plan.resolve()
    try:
        plan, window = validate(path)
    except Exception as error:
        if args.run:
            args.output_dir.mkdir(parents=True, exist_ok=False)
            write(args.output_dir / "status.json", dict(state="INVALID_PLAN", error=repr(error), attempts=[]))
        print(json.dumps(dict(state="INVALID_PLAN", error=str(error))))
        return 2
    if not args.run:
        print(json.dumps(dict(state="PLAN_ONLY", validated_steps=[s["id"] for s in plan["steps"]], window=window)))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    return run(plan, window, path, args.output_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
