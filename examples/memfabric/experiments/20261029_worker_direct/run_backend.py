#!/usr/bin/env python3
"""One container-local backend round; plans by default, no SSH or daemon agent."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request
import control


def shutdown_contract(state, events, shutdown_started, sender_pid):
    """Classify observed exits without changing their native return codes."""
    result = {}
    for role, item in state["roles"].items():
        observed = item.get("exit.json", {})
        process = item.get("process") or {}
        code = observed.get("returncode")
        known = (observed.get("child_exit_observed") is True
                 and not item.get("supervisor-exit.json")
                 and observed.get("child_pid") == process.get("child_pid")
                 and process.get("child_pid") is not None)
        sent_by_runner = any(
            event.get("event") == "signal_sent" and event.get("role") == role
            and event.get("child_pid") == process.get("child_pid")
            and event.get("target_pgid") == process.get("pgid")
            and event.get("sender_pid") == sender_pid and event.get("signal") == signal.SIGINT
            and isinstance(event.get("time"), (int, float)) and event["time"] >= shutdown_started
            for event in events)
        expected = known and (code == 0 or (role == "service" and code == 130 and sent_by_runner))
        result[role] = dict(returncode=code, child_exit_observed=observed.get("child_exit_observed"),
                            normal_exit=observed.get("normal_exit"),
                            runner_sigint_observed=sent_by_runner,
                            expected_controlled_exit=expected,
                            classification=("normal_zero" if expected and code == 0 else
                                            "service_sigint_cli_130" if expected else "unexpected_or_unverified"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("A", "B", "C"), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--existing", action="store_true")
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--pool-gb", type=int, default=1)
    parser.add_argument("--warmup-requests", type=int, default=4)
    parser.add_argument("--warmup-done", type=Path)
    parser.add_argument("--concurrencies", default="1,4")
    parser.add_argument("--dataset", type=Path, default=Path(str(control.ROOT / "bench1373/data/custom.jsonl")))
    parser.add_argument("--input-tokens", type=int, default=1373)
    parser.add_argument("--expected-external-tokens", type=int, default=1280)
    parser.add_argument("--output-root", type=Path, default=Path(str(control.ROOT / "bench1373")))
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    concurrencies = [int(value) for value in args.concurrencies.split(",")]
    if not concurrencies or set(concurrencies) - {1, 4} or len(set(concurrencies)) != len(concurrencies):
        parser.error("--concurrencies must be 1, 4, or 1,4 without duplicates")
    case = Path(str(control.case_path(args.name)))
    runtime = str(control.RUNTIME)
    external = 0 if args.case == "A" else args.expected_external_tokens
    seed = [control.PY, runtime + "/seed_check.py", "--case", args.case, "--run-dir", str(case),
            "--dataset", str(args.dataset), "--input-tokens", str(args.input_tokens),
            "--expected-external-tokens", str(external), "--seed-unique", "--run"]
    benches = [[control.PY, runtime + "/bench_d.py", "--case", args.case, "--round", str(args.round),
                "--requests", str(args.requests), "--output-tokens", str(args.output_tokens), "--concurrency", str(concurrency),
                "--dataset", str(args.dataset), "--input-tokens", str(args.input_tokens),
                "--expected-external-tokens", str(external), "--tokenizer", str(control.MODEL),
                "--model", "mf-night", "--base-url", "http://127.0.0.1:28402",
                "--p-url", "http://127.0.0.1:28401", "--service-url", "http://127.0.0.1:28195",
                "--output-root", str(args.output_root), "--run"]
               + (["--counter-url", "http://127.0.0.1:29402/counters"] if args.case == "C" else [])
               for concurrency in concurrencies]
    warmup_root = Path(str(control.ROOT / f"warmups{args.input_tokens}" / args.name))
    warmup = list(benches[0])
    for flag, value in (("--requests", str(args.warmup_requests)), ("--concurrency", "1"),
                        ("--round", "1"), ("--output-root", str(warmup_root))):
        warmup[warmup.index(flag) + 1] = value
    print(json.dumps(dict(mode="RUN" if args.run else "PLAN_ONLY", existing=args.existing,
                          case=str(case), seed=None if args.existing else seed,
                          warmup=None if args.existing else warmup,
                          warmup_done=str(args.warmup_done) if args.warmup_done else None,
                          benches=benches), indent=2), flush=True)
    if not args.run:
        return 0
    if args.warmup_requests < 1:
        parser.error("--warmup-requests must be positive")
    if args.existing:
        if args.case != "C" or args.warmup_done is None:
            parser.error("Existing C requires --warmup-done with its accepted pilot summary")
        prior = json.loads(args.warmup_done.read_text(encoding="utf-8"))
        if not (prior.get("accepted_for_comparison") is True and prior.get("case") == "C"
                and prior.get("requests") == args.warmup_requests
                and prior.get("input_tokens_per_request") == args.input_tokens
                and prior.get("output_tokens_per_request") == args.output_tokens):
            raise ValueError("Existing warmup evidence does not match this full-length condition")
    elif args.warmup_done is not None or (warmup_root / f"{args.case}_c1_r1").exists():
        raise ValueError("New groups require their own fresh full-length warmup")
    for concurrency in concurrencies:
        if (args.output_root / f"{args.case}_c{concurrency}_r{args.round}").exists():
            raise FileExistsError("Refusing an existing benchmark directory")
    if (case / "backend_progress.json").exists() or (case / "backend.log").exists():
        raise FileExistsError("Use a fresh backend case/progress directory")
    if args.existing:
        _, plan = control.load_case(args.name)
        if plan["kind"] != args.case or not json.loads((case / "consumer_check/summary.json").read_text())["passed"]:
            raise RuntimeError("Existing group kind or consumer preflight is not accepted")
    elif case.exists():
        raise FileExistsError("New groups require a fresh case")
    progress = dict(case=args.case, name=args.name, round=args.round, events=[], success=False)
    owned, log = args.existing, None

    def record(stage, **fields):
        progress["events"].append(dict(stage=stage, time=time.time(), **fields))
        progress["stage"] = stage
        (case / "backend_progress.json").write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
        if log:
            log.write((json.dumps(progress["events"][-1]) + "\n").encode()); log.flush()

    def window():
        if not control.BEGIN <= time.time() < control.NEW_WORK_END:
            raise RuntimeError("Outside the allowed new-experiment window")

    def healthy(url):
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                return response.status == 200
        except OSError:
            return False

    try:
        window()
        if not args.existing:
            owned = True
            control.start(control.build_plan(args.case, args.name, pool_gb=args.pool_gb))
        log = (case / "backend.log").open("xb")
        record("WAIT_READY")
        deadline = time.monotonic() + 450
        while True:
            window()
            state = control.status(args.name)
            dead = [role for role, item in state["roles"].items()
                    if item.get("exit.json") or item.get("supervisor-exit.json")
                    or (item["process"] and (not item["running"] or not item["identity_matches"]))]
            if dead:
                raise RuntimeError("Group roles exited before ready: " + ",".join(dead))
            if all(healthy(url) for url in ("http://127.0.0.1:28401/health", "http://127.0.0.1:28402/health",
                                            "http://127.0.0.1:28195/healthcheck")):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("P/D/service readiness exceeded 450 seconds")
            time.sleep(1)
        if not args.existing:
            record("CONSUMER_CHECK")
            code = subprocess.run(seed, stdout=log, stderr=subprocess.STDOUT).returncode
            record("CONSUMER_CHECK_EXIT", returncode=code)
            if code:
                raise RuntimeError("Consumer check failed")
            window()
            record("FULL_LENGTH_WARMUP", requests=args.warmup_requests, argv=warmup)
            code = subprocess.run(warmup, stdout=log, stderr=subprocess.STDOUT).returncode
            record("FULL_LENGTH_WARMUP_EXIT", returncode=code, output_root=str(warmup_root))
            if code:
                raise RuntimeError("Full-length warmup failed")
        else:
            record("FULL_LENGTH_WARMUP_ALREADY_DONE", summary=str(args.warmup_done))
        for concurrency, argv in zip(concurrencies, benches):
            window()
            record("BENCHMARK", concurrency=concurrency, argv=argv)
            code = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT).returncode
            record("BENCHMARK_EXIT", concurrency=concurrency, returncode=code)
            if code:
                raise RuntimeError("Benchmark failed; preserving the interrupted/invalid condition")
        progress["success"] = True
    except BaseException as error:
        progress["error"] = repr(error)
        if case.exists():
            record("FAILED", error=repr(error))
    finally:
        if owned and case.exists():
            try:
                shutdown_started = time.time()
                progress["shutdown_started_unix"] = shutdown_started
                record("STOPPING")
                progress["stop_result"] = control.stop(args.name, None, 30)
                stop_state = progress["stop_result"]["state"]
                end = time.monotonic() + 5
                while any("exit.json" not in item for item in stop_state["roles"].values()) and time.monotonic() < end:
                    time.sleep(.1); stop_state = control.status(args.name)
                progress["final_state"] = stop_state
                signal_path = case / "signals.jsonl"
                events = [json.loads(line) for line in signal_path.read_text(encoding="utf-8").splitlines()
                          if line.strip()] if signal_path.exists() else []
                contract = shutdown_contract(stop_state, events, shutdown_started, os.getpid())
                progress["shutdown_contract"] = contract
                accepted = all(item["expected_controlled_exit"] for item in contract.values())
                progress["success"] = progress["success"] and not progress["stop_result"]["still_running"] and accepted
            except BaseException as error:
                progress.update(success=False, stop_error=repr(error))
            record("COMPLETE" if progress["success"] else "FAILED_OR_EXIT_UNVERIFIED")
        if log:
            log.close()
    print(json.dumps(progress, indent=2), flush=True)
    return 0 if progress["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
