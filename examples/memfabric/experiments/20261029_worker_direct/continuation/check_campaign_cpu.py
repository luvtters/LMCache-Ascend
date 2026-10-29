"""Small real-process checks: no network, Torch, MemFabric, model, or device calls."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "campaign_runner.py"
FIXTURE = '''from pathlib import Path
import sys,time
operation,target,delay,code,attempt=sys.argv[1:]
time.sleep(float(delay))
Path(target).write_text(operation+":"+attempt,encoding="utf-8")
raise SystemExit(int(code) if operation!="fail_once" or attempt=="1" else 0)
'''


def iso(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def check():
    checks = {}
    with tempfile.TemporaryDirectory(prefix="campaign-cpu-") as temporary:
        root = Path(temporary)
        fixture = root / "fixture.py"
        fixture.write_text(FIXTURE, encoding="utf-8")

        def step(key, code=0, delay=0, **extra):
            return dict(id=key, argv=[sys.executable, str(fixture), "marker", str(root/(key+".txt")),
                                     str(delay), str(code), "1"], timeout_seconds=5, **extra)

        def plan(name, steps, retry=False, times=None):
            now = time.time()
            start, stop_new, stop = times or (now-1, now+15, now+17)
            path = root / (name+".json")
            path.write_text(json.dumps(dict(start_at=iso(start), stop_new_at=iso(stop_new), stop_at=iso(stop),
                                            steps=steps, retry_failed=retry)), encoding="utf-8")
            return path

        def invoke(name, definition):
            output = root / name
            result = subprocess.run([sys.executable, str(RUNNER), "--plan", str(definition),
                                     "--output-dir", str(output), "--run"], capture_output=True, timeout=10)
            return result.returncode, json.loads((output/"status.json").read_text()), output

        # The launcher exits immediately; it owns no loop that can drive the next step.
        definition = plan("detached", [step("one", delay=.1),
                           step("handoff", code=3, handoff_exit_codes=[3]), step("three")])
        output = root / "detached-results"
        launcher = root / "launcher.py"
        launcher.write_text("import subprocess,sys,os\n" +
            "args="+repr([sys.executable, str(RUNNER), "--plan", str(definition), "--output-dir", str(output), "--run"])+"\n"+
            "with open("+repr(str(root/"detached.log"))+",'wb') as log:\n"+
            " p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=log,stderr=log,"+
            "start_new_session=os.name!='nt',creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)\n"+
            "print(p.pid)\n", encoding="utf-8")
        launcher_result = subprocess.run([sys.executable, str(launcher)], capture_output=True, timeout=3)
        assert launcher_result.returncode == 0
        deadline = time.monotonic()+7
        final = None
        while time.monotonic() < deadline:
            if (output/"status.json").exists():
                final = json.loads((output/"status.json").read_text())
                if "finished" in final:
                    break
            time.sleep(.03)
        assert final and final["state"] == "COMPLETE", final
        assert [row["state"] for row in final["attempts"]] == ["PASSED", "HANDOFF", "PASSED"]
        assert (root/"three.txt").exists()
        checks["launcher_exit_does_not_stop_runner"] = True
        checks["declared_exit3_handoff_runs_validated_successor"] = True

        for cleanup_code in (0, 8):
            key = "handoff_cleanup_"+str(cleanup_code)
            release = step(key+"_release", code=cleanup_code)["argv"]
            definition = plan(key, [step(key, code=3, handoff_exit_codes=[3], cleanup_argv=release,
                                         cleanup_timeout_seconds=.5), step(key+"_next")])
            code, state, _ = invoke(key+"-results", definition)
            assert (root/(key+"_release.txt")).exists()
            if cleanup_code == 0:
                assert code == 0 and state["state"] == "COMPLETE" and (root/(key+"_next.txt")).exists()
                assert state["attempts"][0]["cleanup_verified"] is True
            else:
                assert code == 2 and state["state"] == "NEEDS_ATTENTION" and not (root/(key+"_next.txt")).exists()
                assert state["attempts"][0]["state"] == "HANDOFF_BLOCKED"
        checks["declared_handoff_runs_required_cleanup_and_blocks_failed_cleanup"] = True

        first = step("must_not_start")
        bad = step("missing_successor")
        bad["argv"][1] = str(root/"does-not-exist.py")
        code, result, _ = invoke("missing-results", plan("missing", [first, bad]))
        assert code == 2 and result["state"] == "INVALID_PLAN" and not (root/"must_not_start.txt").exists()
        checks["missing_successor_rejected_before_predecessor"] = True

        cleanup = step("cleanup")["argv"]
        failed = step("real_failure", code=7, cleanup_argv=cleanup, cleanup_timeout_seconds=.5)
        code, result, _ = invoke("failure-results", plan("failure", [failed, step("after_failure")]))
        assert code == 2 and result["state"] == "FINISHED_WITH_FAILURES"
        assert [row["state"] for row in result["attempts"]] == ["FAILED", "PASSED"]
        assert result["attempts"][0]["command"]["returncode"] == 7 and (root/"after_failure.txt").exists()
        checks["real_failure_cleanup_then_next_without_false_pass"] = True

        for label, cleanup_arg in (("no_cleanup", None), ("bad_cleanup", step("cleanup_fail", code=8)["argv"])):
            failed = step(label, code=6)
            if cleanup_arg:
                failed.update(cleanup_argv=cleanup_arg, cleanup_timeout_seconds=.5)
            marker = label+"_never"
            code, result, _ = invoke(label+"-results", plan(label, [failed, step(marker)]))
            assert code == 2 and result["state"] == "NEEDS_ATTENTION" and not (root/(marker+".txt")).exists()
        checks["unknown_or_failed_cleanup_blocks_resource_continuation"] = True

        now = time.time()
        code, result, _ = invoke("expired-results", plan("expired", [step("expired_never")],
                                                        times=(now-3, now-2, now-1)))
        assert code == 2 and result["state"] == "OUTSIDE_START_WINDOW" and not result["attempts"]
        checks["expired_window_starts_zero_steps"] = True

        now = time.time()
        code, result, _ = invoke("deadline-results", plan("deadline", [step("last_allowed", delay=1), step("too_late")],
                                                        times=(now-1, now+.8, now+3)))
        assert code == 2 and result["state"] == "DEADLINE_REACHED" and (root/"last_allowed.txt").exists(), result
        assert not (root/"too_late.txt").exists()
        checks["stop_new_prevents_later_step"] = True

        # Advance the observed wall clock during the RUNNING status write,
        # after the loop check but before execute/Popen; no resource child runs.
        delayed = root/"delayed_runner.py"
        delayed.write_text("import importlib.util\n"+
            "spec=importlib.util.spec_from_file_location('actual_campaign',"+repr(str(RUNNER))+")\n"+
            "module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)\n"+
            "wall=module.time.time;offset=[0.0];module.time.time=lambda:wall()+offset[0]\n"+
            "original=module.write\n"+
            "def delayed_write(path,value):\n"+
            " original(path,value)\n"+
            " if path.name=='status.json' and value.get('state')=='RUNNING':offset[0]=6\n"+
            "module.write=delayed_write\nraise SystemExit(module.main())\n", encoding="utf-8")
        now = time.time()
        definition = plan("late-spawn", [step("late_spawn_never")], times=(now-1,now+5,now+20))
        output = root/"late-spawn-results"
        code = subprocess.run([sys.executable,str(delayed),"--plan",str(definition),"--output-dir",str(output),"--run"],
                              capture_output=True, timeout=5).returncode
        state = json.loads((output/"status.json").read_text())
        assert code == 2 and state["state"] == "DEADLINE_REACHED" and not (root/"late_spawn_never.txt").exists()
        assert state["attempts"][0]["state"] == "NOT_STARTED_DEADLINE"
        assert state["attempts"][0]["command"]["pid"] is None
        assert state["pending"] == [dict(id="late_spawn_never",attempt=1)]
        checks["post_filesystem_deadline_recheck_preserves_unstarted_pending_work"] = True

        environment = root/"read_environment.py"
        environment.write_text("from pathlib import Path\nimport os,json,sys\n"+
            "Path(sys.argv[1]).write_text(json.dumps({k:os.environ.get(k) for k in " +
            "('MF_AUTHORIZED_START_AT','MF_AUTHORIZED_STOP_NEW_AT')}))\n", encoding="utf-8")
        output = root/"environment.txt"
        definition = plan("environment", [dict(id="window",argv=[sys.executable,str(environment),str(output)],timeout_seconds=3)])
        code, state, _ = invoke("environment-results", definition)
        values, source = json.loads(output.read_text()), json.loads(definition.read_text())
        assert code == 0 and values == dict(MF_AUTHORIZED_START_AT=source["start_at"],
                                           MF_AUTHORIZED_STOP_NEW_AT=source["stop_new_at"])
        checks["only_explicit_authorized_window_is_forwarded_to_children"] = True

        counter, gate_calls = root/"quality_count.txt", root/"gate_calls.txt"
        count_step, quality_gate = root/"count_step.py", root/"quality_gate.py"
        count_step.write_text("from pathlib import Path\nimport sys\n"+
            "p=Path(sys.argv[1]);n=int(p.read_text())+1 if p.exists() else 1\n"+
            "p.write_text(str(n));Path(sys.argv[2]).write_text(str(n))\n", encoding="utf-8")
        quality_gate.write_text("from pathlib import Path\nimport sys\n"+
            "p=Path(sys.argv[2]);n=int(p.read_text())+1 if p.exists() else 1;p.write_text(str(n))\n"+
            "raise SystemExit(0 if int(Path(sys.argv[1]).read_text())>=4 else 2)\n", encoding="utf-8")
        definition = plan("quality", [dict(id=key,argv=[sys.executable,str(count_step),str(counter),
                          "{attempt_dir}/work.txt"],timeout_seconds=3) for key in ("armA","armB")])
        source = json.loads(definition.read_text())
        source.update(report_gate_argv=[sys.executable,str(quality_gate),str(counter),str(gate_calls)],
                      report_gate_timeout_seconds=3)
        definition.write_text(json.dumps(source), encoding="utf-8")
        code, state, _ = invoke("quality-results", definition)
        assert code == 0 and state["state"] == "COMPLETE" and state["report_ready"] is True
        assert [(r["id"],r["attempt"]) for r in state["attempts"]] == [
            ("armA",1),("armB",1),("armA",2),("armB",2)]
        assert [r["returncode"] for r in state["report_gate_attempts"]] == [2,0]
        assert counter.read_text()=="4" and gate_calls.read_text()=="2"
        checks["commands_pass_but_report_not_ready_reruns_real_steps_then_gate_succeeds"] = True

        retry = step("retry", cleanup_argv=cleanup, cleanup_timeout_seconds=.5)
        retry["argv"] = [sys.executable, str(fixture), "fail_once", "{attempt_dir}/payload.txt", "0", "7", "{attempt}"]
        other = step("other")
        other["argv"][-1] = "{attempt}"
        code, result, _ = invoke("retry-results", plan("retry", [retry, other], retry=True))
        assert code == 2 and result["state"] == "FINISHED_WITH_FAILURES"
        assert [(r["id"],r["attempt"],r["state"]) for r in result["attempts"]] == [
            ("retry",1,"FAILED"),("other",1,"PASSED"),("retry",2,"PASSED")]
        assert result["attempts"][0]["output_dir"] != result["attempts"][2]["output_dir"]
        checks["safe_failed_step_requeues_after_other_work_with_fresh_attempt"] = True

        no_placeholder = step("unsafe_retry", cleanup_argv=cleanup)
        code, result, _ = invoke("unsafe-retry-results", plan("unsafe-retry", [no_placeholder], retry=True))
        assert code == 2 and result["state"] == "INVALID_PLAN" and not result["attempts"]
        checks["retry_without_new_attempt_name_is_rejected"] = True

        timed = step("timeout", delay=8, cleanup_argv=cleanup, cleanup_timeout_seconds=.5)
        timed["timeout_seconds"] = .1
        code, result, _ = invoke("timeout-results", plan("timeout", [timed, step("after_timeout")]))
        first = result["attempts"][0]
        assert code == 2 and first["state"] == "FAILED" and first["command"]["timed_out"]
        assert first["command"]["returncode"] is not None and first["cleanup_verified"]
        assert (root/"after_timeout.txt").exists()
        checks["timed_out_child_reaped_cleanup_then_continue"] = True
    return dict(status="PASS", evidence_class="real CPU-only subprocesses and launcher exit; no device or remote execution",
                checks=checks)


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
