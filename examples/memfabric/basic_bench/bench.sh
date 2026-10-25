#!/usr/bin/env bash
# One official benchmark invocation against an already running A or B service.
set -euo pipefail

dry_run=0
if [[ "${1:-}" == --dry-run ]]; then dry_run=1; shift; fi
if [[ $# != 3 ]]; then
    echo "Usage: bash bench.sh [--dry-run] A|B qwen_2560_unique|qwen_2560_prefix50|qwen_2560_prefix90 UNIQUE_TAG" >&2
    exit 2
fi
arm="$1"; case_name="$2"; tag="$3"
[[ "$arm" == A || "$arm" == B ]] || { echo 'Arm must be A or B' >&2; exit 2; }
case "$case_name" in
    qwen_2560_unique|qwen_2560_prefix50|qwen_2560_prefix90) ;;
    *) echo 'Unknown case' >&2; exit 2 ;;
esac
[[ "$tag" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Tag must contain letters, digits, _ or -' >&2; exit 2; }

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_root="${MF_DATA_ROOT:-${script_dir}/generated/basic/dataset}"
result_root="${MF_RESULT_ROOT:-${script_dir}/results}"
result_dir="${result_root}/${tag}_${arm}_${case_name}"
tokenizer="${MF_TOKENIZER:-/workspace/d00896497/models/Qwen3-8B}"
base_url="${MF_BASE_URL:-http://127.0.0.1:28000}"
model="${MF_MODEL:-mf-test}"
service_url="${MF_SERVICE_URL:-http://127.0.0.1:28080}"
p_url="${MF_P_URL:-http://127.0.0.1:28001}"
d_url="${MF_D_URL:-http://127.0.0.1:28002}"
data="${data_root}/${case_name}/custom.jsonl"

cmd=(vllm bench serve
    --backend openai --base-url "$base_url" --endpoint /v1/completions
    --model "$model" --served-model-name "$model" --tokenizer "$tokenizer"
    --dataset-name custom --dataset-path "$data" --custom-output-len 128
    --num-prompts 8 --max-concurrency 1 --request-rate inf
    --temperature 0
    --num-warmups 0 --ready-check-timeout-sec 0
    --ignore-eos --disable-shuffle --no-oversample --seed 0
    --skip-chat-template --percentile-metrics ttft,tpot,e2el --metric-percentiles 50
    --save-result --save-detailed --result-dir "$result_dir" --result-filename benchmark.json)
printf '%q ' "${cmd[@]}"; printf '\n'
if [[ "$dry_run" == 1 ]]; then
    echo 'DRY_RUN: no file writes, status requests, or inference requests.'
    exit 0
fi

# Validate the small dataset and reserve a new result directory before traffic.
python - "$data" "$result_dir" "$arm" "$case_name" "${MF_CACHE_STATE:-unknown}" "${cmd[@]}" <<'PY'
import json, os, pathlib, sys
data, result, arm, case, cache_state, *command = sys.argv[1:]
rows = [json.loads(line) for line in pathlib.Path(data).read_text(encoding="utf-8").splitlines() if line.strip()]
if len(rows) != 8 or len({r["prompt"] for r in rows}) != 8:
    raise SystemExit("Expected exactly 8 different input prompts")
if any(r.get("output_tokens") != 128 or r.get("metadata", {}).get("input_tokens") != 2560
       or r.get("metadata", {}).get("case") != case for r in rows):
    raise SystemExit("Dataset does not match this basic case")
service_config = None
launch_path = os.environ.get("MF_LAUNCH_CONFIG")
if launch_path:
    try:
        service_config = json.loads(pathlib.Path(launch_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Cannot read MF_LAUNCH_CONFIG JSON: {exc}")
    if not isinstance(service_config, dict):
        raise SystemExit("MF_LAUNCH_CONFIG must contain a JSON object")
path = pathlib.Path(result)
try:
    path.mkdir(parents=True, exist_ok=False)
except FileExistsError:
    raise SystemExit("Result directory already exists; choose a new unique tag")
manifest = {"arm": arm, "case": case, "dataset_id": rows[0]["metadata"]["dataset_id"],
            "data_path": str(pathlib.Path(data).resolve()), "command": command,
            "service_config": service_config,
            "cache_state_before_case": cache_state, "status": "PREPARED",
            "runtime_cache_hit_fraction": None}
(path / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY

# Read-only snapshots; these do not clear caches or wait between individual requests.
snapshot() {
    python - "$service_url" "$p_url" "$d_url" "$result_dir" "$1" "$arm" <<'PY'
import json, pathlib, sys, time, urllib.request
service, p_url, d_url, result, phase, arm = sys.argv[1:]
path = pathlib.Path(result)
def fetch(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.read().decode("utf-8")
status = json.loads(fetch(service.rstrip("/") + "/status"))
(path / (phase + "_status.json")).write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
if phase == "before":
    adapters = status["storage_manager"]["l2_adapters"]
    if arm == "A" and adapters:
        raise SystemExit("A must have no L2 adapters; benchmark not sent")
    if arm == "B" and (len(adapters) != 1 or adapters[0].get("protocol") != "SDMA"
                       or not adapters[0].get("is_healthy")):
        raise SystemExit("B needs exactly one healthy MemFabric SDMA adapter; benchmark not sent")
for role, url in (("p", p_url), ("d", d_url)):
    (path / (phase + "_" + role + "_metrics.txt")).write_text(fetch(url.rstrip("/") + "/metrics"), encoding="utf-8")
(path / (phase + "_time.txt")).write_text(str(time.time()) + "\n", encoding="utf-8")
PY
}

snapshot before
set +e
"${cmd[@]}" 2>&1 | tee "${result_dir}/benchmark.log"
bench_exit=${PIPESTATUS[0]}
snapshot after
snapshot_exit=$?
set -e
python - "$result_dir" "$bench_exit" "$snapshot_exit" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1]) / "run.json"
data = json.loads(path.read_text(encoding="utf-8"))
result_file = path.parent / "benchmark.json"
metrics = json.loads(result_file.read_text(encoding="utf-8")) if result_file.exists() else {}
counts_ok = (metrics.get("completed") == 8 and metrics.get("failed", 0) == 0
             and metrics.get("total_input_tokens") == 8 * 2560
             and metrics.get("total_output_tokens") == 8 * 128
             and metrics.get("output_lens") == [128] * 8)
complete = sys.argv[2:] == ["0", "0"] and counts_ok
data.update(benchmark_exit_code=int(sys.argv[2]), after_snapshot_exit_code=int(sys.argv[3]),
            request_and_token_counts_match=counts_ok,
            status="FINISHED" if complete else "INCOMPLETE")
path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
if not complete:
    raise SystemExit("Incomplete run: inspect benchmark.log, benchmark.json and run.json")
PY
[[ "$bench_exit" == 0 ]] || exit "$bench_exit"
exit "$snapshot_exit"
