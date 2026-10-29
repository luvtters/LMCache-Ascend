#!/usr/bin/env bash
# Against already-running services. This script never starts/stops a service.
set -euo pipefail
dry_run=0
if [[ "${1:-}" == --dry-run ]]; then dry_run=1; shift; fi
if [[ $# != 3 ]]; then
    echo 'Usage: bash run.sh [--dry-run] store|memfabric seed|quick|matrix UNIQUE_TAG' >&2
    exit 2
fi
arm="$1"; mode="$2"; tag="$3"
[[ "$arm" == store || "$arm" == memfabric ]] || exit 2
[[ "$mode" == seed || "$mode" == quick || "$mode" == matrix ]] || exit 2
[[ "$tag" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python}"
vllm_bin="${VLLM_BIN:-vllm}"
data_root="${AB_DATA_ROOT:-/workspace/l2_ab_20261029/data}"
run_root="${AB_RUN_ROOT:-/workspace/l2_ab_20261029/runs}"
tokenizer="${AB_TOKENIZER:-/workspace/models/Qwen3-8B}"
base_url="${AB_BASE_URL:-http://127.0.0.1:28300}"
service_url="${AB_SERVICE_URL:-http://127.0.0.1:28185}"
p_url="${AB_P_URL:-http://127.0.0.1:28301}"
d_url="${AB_D_URL:-http://127.0.0.1:28302}"
backend_metrics_url="${AB_BACKEND_METRICS_URL:-}"
model="${AB_MODEL:-mf-l2-ab}"
parent="${run_root}/${tag}_${arm}"
output_tokens=1
if [[ "$mode" != seed ]]; then
    output_tokens="${AB_OUTPUT_TOKENS:-}"
    if [[ -z "$output_tokens" ]]; then
        output_tokens="$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["output_tokens"])' "${data_root}/manifest.json")"
    fi
    [[ "$output_tokens" =~ ^[1-9][0-9]*$ ]] || {
        echo 'output_tokens in manifest.json or AB_OUTPUT_TOKENS must be a positive integer' >&2
        exit 2
    }
fi

snapshot() {
    local folder="$1" when="$2" phase="$3" count="$4"
    local wait_args=()
    if [[ "$when" == after ]]; then
        wait_args=(--wait-seconds 15 --expected-requests "$count" --data-root "$data_root")
        if [[ "$phase" == timed ]]; then wait_args+=(--require-source-alignment); fi
        if [[ "$phase" == warmup ]]; then wait_args+=(--wait-source-alignment); fi
    fi
    "$python_bin" "${here}/observe.py" snapshot --folder "$folder" --phase "$when" \
        --service-url "$service_url" --p-url "$p_url" --d-url "$d_url" \
        --backend-metrics-url "$backend_metrics_url" "${wait_args[@]}"
}

one() {
    local phase="$1" c="$2" round="$3" data="$4" count="$5" output="$6"
    local folder="${parent}/c${c}_r${round}_${phase}"
    local cmd=("$vllm_bin" bench serve --backend openai --base-url "$base_url" --endpoint /v1/completions
        --model "$model" --served-model-name "$model" --tokenizer "$tokenizer"
        --dataset-name custom --dataset-path "$data" --custom-output-len "$output"
        --num-prompts "$count" --max-concurrency "$c" --request-rate inf
        --temperature 0 --num-warmups 0 --ready-check-timeout-sec 0
        --ignore-eos --disable-shuffle --no-oversample --seed 0 --skip-chat-template
        --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,95
        --save-result --save-detailed --result-dir "$folder" --result-filename benchmark.json)
    printf '%q ' "${cmd[@]}"; printf '\n'
    [[ "$dry_run" == 0 ]] || return 0
    [[ ! -e "$folder" ]] || { echo "Refusing existing result directory: $folder" >&2; exit 2; }
    mkdir -p "$folder"
    printf '%q ' "${cmd[@]}" > "${folder}/command.sh"; printf '\n' >> "${folder}/command.sh"
    if [[ -n "${AB_LAUNCH_CONFIG:-}" ]]; then cp -- "$AB_LAUNCH_CONFIG" "${folder}/launch_config.json"; fi
    snapshot "$folder" before "$phase" "$count"
    set +e
    "${cmd[@]}" 2>&1 | tee "${folder}/benchmark.log"
    local result=${PIPESTATUS[0]}
    snapshot "$folder" after "$phase" "$count"
    local snapshot_result=$?
    set -e
    "$python_bin" "${here}/observe.py" finish --folder "$folder" --data-root "$data_root" \
        --arm "$arm" --phase "$phase" --concurrency "$c" --round "$round" \
        --requests "$count" --output-tokens "$output" --exit-code "$result" --snapshot-exit "$snapshot_result" \
        > "${folder}/validation.log"
}

if [[ "$mode" == seed ]]; then
    one seed 1 0 "${data_root}/seed.jsonl" 4 1
elif [[ "$mode" == quick ]]; then
    # Same dataset as the first formal condition; use a fresh tag and keep it separate.
    one warmup 1 1 "${data_root}/c1_r1/warmup.jsonl" 4 "$output_tokens"
    one timed 1 1 "${data_root}/c1_r1/custom.jsonl" "${AB_QUICK_REQUESTS:-4}" "$output_tokens"
else
    for c in ${AB_CONCURRENCIES:-1 4}; do
        for round in $(seq 1 "${AB_ROUNDS:-3}"); do
            one warmup "$c" "$round" "${data_root}/c${c}_r${round}/warmup.jsonl" 4 "$output_tokens"
            one timed "$c" "$round" "${data_root}/c${c}_r${round}/custom.jsonl" "${AB_REQUESTS:-32}" "$output_tokens"
        done
    done
fi
if [[ "$dry_run" == 0 ]]; then
    "$python_bin" "${here}/observe.py" summary --root "$parent" --output "${parent}/summary.json" > "${parent}/summary.csv"
fi
