# MemFabric worker 直传验证入口

日期：2026-10-28。本目录提供两层独立验收：先用合成分页 KV 验证真实 NPU context，再用 Qwen3-8B 的 P/D 请求验证完整 worker 路径。它们均不是性能测试。

## 已测到的范围

| 检查 | 已有结果 | 证据边界 |
| --- | --- | --- |
| 合成分页 KV context | split-NHD / format 16，真实三成员 BM、RequestClient 网络 RPC、18 MiB NPU gather→池→scatter 通过；非连续逆序 block 表也通过 | 未加载模型，不替代 vLLM Connector / WorkerAdapter 验收；harness 未运行追踪 native 内部 staging |
| 最终 Qwen3-8B / TP1 单请求门控 | `DIRECT_POOL_PATH_PASS`，gate 实际返回 **0**：1373 prompt tokens，D external 1280、local compute 93、local prefix 0，答案 ORCHID；P/D 各完成 10 次、180 MiB 搬运，最终 pin 为 0 | 证明本次单请求的数据路径与内容，不能外推吞吐、稳定时延或 HCCS 收益 |
| 已有池上的四次 D 追加请求 | 两次顺序、两条客户端并发请求均回答 ORCHID，脚本实际返回码 **0**；新增 40 次 G2L，共 **720 MiB**，无新增 P 搬运，普通 L1 为空、最终 pin/pending write 为 0 | 两次顺序每次 external/prompt 为 1280/1373；并发两条合计为 2560/2746，属于聚合指标，不是逐请求命中日志或性能采样 |
| 最终进程退出与释放 | P、D、proxy、gate 均返回 0。service 输出 BM uninit、`MPCacheServer closed` 和 HTTP 关闭完成后，最终由 SIGKILL 退出，来源未确定。验证环境已于北京时间 02:32:33 释放，无 NPU 进程 | **service 正常进程退出仍未验证**，不能把关闭完成日志等同退出码 0；SIGKILL 后没有新增 NPU 实验 |

最终该组 P 写入共 **180 MiB**，D 首次与追加读取合计 **900 MiB**。通用结果快照见 [validation_results_20261028.json](validation_results_20261028.json)。真实数据路径通过、各进程退出状态和资源释放分别记录；仅看到打印的 PASS 不够。

## 文件与版本

- [validate_direct_kv.py](validate_direct_kv.py)：合成 BF16 分页 KV，提供 `primitive` 与 `context` 两种模式；真实设备执行必须显式 `--allow-npu`。
- [traced_direct_connector.py](traced_direct_connector.py)：实验专用 Connector 包装，只记录 metadata、NPU 指针、字节数和调用完成顺序；CPU BM copy 路径被设为报错。
- [direct_pd_gate.py](direct_pd_gate.py)：复用普通 P/D proxy，P store 完成且对象提交后才放行 D，核对来源指标、答案、池保护和追踪。
- [followup_direct_requests.py](followup_direct_requests.py)：严格 gate 后只向 D 发送两次顺序与两条客户端并发请求，核对既有池复用、逐进程 G2L/scatter 和保护释放。
- [适配说明](README_direct_zh.md)：生产代码职责、支持范围与已有配置模板。

配套 core 扩展点基线为 `04e16ff`；最终实跑为该基线加两文件工作区修订，等价修改现归档为 core `8bc9edc`。Ascend runtime 分支的直传父版本为 `be90c96`。`validate_direct_kv.py` 的 `direct_parent_revision` 仅记录这个历史父版本，**不是实际运行 checkout 的核验结果**；`source_revision_verified_by_script` 保持 `false`。执行者应另外记录两个仓库的实际提交、未提交改动和加载模块路径。

## 本轮修订与保留的失败记录

Ascend 增加可选 BM `init_timeout`，service 与 P/D 均可显式设为 300 秒，为模型冷加载留出等待；不填仍为 90 秒，数据搬运的 `operation_timeout=90` 保持不变。本轮完成 29 项 Ascend CPU 检查和 1 项 core 定向检查，CPU 证据与真实模型结果分开。

core 将 `lmcache/v1/distributed/transfer_channel/impl` 从包 `__init__.py` 的立即导入移到 `factory.py` 实际创建 context 时导入。纯 CPU factory 导入探针从退出 -6 变为 0；真实 status RPC 后关闭客户端也返回 0，未加载 Mooncake native engine，最终 service/P/D 的映射记录同样未见该引擎。这项修订避免未选择的可选后端被过早加载，**不代表显式选择 Mooncake 后其 native 退出问题已修复**。

诊断 gate 另外改为只读 HTTP `/status` 观察池状态，不再导入 LMCache/RequestClient；worker 与 service 仍使用原真实控制 RPC。

早期 model03 的数据路径曾通过，但旧 gate、P/D API 随后出现 SIGABRT、service 出现 SIGSEGV；停止记录中的 `clean_stop` 仅表示进程已停止。model04 则在加载模型前采样到空闲 17.93 / 总量 61.27 GiB，低于请求的 42.89 GiB；稍后测得约 60.87 GiB 空闲，原因未确定，不能归因为设备映射错误或延迟导入修订。守卫曾拒绝一次重试；随后只重启本项目容器、相同参数的最终重跑通过。

最终 service 的 SIGKILL 来源仍未知；停止脚本只发送 SIGINT。容器 cgroup OOM kill 与 failcnt 均为 0，内存峰值约 9.5 GiB，现有证据不能认定为容器 OOM。关闭日志已完成，但 service 正常进程退出、长期稳定性、多 TP、吞吐/稳定时延与跨服务器 HCCS 收益仍未验证。

## 准备独立运行目录

以下均为已有容器或已准备 Ascend 环境内的命令，不包含宿主机管理、SSH 或模型下载。先激活匹配的项目 venv，确认 `python` 来自该环境，保留 CANN/SDK 已设置的 Python 路径。下文使用本轮已验证的 `python -m` 入口；它与安装在同一环境中的 `lmcache` / `vllm` CLI 等价，避免全局 console script 的 shebang 绕过项目环境。目录和模型路径由执行者替换，使用新建的个人运行目录。

```bash
export ASCEND=/path/to/LMCache-Ascend
export CORE=/path/to/LMCache
export RUN=/path/to/private/new-direct-run
export MODEL=/path/to/read-only/Qwen3-8B
mkdir -p "$RUN"
export PYTHONPATH="$ASCEND/examples/memfabric:$ASCEND:$CORE${PYTHONPATH:+:$PYTHONPATH}"
```

模型保持只读。不要用源码路径覆盖原 `PYTHONPATH`，否则可能丢失 `acl` 等 SDK Python 模块。

下面的设备分工示例为 P=8、D=10、owner=14；执行时按当次获准设备替换。每个进程只见一张卡，配置中的 `device_id` 为进程内逻辑号 0。BM rank 为 owner/P/D = 0/1/2，与 TP rank 无关。

先从现有模板生成本轮私有配置。三端统一 `pool_gb=1`、BM world size 3；service 使用 32 MiB slot。`init_timeout=300` 留出模型冷加载等待，搬运 `operation_timeout` 仍是 90 秒，控制 RPC 超时另行设置。

```bash
python - <<'PY'
import json, os
from pathlib import Path
source = Path(os.environ["ASCEND"]) / "examples/memfabric"
output = Path(os.environ["RUN"])
server = json.loads((source / "direct_server.json").read_text())
server["config"].update(pool_gb=1, init_timeout=300)
(output / "direct_server.json").write_text(json.dumps(server, indent=2))
for role in ("p", "d"):
    config = json.loads((source / f"direct_{role}.json").read_text())
    config["kv_connector_module_path"] = "traced_direct_connector"
    extra = config["kv_connector_extra_config"]
    extra["lmcache.mp.mq_timeout"] = 180
    extra["lmcache.mp.direct"].update(pool_gb=1, init_timeout=300)
    (output / f"direct_{role}.json").write_text(json.dumps(config, indent=2))
PY
```

## 第一层：真实 NPU 合成分页 KV

在 service 终端启动控制 API 与 BM rank 0：

```bash
ASCEND_RT_VISIBLE_DEVICES=14 python -m lmcache.cli.main server \
  --host 127.0.0.1 --port 25555 \
  --http-host 127.0.0.1 --http-port 28080 \
  --chunk-size 128 --l1-size-gb 1 --max-gpu-workers 2 \
  --supported-transfer-mode lmcache_driven \
  --server-module "$(cat "$RUN/direct_server.json")"
```

P、D 在两个已准备相同环境变量的终端**并行启动**，不能先等 P 完成再启动 D。P 终端：

```bash
ASCEND_RT_VISIBLE_DEVICES=8 python "$ASCEND/examples/memfabric/validate_direct_kv.py" \
  --mode context --role p --device 0 --layout split-nhd \
  --ascend-source "$ASCEND" --core-source "$CORE" \
  --coord-dir "$RUN/synthetic" --run-id context01 \
  --server-url tcp://127.0.0.1:25555 --store-url tcp://127.0.0.1:28680 \
  --pool-gb 1 --timeout 180 --allow-npu
```

D 终端运行同一命令，仅将设备改为 `ASCEND_RT_VISIBLE_DEVICES=10`、角色改为 `--role d`。两端必须使用相同 geometry、coord-dir 与 run-id；重跑使用新的 run-id。

非连续逆序 block 验证在**两端同时**追加以下参数，并换新 run-id：

```text
--block-size 64 --num-blocks 6 --source-blocks 3,1 --target-blocks 4,2
```

`--skip-blocks 1` 可额外检查前缀保护，但不属于上表已通过的两次 context 结果。`stacked-nhd` 会发现 format 1，当前 native 不支持；保留该选项只用于诊断已知失败布局。

通过应同时满足两端返回码 0、result JSON 为 PASS、D 恢复内容相等、未写位置不变、保护归零和正常关闭。合成 KV 没有模型权重；请求字节数与含初始化/同步的总耗时不能换算成传输带宽。

## 第二层：Qwen3-8B 真实 worker 门控

合成 context 成员退出后，固定 BM 组不能原地重建。退出旧 service，使用新运行目录和新 service/P/D 固定组；不要把已发生 peer leave 的旧组用于模型验收。service 命令与前节相同。

P 终端：

```bash
export MF_DIRECT_TRACE_DIR="$RUN/worker-traces"
export MF_DIRECT_TRACE_ROLE=P
export LMCACHE_MP_TRANSFER_MODE=auto
export LMCACHE_REQUEST_TELEMETRY_TYPE=fastapi
export LMCACHE_REQUEST_TELEMETRY_ENDPOINT=http://127.0.0.1:25769/api/v1/telemetry
export VLLM_WORKER_MULTIPROC_METHOD=spawn
ASCEND_RT_VISIBLE_DEVICES=8 python -m vllm.entrypoints.cli.main serve "$MODEL" \
  --served-model-name mf-direct --tensor-parallel-size 1 --dtype bfloat16 \
  --max-model-len 4096 --max-num-batched-tokens 1024 --max-num-seqs 4 \
  --block-size 128 --gpu-memory-utilization 0.7 --enforce-eager \
  --no-enable-prefix-caching --host 127.0.0.1 --port 28001 \
  --kv-transfer-config "$(cat "$RUN/direct_p.json")"
```

D 终端**并行**运行同一模型命令，设备改为 10、端口改为 28002、配置改为 `direct_d.json`；设置 `MF_DIRECT_TRACE_ROLE=D`，并取消 P 专用通知变量：

```bash
unset LMCACHE_REQUEST_TELEMETRY_TYPE LMCACHE_REQUEST_TELEMETRY_ENDPOINT
```

两端共用 trace-dir，但使用各自 role/PID 文件。保持 `LMCACHE_MP_TRANSFER_MODE=auto`，不能沿用兼容版的显式 `lmcache_driven` worker 配置。service 的 `--supported-transfer-mode lmcache_driven` 不改变 Direct Connector 自选 context 的行为。

代理终端：

```bash
python "$CORE/examples/disagg_prefill_mp/disagg_proxy_server.py" \
  --host 127.0.0.1 --port 28000 \
  --prefiller-host 127.0.0.1 --prefiller-port 28001 \
  --decoder-host 127.0.0.1 --decoder-port 28002 --telemetry-port 25768
```

待 P/D 健康后，gate 终端发送一次真实请求：

```bash
python "$ASCEND/examples/memfabric/direct_pd_gate.py" \
  --service-http http://127.0.0.1:28080 \
  --prefiller http://127.0.0.1:28001 --decoder http://127.0.0.1:28002 \
  --proxy http://127.0.0.1:28000 --model mf-direct \
  --telemetry-port 25769 \
  --proxy-telemetry http://127.0.0.1:25768/api/v1/telemetry \
  --trace-dir "$RUN/worker-traces" --require-direct-trace \
  --output "$RUN/gate_run01"
```

必须使用空池、新 P/D 进程，不发送其他请求；脚本不清理 L1。普通 L1 的对象数和已用字节为零、L2 adapter 列表为空；P 完成 commit 后才放行 D。指标与 trace 应覆盖同一次请求：P 首次本地计算，D 外部 KV 命中且本地前缀命中为零，答案正确；每个 PID/owner 线程的 gather/L2G、G2L/scatter 需按 scratch 指针与字节数配对完成。

trace 是 Python/native 调用边界证据，不是硬件 DMA profiler。未加 `--require-direct-trace` 的成功只记为功能复用，不作为完整直传路径证明。

## 追加顺序与客户端并发请求

严格 gate 已写出 `DIRECT_POOL_PATH_PASS`、池和 D 仍健康时，可在同一固定组上重放 gate 的原始请求。此步骤只向 D 发四个请求，不启动 P 搬运、不清 L1，也不修改池控制状态：

```bash
python "$ASCEND/examples/memfabric/followup_direct_requests.py" \
  --gate-dir "$RUN/gate_run01" \
  --decoder http://127.0.0.1:28002 --service-http http://127.0.0.1:28080 \
  --trace-dir "$RUN/worker-traces" --model mf-direct \
  --output "$RUN/followup01"
```

脚本分别保存两次顺序请求和并发两条请求的答案、阶段指标与 trace。并发阶段的来源 token 是整段聚合值；客户端并发不自动等于引擎同时执行两个请求。已有实跑四次新增读取共 720 MiB，结果为功能验证，不用于估算吞吐或带宽。

## 收尾与结果解释

先停止新增请求，等 gate 完成并核对返回码，再排空、退出 P/D 及 scheduler，最后退出 service。RPC 超时或取消不等于 DMA 结束；关闭失败时保留未知操作的保护，不根据超时复用地址。

分别归档结果 JSON、gate 返回码、P/D 指标、逐进程 trace、service 状态和关闭回执。真实数据路径 PASS、进程正常退出、性能收益是三项独立结论。本次尚无吞吐或 HCCS 收益结论。
