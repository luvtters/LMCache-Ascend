# 客户结构裁剪：四档兼容 L2 诊断入口

复用已有 CPU tokenizer 生成器与原生 `vllm bench serve`，兼容 core `6913259` / Ascend `525a710`。自然 P/D 协议 `customer01` 四档与严格协议 `customer02_strict` 三个 128 输出档已完成，共 112 条 timed 请求、56/56 配对文本逐字相同。自然协议 4K 有两组取数路线差异；严格协议 1000 输出档为 **NOT_RUN**。两协议分开报告，精选指标、计数与退出限制见 [`../results_customer.json`](../results_customer.json)。

## 最终诊断结果

每行每后端为 **1 轮 × 8 条 timed、C1**。token/s 为实际输出/原生计时，TTFT/TPOT 为请求均值，单位 ms；失败与路线、退出检查分别保留，不能据表宣称全部验收通过。

自然 P/D（`customer01`）：

| 输入 / 前缀 / 输出 | MF token/s | Store token/s | MF TTFT / TPOT | Store TTFT / TPOT | 路线 |
|---|---:|---:|---:|---:|---|
| 4K / 50% / 128 | 19.817 | 18.881 | 399.679 / 47.709 | 544.129 / 49.090 | Store D 每条 8 L1+24 L2，MF D 全 32 L2 |
| 16K / 50% / 128 | 16.763 | 15.485 | 1471.546 / 48.533 | 1956.819 / 49.676 | 两组 P/D 全 L2 |
| 16K / 89.84% / 128 | 18.881 | 16.690 | 787.969 / 47.172 | 1419.401 / 49.206 | 两组 P/D 全 L2 |
| 16K / 89.84% / 1000 | 19.597 | 18.504 | 800.079 / 50.279 | 1446.958 / 52.647 | 两组 P/D 全 L2 |

严格 P→drain→D（`customer02_strict`）：

| 输入 / 前缀 / 输出 | MF token/s | Store token/s | MF TTFT / TPOT | Store TTFT / TPOT | 路线 |
|---|---:|---:|---:|---:|---|
| 4K / 50% / 128 | 18.579 | 17.053 | 459.122 / 50.629 | 749.706 / 53.195 | 两组 P/D 全 L2 |
| 16K / 50% / 128 | 15.899 | 14.285 | 1528.501 / 51.351 | 2494.146 / 50.909 | 两组 P/D 全 L2 |
| 16K / 89.84% / 128 | 17.332 | 15.126 | 853.002 / 51.431 | 2069.973 / 50.327 | 两组 P/D 全 L2 |

严格协议的 16K / 89.84% / 1000 为 `NOT_RUN`，不补零、不借用自然协议结果。自然协议 128 条 prefetch 只有 8 条 mixed，严格协议已记录三档的 96 条全部 0 L1；这证明已审计范围的取数来源，不证明纯拷贝带宽或独立 H2D 完成时间。

实际运行路线为 MemFabric BM SDMA / rank 1；Store 为 AscendDirectTransport / runtime TENT、LocalCommRes 1.2、BufferPool 4:8，不能按 JSON 的 `tcp` 字段解释。Store 的严格协议 P115/D128 prefetch 日志均值约 486/528 ms，自然协议约 313/346 ms；原因未定，不能归因于已证明的单一机制。

MemFabric 组内累计（含 seed 与 timed）自然协议实际 load/store 为 125325803520/31708938240 bytes，严格协议为 83802193920/24914165760 bytes；日志和最终状态相符。最终对象数自然/严格为 1680/1320，空 slots 为 368/728，pins、locked objects 和 evictions 均为 0。Store 分别查询 1680/1320 个 18 MiB 完整内存副本，均位于独立 owner；实际 Store 搬运字节仍未知。0 次淘汰只表明本轮未触发淘汰，不等于淘汰功能验通。

## 四档与资源口径

按 Qwen3-8B 实际最大上下文 40960 裁剪，下列四档的服务 `max-model-len=32768`。它们使用合成文本，不是客户 GLM 原始五档；公共前缀比例不是实测命中率。

| case | 输入 token | 公共前缀 token | 前缀比例 | 输出 token |
|---|---:|---:|---:|---:|
| `in4096_p50_out128` | 4096 | 2048 | 50% | 128 |
| `in16384_p50_out128` | 16384 | 8192 | 50% | 128 |
| `in16384_p90_out128` | 16384 | 14720 | 89.84375% | 128 |
| `in16384_p90_out1000` | 16384 | 14720 | 89.84375% | 1000 |

每档两个独立前缀组，2 条 seed（每组一条、输出 1）和 8 条 timed（C1），无额外 warmup。seed/timed 的完整输入不同，任意同组请求在公共前缀外不共享额外完整 128-token chunk。最终原始 prompt 全文重新编码并校验长度。

本轮服务使用 64 GiB pool、32 MiB slots、8 GiB CPU L1；每组后端在遍历四档前全新启动，组内保持同一实例和唯一 case ID。BM 每个 rank 保留 64 GiB，共两个 rank；Store 为 owner 64 GiB 加 client 16 MiB segment。这是实现内存保留差异，不能称为两组总保留内存完全一致，也不能据此直接推出性能差异原因。

按 prompt KV 设计，四档依次新增 192、768、360、360 个唯一 chunk，累计 1680 slots，即 52.5 GiB slot 工作集；18 MiB/chunk 对应的 payload 估计为 29.53125 GiB。这些是设计预算，实际对象数、读写字节与 evictions 须另取计数验收。若运行时还写入生成 token 的 chunk，保留这些增量，不能将实际总数强制修成预算。

## CPU 数据准备

以下路径须替换为已确认的个人工作目录、已有 CPU tokenizer 环境、兼容源码和只读模型。`CUSTOMER` 是本目录；数据输出目录必须尚不存在。可用 `taskset` 绑定当前获准使用的少量 CPU，不复制特定服务器的 CPU 编号。

```bash
export CUSTOMER="$PWD"
export TASK=/path/to/confirmed-personal-workdir
export PY=/path/to/existing/cpu-tokenizer/bin/python
export ASCEND_SOURCE=/path/to/compatible/LMCache-Ascend
export MODEL=/path/to/read-only/Qwen3-8B
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
"$PY" "$CUSTOMER/generate.py" \
  --generator-file "$ASCEND_SOURCE/examples/memfabric/basic_bench/generate_cases.py" \
  --tokenizer "$MODEL" --output "$TASK/customer_data" --requests 8
```

可用 `--case in16384_p90_out1000` 单独生成该档至另一个全新目录。生成器只验证输入数据与设计预算，不证明运行时命中、读写路线或性能。

## 对已运行服务请求

`PY` 切换为已有 vLLM client 解释器，`VLLM_BIN` 指向同一环境。端点与模型名须和当前服务一致；下列 localhost 端口只是本套入口的默认配置。

```bash
export PY=/path/to/existing/vllm/bin/python
export VLLM_BIN=/path/to/existing/vllm/bin/vllm
export AB_TOKENIZER="$MODEL"
export AB_BACKEND_METRICS_URL=http://127.0.0.1:29185/metrics
for CASE in in4096_p50_out128 in16384_p50_out128 in16384_p90_out128 in16384_p90_out1000; do
  "$PY" "$CUSTOMER/run.py" --data-root "$TASK/customer_data/$CASE" \
    --run-root "$TASK/customer_results" --arm store --tag customer01 --phase both --requests 8 --dry-run
done
```

`--dry-run` 只打印 seed/timed 两条命令，不发 HTTP 请求；执行须在当前有效设备授权下去掉该参数。相同数据在另一组全新服务上使用 `--arm memfabric`，已有结果目录会被拒绝。`--phase seed` / `--phase measure` 可拆分阶段。运行入口从本目录的上一级加载 `observe.py`。

每次前后快照都等待可观察到的 L1 对象、字节、staging、读写锁、store/prefetch in-flight 和已暴露后端 pins 归零。缺少必需的 L1/controller 字段不算零；默认 drain 上限 60 秒，可用 `CUSTOMER_DRAIN_SECONDS` 调整，未满足 gate 则失败。请求原文、原生逐请求输出、长度和 errors 留在私有运行归档，发布时只提取精选指标与差异位置。

timed 阶段 P 只复用公共前缀并计算独有尾部，再保存所有完整 prompt chunk；D 从 LMCache 取回完整对齐 prompt，但 vLLM 来源计数为 N−1 external、最后 1 token 重算。此处 external 不能区分 LMCache 临时 L1 与 L2。seed P 计算全文，seed D 同样按 N−1/1 记账。输出长度不会改变这些 prompt 来源预期。manifest 将 P 前缀命中和 D 全文命中分开，不能沿用基础 127-token 尾部的来源公式。物理读字节只能来自真实后端计数，不能从来源 token 或成功 chunks 推算。

## 逐请求路线与协议隔离

前后 L1 快照为空、vLLM external 来源符合预期，均不能证明两次快照之间没有临时 L1 读取。自然 P/D `customer01` 的原生路线审计逐条核对 128 条 prefetch：只有 4K 档 Store D 的 8 条为 mixed，每条 32 块为 **8 L1 + 24 L2**；同档 MemFabric D 为 **0 L1 + 32 L2**。其余 120 条全部为 0 L1，包括两个 16K / 128 输出档及两组 1000 输出档。4K 档必须保留两组路线差异，不能用全四档的整体文本相同掩盖。

`customer02_strict` 是另一套实验协议：两组实验 proxy 均在 P event 后等待 service 的 L1/store/prefetch drain，再转发 D；其中检查 L1 读写锁，不专门轮询 L2 pins。该条件改变 P/D 时序，不能与自然 P/D 首轮按重复轮次混合。每个 tag 单独输出吞吐、延迟和 p95；是否全 L2 仍须看该 tag 的原生逐请求 retrieval 日志。

`proxy_pool_barrier.py` 是本轮 C1 实验 shim，包装兼容 core 原 proxy 的 decoder send/stream；通过环境变量 `LMCACHE_SOURCE` 定位 `examples/disagg_prefill_mp/disagg_proxy_server.py`，固定轮询 `http://127.0.0.1:28185/status` 并记录 `L2_POOL_BARRIER`。两组使用相同 shim、64 GiB pool、8 GiB L1、max-model-len 32768。**HTTP 轮询及 drain 等待都包含在端到端 TTFT 中**；不能直接用严格减自然的 TTFT 解释 L2 搬运耗时或收益。该入口启动实验 proxy，不能当 CPU 数据生成步骤运行。

严格文本配对顺序已核对实际运行的 vLLM `benchmarks/serve.py:882` 的 `outputs = await asyncio.gather(*tasks)` 与 `:1017` 的 `generated_texts` 顺序构造，输出数组保留输入 task 顺序。比较还需同 case、输入全文、参数和 request index；本轮字面文本相同不等于 token ID 或 KV 数值相同。

## 离线归档与边界

```bash
python "$CUSTOMER/summarize.py" --root "$TASK/archived_customer_results" \
  --output-prefix "$TASK/customer_summary"
python "$CUSTOMER/audit_prefetch.py" --archive-root "$TASK/archived_run" \
  --tag customer02_strict --store-run customer_store02 --memfabric-run customer_mf02 \
  --output "$TASK/strict_prefetch_audit.json"
```

**最终必须运行逐请求路线审计**。`audit_prefetch.py` 是纯标准库 CLI，按 timed 的 before/after Unix 窗口检查 P-prefix 与 D-full-prompt 日志：混合 L1/L2、少记录、缺证据或未知格式会返回 1。首轮自然 4K 返回 1 是预期的路线审计失败，不是服务请求失败；其余自然三档通过。严格协议 PASS 只指已记录的三个 128 输出档。CLI 枚举两组现存 case 目录并集，因此还须按上面的覆盖表核对两组同时缺失的 case，不能把现存档位 PASS 写成四档完整。

汇总保留不完整轮次、重复项、缺文件、实际对象/读写/淘汰计数、L1/protection 状态和严格文本差异；缺失 Store bytes 保持未知。它只在同一 tag 内聚合通过请求/来源/drain 检查的轮次，拒绝跨 tag 汇总，不虚构尚未运行的轮次；路线验收独立于这些计数检查。吞吐使用同 tag 总实际输出/总原生计时，均值按请求数加权，p95 仅在同 tag 完整可靠明细存在时池化，否则保留各轮范围而不平均。

**两组退出 gate 均未通过**：自然和严格协议的两组 P/D 均为 `SIGABRT`；MemFabric service 分别为退出码 130 和 `SIGTERM`，members 均为 0；Store service 两次为 `SIGSEGV`，owners 两次为 `SIGABRT`。客户端计数成功不能覆盖这些退出失败；八请求轮次只作诊断，不宣称稳定性、统计显著性、token ID/KV 数值一致性或整体收益。全轮含基础共 688 条 timed、0 请求失败，但基础仍有 2 条严格文本差异，自然 4K 路线不一致，退出问题也未解决，不能写成整体 all passed。北京时间 20:05:28.959 已停止个人容器、PID 为 0，最终设备盘点无 NPU 运行进程，守卫与监控均结束。

本次归档本地检查：三个 Python 文件 AST 通过；修正后的相对观察器路径可加载，最小临时 manifest 的 seed/timed 两条 dry-run 通过且未创建结果目录；UTF-8 与暂存空白检查通过。此检查没有发请求、加载模型或初始化设备。
