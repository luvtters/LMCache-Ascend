# 兼容 L2 对照入口（2026-10-29）

本目录归档兼容 LMCache core `6913259` / LMCache-Ascend `525a710` 的本轮通用脚本。比较现有 Mooncake Store L2 与 `MemFabricL2Adapter`，两组均走 **NPU KV → CPU L1 staging → L2** 及其逆向流程；`skip_l1` 排除持久 L1 命中，不能解释为 worker 直传。本目录不包含部署、SSH、模型、原始内部日志或未部署的 native 修补候选。

## 验收边界

本轮 Store 使用已有 SDK 的 relay 路径：`ASCEND_LOCAL_COMM_RES=1.2`、`ASCEND_BUFFER_POOL=4:8`，不强制 `HCCL_INTRA_ROCE_ENABLE=1`，无需修改生产 connector。实际日志确认 `AscendDirectTransport` 和 runtime TENT；环境中的 `MC_USE_TENT=0` 字符串不能覆盖实际运行日志。JSON 的 `protocol=tcp` 是固定版本配置字段，不能据此称为 TCP 搬运基线。

18/36 MiB 完整内容和独立 owner 副本位置检查，与请求、来源计数、L1 状态、真实读池字节分别核验。成功 chunks 或请求完成数不能替代真实读池证据；缺失的字节计数保留为未知。

前后 L1 为空及 vLLM external 来源计数符合预期，仍不能证明请求全过程均从 L2 取回：两次快照之间可能经临时 L1 提供数据。全量读池须逐请求核对原生 retrieval 路线日志，不能只凭 first hot read、计数或边界快照定性；基础与追加结果均适用此限制。

**本轮生命周期退出 gate 未通过**：两组正式停服时的 P/D 均出现 `SIGABRT`，不能将该异常专归 Store。MemFabric 长档 service 按 `SIGINT` 停止，退出码 130，短档 service 收到 `SIGTERM`，两次 pool member 均退出码 0；不能将 service 记为退出 0。Store 长档 service 为 `SIGSEGV`，短档 service 最终退出状态未记录；其他独立探针还分别观察到 native probe `SIGABRT`、owner `SIGABRT`。内容、位置或请求检查通过，不代表这些退出问题已解决。性能材料只作诊断，不宣称稳定性或 MemFabric 收益；`summarize_l2_results.py` 固定保留本轮 Store 的 `PERFORMANCE_DIAGNOSTIC_EXIT_GATE_FAILED` 标记，其 MemFabric 退出状态仍须结合本段记录审阅，不能直接套用于未来已修复实验。

## 结果范围

基础 warm-read 固定三档：1024 前缀 + 127 尾部 / C1，以及 3072 前缀 + 127 尾部 / C1、C4；均为 32 输出 token。两组各档三轮、每轮 32 条 timed，共 18 轮 / 576 条 timed 请求已执行完成并完成离线复核。精选指标、计数与退出边界见 [`results_basic.json`](results_basic.json)；执行完成与全部验收通过是不同状态。

追加客户结构裁剪 case 使用 Qwen3-8B 的 40960 最大上下文，覆盖 4K/16K 输入、128/1000 输出、50%/约90%公共前缀；这不是客户 GLM 原始五档，构造前缀比例也不是实测命中率。自然协议四档和严格协议三个 128 输出档分别归档于 [`customer/README.md`](customer/README.md) 与 [`results_customer.json`](results_customer.json)，严格 1000 输出为 `NOT_RUN`，两协议不混合聚合。112 条 timed 请求全部完成、56/56 配对文本相同；自然 4K 的 Store D 仍有混合 L1/L2 路线，严格三档的 96 条 prefetch 均为 0 L1。

含基础本轮共 **688 条 timed、0 请求失败**；基础 2 处文本差异、自然 4K 路线差异和两组退出 gate 失败均保留，不能宣称整体通过。北京时间 **20:05:28.959** 已确认个人容器停止、PID 0、无 NPU 运行进程，守卫和监控结束。

长尾增加完整 cache chunk 时，可能同时产生读与写，不能沿用基础 127-token 尾部的 external/local-compute 预期，也不能把未知的读写量套成基础 27 GiB 纯读。追加 case 以实际计数区分 warm-read、写入和淘汰；严格输出一致性继续逐条配对，差异保留而不预判原因。现有 `generate_l2_cases.py` 和固定 127-token 预期的离线汇总面向基础数据；追加入口及其判定规则须随对应结果单独归档。

## 基础 warm-read：三档精选结果

以下为完整三档，均为 32 输出 token。每组/条件为 3 轮 × 32 条，全部完成、无请求失败；吞吐取总实际输出 token / 总原生计时，TTFT/TPOT 为按请求数加权的均值。退出与整体严格输出文本检查未通过，因此表中数值只用于本轮诊断。

| 输入 token | L2 后端 | 并发 | 输出 token/s | 平均 TTFT ms | 平均 TPOT ms |
|---:|---|---:|---:|---:|---:|
| 1151 | MemFabric | 1 | 19.9243 | 212.396 | 44.9433 |
| 1151 | Mooncake Store | 1 | 18.0946 | 232.182 | 49.5441 |
| 3199 | MemFabric | 1 | 18.8376 | 268.505 | 46.1221 |
| 3199 | Mooncake Store | 1 | 17.4442 | 304.787 | 49.3290 |
| 3199 | MemFabric | 4 | 58.9954 | 494.956 | 53.1640 |
| 3199 | Mooncake Store | 4 | 53.7678 | 544.337 | 58.2622 |

两组短/长档每轮的 P、D 各记录 external 32768/98304、local compute 4064、local hit 0，L1 前后快照为空；短/长档每轮记录 512/1536 个 L2 completed chunks。MemFabric 短/长档每轮实际 load bytes 为 9/27 GiB，store bytes 增量与 evictions 均为 0。Store 实际读字节未知，不能从 chunks 或逻辑 KV 大小推算。

基础三档另已按每个 timed Unix 时间窗口逐条审计 **1152 条 prefetch 日志**（576 请求 × P/D）：全部为 0 L1，mixed 为 0；短/长档各轮分别核对到 512/1536 个 L2 chunks。因此本组的全 L2 路线有逐条原生日志证据，不是从前后 L1 为空推定；Store 实际搬运字节仍保持未知。

短档严格输出文本相同为 **96/96**，长档为 **190/192**，合计 **286/288**。两处差异位于长档 C4 的第 1、3 轮，均为零起点 index 6：首行结尾差一个句点，其余文本逐字相同。配对轮次的输入/输出长度和 errors 数组一致，但整体严格文本一致性未通过，原因尚未判定；本次比较不能证明差异由 batching 引起，也不构成 token ID 或 KV 数值一致性全通过。

请求数组顺序已核对实际运行的 vLLM benchmark 源码：`benchmarks/serve.py:882` 以 `asyncio.gather(*tasks)` 收集结果，`:1017` 按同一 `outputs` 顺序写入 `generated_texts`，因此原生输出数组保留输入 task 顺序。配对同时要求同一输入、参数、轮次和 request index。

## 文件与设备入口

| 文件 | 用途 |
|---|---|
| `generate_l2_cases.py` | 复用 `../basic_bench/generate_cases.py` 的 CPU tokenizer，生成四组前缀及跨阶段不同尾部；不加载模型权重 |
| `run.sh` / `observe.py` | 对已启动服务执行 seed、quick、matrix；保存失败、逐请求长度、P/D 来源和前后计数，不启停服务 |
| `summarize_l2_results.py` | 离线解析归档；吞吐用总实际输出/总原生计时，p95 不取平均 |
| `results_basic.json` | 完整基础三档的精选指标、实际计数、严格文本检查与退出限制；不含原始输出或内部环境位置 |
| `results_customer.json` | 自然四档和严格三档的分协议指标、原生路线、实际计数、owner 位置及退出摘要 |
| [`customer/`](customer/README.md) | 独立的客户结构裁剪入口、严格 C1 proxy shim、离线逐请求审计 CLI 与两协议结果 |
| `service_device_entry.py` | 在构造现有 L2 adapter 前绑定并核对 service 设备，再转交现有 LMCache CLI |
| `store_gate/` | 16 MiB client segment、4 GiB 独立 owner 的内容/位置/退出检查；默认 `PLAN_ONLY` |

本轮容器映射为物理 `8,10,12,14`：P 在 8，D 在 10，service 的逻辑 2 对应物理 12，池 owner/member 在物理 14。`service_device_entry.py` 明确断言 `logical=2 / physical=12`；改变映射时应先重新审阅这个入口。Store owner 和 native gate 分别用 `ASCEND_RT_VISIBLE_DEVICES=14` 与 `12`、`--device 0`，并核对实际 ACL 映射，形成 **12 ↔ 14** 搬运端点。此表是本轮配置记录，不表示当前设备空闲或新增使用授权。

service 的既有启动参数接在入口的第一个位置参数 `2` 之后。relay 环境设置如下；这里只展示绑定，启停仍由本轮操作记录管理：

```bash
export ASCEND_RT_VISIBLE_DEVICES=8,10,12,14
export ASCEND_LOCAL_COMM_RES=1.2 ASCEND_BUFFER_POOL=4:8
unset HCCL_INTRA_ROCE_ENABLE
# 原 LMCache CLI 的 server 参数交给 service_device_entry.py 2。
# 入口先确认逻辑 2 映射物理 12，再运行同一套 server 参数。
```

## 本地与 CPU 检查

```bash
python store_gate/check_cpu.py
python store_gate/store_owner.py
python store_gate/store_smoke.py
bash -n run.sh
AB_OUTPUT_TOKENS=32 bash run.sh --dry-run store matrix preview
```

这些检查不导入 native SDK、不初始化设备，也不发请求。`service_device_entry.py` 是实际设备入口，只做静态语法检查，不能在 CPU 检查时执行。真正运行 owner/gate 必须显式加 `--allow-npu`，使用新回执目录、已有 native 扩展和 `memory_allocator=offset` 的新 master；最终必须检查子进程退出码与 owner 存活。

本次归档本地检查已通过：7 个 Python 文件 AST、UTF-8/JSON、上述 CPU gate（含两项 `PLAN_ONLY`）、Bash 语法及 C1/C4 三轮的 12 条命令预览。此记录仅证明归档入口的本地检查结果，不新增服务器、NPU 或性能验证。

## 数据与请求口径

先在已有 CPU tokenizer 环境生成数据，两组复用相同文件。以下路径由操作者设定：`BENCH` 为本目录，`ASCEND_SOURCE` 为兼容 Ascend 源码，`MODEL` 为只读 Qwen3-8B tokenizer，`TASK` 为个人工作目录。输出目录必须尚不存在。

```bash
python "$BENCH/generate_l2_cases.py" \
  --generator-file "$ASCEND_SOURCE/examples/memfabric/basic_bench/generate_cases.py" \
  --tokenizer "$MODEL" --output "$TASK/data_3072" \
  --dataset-id l2_retest_3072_20261029 --prefix-tokens 3072 \
  --suffix-tokens 127 --output-tokens 32 --requests 32 --rounds 3 --concurrencies 1 4
```

另一个条件使用 1024 前缀、127 尾部、C1。每轮 4 条 warmup、32 条 timed、32 输出 token，要求三轮；是否完成须看精选结果，命令本身不代表已执行。`run.sh` 的 warmup/timed 输出数默认读取数据 manifest，seed 固定为 1；两组任何显式覆盖必须相同。设置 `AB_DATA_ROOT`、`AB_RUN_ROOT`、`AB_TOKENIZER` 和服务端点后，用新 tag 依次运行 seed / matrix；quick 单独保留，不计入正式三轮覆盖。

`observe.py` 校验逐请求输出数、总输入/输出、完成失败、命令与快照退出码及 P/D 来源总量；完整归档汇总进一步核对每个角色的 external/local-compute/local-hit 预期。它的 accepted 字段只代表该层检查，不能代替路线、真实读池、数值或退出验收。
