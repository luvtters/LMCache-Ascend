# Worker 直传实验入口归档

日期：2026-10-29。结果状态：**PARTIAL_COMPLETION**；功能验收和一轮串行探索对照通过，指定重复性能矩阵及异常实验未完成。详见[阶段实测报告](RESULTS_zh.md)及[精选证据](selected_evidence.json)。CPU检查、真实搬运、性能和退出分别记录。

前提为 LMCache core `8bc9edc`、LMCache-Ascend `7e5d40a`，Qwen3-8B / TP1、固定 BM 成员、DRAM 池及已准备的 Ascend 环境。没有增加 SDK 或依赖版本。新组必须使用新 case；未知 DMA 完成时不能靠超时回收地址，成员失效后不能原地重新加入旧组。

| 组别 | D 的恢复路径 | 要回答的问题 |
|---|---|---|
| A | 保留 LMCache，无 MemFabric、无可复用 KV，D 全量重算 | A/C 比较读池与重算 |
| B | MemFabric 池 → 临时 CPU L1 → NPU；`skip_l1`、默认 prefetch | B/C 比较两条完整接入路径 |
| C | MemFabric 池 → worker 的 NPU scratch → 实际 KV 页 | 当前 worker 直传方案 |

三组 D 均为 `kv_consumer`，关闭 vLLM 本地前缀缓存。P 播种、consumer 准入及四条全长度预热在正式计时之外。每个正式窗口检查成功/失败数、真实输入输出、P 无新增请求、D 的 external/local-compute/local-prefix、适用保护归零及实际完成读池字节。不能用恢复 token 数换算后冒充测得的搬运量。

正式 C4 用例采用四个不同的 1373-token 前缀轮转，输出 128，首个 128-token chunk 互异。B/C 均用 2 GiB 池、32 MiB slot，即 64 槽，工作集占 40 槽。每个前缀预期恢复 1280 tokens；实际是否一致仍由运行指标验收。同前缀并发可能出现临时 L1 合并或重算，恢复量不齐的窗口不能进入等工作量主对照。另有 3000-token 单前缀 C1 基础用例，预期可缓存 2944 tokens。两者均为自建合成材料，不是客户 GLM 原件，也不涉及其他模型部署。

## 文件与使用

| 文件 | 职责 |
|---|---|
| `control.py`、`run_backend.py` | 生成/执行本轮容器内计划，按已有入口启动服务、采样和受控关闭；默认只打印计划 |
| `supervisor.py`、`service_entry.py` | 保存真实子进程 `wait()` 退出码、信号发送与 CLI 收尾标记 |
| `prepare_multi.py`、`prepare_long.py` | 使用本地 tokenizer 生成并校验完整 chat 数据；不下载、不改模型 |
| `prepare_short.py` | 将同输入另存为8-token输出候选数据；本轮未执行该变体，不能混入128-token结果 |
| `seed_check.py`、`bench_d.py` | 播种/连续读池准入及 D 定点官方 `vllm bench serve` |
| `counted_direct_connector.py` | 正式窗口的低开销内存计数与只读 HTTP 快照 |
| `diagnostic_direct_connector.py` | 独立组计时 worker retrieve 和控制 RPC，不替代正式性能组 |
| `pool_cases.py` | 1 GiB 池、两个 512 MiB 逻辑槽的淘汰/保护/不同 NPU 内容恢复实验 |
| `cancel_requests.py`、`check_cancel_requests.py` | 已播种 diagnostic 服务的真实 HTTP 客户端关闭与收尾观察，以及纯 CPU/loopback 检查 |
| `mixed_requests.py`、`check_mixed_requests.py` | 已有 diagnostic 组的两个 P 写入与八个 D 读取功能用例，区分客户端并存和 DMA 重叠 |
| `peerleft_probe.py` | capacity 成员退出且 service 仍存活时，执行 status→lookup→未注册身份 prepare→status；不注册 worker、不 DMA，真实 Factory 可被动导入 CPU Torch，见 [说明](PEER_LEFT.md) |

归档副本移除了个人绝对路径。容器内显式设置以下变量，并把本目录放在已确认个人运行目录的 `runtime/`；`src/LMCache`、`src/LMCache-Ascend` 和保留 SDK 路径的 `env.sh` 由执行者准备：

`runtime/` 还必须同时放入现有 `examples/memfabric/direct_pd_gate.py`、`traced_direct_connector.py`、`followup_direct_requests.py`。它们提供严格 gate、trace Connector 和追加请求入口，本归档不重复复制；本轮实际运行的上传器会从固定 Ascend 源码补齐这三项依赖。

```bash
export MF_EXPERIMENT_ROOT=/path/to/confirmed-private/run
export MF_EXPERIMENT_PYTHON=/path/to/venv/bin/python
export MF_MODEL=/path/to/read-only/Qwen3-8B
python "$MF_EXPERIMENT_ROOT/runtime/control.py" start --kind C --case review --pool-gb 2
```

上面仅打印计划；`--run` 才执行。数据准备使用[既有严格 gate](../../README_direct_validation_zh.md)生成的 `request.json`，保留完整 messages，不从 user 文本单独重建。路径示例不构成挂载或设备使用授权。`control.py` 保留本轮固定开始/截止时间，后续须在新的明确授权后更新，不能沿用过期窗口。操作细节见 [压测说明](bench_d_README.md)、[计数说明](COUNTERS.md)、[诊断说明](DIAGNOSTIC.md)、[容量实验](pool_cases_README.md)、[取消实验](CANCEL.md) 和 [混合功能用例](MIXED.md)。

当前 CLI 收到受控 SIGINT 时可按契约退出 **130**。验收保留实际 `returncode=130`、`normal_exit=false`，只有匹配本次发送者、child PID/PGID 和真实 `wait()` 的信号证据时，才分类为受控关闭；其他角色仍要求 0。日志打印关闭完成不等于退出 0，也不能据此解释历史 SIGKILL。

## 证明边界与本地检查

控制 RPC 传递 key、地址描述符、保护及完成状态等元数据，不传 KV payload。BM `completed_bytes` 只统计同步成功返回的实际 copy 调用字节；它不是物理 HCCS 链路计数器。RPC 计时嵌套在 retrieve 内，不能相加，也不能把 TTFT 剩余全部归因于 RPC。CPU fake/Future 检查不证明 NPU、硬件链路或推理性能。

本归档的路径参数化副本仅做了语法、默认计划和关键纯 CPU 检查；M8 实际执行入口、信号记录及内部原始结果保留在本地，未复制到本目录。诊断 CPU 检查需显式指定固定 core 源码：

```bash
export MF_CORE_SOURCE=/path/to/LMCache
python check_counted_direct_connector.py
python check_diagnostic_direct_connector.py
```
