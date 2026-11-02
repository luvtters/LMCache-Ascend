# MemFabric L2 兼容路线：独立检视入口

**当前结论：** 本轮 03/04/07/08 的 strict gate 均有模型功能通过记录；07/08 使用 Ascend `075186f`、官方 core `6913259`。多 chunk 冷写入与 repeat 已验证；P/D 仍以 `-6` 退出。08 不同前缀并发能正确恢复，07 同键并发的第二次 L1 reservation 被拒，原因未定。功能、并发观察、进程退出与 SDK 告警分开记录；不是性能或生产稳定性验收。

此分支只接入 LMCache 现有 L2 插件接口，不修改 LMCache core，不包含 Direct Connector、Direct TransferContext 或 PoolIndex。SDMA 和 DEVICE_RDMA 是同一 L2 路线的可选搬运配置，不是两条接入路线。

## 改了什么

`LMCacheMPConnector → 原 LMCache CPU L1 → MemFabricL2Adapter → BMSession → 池 DRAM`，读取反向。NPU 与 CPU L1 之间仍用原来的 LMCache 链路。

- `memfabric_l2_adapter.py`：实现已有 L2AdapterInterface，管理 key/slot、查找保护、淘汰、任务通知。
- `memfabric_bm.py`：固定 BM 会话、成员就绪、H2G/G2H、CPU MemoryObj 注册及引用保护。
- `pool_member.py`：贡献 HOST DRAM；`pool_coordination.py` 是本路线独立的纯 CPU HTTP 就绪协调器。

## 固定基线与来源

官方 Ascend 基线 `c5e29f15902437883dba530c22705a48ff7ce0fb`。实现来自已上机的 `25ceb46eb1293e4119cc32f8c841b3b8054dcaea`，本分支删除未使用的 NPU 直传 BM 方法及对应注销分支；现有 CPU 搬运方法体保持一致。未纳入批量搬运、整段 L1 注册复用或 allocator lease 候选。

接口基于未修改的 LMCache `6913259264dbebb2fa5ba1665606c40ffc5c6661`。本轮独立路线已重新使用该官方 core 验证，不依赖 worker 的 core 修改；历史快照仍按原提交保留。

## 已跑通与限制

历史 A3 SDMA 单机 L2/模型恢复通过；后续固定 SDK 1.2.1 的 DEVICE_RDMA 真实跨机 HOST 池、MixedSHM 18 MiB 正反字节和 Qwen3-8B TP1 恢复通过。P/D 同机、池跨机；1373 输入恢复 1280 tokens，实际 180 MiB。自然服务对照没有稳定性能优势；仍有生命周期和部分输出差异问题。功能通过不等于全量验收。

本轮新增 `examples/memfabric/l2_model_gate.py`、说明和配置样例；MemoryObj gate 增加协议选择，默认仍为 DEVICE_RDMA。未改生产 adapter/BM、MemFabric SDK、公共模型或驱动。

两种协议各通过 A/B 两种 18 MiB 内容的 store/load，破坏源缓冲后恢复一致，引用和 pins 归零。TP1 模型请求 1373 输入，D 恢复 1280、本地计算 93，实际读池 188743680 字节，prefetch 为 10/10 且 0 L1/10 L2，答案 `ORCHID`。P/D/池成员均在同一服务器，不能称跨物理机。

模型 gate 关闭本地前缀缓存、使用原生 `skip_l1` 策略，relay 等写池完成且原生 L1 排空后才放行 D，没有手工删除 L1。这是证明来源的受控功能条件，不能用于客户自然服务性能结论。首次监督器误查不存在的 proxy `/health` 而停在请求前，该失败单独归档；改查已有 OpenAPI 接口后两轮功能通过。

退出仍有边界：SDK 有 `free sq_cq -8` 告警；进程组 SIGTERM 及父进程 SIGINT 两种停止试验均出现 P/D `-6`。父进程 SIGINT 时 proxy 0、service 130、member 0；原始 wait 与发送信号分别保留。不把功能通过说成无告警、无退出异常。

## 最小接入入口

已有 `l2.json` 默认 SDMA。使用原 LMCacheMPConnector，service 增加：

```sh
lmcache server --host 127.0.0.1 --port 25555 --chunk-size 128 \
  --l1-size-gb 2 --max-gpu-workers 2 \
  --l2-adapter "$(cat examples/memfabric/l2.json)"
```

运行前仍需准备 Ascend 环境和池成员。跨机 DEVICE_RDMA 必须显式配置 data_op_type、device_nic、enable_56bits_gva、join_coord_url、每次唯一 join_group_id；service 与 pool_member 的固定 BM 组一致。真实共享 L1 的 Host pin 设备与 BM 设备组合需沿用已验证部署，不能仅换这个参数就宣称任意环境可用。测试脚本 `l2_rdma_memoryobj_gate.py` 默认只生成计划，真实执行需 `--allow-npu`。

## CPU 检查

```sh
python -m unittest discover -s tests/v1/distributed -p 'test_memfabric*.py' -v
```

使用假 SDK 和本地 HTTP 控制面，不证明真实 RDMA 或性能。

## 本轮验证补记（2026-11-02）

本轮开发分支为 `work/memfabric-l2-lifecycle-20261102`；设备组 07–11 使用 Ascend `075186f`、官方 LMCache core `6913259`。本节点新增 shutdown timeout 选项、双协议模型 gate、七项基础用例及协调器 SIGINT 收尾修复；没有修改生产 adapter、BM 封装或 core。

- 03 实际使用 Ascend `8bb28ad` / core `6913259`；04 实际使用 Ascend `6817f94` / 同一 core。两轮 strict gate 均为 1373 输入、1280 外部恢复、93 本地计算，P 写入与 D 读取各 180 MiB，0 L1/10 L2，答案 `ORCHID`；repeat 与 partial-prefix 功能项通过。
- 03/04 的 cold 请求各为 73 tokens，小于 128-token chunk，不能证明多 chunk 冷池写入。并发两请求合计 2746 输入、1280 外部恢复、1466 本地计算、0 本地缓存命中；结果与一条复用、一条重算相符，具体内部原因尚未确定。
- 04 设置 30 秒 drain 后 P/D 仍以 `-6` 退出。05 无 connector、无推理请求，P 退出 0；06 使用 LMCache connector/native L1、无 MemFabric L2/BM/池成员且无推理请求，P 仍以 `-6` 退出、service 以 130 退出。这说明本轮 `-6` 不需要 MemFabric 参与，但不足以归因给 LMCache core；SDK、native 扩展和运行环境仍未排除。
- SDMA 组 07 实际使用 Ascend `075186f` / core `6913259`；strict gate 与六项 basic 功能用例通过。多 chunk 冷请求为 1,758 tokens、13 个完整 chunk，首次全量本地计算并新写 234 MiB；同 body repeat 从 L2 恢复 1,664 tokens，本地计算 94。P/D 仍以 `-6` 退出，功能结果不代表生命周期通过。
- 07 同键并发 DEBUG 中，两条请求都 lookup 到 10/10 个 L2 keys；一条恢复，另一条在 L1 `reserve_write` 得到 `KEY_NOT_WRITABLE` 后最终 0 hit、没有第二次 L2 load。不能把它归为池 miss，也不能据此断言 core 缺陷；原因仍未确定。
- RDMA 组 08 使用 Ascend `075186f` / core `6913259`；strict gate 与七项 basic 通过。不同前缀并发两请求共 3,131 tokens，来源为 2,944 外部恢复、187 本地计算、0 本地缓存命中；P/D 仍以 `-6` 退出。
- CPU import-only 最小已测组合 `torch + torch_npu + mooncake.engine` 在不加载 LMCache 或 vLLM 时仍以 `-6` 退出；独立 native block import 对照退出 0。这缩小了当前组合复现范围，但不能证明 ABI 冲突或与设备组相同的堆根因。
- 09 在缺少可选 Mooncake 包的私有依赖视图中完成 DEVICE_RDMA strict 与七项 basic；P、D、service 实际退出 0，未出现此前的堆损坏文本。原安装和 core 未改，这是当前部署对照的结果，不是对 Mooncake 原生扩展或 SDK 的修复；SDK `free sq_cq -8` 仍有记录，旧协调器需 SIGTERM。
- 该私有环境的 SDMA 模型对照仍未完成：10 在 gate 监听端口冲突时结束，11 在模型就绪前遇到不可写的日志目录。两组均没有推理请求，不能判定 SDMA KV 搬运失败；11 清理时 D 退出 -11，保留为未解决异常。
- 协调器修复 `57f51b7` 仅显式恢复 SIGINT 处理并在 KeyboardInterrupt 后关闭 HTTP server；普通启动和继承 SIG_IGN 后 exec 两种纯 CPU 场景均完成真实 GET/POST/GET、SIGINT 退出 0 和端口重绑定。模型组未使用这项后续修复，不称模型已重新验证。
- 当前设备验证复用了历史固定环境中的 native 扩展；对应源码构建提交未建立，不能称为从当前 Ascend HEAD 全量重编。部署须先确认运行用户有可写的日志目录，并验证监听端口实际可绑定；无 LISTEN 不等于端口可绑定。
