# 8 环境：BM / TE 高速搬运对照

日期：2026-10-28，北京时间。四组正式采样全部通过；约 **01:25 已停止本项目实验及准备容器并释放设备**。本轮只验证 CPU DRAM 搬运，没有加载模型、运行 worker-direct 或修改 LMCache 生产代码。

结论：**当前兼容 BM copy 没有表现出普遍优势。** 对 TE relay，BM 在 1 MiB 较快，18/64 MiB 较慢；TE RoCE 三档均较快。不能从昨天 BM 对 TCP 的优势推导出 BM 对高速 TE 的优势，也不能据本轮数据认定接入 MemFabric 会提高推理性能。

## 正式数据

下表为有效 GB/s，单元格为 **写 / 读**。GB 使用十进制，载荷 MiB 使用二进制；带宽按全部成功载荷字节除以同步 API 总耗时计算。

| 载荷 | BM SDMA | TE relay，LocalCommRes 1.2 | TE Ascend RoCE | TE 单机 SHM |
|---|---:|---:|---:|---:|
| 1 MiB | 4.866 / 4.582 | 3.917 / 3.919 | 8.147 / 8.352 | 18.377 / 18.349 |
| 18 MiB | 14.187 / 14.744 | 18.815 / 18.677 | 22.066 / 22.040 | 17.672 / 17.764 |
| 64 MiB | 20.716 / 18.364 | 25.489 / 26.375 | 23.694 / 23.723 | 18.216 / 16.079 |

BM 相对 relay 的平均耗时：1 MiB 写/读分别约低 **19.5% / 14.5%**；18 MiB 分别高 **32.6% / 26.7%**；64 MiB 分别高 **23.0% / 43.6%**。平均值、p50、p95 和全部单次样本见以下归档；不能把每个样本当成独立部署重复。

- [BM / relay 完整六格统计](comparison_final_hccs.md)
- [BM / RoCE 完整六格统计](comparison_final_roce.md)
- [BM / SHM 完整六格统计](comparison_final_shm.md)

## 已统一的条件和证明

同一台 `115.120.47.8`、同一私有容器、物理设备 8/10、两端 CPU affinity 96–119、分配前 MPOL_BIND(4)。三档 1/18/64 MiB，预热 20、100 次计时 × 3 次重复，双向各 300 次；batch=1、outstanding=1、单 Python 提交线程。BM 本地缓冲显式使用 64 KiB 对齐；TE pinned Host 及 SHM 本次观察到的地址也均为 64 KiB 对齐。

每组 **1,800 次计时 copy**、18 个统计单元均通过；四组共 7,200 次。每档都检查了对端写入、破坏本地源数据后的回读，以及对端独立生成不同数据后的读取；双 worker 均以 0 退出。分配、注册、建链、预热和正确性检查在搬运计时之外。固定地址反复使用，属于热缓冲测试。

最终正式序列为 BM → relay → RoCE → SHM，约一分钟完成；没有随机化顺序或跨时段重复。此前反向顺序 RoCE → BM 的有效采样使用了 BM 4 KiB 对齐，只作排障参考，不混入最终 64 KiB 表。最终同步设备监控 21 次，所有观测到的占卡 PID 的 cgroup 均属于本项目实验容器，没有未知归属项。

## 实际路径证据

**TE RoCE 已有运行日志证据。** 对应正式 worker 的 HCCL 日志记录 `transporttype[ROCE]`、两端 NPU IP 和 RDMA QP；设备侧也记录了 ROCE transport。TE 双端均未导入 MemFabric、未映射 `libmf_smem.so`。这组不能再称为 TCP 对照。

**TE relay 已验证数据面，物理 HCCS 流量尚未计量。** 当前 CANN 默认通信资源 1.3 选择 HixlEngine，拒绝非 `0:0` BufferPool；显式 `--ascend-local-comm-res 1.2 --ascend-buffer-pool 4:8` 后通过。日志记录 BufferPool、8↔10 P2P，以及 `HCCS_SW_NUM[2]` 拓扑；本轮没有链路字节计数器，不能把拓扑/P2P 日志等同为每字节实际 HCCS 归因。结果表保留 relay 名称。

**BM 保持兼容 wrapper。** 使用 `525a710` 的 BMSession、两成员各 1 GiB DRAM 池，H2G/G2H，显式请求 SDMA。未修改其拷贝语义，也未声称量化 HCCS 链路收益。SHM 使用独立 CPU TE 原生桥接，没有 Torch/NPU/MemFabric 导入或加速器库映射。

[路径补充证据 JSON](route_evidence.json) 独立于不可变原始测量；原始 configuration 中的 `NOT_VERIFIED` 保持采集时状态，由这份补充证据明确其后核实范围。

## 仍然不同，不能归因成纯后端差异

| 项目 | BM | TE Ascend / SHM | 当前处理 |
|---|---|---|---|
| 本地内存 | 普通 POSIX 页 | ACL pinned Host / 共享 mmap | BM ACL Host 原 H2G 调用实机返回方向不匹配；未改兼容语义，差异保留 |
| 目标内存 | BM 管理的池 DRAM | TE pinned Host / 共享 mmap | 所有权与 allocator 不同 |
| 注册 | wrapper 无显式预注册接口 | RoCE 显式注册；relay 跳过 Host 注册 | 设置阶段单列，BM 内部行为未消除 |
| NUMA | 本地普通页观察到 N4；池页未知 | pinned VMA 无物理节点统计；SHM 观察到 N4 | 仅进程策略相同，不能宣称全部物理内存同节点 |
| 传输 | SDMA | relay / NPU RoCE / CPU memcpy | 分组报告；不是相同链路上的纯库开销 |
| 后台线程 | BM 内部线程 | ADXL/HCCL/HIXL 或 CPU TE 线程 | 记录线程快照；相同提交数不等于相同 native 线程或 CPU 成本 |

因此，本轮能够比较这些**实际可运行路径的综合搬运表现**，不能严格隔离“只换 MemFabric 一个变量”。单机 SHM 很快也不能推出它能替代跨服务器池；relay 的吞吐优势不能直接推出 TTFT 或推理吞吐优势。

## 修复和复现

本轮修复只在独立压测材料：classic TE 使用四个必需 copy 参数；relay 的 RoCE 环境开关 unset；清理并显式记录 LocalCommRes；记录实际加载的 Mooncake engine 路径。17 项定向 CPU 检查通过，独立静态审阅无阻断问题，设备最终四组验收通过。

迁移缺少私有 `mooncake/__init__.py` 曾导致镜像预装包覆盖独立构建，已在实验容器补齐 marker。归档保留失败和修复前记录；正式结果均加载私有 `engine.cpython-312-aarch64-linux-gnu.so`。安装包元数据中的 `mooncake-transfer-engine-npu=0.3.11.post1` 是镜像自带发行包，**不是本次实际加载的源码构建版本**。

源码构建来自准备记录固定的 Mooncake `f751cff2b67668ae97f1ff7e45fb2154fdbff64a`；MemFabric 1.2.1 只在私有 BM venv，CANN 主机库 9.1；设备日志显示既有 hcomm 兼容插件版本 9.2.1。本轮没有升级依赖、驱动或共享插件。

准确参数见每组 `.receipt.json`；可复用 [压测入口](../../README_transfer_bench_fast.md)。设备容器为 `memfabric-d00896497-478-bm-te-20261028`，无 bind mount，文件保留在私有 `/workspace/d00896497/te_fast_20261027/runs_20261028`；已停止，后续启动需遵守实际授权，不能自动恢复测试。

完整归档：本地 research/artifacts/2026-10-28_478_fast_compare 完整运行材料（另行归档）、[释放回执](final_release.json)。初始约 01:05 无占卡任务，随后出现 TP16 工作；复核均属 `test-eillon-dsa` 后按用户授权停止该容器，保留文件和镜像，未清理无关容器。

后续方向：优先让另一个会话继续真实 KV worker 直传验收；完整服务 A/B 仍须比较保留 LMCache 的无 MemFabric / 有 MemFabric，记录强制 L1 未命中、实际读池与恢复/重算 tokens、相同容量和请求。当前底层结果不能替代它。

设备释放并归档后，已向目标“2. 解读 MemFabric 文档与仓库”发送且仅发送 **“执行”**；现有会话 owner 返回成功，新 turn 状态为 inProgress。[交接回执](handoff_receipt.json)。这证明指令已被接受，不代表目标会话后续实验已完成。
