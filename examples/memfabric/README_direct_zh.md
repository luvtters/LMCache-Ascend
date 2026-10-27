# MemFabric worker 直传：适配说明与归档入口

日期：2026-10-27。当前是 **Qwen3-8B / TP1 的实验适配版**：外部 Connector、worker 数据面和 service 控制面已经接线，并完成本地 CPU 检查；真实 RequestClient 网络通信、NPU gather/scatter、三成员 BM 与完整推理仍为 `NOT_VERIFIED`。兼容版的设备通过记录不能算作直传版通过。

## 两条路线和仓库

| 路线 | LMCache-Ascend | LMCache core | 验证状态 |
| --- | --- | --- | --- |
| CPU L1 桥接兼容版 | `compat/memfabric-l2-525a710`，固定 `525a710f55a67b3c2bc54ee153a4dca1d075ea7b` | `compat/memfabric-l2-6913259`，固定 `6913259264dbebb2fa5ba1665606c40ffc5c6661` | 历史 A3 / Qwen3-8B TP1、TP4 严格读池验收通过 |
| 新 worker 直传版 | `feature/memfabric-worker-direct` | 同名分支；当前 core 为 `04e16ff41ca3386df34e6f5349c733b3525a994d` | 本地 CPU 检查通过；真实直传推理未验证 |

个人 fork： [LMCache-Ascend](https://github.com/luvtters/LMCache-Ascend)、[LMCache](https://github.com/luvtters/LMCache)。本地 `origin` 指向个人 fork，`upstream` 指向官方仓库；旧 Ascend origin 保留为 `previous-origin`。本次使用 vLLM 的外部 Connector 加载接口，无需修改 vLLM 源码。

兼容版两种 TP 都在放行 D 前确认 L1 为空，实际从池回读 180 MiB，恢复 1373 个 prompt tokens 中 1280 个的 KV，本地前缀命中为 0；恢复仍经过临时 CPU L1。历史三档普通 A/B 的 B 组没有发生读池，不能据此判断直传收益。

## 修改思路

这次改变的是搬运的执行位置和中转缓冲。LMCache 继续管理对象，MemFabric 提供池与拷贝：

```text
兼容版：P NPU → CPU L1 → 池 DRAM → CPU L1 → D NPU

直传版：P paged KV → NPU gather 缓冲 → BM L2G → 池 DRAM
        D paged KV ← NPU scatter     ← BM G2L ← 池 DRAM

控制面：scheduler / worker ← 元数据 RPC → LMCache service
```

attention 仍读取本机 NPU KV，池仍是 DRAM。允许 NPU 上的 gather/scatter 整理，但直传命中不再调用普通 CPU L1 payload 路径。service 仍可能预留普通 L1 arena，不能把“直传不用它”说成“完全没有 L1 内存预留”。

1. **写入。** worker 先向 service 预留 slot，整理 NPU KV，执行同步 L2G。只有拷贝完成后才 commit，对象此时才可查询。描述符包含 epoch、slot 代次、storage rank、offset/size 和布局，不能把别的进程的 CPU 指针直接传来使用。
2. **读取。** scheduler 查询时加 lookup pin，防止调度等待期间被淘汰；worker 另取 read lease，G2L 到 NPU 缓冲，再 scatter 到真实 KV。完成同步后才报告加载完成。两种保护分别释放。
3. **取消和故障。** cleanup 只停止交付结果，backend 继续保留在途 lookup，end-session 排空后释放服务端保护。RPC 超时不能证明 DMA 已停；未知操作保留 slot、lease 与 scratch。部分 scatter 失败返回错误，不报告恢复成功。
4. **线程与成员。** 每个 worker 用一个专用线程统一创建 BM、执行 NPU/BM 操作和关闭，满足 owner-thread 约束。固定成员失败后不由心跳重建，也不重复注册覆盖旧 context。

## 改动在哪里

| 文件 | 职责 |
| --- | --- |
| `lmcache/integration/vllm/lmcache_mp_connector.py`（core） | 增加对称的 scheduler adapter 工厂，默认仍构造原 adapter |
| `lmcache/integration/vllm/vllm_multi_process_adapter.py`（core） | 将每请求 lookup 的构造提取为可覆写 hook，保留原提交/检查流程 |
| `lmcache_ascend/integration/vllm/memfabric_direct_connector.py` | 选择 Direct Scheduler/WorkerAdapter，接锁释放与完成语义，拒绝不支持的组合 |
| `lmcache_ascend/v1/distributed/memfabric_direct.py` | 元数据控制、稳定 chunk key、worker 传输上下文、取消与关闭排空 |
| `lmcache_ascend/v1/distributed/memfabric_direct_transfer.py` | 真实 KV 视图、NPU scratch、block gather/scatter |
| `lmcache_ascend/v1/distributed/memfabric_direct_server.py`、`memfabric_pool_index.py` | 池 owner、slot、提交可见性、pin、lease 与失效 |
| `lmcache_ascend/v1/distributed/memfabric_bm.py` | 增加 L2G/G2L，保留兼容版 H2G/G2H |

未新增通用后端加载框架；原 `memfabric_l2_adapter.py` 保持不变。默认 Connector 不加载这条直传路径。旧的 `lmcache.mp.direct_backend` 草案没有实现，不要使用它。

## 配置和后续设备入口

首版仅支持 TP1/PP1/DCP1、一个全注意力 GQA KV group、单 service、单机固定三成员和同步搬运。TP 扩展、滑窗/循环状态、多组、MLA、KV events、CacheBlend、lazy offload 和动态恢复均不支持。KV chunk 必须能放入一个 slot。

配置文件： [service](direct_server.json)、[P](direct_p.json)、[D](direct_d.json)。BM rank 分别为 0、1、2；每个推理实例自己的 TP rank 都是 0。三个进程的 BM world size、容量与 store URL 必须一致。示例每 rank 预留 2 GiB，合计 6 GiB；仅 rank 0 的 2 GiB 存对象，不能把总预留说成有效缓存容量。

以下是**待设备验证的入口示例**，执行会占卡；本轮没有执行。必须先获得环境授权、激活匹配的 Ascend 与两个 fork 环境，替换设备和只读模型路径。`device_id=0` 对应每进程掩码后的逻辑设备，实际映射须先核实。不要沿用兼容版的显式 `lmcache_driven` 环境变量；示例为 `auto`，选择的是 Direct Connector 自己的 context。

service 终端（在 Ascend checkout 根目录）：

```bash
ASCEND_RT_VISIBLE_DEVICES=14 lmcache server --host 127.0.0.1 --port 25555 \
  --chunk-size 128 --l1-size-gb 1 --max-gpu-workers 2 \
  --server-module "$(cat examples/memfabric/direct_server.json)"
```

P、D 要在两个终端**并行启动**，不能等 P 完全 ready 后再启动 D，否则固定 BM 组会等不到全部成员。P 示例：

```bash
ASCEND_RT_VISIBLE_DEVICES=8 vllm serve /path/to/Qwen3-8B \
  --served-model-name mf-direct --tensor-parallel-size 1 --dtype bfloat16 \
  --max-model-len 4096 --max-num-batched-tokens 1024 --max-num-seqs 1 \
  --block-size 128 --gpu-memory-utilization 0.7 --enforce-eager \
  --no-enable-prefix-caching --host 127.0.0.1 --port 28001 \
  --kv-transfer-config "$(cat examples/memfabric/direct_p.json)"
```

D 使用相同命令，将物理设备换为 `10`、端口换为 `28002`、配置换为 `direct_d.json`。P、D 的模型标识、dtype、chunk 与布局保持一致。这些命令只是两个引擎与池的接入入口，不包含 P/D proxy；完整 1P1D 还需要后续接入现有代理和验收流程。

退出时先停止新请求，排空并退出 P/D 引擎及 scheduler，最后退出 service。service 必须保留到 worker 和 lookup 保护释放完成；遇到控制响应或 DMA 完成未知时，不按超时强制回收池。

## 已检查和仍未验证的边界

本轮共 47 项定向 CPU 检查通过，边界如下：

| 检查 | 数量 | 实际覆盖范围 |
| --- | ---: | --- |
| PoolIndex | 10 | 真实纯 CPU 索引、提交可见性、pin 与代次 |
| direct worker / scheduler backend | 17 | 真实索引；BM、NPU、消息边界为 Fake，覆盖线程、迟到回复和资源排空 |
| Connector | 9 | 上游依赖为 stand-in；另提取真实 `get_finished` 验证 False 的失败标记 |
| BM wrapper | 2 | Fake SDK 检查 H2G/G2H 与 L2G/G2L 分发及同步 flags，不初始化 SDK |
| service | 4 | FakeSession 与最小 core stub 的启动、失效、关闭检查 |
| core 扩展点 | 3 | 执行实际方法体的源代码级契约检查，不是完整模块导入 |
| core 模块发现与消息路由 | 2 | 真实 core loader/router/envelope，CPU session 替代 BM，跳过 Ascend 包原生初始化 |

本地 Windows 缺少 LMCache native/device backend，完整 MP adapter 导入未通过。因此，接线检查和进程内消息路由不等于真实 RequestClient 网络 RPC 或完整推理验证。

在已准备的 Python 环境中，从 Ascend checkout 运行本轮定向 CPU 检查：

```bash
PYTHONPATH=../LMCache python -m unittest discover \
  -s tests/v1/distributed -p 'test_memfabric*.py' -v
python ../LMCache/tests/v1/test_mp_adapter_extension_hooks.py
```

Windows 可在 PowerShell 设置 `$env:PYTHONPATH='../LMCache'` 后执行相同 Python 命令。无需导入 MemFabric SDK 或初始化 NPU；真实 NPU scratch 的行为仍须上机检查。

下一轮依次验证真实模块导入与 RequestClient、三成员 BM 的设备映射、真实 Qwen KV gather→池→scatter 内容一致性，再做严格读池的 1P1D。必须证明拷贝端点是 NPU，并排除独立 CPU L1 中转；仅凭 L1 最终为空不足以证明直传。性能再比较无 MemFabric、兼容桥接和 worker 直传三组，分别报告总预留、有效容量和实际恢复量。
