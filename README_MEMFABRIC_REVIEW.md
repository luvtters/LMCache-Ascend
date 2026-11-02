# MemFabric worker 路线：原生 L1 基础接入

当前目标是让 MemFabric HOST 池成为 LMCache 原生 L1 的实际 backing memory。对象、写入暂存、读写保护、预取与释放由原生 `L1Manager` / `StorageManager` 管理；`TensorMemoryAllocator` 直接在 BM 本地 HOST 映射上分配 `TensorMemoryObj`。NPU上的KV形状测试数据通过BM同步L2G/G2L在NPU与该对象之间搬运，控制文件只携带地址、状态等元数据。不加载L2 adapter，不修改MemFabric SDK。

**当前完成范围是原生 L1 管理与真实 NPU 搬运基础 gate，不是完整 vLLM、模型推理或 P/D 接入。** 本目录没有新的 L1 connector，也没有通过给旧 Direct connector 改名来宣称完成兼容。

## 配套源码与验证边界

- 当前开发分支：两仓库同名 `work/memfabric-l1-compat-20261102`；长期检视入口仍为 `review/memfabric-worker-direct`，是否已同步以实际分支 tip 为准。
- 本轮 Ascend 起点：`13ad6e39e6a1b300dc310a7219d1d6bcfae97430`，官方固定基线 `c5e29f15902437883dba530c22705a48ff7ce0fb`。
- 配套 core 起点：`26e0d7e01f95d81edcc3c8b7d77c89399c851a23`，官方固定基线 `6913259264dbebb2fa5ba1665606c40ffc5c6661`。必须同时应用本目标的 core 差异；core 起点本身没有 `memory_manager` 注入接口。
- 2026-11-02，本路线独立分支加本轮源码差异在 M8 完成 native L1 CPU gate（退出码 0）及 18 MiB BF16 KV 形状对象的真实 NPU→BM HOST 原生 L1→NPU gate（pool、worker 均退出码 0）。结果覆盖写完成前不可读、L1 预取命中、读保护阻止删除、内容一致、读完成后释放和零占用；不能推导完整模型性能、P/D 或跨物理机能力。
- 上述设备 gate 使用下方两个入口。随后只移除无引用的旧 Direct 实现与专用 core hooks，保留已验证的 BM、L1 backing、core 注入及 gate 方法体；清理后的 CPU 复核由本轮原件单独记录。

## 入口

`lmcache_ascend/v1/distributed/memfabric_l1_memory.py` 只提供实际内存 backing 和 native 对象地址转换；BM session 继续由调用方持有。关闭顺序是原生读写保护及对象释放→L1 allocator→BM session。存在活跃 reservation 时显式注入的 L1 manager 拒绝关闭；未知设备完成时 gate 保留对象和 session 等待外部守卫，超时不视为 DMA 已完成。

CPU gate 使用真实 native L1 管理与 allocator，以 CPU tensor 模拟 BM arena，不初始化 SDK 或 NPU；需要正常安装的 `lmcache_native` 扩展：

```sh
python examples/memfabric/validate_l1_backing_cpu.py \
  --core-source /absolute/path/to/worker-core \
  --ascend-source /absolute/path/to/worker-ascend
```

设备 gate 需要已有授权、独立截止守卫和两张已分配设备；以下是入口说明，不代表新的执行授权。两个进程在同一个全新结果目录并行运行，`--device` 使用各自可见设备中的逻辑号：

```sh
python examples/memfabric/validate_l1_backing.py \
  --role pool --device 0 --core-source /absolute/path/to/worker-core \
  --ascend-source /absolute/path/to/worker-ascend --run-dir /private/new-run \
  --store-url tcp://127.0.0.1:28890 --device-nic tcp://127.0.0.1:10005 --allow-npu
# 另一进程同参数，--role worker，分配另一张设备并映射为其逻辑 0。
```

默认对象形状为 `[2, 36, 128, 1024]`、BF16，单向 18 MiB。双方均等待真实 after-mmap JOIN 后才发布就绪并开始搬运。脚本不读取模型，不启动服务或完整推理。

## 下一步生产接线

当前生产service尚未创建此backing；gate使用显式原生管理器注入。下一步接入service的L1创建路径，并扩展原生engine-driven的service transfer strategy及worker内层context/搬运hook；复用prepare/commit RPC和scheduler，不重建PoolIndex。BM arena不等于具名SHM，不能留空描述后默默回退inline RPC。

固定BM组还须处理service RPC就绪与worker join的启动依赖，避免互等。生产接线、真实分页模型KV和P/D不属于本节点完成范围。

## 旧 worker 直传归档

旧 DirectConnector、DirectTransferContext、PoolIndex、服务模块、样例及专用测试已从当前 L1 目标移除。旧实现保留在本地 tag `archive/memfabric-worker-direct-pre-l1-20261102`，对应 Ascend `13ad6e39e6a1b300dc310a7219d1d6bcfae97430` 与 core `26e0d7e01f95d81edcc3c8b7d77c89399c851a23`。归档 tag 是否已推送以远端回读为准；旧模型记录仍按其原快照解释，不转移为新 L1 接入验收。
