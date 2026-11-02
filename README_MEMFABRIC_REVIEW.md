# MemFabric worker 直传路线：独立检视入口

**独立分支设备复验：NOT_VERIFIED。** 以下设备记录只属于原运行快照；源码提取和本次 CPU 通过不能替代独立分支上机复验。

配套 core：luvtters/LMCache 的 `review/memfabric-worker-direct`。不加载 L2 adapter；此分支不是已经完成的 L1 兼容实现，L1 兼容仍是后续目标。

## 数据与控制怎么走

`P 真实 KV → NPU gather scratch → BM L2G → HOST DRAM 池`；D 通过 G2L 回到 NPU scratch，再 scatter 到真实 KV。service 管理对象、地址、提交、pin 与 read lease；RPC 只带元数据。

- `memfabric_direct_connector.py` 选择配套 scheduler/worker adapter。
- `memfabric_direct.py` 的 MemFabricDirectTransferContext 是本路线新增的 worker 传输实现，不能描述为只改 L2 插件。
- `memfabric_direct_transfer.py` 复用原 NPU gather/scatter 能力。
- `memfabric_direct_server.py`、`memfabric_pool_index.py` 管理独立池对象。
- `memfabric_bm.py` 封装 BM；移除了此路线不用的 CPU MemoryObj L2 注册方法，不修改 SDK。

## 固定基线与真实验证

官方 Ascend 基线 `c5e29f15902437883dba530c22705a48ff7ce0fb`；生产实现提取自 `25ceb46eb1293e4119cc32f8c841b3b8054dcaea`，五个直传实现文件逐字保留。配套 core 实现来自 `5b565d211416cd48bebf20bae12a4abc15a72ba9`。

2026-10-31 Qwen3-8B 跨机 TP1 三种放置、P/D 分机 TP2/TP4 严格功能检查通过。HOST DRAM 池、真实 NPU scratch 注册、同步 L2G/G2L；1373 输入中恢复 1280、本地计算 93、本地前缀 0，答案 ORCHID，单侧搬运 180MiB。不是性能结论。关闭时仍存在强制退出、资源追踪和 native 告警，不能称稳定性验收通过。

本次只整理分支和运行 CPU 检查，没有重新进行设备实验，也没有加入尚未验证的优化。

## 入口

`direct_server.json`、`direct_p.json`、`direct_d.json` 是原 TP1 SDMA 接入样例；实际物理设备和只读模型路径按获批环境选择。固定成员的 P/D 应并行启动。

```sh
lmcache server --host 127.0.0.1 --port 25555 --chunk-size 128 \
  --l1-size-gb 1 --max-gpu-workers 2 \
  --server-module "$(cat examples/memfabric/direct_server.json)"

vllm serve /path/to/Qwen3-8B --served-model-name mf-direct \
  --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 4096 \
  --block-size 128 --enforce-eager --port 28001 \
  --kv-transfer-config "$(cat examples/memfabric/direct_p.json)"
```

D 使用 direct_d.json 和独立端口；完整 1P1D 还需原 P/D proxy。跨机 DEVICE_RDMA 在 service config 和每个 worker 的 lmcache.mp.direct 中显式加入相同协议/56位配置、各自 NIC、就绪协调地址/唯一组号，`init_timeout` 可设置 300。保持已验证设备映射和 HOST 池；使用 MF_HYBM_RDMA_SWAP_SPACE_SIZE=0。示例不是自动部署或新的设备授权。

## CPU 检查

设置 LMCACHE_CORE_SOURCE 指向配套 core，再运行：

```sh
python -m unittest discover -s tests/v1/distributed -p 'test_memfabric*.py' -v
```

这些检查使用假 SDK/引擎边界，不能替代设备验证。`direct_pd_gate.py`、`traced_direct_connector.py` 保留功能检查入口，不用于自然条件性能结论。
