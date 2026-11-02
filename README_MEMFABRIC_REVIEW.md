# MemFabric L2 兼容路线：独立检视入口

**独立分支设备复验：NOT_VERIFIED。** 以下设备记录只属于原运行快照；源码提取和本次 CPU 通过不能替代独立分支上机复验。

此分支只接入 LMCache 现有 L2 插件接口，不修改 LMCache core，不包含 Direct Connector、Direct TransferContext 或 PoolIndex。SDMA 和 DEVICE_RDMA 是同一 L2 路线的可选搬运配置，不是两条接入路线。

## 改了什么

`LMCacheMPConnector → 原 LMCache CPU L1 → MemFabricL2Adapter → BMSession → 池 DRAM`，读取反向。NPU 与 CPU L1 之间仍用原来的 LMCache 链路。

- `memfabric_l2_adapter.py`：实现已有 L2AdapterInterface，管理 key/slot、查找保护、淘汰、任务通知。
- `memfabric_bm.py`：固定 BM 会话、成员就绪、H2G/G2H、CPU MemoryObj 注册及引用保护。
- `pool_member.py`：贡献 HOST DRAM；`pool_coordination.py` 是本路线独立的纯 CPU HTTP 就绪协调器。

## 固定基线与来源

官方 Ascend 基线 `c5e29f15902437883dba530c22705a48ff7ce0fb`。实现来自已上机的 `25ceb46eb1293e4119cc32f8c841b3b8054dcaea`，本分支删除未使用的 NPU 直传 BM 方法及对应注销分支；现有 CPU 搬运方法体保持一致。未纳入批量搬运、整段 L1 注册复用或 allocator lease 候选。

接口基于未修改的 LMCache `6913259264dbebb2fa5ba1665606c40ffc5c6661`。历史 SDMA L2 实验使用该 core；后续真实 RDMA 服务实验复用了公共运行基线 `5b565d2`，不能声称拆分后的组合已重新上机。

## 已跑通与限制

历史 A3 SDMA 单机 L2/模型恢复通过；后续固定 SDK 1.2.1 的 DEVICE_RDMA 真实跨机 HOST 池、MixedSHM 18MiB 正反字节和 Qwen3-8B TP1 恢复通过。P/D 同机、池跨机；1373 输入恢复 1280 tokens，实际 180MiB。自然服务对照没有稳定性能优势；仍有生命周期和部分输出差异问题。功能通过不等于全量验收。

本次仅做源码整理及本地 CPU 检查，不重新占卡，不修改 SDK。

## 最小接入入口

已有 `l2.json` 默认 SDMA。使用原 LMCacheMPConnector，service 增加：

```sh
lmcache server --host 127.0.0.1 --port 25555 --chunk-size 128 \
  --l1-size-gb 2 --max-gpu-workers 2 \
  --l2-adapter "$(cat examples/memfabric/l2.json)"
```

运行前仍需准备 Ascend 环境和池成员。跨机 DEVICE_RDMA 必须显式配置 data_op_type、device_nic、enable_56bits_gva、join_coord_url、每次唯一 join_group_id；service 与 pool_member 的固定 BM 组一致。真实共享 L1 的 Host pin 设备与 BM 设备组合需沿用已验证部署，不能仅换这个参数就宣称任意环境可用。测试脚本 `l2_rdma_memoryobj_gate.py` 默认只生成计划，真实执行需 --allow-npu。

## CPU 检查

```sh
python -m unittest discover -s tests/v1/distributed -p 'test_memfabric*.py' -v
```

使用假 SDK 和本地 HTTP 控制面，不证明真实 RDMA 或性能。
