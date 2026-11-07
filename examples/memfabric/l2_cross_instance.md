# 两套 L2 实例共享 DDR 对象与双 owner 聚合

2026-11-07 本地维护说明：当前生产源码来自 Ascend `3012c71`，配套官方 core `6913259`。双 owner 的真实容量、GLM 和性能依据见[当前验收](../../../报告/当前验收.md)。本页前半部分保留单 owner 的有效配置，双 owner 需按专节显式配置，不能把默认单 owner 示例当作聚合模式。没有本轮设备授权时只阅读方案，不执行启动命令。

本用例在两台 Linux 主机上各启动一个 `MemFabricL2Adapter`。A 为 BM rank 0，B 为 BM rank 1，两者各贡献 1 GiB HOST DRAM。默认 `storage_rank=0`，两个适配器都访问 A 的 32 MiB 槽位。A 写本机池后，B 通过 BM 从 A 的 DRAM 读取到 B 的 CPU `MemoryObj`。共享目录保存 `ObjectKey`、槽位、布局和 pin 等元数据，HTTP 事件只协调步骤。此配置可使用的对象容量为 1 GiB。若改为 `storage_rank=1`，则 A 远端写、B 本机读，需单独记录方向。

## 一次尝试的三个入口

共享模式仅用于固定的两个可信成员。协调器在内存中保存目录，每次启动产生新的 epoch，两端 BM JOIN 都必须带同一个 epoch。目录只在 BM 写入返回成功后发布对象。同一 key 写入中、等待删除或全部槽位受保护时，新写入会明确失败。若 BM 拷贝完成情况不明，保留对应预留或读锁；若控制面回复丢失，客户端停止提交。此时结束两端进程和协调器，使用新组重新预热。该实现按对象发送 HTTP 元数据请求，模型 TTFT 和吞吐须另行实测。

在两台主机的个人容器中使用同一份共享目录版本的 Ascend 和固定官方 LMCache core。每台机器先核实设备、NIC、容器和剩余授权时间。以下命令假定当前目录为该版本的 `examples/memfabric`，`python` 能导入此次检出的代码。`RUN_ID` 在两端设置为**同一个全新值**；`RUN_DIR` 是各自主机内已确认归属的本次空目录。`A_COORD_IP`、`A_STORE_IP` 是 B 能连接到的 A 端地址；每端单独填写本机的 `LOCAL_DEVICE`、`LOCAL_NIC_IP`。先明确空闲端口，再替换示例端口。

在 A 的独立终端以前台方式启动 CPU 协调器，保留终端直至 A、B 两个入口均已结束。先创建 `RUN_DIR`，等待 `listening` 记录。重试需更换协调器、run ID 和结果目录。

```bash
set -o pipefail
python -u pool_coordination.py --bind "$A_COORD_IP" --port 28581 \
  --directory "$RUN_DIR/events" --run-id "$RUN_ID" --shared-l2 \
  --storage-rank 0 --pool-gb 1 --slot-size-mb 32 2>&1 | tee "$RUN_DIR/coordinator.log"
```

在 A、B 的各自终端执行下面的 Bash 命令。A 设置 `RANK=0`，B 设置 `RANK=1`。当前共享模式仅支持 `DEVICE_RDMA`，使用各自本机的 NIC，两端均显式开启 56-bit GVA。两端都是真实的适配器成员，无需额外启动 `pool_member.py`。

```bash
export PYTHONHASHSEED=0
GATE_ARGS=(--rank "$RANK" --device "$LOCAL_DEVICE" --storage-rank 0
  --store-url "tcp://$A_STORE_IP:28580" --device-nic "tcp://$LOCAL_NIC_IP:28582"
  --join-coord-url "http://$A_COORD_IP:28581" --join-group-id "$RUN_ID"
  --data-op-type DEVICE_RDMA --enable-56bits-gva --payload-bytes 18874368)

# 计划模式仅写本地 JSON，可在协调器未启动时运行。
python -B l2_cross_instance_gate.py "${GATE_ARGS[@]}" --output "$RUN_DIR/plan-$RANK.json"

# 获得本轮设备授权并确认独立停止措施后，再并行执行两端。
set -o pipefail
python -u l2_cross_instance_gate.py "${GATE_ARGS[@]}" --allow-npu \
  --output "$RUN_DIR/rank-$RANK.json" 2>&1 | tee "$RUN_DIR/rank-$RANK.log"
rc=${PIPESTATUS[0]}; printf '%s\n' "$rc" > "$RUN_DIR/rank-$RANK.launcher-exit"
```

每端都先核对 rank、共同配置和对端 Linux boot ID。物理主机身份另由本轮主机与容器清单确认。两个原生 worker 各自完成最后一次 I/O 和解锁检查，并交换 `io-done` 后才关闭 BM。外层进程各自 `wait` 本地 worker，再通过协调器核对对方的真实退出码。单项任务或对端事件默认超时 90 秒，整个本地 worker 默认上限 480 秒，超时会判失败并记录退出状态。读回两端 launcher 退出码后再停止协调器，保存其退出状态和设备释放回执。

## 判读回执

两端各构造 `ObjectKey(b'X' * 32, 'l2-cross-instance-gate', 0, 0, '')`。`kv_rank=0` 描述模型 KV 分片，和 B 的 `rank_id=1` 可以同时成立。下列步骤对**同一个 key**执行两轮，期望字节分别为 `(位置 + 17) % 251` 和 `(位置 + 93) % 251`。

| 操作顺序 | 必须取得的证据 |
|---|---|
| B 查找，随后 A 写入 | B 首查 miss，A 的 store completion 等于对象长度。A 在发布 `stored` 前覆盖原源缓冲。 |
| B 查找并 pin，首次 load | B 收到 hit，每次先将目的缓冲填为 255。适配器 `load_bytes` 增量等于对象长度，逐字节比对全量成功。 |
| A 删除，B 仍持 pin | A、B 分别发起新 lookup，均须 miss。B 使用旧 pin 再次 load，同样核对完成字节和全量内容。这里的新读者是另一次 lookup，物理参与者始终为 A、B。 |
| B 解锁，进入下一轮 | B 在同一 I/O 队列提交后续 lookup 并等待完成，确认 miss、全局 `total_pins=0`、`objects=0`。第二轮通过同 key 新内容验证删除后重写。 |
| 结束 | 每端 `rank-N.json.worker.json` 保存逐项证据和 close 后的 allocator 检查。每端 `rank-N.json` 需为 `TWO_ADAPTER_CONTENT_AND_PROCESS_EXIT_PASSED`，两份 `launcher-exit` 均为 `0`。 |

## 双 owner 容量验收

协调器和两端参数都改用 `--storage-ranks 0 1`，每端容量和槽大小必须一致。模型插件配置对应 `"storage_ranks": [0, 1]`，省略 `storage_rank`。`pool_gb` 按 owner 计算。目录全局槽 `0, 1, 2, 3, …` 依次表示两台的本地槽 `0, 0, 1, 1, …`。

- 137 单机基础关使用同一容器内两个进程、不同的设备和 NIC 端口。两端均加 `--same-host --storage-ranks 0 1 --pool-gb 1 --slot-size-mb 32 --object-count 4`，保存为单机双 rank 证据。
- 137＋25 容量关使用不同物理机。协调器配置 `--storage-ranks 0 1 --pool-gb 8 --slot-size-mb 16`。两端使用同一组几何并加 `--object-count 960 --payload-bytes 9222144 --require-overflow --worker-timeout 1800`。按现场实际剩余授权另设独立停止措施。
- A 写入全部对象后才放行 B。B 通过同一适配器读回并逐对象比较完整内容。两份 worker 回执中的 `two-owner-payload` 都须通过，`completed_bytes` 为 `8853258240`，两个 owner 的对象和完成字节均为正，且无待发布写入或淘汰。最后还须确认跨两个 owner 的删除释放、两个 worker 的真实退出和设备释放。此关使用合成对象。真实 GLM 精度及性能另跑。

使用默认 18 MiB 对象时，A 两次 store 应合计完成 36 MiB，B 四次 load 应合计完成 72 MiB。`report_status()` 中 `objects`、`free_slots`、`locked_objects`、`total_pins` 等来自同一份共享目录，两端看到的是全池状态，不能相加。`store_bytes`、`load_bytes` 是各自适配器进程完成的对象载荷字节。链路计数、带宽、实际模型 KV 命中和生成质量均需另测。超时、原生异常退出或任一侧缺回执时保留日志和失败状态。

## 迁移到真实模型

真实模型运行时，在 A、B 各自的 LMCache 服务 `--l2-adapter` 参数中传入一份插件 JSON。下面是 A 的模板，地址为示例地址，请在启动前替换。B 将 `rank_id` 改为 `1`，将 `device_nic` 换为 B 的通信地址，并填 B 容器分配的逻辑 `device_id`。两份配置的 `storage_rank` 都保留 `0`，`store_url`、`join_coord_url`、`join_group_id` 及池参数完全一致。模型实验单独使用新的协调器、run ID 和结果目录。

```json
{
  "type": "plugin",
  "module_path": "lmcache_ascend.v1.distributed.memfabric_l2_adapter",
  "class_name": "MemFabricL2Adapter",
  "shared": true,
  "adapter_params": {
    "shared_directory": true,
    "rank_id": 0,
    "device_id": 0,
    "world_size": 2,
    "storage_rank": 0,
    "pool_gb": 1,
    "slot_size_mb": 32,
    "data_op_type": "DEVICE_RDMA",
    "enable_56bits_gva": true,
    "device_nic": "tcp://192.0.2.10:28582",
    "store_url": "tcp://192.0.2.10:28580",
    "join_coord_url": "http://192.0.2.10:28581",
    "join_group_id": "replace-with-fresh-model-run-id"
  }
}
```

顶层 `shared: true` 将这个 L2 存储域标为多实例共享；`adapter_params.shared_directory` 启用插件的跨 rank 目录。池满时，由共享目录统一选择可驱逐且未锁定的对象。模板省略顶层 `eviction`，使每个 LMCache 实例的 `eviction_config` 保持关闭。

两个模型实例要生成相同的 `chunk_hash`、`model_name`、`kv_rank`、`object_group_id` 和 `cache_salt`。若选用 Python 哈希，须在两边相关解释器启动前配置同一整数 `PYTHONHASHSEED`，并统一哈希实现、实际 token 序列、前缀上下文及 chunk 边界。本用例直接构造合成 key，未执行模型端的 chunk hash 计算。模型名相同仍须核对实际权重版本、tokenizer、KV dtype、分片对应关系，以及每个 KV 组的 shape、物理布局和内存格式。

真实模型验收应只在 A 预热该前缀，随后用独立的 B 实例首次请求相同 token 前缀，控制或排除 B 的 L1 命中及 vLLM 自动前缀缓存（APC）。逐请求关联 A 的已完成 store、B 的 L2 lookup、BM load 完成字节、恢复与重算 token、输出正确性及服务真实退出。只观察到 HTTP 成功、响应相同或 TTFT 变化，仍无法定位 KV 的来源。
