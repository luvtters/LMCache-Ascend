# 85/86：SDMA 跨机 DDR 池化测试

本分支 `feature/memfabric-sdma-85-86-20261111` 从 L2 节点 `6a07d75` 分出，开放共享目录的 SDMA 路径。两机各贡献 8 GiB HOST DDR，总池容量 16 GiB；每机原有 LMCache L1 容量另外保留。没有启用 HBM 池，也没有接入尚未完成的 PLUGIN L1。

客户两机组成一个 TP16/PP2 Prefill：85 是 PP0，86 是 PP1，41/37 层、MTP1，两个本地 LMCache 服务为 `10.48.96.85:5565`、`10.48.96.86:5565`。共享 DDR 容量不意味着 PP0 能复用 PP1 的 KV 内容。原生 PP core 的 stage namespace 保留分片身份。P→D 的 Mooncake/MemFabric 配置继续由原启动文件管理。

本地只做 CPU 检查。**85/86 的跨机 SDMA 映射、内容搬运、真实模型恢复及性能尚未验证。** 本分支不会自动切换到 RDMA。SDMA 保留底层 `gva_to_va(..., LOCAL_DEVICE)` 就绪检查；映射超时即未通过，不能删除这个检查继续跑。

## 1. 两机拉到独立目录，保留现有 PP 安装

在各自运行 LMCache 的个人容器内操作，使用原来的系统 Python 和环境。不要在本分支执行 `pip install -e .`，也不要把仓库根目录加到 PYTHONPATH，否则会覆盖正在使用的 PP 插件。

```bash
mkdir -p /opt/ks-luv-pp
git clone --single-branch --branch feature/memfabric-sdma-85-86-20261111 \
  https://github.com/luvtters/LMCache-Ascend.git /opt/ks-luv-pp/memfabric-sdma
export MF_REPO=/opt/ks-luv-pp/memfabric-sdma
export PYTHONPATH="$MF_REPO/customer/sdma${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
python "$MF_REPO/customer/sdma/check.py"
```

预期 `IMPORTS_OK`，core 与 PP plugin 路径指向原安装，SDMA L2 路径指向新目录。这只是导入检查，不初始化 BM。参考已有 native PP 配对：core `89d93a2`，Ascend `ac1e4c7`／`b06a829`；如现场已有后续修复，保留现场版本。旧 `AscendPPStageLocalConnector` 分支不能直接共用目录，先确认采用包含 stage namespace 的原生 PP 路径。

现有环境需有 `memfabric-hybrid>=1.2.1` 及匹配的 CANN/驱动；不要为此升级现有框架、驱动或公共 SDK。SDK、NPU 设备节点及跨机 SDMA 互联须在客户容器实际可见。检查 28580/28581/28582 未被占用，85/86 互通；`device85/device86` 是容器内逻辑 NPU 编号，默认 0，现场可改。这里不登录客户机器，不启动已有服务。

## 2. 先做跨机池容量与内容检查

两机各自的每个测试终端都先执行上节的四条 `export`。两机手动填同一个全新 RUN_ID，不能分别生成不同时间。以下示例的 ID 仅供首次使用；重试更换 ID 及结果目录。

```bash
export RUN_ID=sdma8586-capacity-01
export MF_RUN=/opt/ks-luv-pp/$RUN_ID
python "$MF_REPO/customer/sdma/configure.py" --run-id "$RUN_ID" --output "$MF_RUN"
```

85 的单独终端启动元数据协调器（只保存目录和状态，KV 不走 HTTP）：

```bash
python -u "$MF_REPO/examples/memfabric/pool_coordination.py" \
  --bind 10.48.96.85 --port 28581 --directory "$MF_RUN/events" \
  --run-id "$RUN_ID" --shared-l2 --storage-ranks 0 1 --pool-gb 8 --slot-size-mb 32
```

在 85 设置 `RANK=0; LOCAL_IP=10.48.96.85`；86 设置 `RANK=1; LOCAL_IP=10.48.96.86`。然后在两端运行相同命令。先不加 `--allow-npu` 可以生成计划；真正执行才加该选项，两端需同时开始，使用已确认可用的设备。

```bash
set -o pipefail
python -u "$MF_REPO/examples/memfabric/l2_cross_instance_gate.py" \
  --rank "$RANK" --device 0 --storage-ranks 0 1 --pool-gb 8 --slot-size-mb 32 \
  --store-url tcp://10.48.96.85:28580 --device-nic "tcp://$LOCAL_IP:28582" \
  --join-coord-url http://10.48.96.85:28581 --join-group-id "$RUN_ID" \
  --data-op-type SDMA --adapter-module memfabric_sdma.memfabric_l2_adapter \
  --object-count 384 --payload-bytes 25165824 --require-overflow \
  --worker-timeout 1800 --allow-npu --output "$MF_RUN/rank-$RANK.json" \
  2>&1 | tee "$MF_RUN/rank-$RANK.log"
rc=${PIPESTATUS[0]}; printf 'exit=%s\n' "$rc"
```

85 写入 384×24 MiB＝9 GiB，超过单 owner 的 8 GiB；86 从两个 owner 读回全部内容。通过需两端 `TWO_ADAPTER_CONTENT_AND_PROCESS_EXIT_PASSED`、launcher 退出 0，`two-owner-payload` 的 `completed_bytes=9663676416`、两个 owner 的完成字节均大于 0、全量内容一致、无淘汰，且删除/解锁与两个 worker 实际退出通过。单机映射或单 owner 远程读不算双 owner 聚合。

本检查有 1800 秒 worker 上限；仍应按客户可用资源窗口控制启动与停止。失败保存两端 JSON/日志，结束本次两个 gate 进程并确认设备释放，再停止协调器；不要保留半个组重试。正常情况下等待两端 launcher 退出后，在协调器终端 Ctrl+C。容量关结束后再启动模型关，不同时复用这组端口、目录或 RUN_ID。

## 3. 接回现有 LMCache On + PP

使用新的共同 ID，如 `sdma8586-model-01`，重新执行 configure，并在 85 启动同参数的新协调器。两端缓存终端设置相同的 PYTHONHASHSEED 和上面的独立 PYTHONPATH；两端模型终端也设置相同 PYTHONHASHSEED，沿用原有 PP 环境变量。

85 选 `MF_CONFIG="$MF_RUN/85.json"`，86 选 `MF_CONFIG="$MF_RUN/86.json"`。在各自现有 `cache-foreground.sh` 的原生 `lmcache server ...` 命令尾部追加：

```bash
  --l2-adapter "$(cat "$MF_CONFIG")"
```

其余 L1 大小、chunk-size、端口和 PP 参数照原配置。保留原启动脚本副本，先结束自己原来的缓存服务，再用原生 `lmcache server` 命令以前台启动新的两端服务；不要在旧服务占用 5565 时并起第二份。协调器先启动，两端缓存随后尽快启动。两端均需看到 `MEMFABRIC_READY protocol=SDMA storage_rank=(0, 1) capacity=17179869184` 和各自 `MEMFABRIC_SHARED_DIRECTORY rank=0/1`，此时才继续用原 `bash /opt/ks-luv-pp/startpp.sh` 或实际 `run_dp_template.sh` 启动模型。不要把 `AscendStoreConnector`、`AscendPPStageLocalConnector` 或已有 MultiConnector 配置临时换成别的模式；现场原生 PP 配置不同于参考版本时先对齐。

在 85 检查 `/health` HTTP 200 和 `/v1/models`；86 是 headless，没有独立 8000 ready 要求。这只证明服务就绪。正式验证用同一原始请求先写池，再在保留两个 cache/BM 进程的条件下排除模型 APC 和缓存 L1 命中，核对 `MEMFABRIC_STORE`、`MEMFABRIC_LOAD`、实际恢复 token、完整输出和错误日志。不能仅凭 Retrieved 或热请求加速证明来自 SDMA 池；如尚无可靠的 L1 排除步骤，就把模型读池保留为待验证。

回退：停止本轮模型、两端缓存与协调器，移除 `--l2-adapter` 和本轮 PYTHONPATH，使用保存的原缓存命令和原模型脚本重启。共享目录是易失缓存，不支持保留一端在线时重建另一端。此分支不改 PP core、模型启动文件、驱动或 SDK。
