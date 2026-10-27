# TE Ascend Direct / 单机 SHM 对照入口

本入口用于独立搬运诊断，不加载 LMCache adapter、模型或推理服务。2026-10-28 已在 A3 完成 BM、TE relay、TE RoCE 和 SHM 的三档正确性及正式采样，见 [结果与证明边界](results/20261028_478/README_zh.md)。命令示例仍须按当次设备授权和实际路径调整。

## 当前能力

| 入口 | 分配/搬运 | 验证边界 |
| --- | --- | --- |
| `--backend te --protocol tcp` | POSIX 普通页；classic TE，用 `MC_FORCE_TCP=1` 约束路径 | 本机 CPU 正确性及短采样已完成 |
| `--backend te --protocol shm` | TE `allocateSharedMemory`；C ABI 小桥接只安装 SHM，实际 CPU memcpy | 本机双进程正确性及短采样已完成 |
| `--backend te --protocol ascend-roce` | ACL pinned Host；classic Ascend Direct，要求 NPU RoCE | A3 三档通过；日志确认 ROCE transport |
| `--backend te --protocol ascend-hccs-relay` | ACL pinned Host；ADXL BufferPool 中转 | LocalCommRes 1.2 三档通过；P2P/HCCS 拓扑有证据，物理链路流量未计量 |
| `--backend bm` | 保留兼容 BMSession / SDMA；本地 buffer 可选 POSIX 或 ACL Host | POSIX 三档通过；ACL Host 原 H2G 返回方向不匹配，不能作为已通过配置 |

固定 Mooncake revision 为 `f751cff2b67668ae97f1ff7e45fb2154fdbff64a`。不启用 TENT / MemFabric transport，不将 HIXL Fabric Memory 当作已接通；后者有不同分配和 Store-init 入口。

## 本机 CPU 验证环境

实际在 Windows 本机的 WSL `Ubuntu-22.04`、普通用户 `luv_l` 下准备，私有目录 `/home/luv_l/memfabric_te_fast_20261027`。无 NPU 设备，未 SSH 使用服务器。CMake、头文件和库只放私有 venv/sysroot，未安装系统包或升级现有环境。

下面命令在该 WSL 内执行；`0,1` 是本机已确认可用 CPU，仅是本机回执的资源配置。

```bash
TE_LOCAL_ROOT=/home/luv_l/memfabric_te_fast_20261027
export LD_LIBRARY_PATH="$TE_LOCAL_ROOT/build-cpu/mooncake-transfer-engine/src:$TE_LOCAL_ROOT/build-cpu/mooncake-common/src:$TE_LOCAL_ROOT/build-cpu/mooncake-common:$TE_LOCAL_ROOT/sysroot/usr/lib/x86_64-linux-gnu"
export PYTHONPATH="$TE_LOCAL_ROOT/python"
python3 "$TE_LOCAL_ROOT/harness/transfer_bench.py" \
  --backend te --protocol shm \
  --shm-library "$TE_LOCAL_ROOT/libte_shm_bridge.so" \
  --cpu-affinity 0,1 --sizes-mib 1,18,64 --check-only \
  --output "$TE_LOCAL_ROOT/shm_correctness.json"
```

短采样用 `--warmup 2 --iterations 5 --repeats 2` 替换 `--check-only`；正式三档仍用 `20 / 100 / 3`。TCP 使用 `--protocol tcp` 并去掉 `--shm-library`。CPU 模块路径必须对应独立 CPU 构建，不能加载有 CUDA/NPU 依赖的通用 wheel。

## 构建 SHM 桥接

对已有、固定版本的 CPU-only TE 构建执行：

```bash
export MOONCAKE_SOURCE=/home/luv_l/memfabric_te_fast_20261027/Mooncake-f751cff2b67668ae97f1ff7e45fb2154fdbff64a
export MOONCAKE_BUILD=/home/luv_l/memfabric_te_fast_20261027/build-cpu
export MOONCAKE_SYSROOT=/home/luv_l/memfabric_te_fast_20261027/sysroot
bash build_te_shm_bridge.sh /home/luv_l/memfabric_te_fast_20261027/libte_shm_bridge.so
```

源码路径以构建回执为准；脚本只链接已有 CPU TE，不下载、不安装、不导入原生模块。桥接必须与所选 TE 公共头和构建模式匹配；脚本拒绝开启的加速器/TENT/multi-protocol 配置。

桥接暴露 create/allocate/register/connect/copy/unregister/free/destroy，Python 经 `te_shm.py` 调用。共享缓冲由 TE 自己分配并发布对象名，避免普通 POSIX 指针 register 返回成功但没有 SHM 元数据。只安装 SHM，不会自动退回 TCP。控制管道仍只传地址和回执，不传载荷。

## 准备 Ascend Direct

目标机需已有匹配 CANN 9.1 和 ADXL/HIXL，另建源码构建目录和解释器，保留 CPU TE 与 LMCache 环境。`prepare_te_ascend_direct.sh` 默认只打印构建计划，不初始化设备：

```bash
export MOONCAKE_SOURCE=/workspace/d00896497/te_source_fixed
export MOONCAKE_ASCEND_BUILD=/workspace/d00896497/te_ascend_direct_build
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
bash prepare_te_ascend_direct.sh --print-plan
```

`--configure-only` / `--build` 分别配置或配置并构建，依赖须已经准备；可以追加本环境的 CMake 依赖路径参数。脚本不安装、不运行 TE、不覆盖既有 CPU build。固定 classic 目标链接 `llm_datadist`，其内部使用 `libcann_hixl`；不能将只有 CPU-only 的模块当作支持 Ascend Direct。

下列命令在当前机器也可安全检查计划，**`--dry-run` 不导入原生模块、不分配缓冲、不创建进程或设备 context**：

```bash
python transfer_bench.py --backend te --protocol ascend-hccs-relay \
  --allow-npu --devices 8,10 --cpu-affinity 96-119 --numa-node 4 \
  --te-thread-pool-size 1 --ascend-buffer-pool 4:8 --ascend-local-comm-res 1.2 \
  --timeout 90 --sizes-mib 1,18,64 --dry-run
```

未来有明确设备窗口后，才能去掉 `--dry-run`；先用 `--check-only` 验证数据，另加 `--output` 保存回执。`--devices` 必须是两个不同的 8–15 物理设备，运行前按当次授权重新盘点。设备路径会在 native import 前拒绝缺失许可，进程内按可见列表映射到逻辑 rank 0/1。

RoCE 改为 `--protocol ascend-roce`，使用 `HCCL_INTRA_ROCE_ENABLE=1`、BufferPool=0:0；HCCS relay 明确使用 BufferPool，并保持 `HCCL_INTRA_ROCE_ENABLE` unset，固定源码只接受整数 `1` 为启用值，设置 `0` 会触发无效值告警。两者均禁用 TENT、Fabric Memory 和 SHM 自动路由，移除继承的 `ASCEND_GLOBAL_RESOURCE_CONFIG` 等覆盖项。同步模式保持 `ASCEND_USE_ASYNC_TRANSFER` unset，不能写 `0` 来关闭。

当前 8 环境 CANN 9.1 的 HCCS relay / BufferPool=4:8 需要显式加 `--ascend-local-comm-res 1.2`。该选项只允许 Ascend Direct 使用非 `auto` 值；可选 `auto`、`1.2`、`1.3`，默认 `auto` 会清除继承的 `ASCEND_LOCAL_COMM_RES` 并保持 unset，由 TE 选择默认值。显式版本写成 ADXL 所需的 JSON，结果配置的 `ascend_local_comm_res_requested` 记录请求值，非 Ascend 记录 null；请求值不作为实际引擎或链路验证证据。

固定 classic TE 的 copy `transport_hint` 会被忽略；实际选择由初始化协议与环境控制。这里初始化明确请求 `ascend`，对应独立构建只安装 Ascend transport；TCP 则以 `MC_FORCE_TCP=1` 控制，SHM 桥接只安装 SHM。copy 统一使用 session、本地地址、对端地址、字节数四个必需参数，兼容只暴露四参的绑定与提供可选 hint 的 CPU 绑定；不通过捕获 `TypeError` 重试，以免掩盖原生错误或重复搬运。真实 HCCS/RoCE 必须再核对 HIXL 日志/计数器，JSON 保留 `NOT_VERIFIED`。

`prepare_te_ascend_direct.sh` 只构建，不安装 Python 包。将 engine 放入独立解释器的 `site-packages/mooncake/` 时，必须同时保留该目录的 `__init__.py`；迁移产物也须包含此文件。否则使用 `--system-site-packages` 的解释器可能优先加载镜像自带的完整 `mooncake` 包。结果 JSON 的 `initiator_native_state` / `target_native_state` 中，`mooncake_engine_realpath` 记录实际已加载模块的真实路径，核对其指向私有构建；该记录不额外导入模块。

## 已统一与剩余差异

- 同一 Python 测量/校验循环、payload、轮数、一次提交线程、batch=1、outstanding=1、完整同步 API 计时；预热/分配/注册/建链/检查在搬运计时外。
- `--cpu-affinity` 对两端生效；`--numa-node` 在分配前请求 Linux MPOL_BIND，失败直接终止，不假装应用成功。JSON 保存实际 affinity、允许节点、包含缓冲的 NUMA 映射、线程名称/数量和地址对齐余数。
- Ascend 使用 `aclrtMallocHost`；BM 可选 `--allocator acl-host` 对齐本地内存种类，但会触发设备初始化，仍需要 `--allow-npu`。默认 BM/TCP 保持 POSIX，`--alignment` 可调；SHM 必须使用 TE 自有共享分配。
- 无法直接对齐：BM 池与 TE 目标的 allocator/所有权、BM 内部注册、池的物理 NUMA、SHM 与 DMA 的机制、HIXL relay 中转、native 后台线程职责和绑定层。资源快照不会把这些差异消除。
- Ascend executor 请求线程数限制为固定版本支持的 1–16，避免超范围静默回到默认；HCCS BufferPool 分支固定使用 1 个 executor 线程，仍以资源快照为准，不把请求值当作实测。
- NUMA 映射统计的是包含 buffer 的 VMA，不一定等于该 buffer 的精确页数。固定热缓冲的短采样不代表持续流式带宽。

## 汇总与通过判据

```bash
python summarize_transfer_bench.py --bm bm_new.json --te te_new.json --output comparison.md
```

只接受数据校验完整、双 worker 退出 0、成功计时和字节一致的结果。新 schema 额外拒绝混合不同机器、平台、请求/实际 CPU affinity、NUMA 策略及测试口径；旧 schema 结果仍可成对读取，但不与新 schema 混比。报告逐项列出剩余差异，Ascend 路由未核实会原样显示。

未知完成/失败不释放可能在途的缓冲；失败子进程由父进程终止。SHM 桥接确认 COMPLETED 和字节数才返回成功；首次共享映射在正确性/预热中建立。CPU 验证不替代 Ascend 高速验收、BM 新内存模式或 LMCache 完整服务 A/B。
