# BM / Mooncake TE 同口径 CPU DRAM 测试

2026-10-27 新增 Ascend Direct 准备代码、TE SHM 与资源对齐参数，最新入口见 [高速与 SHM 说明](README_transfer_bench_fast.md)。下文保留最初 TCP 准备阶段说明，历史环境/验收边界以对应日期回执为准；当前能力以新说明和审阅包为准。

本入口独立于 LMCache，不加载 adapter、模型或推理服务。TE 路径只使用 CPU DRAM 和明确指定的 classic TCP；BM 路径复用兼容提交 `525a710` 的 `BMSession.copy()`，需要设备，当前仅准备脚本。

## 已准备的环境

- 容器：`137:memfabric-d00896497-qwen3`。
- 私有目录：`/workspace/d00896497/te_cpu_20261026`。
- 独立解释器：上述目录的 `venv/bin/python`，不继承系统 site-packages。
- CPU 源码：Mooncake `f751cff2b67668ae97f1ff7e45fb2154fdbff64a`；CUDA、Ascend、Ascend Direct、MemFabric、TENT、Store、EP 关闭。
- 通用 PyPI ARM64 wheel `0.3.13.post1` 链接 CUDA，不能直接用于本容器；后续以 CPU 源码产物和构建回执为准，不把 wheel 的版本标签当成源码产物版本。
- 构建依赖解压在本目录 `sysroot`，不安装到系统。CPU 环境需要自己的库路径，不在原 LMCache 终端 source 它。

实际验收结果、完整命令及证据以 [CPU 环境验收记录](../../../research/2026-10-26_TE_CPU环境与同口径压测入口.md) 为准；构建中的状态不得当作已通过。

## 终端入口

下面前两项仅使用 CPU，不启动 BM 或推理。命令在原个人容器内执行：

```bash
TE_CPU_ROOT=/workspace/d00896497/te_cpu_20261026
source "$TE_CPU_ROOT/te_cpu_env.sh"
```

三档正确性：

```bash
"$TE_CPU_ROOT/venv/bin/python" "$TE_CPU_ROOT/transfer_bench.py" \
  --backend te --protocol tcp --sizes-mib 1,18,64 --check-only \
  --output "$TE_CPU_ROOT/te_correctness.json"
```

三档正式 TE 采样，仍仅 CPU；不能把这项结果叫作 BM / TE 已完成对比：

```bash
"$TE_CPU_ROOT/venv/bin/python" "$TE_CPU_ROOT/transfer_bench.py" \
  --backend te --protocol tcp --sizes-mib 1,18,64 \
  --warmup 20 --iterations 100 --repeats 3 \
  --output "$TE_CPU_ROOT/te_benchmark.json"
```

将来取得设备窗口后，**在新终端恢复原 Ascend / 项目 venv 环境**，才可运行下面的 BM 命令。它会初始化并使用两张 NPU；本次没有执行。`8,9` 只是允许范围内的示例，运行前重新确认空闲和逻辑映射。

```bash
TE_CPU_ROOT=/workspace/d00896497/te_cpu_20261026
/workspace/d00896497/venv/bin/python "$TE_CPU_ROOT/transfer_bench.py" \
  --backend bm --allow-npu --devices 8,9 \
  --bm-module "$TE_CPU_ROOT/compat_525a710_memfabric_bm.py" \
  --store-url tcp://127.0.0.1:19867 --sizes-mib 1,18,64 \
  --warmup 20 --iterations 100 --repeats 3 \
  --output "$TE_CPU_ROOT/bm_benchmark.json"
```

BM 缺少 `--allow-npu`、两个 8–15 内的设备或兼容源码路径时，会在导入原生模块前拒绝运行。TE 命令不接受其他 transport，不会把未支持的协议自动改成 TCP。

## 如何判定通过

每档先做三种校验：发起端写入后目标进程核对；覆盖本地源后从目标读回；目标独立预置另一种模式后发起端读取。进程间控制通道只传地址、命令和回执，不传 payload。BM 的池地址经 BM copy 访问，不把 GVA 当作 CPU 指针解引用。

只有数据检查通过且两个子进程正常退出，JSON 才会写 `status=PASS`。失败返回非零；未确认完成的操作不会被算入成功样本。失败时由父进程终止本次测试子进程，不正常释放仍可能在途的缓冲；不操作其他服务。

正常计时将完整的同步完成 API 包在 `perf_counter_ns()` 内。分配、首次触页、注册、连接准备、预热、校验、统计和清理在该计时外。`samples_ns` 保留逐次样本；`measurement_wall_ns` 为测量循环墙钟；`effective_GB_s` 为成功 payload bytes / 成功 API 总耗时，采用十进制 GB/s。3 轮 × 100 次各自保存，不丢弃异常高值。

## 结果能比较到什么程度

两种入口使用同一统计与校验循环、同一 POSIX 4 KiB 对齐 CPU buffer、相同三档字节数、一次提交线程、batch=1、outstanding=1。native 后台线程使用各后端默认值，不能把它描述为整个进程只有一个线程。

仍有必须显式报告的差异：TE 显式注册本地 DRAM，BM 保留兼容版的无显式预注册调用；TE 对端由 POSIX allocator 分配，BM 对端由池分配；TE 本轮实际协议为本机 TCP loopback，BM 为 SDMA；NUMA 位置尚未验证。结果属于路径组合比较，不能直接归因于相同链路下的纯传输实现。

缓冲在固定地址重复使用，属于热缓冲测试。输出包含两端亲和、setup 时间、模块加载位置及 native state。setup 中的控制交换等待不等于纯建链时间；真正首次传输可能包含延迟建链，位于正确性轮和预热中。

本入口不替代 LMCache 服务 A/B，也不证明 NPU↔池直传、HCCS 跨机收益或真实 KV 恢复。正式推理仍须采集池读写、恢复/重算 tokens、缓存状态、TTFT 和吞吐。
