# 兼容 L2 跨机角色入口

冻结 Ascend `525a710` / core `6913259`，不修改生产 adapter。此目录仅提供验收入口；CPU 检查不能证明跨机 BM、模型或正常退出已通过。

所有入口接收 `--config /个人目录/run.json`。`run_root/run_id` 对应全新 attempt，driver 与模型组必须使用不同 run_id/BM 端口。远端 member 的生命周期由外部控制器记录，本目录不连接服务器、不抢占设备。

```json
{
  "run_root": "/workspace/d00896497/crosshost_20261030/runs",
  "run_id": "L2-F-driver-attempt01",
  "core_path": "/workspace/d00896497/crosshost_20261030/src/core_compat",
  "ascend_path": "/workspace/d00896497/crosshost_20261030/src/ascend_l2",
  "python": "/workspace/d00896497/venv/bin/python",
  "communication_ip": "192.168.0.223",
  "store_port": 28580,
  "pool_gb": 1,
  "slot_size_mb": 32,
  "driver_devices": [8],
  "driver_bm_device": 0,
  "local_devices": {"p": [8], "d": [10], "service": [8, 10, 12]},
  "service_bm_device": 2,
  "tp_size": 1,
  "model_path": "/workspace/d00896497/models/Qwen3-8B",
  "ports": {"rpc": 25555, "http": 28080, "metrics": 29090, "p": 28001, "d": 28002,
            "proxy": 28000, "telemetry": 25768, "gate": 25769}
}
```

如需环境脚本，模型 supervisor 的 JSON 可加 `env_file`；driver 由外部控制器先加载该脚本再运行。不要在配置里写个人代理凭据。

无模型 rank0：`python -u l2_driver.py --config run.json`。远端同时启动冻结版 `examples/memfabric/pool_member.py --device 0 --rank 1 --world-size 2 --pool-gb 1 --store-url tcp://192.168.0.223:28580`。两端紧邻发起，不要等 member READY 后才启动 rank0。driver 通过真实 adapter/store/lookup/load/unlock/delete；18 MiB 全内容比对，同时留存 `expected.pt`、`actual.pt`、差异、原始 status、STORE/LOAD 与 wait/close 日志。外部控制器必须真实 wait 两端，脚本内功能 PASS 不等于生命周期通过。

模型计划：`python l2_model.py --config model-run.json --plan-only`。实际执行去掉 `--plan-only`，外部控制器同时启动同一组远端 member。supervisor 前台启动 service、P、D、原 proxy、严格 gate，最后逐角色停止并真实 wait；`processes.json` 原样记录返回码和发送信号，不把 SIGABRT/SIGKILL 改为正常退出。无自动重启。`l2_model.py --role service` 也是独立角色 CLI，但通常由 supervisor 调用。

严格 gate 收齐所有 TP P 完成事件，等 10×TP 对象、180 MiB STORE 与自动排空 L1 后才放行 D。检查 1373 prompt、1280 external、93 compute、0 prefix、ORCHID，以及 D 的每条真实 prefetch 原文零 L1、L2 总数等于 10×TP。冻结 MP 的一次 PrefetchTaskSpec 已包含所有 kv rank，因此期望单条聚合预取记录。TP1 的 18 MiB/对象及 TP2/4/8 的动态分片计算均须用本轮原件验证，不是已验证结论。TP>1 需配置足够且不重叠的 P/D 设备，并让 service 看见所有 P/D 和独立 BM 设备；资源不足应停止在此，不扩大授权。

反向独立新 attempt：将 communication_ip 改为新 rank0 通信 IP，模型、设备、已许可目录及端口按新主机配置；外部 member 交换到原主机。不要复用旧 BM 组或就绪文件。

CPU 检查：`python l2_cpu_checks.py`。仅验证外部配置/命令规划/TP 几何和来源审计能拒绝混合、缺失、额外记录；没有初始化 NPU 或模拟 BM 正确性。
