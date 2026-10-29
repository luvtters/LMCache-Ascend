# 实验专用搬运计数

`counted_direct_connector.py` 继承生产 `MemFabricDirectConnector`，只在 worker 工厂安装计数包装；不修改仓库生产代码。把本目录加到容器 `PYTHONPATH`，将 `kv_connector_module_path` 设为 `counted_direct_connector`，保留类名 `MemFabricDirectConnector`。

P、D 分别设置 `MF_DIRECT_COUNTER_ROLE=P` / `D`，并设置不同的 `MF_DIRECT_COUNTER_PORT`（例如 29401、29402，启动前检查端口）。仅监听 `127.0.0.1`，`GET /counters` 返回累计内存快照；不提供重置接口。窗口前后由压测脚本读取并求差，必须确认 `pid` 与 `session_id` 未变，不能跨重启相减。

- `operations.L2G/G2L.completed_bytes`：实际传给 BM、同步成功返回的字节数，未从 token 数估算。
- `completed_calls/completed_ns`：成功调用次数、其同步调用耗时之和。不是硬件 DMA profiler，也不包含整个请求耗时。
- `failed_calls/failed_ns`：失败调用及其耗时，不增加完成字节；失败不代表 DMA 已结束。
- `started_calls/in_flight`：开始次数和仍未返回次数；未返回时不会计为完成。
- `gather/scatter` 只统计调用与耗时，`completed_bytes=0`，避免把跳过部分前缀的 scatter 误称为完整 buffer 搬运。
- `cpu_copy_attempts`：CPU BM 路径尝试次数。该路径在本直传实验中报错，避免静默中转。

每次搬运仅更新内存计数；没有逐次文件或网络写入。HTTP 线程只读取元数据，不调用 BM/NPU。仍有少量 Python 锁和计时开销，不能称为零开销。原严格 trace gate 与正式计时分开运行，不要同时加载 `traced_direct_connector`。

服务器线程随进程退出由 `atexit` 关闭；`close_counter_server()` 只停止 HTTP 辅助线程，不会关闭或排空 BM/传输 context。先在压测结束时保存快照，再做模型/池的正常关闭验收。

本地验证：`python -X utf8 check_counted_direct_connector.py`，使用完全替代的 connector、BM 与 NPU transfer 类型，不导入 Torch/Ascend，不代表真实设备验通。设备执行前仍需用真实已知请求确认成功计数与原 gate 匹配。
