# Worker 控制等待诊断

单独启动新组，`kv_connector_module_path=diagnostic_direct_connector`，类名仍为 `MemFabricDirectConnector`。沿用 `MF_DIRECT_COUNTER_ROLE` 与 `MF_DIRECT_COUNTER_PORT`（例如 29401/29402，先检查端口）。正式 ABC 的 `counted_direct_connector.py` 完全不变，不用本诊断组替代正式结果。

模块在 worker 工厂先安装原 BM/gather/scatter 计数，再延迟导入并透明包装真实 `_Control.submit` 和 `MemFabricDirectTransferContext._retrieve`。不创建新 transport、不替换 Future、不调用取消/排空接口。真实 `MessagingFuture` 无 `__slots__`、有实例字典，包装只覆盖该实例的 `result`，返回原响应、抛原异常；`wait/query/callback` 和对象身份保持不变。

`GET /counters` 除原计数外增加：

- `retrieve`：owner 线程执行 `_retrieve` 的次数、请求 chunk 数、完整恢复/未命中/失败次数及耗时。它不包含进入 owner 前的排队、scheduler 或模型执行耗时。`restored_chunks` 仅来自完整成功调用，不把失败中已搬运的部分记为完整恢复。
- `rpc.lookup/acquire/release/other`：submit 调用成本、`result` 等待成本、正常返回/异常/等待超时次数。重复读取原 Future 会增加 result 调用/等待次数，但不会重复累计该 Future 的首次结果、成功元数据或观察到的完成数。
- `submit_to_first_result_ns`：提交调用开始到首次正常返回 `result()` 的观察延迟。`submit_to_first_success_ns` 进一步要求该响应 `success=true` 且 JSON `ok=true`。两者都不是纯网络 RTT，可能包含排队或调用方延迟读取。
- `metadata_failures`：正常返回的响应明确失败，仍原样交给生产 `_Control.decode` 处理；不在诊断层主动抛新错误。
- `result_timeout_calls`：原 `result()` 等待超时，不代表 Future 完成或 DMA 结束，后续迟到结果仍可在同一对象上观察一次。Future 内部保存的 TimeoutError 是终态异常，按异常记录，与 Event.wait 超时区分。

真实 `MessagingFuture.result()` 调用 `wait()`，后者直接调用 `threading.Event.wait(timeout)`；没有固定 sleep 轮询。本诊断不添加 sleep。

RPC 等待通常嵌套在 `_retrieve` 内，不能把两者相加。当前只在 worker 进程安装，因此不包含 scheduler/API 进程的 lookup 和调度等待；TTFT 减去 BM/scatter 后的剩余不能全部归因于 RPC。

CPU 检查先显式设置 `MF_CORE_SOURCE=/path/to/LMCache`（固定 core `8bc9edc`），再执行 `python -X utf8 check_diagnostic_direct_connector.py`。从该源码 AST 提取真实 `MessagingFuture` 类执行，替代外部依赖与 context/control，不导入 Torch/Ascend。覆盖成功、重复 result、超时后迟到、元数据拒绝、保存的终态异常、submit 异常、retrieve 成功/未命中/失败及 HTTP 快照。它不是实机诊断结果。
