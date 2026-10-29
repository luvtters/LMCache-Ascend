# 固定 BM 组成员离开后的元数据拒绝探针

`peerleft_probe.py` 只在已经运行的 service 上调用真实 LMCache 控制 RPC；不创建 worker、BM session，不注册 KV，不触发 DMA。默认 `PLAN_ONLY` 仅依赖标准库；`--run` 才导入真实 `RequestClientFactory` 和 `ServerModuleCallRequest`。

固定 core 的 RequestClient 依赖会被动导入 **CPU Torch**，不能宣称零 Torch。脚本在导入前设置本进程 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，记录模块与映射；发现 Torch-NPU、MemFabric 或指定设备库时，在发送探针前拒绝继续。脚本不调用任何设备初始化 API，也不改 service/宿主机设置。

使用时先确认 capacity P/D 已正常退出、service 仍存活。容器内的原生命令形态：

```bash
source "$MF_EXPERIMENT_ROOT/env.sh"
PYTHONPATH="$MF_EXPERIMENT_ROOT/runtime:$MF_EXPERIMENT_ROOT/src/LMCache${PYTHONPATH:+:$PYTHONPATH}" \
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
"$MF_EXPERIMENT_PYTHON" "$MF_EXPERIMENT_ROOT/runtime/peerleft_probe.py" \
  --server-url tcp://127.0.0.1:25595 \
  --output-root "$MF_EXPERIMENT_ROOT/capacity01/peerleft-probe01" --run
```

`MF_EXPERIMENT_ROOT`、`MF_EXPERIMENT_PYTHON` 沿用归档入口约定，分别指向已确认的个人运行目录和对应 venv 解释器。将输出目录改为本次真实 case 下的新目录；已有目录拒绝覆盖。删去 `--run` 可查看计划，不连接服务器。

探针先调用 `memfabric_direct.status`，要求同一实例的 `ready=false`、`last_error` 含固定组 peer left 原因，再调用：

1. `lookup`：空 keys、不带 request_id，不创建查询 pin。
2. `prepare_store`：合法 size=1、当前 epoch，但使用从未注册的随机 member_id。即使服务意外变为 ready，也应在 owner 校验拒绝，不可能创建写预约。

最后再次 status，要求 epoch 一致，objects/free_slots/pin/pending/quarantine 都实际观察且不变。两次操作均须收到模块层 `ok:false` 和固定组失效错误。外层 `response.success=True` 仅表示调用 handler 成功，不等于池操作成功。

所有原始 payload、外层状态、返回体、模块审计和 client.close 结果保存到 `receipt.json`。真实超时记为 `TIMEOUT_UNKNOWN`，保留原 Future 引用直到关闭；不取消、不重试，不把迟到响应追认为通过。关闭客户端也不等于未知 RPC 没有执行。

通过只证明 **已经空闲的失效组拒绝查询与新写预约**，不证明 DMA 故障、未知在途资源回收、自动重建或旧 generation 校验。generation 校验位于 `_available()` 之后，本探针不会走到那里。
