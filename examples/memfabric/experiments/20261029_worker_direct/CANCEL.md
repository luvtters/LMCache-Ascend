# 客户端取消与资源收尾

此入口只向已经启动、已经播种的 C diagnostic 服务发请求，不启动/停止服务，不清缓存，不代替服务释放 pin 或 lease。

```bash
python "$MF_EXPERIMENT_ROOT/runtime/cancel_requests.py" \
  --dataset "$MF_EXPERIMENT_ROOT/bench1373/data/custom.jsonl" \
  --output-root "$MF_EXPERIMENT_ROOT/C-diagnostic01/cancel01" \
  --input-tokens 1373 --expected-external-tokens 1280 --run
```

默认仅打印计划。实际执行时，依次在收到 HTTP 响应头后延迟 0、5、20、50 ms，再关闭连接；客户端不调用流正文读取。请求使用 `stream=true`、`max_tokens=128`、`ignore_eos=true`，以免自然遇 EOS 的短回答被当作取消。

`MF_EXPERIMENT_ROOT` 与本目录其余归档入口相同，须由调用者指向已确认的个人运行目录。两个 Python 文件的输入与输出位置均由命令行提供，没有个人绝对路径。

每条请求独立保存连接动作/时间/响应头及前后原始 metrics、status、counter。必须观察到：HTTP 200 和 SSE、客户端关闭动作、服务的 abort 计数增量为 1、生成量小于 128、P 没有新增请求、实际 native 操作无新增失败、同一 worker counter 会话，以及请求 running/waiting、池保护和在途操作已知归零。排空最多等待 30 秒，要求至少一秒持续空闲；缺必需字段为未知，不记通过。若请求在到达 engine 前已被关闭而没有 abort 计数，记录未观察到服务端 abort，不能冒充已验过 engine 取消。

取消阶段不要求固定恢复 token 数或每次产生 G2L；真实完成字节和 RPC/retrieve 差值按观察保存。最后发送一个普通完整请求，检查 ORCHID、1373/1280/93/0、真实读池和保护归零。资源不明确或出现 native 错误时停止后续请求，不强行回收。

`application_body_bytes_read=0` 只描述客户端调用；HTTP 头解析器可能已缓冲正文，模型也可能已经产生 token。结果只证明真实客户端取消、服务端 abort 观察与资源收尾，不能称为 native DMA 中断、网络故障或迟到 RPC 注入。RPC/retrieve 计数仅覆盖 worker，不覆盖 scheduler。

本地 `check_cancel_requests.py` 仅检查解析、未知/失败拒绝和本机 loopback 连接关闭；不等于 NPU 或推理验收。
