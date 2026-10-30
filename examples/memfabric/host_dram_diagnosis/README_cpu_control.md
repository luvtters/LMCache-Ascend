# 双物理服务器 CPU 控制面探针

本入口只使用 Python 标准库，以真实 TCP/HTTP 验证双节点元数据通信和现有 `direct_coordination` 的事件契约。它不导入 torch、torch_npu、MemFabric 或 ACL，不初始化设备，不传输 KV，不模拟 LMCache RPC 成功。

**CPU 通信通过不能证明 BM、SDMA、HCCS、HOST/DRAM 搬运或完整推理通过。当前跨机 HOST/DRAM 的 native SDMA CQE 9/10 故障不因本探针通过而解决。**

## 复制范围

仅运行服务器和客户端需要两个文件，保留以下相对目录结构；无需安装依赖：

```text
examples/memfabric/direct_coordination.py
examples/memfabric/host_dram_diagnosis/cpu_control_probe.py
```

本地检查另需同目录 `check_cpu_control_probe.py`。只能部署到已核实的个人 CPU 容器，要求无设备、无挂载、非 privileged、非 root。应由外层独立限时器停止个人 CPU 容器，避免依赖当前会话等待。容器需能够通过两机通信 IP 互访，例如已核实符合约束的 host 网络；本文件不创建容器、不修改网络或防火墙。

## 最简单原生命令

以下命令在个人 CPU 容器内、已复制的文件树根目录执行。示例使用节点 A 地址 `192.0.2.10`、节点 B 地址 `192.0.2.20`，均为文档占位地址，执行前必须替换为实际通信 IP；端口为 29635。**每次重试必须同时换两端 run-id 与结果目录**。不要把两端命令放进同一容器冒充跨机。

先在 节点A 启动监听：

```sh
python3 examples/memfabric/host_dram_diagnosis/cpu_control_probe.py --role server --bind 192.0.2.10 --port 29635 --peer-url http://192.0.2.20:29635 --run-id cpu-twohost-001 --output /tmp/cpu-twohost-001-serverA --duration 300 --run
```

在 节点B 启动监听：

```sh
python3 examples/memfabric/host_dram_diagnosis/cpu_control_probe.py --role server --bind 192.0.2.20 --port 29635 --peer-url http://192.0.2.10:29635 --run-id cpu-twohost-001 --output /tmp/cpu-twohost-001-serverB --duration 300 --run
```

两端监听均已实际启动后，在 节点A 的另一个终端运行 A→B 及 B→A 回调：

```sh
python3 examples/memfabric/host_dram_diagnosis/cpu_control_probe.py --role client --peer-url http://192.0.2.20:29635 --local-url http://192.0.2.10:29635 --run-id cpu-twohost-001 --output /tmp/cpu-twohost-001-clientA --deadline 90 --require-distinct-hosts --run
```

然后在 节点B 运行反方向的独立契约检查：

```sh
python3 examples/memfabric/host_dram_diagnosis/cpu_control_probe.py --role client --peer-url http://192.0.2.10:29635 --local-url http://192.0.2.20:29635 --run-id cpu-twohost-001 --output /tmp/cpu-twohost-001-clientB --deadline 90 --require-distinct-hosts --run
```

省略 `--run` 时仅打印 `PLAN_ONLY`，不监听、不发送 HTTP、不创建结果目录。`--duration` 上限 300 秒，`--deadline` 上限 180 秒；所有 HTTP 调用另有限时。服务器收到 SIGTERM/SIGINT 时走正常 shutdown/server_close，并保存 `STOPPED` 回执；自动时限也执行相同关闭。客户端不会远程停止服务器。确认两端客户端完成后，由执行器向**本轮已核实 PID**发送 SIGTERM，或等待 300 秒自动退出，再确认两个容器均已停止、监听端口已释放。

## 验收与证据

每个客户端正常返回 0，`result.json` 为 `PASS_CPU_HTTP_ONLY`，19 个检查均为 true（不使用 `--require-distinct-hosts` 的本地检查为 18 个）。单台客户端验收：

- 对端身份、两端未导入禁止模块；两台 Linux 内核 boot ID 必须不同。容器 hostname 不作为物理服务器区分依据；执行器还应保留 两台服务器 的实际启动位置与容器身份回执。
- 真实 `make_server` 事件首次写入 201、相同重复 200、冲突 409，随后 GET 验证原值没有被覆盖。
- 未知旧 attempt 以及与当前 attempt 共享前缀的未知 attempt，GET 为 404、POST 为 400。拒绝状态与当前生产模块的 HTTP 契约一致。
- 真实 `publish`/`wait_event` 成功完成，另两个隔离 attempt 分别触发失败事件 RuntimeError 与不存在事件 TimeoutError；预期异常保留原消息和 traceback。
- 对端调用真实 `publish` 回调本机协调服务器，并从本机真实 `wait_event` 读回相同完成事件，证明反方向 HTTP 实际可达。回调只允许服务器启动时明确配置的 peer URL 与当前 run-id。
- 小于 64 KiB 的元数据在对端回环后逐字段完整比较，不进行吞吐基准。

每个 case 的 attempt 为 `<run-id>-immutable/complete/failure/timeout/callback/echo`，服务端各自使用真实 `make_server` 的独立闭包、目录与锁，避免失败事件污染其他 case。这里的额外 HTTP 路由只存在于诊断程序，不改生产 `direct_coordination.py`。

每端 `http.jsonl` 保存实际 HTTP 请求与响应的状态、headers、原始 body 的 base64/UTF-8、本机开始/结束单调时序和墙钟。接收端保存实际收到的完整 headers 与请求行；发送端另存 URL、准备的 headers、响应和传输异常。每份记录有本机 hostname/PID，**两机 monotonic 时间不能直接相减**。`result.json` 保存检查明细、身份与原始异常。不要只凭终端 PASS 字样验收；保留两端服务端和客户端四份目录。

意外 HTTP 状态、内容不同、相同 boot ID、缺失回调、连接拒绝或超时均应退出 1 并保存 `FAIL`。已有非空输出目录直接拒绝执行；重复使用同一 run-id 的原服务器会使“首次 201”失败，不自动抹掉历史事件。

## 本地 CPU 检查

```sh
python3 examples/memfabric/host_dram_diagnosis/check_cpu_control_probe.py
```

两台本地 loopback 监听分别完成 18 项契约检查；负例验证旧 attempt 重用、同物理机拒绝、对端不可达，以及自动正常退出和端口释放。正常输出 `PASS_LOCAL_CPU_ONLY`。这只是本地 CPU 验证，不能代替 两台服务器 真双机执行回执。
