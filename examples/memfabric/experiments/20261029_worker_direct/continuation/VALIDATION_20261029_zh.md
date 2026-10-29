# Worker 直传：2026-10-29 精选验证记录

本文件记录单机 Qwen3-8B、TP1、1P1D 的重复观测。运行源码固定为 LMCache `8bc9edc`、LMCache-Ascend `7e5d40a`；兼容基线 `525a710` 未修改。

原始服务日志、HTTP 查询正文、原生压测日志与完整逐请求结果保存在私有归档中。本文件是去标识后的精选记录，**不替代原件，也不以编排脚本或质量门的 PASS 打印作为验证依据**。没有公开服务器、个人目录、进程身份或设备地址。

## 对照条件与取样范围

| 组别 | D 使用的路径 |
|---|---|
| A | 保留 LMCache，无可复用 KV，全部输入本地重算 |
| B | 兼容 MemFabric L2，池 → 临时 CPU L1 → NPU |
| C | worker 直传，池 → NPU scratch → 实际 KV 页 |

每窗32条计时请求，输入1373 token，输出8 token，并发1；三组 D 均为 consumer，关闭本地 prefix caching。B/C 采用相同四前缀工作集及有效池配置。P 播种、先验检查与预热不计入 D-only 正式计时。

只使用私有快照04中第二批 `attempt2` 的完整九窗：三轮顺序为 ABC、BCA、CAB，每个窗口来自独立部署。没有挑最优值，也没有把第一批监控证据不完整的样本拼入本集合。

## 原生指标逐轮照录

以下数值直接照录每窗原生 `vllm bench serve` 的 `benchmark.log`，保留其两位小数。对应原生 `benchmark.json` 保留完整精度和逐请求字段；这里不求三轮平均。

| 执行顺序 | 组别 / 轮次 | Mean TTFT（ms） | Mean TPOT（ms） | Output token throughput（token/s） |
|---|---|---:|---:|---:|
| 1 | A-r1 | 152.46 | 38.49 | 18.95 |
| 2 | B-r1 | 136.88 | 43.15 | 18.22 |
| 3 | C-r1 | 124.51 | 40.81 | 19.49 |
| 4 | B-r2 | 133.98 | 41.84 | 18.73 |
| 5 | C-r2 | 121.94 | 39.84 | 19.95 |
| 6 | A-r2 | 149.88 | 37.97 | 19.24 |
| 7 | C-r3 | 124.88 | 40.64 | 19.53 |
| 8 | A-r3 | 154.65 | 39.52 | 18.54 |
| 9 | B-r3 | 127.59 | 39.89 | 19.66 |

九份原生 `benchmark.log` 的第55、56、59、60行均分别记录32条成功、0条失败、43936输入 token、256输出 token；第62、67、72行分别给出上表吞吐、TTFT、TPOT。原生 `benchmark.json` 的 `input_lens` 每项均为1373，`output_lens` 每项均为8，九窗对应的 `generated_texts` 逐字一致：

```json
"ORCHID\n\n</think>\n\nOR"
```

这验证的是指定8-token输出下的对齐，不表示长输出正确性已覆盖。三轮中 C 的 TTFT 均低于 A/B，但没有所有指标一致改善：第三轮 C 的输出吞吐低于 B，TPOT 也高于 B。三次独立部署不等于统计显著性证明。

## KV 来源与实际完成字节

来源依据是每窗 D `/metrics` 的原始前后正文。A 三窗的 `local_compute` 均由8238增至52174，`external_kv_transfer` 与 `local_cache_hit` 保持0；B/C 六窗的 `external_kv_transfer` 均由7680增至48640，`local_compute` 由558增至3534，`local_cache_hit` 保持0。每窗 P 的来源计数不增加，计时内没有新增 P 请求。

B 三个部署在各自计时窗口内，原始 service 日志各有32次 `MEMFABRIC_LOAD` 及对应的预取完成记录。每次恢复10个块，记录为 `0 L1, 10 L2`，每次成功 load 为188743680字节。不能只用空闲时的 L1 快照说明请求过程没有命中 L1。

C 三个部署的原始 worker `/counters` 响应中，G2L `completed_bytes` 均由1132462080增至7172259840，每个窗口实际增加6039797760字节，即 **5.625 GiB**；`cpu_copy_attempts` 前后均为0，计时内 L2G 完成字节不增加，同一窗口的 worker 会话保持一致。服务原始状态同时保留对象、pin、pending write 和 quarantine 的查询值。

这些计数表示同步成功 BM 调用的完成字节，不是物理 HCCS 链路流量。A/C 比较预置 KV 后读取与重算的整体行为；B/C 比较直传相对 CPU 中转的整体变化。首次 prefill 与写池成本不包含在本次 D-only 计时中。

## 退出记录与尚未解决的问题

正式九部署的独立父进程 wait 原记录中，P、D、proxy 均退出0；B 的 member 退出0。service 收到可核对的 SIGINT 后退出 **130**，按受控退出记录，**不能改写成自然退出0**。请求完成、native 收尾日志与最终 wait 码分别核对。

之后的一次取消部署在 BM 收尾期间挂住。原始线程转储定位到当时实际加载的 `memfabric_bm.py` 第142行：

```python
self.handle.destroy()
```

结合当时的实际源码，前面的 `self.wait()` 与 `self.handle.leave()` 已返回；不能沿用早期清理回执中的“leave hang”标签作为最终定位。该 service 最终由人工清理，父进程实际观察退出 **-9**，生命周期验收未通过。此问题尚未解决，也不能据此解释其他历史 SIGKILL 的来源。

原 headers-only 取消协议未观察到 engine abort：代表性失败窗口只见 lookup/release 增量，G2L 与完整恢复增量为0；尾部轮询还出现查询超时。该记录保持失败或未观察到，不能计作 worker 直传取消通过。

after-first-SSE 新协议已经实机执行四次。每次均保存了首个非空、非终止的 completion SSE 原始字节，关闭连接前已观察到生成；每个窗口的原始 worker 查询都显示 G2L 增加188743680字节、CPU copy 为0，随后保护与在途操作排空，最后完整请求恢复成功。但四次原始 D 指标中的 `finished_reason="abort"` 均由0保持为0，未取得 engine abort 的充分证据，严格验收仍未通过，不能改写为取消成功或 native DMA 中断。拟启用原生请求日志的追加诊断本轮 **NOT_EXECUTED**，没有放宽最后启动余量来挤入运行。

## 扩展条件中的 native 收尾异常

除上面的 destroy 挂起外，另有两个独立服务退出异常。二者都发生在计时请求结束、收到停止信号后的收尾阶段，不能归类成恢复量不足：

| 扩展部署 | 原始 service wait | 原始服务日志与 Python 栈位置 |
|---|---|---|
| 3000输入第一轮 C | -7 / SIGBUS | `Fatal Python error: Bus error`；BM uninit 完成日志之后，正在停止 `DeviceHostFuncDispatcher` |
| 1373输入、C4第三轮 C | -11 / SIGSEGV | `Fatal Python error: Segmentation fault`；BM uninit 完成日志之后，正在停止 `DeviceHostFuncDispatcher` |

两份原始栈的当前 Python 线程均位于 `periodic_thread.py:270` 的 `stop`，上层依次为 `native_completion.py:95` 的 `stop` 和 `lmcache_driven_transfer.py:449` 的 `close`。这是故障现场，不能据此确定 native 崩溃的具体指令或根因，也不能把它们与此前第142行 destroy 挂起当作同一个已解释问题。

其中1373/C4第三轮的原生压测仍是32条成功、0失败、43936输入、256输出，输出与前述 ORCHID 文本一致，external恢复量为40960 token；P、D、proxy 实际退出0，service 在受控 SIGINT 后实际退出-11。3000/C1的九窗请求、来源、字节与资源观测对齐，但第一轮 C 的 service 退出-7，因此**不能把该扩展条件写成三轮完整生命周期全部通过**。两个 native 收尾异常均未解决，属于完整运行验收的阻塞项。

并发路径还须按实际来源分开：1373/C4 的 B 三窗原始 service 日志各有32次全量 `0 L1, 10 L2` 预取及相应 load。3000/C4 的 B 则三窗均出现8次全量 L1 预取，首窗仅22次全量 L2 load，后两窗各23次；同时存在额外重算。该扩展是混合路径，不能与 C 的全量读池结果作为等恢复量的传输性能比较，也不能用空闲时 L1 为0覆盖请求过程中的 L1 使用。

## 容量原始 RPC 补证的失败与本地修正

追加的原始 RPC 审计组实际失败：P、D 退出1，service 受控退出130。P 在第一次 store A 的 NPU gather 中报：

```text
num_planes must be in [1, 4] (explicit plane geometry from registration), got 0
```

P 的实际 L2G/G2L 完成字节均为0；D 因收到 P 失败通知退出。已保存的12份服务 payload 只覆盖注册、状态、成功的 `prepare_store` 与后续 `abort_store`，没有执行到 stale descriptor、旧 generation 或全部 pin 住时的 pool-full 拒绝。因此这些原始 payload 不能被当作目标拒绝语义已补证。

该新增审计封装提前导入 `_Control`，发生在原合成用例 helper 导入 `torch_npu` 之前；失败组原始日志显示平台选择为 `StubCPUDevice`，原先成功的容量组则为 NPU。这是新增封装可确认的启动顺序差异，不能把本次失败直接归因于 MemFabric 的淘汰或保护逻辑。仅本地将顺序改回 `torch → torch_npu → _Control`，并检查原 Future、响应、异常及 payload 字节保持不变；CPU检查通过，**修正后没有再次做 NPU 验证**。原实机失败记录保持失败。

## 最终释放与结论范围

北京时间 **2026-10-29 22:02:07** 的最终原始释放记录确认自有容器已停止、PID为0，并完成设备释放核查；没有在到期后重新启动实验。容器停止结果与上述 service 原始 wait 码分别保留，不能互相替代。

正式可重复性能结论仍以本文的1373输入、8输出、C1、第二批完整九窗为准。扩展请求完成、取消证据不足、审计封装失败以及 native 生命周期异常分别记录。本文件不宣布跨机 HCCS 收益、TP>1或长期稳定性。
