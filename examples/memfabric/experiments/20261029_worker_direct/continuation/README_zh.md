# 独立接续执行：本地修复与验收

本轮状态：**本地代码及CPU接续验证完成；未部署、未安排新的设备任务。** 中断原因及时间线见[本轮阶段报告](../RESULTS_zh.md)。内部会话原始记录不包含在本目录。

这次执行规则是：授权窗口内没有达到可汇报条件，就继续已批准清单与清理成功后的新组重试；不能把单窗成功、脚本准备完或命令返回0当成可以收尾。先完成一个基础A/B/C条件的独立重复，再扩展输入、并发和功能点。所有工作受停止新增与最终释放时间约束，不补跑已过期窗口。

## 修复了什么

- `campaign_runner.py`：普通Python独立进程持有完整清单。开始前检查每个后继脚本、清理入口及质量检查入口；全部验证后才执行。默认`PLAN_ONLY`，`--run`才运行。
- 显式声明的交接码记`HANDOFF`，不冒充实验通过。声明了清理命令时，交接也必须先确认清理成功。
- 真失败保存原始退出码。清理成功后可继续下一批准步骤，`retry_failed=true`会将失败步骤加入队尾。每次重试新建attempt目录；命令必须使用`{attempt}`或`{attempt_dir}`构造新组或新结果位置。
- `report_gate_argv`定义“有可汇报结论”的严格检查。队列做完但质量检查非0时，整份实际步骤以新attempt再跑，直到质量检查通过或不能新增实验；不是只轮询检查脚本。
- `report_gate.py`只读核对汇总状态：要求`COMPLETE`、缺失/不合法/未知监控项明确为空、ABC齐备及至少指定重复数。最小一个条件用`--groups 3`，三个条件用`--groups 9`。它不生成或补造样本。
- `stop_new_at`在实际创建子进程前再次核对；清理预留时间并受`stop_at`约束。子进程收到两项明确窗口环境变量，既有`control.py`读取这些变量；不再依赖修改旧日期。

完整清单是执行的前提。若计划在跑到一半时改变，应先生成并验证完整替代清单，并取得其可执行回执，才能在旧组已清理的边界交接；不再先放置`STOP_LONG_MATRIX`后等待聊天临时补代码。本执行器不支持随意热改正在运行的清单。

## CPU证据

```bash
python check_campaign_cpu.py
```

[campaign_cpu_check.json](campaign_cpu_check.json)记录14项真实CPU子进程检查，包括启动者退出后仍接续三步、退出3交接、清理失败阻止后继、缺失后继在前继运行前被拒绝、过期零启动、截止前复查、失败新attempt重试，以及A1/B1→质量检查2→实际A2/B2→质量检查0。

另见[report_gate_cpu_check.json](report_gate_cpu_check.json)、[window_override_cpu_check.json](window_override_cpu_check.json)。没有Torch/MemFabric/模型/设备执行。这证明独立进程的接续行为，不代替真实NPU链路验收。

同一14项检查也在本机非root WSL Linux中通过，覆盖POSIX进程分支，见[campaign_linux_cpu_check.json](campaign_linux_cpu_check.json)；没有连接实验服务器。

## 下一窗口怎样实际启动

在新的明确授权日期和范围内，准备真实`campaign.json`：

- `start_at`、`stop_new_at`、`stop_at`：带时区的ISO时间，严格递增。
- `steps`：按优先级排列的完整已批准命令；每项含唯一`id`、`argv`、`timeout_seconds`，资源步骤配明确的`cleanup_argv`及`cleanup_timeout_seconds`。
- `retry_failed`：是否清理后用新attempt重试。
- `report_gate_argv`、`report_gate_timeout_seconds`：实际汇总与严格完成判定入口。不能直接用“打印IN_PROGRESS但退出0”的分析命令充当完成检查。

资源步骤应调用现有容器内实验入口，使用本次新组名和新结果目录。质量检查应收集这些结果并调用严格检查，不得单独重试一个永远不会新增样本的分析命令。所有脚本、数据、源码、环境和只读模型路径须在启动前核实；这份说明本身不是已完成的实机清单。

以下仅为**尚未执行的容器内命令形态**；`RUN`和`PY`由新授权窗口的个人目录及已有解释器确定：

```bash
source "$RUN/env.sh"
"$PY" "$RUN/runtime/campaign_runner.py" --plan "$RUN/campaign.json" --output-dir "$RUN/campaign_results"
nohup "$PY" "$RUN/runtime/campaign_runner.py" --plan "$RUN/campaign.json" --output-dir "$RUN/campaign_results" --run > "$RUN/campaign_launch.log" 2>&1 < /dev/null &
echo $!
```

默认计划检查不创建结果目录；真实启动要求新的结果目录。回读本轮新`ready.json`、`status.json`，核对executor及当前child PID、开始时间、存活状态和新日志，才能称“正在执行”。`READY`只表示清单验证，不表示定时触发已经注册或NPU已经运行。

若是未来时点，还须实际注册并回读触发器，使其启动上述独立执行入口；仅唤醒聊天不足以承受再次出现的模型容量错误。必须说明服务器/电脑/连接等触发依赖。当前没有注册任何这样的未来任务。

## 状态与截止

`PASSED`、`FAILED`、`HANDOFF`逐项保留；曾失败后恢复不会删除失败历史。`report_ready=true`才表示已配置质量检查通过；不配置质量检查时，`COMPLETE`只表示命令清单完成，不能当成可汇报结论。

`NEEDS_ATTENTION`表示旧组清理未知或失败，不能复用可能仍在访问的池地址；必须先处理本项目旧组及保护，再使用新组继续。`DEADLINE_REACHED`保留未执行待办。设备容器的独立停止/释放守卫仍必须另行启动并回读，不能用执行器或聊天承诺代替。当前修复不自动恢复任何过期服务器实验。
