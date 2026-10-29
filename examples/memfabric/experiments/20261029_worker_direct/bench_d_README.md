# D 定点 A/B/C 压测入口

`bench_d.py` 只请求已运行的 D，不启动、停止服务，不通过 SSH。默认只打印计划；`--run` 才渲染数据、采样 HTTP 和调用官方 `vllm bench serve`。容器内模型和输出位置由本轮编排提供。

先以本地 tokenizer 渲染完整 gate chat，保留 system/user 消息，关闭 thinking：

```bash
python bench_d.py --run --prepare-only --case C \
  --request-json "$GATE/request.json" --tokenizer "$MODEL" \
  --input-tokens 1373 --output-root "$RUN" --requests 32
```

这一步不发送模型请求、不下载 tokenizer。生成 `data/{custom.jsonl,manifest.json,prompt.txt}`。1373 是上轮 gate 的可选预期校验值；输入变化时按真实分词结果填写或不传，不硬套旧长度。后续三组都复用同一 `custom.jsonl`。

本轮正式 C4 使用 `prepare_multi.py` 的四个不同前缀轮转数据；上面的单前缀输出只用于基础准入或单前缀探索。并发同前缀出现临时 L1 合并或恢复量变化时，不能将该窗口作为等搬运量主对照。

一次正式 C 采样示例：

```bash
python bench_d.py --run --case C --round 1 --concurrency 1 \
  --dataset "$RUN/data/custom.jsonl" --input-tokens 1373 \
  --expected-external-tokens 1280 --tokenizer "$MODEL" --model mf-night \
  --base-url http://127.0.0.1:28402 --service-url http://127.0.0.1:28195 \
  --p-url http://127.0.0.1:28401 \
  --counter-url http://127.0.0.1:29402/counters --output-root "$RUN"
```

- `--case A` 要求 D 全量重算、external=0、local prefix=0；不要传直传 counter URL，`--expected-external-tokens 0` 可作显式校验。
- `--case B` 从 service status 的 adapter `load_bytes` 取完成读池字节，不能使用 `tokens × 每 token 大小` 冒充实测流量。
- `--case C` 从实验 counter 的 `G2L.completed_bytes` 取差；前后 `pid/session_id` 必须一致，CPU copy attempts 与错误增量为 0。
- 三组 D 需先验通过 `kv_consumer`，关闭本地 prefix。脚本不能仅从结果证明启动角色配置正确，须保留编排配置。
- 正式运行必须指定 `--p-url`，检查计时窗口 P 的 prompt source 增量为 0；B/C 还必须指定 `--expected-external-tokens`，C 必须指定 counter URL。计划与仅准备数据的模式不要求这些运行端点。
- 改 `--concurrency 4` 使用 C4，改 `--round` 或 `--output-root` 保持目录唯一。已有结果目录拒绝覆盖。
- 每轮保存官方 `benchmark.json`、`benchmark.log`、子进程返回码、前后 HTTP 原文及 `summary.json`。前后适用的锁、pin、在途和暂存字段必须实际观察到且归零；前置未排空或状态未知时不会启动计时。缺计数、重启导致计数回退、来源不对齐或失败请求均不能记为通过。返回 2 表示该轮不接受用于对照，原始失败保留。

本地只检查了纯解析、计数回退/重启拒绝和默认计划模式；真实 tokenizer 渲染、官方 benchmark、HTTP 指标和 NPU 路径须在本轮设备窗口另外验收。
