# P 写 / D 读混合功能用例

这是已有 C diagnostic 组上的 CPU HTTP 客户端，不启动服务、不清缓存、不直接释放池保护，也不做故障注入。要求 Qwen3-8B/BF16/TP1、2 GiB DRAM 池、32 MiB 槽位（64 槽）。

```bash
python "$MF_EXPERIMENT_ROOT/runtime/mixed_requests.py" \
  --dataset "$MF_EXPERIMENT_ROOT/bench1373_multi4/data/custom.jsonl" \
  --tokenizer "$MF_MODEL" \
  --output-root "$MF_EXPERIMENT_ROOT/C-diagnostic01/mixed01" --run
```

默认只打印计划。`--run` 才使用已有本地 tokenizer（`local_files_only=True`，不下载）与 HTTP 端点。新结果目录必须不存在。

`MF_EXPERIMENT_ROOT` 和 `MF_MODEL` 沿用本目录归档约定，由调用者提供已确认的个人运行目录和只读本地模型路径；两个 Python 文件不包含个人绝对路径。

脚本先检查池容量与空闲状态。初始池只能为空，或仅含原四前缀的一部分（10/20/30/40 个对象）；需要时补齐四前缀，再分别通过 D 读取确认 40 个旧对象均可恢复。此阶段的流量不计入混合窗口；初始包含其他对象时不清理，而是拒绝继续。

两份新前缀只将原始首个 Record 000 改为 Record 400/500。完整原始 prompt 不重渲染、不截断；本地 tokenizer 必须确认六份均为 1373 tokens、首个 128-token 块六份互异。运行时另核对实际 HTTP usage。

三个客户端线程经 barrier 发起：两个 P 请求各写一份新前缀、输出 1 token；另一个线程循环读取旧四前缀，共八条 D 请求、输出上限 8，答案应含 ORCHID。两份新对象共 20 槽，最终工作集 60 槽，不应淘汰。

混合窗口要求实际 P L2G 新增 **360 MiB**、D G2L 新增 **1440 MiB**；P 两请求均重算，D external/compute/prefix 合计 **10240/744/0**。两 worker 会话不变、CPU BM copy 和 native 失败增量为零，最终 pin/pending/quarantine、请求以及 RPC/retrieve/copy 在途计数必须已知归零。HTTP 返回后继续等真实写入和状态完成，不把 API 返回当作 store 已提交。

保存 barrier 时间、各请求实际客户端调用区间与相交区间。没有相交时标 `NOT_OBSERVED`。客户端区间相交只证明请求并存，不证明数据搬运同时执行，也不报告 DMA 重叠或性能收益。RPC/retrieve 数据仅覆盖对应 worker，不能推广为 scheduler 或纯网络耗时。

`check_mixed_requests.py` 仅检查 CPU 假 tokenizer、计数判据与 barrier/区间逻辑；不替代真实模型验收。
