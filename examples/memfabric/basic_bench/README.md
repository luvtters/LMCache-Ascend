# Qwen3-8B：三档基础用例与 MemFabric A/B 入口

准备日期：2026-10-24。用于单机 1P1D 的基础功能检查和初步观察，不是客户 GLM 五档的原样复现，也不是正式性能验收。模型保持 Qwen3-8B，不涉及 DeepSeek-V4 部署。

本目录只提供数据生成与向**已有服务**发送请求的入口，不启动、停止或重启推理、LMCache、MemFabric、代理或容器。`bench.sh --dry-run` 只展开命令；不加该参数会真正发送推理请求，须等设备使用窗口恢复后运行。

## 1. 来源与裁剪

参考的本地材料：

- `E:\work\ks\output\customer_2p1d_scripts_20261019\m12\glm5\glm5.1+vllm - ks98.md`：客户五档原文；50%/90%/99% 的定义仍未查实。
- `E:\work\ks\output\ks_customer_glm_bench_audit_20261019\README.md`：来源和统计口径核查。
- `E:\work\ks\output\ks_glm_fivecases_prepare_20261019\`：候选说明、`build_validate.py`、`benchmarks/128k_p90.sh`、`benchmarks/bench参数说明.md` 和分词报告。

沿用候选材料的公共前缀定义、完整输入重新分词校验、8 条顺序请求、并发 1、官方 `vllm bench serve` 和结果保存方式。这里使用新构造的英文短句填充文本，没有复制客户 prompt；仅复用构造方法与执行口径，数据标记为 `synthetic_basic_case`。原件及 GLM 候选材料均未修改。

| case | 完整输入 token | 公共前缀目标 | 输出 token | 请求数 / 并发 |
| --- | ---: | ---: | ---: | --- |
| qwen_2560_unique | 2560 | 不共享完整的首个 128-token 块 | 128 | 8 / 1 |
| qwen_2560_prefix50 | 2560 | 1280（50%，10 个块） | 128 | 8 / 1 |
| qwen_2560_prefix90 | 2560 | 2304（90%，18 个块） | 128 | 8 / 1 |

输入加输出为 2688，适用于此前验证的 `max-model-len=4096`。不同请求的全文和 token 序列均不相同；不同 case 不共享完整首块。unique 档仍可能共享不足一个块的说明文字，因此不把它声称为“绝对零公共 token”。

50%/90% 是数据构造比例，既不是实际命中率，也不是延迟分位数。首次请求是否冷、缓存容量、块边界、路由和 P→D 传递都影响实测统计。8 条请求只用于基础检查，不用来宣称稳定的 P95/P99 或性能提升。

## 2. 文件与生成数据

- `generate_cases.py`：仅依赖标准库与 `tokenizers`，不导入 torch、torch_npu、vLLM、LMCache 或 MemFabric。
- `bench.sh`：同一个入口用参数 `A` / `B` 区分组别，执行一次官方 benchmark。
- `generated/basic/`：本次已生成的 24 条输入、CPU 校验报告；此目录被 Git 忽略，随本地交付包提供。
- `results/`：将来每次实际执行的独立结果目录，被 Git 忽略。

离线交付时将整个 `basic_bench` 目录放到已有仓库的 `examples/memfabric/` 下，保留 `generated/basic` 即可使用本次小数据。压缩包不包含 tokenizer、模型权重和依赖；复用容器内现有环境。已有 MemFabric 链路验收脚本仍位于上一级，不在包内重复提供。

生成器需要已有 Qwen3-8B 的 `tokenizer.json` 和 `tokenizer_config.json`，不读取模型权重、不下载文件、不修改模型。只支持本次核查的 Qwen2 tokenizer 包装，遇到其他包装或特殊 token 行为会停止。

以下为容器内 CPU 准备命令。文件应放在已确认的个人目录；当前可沿用 `/workspace/d00896497/src/LMCache-Ascend`。不需要复制或修改公共模型。

```bash
cd /workspace/d00896497/src/LMCache-Ascend/examples/memfabric/basic_bench
python generate_cases.py \
  --tokenizer /workspace/d00896497/models/Qwen3-8B \
  --output generated/pair01 \
  --dataset-id pair01
export MF_DATA_ROOT="$PWD/generated/pair01/dataset"
```

`pair01` 只是示例；每轮新的冷态 A/B 对照用一个未使用过的 dataset-id。A 和 B 必须复用同一份数据，不能各自改 ID。已有输出目录会被拒绝覆盖。数据按最终完整文本重新编码，不依赖字符串截断或 GLM 的 token 计数。

本地已准备的样本位于 `generated/basic/dataset`，不设置 `MF_DATA_ROOT` 时入口默认使用它。仅从 Git 获取脚本时，先按上面的命令生成数据。新 ID 可以避免复用之前相同前缀，但不能证明整套缓存为空；仍需记录缓存初态。

## 3. 无卡时：只展开命令

```bash
bash bench.sh --dry-run A qwen_2560_prefix50 preview01
bash bench.sh --dry-run B qwen_2560_prefix50 preview01
```

此模式不检查服务、不创建结果文件、不导入设备框架、不发送推理请求。输出是一条可复制的完整 `vllm bench serve` 命令。真正执行时脚本还会检查数据与 LMCache 后端；单独复制原生命令执行则不包含这些检查和快照。

## 4. 有卡之后：服务应满足的 A/B 条件

本轮采用 [TP1 后八设备终端指导](TP1_8_15_操作指导.md)：P 为物理设备 8、D 为 10，各 TP1；service 可见范围严格为 8–15，BM owner local 4 对应物理 12，member 仅可见 14、使用 local 0。客户端代理 28000、P 28001、D 28002、LMCache 25555 / HTTP 28080。A/B 顺序执行，所有设备工作须处于当次授权窗口。用户最新指定 2026-10-25 本轮窗口于北京时间 16:15 截止。

共同设置：Qwen3-8B、服务名 `mf-test`、chunk/block size 128、L1 4 GiB、LRU、`max-model-len=4096`、`max-num-seqs=1`、`max-gpu-workers=2`，其余推理参数保持相同。两组都关闭 vLLM 本地前缀缓存，保留 LMCache L1，并显式使用 prefetch `retain`。

| 设置 | A：保留 LMCache，不接 MemFabric | B：LMCache + MemFabric |
| --- | --- | --- |
| `--l2-store-policy` | `default` | `default` |
| `--l2-prefetch-policy` | `retain` | `retain` |
| `--l2-adapter` | 不设置，L2 列表为空 | 本项目 MemFabric 插件 |
| BM member | 不运行 | rank 0 为 service local 4/物理 12；rank 1 为 mask 14/local 0/物理 14 |
| P 通知 | 直接到普通代理 `:25768` | 相同 |
| L1 清空门控 | 不使用 | 不使用 |

本轮 B 使用 world size 2、每 rank 池 4 GiB、storage rank 1、槽位 **32 MiB**、store URL `tcp://127.0.0.1:28580`；完整个人 JSON 与人工启动命令见上述指导。TP1 对象约 18 MiB，不能复用 TP8 的 4 MiB 槽。旧 `tp8_service.sh`、`l2_tp8_memfabric.json` 与 16 卡布局仅保留为[历史资料](../README.md)，不作为本轮默认入口。两组保持相同日志设置。

这里保持已有容量，目标是基础验证。该槽位布局下 L2 有效 KV 容量约 2.25 GiB，unique 档可能触发淘汰，不要求所有对象最终驻留。共享前缀档通常可以命中 L1，因此不能把一次成功请求当成读池证据。

切换 A/B 前确认上一组工作已结束，采用独立或重新初始化的实验缓存，不能把 A 的已热 L1 直接作为 B 的初态。相关启停需在新的设备窗口内人工完成，本入口不会代办。当前脚本只自动核对 A 没有 L2、B 恰好有一个健康的 SDMA 适配器；存储策略、P/D 映射及全部启动参数仍按本表人工核对。

## 5. 向已就绪服务实际发送请求

仅在允许使用设备的窗口执行。使用同一 tokenizer、数据、普通代理和模型别名。以下每一行独立执行一档，不是后台批处理。

```bash
source /workspace/d00896497/env.sh
export MF_BASE_URL=http://127.0.0.1:28000
export MF_TOKENIZER=/workspace/d00896497/models/Qwen3-8B
export MF_MODEL=mf-test
export MF_CACHE_STATE=target_prefixes_unseen_other_cache_state_unknown

# A 服务准备好后：
bash bench.sh A qwen_2560_unique pair01
bash bench.sh A qwen_2560_prefix50 pair01
bash bench.sh A qwen_2560_prefix90 pair01

# B 服务另外准备好、缓存初态核对后，使用同一份 MF_DATA_ROOT：
bash bench.sh B qwen_2560_unique pair01
bash bench.sh B qwen_2560_prefix50 pair01
bash bench.sh B qwen_2560_prefix90 pair01
```

只有确认目标前缀未提交过时才使用上面的 `MF_CACHE_STATE` 描述；否则填写实际情况或保留默认 `unknown`。每次重跑使用新的结果 tag，旧结果目录拒绝覆盖。新 tag 仅改变结果文件名，不改变输入或清除缓存。

| 环境变量 | 默认值 | 用途 |
| --- | --- | --- |
| MF_DATA_ROOT | 本目录 `generated/basic/dataset` | 三档数据所在目录 |
| MF_RESULT_ROOT | 本目录 `results` | 已确认的个人结果目录 |
| MF_BASE_URL | `http://127.0.0.1:28000` | 普通 P/D 代理 |
| MF_MODEL | `mf-test` | 已就绪服务的模型别名 |
| MF_TOKENIZER | `/workspace/d00896497/models/Qwen3-8B` | 与服务一致的 tokenizer |
| MF_SERVICE_URL | `http://127.0.0.1:28080` | LMCache HTTP 状态接口 |
| MF_P_URL / MF_D_URL | `http://127.0.0.1:28001` / `http://127.0.0.1:28002` | P、D 的只读 metrics 接口 |
| MF_CACHE_STATE | `unknown` | 人工记录缓存初态，不执行清理 |
| MF_LAUNCH_CONFIG | 未设置 | 可选的真实服务启动 JSON，保存到 run.json 并用于区分 TP/模式/预取策略；不能代替全部参数核对 |

基础入口固定 N=8、C=1、输出 128、seed 0、temperature 0、无额外预热、无就绪探测推理、无随机打乱、无超采样。使用 `/v1/completions` 和 `--skip-chat-template`，因此不会再给输入套聊天模板；不要改成 chat 接口后继续沿用本次 token 校验。原生 benchmark 采用流式响应测量延迟，`--ignore-eos` 用于请求 128 个输出 token。生成文本只是合成负载，模型语义正确性由已有固定答案验收用例检查。

2026-10-25 真机发现当前 vLLM benchmark 不再默认使用 temperature 0，因此入口显式固定该值。修改前后的结果按记录的命令分别配对，不能作为相同采样设置合并统计。

若要单独检查 L1 清空后的恢复，可在选定 case 已运行、服务空闲时使用 `python probe_l1.py A --tag probe01` 或对应的 B 命令。它先核对后端，非强制清空本项目服务的 L1，保留 B 的池，再重放默认的 90% 档。该诊断会改变实验缓存，需与普通三档分开记录；A/B 仍使用同一数据。`summarize_results.py RESULTS_ROOT` 可汇总已有结果，按 case、数据 ID、缓存初态标签和压测命令匹配 A/B，服务启动参数仍需另外核对。

当前固定 core 的 store default 和 prefetch default 是两个独立设置：后者默认不保留从 L2 恢复到 L1 的数据。若目标是恢复后继续利用 L1，可在两组共同显式设置 `--l2-prefetch-policy retain`，并独立记录这一轮参数；A 无 L2 时该预取设置没有作用。门控和取消验收仍沿用要求 L1 排空的 default 预取策略。

有卡且专用 B 服务空闲时，可用 `python retain_smoke.py --output results/retain_唯一名称` 检查 retain：3 次写入、非强制清 L1、3 次恢复、3 次热重放；逐次核对不同密码和真实 BM 读取，恢复后检查 L1 驻留及锁/pin 归零。此检查会发送 9 次真实推理请求并改变实验缓存，需与 benchmark 分开执行。2026-10-25 的后八设备 TP1 实测已通过这条路径。

两份 retain 验收还会核对 P/D 实际 token 来源，排除“不读池但完整重算”的假通过。`concurrent_retain_smoke.py --output results/concurrent_唯一名称 --concurrency 4` 测试客户端并发：冷恢复允许并明确报告部分重算，热重放则要求 P/D 完整块来自 external KV、引擎本地前缀命中为 0、BM 新增读取为 0。引擎 `max-num-seqs=1` 时，这不代表单引擎同时执行 4 请求。

## 6. 结果怎么读

每档结果目录为 `results/<tag>_<A或B>_<case>/`：

- `run.json`：组别、数据 ID、展开命令、缓存初态、退出码、请求/token 数是否符合预期。`FINISHED` 只表示本次数据收集完成，不表示获得性能收益。
- `benchmark.json` / `benchmark.log`：官方结果、逐请求输出与耗时。应成功 8 条、总输入 20480、总输出 1024，每条输出 128。
- `before_status.json` / `after_status.json`：L1、L2、写入和读取字节等状态。
- `before_p_metrics.txt`、`after_p_metrics.txt` 以及 D 的对应文件：分开保留两端指标。

入口遇到组别/后端不符会在发送请求前停止。请求失败、输出不足或采集失败不能算作完整 A/B 对照。无服务的负面检查可能只留下 `PREPARED` 和部分快照，实际错误见终端。

状态与指标在 benchmark 前后读取，读取发生在官方计时窗口外；结束时异步写入或指标发布可能尚未完成。这些快照不是请求级精确归因，也不是 L2 提交屏障。若仍有 pending/in-flight，记录此事实，不能直接把差值当成完整 KV 流量。脚本不会逐请求轮询和等待，以免改变正常调度。

P、D 分开看实际计算量。D 的 `external_kv_transfer` 可能来自 LMCache L1，也可能来自 L2；不能单独等同 MemFabric 命中，P/D token 数也不能相加当作客户端输入。实际命中率字段默认保留 null，不能用 50%/90% 替代。

## 7. MemFabric 链路验收：直接复用

继续使用已有 [pd_inference_smoke.py](../pd_inference_smoke.py)，不复制或修改验收逻辑。本轮设备映射、空池准备与完整人工命令见 [TP1 后八设备指导第 6 节](TP1_8_15_操作指导.md#6-严格-p池d-验收使用新的空池)；上一级 [TP8 门控说明](../README.md#one-pd-request-through-memfabric) 作为历史参考。

该检查需要独立的新空池、store `skip_l1`、prefetch `default`，P 通知指向 25769；代理仍监听 25768。P 写池完成、L1 为空后才放行 D，核对 MemFabric 实际读取及固定答案。准备好独立 TP1 验收环境后，原生入口是：

```bash
python ../pd_inference_smoke.py --tp-size 1 --output results/pd_gate_tp1_8_15_pair01
```

不能直接对普通 A/B 的 store `default`＋prefetch `retain` 服务运行此验收；它会等待 L1 清空。门控验收的延迟独立记录，不混入普通 A/B 结果。

## 8. 本次准备与验证记录

本地生成 3 档共 24 条输入，每条完整输入恰好 2560 tokens；50%/90% 档所有请求对的公共前缀分别为 1280/2304。unique 档本次共享 28 tokens，小于一个 128-token 块，8 个首块均不同。报告见 `generated/basic/report.json`。

2026-10-24 的离线准备仅进行文件读取和 CPU tokenizer 校验。`generated/basic/remote_tokenizer_check.json` 记录 Transformers 5.5.4 / tokenizers 0.22.2 下全部 24 条、61440 个 token ID 对照一致。该检查只在临时 Python 进程内隐藏设备框架可用性并阻止导入 torch/torch_npu 等模块。

`generated/basic/entry_checks.json` 是当时的本地 CPU 入口检查，不是推理服务测试。2026-10-25 已在 137 的 8–15 设备范围实跑 TP1 两轮及 TP4 一轮三档 A/B，普通请求合计 144/144 成功；另完成严格读池、保留恢复数据、取消、成员退出与降级检查。原始请求、结果、服务配置和测量边界保存在项目 `research/artifacts/2026-10-25_137_tp1_ab`。生成报告中的 NOT_MEASURED 是离线准备时的历史状态，不替代这些设备结果。

本轮小样本没有证明 MemFabric 稳定加速。默认冷并发还观察到 P 端部分重算，热并发则已按 P/D token 来源确认真实复用。后续需分别研究并发预取争用和较大工作集的收益，不把构造比例或 byte 下降直接等同延迟提升。
