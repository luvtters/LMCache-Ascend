# L2 对象与搬运机理诊断入口

本目录归档兼容 LMCache core `6913259` / LMCache-Ascend `525a710` 的无模型诊断入口，比较实际 `MemFabricL2Adapter` 与现有 Mooncake Store `NativeConnectorL2Adapter`。没有修改生产搬运、调度或退出语义，不包含 worker 直传。

## 三个入口

- `l2_api_bench.py`：E00 内容与删除检查、E01a 实际 adapter 生命周期、E02 对象数/大小扫描；`--rw-probe` 为 E05 读写排队对照。
- `l2_controller_bench.py`：E01b 实际 StorageManager、StoreController、PrefetchController 路径；逐轮要求 0 L1、完整 L2 恢复，正常结束读取并删除 L1。
- `mechanism_store_owner.py`：独立 owner 预置及副本查询。`preferred_segment` 只是偏好，仍逐对象核对实际完整副本属于 owner。

默认只打印 `PLAN_ONLY`，不初始化设备。实际运行必须同时给出 `--run --allow-npu`、明确的设备可见范围、固定源码路径、已准备的配置及新结果目录。MF 的独立 BM member、Store 的 master/owner 和外部截止守卫由调用方管理；本目录不部署服务器、下载依赖或创建容器。

## 内存与计时口径

`--memory-kind shared` 为默认：实际使用 L1MemoryManager → MixedMemoryAllocator → SHM pinned arena → TensorMemoryAllocator。其对象是合法的合成四维 uint8 KV_2LTD 布局，**不是 Qwen 的真实 KV 布局**。`--memory-kind torch` 是无 parent allocator 的合成 MemoryObj 诊断，布局和分配路径都单独记录，不能混入 shared 组或归因为只改变一个 allocator 参数。

源数据准备、预触页、分配和完整内容校验与 load 计时分开。每格首个完成 load 保存实际对象头尾各16字节及预期值；完整 body 检查仍执行。原始服务/native 日志、事件和查询应完整保存在私有结果目录，不能以脚本布尔判断替代。

`--trace` 默认关闭。启用时观察 MF 既有 executor 提交/出队和同步 BM 返回；Store 观察 SDK 提交、原生完成 drain 和 L2 结果消费。**Store SDK 内部 worker 出队、纯 copy 耗时及实际 transport bytes 仍为 UNKNOWN**；成功 payload 字节不冒充链路字节。嵌套或重叠的 span 不能相加当作总时延。

E05 固定64×18MiB读集合，另提交32×18MiB新 key 写入，总96对象，适配4GiB/32MiB槽的128槽池。两条件保持相同源/目标地址和内存占用，仅切换是否提交 SET；正常删除新增写对象并核对无淘汰。若同一次 poll 同时观察到两个完成 fd，原生完成先后为 UNKNOWN，不能用随后 Python 逐项消费时间推断。

## 最小调用示例

准备好设备、独立 owner/member、配置和外部超时监督后，例如：

```bash
python l2_api_bench.py --run --allow-npu --backend memfabric \
  --config /path/to/memfabric.json \
  --core-source /path/to/LMCache --ascend-source /path/to/LMCache-Ascend \
  --output /path/to/fresh-smoke --cases smoke --warmups 0 --repeats 1 --blocks 1 --trace
```

MF 配置使用纯 BM 参数对象（`device_id/world_size/pool_gb/storage_rank/slot_size_mb/store_url`），或用 `params` 包装该对象。Store 组改为 `--backend store`，增加 `--native-so /path/to/lmcache_mooncake.so --owner-receipt /path/to/owner.json`，配置使用已有 `mooncake_store` 字段。配置、SDK、实际 transport、owner 落点和角色设备必须与对应记录匹配。

E01/E02 用 `--cases all`，或按脚本 `CASES` 中的 case id 选格。E05 用 `--rw-probe --warmups 1 --repeats 2 --blocks 1 --trace`。E01b 的对应 source/config/native/owner 参数相同，可先用 `--objects 1 --l1-bytes 67108864 --warmups 0 --repeats 1`，128对象使用默认3GiB L1。所有性能窗口须分别保存实际 argv、版本、计时范围、来源、原始结果和父进程真实 wait 码。

## 本轮实际运行与未解决项

2026-10-29 的单机窗口已实际运行 shared 路径三轮13格 M/S 配对、Store 13格 trace、E05 两后端、torch 诊断两后端及 MF 槽大小变体。MF E01b 的1对象与128对象两次实机 client 进程 wait 均为0。这里仅说明执行范围；逐格数值、落点、内容、完成状态和退出准入以私有原件审阅为准，不把同进程重复称为独立实例。

生命周期仍未全部通过：Store 独立 owner 曾在 `close()` 返回0之后发生 heap corruption，父进程真实 wait 为 **-6**；Store controller 原路径及额外 GC 诊断变体出现真实 wait **-11**，尚未解决。早期 MF member 因监督器过早发送停止信号而出现 **-2** 的失败样本也应保留，与自然退出复测分开。不能用 `CLOSE_RETURNED`、请求完成或退出前打印替代真实 wait，更不能通过 `os._exit()` 掩盖异常。

另提供 Store-only 可选诊断 `--release-native-after-close`，默认关闭：在正常 StorageManager.close 返回、demux worker 确认已退出且当前 ACL context 有效后，释放该已关闭 adapter 保留的 native client 引用，保存析构前后 context 原值。独立的1对象与128对象实机诊断中，客户端由原路径的-11变为0；单纯清引用环并做GC的组仍为-11。该结果支持继续检查 native 对象释放时序，**没有修改生产 core，也没有解决独立 owner 的-6**。该诊断和原始基线应使用不同目录，不能混写成全部生命周期已修复。

本目录不包含内部原始日志、主机地址、设备地址、运行身份、SDK 二进制或带个人路径的编排器。尚未证明跨机 HCCS 收益，也不将底层或 adapter 数字直接解释为完整推理服务收益。
