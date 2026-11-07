# MemFabric L2 兼容：Ascend 实现

更新日期：2026-11-07（北京时间）。本库通过原生 `L2AdapterInterface` 接入 MemFabric HOST DDR 池，保留官方 worker、NPU↔CPU L1 和原生 L1 管理链路。路线入口见[方案说明](../README.md)，分类原件见[当前验收](../报告/当前验收.md)。

## 版本与隔离

- 官方 Ascend 比较基线固定为 `c5e29f15902437883dba530c22705a48ff7ce0fb`。
- core 固定官方 `6913259264dbebb2fa5ba1665606c40ffc5c6661`，独立仓库、detached HEAD，不修改。
- 当前双 owner payload 为 `3012c71`；本地整理分支为 `work/memfabric-l2-local-docs-20261107`，最终文档整理 SHA 由路线 TRACKING 记录。
- 远端检视 `review/memfabric-l2` 为 `f598eb3`，与双 owner 开发节点区分，本轮不推送。
- NPU pin `fce581a` 仍在本路线独立候选目录，仅 production CPU 验证，不混入当前已设备验证实现。

本路线不依赖 L1 兼容方案的 core、worker 工厂、checkout、配置或运行产物，不使用归档或网络盘对象库。

## 接入与配置

service 使用原生 `lmcache server` 的 `--l2-adapter` 加载
`lmcache_ascend.v1.distributed.memfabric_l2_adapter:MemFabricL2Adapter`。
模型使用原生 `LMCacheMPConnector`。组件命令、环境和池成员必须按角色独立准备，不能把示例配置当成已验证的任意部署。

[默认 l2.json](examples/memfabric/l2.json)仍是 **SDMA、单 owner** 示例，不代表默认开启双机聚合。双 owner 当前实测配置需显式使用 `DEVICE_RDMA`、共享目录参数 `shared_directory` 和 `storage_ranks: [0, 1]`；`pool_gb` 表示每 owner 的贡献，两个 owner 各 8 GiB 时总贡献为 16 GiB。两端还须匹配固定 BM 组、协调端点、设备及各自网卡。详见[跨实例说明](examples/memfabric/l2_cross_instance.md)。

## 已有实际依据

`3012c71` 配官方 core `6913259` 在 137＋25 各 8 GiB 的真实双 owner 容量关写入并读回 960 个对象，共 8,853,258,240 B，全量逐字节一致、无淘汰。合成内容一致不替代全部真实模型 KV 逐字节证明。

GLM 26 题完整结束并核实两端 LOAD；本轮写入端与恢复端 26/26 同答，NIAH 6/6、MMLU 18/20。相对历史恢复答案仅 25/26，严格历史答案等价未通过，同服务强制重算不计正式 OFF。

完整 MemFabric L2＋`DEVICE_RDMA` 对 Mooncake Store＋经典 TE `ascend` 完成四组配对；输出吞吐为 5.80/5.27、16.68/16.20、24.07/17.90、3.95/5.38 tok/s。两套池管理、放置和 transport 路径不同，不是纯传输对照。长输入只有 3 条；4K→512 C1 仅 Mooncake 完成，不补作配对。

## 未完成及生命周期

正式单机 OFF MMLU20 为实机 0/20；TE 只有 fake CPU 契约，真实 TE 注册/搬运和同目录 BM→TE→TE→BM 未验。核心 `mooncake_te_impl.py` 只有读取能力，不能直接替换完整存取路径。

计划 SIGINT 服务退出130、SDK关闭告警及Mooncake Store释放阶段-9分别保留；功能完成不改写为所有角色干净退出。11月6日释放回执是历史现场，不是本轮设备状态或授权。本轮仅本地迁移、CPU检查与文档整理。
