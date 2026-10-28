# DeepSeek V4.1 Flash Prefill Benchmark · Ascend 950

**当前状态：v0.1 实现已提供，含原生执行器、配置、部署/采集脚本和 CPU 测试。当前开发机是 Windows，无 Ascend NPU；尚未完成 Linux/950 编译、八卡运行或性能对齐验收，没有实测性能数据。** 设计及实现取舍见[设计文档](./DESIGN.md)。

目标是在8张Ascend 950上，扫描完整输入长度、虚拟前缀缓存命中率和四种并行策略，得到可比较的Prefill forward时延、吞吐及全部rank的profiling文件。

直接调用DeepInfer原生模型、metadata和Ascend kernel。默认用完整维度dummy权重，不重写一套近似的attention/MoE图；也支持真实checkpoint对照。

## 支持范围

| 项目 | 当前实现 |
| --- | --- |
| 硬件 | 同一高速互联域内8张Ascend 950；实际HBM/驱动/固件在doctor中检查 |
| 并行 | DP8CP1EP8 / DP4CP2EP8 / DP2CP4EP8 / DP1CP8EP8 |
| 输入 | 纯文本，各DP组使用相同L和h；每组可有多条请求 |
| 权重 | 默认dummy，完整原生模型与量化布局；可选real |
| cache | 构造已驻留HBM的合法synthetic prefix，无需真实先计算缓存前缀 |
| SWA | decoder YOCO off / mask / compact；默认compact140 |
| MoE | 原生强制负载均衡；关闭强制倾斜和动态EPLB重排 |
| 执行 | single_forward；可显式选择chunked_suffix处理长输入 |
| 比较口径 | 默认fixed_global；另支持fixed_per_dp |
| profiling | 独立采集pass，默认只采 rank0（`ranks=all` 可全采），采集时不解析 |

首版不包含encoder SWA miss回放、DP异构长度/命中率、decode、采样、图像VE、外部KV存储或P→D网络传输。模型forward内原有的Engram、CP、HC、KV量化/读写、YOCO边界和DSpark cache preparation仍在测量中。

## 环境基线

| 组件 | 版本/要求 |
| --- | --- |
| DeepInfer | `553741557a57df44a46fb3ab72b6aafeaf945337` |
| 模型 | DeepSeek-V4.1-Flash-1017；revision `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| 系统 | 厂商支持CANN 9.2.0的Linux；具体发行版、CPU架构和glibc在目标机记录锁定 |
| 驱动/HDK/固件 | 目标950与高速互联所需版本；由机器预装，doctor记录并核验 |
| CANN | 9.2.0；含Bisheng、HCCL/HCOMM和相关库 |
| Python | 3.12 |
| PyTorch | `torch==2.13.0+cpu` |
| torch_npu | `torch-npu==2.13.0rc1` |
| host compiler | GCC≥13，C++20 |
| TileLang / TileKernels | `0.1.15` / `2.1.0` |
| kernel库 | 使用DeepInfer gitlink固定的DeepEP-Ascend、DeepGEMM-Ascend、FlashMLA和DeepSelect |

完整Python依赖沿用仓库requirements并生成lock。`deep_ep`和`deep_gemm`必须实际来自Ascend实现，不能安装同包名的CUDA版本。

dummy模式仍需要模型配置和tokenizer等原生loader访问的小文件；不下载权重shard，也不意味着模型的HBM占用可以忽略。real模式需要完整checkpoint。8卡是否容纳某个case由完整模型、Engram、KV、state和workspace共同决定；OOM会作为结果保留，不自动缩小模型。

一键部署覆盖Python环境和源码构建，前提是950系统驱动/固件和CANN已经安装。具体容器镜像digest尚未指定，不提供虚构的可下载镜像。

## 一键部署与一键采集

在 Linux/950 目标机的 DeepInfer 仓库根目录执行。先检查 CANN、驱动、Bisheng 和 GCC 已安装；脚本会记录实际环境并严格检查版本。

```bash
# 1. 创建隔离Python环境、安装固定依赖、构建原生扩展、准备dummy小文件并检查环境。
bash benchmarks/prefill_ascend950/deploy.sh \
  --venv .venv-prefill950 \
  --cann-env /usr/local/Ascend/ascend-toolkit/set_env.sh \
  --model-dir /data/models/DeepSeek-V4.1-Flash-1017 \
  --weights dummy

# 2. 一键跑四种策略；默认只采 rank0 的 trace，解析留到离线。
bash benchmarks/prefill_ascend950/collect.sh \
  --venv .venv-prefill950 \
  --config benchmarks/prefill_ascend950/configs/smoke.json \
  --devices 0,1,2,3,4,5,6,7 \
  --model-dir /data/models/DeepSeek-V4.1-Flash-1017 \
  --out results/prefill950_smoke
```

`--cann-env`应指向实际安装的环境脚本，上例路径不是硬性安装位置。部署日志保存在 `build/prefill950-deploy/`；版本、wheel/扩展 SHA256、完整 pip freeze 和 doctor 结果保存在 `<venv>/prefill950/`。collect 自动读取部署时保存的 CANN 脚本路径，也接受显式 `--cann-env`。

离线时加 `--wheelhouse /data/wheelhouse`，预先放齐 Python 依赖及构建包、模型小文件、四个 Ascend vendor 的递归子模块。离线依赖清单会去掉上游的远程 find-links。real 模式须预先提供完整 checkpoint；脚本只下载小文件，不自动下载权重 shard。构建引导包固定 `setuptools==77.0.3`、`wheel==0.45.1`，其余版本沿用仓库 requirements。

collect依次完成doctor、case plan、warmup、无profiler计时、独立profiling及汇总。不会自动执行decode或将profile期间的读数用于吞吐排名。

## 配置示例

现有 [configs/smoke.json](./configs/smoke.json) 使用四拓扑、L8192、h0.5。未指定字段使用以下完整默认值；`plan --out` 可导出实际配置：

```json
{
  "schema_version": 1,
  "runtime": {
    "weights": "dummy",
    "engram": true,
    "max_model_len": 1048576,
    "mtp_draft_length": 5,
    "seed": 20261026
  },
  "parallel": {
    "strategies": ["DP8CP1EP8", "DP4CP2EP8", "DP2CP4EP8", "DP1CP8EP8"],
    "batch_mode": "fixed_global",
    "global_batch_size": 8
  },
  "workload": {
    "context_lengths": [8192],
    "cache_hit_rates": [0.5],
    "alignment_policy": "floor",
    "cache_source": "synthetic_hbm",
    "execution": "single_forward",
    "chunk_tokens_per_request": null
  },
  "yoco": {
    "mode": "compact",
    "keep_tokens": 140,
    "store_extra_state": true,
    "max_backtracking_tokens": 6,
    "encoder_swa_state": "available"
  },
  "moe": {
    "force_load_balance": true,
    "force_imbalance_ratio": 0,
    "extra_experts": 32,
    "rearrangement_interval": 0
  },
  "measurement": {
    "warmup": 0,
    "iterations": 10,
    "rounds": 1,
    "timing_scope": "native_model_forward"
  },
  "profile": {
    "enabled": true,
    "ranks": "rank0",
    "level": "Level1",
    "wait_steps": 0,
    "warmup_steps": 0,
    "active_steps": 3,
    "record_shapes": true,
    "with_stack": false,
    "profile_memory": false
  }
}
```

L是包含缓存前缀的完整输入长度。h使用[0,1]比例，例如0.9表示目标90%，不是90。

`chunk_tokens_per_request` 默认关闭（`null`），默认执行方式是 `single_forward`，即每个 case 每次只做一次前向；只有显式设置 `execution=chunked_suffix` 并给出 `chunk_tokens_per_request` 时才启用分块。分块时 `forward` 是整条后缀（各 chunk 之和）的平均耗时，单 chunk 见 `forward_step`。

`runtime.engram=false` 关闭原生 Engram 路径（显存受限或做对照实验时使用），此时不分配 Engram 参数和 `EngramBuffer`；结果必须与开启 Engram 的运行分开报告。环境变量 `PREFILL950_DISABLE_ENGRAM=1` 仍可作为临时覆盖。

`runtime.engram=true`（默认）时，benchmark 默认设置 `DEEPINFER_ENGRAM_CPU_LOAD=1`：两张 Engram 表在 CPU 暂存，初始化时写入每 rank 1/8 的 NPU `EngramBuffer`（`EngramBuffer.write` 走 H2D），forward 由设备侧跨 8 rank RDMA fetch。这样 96 GB 卡上也能容纳；显式设置 `DEEPINFER_ENGRAM_CPU_LOAD=0` 可恢复旧的“表直接驻留 GPU”行为（96 GB 卡会 OOM）。

固定G=8时，每DP请求数依次为1/2/4/8。所有请求L和h相同，总共仍只有8条逻辑请求；CP副本不重复计数。

现有 [configs/sweep.json](./configs/sweep.json) 使用长序列扫描，workload 为：

```json
{
  "workload": {
    "context_lengths": [8192, 32768, 131072, 524288, 1048576],
    "cache_hit_rates": [0.0, 0.5, 0.9, 0.99],
    "alignment_policy": "floor",
    "cache_source": "synthetic_hbm",
    "execution": "chunked_suffix",
    "chunk_tokens_per_request": 4096
  }
}
```

以上片段也能独立作为配置，其他段使用默认值。长 context 且低命中率时显式 chunk，以限制 query 和 MoE workspace；完整 KV 仍可能 OOM。固定每 DP 请求数可直接使用 [configs/fixed_per_dp.json](./configs/fixed_per_dp.json)；两种 batch 模式的字段互斥。

`max_model_len`默认固定1M，不随每个L自动缩小；它可能影响metadata容量及indexer分块/stride。runtime派生长度、query容量和KV实际分配量都会保存在manifest中。

## 缓存命中率如何生效

benchmark跳过C个前缀token的计算，为这些位置准备真实布局的synthetic HBM KV。它不会只缩短query却把所有历史K也缩短。

默认C向下按128-token block对齐；compact/mask还保证至少有140个新后缀token（短输入则C=0）。实际值保存为`cached_tokens_effective`和`cache_hit_rate_effective`。

| 输入 | compact140下实际prefix | 实际新token | 实际命中率 |
| --- | --- | --- | --- |
| L=8192，目标0% | 0 | 8192 | 0% |
| L=8192，目标50% | 4096 | 4096 | 50% |
| L=8192，目标99% | 7936 | 256 | 96.875% |
| L=8192，目标100% | 7936 | 256 | 96.875% |

`alignment_policy=exact`时无法精确满足的点返回INVALID_CONFIG。所有策略使用相同规范化C，不能一个策略按99%原值、另一个按block取整后比较。

该h不表示request hit rate、SSD/3FS命中率、NPU L2命中率或真实服务缓存命中率。

## SWA bounded replay与原生执行路径

`yoco.mode=compact`是真正裁剪decoder query行；`mask`保留全部query计算、只限制SWA可见域；`off`为关闭截断。三种模式分别报告。

keep=140必须覆盖128-token窗口与实际回退余量；默认MTP=5时实际回退至少6。虽然不执行decode，仍保留原生forward里的DSpark特征与cache preparation，不自行删除。

decoder第20层仍通过full-hidden分支更新完整新增区间的main/index KV，后续层复用。benchmark不能把模型改成“encoder完整计算，然后后20层所有算子仅输入140 token”。

chunked_suffix中，早期chunk可能只有一个decoder占位query；full_len始终是完整L，不能把当前chunk末尾当作整条序列结尾。不同chunk之间保留生成的state，下一轮trajectory开始前再恢复初始state。

## 时延与吞吐读法

| 指标 | 定义 |
| --- | --- |
| forward_p50/p95/p99_ms | 批计时：第一个 forward 前记 event，最后一个 forward 结束后记 event，中间不插入 sync/barrier/restore。取 8 rank 批总时长的最大值除以批内 forward 数；批量模式下各样本均等于该平均值，故 p50=p95=p99=mean、CV=0 |
| forward / forward_step | `forward` 是整条后缀（各 chunk 之和）的平均耗时，`forward_step` 是单 chunk 平均；single_forward 两者相同 |
| new_tokens_per_s | 有效新token总数 / 累计forward时间；吞吐分子不含命中前缀 |
| new_tokens_per_s_per_card | 上项除以8 |
| requests_per_s | 本轮逻辑请求数 / 累计forward时间 |
| logical_input_tokens_per_s | 可选，分子包含cached prefix，明确区别于new-token吞吐 |
| host_preparation_and_forward_ms | 每 rank 原始样本中的辅助 wall 成本，包含请求描述、同步和 forward；不与纯 forward 混排 |
| observed_moe_lb | 计时外读取原生强制均衡后的实际expert负载；不保证严格等于1 |

主计时包含原生wrapper、embedding、Engram、40层、内部metadata工作、CP通信、YOCO full-KV/compact和DSpark缓存准备；不含外部metadata构造、造数/状态恢复、sampler、模型加载或profile开销。批计时只在第一个 forward 前恢复一次 state 并同步，之后 `iterations` 次 forward 连续下发；默认 `warmup=0, iterations=10, rounds=1`。

fixed_global适合比较相同总工作量的延迟和吞吐。fixed_per_dp时总请求数随DP改变，输出延迟/吞吐Pareto。若数值差异落在测量噪声内，标为tie/不确定。

## 批量执行（一次加载、多个 case）

同一并行策略下逐个 case 重新加载模型（dummy 权重约 60s/次）在扫描大量 L×命中率时占主导。`batch.enabled=true` 时，launcher 把同一策略的 case 分组，每组只加载一次原生模型，随后在同样的 8 个 rank 进程内依次执行组内 case：每个 case 重新分配 KV cache/state、造数、计时和 profiling，结束后释放；模型、Engram buffer 和通信 workspace 保持常驻。

模型构建时 `max_num_batched_tokens` 会按 token 数预分配通信 workspace（实测约 128 KB/token，7936 token ≈ 0.95 GiB SymmBuffer），因此不能按整个扫描的最大 L 只加载一次：1M/低命中 case 的容量会拖垮全部 case。默认 `batch.capacity_grouping=power2`，按每个 case 需要的 `max_rows_per_rank` 向上取 2 的幂分桶，每组只在同一 2× 容量带内加载；`single` 表示每策略只按最大容量加载（可能 OOM）。完整扫描示例见 `configs/sweep_v41_full.json`（20 个 L × 11 个命中率，220 个 case 分为 13 组、共 13 次加载）。

容错：某个 case 在任一 rank 抛错（含 OOM）时，该 rank 写 `rank_*/error.json` 后退出；launcher 终止本组进程、把该 case 标为 FAILED/OOM，然后用剩余 case 重新加载并继续，不会中断整个扫描。整组模型加载失败时只把该组剩余 case 标为 FAILED 并继续下一组，不会影响其他容量带。每个 case 仍受 `runtime.case_timeout_s` 限制。

进度：stdout 按 `[完成数/总数]` 输出每个 case 的 `running`/`OK`/`FAILED` 与 `forward_p50_ms`；每组开始时打印 `loading model ... capacity=...` 和加载耗时。组级日志与 `ready_rank_*.json` 在 `groups/<strategy>_cap<capacity>_<stamp>/`，case 结果目录结构不变。`max_cases_per_group` 可限制单组 case 数（0 表示不限制），CLI 可用 `--batch-cases/--no-batch-cases` 覆盖开关。诊断时可用 `PREFILL950_FAIL_CASE=<case_id>[,<case_id>]` 注入 case 失败，验证失败续跑。

## Profiling与结果目录

默认在独立pass**只采 rank0**（`profile.ranks=rank0`；设为 `all` 才全采），所有 rank 仍执行相同的 profiling steps。采集阶段**不解析** trace：`profiler_index.json` 只记录 `status=COLLECTED` 和 `expected_active_forwards`，解析留到离线（可调用 `profiler.save_profile_index`）。默认 `wait=0, warmup=0, active=3`，即 profiling pass 共 3 次 forward。profiling 不会污染正式 timing samples。

profiling pass 与 timing 一样采用连续批执行：整批开始前只 restore state 并同步一次，3 次 forward 连续下发，中间不插入 `synchronize`/`barrier`/`restore`，避免 trace 中两次 forward 之间出现设备空泡；`profiler.step()` 只做 host 侧步进。

256K 用例 trace 起始处约 33ms 的空泡来自首个 forward 的 host metadata 构造（raw 与 compressed 两个 262144-token `SequenceData` 各约 15.5ms），发生在 `prefill950:native_model_forward` 范围之外。计时 pass 在 `start.record()` 之前构造首个 metadata，因此该空泡不计入单 forward 平均时延；profiling 开/关的正式 timing 差异在测量噪声内。

CANN 9.2 在本平台不支持对自定义 AscendC 内核导出 `Input Shapes`（`msprof --aicore-shape=on` 对这些内核仍返回 N/A，只有 aclnn/aten 算子带 shape）。因此 profiling 的 active steps 还会额外写入 `module_shapes.jsonl`，按模块记录输入 tensor 的 shape/dtype，作为 kernel 形状的补充数据；它不影响 kernel_details.csv 本身的 N/A。

```text
results/<run_id>/
  manifest.json / resolved_config.json / cases.jsonl
  summary.csv / summary.json / best_strategy.csv
  heatmap_latency.png / heatmap_throughput.png
  cases/<strategy>_L<length>_h<hit>_<timestamp>/
    case.json / result.json / rank_mapping.json
    rank_0/ ... rank_7/
      worker.log / native_contract.json
      samples.jsonl / outcome.json / module_shapes.jsonl
      graph_signature_0.json / graph_signature_1.json
      profiler_index.json         # 仅选中 rank（默认 rank0），采集后 status=COLLECTED
      profiler/...                # 原生 Ascend trace/summary，仅选中 rank
  doctor/doctor.json
  attempts/...                   # resume 时归档的失败尝试
```

表格按L、实际C、batch口径、YOCO、execution和其他固定参数分组，展示四策略p50/p95、new-token吞吐、HBM峰值、状态和profile入口。软件/固件/配置hash不同的run不会自动合并排名。

每个 case 独立启动八进程，复用原生 PG/rank 映射和 CP metadata 广播；CP follower 进入同步 benchmark 循环。模型加载、准备和退出都不计入 forward 时延。这样隔离 buffer/state，代价是扫描启动较慢。

OOM、单 rank 异常或超时会终止整组，继续后续 case。profile 缺失、缺少实际 NPU kernel 或 active forward 范围不完整时保留 timing 并标记 `timing_ok_profile_failed`，整次 collect 非零退出。结果写入采用临时文件原子替换。`--resume` 仅在 manifest 完全一致时跳过成功点；失败点整 case 重跑，原尝试移到 `attempts/`。

`peak_torch_*` 是 allocator 峰值；`minimum_device_free_bytes` 是准备后及每次 forward 后的设备空闲显存最小采样，包含持久非 torch 分配，不能当作连续 HBM 峰值。准备峰值、原生 KV/state 字节数和快照大小另存在 rank outcome 中。容量规划仅给出 block KV 下界，不保证完整模型能运行该点。

## 性能对齐的验收方式

复用代码只是第一层保证，还需在950上对照原生ModelRunner：检查metadata、kernel分支、次数、GEMM尺寸、量化layout、通信payload和YOCO完整KV分支；随后在相同计时边界做稳态对比。

设计目标为稳定点 CV≤3% 时，至少 3 轮×20 samples，median 偏差≤5%、p95 偏差≤10%。`compare` 检查相同环境/原生源码/模型及工作负载下的时延结果。独立原生基线需导出相同结果 schema；两次 benchmark run 的比较只证明两次测量一致，不能充当独立基线验收。

`graph_alignment_status=NATIVE_PATH_REPEATABLE` 表示两次新建原生 metadata 的 CPU 值与模块输入布局签名一致，**不是**与独立原生基线逐 kernel 对齐。`performance_alignment_status` 默认始终为 `NOT_MEASURED_AGAINST_EXTERNAL_REFERENCE`。仍须在 950 人工核对 trace 中的 FlashMLA、MegaMoE、YOCO full-KV 分支、shape、通信和事件子流依赖。dummy/real 数据依赖访存不同，分开报告。

## 其他命令与本地验证

```bash
# 不执行模型，列出规范化后的C/U、batch分配、chunk轨迹和容量预算。
python -m benchmarks.prefill_ascend950.cli plan --config /path/to/config.json

# 预检8卡、互联、软件版本、包来源、模型小文件及profiler能力。
python -m benchmarks.prefill_ascend950.cli doctor --devices 0,1,2,3,4,5,6,7 \
  --model-dir /data/models/DeepSeek-V4.1-Flash-1017

# 沿用配置，切换真实checkpoint；与dummy结果独立分组。
bash benchmarks/prefill_ascend950/collect.sh \
  --venv .venv-prefill950 --config /path/to/config.json \
  --devices 0,1,2,3,4,5,6,7 \
  --weights real --model-dir /data/models/DeepSeek-V4.1-Flash-1017 \
  --out results/prefill950_real

# 从已有raw samples重建报表，不重新执行forward。
python -m benchmarks.prefill_ascend950.cli summarize --run results/prefill950_smoke

# 对已有相同 schema 的独立原生基线与 benchmark 结果做显式对照。
python -m benchmarks.prefill_ascend950.cli compare \
  --reference results/native_reference --candidate results/prefill950_smoke \
  --out results/prefill950_comparison.json

# CPU 测试不导入 torch，也不要求 CANN/NPU。
python -m unittest discover -s tests/benchmarks/prefill_ascend950 -v

# 在目标机额外启用真实八卡测试（默认跳过）。
PREFILL950_HARDWARE_TEST=1 PREFILL950_MODEL_DIR=/data/models/DeepSeek-V4.1-Flash-1017 \
  python -m unittest discover -s tests/benchmarks/prefill_ascend950 -v
```

CPU 测试覆盖命中率边界、CP padding/rank 映射、完整 prompt metadata、状态恢复、全 rank 统计、失败清理及 trace 检查。当前未执行目标机测试；一键部署/采集接口已实现，950 可用性与性能验收状态仍为未验证。
