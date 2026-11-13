# DeepInfer 降频根因定位：MegaMoE 切分、extra experts 与 MFU 假设验证

生成时间：2026-11-05  
实验平台：Ascend950DT ×8，DP4CP2EP8，L=131072，HBM 4092 GB/s  
结论一句话：**降频是真实的，直接触发因素是 ~950 W 板级功耗/DVFS；可以排除 MegaMoE 切分、extra_experts 和 MoE MFU 更高。DeepInfer 的 AIV 累计工作量显著高于 ds41（GEMM epilogue、sparse attention），是“功率余量更小”的主要候选，但 AIV 单独把功耗打满尚未用 per-rail power 或单独降 AIV 的 A/B 直接证明。**

---

## 0. 结论摘要

1. **降频真实存在**。用两边 profiler 里同一张 `AICORE_FREQ` 表对比：
   - DeepInfer h0.8：均值 **1604.9 MHz**，p10=1544，33.9% 采样 <1600 MHz，最低 1419 MHz；
   - ds41 h0.8：均值 **1649.0 MHz**，p10=1650，只有 0.2% 采样 <1600 MHz；
   - DeepInfer h0.9：均值 **1618.0 MHz**（E0 为 1614.5），24.2% <1600；
   - ds41 h0.9：**1650 MHz 恒定**。
2. **切分不是解药**。DeepInfer h0.8 开 `DEEPINFER_MOE_CHUNK_TOKENS=8192` 后仍然降频，且更差：频率均值 **1579.3 MHz**，<1600 占比 48.7%，MoE 总时间 182.7→190.3 ms，forward 489.6→491.1 ms。
3. **h0.9 是切分假设的反例**。h0.9 每 die 只有 6592 token，小于 8192；DeepInfer 和 ds41 都只发一次 MegaMoE，DeepInfer 仍降到均值 1618 MHz，ds41 保持 1650 MHz。
4. **extra_experts 也不是原因**。DeepInfer `extra_experts=32→0` 后，h0.8 频率均值 1604.9→1601.9 MHz，h0.9 1618.0→1614.5 MHz；降频没有消失，端到端也没有变快（h0.8 489.6→493.1 ms）。
5. **“我们的算子 MFU 更高”不成立，至少 MoE 不是**。同一 MoE 形状下，DeepInfer/ds41 的单层 MegaMoE 时间几乎相同，理论 TFLOPS 约 200 vs 199（DeepInfer 仅高约 1%–3%，降频后反而低到约 186–188）。差距在 **AIV 累计执行时间**：
   - h0.9 每 forward AIV 累计执行时间：DeepInfer **261.0 ms / span 264.6 ms = 98.6%**（相对密度，非利用率）；
   - ds41：**209.0 ms / span 262.3 ms = 79.7%**；
   - AIC MAC 时间反而是 DeepInfer 更低：**173.3 vs 175.3 ms**。
6. **功耗与 DVFS 证据是直接的；AIV 的具体贡献是推断**：外部采样看到板级功率反复到 **949–950 W**，同时 AICORE_FREQ 下降；MegaMoE-only probe 只跑 MoE 也能到 950 W/降频。DeepInfer 的 FP8 GEMM epilogue（AIV +34.5 ms/forward）和 sparse attention（AIV +20.3 ms）使其 AIV 累计时间显著高于 ds41（相对 span 98.6% vs 79.7%），这是“更容易贴近功耗墙”的主要候选，但不能把 98.6% 当作 AIV 硬件利用率，也不能仅凭它证明 AIV 单独触发降频。
7. **ds41 并不是“永远不降频”**。把 ds41 h0.8 从 4 iterations 拉到 20 iterations 后，外部 `npu-smi` 采样也到 **944.1 W**，并出现 1425/1600/1625 MHz 的瞬时下降；只是它 4-iteration profiling 太短，没到持续触发点。差异是功率余量，不是“代码上有没有切分”。

---

## 1. 实验与口径

所有对比都是同一节点、同一组 8 卡、DP4CP2EP8、L=131072、`cache_source=synthetic_hbm`、HBM 限速 4092 GB/s。

| 编号 | 模型 | hit | 配置 | 说明 |
|---|---|---:|---|---|
| A | DeepInfer | 0.8 | E32，不切 MoE | 基线，profile 10 forward |
| B | DeepInfer | 0.8 | E32，`DEEPINFER_MOE_CHUNK_TOKENS=8192` | 临时 patch，仅实验 |
| C | DeepInfer | 0.8 | E0，不切 MoE | 关闭 32 个冗余专家 |
| D | DeepInfer | 0.9 | E32，不切 MoE | 反例：T=6592 < 8192（新跑） |
| E | DeepInfer | 0.9 | E0，不切 MoE | 反例 + 新跑端到端 |
| F | ds41 | 0.8 | chunk auto（8192），4 iters | 原始 ds41 口径 |
| G | ds41 | 0.8 | `DS41M_MOE_CHUNK=0`，4 iters | 反向 A/B |
| H | ds41 | 0.8 | chunk auto，20 iters | 持续负载测试 |
| I | ds41 | 0.9 | chunk auto，4 iters | 同 token 量对照 |
| J | 单算子 probe | — | DI MegaMoE-only，600 iters | T=13120/E416、T=13120/E384、T=8192/E416 |

说明：
- `AICORE_FREQ` 来自 msprof 的 `AICORE_FREQ` 表；DeepInfer 的 `ascend_pytorch_profiler_0.db` 和 ds41 的 `msprof_*.db` 中表结构完全相同，可直接对比。
- 功率来自外部 `npu-smi info -t common -i 0` 的 `NPU Real-time Power(W)`，0.5 s 采样。
- `AIV`/`MAC` 来自 op summary/kernel details 的 `aiv_time(us)`、`aic_mac_time(us)`，逐 forward 求和；两边算子集合不同，但总量和主要差异项可比。
- 本文的 TFLOPS 是近似值：`6*H*I*(T*topk/EP + T*shared)`，均匀路由假设。

---

## 2. 降频证据（同源频率表）

| 配置 | AIC 频率均值 | p10 | 最低 | <1600 MHz 占比 | 结论 |
|---|---:|---:|---:|---:|---|
| DeepInfer h0.8 E32 不切 | **1604.9** | 1544 | 1419 | **33.9%** | 持续降频 |
| DeepInfer h0.8 E32 切 8192 | **1579.3** | 1488 | 1313 | **48.7%** | 更差 |
| DeepInfer h0.8 E0 不切 | **1601.9** | 1535 | 1396 | **40.0%** | 仍降频 |
| DeepInfer h0.9 E32 | **1618.0** | 1575 | 1475 | **24.2%** | 仍降频 |
| DeepInfer h0.9 E0 | **1614.5** | 1563 | 1463 | **27.9%** | 仍降频 |
| ds41 h0.8 auto，4 iters | **1649.0** | 1650 | 1575 | **0.2%** | 基本不降 |
| ds41 h0.9 auto，4 iters | **1650.0** | 1650 | 1650 | **0.0%** | 不降 |

外部功率/频率采样进一步确认“功率上限”：

| 负载 | 峰值功率 | 高频段 curFreq 采样 |
|---|---:|---|
| DeepInfer h0.8 E0，10 forward continuous | **948.0 W** | 1650×6、1475×2、1600×1 |
| DeepInfer h0.9 E32，10 forward continuous | **942.6 W** | 1650×213、1600×1（npu-smi 瞬时；trace 均值 1618） |
| DeepInfer MegaMoE-only，T=13120/E416 | **950.0 W** | 1450×8、1650×2、1425×1 |
| DeepInfer MegaMoE-only，T=13120/E384 | **950.0 W** | 1400–1475 为主 |
| DeepInfer MegaMoE-only，T=8192/E416 | **950.1 W** | 1350–1500 |
| ds41 h0.8 auto，4 iters | 922.7 W | 1650×6 |
| ds41 h0.8 chunk=0，4 iters | 937.6 W | 1650×4 |
| ds41 h0.8 auto，20 iters | **944.1 W** | 1650×16、1425/1600/1625 各 1 |
| ds41 h0.9 auto，4 iters | 830.0 W | 1650×3 |

**解释**：949–950 W 反复精确出现，像是板级功耗上限。DeepInfer 在 h0.8 已经贴住上限并持续降频；ds41 4-iteration 只在 923–938 W，未持续越线，所以看起来“不降频”。20 iterations 后 ds41 也到 944 W，并开始出现瞬时下降。

---

## 3. 假设 1：ds41 切 MegaMoE、DeepInfer 不切，是否原因？

### 3.1 DeepInfer 主动切 8192 的 A/B

h0.8、T=13120：

| 指标 | 不切（A） | 切 8192（B） | 变化 |
|---|---:|---:|---:|
| MoE kernel 调用数/forward | 40 | 60（encoder 20×2 + decoder 20×1） | +20 |
| MoE 总时间/forward | **182.7 ms** | **190.3 ms** | **+4.2%** |
| forward steady span | 503.2 ms | 515.6 ms | +2.5% |
| 端到端 batch average | 489.6 ms | 491.1 ms | +0.3% |
| AIC 频率均值 | 1604.9 MHz | **1579.3 MHz** | **更低** |
| <1600 MHz 占比 | 33.9% | **48.7%** | 更差 |
| AIV累计/span（相对密度） | 99.5% | 99.7% | 仍高 |

切分没有降低 AIV 累计工作量，反而增加两次 kernel 启动、chunk 拷贝和同步开销；因此频率更低、MoE 更慢。**DeepInfer 上加 chunk 不能复现 ds41 的不降频。**

### 3.2 h0.9 反例

h0.9 的 T=6592，小于 8192，DeepInfer 和 ds41 都只发一次 MegaMoE。若“切分”是根因，两边都不切，应该都不降频；实际 DeepInfer 均值 1618 MHz（E0 1614.5），ds41 恒 1650。**切分假设在 h0.9 直接失效。**

### 3.3 ds41 反向 A/B

ds41 h0.8，CP2：

| 配置 | forward 稳态 | 峰值功率 | curFreq |
|---|---:|---:|---|
| chunk auto（8192） | 474 ms | 922.7 W | 1650（4 iters） |
| `DS41M_MOE_CHUNK=0` | 471 ms | 937.6 W | 1650（4 iters） |
| chunk auto，20 iters | 480 ms | 944.1 W | 1650 为主，偶发下降 |

- 切/不切对 ds41 端到端几乎无差异（471 vs 474 ms；20it 480 ms）。
- 不切时功率反而略高（937.6 vs 922.7 W），但短跑仍在 1650；20it 才出现偶发下降。
- ds41 **不切**时 SymmBuffer 是 T=131072，**13.6 GB**；切 8192 时是 982 MB。ds41 切分首先是**显存/缓冲容量约束**，不是“为了省功耗/防降频”。
- DeepInfer 的 SymmBuffer 按每 rank 实际 T 分配（h0.8 13120），本来就不需要为了 13.6 GB 去切。

### 3.4 MegaMoE-only probe

固定 MoE workload，连续 600 次调用，只跑 MegaMoE：

| shape | 前 10 次 | 后 30 次 | 时间涨幅 | 峰值功率 | curFreq |
|---|---:|---:|---:|---:|---|
| T=13120，E416 | 7.885 ms | 9.004 ms | **+14.2%** | 950 W | 1450 MHz 为主 |
| T=13120，E384 | 7.878 ms | 9.004 ms | **+14.3%** | 950 W | 1400–1475 MHz |
| T=8192，E416 | 4.996 ms | 5.972 ms | **+19.5%** | 950 W | 1350–1500 MHz |

- **T=8192 的 MegaMoE 单独跑也到 950 W、也降频**；切到 8K 以内并不能避开功耗墙。
- E384 与 E416 的曲线几乎一样；extra_experts 也不是 probe 的功耗差异来源。
- 时间涨幅与频率比例吻合：7.885→9.004 是 +14.2%，1650→1450 是 +13.8%；T8192 的 4.996→5.972 是 +19.5%，1650→1350 是 +22.2%。同一 workload 变慢，基本全是 DVFS，不是做了更多计算。

**结论：假设 1 不成立。**

---

## 4. 假设 2：我们的算子 MFU 更高，是否原因？

### 4.1 MoE 本身的 MFU / 单层时间

h0.9（每 die 约 6592–6624 token，EP8，topk6，shared1）：

| 模型/配置 | 每层 big MoE 时间（稳态前段） | 每层 big MoE 时间（降频后） | 理论 TFLOPS（前段/降频后） |
|---|---:|---:|---:|
| DeepInfer E416 | 4.03 ms | 4.39 ms | **202 / 186** |
| DeepInfer E0 | 4.03 ms | 4.33 ms | **202 / 188** |
| ds41 | 4.11 ms | 4.11 ms（不降） | **199 / 199** |

h0.8：

| 模型/配置 | 每层 MoE 时间 | 理论 TFLOPS |
|---|---:|---:|
| DeepInfer 不切，T=13120 | 7.86 ms（前段）/8.65 ms（降频后） | **207 / 188** |
| ds41 切 8192+4985 | 5.05+3.07=8.12 ms | **201** |

- 降频前 DeepInfer 的 MoE 只比 ds41 高约 **1%–3%**，不是数量级；
- 降频后反而比 ds41 低；
- 因此“我们的 MoE MFU 普遍更高导致降频”不成立。MoE 本身的效率不是主要异常点。

### 4.2 全 forward：AIC MAC 相近，AIV 累计执行时间差 25%

h0.9，逐 forward 平均（DeepInfer 取 step1–9，ds41 取 3 个完整 forward）：

| 指标 | DeepInfer | ds41 | 差异 |
|---|---:|---:|---:|
| forward span | 264.6 ms | 262.3 ms | +0.9% |
| AIC MAC 时间 | **173.3 ms** | **175.3 ms** | **-1.1%** |
| AIV 时间 | **261.0 ms** | **209.0 ms** | **+24.9%** |
| AIV 累计/ span（相对密度） | **98.6%** | **79.7%** | +18.9pp |
| MAC / span | 65.5% | 66.8% | -1.3pp |
| MegaMoE 时间 | 95.7 ms | 91.2 ms | +4.9% |

h0.8：

| 指标 | DeepInfer | ds41 |
|---|---:|---:|
| AIV 累计/ span（相对密度） | **99.5%** | **~80%**（profile 窗口 81.9%） |
| AIC MAC / span | 67.7% | 72.4%（窗口口径，偏乐观） |
| AIV 总时间 | 500.8 ms | 377.6 ms（窗口） |
| MegaMoE 时间 | 182.7 ms | 172.6 ms（旧 profile 窗口） |

**MAC 时间没有更高；高的是 AIV 累计执行时间。** 这里 `AIV累计/span` 是 `sum(aiv_time)/forward_span`，只是“每单位 wall time 内各 op 报告的 AIV active time 之和”，**不是 AIV 硬件利用率，也不是 AIV 功耗**。两边用同一 profiler 字段、同一 forward 划分，因此可做相对比较：DeepInfer 的 AIV/MAC 比约 **1.51**，ds41 约 **1.19**——同样的矩阵乘工作量，DeepInfer 多做了约 27% 的 vector 侧工作。这是功耗压力更大的一个强相关因素，但还不能单独证明它触发降频。

### 4.3 AIV 增量来自哪里（h0.9，每 forward）

| 类别 | DeepInfer AIV | ds41 AIV | 差 |
|---|---:|---:|---:|
| FP8 GEMM / QuantBatchMatmul | **45.3 ms** | **10.8 ms** | **+34.5 ms** |
| Attention core | **54.4 ms** | 34.1 ms | **+20.3 ms** |
| MegaMoE | 95.3 ms | 88.1 ms | +7.1 ms |
| MQA logits | 13.1 ms | 13.0 ms | ≈0 |
| HcPre/MhcPost/DynamicMxQuant/Rotary/RmsNorm/TensorMove 合计 | 37.5 ms | 47.9 ms | -10.4 ms |
| 其他 | ~15 ms | ~15 ms | ≈0 |
| **合计** | **260.5 ms** | **209.0 ms** | **+51.5 ms** |

两个主要异常：
1. **DeepGEMM FP8 GEMM 的 AIV epilogue**：两边 GEMM 的 MAC 时间相近（DI 45.5 ms vs ds41 QuantBatchMatmul+fp8 GEMM 约 46.1 ms），但 DeepInfer 的 AIV 时间多 **34.5 ms**；ds41 的 QuantBatchMatmul 走 AIC，AIV 只有 0.38 ms。
2. **Sparse attention**：DeepInfer sparse_fwd 的 AIV 54.4 ms，ds41 Mqsmla 34.1 ms；MAC 时间反而 DI 少 1.2 ms。也就是说 attention 做了同样多的矩阵乘，但 vector 侧多花 20 ms。

这不是“MFU 更高”，而是 **同样 MAC 工作量下 AIV 辅助开销更高**，使总功耗更容易贴近上限。

### 4.4 这个 AIV 公式能说明什么、不能说明什么

`aiv_time(us)` 是 profiler 对每个 op 给出的 AIV 执行时间，`sum(aiv_time)` 是这些 op 的 AIV 时间累计值。它回答的是“这个 op/这段 forward 在 AIV 上花了多少执行时间”，不是：

- **AIV 硬件利用率**：没有除以 AIV 核数、block 并行度或 kernel 重叠；
- **AIV 功耗**：功耗还取决于指令 mix、频率/电压、clock gating、AIC/AIV/HBM 并行；
- **AIV 单独触发降频**：本轮没有 per-rail 功耗，也没有“只降低 AIV、保持 MAC 不变”的 A/B。

因此 `AIV累计/span = 98.6%` 只能作为 **相对工作量密度** 的证据：DeepInfer 在同样 wall time 内累计了更多 AIV 执行时间，比 ds41 更可能把功耗顶到上限。真正的因果证据仍然是：

1. 外部 `npu-smi` 反复看到 **949–950 W**；
2. 同一时间 `AICORE_FREQ` 下降；
3. MegaMoE-only probe 的同一 workload 时间涨幅与 1650→1450 MHz 吻合，证明 DVFS 真实发生。

probe 也说明：功耗墙不是“只有非 MoE 的 AIV 才会触发”，MegaMoE 本身（AIC+AIV+访存）就能触发。AIV 与非 MoE 算子在整模型中的额外贡献，目前是 **强相关 + 机制推断**，不是已被单独隔离的因果。

---

## 5. 根因链

1. DeepInfer 的 FP8 GEMM epilogue + sparse attention + 其他 vector 算子，使其 AIV 累计执行时间从 ds41 的相对密度约 80% 提升到约 99%（非硬件利用率）。
2. 板级功率在 h0.8/h0.9 达到 **949–950 W** 上限。
3. 硬件 DVFS 把 AIC 从 1650 降到约 1450–1600 MHz；MoE/attention 等 kernel 时间按频率比例变长。
4. 因为 AIV 本身没有变少，降频后 AIV 仍接近满占，形成“高功率→降频→时间变长”的稳态；h0.8 频率均值 1605，h0.9 1618，h0.96+ 的 workload 更小才回到 1650。

`npu-smi` 外部采样与 profile 的 `AICORE_FREQ` 方向一致；probe 的时间涨幅与频率比例吻合，排除“只是 profiler 统计 artifact”的解释。**但 AIV 的具体功耗占比和单独因果，仍需 per-rail power 或 AIV 降载 A/B 复核。**

---

## 6. 性能影响与旧数据异常

- 本次同条件复测：
  - h0.9：DeepInfer E0 **262.3 ms**、E32 **264.5 ms**，ds41 **271.0 ms**；DeepInfer 反而更快。
  - h0.8：DeepInfer **489.6 ms**，ds41 auto **474 ms** / chunk=0 **471 ms** / 20it **480 ms**；ds41 略快。
- 所以至少这两个 case 不是“DeepInfer 数量级失败”。降频带来的损失约 **2%–4%**，不是造成数十个百分点差异的主因。
- **旧 summary 里 h0.9=469.4 ms 与同 case 的 profiling step span 256–269 ms 自相矛盾**，是采集异常（很可能计时 forward 撞上 profiler/缓存状态），不能作为 DeepInfer vs ds41 的基准；本文用新跑的 264.5/262.3 ms 和 profile span。
- 旧 ds41 h0.8 profile forward 515.8 ms 也来自 rank0 prof 轮；本次 4it/20it 稳态为 471–480 ms。跨轮对比时要统一“是否 prof、rank、iters”。

### 6.1 “DeepInfer h0.9 失败”这个印象主要来自异常值

旧汇总里 h0.9：DeepInfer 469.4 ms vs ds41 约 271 ms，看起来慢 73%。但同一个 DeepInfer case 的 10 次 forward profiling span 只有 256–269 ms，新跑 E32/E0 分别是 264.5/262.3 ms；469.4 ms 与这些独立测量都矛盾，应该重采/剔除，而不是归因于降频。降频实测只解释约 2%–4%，不足以造成 73% 的差距。

---

## 7. 建议

1. **不要为了降频给 DeepInfer 加 MegaMoE chunk**。实测 MoE +4.2%，forward +2.5%，频率更低；ds41 反向 A/B 也没有因切分而变快。
2. **extra_experts 保持 32**。E0 没有降频收益，h0.8 还慢 0.7%；没有证据表明它是功耗/降频原因。
3. **优先降低 AIV 辅助开销**：
   - DeepGEMM FP8 GEMM 的 vector epilogue（最大单项 +34.5 ms/forward）：评估改用 vendor QuantBatchMatmul、AIC epilogue、或把 per-token cast/store 融合进 cube 路径；
   - sparse attention 的 AIV 路径（+20.3 ms）：和 ds41 的 Mqsmla 对比，减少 vector 侧处理；
   - 这两项若各降一半，AIV 可下降约 27 ms/forward，给 ~950 W 上限留出余量。
4. **验收指标**：不要只看 `AICORE_FREQ` 均值；用 MegaMoE-only probe（T=8192 和 T=13120）看 600 次迭代的前后时间漂移。当前 T=13120/E416 漂移 **+14.2%**，T=8192/E416 **+19.5%**；如果优化后这个漂移明显下降，才算真正解决功耗问题。
5. 如果短期只能在现有代码上跑，接受 h0.8/h0.9 约 2%–4% 的 DVFS 损失；它比“加 chunk 带来的 2.5% forward 恶化”更划算。

---

## 8. 复现入口

- DeepInfer 基线/chunk A/B：
  - `/mnt/xds/sfs/y30048898/DeepInfer/results/chunk_exp_h08_baseline_v2_20261105_bw4092`
  - `/mnt/xds/sfs/y30048898/DeepInfer/results/chunk_exp_h08_chunk8192_v2_20261105_bw4092`
- DeepInfer extra_experts=0：
  - `/mnt/xds/sfs/y30048898/DeepInfer/results/extra0_h08_h09_20261105_bw4092`
- ds41 h0.8 auto / chunk=0 / 20it：
  - `/mnt/xds/sfs/l00972766/tmp/diag_ds41_h08_chunkauto`
  - `/mnt/xds/sfs/l00972766/tmp/diag_ds41_h08_chunk0`
  - `/mnt/xds/sfs/l00972766/tmp/diag_ds41_h08_20it`
- ds41 h0.9：
  - `/mnt/xds/sfs/l00972766/tmp/diag_ds41_h09_chunkauto`
  - 旧 profile：`/mnt/xds/sfs/l00972766/grid_prof_1103/prof_cp2/cp2dp4ep8_L131072_hit0.9_batch1_replayon_prefill`
- MegaMoE-only probe 日志：
  - `/tmp/mega_probe_13120_e416.log`
  - `/tmp/mega_probe_13120_e384.log`
  - `/tmp/mega_probe_8192_e416.log`
  - 功率采样：`/tmp/npu_common_megamoe_13120.log`、`/tmp/npu_common_mega_more.log`

数据表：`moe_downclock_root_cause_20261105.xlsx`（本目录）。
  
实验用的 DeepInfer MegaMoE chunk patch 已在实验后回滚到原始 `mlp.py`；实验版留档在 `results/chunk_exp_h08_chunk8192_v2_20261105_bw4092/mlp.py.chunk_experiment`，仅用于复现，不在默认代码路径中。

---

## 9. 口径限制

- 功率是板级实时值、0.5 s 采样，不是 per-op 能耗；峰值可能被采样错过。
- AIV/AIC 时间是 profiler 的 per-op 汇总，跨实现比较的是总量和主要差异，不是逐指令效率。
- ds41 20it 的外部频率是 `npu-smi` 瞬时值，和 msprof `AICORE_FREQ` 不是同一采样机制；但方向一致。
- TFLOPS 用均匀路由近似，绝对值有误差；用于同形状相对比较足够。
- “AIV 高 → 功耗高 → 降频”是跨模型对照 + MegaMoE-only probe 的推断；`sum(aiv_time)/span` 不是 AIV 利用率或功耗。本轮没有 per-rail power，也没有“只降低 AIV 工作量”的 A/B，后续优化时建议用 probe 漂移和实际功耗复核。
