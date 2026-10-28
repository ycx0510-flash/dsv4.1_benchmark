# 64K / 97% 命中时延记录（2026-10-27）

工作负载：`DP8CP1EP8`，`L=65536`，`h=0.97` → 实际 `C=63488`、`effective=0.96875`、新 token `2048/请求`、`G=8`（共 16384 新 token）。
计时口径：`native_model_forward`，每 step 取 8 rank 的 NPU event span 最大值；5 warmup + 20 iterations × 3 rounds = 60 samples；profiling 为独立 pass，不计入计时。

| Engram | profiling | forward p50 | forward p95 | CV | new tok/s | 单卡 tok/s | 设备最低剩余 | case 墙钟 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 开 | 开 | 116.64 ms | 117.47 ms | 0.51% | 140,496 | 17,562 | 15.62 GiB | 91.1 s |
| 开 | 关 | 117.03 ms | 118.37 ms | 0.65% | 139,933 | 17,492 | 15.62 GiB | 77.1 s |
| 关 | 开 | 113.69 ms | 114.32 ms | 0.34% | 144,043 | 18,005 | 40.37 GiB | 95.1 s |
| 关 | 关 | 116.13 ms | 116.52 ms | 0.22% | 141,067 | 17,633 | 40.37 GiB | 51.5 s |

参考：同机 `L=8192 / h=0.5 / Engram 关 / profiling 开`（新 token 4096/请求）p50 = 178.91 ms、p95 = 180.70 ms、183,454 new tok/s。

## 观察

- Engram 开/关（同为 profiling 关）：117.03 vs 116.13 ms，约 **+0.8%**；Engram 只涉及 2/40 层，主要成本仍是 MoE 与 MLA attention。
- profiling 开/关不影响计时样本（独立 pass），差异属于跨运行噪声。
- 跨运行 p50 波动约 2%（例如“Engram 关 + profiling 开”113.69 ms 落在下沿），小于约 2% 的差异不应解读为真实差别。
- p99 在 60 样本下标记为不可靠（`p99_reliable=false`），对外只使用 p50/p95。

## 结果目录

```text
results/prefill950_64k_h097_engram_default/      # 开 + 开（runtime.engram=true 默认路径）
results/prefill950_64k_h097_engram_noprofile/    # 开 + 关
results/prefill950_64k_h097_noengram/            # 关 + 开
results/prefill950_64k_h097_noengram_noprofile/  # 关 + 关
```
