# dsv4.1_benchmark

DeepSeek V4.1 Flash prefill benchmark on 8×Ascend 950 (DP8CP1EP8 / DP4CP2EP8 / DP2CP4EP8).

- Benchmark source: [`benchmarks/prefill_ascend950/`](benchmarks/prefill_ascend950/) — see its README and DESIGN for the full contract
- Runtime environment and local patches: [`ENVIRONMENT.md`](ENVIRONMENT.md), [`patches/`](patches/), [`env/`](env/)
- 64K / 97% hit latency records: [`RESULTS-64K-h097.md`](RESULTS-64K-h097.md)

## Quick start

```bash
cd /mnt/xds/sfs/y30048898/DeepInfer
source build/prefill950-tools/prefill950-env.sh
export PREFILL950_ALLOW_LOCAL_SOURCE=1
export PREFILL950_JIT_ASCEND_HOME=$PWD/build/prefill950-tools/ascend-toolkit-shim

python -m benchmarks.prefill_ascend950.cli run \
  --config benchmarks/prefill_ascend950/configs/smoke.json \
  --devices 0,1,2,3,4,5,6,7 \
  --model-dir /mnt/xds/sfs/DeepSeek-V4.1-Flash/DeepSeek-V4.1-Flash \
  --out results/prefill950_smoke
```

Timing uses one NPU event pair around `iterations` back-to-back forwards (default 10, no sync between), reporting the mean; profiling is a separate pass. Case directories are named `<strategy>_L<length>_h<hit>_<timestamp>`.

## Notes

- `runtime.engram=true` (default) stages the Engram tables on CPU and writes the per-rank 1/8 shard into the device `EngramBuffer` (device-side cross-rank fetch).
- Profiling defaults to Level1: `kernel_details.csv` carries the full 49-column CANN table (shapes for aclnn/aten ops, AIC/AIV metrics for all kernels). Custom AscendC kernels do not expose input shapes on this CANN build; `module_shapes.jsonl` records module-level shapes as a companion dataset.
