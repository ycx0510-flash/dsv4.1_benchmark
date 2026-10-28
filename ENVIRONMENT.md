# 运行环境与本地补丁（2026-10-27 存档）

## 容器/硬件

| 项 | 值 |
| --- | --- |
| 容器 | `y30048898_ds_benchmark_1027`（host `liteserver-hps-tokens-001-00003`，容器 ID `b255c474…`） |
| 系统 | openEuler 24.03 (LTS-SP4)，kernel 6.6.0-72.0.1.76.r1561_118.hce3.aarch64 |
| NPU | 8 × `Ascend950DT_9582`，单卡 torch 可见 91.75 GiB，8 卡两两 `UB` 互联 |
| CANN | 9.2.0（`V100R001C25B114`），Bisheng / HCCL / ATB / NNAL |
| 驱动 | `npu-smi` 25.1.rc2，driver 25.1.rc2（ascendhal 7.35.23） |
| 额外系统包 | `elfutils-devel 0.190`（deep_jit 编译期需要 `elfutils/libdwfl.h`） |

## Python 环境

venv：`/mnt/xds/sfs/y30048898/DeepInfer/.venv-prefill950`（Python 3.12.16）

| 包 | 版本 |
| --- | --- |
| torch | 2.13.0+cpu |
| torch_npu | 2.13.0rc1（模块内 `__version__` 为 `2.13.0.rc1`） |
| tilelang / tile_kernels | 0.1.15 / 2.1.0 |
| deep_gemm | 0.1.1+3a35326 |
| deep_ep | **1.0.0+local**（见下方补丁，原 1.0.0+3b25377 备份于 `/tmp/wheel-backup/`） |
| flash_mla | 1.0.0+2e5429f |
| deep_select | 1.0.0+bfa4507.20261020.173557 |

DeepInfer 基线：`553741557a57df44a46fb3ab72b6aafeaf945337`（工作区含本地补丁，见下）。
模型：`/mnt/xds/sfs/DeepSeek-V4.1-Flash/DeepSeek-V4.1-Flash`，`config.json` 规范化 SHA256 = `d7637228…669d`（与 benchmark 锁定 revision 一致）；备选 `/mnt/xds/sfs/DeepSeek-V4.1-Flash-1017/v1`。

## 本地补丁

本存档包含两组补丁（`patches/`）：

1. `deepinfer-engram-cpu-load.patch`
   - `deepinfer/environ.py`：`DEEPINFER_ENGRAM_FORCE_GPU_STORAGE` 支持显式 `0/false/off/no`。
   - `deepinfer/model_executor/models/deepseek.py`：`DEEPINFER_ENGRAM_CPU_LOAD=1` 时加载期 Engram 表放 CPU。
   - `deepinfer/model_executor/models/v4_modules/engram.py`：NPU 上 CPU 权重直接交给 `EngramBuffer.write()`。
2. `deepep-ascend-engram-cpu-source.patch`
   - `vendor/DeepEP-Ascend/csrc/buffers/engram.hpp`：`write()` 接受 CPU 源，走 `ACL_MEMCPY_HOST_TO_DEVICE`。

应用方式（在 DeepInfer 根目录）：

```bash
git apply patches/deepinfer-engram-cpu-load.patch
git -C vendor/DeepEP-Ascend apply ../dsv4.1_benchmark/patches/deepep-ascend-engram-cpu-source.patch
# 然后重编 deep_ep：
cd vendor/DeepEP-Ascend && python setup.py bdist_wheel --dist-dir ../../build/ascend-aarch64/vendor-wheels
python -m pip install --force-reinstall --no-deps ../../build/ascend-aarch64/vendor-wheels/deep_ep-*.whl
```

## JIT 编译器加固

DeepGEMM/DeepEP 的 `deep_jit` 用 `$ASCEND_HOME_PATH/bin/bisheng` 直接编译运行时内核，CANN 9.2 下会因 `ascendc_assert` 未定义和 `-std=c++20` + GCC 12 头文件冲突失败。解决办法：

- `env/bisheng-wrapper.sh`：包装真实 bisheng，`.asc` 增加 `-include utils/debug/asc_assert.h`、`-include limits`、`-U__cpp_concepts -D__cpp_concepts=0`、`-Dasc_float2bfloat16_rn_v2_impl=…`，并指定 `--gcc-toolchain`。
- `env/make-shim.sh`：生成一个只改 `bin/bisheng` 指向、其余全部软链到真实 CANN 的 shim 目录。
- 运行时设置 `PREFILL950_JIT_ASCEND_HOME=<shim>`，worker 会在子进程里把它设为 `ASCEND_HOME_PATH`（只影响 benchmark worker）。

## 运行参数

```bash
cd /mnt/xds/sfs/y30048898/DeepInfer
source build/prefill950-tools/prefill950-env.sh      # GCC14、SOC_VERSION、本地 tmp/cache
export PREFILL950_ALLOW_LOCAL_SOURCE=1               # 允许已记录的本地补丁通过 doctor
export PREFILL950_JIT_ASCEND_HOME=$PWD/build/prefill950-tools/ascend-toolkit-shim
```

## Benchmark 代码相对初始版本的改动

- `doctor.py`：torch_npu 版本字符串归一化（`2.13.0.rc1` vs `2.13.0rc1`）；新增 `PREFILL950_ALLOW_LOCAL_SOURCE` 记录式放行。
- `config.py`：新增 `runtime.engram`（默认 true）；`engram=true` 时默认注入 `DEEPINFER_ENGRAM_CPU_LOAD=1`。
- `worker.py`：dummy 模式传 `skip_load`；按 `runtime.engram` 置空 Engram；JIT shim 环境变量。
- `alignment.py`：签名时忽略 forward/backward hook 注册表（诊断自身产生的 hook id 会变化）。
- `README.md`：补充 Engram 开/关与 CPU 暂存说明。

> 注意：本地补丁使 `deepinfer` 跟踪文件与 `vendor/DeepEP-Ascend` 子模块不再是纯净基线。`doctor.json` / `manifest.json` 仍记录文件 SHA256 与子模块 dirty 状态，带补丁的运行不会与干净基线的结果混排。
