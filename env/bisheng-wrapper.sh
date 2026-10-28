#!/bin/bash
# Ascend 950 prefill benchmark build wrapper.
# Host C++20 code is compiled by GCC 14 (Bisheng's device frontend lacks concepts);
# device .asc code goes to the stock CANN Bisheng with the missing float2bf16 alias
# mapped to the public _v2 entry point.  <limits> is force-included for .asc because
# FlashMLA uses std::numeric_limits but only includes <numeric>, which the device
# frontend does not transitively expand.
set -e

GCC14=/opt/openEuler/gcc-toolset-14/root/usr/bin
REAL_BISHENG=/usr/local/Ascend/cann/tools/bisheng_compiler/bin/bisheng

host_cpp=false
device_src=false
for arg in "$@"; do
    case "$arg" in
        *.cpp|*.cc|*.cxx) host_cpp=true ;;
        *.asc) device_src=true ;;
    esac
done

if $host_cpp; then
    args=()
    skip_next=false
    for arg in "$@"; do
        if $skip_next; then
            skip_next=false
            continue
        fi
        case "$arg" in
            --npu-arch=*|--cce-res-usage) continue ;;
            -mllvm) skip_next=true; continue ;;
        esac
        args+=("$arg")
    done
    exec "$GCC14/g++" "${args[@]}"
fi

extra=()
if $device_src; then
    # CANN 9.2's c_api/asc_simd.h no longer pulls in the ascendc_assert macro.
    extra=(-include limits -include utils/debug/asc_assert.h)
fi

exec "$REAL_BISHENG" \
    --gcc-toolchain=/usr/local/Ascend/cann/tools/hcc \
    --target=aarch64-target-linux-gnu \
    -U__cpp_concepts -D__cpp_concepts=0 \
    -Dasc_float2bfloat16_rn_v2_impl=asc_float2bfloat16_rn_v2 \
    -L/opt/openEuler/gcc-toolset-14/root/usr/lib64 \
    "${extra[@]}" \
    "$@"
