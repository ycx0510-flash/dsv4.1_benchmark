"""Native sequence and cache preparation; accelerator imports stay in functions."""

from __future__ import annotations

from typing import Any

from .config import digest


def block_table(case: dict, request: int) -> list[int]:
    count = (case["context_length"] + 127) // 128
    return list(range(1 + request * count, 1 + (request + 1) * count))


def cache_slots(case: dict, request: int, compressed_block_size: int) -> range:
    """Reserve block zero and disjoint physical blocks, including last-block padding."""
    table = block_table(case, request)
    return range(table[0] * compressed_block_size, (table[-1] + 1) * compressed_block_size)


class NativeSequences:
    def __init__(self, case: dict, seed: int, global_request_start: int = 0) -> None:
        import numpy as np

        from deepinfer.model_executor.models.v4_modules.engram import CompressedTokenizer

        self.case = case
        self.tokens = [
            np.random.default_rng(np.random.SeedSequence([seed, global_request_start + i])).integers(
                0, len(CompressedTokenizer.lookup_table), size=case["context_length"], dtype=np.int32
            )
            for i in range(case["per_dp_batch_size"])
        ]
        self.compressed = [CompressedTokenizer.encode(t) for t in self.tokens]

    def metadata(self, step: dict) -> list:
        from deepinfer.sampling_params import SamplingParams
        from deepinfer.sequence import SequenceData, SequenceMetadata

        # The entire prompt (not a q-token slice) is required by YOCO and Engram.
        return [
            SequenceMetadata(
                request_id=f"benchmark-{i}",
                seq_id=i,
                is_prompt=True,
                seq_data=SequenceData(raw),
                compressed_seq_data=SequenceData(compressed),
                multi_modal_data=None,
                sampling_params=SamplingParams(temperature=0),
                block_table=block_table(self.case, i),
                seqlen_q=step["seqlen_q"],
                seqlen_k=step["seqlen_k"],
                last_output_token=0 if step["seqlen_k"] == step["full_len"] else -1,
                last_accepted_token=[],
                last_next_n_token=[],
                mtp_info_write_slot=0,
                state_id=i + 1,
            )
            for i, (raw, compressed) in enumerate(zip(self.tokens, self.compressed, strict=True))
        ]


def seed_cache(engine: Any, case: dict, seed: int, global_request_start: int) -> dict:
    import torch

    from deepinfer.model_executor.layers.tilelang_ops.cache_common import FP4_INDEX_KV_FORMAT, V41_FORMAT, V41_FP4_FORMAT
    from deepinfer.model_executor.layers.tilelang_ops.fused_transform import (
        CastArgs,
        KVCacheOutArgs,
        fused_norm_rope_hadamard_cast_store_kvcache,
    )

    engine.gpu_blocks.zero_()
    engine.gpu_states.zero_()
    seen, views = set(), []
    # Native cache views preserve shared producer/consumer aliases and physical strides.
    for layer, cache in enumerate(engine.gpu_cache.cache_list):
        for field, layout in (("ca_cache", V41_FP4_FORMAT), ("ica_cache", FP4_INDEX_KV_FORMAT), ("swa_cache", V41_FORMAT)):
            tensor = getattr(cache, field, None)
            if tensor is None:
                continue
            key = (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()))
            if key in seen:
                continue
            seen.add(key)
            if tensor.ndim != 4 or tensor.shape[-1] != layout.bytes_per_token:
                raise RuntimeError(f"Unexpected native {field} layout: {tuple(tensor.shape)}")
            views.append(
                {"layer": layer, "field": field, "shape": list(tensor.shape), "stride": list(tensor.stride()), "format": layout.note}
            )
            for request in range(case["per_dp_batch_size"]):
                # A logical request receives identical synthetic data across CP
                # replicas and strategies, without changing native routing RNG.
                generator = torch.Generator(device=tensor.device)
                generator.manual_seed(int(digest([seed, global_request_start + request, layer, field])[:15], 16))
                if field == "swa_cache":
                    slots = range((request + 1) * tensor.shape[1], (request + 2) * tensor.shape[1])
                else:
                    slots = cache_slots(case, request, tensor.shape[1])
                for start in range(slots.start, slots.stop, 2048):
                    stop = min(start + 2048, slots.stop)
                    mapping = torch.arange(start, stop, dtype=torch.int32, device=tensor.device)
                    values = torch.randn(
                        (stop - start, layout.num_elements), dtype=torch.bfloat16, device=tensor.device, generator=generator
                    )
                    # Quantize real finite BF16 values through the production store kernel,
                    # rather than inventing FP4/FP8 bytes or invalid scale encodings.
                    fused_norm_rope_hadamard_cast_store_kvcache(
                        values, cast_args=CastArgs.from_cache_format(layout), out_args=KVCacheOutArgs(tensor, mapping, layout=layout)
                    )
    torch.cuda.synchronize()
    return {
        "views": views,
        "block_bytes": engine.gpu_blocks.numel(),
        "state_bytes": engine.gpu_states.numel(),
        "seeded_region": "all allocated request blocks and SWA ring; compressed partial states zero",
        "prefix_tokens": case["cached_tokens_effective"],
        "reserved_block": 0,
        "reserved_state": 0,
    }


class StateSnapshot:
    def __init__(self, engine: Any) -> None:
        self.live = engine.gpu_states
        self.saved = self.live.clone()

    def restore(self) -> None:
        # Outside model timing; only at trajectory boundaries, never between chunks.
        self.live.copy_(self.saved)
