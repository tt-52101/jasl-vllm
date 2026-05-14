# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Environment controls for the portable Triton sparse MLA path."""

import torch

import vllm.envs as envs
from vllm.platforms import current_platform


def _is_sm12x_device(device: torch.device) -> bool:
    if not current_platform.is_cuda():
        return False
    index = (
        device.index
        if device.index is not None
        else torch.accelerator.current_device_index()
    )
    capability = current_platform.get_device_capability(device_id=index)
    return capability is not None and capability[0] == 12


def triton_sparse_mla_configured() -> bool | None:
    return envs.VLLM_TRITON_MLA_SPARSE


def is_triton_sparse_mla_enabled_for_platform() -> bool:
    configured = triton_sparse_mla_configured()
    if configured is not None:
        return configured
    return current_platform.is_device_capability_family(120)


def is_triton_sparse_mla_enabled(device: torch.device) -> bool:
    configured = triton_sparse_mla_configured()
    if configured is not None:
        return configured
    return _is_sm12x_device(device)


def triton_sparse_mla_topk_chunk_size() -> int:
    return envs.VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE


def triton_sparse_mla_query_chunk_size() -> int:
    return envs.VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE


def triton_sparse_mla_head_block_size() -> int | None:
    value = envs.VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE
    if value in (1, 2, 4):
        return value
    return None


def triton_sparse_mla_matmul_decode_enabled() -> bool:
    configured = envs.VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE
    if configured is not None:
        return configured
    return current_platform.is_device_capability_family(120)


def triton_sparse_mla_splitkv_decode_enabled() -> bool:
    return envs.VLLM_TRITON_MLA_SPARSE_SPLITKV_DECODE


def _uses_speculative_decoding(vllm_config) -> bool:
    """True iff ``vllm_config`` requests speculative decoding (e.g. MTP).

    Spec decode forces ``query_len > 1`` on the decode hot path, which the
    Triton sparse MLA cudagraph capture cannot represent today. The smart
    default keeps cudagraphs ON for the no-MTP path (where the decode is
    pure ``query_len == 1`` and cudagraph capture is safe) and OFF for any
    spec-decode configuration so we don't trip the capture-time shape
    assertions.
    """
    if vllm_config is None:
        return False
    spec_config = getattr(vllm_config, "speculative_config", None)
    if spec_config is None:
        return False
    num_spec_tokens = getattr(spec_config, "num_speculative_tokens", 0) or 0
    return num_spec_tokens > 0


def triton_sparse_mla_cudagraphs_allowed(vllm_config) -> bool:
    """Smart default for whether to enable cudagraph capture on the Triton
    sparse MLA path.

    Returns ``False`` when speculative decoding is active (the capture path
    cannot represent ``query_len > 1``); otherwise returns ``True``. The
    caller (``flashmla_sparse`` / ``sparse_swa`` backends) interprets a
    ``True`` return as "no kill-switch needed — defer to the metadata
    builder's normal cudagraph support level".
    """
    return not _uses_speculative_decoding(vllm_config)
