# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Environment controls for the portable Triton sparse MLA path."""

import os

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

_TRITON_MLA_SPARSE_ENV = "VLLM_TRITON_MLA_SPARSE"
_TRITON_MLA_SPARSE_TOPK_CHUNK_ENV = "VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE"
_TRITON_MLA_SPARSE_QUERY_CHUNK_ENV = "VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE"
_TRITON_MLA_SPARSE_HEAD_BLOCK_ENV = "VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE"
_TRITON_MLA_SPARSE_MATMUL_DECODE_ENV = "VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE"
_TRITON_MLA_SPARSE_SPLITKV_DECODE_ENV = "VLLM_TRITON_MLA_SPARSE_SPLITKV_DECODE"
_ENV_TRUE_VALUES = {"1", "true", "yes", "on"}
_ENV_FALSE_VALUES = {"0", "false", "no", "off"}

logger = init_logger(__name__)


def _optional_env_flag(name: str) -> bool | None:
    raw_value = os.getenv(name)
    if raw_value is None:
        return None
    value = raw_value.lower()
    if value in _ENV_TRUE_VALUES:
        return True
    if value in _ENV_FALSE_VALUES:
        return False
    logger.warning(
        "Ignoring unrecognized value %r for env var %s; expected one of %s. "
        "Falling back to platform default.",
        raw_value,
        name,
        sorted(_ENV_TRUE_VALUES | _ENV_FALSE_VALUES),
    )
    return None


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
    return _optional_env_flag(_TRITON_MLA_SPARSE_ENV)


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


def _positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        parsed = int(raw_value)
    except ValueError:
        logger.warning(
            "Ignoring non-integer value %r for env var %s; using default %d.",
            raw_value,
            name,
            default,
        )
        return default
    if parsed < 1:
        logger.warning(
            "Ignoring non-positive value %d for env var %s; using default %d.",
            parsed,
            name,
            default,
        )
        return default
    return parsed


def triton_sparse_mla_topk_chunk_size() -> int:
    return _positive_int_env(_TRITON_MLA_SPARSE_TOPK_CHUNK_ENV, 512)


def triton_sparse_mla_query_chunk_size() -> int:
    return _positive_int_env(_TRITON_MLA_SPARSE_QUERY_CHUNK_ENV, 256)


def triton_sparse_mla_head_block_size() -> int | None:
    raw_value = os.getenv(_TRITON_MLA_SPARSE_HEAD_BLOCK_ENV)
    if raw_value is None:
        return None
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning(
            "Ignoring non-integer value %r for env var %s.",
            raw_value,
            _TRITON_MLA_SPARSE_HEAD_BLOCK_ENV,
        )
        return None
    if value in (1, 2, 4):
        return value
    logger.warning(
        "Ignoring unsupported value %d for env var %s; expected one of (1, 2, 4).",
        value,
        _TRITON_MLA_SPARSE_HEAD_BLOCK_ENV,
    )
    return None


def triton_sparse_mla_matmul_decode_enabled() -> bool:
    configured = _optional_env_flag(_TRITON_MLA_SPARSE_MATMUL_DECODE_ENV)
    if configured is not None:
        return configured
    return current_platform.is_device_capability_family(120)


def triton_sparse_mla_splitkv_decode_enabled() -> bool:
    configured = _optional_env_flag(_TRITON_MLA_SPARSE_SPLITKV_DECODE_ENV)
    if configured is not None:
        return configured
    return False
