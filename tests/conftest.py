"""Shared pytest fixtures for the encoder test suite."""

from __future__ import annotations

import pytest

from ._util import Encoder, NVDISASM


@pytest.fixture(scope="session")
def encoder() -> Encoder:
    return Encoder()


requires_nvdisasm = pytest.mark.skipif(
    NVDISASM is None,
    reason="nvdisasm not on PATH — install CUDA toolkit to run round-trip tests",
)
