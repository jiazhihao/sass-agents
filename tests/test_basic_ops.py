"""Non-tensor smoke tests — a handful of common arithmetic / control ops.

Kept here as regression anchors while the tensor-heavy modules expand.
"""

from __future__ import annotations

import pytest

from ._util import (
    Encoder,
    assert_byte_exact,
    assert_roundtrip,
    requires_nvdisasm,
)


SAMPLES: list[tuple[str, str]] = [
    ("NOP", "18790000000000000000000000c00f00"),
    ("IMAD.MOV.U32 R4, RZ, RZ, RZ",      "247204ffff000000ff008e0700e20f00"),
    ("IMAD.MOV.U32 R0, RZ, RZ, 0x1",     "247400ff01000000ff008e0700ea0f00"),
    ("IMAD R2, R0, UR5, R5",              "247c02000500000005028e0f00e20f00"),
    ("IADD3 R0, P0, PT, R0, UR4, RZ",    "107c000004000000ffe0f10f00ca0f00"),
    ("MOV R13, UR6",                      "027c0d0006000000000f000800e20f00"),
    ("FMUL R10, R2, R10",                "20720a020a0000000000400000e20f00"),
    ("FFMA R3, R8, R11, 1.1641532182693481445e-10",
     "237403080000002f0b00000000e20f00"),
    ("EXIT",                              "4d790000000000000000800300ea0f00"),
]


@pytest.mark.parametrize("sass_text,cubin_hex", SAMPLES, ids=[s[0] for s in SAMPLES])
def test_encoder_matches_cubin(encoder: Encoder, sass_text: str, cubin_hex: str):
    assert_byte_exact(encoder, sass_text, cubin_hex)


@requires_nvdisasm
@pytest.mark.parametrize("sass_text,cubin_hex", SAMPLES, ids=[s[0] for s in SAMPLES])
def test_nvdisasm_roundtrip(encoder: Encoder, sass_text: str, cubin_hex: str):
    assert_roundtrip(encoder, sass_text, cubin_hex)
