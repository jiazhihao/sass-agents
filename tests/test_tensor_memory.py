"""Tensor-memory load/store family.

  LDTM.x{2,4,8,16,32,64,128}
  LDTM.16dp256bit.x{2,4,8,16}
  STTM.x{2,4,8,...} + STTM.16dp128bit.x16
  LDSM.16.{M,MT}88.{2,4,8}
  STSM.16.{M,MT}88.{2,4}
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
    # tmem loads (various widths)
    ("LDTM.x2 R4, tmem[UR4]",          "ee7904000400000000000c0800220e00"),
    ("LDTM.x8 R4, tmem[UR27]",         "ee7904001b00000000001c0800620e00"),
    ("LDTM.x16 R4, tmem[UR4]",         "ee790400040000000000240800220f01"),
    ("LDTM.x32 R4, tmem[UR5]",         "ee7904000500000000002c0800642e00"),
    # 16dp256bit tmem loads
    ("LDTM.16dp256bit.x2 R12, tmem[UR5]", "ee790c000500000000000a0800e20f00"),
    ("LDTM.16dp256bit.x4 R84, tmem[UR7]", "ee795400070000000000120800e42f00"),
    ("LDTM.16dp256bit.x8 R36, tmem[UR4]", "ee7924000400000000001a0800640e00"),
    ("LDTM.16dp256bit.x16 R68, tmem[UR5]","ee794400050000000000220800a20e00"),
    # tmem stores
    ("STTM.x2 tmem[UR28], R134",        "ed790000860000001c000c0800e20300"),
    ("STTM.x8 tmem[UR9], R12",          "ed7900000c00000009001c0800e20f00"),
    ("STTM.x32 tmem[UR9], R164",        "ed790000a400000009002c0800e20f00"),
    ("STTM.16dp128bit.x16 tmem[UR16], R36", "ed790000240000001000200800e20f00"),
    # Shared-memory matrix (LDSM / STSM)
    ("LDSM.16.M88.4 R24, [R18]",        "3b781812000000000002000000a20200"),
    ("LDSM.16.MT88.4 R108, [R0+UR13+0x4400]",
                                         "3b786c000d0044000042000800220700"),
    ("STSM.16.MT88.4 [R4], R28",        "447800041c0000000042000000e20300"),
]


@pytest.mark.parametrize("sass_text,cubin_hex", SAMPLES, ids=[s[0] for s in SAMPLES])
def test_encoder_matches_cubin(encoder: Encoder, sass_text: str, cubin_hex: str):
    assert_byte_exact(encoder, sass_text, cubin_hex)


@requires_nvdisasm
@pytest.mark.parametrize("sass_text,cubin_hex", SAMPLES, ids=[s[0] for s in SAMPLES])
def test_nvdisasm_roundtrip(encoder: Encoder, sass_text: str, cubin_hex: str):
    assert_roundtrip(encoder, sass_text, cubin_hex)
