"""Tensor-core copy / barrier / atomic-find-and-set family.

  UTCCP.T.S.* (tensor copy)
  UTCBAR / UTCBAR.MULTICAST / UTCBAR.2CTA.MULTICAST
  UTCATOMSWS.AND / UTCATOMSWS[.2CTA].FIND_AND_SET.ALIGN
  ACQBULK
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
    # Tensor copy
    ("UTCCP.T.S.4x32dp128bit tmem[UR27+0x100], gdesc[UR66]",
     "e779001b420001000000100900e20f00"),
    ("UTCCP.T.S.2CTA.4x32dp128bit tmem[UR28+0x180], gdesc[UR60]",
     "e779001c3c8001000000300900e20500"),
    ("UTCCP.T.S.2CTA.128dp128bit tmem[UR30+0x100], gdesc[UR18]",
     "e779001e120001000000380800e20700"),
    # Barriers
    ("UTCBAR [UR23], URZ",                              "e9730017ff000000ff00000800e20700"),
    ("UTCBAR.MULTICAST [UR13], URZ, UR9",               "e973000dff0000000908000800e20300"),
    ("UTCBAR.2CTA.MULTICAST [UR15], URZ, UR10",         "e973000fff0000000a08200800e20700"),
    # Atomic find-and-set
    ("UTCATOMSWS.AND URZ, UR8",                          "e379ff00080000000000000800e20300"),
    ("UTCATOMSWS.FIND_AND_SET.ALIGN UP1, UR4, UR4",      "e3750400040000000008020800640e00"),
    ("UTCATOMSWS.2CTA.FIND_AND_SET.ALIGN UP0, UR5, UR5", "e3750500050000000008200800a40e00"),
    # Bulk-copy commit
    ("ACQBULK",                                          "2e780000000000000000000000e20f00"),
]


@pytest.mark.parametrize("sass_text,cubin_hex", SAMPLES, ids=[s[0] for s in SAMPLES])
def test_encoder_matches_cubin(encoder: Encoder, sass_text: str, cubin_hex: str):
    assert_byte_exact(encoder, sass_text, cubin_hex)


@requires_nvdisasm
@pytest.mark.parametrize("sass_text,cubin_hex", SAMPLES, ids=[s[0] for s in SAMPLES])
def test_nvdisasm_roundtrip(encoder: Encoder, sass_text: str, cubin_hex: str):
    assert_roundtrip(encoder, sass_text, cubin_hex)
