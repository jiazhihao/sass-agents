"""TMA descriptor-based async copy family.

  UTMALDG.{2D,3D,4D,5D}[.MULTICAST][.2CTA]
  UTMALDG.5D.IM2COL
  UTMASTG.{2D,3D,4D,5D}
  UTMAREDG.3D.ADD
  UTMACCTL.PF / .IV
  UTMACMDFLUSH
  UTMAPF.L2.3D
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
    # TMA global-memory loads
    ("@!UP0 UTMALDG.3D [UR48], [UR4], desc[UR8]",
     "b4850004300800000010010800e20f00"),
    ("UTMALDG.3D.MULTICAST.2CTA [UR16], [UR42], UR6, desc[UR44]",
     "b473002a102c00000618210800e20700"),
    ("UTMALDG.2D [UR44], [UR54], desc[UR72]",
     "b47500362c4800000090000800e20500"),
    ("UTMALDG.5D.IM2COL [UR16], [UR62], UR68",
     "b473003e100000004400060800e20f00"),
    # TMA stores
    ("UTMASTG.3D [UR8], [UR74]",
     "b573004a080000000000010800e20f00"),
    ("UTMASTG.2D [UR16], [UR8]",
     "b5730008100000000080000800e20500"),
    # TMA reductions / control
    ("UTMAREDG.3D.ADD [UR8], [UR4]",
     "b6730004080000000000010800f00300"),
    ("UTMACCTL.PF [UR8]",
     "b9790008000000000000040800e40300"),
    ("UTMACCTL.IV [UR30]",
     "b979001e000000000000000800e22300"),
    ("UTMACMDFLUSH",
     "b7790000000000000000000000e20100"),
    ("UTMAPF.L2.3D [UR20], [UR14]",
     "b875000e140000000000010800e20b00"),
]


@pytest.mark.parametrize("sass_text,cubin_hex", SAMPLES, ids=[s[0] for s in SAMPLES])
def test_encoder_matches_cubin(encoder: Encoder, sass_text: str, cubin_hex: str):
    assert_byte_exact(encoder, sass_text, cubin_hex)


@requires_nvdisasm
@pytest.mark.parametrize("sass_text,cubin_hex", SAMPLES, ids=[s[0] for s in SAMPLES])
def test_nvdisasm_roundtrip(encoder: Encoder, sass_text: str, cubin_hex: str):
    assert_roundtrip(encoder, sass_text, cubin_hex)
