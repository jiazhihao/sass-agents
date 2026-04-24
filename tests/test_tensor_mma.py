"""Tensor-core matmul family: UTCHMMA / UTCQMMA / UTCOMMA (2CTA and 4X variants).

Covers the two shapes for UTC*MMA:
  (a) gdesc-first : gdesc, gdesc, tmem, tmem, idesc[, tmem_c], Ps
  (b) tmem-first  : tmem,  gdesc, tmem, tmem, idesc, Ps        (accumulator in A slot)
plus UTCHMMA.2CTA's special trailing-imm form (imm = 0x8 only):
      gdesc, gdesc, tmem, tmem, idesc, Ps, 0x8

Ground-truth (text, cubin_bytes) samples come from real cubins produced by
the cutlass sm_100a kernels. Mutation tests compose the encoder output against
a known-good scheduling region and round-trip it through nvdisasm.
"""

from __future__ import annotations

import pytest

from ._util import (
    Encoder,
    assert_byte_exact,
    assert_roundtrip,
    compose_with_template_sched,
    parse_nvdisasm_output,
    run_nvdisasm,
    canon_sass,
    requires_nvdisasm,
)


# ---------------------------------------------------------------------------
# Ground-truth (text, cubin_bytes) samples — byte-exact + round-trip checked
# ---------------------------------------------------------------------------

SAMPLES: list[tuple[str, str]] = [
    # ---- UTCHMMA (gdesc-first, 6-op) ------------------------------------
    ("UTCHMMA gdesc[UR60], gdesc[UR62], tmem[UR11], tmem[UR30], idesc[UR31], !UPT",
     "ea75003c3e1eff000b00800f00e20f00"),
    ("UTCHMMA gdesc[UR56], gdesc[UR58], tmem[UR10], tmem[UR30], idesc[UR31], UPT",
     "ea7500383a1eff000a00800b00e20f00"),
    ("UTCHMMA gdesc[UR34], gdesc[UR36], tmem[UR8], tmem[UR20], idesc[UR21], UP4",
     "ea7500222414ff000800000a00e20300"),
    # ---- UTCHMMA (tmem-first, accumulator in A slot) --------------------
    ("UTCHMMA tmem[UR11], gdesc[UR38], tmem[UR21], tmem[UR28], idesc[UR29], !UPT",
     "ea79000b261cff001500800f00e20f00"),
    ("UTCHMMA tmem[UR34], gdesc[UR44], tmem[UR35], tmem[UR28], idesc[UR29], UPT",
     "ea7900222c1cff002300800b00e20700"),
    # ---- UTCHMMA.2CTA (gdesc-first, 6-op) -------------------------------
    ("UTCHMMA.2CTA gdesc[UR38], gdesc[UR40], tmem[UR8], tmem[UR24], idesc[UR25], UP2",
     "ea7500262818ff000800200900e20700"),
    ("UTCHMMA.2CTA gdesc[UR6], gdesc[UR4], tmem[UR28], tmem[UR44], idesc[UR45], UP0",
     "ea750006042cff001c00200800e20300"),
    # ---- UTCHMMA.2CTA with trailing 0x8 imm (7-op) ----------------------
    ("UTCHMMA.2CTA gdesc[UR36], gdesc[UR42], tmem[UR27], tmem[UR34], idesc[UR35], UPT, 0x8",
     "ea7500242a22ff001b40a00b00e20900"),
    # ---- UTCQMMA (gdesc-first, 6-op) ------------------------------------
    ("UTCQMMA gdesc[UR34], gdesc[UR36], tmem[UR10], tmem[UR20], idesc[UR21], UP2",
     "ea7500222414ff000a03000900e20300"),
    ("UTCQMMA gdesc[UR54], gdesc[UR28], tmem[UR21], tmem[UR46], idesc[UR47], !UPT",
     "ea7500361c2eff001503800f00e20100"),
    # ---- UTCQMMA (tmem-first) -------------------------------------------
    ("UTCQMMA tmem[UR23], gdesc[UR32], tmem[UR28], tmem[UR44], idesc[UR45], !UPT",
     "ea790017202cff001c03800f00e20100"),
    # ---- UTCQMMA.2CTA (gdesc-first, 6-op) -------------------------------
    ("UTCQMMA.2CTA gdesc[UR30], gdesc[UR6], tmem[UR11], tmem[UR28], idesc[UR29], UP2",
     "ea75001e061cff000b03200900e20700"),
    # ---- UTCQMMA.2CTA (gdesc-first, 7-op with tmemC) --------------------
    ("UTCQMMA.2CTA gdesc[UR16], gdesc[UR14], tmem[UR12], tmem[UR18], idesc[UR19], tmem[UR20], !UPT",
     "ea7d00100e1214000c03a00f00e80500"),
    ("UTCQMMA.2CTA gdesc[UR48], gdesc[UR46], tmem[UR12], tmem[UR10], idesc[UR11], tmem[UR74], UPT",
     "ea7d00302e0a4a000c03a00b00e80900"),
    # ---- UTCOMMA.4X (always 7-op with tmemC) ----------------------------
    ("UTCOMMA.4X gdesc[UR48], gdesc[UR50], tmem[UR16], tmem[UR46], idesc[UR47], tmem[UR10], UP1",
     "ea750030322e0a801000800800e20300"),
    ("UTCOMMA.4X gdesc[UR6], gdesc[UR8], tmem[UR15], tmem[UR32], idesc[UR33], tmem[UR68], !UPT",
     "ea750006082044800f00800f00e20300"),
    ("UTCOMMA.4X gdesc[UR56], gdesc[UR58], tmem[UR17], tmem[UR54], idesc[UR55], tmem[UR10], UP0",
     "ea7500383a360a801100000800e20300"),
    # ---- UTCOMMA.2CTA.4X ------------------------------------------------
    ("UTCOMMA.2CTA.4X gdesc[UR38], gdesc[UR40], tmem[UR16], tmem[UR60], idesc[UR61], tmem[UR10], !UPT",
     "ea750026283c0a801000a00f00e20700"),
    ("UTCOMMA.2CTA.4X gdesc[UR50], gdesc[UR52], tmem[UR16], tmem[UR36], idesc[UR37], tmem[UR10], UP1",
     "ea75003234240a801000a00800e20700"),
    ("UTCOMMA.2CTA.4X gdesc[UR6], gdesc[UR8], tmem[UR15], tmem[UR32], idesc[UR33], tmem[UR70], !UPT",
     "ea750006082046800f00a00f00e20300"),
]


@pytest.mark.parametrize("sass_text,cubin_hex", SAMPLES, ids=[s[0] for s in SAMPLES])
def test_encoder_matches_cubin(encoder: Encoder, sass_text: str, cubin_hex: str):
    assert_byte_exact(encoder, sass_text, cubin_hex)


@requires_nvdisasm
@pytest.mark.parametrize("sass_text,cubin_hex", SAMPLES, ids=[s[0] for s in SAMPLES])
def test_nvdisasm_roundtrip(encoder: Encoder, sass_text: str, cubin_hex: str):
    assert_roundtrip(encoder, sass_text, cubin_hex)


# ---------------------------------------------------------------------------
# Mutation harness — one slot at a time against a known-good template
# ---------------------------------------------------------------------------

def _check_roundtrip(encoder: Encoder, text: str, template: bytes) -> None:
    composed = compose_with_template_sched(encoder, text, template)
    disasm = parse_nvdisasm_output(run_nvdisasm(composed))
    assert canon_sass(text) == canon_sass(disasm), (
        f"mutation round-trip failed\n"
        f"  original : {canon_sass(text)!r}\n"
        f"  disasm'd : {canon_sass(disasm)!r}\n"
        f"  bytes    : {composed.hex()}"
    )


# ---- UTCHMMA (gdesc-first) -------------------------------------------------

_TEMPLATE_UTCHMMA_GD = bytes.fromhex("ea75003c3e1eff000b00800f00e20f00")
_SHAPE_UTCHMMA_GD = ("UTCHMMA gdesc[UR{gA}], gdesc[UR{gB}], "
                     "tmem[UR{tA}], tmem[UR{tB}], idesc[UR{id_}], {ps}")
def _utchmma_gd(**ov) -> str:
    d = dict(gA=60, gB=62, tA=11, tB=30, id_=31, ps="!UPT"); d.update(ov)
    return _SHAPE_UTCHMMA_GD.format(**d)

_MUT_UTCHMMA_GD: list[str] = []
for gA in (0, 4, 22, 62):       _MUT_UTCHMMA_GD.append(_utchmma_gd(gA=gA))
for gB in (2, 18, 46):          _MUT_UTCHMMA_GD.append(_utchmma_gd(gB=gB))
for tA in (0, 7, 40, 62):       _MUT_UTCHMMA_GD.append(_utchmma_gd(tA=tA))
for tB in [(0,1),(20,21),(44,45),(60,61)]:
    _MUT_UTCHMMA_GD.append(_utchmma_gd(tB=tB[0], id_=tB[1]))
for ps in ("UP0","UP1","UP2","UP3","UPT","!UP0","!UP2"):
    _MUT_UTCHMMA_GD.append(_utchmma_gd(ps=ps))


# ---- UTCHMMA (tmem-first) --------------------------------------------------

_TEMPLATE_UTCHMMA_TM = bytes.fromhex("ea79000b261cff001500800f00e20f00")
_SHAPE_UTCHMMA_TM = ("UTCHMMA tmem[UR{acc}], gdesc[UR{gB}], "
                     "tmem[UR{tA}], tmem[UR{tB}], idesc[UR{id_}], {ps}")
def _utchmma_tm(**ov) -> str:
    d = dict(acc=11, gB=38, tA=21, tB=28, id_=29, ps="!UPT"); d.update(ov)
    return _SHAPE_UTCHMMA_TM.format(**d)

_MUT_UTCHMMA_TM: list[str] = []
for acc in (0, 5, 32, 62):      _MUT_UTCHMMA_TM.append(_utchmma_tm(acc=acc))
for gB in (0, 14, 50, 62):      _MUT_UTCHMMA_TM.append(_utchmma_tm(gB=gB))
for tA in (0, 15, 33, 62):      _MUT_UTCHMMA_TM.append(_utchmma_tm(tA=tA))
for tB in [(0,1),(16,17),(40,41),(62,63)]:
    _MUT_UTCHMMA_TM.append(_utchmma_tm(tB=tB[0], id_=tB[1]))
for ps in ("UP0","UP3","UPT","!UP1"):
    _MUT_UTCHMMA_TM.append(_utchmma_tm(ps=ps))


# ---- UTCHMMA.2CTA (gdesc-first, 6-op) --------------------------------------

_TEMPLATE_UTCHMMA_2CTA_A = bytes.fromhex("ea7500262818ff000800200900e20700")
_SHAPE_UTCHMMA_2CTA_A = ("UTCHMMA.2CTA gdesc[UR{gA}], gdesc[UR{gB}], "
                         "tmem[UR{tA}], tmem[UR{tB}], idesc[UR{id_}], {ps}")
def _utchmma_2cta_a(**ov) -> str:
    d = dict(gA=38, gB=40, tA=8, tB=24, id_=25, ps="UP2"); d.update(ov)
    return _SHAPE_UTCHMMA_2CTA_A.format(**d)

_MUT_UTCHMMA_2CTA_A: list[str] = []
for gA in (0, 2, 18, 62):       _MUT_UTCHMMA_2CTA_A.append(_utchmma_2cta_a(gA=gA))
for gB in (4, 16, 50):          _MUT_UTCHMMA_2CTA_A.append(_utchmma_2cta_a(gB=gB))
for tA in (0, 16, 32, 62):      _MUT_UTCHMMA_2CTA_A.append(_utchmma_2cta_a(tA=tA))
for tB in [(0,1),(22,23),(48,49),(60,61)]:
    _MUT_UTCHMMA_2CTA_A.append(_utchmma_2cta_a(tB=tB[0], id_=tB[1]))
for ps in ("UP0","UP1","UP3","UP6","UPT","!UPT","!UP0","!UP2"):
    _MUT_UTCHMMA_2CTA_A.append(_utchmma_2cta_a(ps=ps))


# ---- UTCHMMA.2CTA (gdesc-first, 7-op with imm=0x8) -------------------------

_TEMPLATE_UTCHMMA_2CTA_B = bytes.fromhex("ea7500242a22ff001b40a00b00e20900")
_SHAPE_UTCHMMA_2CTA_B = ("UTCHMMA.2CTA gdesc[UR{gA}], gdesc[UR{gB}], "
                         "tmem[UR{tA}], tmem[UR{tB}], idesc[UR{id_}], {ps}, 0x{imm:x}")
def _utchmma_2cta_b(**ov) -> str:
    d = dict(gA=36, gB=42, tA=27, tB=34, id_=35, ps="UPT", imm=0x8); d.update(ov)
    return _SHAPE_UTCHMMA_2CTA_B.format(**d)

_MUT_UTCHMMA_2CTA_B: list[str] = []
for gA in (0, 4, 22):           _MUT_UTCHMMA_2CTA_B.append(_utchmma_2cta_b(gA=gA))
for gB in (8, 24, 54):          _MUT_UTCHMMA_2CTA_B.append(_utchmma_2cta_b(gB=gB))
for tA in (0, 15, 31):          _MUT_UTCHMMA_2CTA_B.append(_utchmma_2cta_b(tA=tA))
for tB, id_ in [(10,11),(40,41),(58,59)]:
    _MUT_UTCHMMA_2CTA_B.append(_utchmma_2cta_b(tB=tB, id_=id_))
for ps in ("UP0","UP2","!UP1","!UPT"):
    _MUT_UTCHMMA_2CTA_B.append(_utchmma_2cta_b(ps=ps))


# ---- UTCQMMA (gdesc-first) -------------------------------------------------

_TEMPLATE_UTCQMMA_GD = bytes.fromhex("ea7500222414ff000a03000900e20300")
_SHAPE_UTCQMMA_GD = ("UTCQMMA gdesc[UR{gA}], gdesc[UR{gB}], "
                     "tmem[UR{tA}], tmem[UR{tB}], idesc[UR{id_}], {ps}")
def _utcqmma_gd(**ov) -> str:
    d = dict(gA=34, gB=36, tA=10, tB=20, id_=21, ps="UP2"); d.update(ov)
    return _SHAPE_UTCQMMA_GD.format(**d)

_MUT_UTCQMMA_GD: list[str] = []
for gA in (0, 2, 50, 62):       _MUT_UTCQMMA_GD.append(_utcqmma_gd(gA=gA))
for gB in (4, 30, 58):          _MUT_UTCQMMA_GD.append(_utcqmma_gd(gB=gB))
for tA in (0, 19, 33, 62):      _MUT_UTCQMMA_GD.append(_utcqmma_gd(tA=tA))
for tB, id_ in [(0,1),(30,31),(46,47),(60,61)]:
    _MUT_UTCQMMA_GD.append(_utcqmma_gd(tB=tB, id_=id_))
for ps in ("UP0","UP1","UP4","UPT","!UPT","!UP3"):
    _MUT_UTCQMMA_GD.append(_utcqmma_gd(ps=ps))


# ---- UTCQMMA.2CTA (gdesc-first, 7-op with tmemC) ---------------------------

_TEMPLATE_UTCQMMA_2CTA_C = bytes.fromhex("ea7d00100e1214000c03a00f00e80500")
_SHAPE_UTCQMMA_2CTA_C = ("UTCQMMA.2CTA gdesc[UR{gA}], gdesc[UR{gB}], "
                         "tmem[UR{tA}], tmem[UR{tB}], idesc[UR{id_}], tmem[UR{tC}], {ps}")
def _utcqmma_2cta_c(**ov) -> str:
    d = dict(gA=16, gB=14, tA=12, tB=18, id_=19, tC=20, ps="!UPT"); d.update(ov)
    return _SHAPE_UTCQMMA_2CTA_C.format(**d)

_MUT_UTCQMMA_2CTA_C: list[str] = []
for gA in (0, 30, 62):          _MUT_UTCQMMA_2CTA_C.append(_utcqmma_2cta_c(gA=gA))
for gB in (0, 44, 60):          _MUT_UTCQMMA_2CTA_C.append(_utcqmma_2cta_c(gB=gB))
for tA in (0, 32, 62):          _MUT_UTCQMMA_2CTA_C.append(_utcqmma_2cta_c(tA=tA))
for tB, id_ in [(0,1),(40,41),(60,61)]:
    _MUT_UTCQMMA_2CTA_C.append(_utcqmma_2cta_c(tB=tB, id_=id_))
for tC in (0, 4, 66, 76):       _MUT_UTCQMMA_2CTA_C.append(_utcqmma_2cta_c(tC=tC))
for ps in ("UP0","UP2","UPT","!UP1"):
    _MUT_UTCQMMA_2CTA_C.append(_utcqmma_2cta_c(ps=ps))


# ---- UTCOMMA.4X (always 7-op with tmemC) -----------------------------------

_TEMPLATE_UTCOMMA_4X = bytes.fromhex("ea750030322e0a801000800800e20300")
_SHAPE_UTCOMMA_4X = ("UTCOMMA.4X gdesc[UR{gA}], gdesc[UR{gB}], "
                     "tmem[UR{tA}], tmem[UR{tB}], idesc[UR{id_}], tmem[UR{tC}], {ps}")
def _utcomma_4x(**ov) -> str:
    d = dict(gA=48, gB=50, tA=16, tB=46, id_=47, tC=10, ps="UP1"); d.update(ov)
    return _SHAPE_UTCOMMA_4X.format(**d)

_MUT_UTCOMMA_4X: list[str] = []
for gA in (0, 6, 32, 62):       _MUT_UTCOMMA_4X.append(_utcomma_4x(gA=gA))
for gB in (0, 22, 58):          _MUT_UTCOMMA_4X.append(_utcomma_4x(gB=gB))
for tA in (0, 15, 32, 62):      _MUT_UTCOMMA_4X.append(_utcomma_4x(tA=tA))
for tB, id_ in [(0,1),(28,29),(60,61),(62,63)]:
    _MUT_UTCOMMA_4X.append(_utcomma_4x(tB=tB, id_=id_))
for tC in (0, 4, 66, 70):       _MUT_UTCOMMA_4X.append(_utcomma_4x(tC=tC))
for ps in ("UP0","UP2","UPT","!UPT","!UP0"):
    _MUT_UTCOMMA_4X.append(_utcomma_4x(ps=ps))


# ---- UTCOMMA.2CTA.4X -------------------------------------------------------

_TEMPLATE_UTCOMMA_2CTA_4X = bytes.fromhex("ea750026283c0a801000a00f00e20700")
_SHAPE_UTCOMMA_2CTA_4X = ("UTCOMMA.2CTA.4X gdesc[UR{gA}], gdesc[UR{gB}], "
                          "tmem[UR{tA}], tmem[UR{tB}], idesc[UR{id_}], tmem[UR{tC}], {ps}")
def _utcomma_2cta_4x(**ov) -> str:
    d = dict(gA=38, gB=40, tA=16, tB=60, id_=61, tC=10, ps="!UPT"); d.update(ov)
    return _SHAPE_UTCOMMA_2CTA_4X.format(**d)

_MUT_UTCOMMA_2CTA_4X: list[str] = []
for gA in (0, 6, 50):           _MUT_UTCOMMA_2CTA_4X.append(_utcomma_2cta_4x(gA=gA))
for gB in (8, 26, 52):          _MUT_UTCOMMA_2CTA_4X.append(_utcomma_2cta_4x(gB=gB))
for tA in (0, 17, 32):          _MUT_UTCOMMA_2CTA_4X.append(_utcomma_2cta_4x(tA=tA))
for tB, id_ in [(30,31),(36,37),(58,59)]:
    _MUT_UTCOMMA_2CTA_4X.append(_utcomma_2cta_4x(tB=tB, id_=id_))
for tC in (0, 4, 8, 70):        _MUT_UTCOMMA_2CTA_4X.append(_utcomma_2cta_4x(tC=tC))
for ps in ("UP0","UP1","UPT","!UP2"):
    _MUT_UTCOMMA_2CTA_4X.append(_utcomma_2cta_4x(ps=ps))


# ---- Parametrize dispatch --------------------------------------------------

# Each entry: (id_prefix, template_bytes, list_of_mutation_texts)
_MUTATION_GROUPS: list[tuple[str, bytes, list[str]]] = [
    ("UTCHMMA",              _TEMPLATE_UTCHMMA_GD,        _MUT_UTCHMMA_GD),
    ("UTCHMMA.tm",           _TEMPLATE_UTCHMMA_TM,        _MUT_UTCHMMA_TM),
    ("UTCHMMA.2CTA",         _TEMPLATE_UTCHMMA_2CTA_A,    _MUT_UTCHMMA_2CTA_A),
    ("UTCHMMA.2CTA.imm8",    _TEMPLATE_UTCHMMA_2CTA_B,    _MUT_UTCHMMA_2CTA_B),
    ("UTCQMMA",              _TEMPLATE_UTCQMMA_GD,        _MUT_UTCQMMA_GD),
    ("UTCQMMA.2CTA.7op",     _TEMPLATE_UTCQMMA_2CTA_C,    _MUT_UTCQMMA_2CTA_C),
    ("UTCOMMA.4X",           _TEMPLATE_UTCOMMA_4X,        _MUT_UTCOMMA_4X),
    ("UTCOMMA.2CTA.4X",      _TEMPLATE_UTCOMMA_2CTA_4X,   _MUT_UTCOMMA_2CTA_4X),
]

_MUTATION_CASES = [
    pytest.param(tpl, text, id=f"{prefix}::{text}")
    for (prefix, tpl, texts) in _MUTATION_GROUPS
    for text in texts
]


@requires_nvdisasm
@pytest.mark.parametrize("template,sass_text", _MUTATION_CASES)
def test_mma_operand_mutations(encoder: Encoder, template: bytes, sass_text: str):
    """Mutate each operand slot of every MMA variant; encoder output must round-trip."""
    _check_roundtrip(encoder, sass_text, template)
