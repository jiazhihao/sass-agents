"""SASS-to-128-bit encoder for Blackwell (sm_100a).

Usage:
    from sass_encoder import Encoder, UnsupportedInstruction
    enc = Encoder()
    bits, struct_mask = enc.encode(sass_text)

Returns (bits, struct_mask), both 128-bit ints, where `struct_mask` covers the
bits that the encoder claims authority over (everything not in the 17-bit
scheduling region: bits 105..121). A caller that wants the full 16-byte image
matching the cubin must OR in the scheduling bits observed from the cubin.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


# Bits 105..121 are scheduling (stall/yield/wait/rd/wr barriers). Everything
# else is considered "structural" and the encoder is responsible for it.
SCHED_BITS_LO = 105
SCHED_BITS_HI = 121  # inclusive
SCHED_MASK = ((1 << (SCHED_BITS_HI - SCHED_BITS_LO + 1)) - 1) << SCHED_BITS_LO
STRUCT_MASK = ((1 << 128) - 1) & ~SCHED_MASK


class UnsupportedInstruction(Exception):
    pass


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

@dataclass
class ParsedInsn:
    pred: str | None          # e.g. "P0", "!P0", "UP0", "!UP0", or None for @PT
    mnemonic: str             # e.g. "IMAD.MOV.U32"
    operands: list[str]       # raw operand tokens, trimmed


def _split_operands(rest: str) -> list[str]:
    # Operands are comma-separated; commas never appear inside tokens we care
    # about (no nested parens at this level except `c[bank][off]` and
    # `desc[URx][R.64]`, which don't contain commas).
    if not rest.strip():
        return []
    out = []
    depth = 0
    cur = ""
    for ch in rest:
        if ch == "[":
            depth += 1; cur += ch
        elif ch == "]":
            depth -= 1; cur += ch
        elif ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


PRED_RE = re.compile(r"^@(![!]?)?(U?P[T0-9]+)\s+")


def parse(text: str) -> ParsedInsn:
    text = text.strip().rstrip(";").strip()
    pred = None
    m = PRED_RE.match(text)
    if m:
        bang = "!" if m.group(1) == "!" else ""
        pred = bang + m.group(2)
        text = text[m.end():]
    # mnemonic = first whitespace-delimited token
    parts = text.split(None, 1)
    mnemonic = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    operands = _split_operands(rest)
    return ParsedInsn(pred=pred, mnemonic=mnemonic, operands=operands)


# ---------------------------------------------------------------------------
# Operand parsing
# ---------------------------------------------------------------------------

REUSE_RE = re.compile(r"\.reuse\b")


def parse_reg(tok: str) -> tuple[int, bool]:
    """Return (reg_index, reuse). RZ -> 255. Accepts trailing .reuse."""
    reuse = bool(REUSE_RE.search(tok))
    tok = REUSE_RE.sub("", tok).strip()
    if tok == "RZ":
        return 255, reuse
    m = re.fullmatch(r"R(\d+)", tok)
    if not m:
        raise UnsupportedInstruction(f"bad reg {tok!r}")
    return int(m.group(1)), reuse


def parse_ureg(tok: str) -> tuple[int, bool]:
    """Return (index, reuse). URZ is encoded as 0xff (matching the RZ
    convention where the all-ones pattern denotes the zero-register)."""
    reuse = bool(REUSE_RE.search(tok))
    tok = REUSE_RE.sub("", tok).strip()
    if tok == "URZ":
        return 255, reuse
    m = re.fullmatch(r"UR(\d+)", tok)
    if not m:
        raise UnsupportedInstruction(f"bad ureg {tok!r}")
    return int(m.group(1)), reuse


def parse_pred(tok: str) -> tuple[int, bool]:
    """Return (index, negate). PT -> 7."""
    neg = tok.startswith("!")
    if neg:
        tok = tok[1:]
    if tok == "PT":
        return 7, neg
    m = re.fullmatch(r"P(\d)", tok)
    if not m:
        raise UnsupportedInstruction(f"bad pred {tok!r}")
    return int(m.group(1)), neg


def parse_upred(tok: str) -> tuple[int, bool]:
    neg = tok.startswith("!")
    if neg:
        tok = tok[1:]
    if tok == "UPT":
        return 7, neg
    m = re.fullmatch(r"UP(\d)", tok)
    if not m:
        raise UnsupportedInstruction(f"bad upred {tok!r}")
    return int(m.group(1)), neg


def parse_imm(tok: str) -> int:
    tok = tok.strip()
    if tok.startswith("+"):
        return parse_imm(tok[1:])
    if tok.startswith("-"):
        return -parse_imm(tok[1:])
    if tok.startswith("0x") or tok.startswith("0X"):
        return int(tok, 16)
    if re.fullmatch(r"\d+", tok):
        return int(tok)
    raise UnsupportedInstruction(f"bad immediate {tok!r}")


CONST_RE = re.compile(r"^c\[(0x[0-9a-fA-F]+|\d+)\]\[(.+)\]$")


def parse_const(tok: str) -> tuple[int, int, int]:
    """Parse a constant-bank reference c[bank][inner].

    Returns (bank, offset, indirect_reg) where indirect_reg is 0xff for the
    non-indirect forms (either `[imm]` or `[RZ]`).

    `inner` forms accepted:
      [imm]            -> (bank, imm, 0xff)
      [RZ]             -> (bank, 0,   0xff)
      [Rn]             -> (bank, 0,   n)
      [Rn + imm]       -> (bank, imm, n)
      [Rn - imm]       -> (bank, -imm, n)
    """
    m = CONST_RE.match(tok.strip())
    if not m:
        raise UnsupportedInstruction(f"bad constref {tok!r}")
    bank = int(m.group(1), 0)
    inner = m.group(2).strip()
    if inner.startswith("0x") or inner.startswith("0X") or re.fullmatch(r"\d+", inner):
        return bank, int(inner, 0), 0xff
    if inner == "RZ":
        return bank, 0, 0xff
    mm = re.fullmatch(r"(R\d+|UR\d+)\s*(?:([+-])\s*(0x[0-9a-fA-F]+|\d+))?", inner)
    if mm:
        reg_tok = mm.group(1)
        sign = mm.group(2)
        imm = mm.group(3)
        if reg_tok.startswith("UR"):
            rn, _ = parse_ureg(reg_tok)
        else:
            rn, _ = parse_reg(reg_tok)
        off = 0
        if imm is not None:
            off = int(imm, 0)
            if sign == "-":
                off = -off
        return bank, off, rn
    raise UnsupportedInstruction(f"indirect const {tok!r} not supported")


# ---------------------------------------------------------------------------
# Common field helpers
# ---------------------------------------------------------------------------

def set_bits(word: int, lo: int, width: int, value: int) -> int:
    mask = ((1 << width) - 1) << lo
    word &= ~mask
    word |= (value & ((1 << width) - 1)) << lo
    return word


def get_bits(word: int, lo: int, width: int) -> int:
    return (word >> lo) & ((1 << width) - 1)


# Common bit fields that appear consistent across Blackwell opcodes (observed
# by diffing many instances):
#   - opcode low byte   : bits 0..8   (9 bits)  [byte 0 + low bit of byte 1]
#   - predicate         : bits 12..15 (4 bits; bit15=negate, bits12..14=idx)
#   - Rd field          : bits 16..23 (8 bits)
#   - Ra field          : bits 24..31 (8 bits)
# These are patterns I'll refine as encoders are built up.


def encode_pred_guard(pred: str | None) -> int:
    """Encode the @pred guard into bits 12..15. PT is 0x7, no guard => PT."""
    if pred is None:
        return 0x7 << 12   # @PT
    neg = 0
    tok = pred
    if tok.startswith("!"):
        neg = 1
        tok = tok[1:]
    if tok.startswith("U"):
        # uniform predicate guard occupies a different bit? For now treat as
        # regular — will be refined when we encounter @UP* instructions.
        tok = tok[1:]
    if tok == "PT":
        idx = 7
    else:
        m = re.fullmatch(r"P(\d)", tok)
        if not m:
            raise UnsupportedInstruction(f"pred guard {pred!r}")
        idx = int(m.group(1))
    return ((neg & 1) << 15) | (idx << 12)


# ---------------------------------------------------------------------------
# Encoder dispatch
# ---------------------------------------------------------------------------

@dataclass
class Context:
    """Optional context for position-dependent encoders (branches, etc)."""
    pc: int = 0                                     # byte offset of this insn
    labels: dict[str, int] | None = None            # {label: offset}


EncoderFn = Callable[..., int]   # fn(ParsedInsn, ctx: Context) -> int
_TABLE: dict[str, EncoderFn] = {}


def register(mnemonic: str):
    def deco(fn: EncoderFn):
        _TABLE[mnemonic] = fn
        return fn
    return deco


class Encoder:
    def encode(self, text: str, ctx: Context | None = None) -> tuple[int, int]:
        p = parse(text)
        fn = _TABLE.get(p.mnemonic)
        if fn is None:
            raise UnsupportedInstruction(f"no encoder for {p.mnemonic!r}")
        try:
            bits = fn(p, ctx)
        except TypeError:
            # Encoder defined with the old 1-arg signature.
            bits = fn(p)
        return bits & STRUCT_MASK, STRUCT_MASK


# ---------------------------------------------------------------------------
# Per-opcode encoders (populated incrementally)
# ---------------------------------------------------------------------------

# Bit layout for Blackwell opcodes (empirical): the low 9 bits at [0..8] are
# the primary opcode; many of our observed encodings have byte 0 / byte 1 as
# the primary opcode, with bit 9 set when the source encodes a URx/immediate
# variant, etc.


def _apply_pred(w: int, pred: str | None) -> int:
    """Set bits 12..15 to the predicate guard. PT is 0b0111 (idx=7, neg=0)."""
    # Clear bits 12..15 first
    w = set_bits(w, 12, 4, 0)
    if pred is None:
        return set_bits(w, 12, 4, 0x7)
    neg = 0
    tok = pred
    if tok.startswith("!"):
        neg = 1
        tok = tok[1:]
    # Uniform-predicate guards also use this same 4-bit field in Blackwell
    # (observed: @UP0 and @P0 guards both encode index in bits 12..14 with
    # bit 15 = negate — but the uniform form is selected by a separate bit
    # elsewhere in the encoding for some opcodes. We'll refine per-opcode.)
    if tok.startswith("U"):
        tok = tok[1:]
    if tok == "PT":
        idx = 7
    else:
        m = re.fullmatch(r"P(\d)", tok)
        if not m:
            raise UnsupportedInstruction(f"pred guard {pred!r}")
        idx = int(m.group(1))
    return set_bits(w, 12, 4, (neg << 3) | idx)


def _parse_syncs_addr(tok: str) -> tuple[int, int, int]:
    """Parse a SYNCS-style address operand.

    Accepts combinations of R base, UR base and immediate offset inside [].
    Returns (Ra_idx_or_0xff, URa_idx_or_0xff, offset).
    """
    m = re.fullmatch(r"\[\s*(.+?)\s*\]", tok.strip())
    if not m:
        raise UnsupportedInstruction(f"bad syncs addr {tok!r}")
    inside = m.group(1)
    # Split on + or - while keeping the sign with the offset literal.
    tokens = re.findall(r"(UR\d+|URZ|R\d+|RZ|[-+]?0x[0-9a-fA-F]+|[-+]?\d+)", inside)
    ra = 0xff
    ur = 0xff
    off = 0
    for piece in tokens:
        if re.fullmatch(r"R\d+|RZ", piece):
            ra, _ = parse_reg(piece)
        elif re.fullmatch(r"UR\d+|URZ", piece):
            ur, _ = parse_ureg(piece)
        else:
            off = parse_imm(piece)
    return ra, ur, off


def _syncs_phasechk(p: ParsedInsn, *, trywait: bool) -> int:
    if len(p.operands) != 3:
        raise UnsupportedInstruction("SYNCS.PHASECHK arity")
    pd = _parse_pred_dst(p.operands[0])
    ra, ur, off = _parse_syncs_addr(p.operands[1])
    rb, _ = parse_reg(p.operands[2])
    w = 0
    w = set_bits(w, 0, 16, 0x75a7)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 32, 8, rb)
    w = set_bits(w, 40, 24, off & ((1 << 24) - 1))
    w = set_bits(w, 64, 8, ur)
    w = set_bits(w, 72, 8, 0x11 if trywait else 0x10)
    w = set_bits(w, 81, 3, pd)
    w = set_bits(w, 91, 1, 1)
    return _apply_pred(w, p.pred)


@register("SYNCS.PHASECHK.TRANS64")
def enc_syncs_phasechk(p: ParsedInsn) -> int:
    return _syncs_phasechk(p, trywait=False)


@register("SYNCS.PHASECHK.TRANS64.TRYWAIT")
def enc_syncs_phasechk_trywait(p: ParsedInsn) -> int:
    return _syncs_phasechk(p, trywait=True)


@register("SYNCS.EXCH.64")
def enc_syncs_exch_64(p: ParsedInsn) -> int:
    """SYNCS.EXCH.64 URd, [addr], URb_data"""
    if len(p.operands) != 3:
        raise UnsupportedInstruction("SYNCS.EXCH.64 arity")
    urd, _ = parse_ureg(p.operands[0])
    ra, ur, off = _parse_syncs_addr(p.operands[1])
    urb, _ = parse_ureg(p.operands[2])
    w = 0
    w = set_bits(w, 0, 16, 0x75b2)
    w = set_bits(w, 16, 8, urd)
    if ur != 0xff and ra == 0xff:
        w = set_bits(w, 24, 8, ur)
    else:
        w = set_bits(w, 24, 8, ra)
        w = set_bits(w, 64, 8, ur)
    w = set_bits(w, 32, 8, urb)
    w = set_bits(w, 40, 24, off & ((1 << 24) - 1))
    w = set_bits(w, 72, 8, 0x01)
    w = set_bits(w, 91, 1, 1)
    return _apply_pred(w, p.pred)


def _syncs_arrive(p: ParsedInsn, *, byte9: int, byte10: int) -> int:
    if len(p.operands) != 3:
        raise UnsupportedInstruction("SYNCS.ARRIVE arity")
    rd, _ = parse_reg(p.operands[0])
    ra, ur, off = _parse_syncs_addr(p.operands[1])
    rc, _ = parse_reg(p.operands[2])
    w = 0
    w = set_bits(w, 0, 16, 0x79a7)
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 32, 8, rc)
    w = set_bits(w, 40, 24, off & ((1 << 24) - 1))
    w = set_bits(w, 64, 8, ur)
    w = set_bits(w, 72, 8, byte9)
    w = set_bits(w, 80, 8, byte10)
    w = set_bits(w, 91, 1, 1)
    return _apply_pred(w, p.pred)


@register("SYNCS.ARRIVE.TRANS64.A1T0")
def enc_syncs_arrive_a1t0(p: ParsedInsn) -> int:
    return _syncs_arrive(p, byte9=0x00, byte10=0x10)


@register("SYNCS.ARRIVE.TRANS64.RED.A1T0")
def enc_syncs_arrive_red_a1t0(p: ParsedInsn) -> int:
    return _syncs_arrive(p, byte9=0x04, byte10=0x10)


@register("SYNCS.ARRIVE.TRANS64")
def enc_syncs_arrive_plain(p: ParsedInsn) -> int:
    return _syncs_arrive(p, byte9=0x00, byte10=0x00)


@register("SYNCS.ARRIVE.TRANS64.RED")
def enc_syncs_arrive_red_plain(p: ParsedInsn) -> int:
    return _syncs_arrive(p, byte9=0x04, byte10=0x00)


@register("SYNCS.ARRIVE.TRANS64.A0TR")
def enc_syncs_arrive_a0tr(p: ParsedInsn) -> int:
    return _syncs_arrive(p, byte9=0x00, byte10=0x30)


@register("SYNCS.ARRIVE.TRANS64.RED.A0TR")
def enc_syncs_arrive_red_a0tr(p: ParsedInsn) -> int:
    return _syncs_arrive(p, byte9=0x04, byte10=0x30)


def _parse_ur_addr(tok: str) -> int:
    """Parse a "[URx]" address — returns UR index."""
    m = re.fullmatch(r"\[\s*(UR\d+|URZ)\s*\]", tok.strip())
    if not m:
        raise UnsupportedInstruction(f"bad UR addr {tok!r}")
    ur, _ = parse_ureg(m.group(1))
    return ur


def _parse_desc_ur(tok: str) -> int:
    m = re.fullmatch(r"desc\[\s*(UR\d+|URZ)\s*\]", tok.strip())
    if not m:
        raise UnsupportedInstruction(f"bad desc[UR] {tok!r}")
    ur, _ = parse_ureg(m.group(1))
    return ur


def _utmaldg_impl(p: ParsedInsn, *, byte9: int, byte10: int,
                   multicast: bool) -> int:
    """UTMALDG.<dim>[.MULTICAST][.2CTA] — 3 or 4 operand TMA global load.

    Non-multicast: [URa], [URb], desc[URc]      (3 operands)
    Multicast:     [URa], [URb], URm, desc[URc] (4 operands)
    """
    expected = 4 if multicast else 3
    if len(p.operands) != expected:
        raise UnsupportedInstruction(f"UTMALDG arity {len(p.operands)}")
    ura = _parse_ur_addr(p.operands[0])
    urb = _parse_ur_addr(p.operands[1])
    if multicast:
        urm, _ = parse_ureg(p.operands[2])
        urc = _parse_desc_ur(p.operands[3])
    else:
        urm = 0x00
        urc = _parse_desc_ur(p.operands[2])
    w = 0
    w = set_bits(w, 0, 16, 0x75b4)
    w = set_bits(w, 24, 8, urb)
    w = set_bits(w, 32, 8, ura)
    w = set_bits(w, 40, 8, urc)
    if multicast:
        w = set_bits(w, 64, 8, urm)
        # UTMALDG.3D.MULTICAST uses the 0x73xx opcode (bit 9 = 0 vs 0x75xx's bit 10).
        w = set_bits(w, 0, 16, 0x73b4)
    w = set_bits(w, 72, 8, byte9)
    w = set_bits(w, 80, 8, byte10)
    w = set_bits(w, 91, 1, 1)
    return _apply_pred(w, p.pred)


# Mapping of UTMALDG modifier combos to (byte9, byte10) empirical values.
_UTMALDG_VARIANTS = {
    # dim, multicast, 2cta
    ("2D", False, False): (0x90, 0x00),
    ("2D", True,  False): (0x98, 0x00),
    ("2D", False, True ): (0x90, 0x20),
    ("2D", True,  True ): (0x98, 0x20),
    ("3D", False, False): (0x10, 0x01),
    ("3D", True,  False): (0x18, 0x01),
    ("3D", False, True ): (0x10, 0x21),
    ("3D", True,  True ): (0x18, 0x21),
    ("4D", False, False): (0x90, 0x01),
    ("4D", True,  False): (0x98, 0x01),
    ("4D", False, True ): (0x90, 0x21),
    ("4D", True,  True ): (0x98, 0x21),
    ("5D", False, False): (0x10, 0x02),
}


def _mk_utmaldg(dim: str, mc: bool, cta2: bool):
    byte9, byte10 = _UTMALDG_VARIANTS[(dim, mc, cta2)]
    def enc(p: ParsedInsn, ctx=None) -> int:
        return _utmaldg_impl(p, byte9=byte9, byte10=byte10, multicast=mc)
    return enc


for _dim in ("2D", "3D", "4D", "5D"):
    for _mc in (False, True):
        for _cta2 in (False, True):
            if (_dim, _mc, _cta2) not in _UTMALDG_VARIANTS:
                continue
            _name = f"UTMALDG.{_dim}"
            if _mc: _name += ".MULTICAST"
            if _cta2: _name += ".2CTA"
            _TABLE[_name] = _mk_utmaldg(_dim, _mc, _cta2)


@register("UTMALDG.5D.IM2COL")
def enc_utmaldg_5d_im2col(p: ParsedInsn) -> int:
    """UTMALDG.5D.IM2COL [URa], [URb], URc — image-to-column variant."""
    if len(p.operands) != 3:
        raise UnsupportedInstruction("UTMALDG.5D.IM2COL arity")
    ura = _parse_ur_addr(p.operands[0])
    urb = _parse_ur_addr(p.operands[1])
    urc, _ = parse_ureg(p.operands[2])
    w = 0
    w = set_bits(w, 0, 16, 0x73b4)
    w = set_bits(w, 24, 8, urb)
    w = set_bits(w, 32, 8, ura)
    w = set_bits(w, 64, 8, urc)                  # URc at byte 8
    w = set_bits(w, 80, 8, 0x06)
    w = set_bits(w, 91, 1, 1)
    return _apply_pred(w, p.pred)


@register("UTCATOMSWS.AND")
def enc_utcatomsws_and(p: ParsedInsn) -> int:
    if len(p.operands) != 2:
        raise UnsupportedInstruction("UTCATOMSWS.AND arity")
    urd, _ = parse_ureg(p.operands[0])
    ura, _ = parse_ureg(p.operands[1])
    w = 0
    w = set_bits(w, 0, 16, 0x79e3)
    w = set_bits(w, 16, 8, urd)
    w = set_bits(w, 32, 8, ura)
    w = set_bits(w, 91, 1, 1)
    return _apply_pred(w, p.pred)


def _utcatomsws_fas_impl(p: ParsedInsn, *, cta2: bool) -> int:
    """UTCATOMSWS[.2CTA].FIND_AND_SET.ALIGN UPd, URd, URa"""
    if len(p.operands) != 3:
        raise UnsupportedInstruction("UTCATOMSWS.FAS arity")
    upd = _parse_upred_dst(p.operands[0])
    urd, _ = parse_ureg(p.operands[1])
    ura, _ = parse_ureg(p.operands[2])
    w = 0
    w = set_bits(w, 0, 16, 0x75e3)
    w = set_bits(w, 16, 8, urd)
    w = set_bits(w, 32, 8, ura)
    w = set_bits(w, 72, 8, 0x08)
    w = set_bits(w, 81, 3, upd)
    if cta2:
        w = set_bits(w, 80, 8, 0x20 | (upd << 1))
    w = set_bits(w, 91, 1, 1)
    return _apply_pred(w, p.pred)


@register("UTCATOMSWS.FIND_AND_SET.ALIGN")
def enc_utcatomsws_fas(p: ParsedInsn) -> int:
    return _utcatomsws_fas_impl(p, cta2=False)


@register("UTCATOMSWS.2CTA.FIND_AND_SET.ALIGN")
def enc_utcatomsws_2cta_fas(p: ParsedInsn) -> int:
    return _utcatomsws_fas_impl(p, cta2=True)


@register("UTMASTG.5D.IM2COL")
def enc_utmastg_5d_im2col(p: ParsedInsn) -> int:
    if len(p.operands) != 2:
        raise UnsupportedInstruction("UTMASTG.5D.IM2COL arity")
    ura = _parse_ur_addr(p.operands[0])
    urb = _parse_ur_addr(p.operands[1])
    w = 0
    w = set_bits(w, 0, 16, 0x73b5)
    w = set_bits(w, 24, 8, urb)
    w = set_bits(w, 32, 8, ura)
    w = set_bits(w, 80, 8, 0x06)
    w = set_bits(w, 91, 1, 1)
    return _apply_pred(w, p.pred)


@register("UTCBAR.MULTICAST")
def enc_utcbar_multicast(p: ParsedInsn) -> int:
    """UTCBAR.MULTICAST [URa], URb, URc"""
    if len(p.operands) != 3:
        raise UnsupportedInstruction("UTCBAR.MULTICAST arity")
    ura = _parse_ur_addr(p.operands[0])
    urb, _ = parse_ureg(p.operands[1])
    urc, _ = parse_ureg(p.operands[2])
    w = 0
    w = set_bits(w, 0, 16, 0x73e9)
    w = set_bits(w, 24, 8, ura)
    w = set_bits(w, 32, 8, urb)
    w = set_bits(w, 64, 8, urc)
    w = set_bits(w, 72, 8, 0x08)                 # MULTICAST only
    w = set_bits(w, 91, 1, 1)
    return _apply_pred(w, p.pred)


@register("UTMAREDG.3D.ADD")
def enc_utmaredg_3d_add(p: ParsedInsn) -> int:
    """UTMAREDG.3D.ADD [URa], [URb] — reduction variant of UTMASTG."""
    if len(p.operands) != 2:
        raise UnsupportedInstruction("UTMAREDG.3D.ADD arity")
    ura = _parse_ur_addr(p.operands[0])
    urb = _parse_ur_addr(p.operands[1])
    w = 0
    w = set_bits(w, 0, 16, 0x73b6)
    w = set_bits(w, 24, 8, urb)
    w = set_bits(w, 32, 8, ura)
    w = set_bits(w, 80, 8, 0x01)
    w = set_bits(w, 91, 1, 1)
    return _apply_pred(w, p.pred)


@register("UTMASTG.4D")
def enc_utmastg_4d(p: ParsedInsn) -> int:
    if len(p.operands) != 2:
        raise UnsupportedInstruction("UTMASTG.4D arity")
    ura = _parse_ur_addr(p.operands[0])
    urb = _parse_ur_addr(p.operands[1])
    w = 0
    w = set_bits(w, 0, 16, 0x73b5)
    w = set_bits(w, 24, 8, urb)
    w = set_bits(w, 32, 8, ura)
    w = set_bits(w, 72, 8, 0x80)                # 4D flag
    w = set_bits(w, 80, 8, 0x01)
    w = set_bits(w, 91, 1, 1)
    return _apply_pred(w, p.pred)


@register("UTMASTG.5D")
def enc_utmastg_5d(p: ParsedInsn) -> int:
    if len(p.operands) != 2:
        raise UnsupportedInstruction("UTMASTG.5D arity")
    ura = _parse_ur_addr(p.operands[0])
    urb = _parse_ur_addr(p.operands[1])
    w = 0
    w = set_bits(w, 0, 16, 0x73b5)
    w = set_bits(w, 24, 8, urb)
    w = set_bits(w, 32, 8, ura)
    w = set_bits(w, 80, 8, 0x02)
    w = set_bits(w, 91, 1, 1)
    return _apply_pred(w, p.pred)


@register("UTMASTG.3D")
def enc_utmastg_3d(p: ParsedInsn) -> int:
    """UTMASTG.3D [URa], [URb]"""
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"UTMASTG arity {len(p.operands)}")
    ura = _parse_ur_addr(p.operands[0])
    urb = _parse_ur_addr(p.operands[1])
    w = 0
    w = set_bits(w, 0, 16, 0x73b5)
    w = set_bits(w, 24, 8, urb)
    w = set_bits(w, 32, 8, ura)
    w = set_bits(w, 80, 8, 0x01)
    w = set_bits(w, 91, 1, 1)
    return _apply_pred(w, p.pred)


@register("UTMACCTL.PF")
def enc_utmacctl_pf(p: ParsedInsn) -> int:
    if len(p.operands) != 1:
        raise UnsupportedInstruction(f"UTMACCTL.PF arity {len(p.operands)}")
    ur = _parse_ur_addr(p.operands[0])
    w = 0
    w = set_bits(w, 0, 16, 0x79b9)
    w = set_bits(w, 24, 8, ur)
    w = set_bits(w, 72, 8, 0x00)
    w = set_bits(w, 80, 8, 0x04)
    w = set_bits(w, 91, 1, 1)
    return _apply_pred(w, p.pred)


def _atoms_logical_impl(p: ParsedInsn, *, subop: int) -> int:
    """ATOMS.{AND,OR,XOR} Rd, [addr], Rb — shared-memory atomic bitwise op.

    Subop bits 87..88: AND=0b01, OR=0b10, XOR=0b11.
    """
    if len(p.operands) != 3:
        raise UnsupportedInstruction(f"ATOMS arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    ra, ur, off = _parse_syncs_addr(p.operands[1])
    rb, _ = parse_reg(p.operands[2])
    w = 0
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 32, 8, rb)
    w = set_bits(w, 40, 24, off & ((1 << 24) - 1))
    w = set_bits(w, 87, 2, subop)
    w = set_bits(w, 89, 1, 1)                   # fixed tail flag
    if ur != 0xff:
        w = set_bits(w, 0, 16, 0x798c)
        w = set_bits(w, 24, 8, ra)
        w = set_bits(w, 64, 8, ur)
        w = set_bits(w, 91, 1, 1)
    else:
        w = set_bits(w, 0, 16, 0x738c)
        w = set_bits(w, 24, 8, ra)
    return _apply_pred(w, p.pred)


@register("ATOMS.AND")
def enc_atoms_and(p: ParsedInsn) -> int:
    return _atoms_logical_impl(p, subop=0b01)


@register("ATOMS.OR")
def enc_atoms_or(p: ParsedInsn) -> int:
    return _atoms_logical_impl(p, subop=0b10)


@register("ATOMS.XOR")
def enc_atoms_xor(p: ParsedInsn) -> int:
    return _atoms_logical_impl(p, subop=0b11)


@register("STSM.16.MT88.4")
def enc_stsm_16_mt88_4(p: ParsedInsn) -> int:
    """STSM.16.MT88.4 [Ra], Rb — matrix shared store."""
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"STSM arity {len(p.operands)}")
    ra, ur, off = _parse_syncs_addr(p.operands[0])
    rb, _ = parse_reg(p.operands[1])
    w = 0
    w = set_bits(w, 0, 16, 0x7844)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 32, 8, rb)
    w = set_bits(w, 40, 24, off & ((1 << 24) - 1))
    if ur != 0xff:
        w = set_bits(w, 64, 8, ur)
        w = set_bits(w, 91, 1, 1)
    w = set_bits(w, 72, 8, 0x42)
    return _apply_pred(w, p.pred)


@register("LDSM.16.MT88.4")
def enc_ldsm_16_mt88_4(p: ParsedInsn) -> int:
    """LDSM.16.MT88.4 Rd, [Ra + URx + off] — matrix shared load."""
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"LDSM arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    ra, ur, off = _parse_syncs_addr(p.operands[1])
    w = 0
    w = set_bits(w, 0, 16, 0x783b)
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 32, 8, 0x00 if ur == 0xff else ur)
    w = set_bits(w, 40, 24, off & ((1 << 24) - 1))
    w = set_bits(w, 72, 8, 0x42)
    if ur != 0xff:
        w = set_bits(w, 91, 1, 1)
    return _apply_pred(w, p.pred)


TMEM_RE = re.compile(r"^tmem\[\s*(UR\d+|URZ)\s*(?:([+-])\s*(0x[0-9a-fA-F]+|\d+))?\s*\]$")
GDESC_RE = re.compile(r"^gdesc\[\s*(UR\d+|URZ)\s*\]$")


def _parse_tmem(tok: str) -> tuple[int, int]:
    m = TMEM_RE.fullmatch(tok.strip())
    if not m:
        raise UnsupportedInstruction(f"bad tmem {tok!r}")
    ur, _ = parse_ureg(m.group(1))
    off = 0
    if m.group(2) is not None:
        off = int(m.group(3), 0)
        if m.group(2) == "-":
            off = -off
    return ur, off


def _parse_gdesc(tok: str) -> int:
    m = GDESC_RE.fullmatch(tok.strip())
    if not m:
        raise UnsupportedInstruction(f"bad gdesc {tok!r}")
    ur, _ = parse_ureg(m.group(1))
    return ur


# --- LDTM / STTM ---------------------------------------------------------
# byte 10 (bits 80..87) encodes both a fixed LDTM/STTM marker and the "count"
# suffix (.x2/.x8/.x16/.x32/.x64). Observed byte 10 values:
LDTM_BYTE10 = {
    "x2":   0x0c,
    "x4":   0x14,
    "x8":   0x1c,
    "x16":  0x24,
    "x32":  0x2c,
    "x64":  0x34,
    "x128": 0x3c,
}


LDTM_16DP256_BYTE10 = {
    "x2":  0x0a,
    "x4":  0x12,
    "x8":  0x1a,
    "x16": 0x22,
}


def _ldtm_impl(p: ParsedInsn, *, width: str) -> int:
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"LDTM arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    ur, off = _parse_tmem(p.operands[1])
    if width not in LDTM_BYTE10:
        raise UnsupportedInstruction(f"LDTM width {width!r}")
    w = 0
    w = set_bits(w, 0, 16, 0x79ee)
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 32, 8, ur)
    w = set_bits(w, 40, 24, off & ((1 << 24) - 1))
    w = set_bits(w, 80, 8, LDTM_BYTE10[width])
    w = set_bits(w, 91, 1, 1)
    return _apply_pred(w, p.pred)


def _sttm_impl(p: ParsedInsn, *, width: str) -> int:
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"STTM arity {len(p.operands)}")
    ur, off = _parse_tmem(p.operands[0])
    rb, _ = parse_reg(p.operands[1])
    if width not in LDTM_BYTE10:
        raise UnsupportedInstruction(f"STTM width {width!r}")
    w = 0
    w = set_bits(w, 0, 16, 0x79ed)
    w = set_bits(w, 32, 8, rb)
    w = set_bits(w, 40, 24, off & ((1 << 24) - 1))
    w = set_bits(w, 64, 8, ur)                  # URa at bits 64..71 (not 24)
    w = set_bits(w, 80, 8, LDTM_BYTE10[width])
    w = set_bits(w, 91, 1, 1)
    return _apply_pred(w, p.pred)


for _wid in LDTM_BYTE10:
    _TABLE[f"LDTM.{_wid}"] = (lambda p, ctx=None, _w=_wid: _ldtm_impl(p, width=_w))
    _TABLE[f"STTM.{_wid}"] = (lambda p, ctx=None, _w=_wid: _sttm_impl(p, width=_w))


def _ldtm_16dp256_impl(p: ParsedInsn, *, width: str) -> int:
    w = _ldtm_impl(p, width="x2")  # temp to reuse machinery
    w = set_bits(w, 80, 8, LDTM_16DP256_BYTE10[width])
    return w


def _sttm_16dp128_impl(p: ParsedInsn, *, width: str, byte10: int) -> int:
    w = _sttm_impl(p, width="x2")
    w = set_bits(w, 80, 8, byte10)
    return w


for _wid in LDTM_16DP256_BYTE10:
    _TABLE[f"LDTM.16dp256bit.{_wid}"] = (
        lambda p, ctx=None, _w=_wid: _ldtm_16dp256_impl(p, width=_w))


# STTM.16dp128bit.x16 observed byte 10 = 0x20.
_TABLE["STTM.16dp128bit.x16"] = (
    lambda p, ctx=None: _sttm_16dp128_impl(p, width="x16", byte10=0x20))


# --- LDSM / STSM 16-bit matrix moves ------------------------------------
def _lds_matrix_impl(p: ParsedInsn, *, opcode: int, dst_is_reg: bool,
                     mt: bool, count_log2: int) -> int:
    """LDSM.16.[M|MT]88.{x2,x4,x8} Rd, [addr]  or
    STSM.16.[M|MT]88.{x2,x4,x8} [addr], Rb
    """
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"{p.mnemonic} arity {len(p.operands)}")
    if dst_is_reg:
        rd, _ = parse_reg(p.operands[0])
        addr = p.operands[1]
        ra, ur, off = _parse_syncs_addr(addr)
        w = 0
        w = set_bits(w, 0, 16, opcode)
        w = set_bits(w, 16, 8, rd)
        w = set_bits(w, 24, 8, ra)
        w = set_bits(w, 32, 8, 0x00 if ur == 0xff else ur)
        w = set_bits(w, 40, 24, off & ((1 << 24) - 1))
        if ur != 0xff:
            w = set_bits(w, 91, 1, 1)
    else:
        addr = p.operands[0]
        ra, ur, off = _parse_syncs_addr(addr)
        rb, _ = parse_reg(p.operands[1])
        w = 0
        w = set_bits(w, 0, 16, opcode)
        w = set_bits(w, 24, 8, ra)
        w = set_bits(w, 32, 8, rb)
        w = set_bits(w, 40, 24, off & ((1 << 24) - 1))
        if ur != 0xff:
            w = set_bits(w, 64, 8, ur)
            w = set_bits(w, 91, 1, 1)

    byte9 = count_log2 | (0x40 if mt else 0)
    w = set_bits(w, 72, 8, byte9)
    return _apply_pred(w, p.pred)


for _ct, _cl in [("2", 1), ("4", 2), ("8", 3)]:
    # LDSM.16.M88.*
    _TABLE[f"LDSM.16.M88.{_ct}"] = (lambda p, ctx=None, _c=_cl:
        _lds_matrix_impl(p, opcode=0x783b, dst_is_reg=True, mt=False, count_log2=_c))
    # LDSM.16.MT88.*
    _TABLE[f"LDSM.16.MT88.{_ct}"] = (lambda p, ctx=None, _c=_cl:
        _lds_matrix_impl(p, opcode=0x783b, dst_is_reg=True, mt=True, count_log2=_c))
    # STSM.16.M88.*
    _TABLE[f"STSM.16.M88.{_ct}"] = (lambda p, ctx=None, _c=_cl:
        _lds_matrix_impl(p, opcode=0x7844, dst_is_reg=False, mt=False, count_log2=_c))
    # STSM.16.MT88.* (covers MT88.4 already, re-registered here for completeness)
    _TABLE[f"STSM.16.MT88.{_ct}"] = (lambda p, ctx=None, _c=_cl:
        _lds_matrix_impl(p, opcode=0x7844, dst_is_reg=False, mt=True, count_log2=_c))


# --- UTCBAR family -------------------------------------------------------
@register("UTCBAR")
def enc_utcbar(p: ParsedInsn) -> int:
    """UTCBAR [URa], URb — tensor-core barrier."""
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"UTCBAR arity {len(p.operands)}")
    ura = _parse_ur_addr(p.operands[0])
    urb, _ = parse_ureg(p.operands[1])
    w = 0
    w = set_bits(w, 0, 16, 0x73e9)
    w = set_bits(w, 24, 8, ura)
    w = set_bits(w, 32, 8, urb)
    w = set_bits(w, 64, 8, 0xff)
    w = set_bits(w, 91, 1, 1)
    return _apply_pred(w, p.pred)


@register("UTCBAR.2CTA.MULTICAST")
def enc_utcbar_2cta_multicast(p: ParsedInsn) -> int:
    """UTCBAR.2CTA.MULTICAST [URa], URb, URc"""
    if len(p.operands) != 3:
        raise UnsupportedInstruction(f"UTCBAR.2CTA.MULTICAST arity {len(p.operands)}")
    ura = _parse_ur_addr(p.operands[0])
    urb, _ = parse_ureg(p.operands[1])
    urc, _ = parse_ureg(p.operands[2])
    w = 0
    w = set_bits(w, 0, 16, 0x73e9)
    w = set_bits(w, 24, 8, ura)
    w = set_bits(w, 32, 8, urb)
    w = set_bits(w, 64, 8, urc)
    w = set_bits(w, 72, 8, 0x08)                # 2CTA flag
    w = set_bits(w, 80, 8, 0x20)                # MULTICAST flag
    w = set_bits(w, 91, 1, 1)
    return _apply_pred(w, p.pred)


# --- ACQBULK -------------------------------------------------------------
@register("ACQBULK")
def enc_acqbulk(p: ParsedInsn) -> int:
    if p.operands:
        raise UnsupportedInstruction("ACQBULK has no operands")
    w = 0
    w = set_bits(w, 0, 16, 0x782e)
    return _apply_pred(w, p.pred)


# --- UTMACCTL.IV --------------------------------------------------------
@register("UTMACCTL.IV")
def enc_utmacctl_iv(p: ParsedInsn) -> int:
    if len(p.operands) != 1:
        raise UnsupportedInstruction(f"UTMACCTL.IV arity {len(p.operands)}")
    ur = _parse_ur_addr(p.operands[0])
    w = 0
    w = set_bits(w, 0, 16, 0x79b9)
    w = set_bits(w, 24, 8, ur)
    w = set_bits(w, 91, 1, 1)
    return _apply_pred(w, p.pred)


# --- UTMAPF.L2.3D -------------------------------------------------------
@register("UTMAPF.L2.3D")
def enc_utmapf_l2_3d(p: ParsedInsn) -> int:
    """UTMAPF.L2.3D [URa], [URb] — prefetch L2."""
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"UTMAPF.L2.3D arity {len(p.operands)}")
    ura = _parse_ur_addr(p.operands[0])
    urb = _parse_ur_addr(p.operands[1])
    w = 0
    w = set_bits(w, 0, 16, 0x75b8)
    w = set_bits(w, 24, 8, urb)
    w = set_bits(w, 32, 8, ura)
    w = set_bits(w, 80, 8, 0x01)
    w = set_bits(w, 91, 1, 1)
    return _apply_pred(w, p.pred)


# UTMALDG.2D/3D/4D/5D and their .MULTICAST/.2CTA variants are registered
# above via the _mk_utmaldg loop. UTMASTG.2D:
@register("UTMASTG.2D")
def enc_utmastg_2d(p: ParsedInsn) -> int:
    if len(p.operands) != 2:
        raise UnsupportedInstruction("UTMASTG.2D arity")
    ura = _parse_ur_addr(p.operands[0])
    urb = _parse_ur_addr(p.operands[1])
    w = 0
    w = set_bits(w, 0, 16, 0x73b5)
    w = set_bits(w, 24, 8, urb)
    w = set_bits(w, 32, 8, ura)
    w = set_bits(w, 72, 8, 0x80)                # 2D flag
    w = set_bits(w, 91, 1, 1)
    return _apply_pred(w, p.pred)


# --- UTCCP tensor copy --------------------------------------------------
@register("UTCCP.T.S.4x32dp128bit")
def enc_utccp_t_s_4x32dp128bit(p: ParsedInsn) -> int:
    """UTCCP.T.S.4x32dp128bit tmem[URa (+off)?], gdesc[URb]"""
    if len(p.operands) != 2:
        raise UnsupportedInstruction("UTCCP arity")
    ura, off = _parse_tmem(p.operands[0])
    urb = _parse_gdesc(p.operands[1])
    w = 0
    w = set_bits(w, 0, 16, 0x79e7)
    w = set_bits(w, 24, 8, ura)
    w = set_bits(w, 32, 8, urb)
    w = set_bits(w, 40, 24, off & ((1 << 24) - 1))
    w = set_bits(w, 80, 8, 0x10)
    w = set_bits(w, 88, 8, 0x09)
    return _apply_pred(w, p.pred)


def _parse_idesc(tok: str) -> int:
    m = re.fullmatch(r"idesc\[\s*(UR\d+|URZ)\s*\]", tok.strip())
    if not m:
        raise UnsupportedInstruction(f"bad idesc {tok!r}")
    ur, _ = parse_ureg(m.group(1))
    return ur


def _parse_gdesc_or_tmem(tok: str) -> tuple[int, str]:
    """Accept either gdesc[URx] or tmem[URx] operand 0 of UTC*MMA."""
    tok = tok.strip()
    if tok.startswith("gdesc["):
        return _parse_gdesc(tok), "gdesc"
    if tok.startswith("tmem["):
        ur, _ = _parse_tmem(tok)
        return ur, "tmem"
    raise UnsupportedInstruction(f"UTC*MMA op0 {tok!r}")


def _utc_mma_impl(p: ParsedInsn, *, byte9: int, cta2: bool,
                  omma: bool = False) -> int:
    """UTCHMMA / UTCQMMA / UTCHMMA.2CTA / UTCQMMA.2CTA / UTCOMMA[.2CTA].4X

    Supports two operand-0 forms:
      gdesc-first : gdescA, gdescB, tmemA(dst), tmemB(acc), idesc, [tmemC,] Ps
      tmem-first  : tmemA,  gdesc,  tmemB,      tmemC,      idesc, Ps
    """
    n = len(p.operands)
    if n not in (6, 7):
        raise UnsupportedInstruction(f"UTC*MMA arity {n}")
    op0, op0_type = _parse_gdesc_or_tmem(p.operands[0])

    extra_imm = 0
    if op0_type == "gdesc":
        g_a = op0
        g_b = _parse_gdesc(p.operands[1])
        t_a, _ = _parse_tmem(p.operands[2])
        t_b, _ = _parse_tmem(p.operands[3])
        _parse_idesc(p.operands[4])
        if n == 7:
            # Distinguish the two 7-operand forms:
            #   (a) ..., tmemC, Ps           — extra accumulator tmem
            #   (b) ..., Ps,     0x<imm>     — extra immediate control flag
            op5 = p.operands[5].strip()
            if op5.startswith("tmem["):
                t_c, _ = _parse_tmem(op5)
                ps_tok = p.operands[6]
            else:
                t_c = 0xff
                ps_tok = op5
                extra_imm = parse_imm(p.operands[6]) & 0xff
        else:
            t_c = 0xff
            ps_tok = p.operands[5]
    else:
        # tmem-first: op0=tmemA, op1=gdesc, op2=tmemB, op3=tmemC, op4=idesc, op5=Ps
        t_a = op0
        g_b = _parse_gdesc(p.operands[1])
        t_b, _ = _parse_tmem(p.operands[2])
        t_c, _ = _parse_tmem(p.operands[3])
        _parse_idesc(p.operands[4])
        g_a = 0x00                       # unused in tmem-first form
        ps_tok = p.operands[5] if n >= 6 else None
    ps_idx, ps_neg = _parse_pred_src(ps_tok)

    w = 0
    # Opcode selection:
    #   tmem-first non-QMMA:              0x79ea  (bit 10 = 1)
    #   tmem-first QMMA:                  0x79ea
    #   gdesc-first HMMA/OMMA:            0x75ea
    #   gdesc-first QMMA 7-op:            0x7dea
    #   gdesc-first HMMA/QMMA 6-op:       0x75ea
    # The "7-op with extra tmemC" form uses 0x7dea for QMMA; 0x75ea otherwise.
    # The "7-op with Ps + imm" form uses 0x75ea regardless.
    seven_op_with_tmemC = (n == 7) and (extra_imm == 0) and (op0_type == "gdesc")
    if op0_type == "tmem":
        opcode = 0x79ea
    elif (byte9 == 0x03) and seven_op_with_tmemC:
        opcode = 0x7dea
    else:
        opcode = 0x75ea
    w = set_bits(w, 0, 16, opcode)
    if op0_type == "gdesc":
        w = set_bits(w, 24, 8, g_a)          # gdesc A at byte 3
        w = set_bits(w, 32, 8, g_b)          # gdesc B at byte 4
    else:
        w = set_bits(w, 24, 8, t_a)          # tmem A at byte 3 (dst)
        w = set_bits(w, 32, 8, g_b)          # gdesc at byte 4
    # tmem B at byte 5 (both forms; holds tmemB for gdesc-first, tmemC for
    # tmem-first because of the text-op re-ordering).
    w = set_bits(w, 40, 8, t_c if op0_type == "tmem" else t_b)
    w = set_bits(w, 48, 8, 0xff if op0_type == "tmem" else t_c)
    w = set_bits(w, 64, 8, t_b if op0_type == "tmem" else t_a)
    # byte 9: main flag byte; extra imm (e.g. UTCHMMA.2CTA "0x8" tail) is
    # encoded as (imm << 3) OR'd in.
    w = set_bits(w, 72, 8, byte9 | ((extra_imm & 0x1f) << 3))
    # byte 10 flags: 2CTA bit 85, QMMA bit 87.
    byte10 = 0x00
    if cta2:
        byte10 |= 0x20
    if byte9 == 0x03:
        byte10 |= 0x80
    w = set_bits(w, 80, 8, byte10)
    if omma:
        w = set_bits(w, 63, 1, 1)
    # Ps idx at bits 87..89, bit 90 = neg.
    w = set_bits(w, 87, 3, ps_idx)
    w = set_bits(w, 90, 1, ps_neg)
    w = set_bits(w, 91, 1, 1)
    return _apply_pred(w, p.pred)


@register("UTCHMMA")
def enc_utchmma(p: ParsedInsn) -> int:
    return _utc_mma_impl(p, byte9=0x00, cta2=False)


@register("UTCHMMA.2CTA")
def enc_utchmma_2cta(p: ParsedInsn) -> int:
    return _utc_mma_impl(p, byte9=0x00, cta2=True)


@register("UTCQMMA")
def enc_utcqmma(p: ParsedInsn) -> int:
    return _utc_mma_impl(p, byte9=0x03, cta2=False)


@register("UTCQMMA.2CTA")
def enc_utcqmma_2cta(p: ParsedInsn) -> int:
    return _utc_mma_impl(p, byte9=0x03, cta2=True)


@register("UTCOMMA.4X")
def enc_utcomma_4x(p: ParsedInsn) -> int:
    return _utc_mma_impl(p, byte9=0x00, cta2=False, omma=True)


@register("UTCOMMA.2CTA.4X")
def enc_utcomma_2cta_4x(p: ParsedInsn) -> int:
    return _utc_mma_impl(p, byte9=0x00, cta2=True, omma=True)


@register("UTCCP.T.S.2CTA.128dp128bit")
def enc_utccp_2cta_128dp128bit(p: ParsedInsn) -> int:
    if len(p.operands) != 2:
        raise UnsupportedInstruction("UTCCP.128dp arity")
    ura, off = _parse_tmem(p.operands[0])
    urb = _parse_gdesc(p.operands[1])
    w = 0
    w = set_bits(w, 0, 16, 0x79e7)
    w = set_bits(w, 24, 8, ura)
    w = set_bits(w, 32, 8, urb)
    w = set_bits(w, 40, 24, off & ((1 << 24) - 1))
    w = set_bits(w, 80, 8, 0x38)
    w = set_bits(w, 88, 8, 0x08)
    return _apply_pred(w, p.pred)


@register("UTCCP.T.S.2CTA.4x32dp128bit")
def enc_utccp_2cta_t_s_4x32dp128bit(p: ParsedInsn) -> int:
    if len(p.operands) != 2:
        raise UnsupportedInstruction("UTCCP.2CTA arity")
    ura, off = _parse_tmem(p.operands[0])
    urb = _parse_gdesc(p.operands[1])
    w = 0
    w = set_bits(w, 0, 16, 0x79e7)
    w = set_bits(w, 24, 8, ura)
    w = set_bits(w, 32, 8, urb)
    w = set_bits(w, 40, 24, off & ((1 << 24) - 1))
    w = set_bits(w, 80, 8, 0x30)                # .2CTA flag (bit 85) + 0x10
    w = set_bits(w, 88, 8, 0x09)
    return _apply_pred(w, p.pred)


@register("UTMACMDFLUSH")
def enc_utmacmdflush(p: ParsedInsn) -> int:
    if p.operands:
        raise UnsupportedInstruction("UTMACMDFLUSH has no operands")
    w = 0
    w = set_bits(w, 0, 16, 0x79b7)
    return _apply_pred(w, p.pred)


@register("NOP")
def enc_nop(p: ParsedInsn) -> int:
    if p.operands:
        raise UnsupportedInstruction("NOP with operands")
    # Observed baseline: 18 79 00 00 00 00 00 00 00 00 00 00 | 00 c0 0f 00
    #  low 16-bit = 0x7918
    w = 0
    w = set_bits(w, 0, 16, 0x7918)
    return _apply_pred(w, p.pred)


@register("EXIT")
def enc_exit(p: ParsedInsn) -> int:
    if p.operands:
        raise UnsupportedInstruction("EXIT with operands")
    # bits 0..11 = 0x94d, bits 12..15 = pred guard, then a fixed high region.
    w = 0
    w = set_bits(w, 0, 16, 0x794d)   # PT form
    # Bytes 10..11 (observed LE) = 0x80, 0x03 -> bits 80..95 = 0x0380.
    w |= (0x0380 << 80)
    return _apply_pred(w, p.pred)


# Source-type selector bits observed at byte 10 (bits 80..95). When the
# "source operand 2" is a URx, bit 83 appears set for many opcodes. Track
# empirically per opcode.


def _encode_cbank_direct(bank: int, off: int, shift: int) -> int:
    """Encode a direct c[bank][off] reference into the 32-bit field that
    lives at bits 32..63 of LDC-family instructions. Empirically the offset
    is left-shifted and OR'd with (bank << 22)."""
    if off < 0 or off >= (1 << 22):
        raise UnsupportedInstruction(f"const offset {off:#x} out of range")
    if bank < 0 or bank >= 32:
        raise UnsupportedInstruction(f"const bank {bank:#x} out of range")
    return ((bank & 0x1f) << 22) | ((off & ((1 << 22) - 1)) << shift)


def _ldc_family(p: ParsedInsn, *, opcode: int, dtype: int, shift: int,
                is_ureg_dst: bool) -> int:
    """LDC/LDC.64 or LDCU/LDCU.64/LDCU.128. Supports c[bank][imm] direct form.

    dtype: 0x08=.32 (default), 0x0a=.64, 0x0c=.128.
    """
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"LDC arity {len(p.operands)}")
    dst = p.operands[0]
    src = p.operands[1].strip()
    if is_ureg_dst:
        drd, _ = parse_ureg(dst)
    else:
        drd, _ = parse_reg(dst)

    bank, off, indirect = parse_const(src)
    w = 0
    w = set_bits(w, 0, 16, opcode)
    w = set_bits(w, 16, 8, drd)
    w = set_bits(w, 24, 8, indirect)
    w = set_bits(w, 32, 32, _encode_cbank_direct(bank, off, shift))
    w = set_bits(w, 72, 8, dtype)
    if is_ureg_dst:
        # LDCU has bit 91 set (UR-destination selector, mirrors the general
        # "UR form" bit we saw in other opcodes).
        w = set_bits(w, 91, 1, 1)
    return _apply_pred(w, p.pred)


@register("LDC")
def enc_ldc(p: ParsedInsn) -> int:
    return _ldc_family(p, opcode=0x7b82, dtype=0x08, shift=6, is_ureg_dst=False)


@register("LDC.64")
def enc_ldc_64(p: ParsedInsn) -> int:
    return _ldc_family(p, opcode=0x7b82, dtype=0x0a, shift=6, is_ureg_dst=False)


@register("LDC.U8")
def enc_ldc_u8(p: ParsedInsn) -> int:
    return _ldc_family(p, opcode=0x7b82, dtype=0x00, shift=6, is_ureg_dst=False)


@register("LDC.S8")
def enc_ldc_s8(p: ParsedInsn) -> int:
    return _ldc_family(p, opcode=0x7b82, dtype=0x02, shift=6, is_ureg_dst=False)


@register("LDC.U16")
def enc_ldc_u16(p: ParsedInsn) -> int:
    return _ldc_family(p, opcode=0x7b82, dtype=0x04, shift=6, is_ureg_dst=False)


@register("LDC.S16")
def enc_ldc_s16(p: ParsedInsn) -> int:
    return _ldc_family(p, opcode=0x7b82, dtype=0x06, shift=6, is_ureg_dst=False)


@register("LDC.128")
def enc_ldc_128(p: ParsedInsn) -> int:
    return _ldc_family(p, opcode=0x7b82, dtype=0x0c, shift=6, is_ureg_dst=False)


@register("LDCU")
def enc_ldcu(p: ParsedInsn) -> int:
    return _ldc_family(p, opcode=0x77ac, dtype=0x08, shift=5, is_ureg_dst=True)


@register("LDCU.64")
def enc_ldcu_64(p: ParsedInsn) -> int:
    return _ldc_family(p, opcode=0x77ac, dtype=0x0a, shift=5, is_ureg_dst=True)


@register("LDCU.128")
def enc_ldcu_128(p: ParsedInsn) -> int:
    return _ldc_family(p, opcode=0x77ac, dtype=0x0c, shift=5, is_ureg_dst=True)


SREG_TABLE: dict[str, int] = {
    "SR_LANEID":    0x00,
    "SR_CLOCKLO":   0x20,
    "SR_TID.X":     0x21,
    "SR_TID.Y":     0x22,
    "SR_TID.Z":     0x23,
    "SR_CTAID.X":   0x25,
    "SR_CTAID.Y":   0x26,
    "SR_CTAID.Z":   0x27,
    "SR_NTID.X":    0x29,
    "SR_NTID.Y":    0x2a,
    "SR_NTID.Z":    0x2b,
    "SR_CgaCtaId":  0x88,
    "SR_CgaSize":   0x8a,
}


@register("CS2R.32")
def enc_cs2r_32(p: ParsedInsn) -> int:
    """CS2R.32 Rd, SR_<name> — 32-bit special register move."""
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"CS2R.32 arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    sr = p.operands[1].strip()
    sr_idx = SREG_TABLE.get(sr, None)
    if sr_idx is None:
        if sr == "SRZ":
            sr_idx = 0xff
        else:
            raise UnsupportedInstruction(f"CS2R.32 SR {sr!r}")
    w = 0
    w = set_bits(w, 0, 16, 0x7805)
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 72, 8, sr_idx)
    return _apply_pred(w, p.pred)


def _i2f_impl(p: ParsedInsn, *, byte9: int, byte10: int,
              opcode_r: int = 0x7306, opcode_ur: int = 0x7d06) -> int:
    """Int-to-float conversion with a selectable byte-9/byte-10 flag pair.

    byte9 holds input type + rounding-mode flags; byte10 holds the output
    width field. The 32/16-bit input family uses opcode 0x7306/0x7d06 and
    the 64-bit input family uses 0x7312/0x7d12.
    """
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"{p.mnemonic} arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    src = p.operands[1]
    w = 0
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 72, 8, byte9)
    w = set_bits(w, 80, 8, byte10)
    if src.startswith("UR") or src.replace(".reuse", "").strip() == "URZ":
        urb, _ = parse_ureg(src)
        w = set_bits(w, 0, 16, opcode_ur)
        w = set_bits(w, 32, 8, urb)
        w = set_bits(w, 91, 1, 1)
    elif src.startswith("R") or src == "RZ":
        rb, _ = parse_reg(src)
        w = set_bits(w, 0, 16, opcode_r)
        w = set_bits(w, 32, 8, rb)
    else:
        raise UnsupportedInstruction(f"{p.mnemonic} src {src!r}")
    return _apply_pred(w, p.pred)


@register("I2F.RP")
def enc_i2f_rp(p: ParsedInsn) -> int:
    return _i2f_impl(p, byte9=0x94, byte10=0x20)


@register("I2F.U32.RP")
def enc_i2f_u32_rp(p: ParsedInsn) -> int:
    return _i2f_impl(p, byte9=0x90, byte10=0x20)


@register("I2F.U16")
def enc_i2f_u16(p: ParsedInsn) -> int:
    return _i2f_impl(p, byte9=0x10, byte10=0x10)


@register("I2F.S64")
def enc_i2f_s64(p: ParsedInsn) -> int:
    return _i2f_impl(p, byte9=0x14, byte10=0x30,
                     opcode_r=0x7312, opcode_ur=0x7d12)


@register("I2F.U64.RP")
def enc_i2f_u64_rp(p: ParsedInsn) -> int:
    return _i2f_impl(p, byte9=0x90, byte10=0x30,
                     opcode_r=0x7312, opcode_ur=0x7d12)


@register("I2F.F64")
def enc_i2f_f64(p: ParsedInsn) -> int:
    return _i2f_impl(p, byte9=0x1c, byte10=0x20,
                     opcode_r=0x7312, opcode_ur=0x7d12)


@register("REDUX")
def enc_redux(p: ParsedInsn) -> int:
    """REDUX URd, Ra — warp-shuffle reduction into a uniform register."""
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"REDUX arity {len(p.operands)}")
    urd, _ = parse_ureg(p.operands[0])
    ra, _ = parse_reg(p.operands[1])
    w = 0
    w = set_bits(w, 0, 16, 0x73c4)
    w = set_bits(w, 16, 8, urd)
    w = set_bits(w, 24, 8, ra)
    return _apply_pred(w, p.pred)


@register("FCHK")
def enc_fchk(p: ParsedInsn) -> int:
    """FCHK Pd, Ra, Rb — FP finite-check, sets Pd if Ra/Rb are both finite."""
    if len(p.operands) != 3:
        raise UnsupportedInstruction(f"FCHK arity {len(p.operands)}")
    pd = _parse_pred_dst(p.operands[0])
    ra, _ = parse_reg(p.operands[1])
    rb, _ = parse_reg(p.operands[2])
    w = 0
    w = set_bits(w, 0, 16, 0x7302)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 32, 8, rb)
    w = set_bits(w, 81, 3, pd)
    return _apply_pred(w, p.pred)


def _vote_impl(p: ParsedInsn, *, subop: int) -> int:
    """VOTE.<subop> Rd, Pd_vote, Ps  or  VOTE.<subop> Pd_vote, Ps."""
    if len(p.operands) == 3:
        rd, _ = parse_reg(p.operands[0])
        pd_vote = _parse_pred_dst(p.operands[1])
        ps_idx, ps_neg = _parse_pred_src(p.operands[2])
    elif len(p.operands) == 2:
        rd = 0xff                               # no-Rd form: RZ in the Rd slot
        pd_vote = _parse_pred_dst(p.operands[0])
        ps_idx, ps_neg = _parse_pred_src(p.operands[1])
    else:
        raise UnsupportedInstruction(f"VOTE arity {len(p.operands)}")
    w = 0
    w = set_bits(w, 0, 16, 0x7806)
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 72, 8, subop)
    w = set_bits(w, 81, 3, pd_vote)
    w = set_bits(w, 87, 3, ps_idx)
    w = set_bits(w, 90, 1, ps_neg)
    return _apply_pred(w, p.pred)


@register("VOTE.ANY")
def enc_vote_any(p: ParsedInsn) -> int:
    return _vote_impl(p, subop=0x01)


@register("VOTE.ALL")
def enc_vote_all(p: ParsedInsn) -> int:
    return _vote_impl(p, subop=0x00)


@register("VOTE.UNI")
def enc_vote_uni(p: ParsedInsn) -> int:
    return _vote_impl(p, subop=0x02)


@register("USETMAXREG.DEALLOC.CTAPOOL")
def enc_usetmaxreg_dealloc_ctapool(p: ParsedInsn) -> int:
    """USETMAXREG.DEALLOC.CTAPOOL imm — release register-pool lanes back to CTA."""
    if len(p.operands) != 1:
        raise UnsupportedInstruction(f"USETMAXREG arity {len(p.operands)}")
    imm = parse_imm(p.operands[0]) & 0xff
    w = 0
    w = set_bits(w, 0, 16, 0x79c8)
    w = set_bits(w, 32, 8, imm)
    w = set_bits(w, 72, 8, 0x05)
    w = set_bits(w, 80, 8, 0x0e)
    w = set_bits(w, 88, 8, 0x08)
    return _apply_pred(w, p.pred)


@register("R2UR.BROADCAST")
def enc_r2ur_broadcast(p: ParsedInsn) -> int:
    """R2UR.BROADCAST URd, Ra — broadcast lane-0 value into a uniform reg."""
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"R2UR.BROADCAST arity {len(p.operands)}")
    urd, _ = parse_ureg(p.operands[0])
    ra, ra_reuse = parse_reg(p.operands[1])
    w = 0
    w = set_bits(w, 0, 16, 0x72ca)
    w = set_bits(w, 16, 8, urd)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 80, 8, 0x8e)
    if ra_reuse:
        w = set_bits(w, 122, 1, 1)
    return _apply_pred(w, p.pred)


def _i2fp_impl(p: ParsedInsn, *, byte9: int, byte10: int) -> int:
    """I2FP — the 'packed' int→float form (different opcode family from I2F).

    Opcode 0x7245 for Ra input, 0x7c45 for UR input (bit 91 set).
    """
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"{p.mnemonic} arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    src = p.operands[1]
    w = 0
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 72, 8, byte9)
    w = set_bits(w, 80, 8, byte10)
    if src.startswith("UR") or src.replace(".reuse", "").strip() == "URZ":
        urb, _ = parse_ureg(src)
        w = set_bits(w, 0, 16, 0x7c45)
        w = set_bits(w, 32, 8, urb)
        w = set_bits(w, 91, 1, 1)
    else:
        rb, _ = parse_reg(src)
        w = set_bits(w, 0, 16, 0x7245)
        w = set_bits(w, 32, 8, rb)
    return _apply_pred(w, p.pred)


@register("I2FP.F32.S32")
def enc_i2fp_f32_s32(p: ParsedInsn) -> int:
    return _i2fp_impl(p, byte9=0x14, byte10=0x20)


@register("I2FP.F32.U32")
def enc_i2fp_f32_u32(p: ParsedInsn) -> int:
    return _i2fp_impl(p, byte9=0x10, byte10=0x20)


@register("I2FP.F32.U32.RZ")
def enc_i2fp_f32_u32_rz(p: ParsedInsn) -> int:
    return _i2fp_impl(p, byte9=0xd0, byte10=0x20)


@register("I2FP.F32.S32.RZ")
def enc_i2fp_f32_s32_rz(p: ParsedInsn) -> int:
    return _i2fp_impl(p, byte9=0xd4, byte10=0x20)


@register("FFMA.RM")
def enc_ffma_rm(p: ParsedInsn) -> int:
    """FFMA.RM — FFMA with round-toward-negative-infinity rounding mode."""
    if len(p.operands) != 4:
        raise UnsupportedInstruction(f"FFMA.RM arity {len(p.operands)}")
    w = enc_ffma(p)
    w = set_bits(w, 78, 1, 1)
    return w


@register("FFMA.SAT")
def enc_ffma_sat(p: ParsedInsn) -> int:
    """FFMA.SAT — FFMA with saturation to [0,1]."""
    if len(p.operands) != 4:
        raise UnsupportedInstruction(f"FFMA.SAT arity {len(p.operands)}")
    w = enc_ffma(p)
    # SAT flag at bit 77.
    w = set_bits(w, 77, 1, 1)
    return w


@register("ULEA.HI")
def enc_ulea_hi(p: ParsedInsn) -> int:
    """ULEA.HI URd, URa, URb, URc, shift — uniform analogue of LEA.HI."""
    if len(p.operands) != 5:
        raise UnsupportedInstruction(f"ULEA.HI arity {len(p.operands)}")
    urd, _ = parse_ureg(p.operands[0])
    ura, ra_reuse = parse_ureg(p.operands[1])
    urb, rb_reuse = parse_ureg(p.operands[2])
    urc, rc_reuse = parse_ureg(p.operands[3])
    shift = parse_imm(p.operands[4]) & 0x1f
    w = 0
    w = set_bits(w, 0, 16, 0x7291)
    w = set_bits(w, 16, 8, urd)
    w = set_bits(w, 24, 8, ura)
    w = set_bits(w, 32, 8, urb)
    w = set_bits(w, 64, 8, urc)
    w = set_bits(w, 72, 8, shift << 3)
    w = set_bits(w, 80, 8, 0x8f)
    w = set_bits(w, 88, 8, 0x0f)                 # UR form
    if ra_reuse: w = set_bits(w, 122, 1, 1)
    if rb_reuse: w = set_bits(w, 123, 1, 1)
    if rc_reuse: w = set_bits(w, 124, 1, 1)
    return _apply_pred(w, p.pred)


@register("VOTEU.ALL")
def enc_voteu_all(p: ParsedInsn) -> int:
    """VOTEU.ALL UPd, Ps  (2-operand form)."""
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"VOTEU.ALL arity {len(p.operands)}")
    upd = _parse_upred_dst(p.operands[0])
    ps_idx, ps_neg = _parse_pred_src(p.operands[1])
    w = 0
    w = set_bits(w, 0, 16, 0x7886)
    w = set_bits(w, 16, 8, 0xff)
    # ALL subop at byte 9 = 0x00 (vs ANY = 0x01).
    w = set_bits(w, 81, 3, upd)
    w = set_bits(w, 87, 3, ps_idx)
    w = set_bits(w, 90, 1, ps_neg)
    return _apply_pred(w, p.pred)


@register("LEA.HI")
def enc_lea_hi(p: ParsedInsn) -> int:
    """LEA.HI Rd, Ra, {Rb|URb}, Rc, shift — 5-operand, high-half LEA."""
    if len(p.operands) != 5:
        raise UnsupportedInstruction(f"LEA.HI arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    ra, ra_reuse = parse_reg(p.operands[1])
    b_tok = p.operands[2]
    rc, rc_reuse = parse_reg(p.operands[3])
    shift = parse_imm(p.operands[4]) & 0x1f
    w = 0
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 64, 8, rc)
    w = set_bits(w, 72, 8, shift << 3)
    # byte 10 = 0x8f: bit 80=1 (HI flag), bits 81-83=111 (Pd=PT), bit 87=1.
    w = set_bits(w, 80, 8, 0x8f)
    # byte 11 = 0x07: implicit Ps=PT.
    w = set_bits(w, 88, 8, 0x07)
    if b_tok.startswith("UR") or b_tok.replace(".reuse", "").strip() == "URZ":
        urb, rb_reuse = parse_ureg(b_tok)
        w = set_bits(w, 0, 16, 0x7c11)
        w = set_bits(w, 32, 8, urb)
        w = set_bits(w, 91, 1, 1)
    else:
        rb, rb_reuse = parse_reg(b_tok)
        w = set_bits(w, 0, 16, 0x7211)
        w = set_bits(w, 32, 8, rb)
    if ra_reuse: w = set_bits(w, 122, 1, 1)
    if rb_reuse: w = set_bits(w, 123, 1, 1)
    if rc_reuse: w = set_bits(w, 124, 1, 1)
    return _apply_pred(w, p.pred)


@register("S2R")
def enc_s2r(p: ParsedInsn) -> int:
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"S2R arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    sr = p.operands[1].strip()
    if sr == "SRZ":
        sr_idx = 0xff
    elif sr in SREG_TABLE:
        sr_idx = SREG_TABLE[sr]
    else:
        raise UnsupportedInstruction(f"S2R SR {sr!r}")
    w = 0
    w = set_bits(w, 0, 16, 0x7919)
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 72, 8, sr_idx)
    return _apply_pred(w, p.pred)


@register("S2UR")
def enc_s2ur(p: ParsedInsn) -> int:
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"S2UR arity {len(p.operands)}")
    urd, _ = parse_ureg(p.operands[0])
    sr = p.operands[1].strip()
    if sr == "SRZ":
        sr_idx = 0xff
    elif sr in SREG_TABLE:
        sr_idx = SREG_TABLE[sr]
    else:
        raise UnsupportedInstruction(f"S2UR SR {sr!r}")
    w = 0
    w = set_bits(w, 0, 16, 0x79c3)
    w = set_bits(w, 16, 8, urd)
    w = set_bits(w, 72, 8, sr_idx)
    return _apply_pred(w, p.pred)


@register("CS2R")
def enc_cs2r(p: ParsedInsn) -> int:
    # Observed: CS2R Rd, SRZ => "zero out a 64-bit register pair" idiom.
    # Fixed pattern: 05 78 Rd 00 00 00 00 00 00 ff 01 00 00 <sched>
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"CS2R arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    sr = p.operands[1].strip()
    if sr != "SRZ":
        raise UnsupportedInstruction(f"CS2R SR {sr!r}")
    w = 0
    w = set_bits(w, 0, 16, 0x7805)
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 72, 8, 0xff)
    w = set_bits(w, 80, 8, 0x01)
    return _apply_pred(w, p.pred)


def _parse_pred_dst(tok: str) -> int:
    """Parse a predicate destination like P0, PT (no negation allowed)."""
    if tok == "PT":
        return 7
    m = re.fullmatch(r"P(\d)", tok.strip())
    if not m:
        raise UnsupportedInstruction(f"pred dst {tok!r}")
    return int(m.group(1))


def _parse_upred_dst(tok: str) -> int:
    if tok == "UPT":
        return 7
    m = re.fullmatch(r"UP(\d)", tok.strip())
    if not m:
        raise UnsupportedInstruction(f"upred dst {tok!r}")
    return int(m.group(1))


@register("IADD3.X")
def enc_iadd3_x(p: ParsedInsn) -> int:
    """IADD3.X Rd, Pc1, Pc2, Ra, Rb|URb|imm, Rc, Pc_in, <negate_flag>

    Same field layout as IADD3 but byte9 bit 74 set (.X flag), and bits 87..89
    encode the carry-in predicate index.
    """
    if len(p.operands) not in (7, 8):
        raise UnsupportedInstruction(f"IADD3.X arity {len(p.operands)}")
    pc_in_idx, _ = _parse_pred_src(p.operands[6])
    # 8th operand: negate-out predicate (PT means "no inversion").
    if len(p.operands) == 8:
        neg_out_idx, _ = _parse_pred_src(p.operands[7])
    else:
        neg_out_idx = 7
    p6 = ParsedInsn(pred=p.pred, mnemonic=p.mnemonic, operands=p.operands[:6])
    w = enc_iadd3(p6)
    # Flip .X flag: byte9 = 0xe0 -> 0xe4 (plus neg-out idx in bits 77..79).
    w = set_bits(w, 74, 1, 1)
    w = set_bits(w, 77, 3, neg_out_idx)
    # Bit 80 = 1 only when neg_out is PT (the "unused" sentinel).
    w = set_bits(w, 80, 1, 1 if neg_out_idx == 7 else 0)
    # Replace Ps (bits 87..90) with Pc_in.
    w = set_bits(w, 87, 4, pc_in_idx & 0x7)
    return w


@register("UIADD3.X")
def enc_uiadd3_x(p: ParsedInsn) -> int:
    """Uniform IADD3.X — same layout as IADD3.X with UR operands."""
    if len(p.operands) not in (7, 8):
        raise UnsupportedInstruction(f"UIADD3.X arity {len(p.operands)}")
    pc_in_idx, _ = _parse_pred_src(p.operands[6])
    neg_out_idx = 7
    if len(p.operands) == 8:
        neg_out_idx, _ = _parse_pred_src(p.operands[7])
    p6 = ParsedInsn(pred=p.pred, mnemonic=p.mnemonic, operands=p.operands[:6])
    w = enc_uiadd3(p6)
    w = set_bits(w, 74, 1, 1)
    w = set_bits(w, 77, 3, neg_out_idx)
    w = set_bits(w, 80, 1, 1 if neg_out_idx == 7 else 0)
    w = set_bits(w, 87, 4, pc_in_idx & 0x7)
    return w


@register("UIADD3")
def enc_uiadd3(p: ParsedInsn) -> int:
    """UIADD3 URd, UPcarry1, UPcarry2, URa, URb|imm, URc

    Same field layout as IADD3 but uniform registers. bit 91 is the "UR in
    operand-B slot" selector and is always set for UIADD3. The URb operand
    can be prefixed with `-` for negation, stored at bit 63.
    """
    if len(p.operands) != 6:
        raise UnsupportedInstruction(f"UIADD3 arity {len(p.operands)}")
    urd, _ = parse_ureg(p.operands[0])
    pc1 = _parse_upred_dst(p.operands[1])
    pc2 = _parse_upred_dst(p.operands[2])
    a_tok = p.operands[3]
    b_tok = p.operands[4]
    urc, urc_reuse = parse_ureg(p.operands[5])

    ura_neg = False
    if a_tok.startswith("-") or a_tok.startswith("~"):
        ura_neg = True
        a_tok = a_tok[1:].strip()
    ura, ura_reuse = parse_ureg(a_tok)

    w = 0
    w = set_bits(w, 16, 8, urd)
    w = set_bits(w, 24, 8, ura)
    w = set_bits(w, 64, 8, urc)
    # bits 72..79 = 0xe0 (fixed), plus bit 72 is the URa-negate flag.
    w = set_bits(w, 72, 8, 0xe0 | (1 if ura_neg else 0))
    w = set_bits(w, 80, 1, 1)
    w = set_bits(w, 81, 3, pc1)
    w = set_bits(w, 84, 3, pc2)
    w = set_bits(w, 87, 1, 1)
    w = set_bits(w, 88, 3, 7)                   # Ps = UPT (fixed)
    w = set_bits(w, 91, 1, 1)                   # UR-form flag, always on

    # Detect UR operand (possibly negated or bitwise-negated) vs immediate.
    # `-`/`~` on a UR operand sets the B-negate flag at bit 63; on an immediate
    # the sign stays in the literal.
    b_has_neg = b_tok.startswith("-") or b_tok.startswith("~")
    b_stripped = b_tok[1:].strip() if b_has_neg else b_tok
    is_ureg = b_stripped.startswith("UR") or b_stripped.replace(".reuse", "").strip() == "URZ"

    if is_ureg:
        urb, urb_reuse = parse_ureg(b_stripped)
        w = set_bits(w, 0, 16, 0x7290)
        w = set_bits(w, 32, 8, urb)
        if b_has_neg:
            w = set_bits(w, 63, 1, 1)
    else:
        imm = parse_imm(b_tok) & 0xffffffff
        w = set_bits(w, 0, 16, 0x7890)
        w = set_bits(w, 32, 32, imm)
        urb_reuse = False

    if ura_reuse:
        w = set_bits(w, 122, 1, 1)
    if urb_reuse:
        w = set_bits(w, 123, 1, 1)
    if urc_reuse:
        w = set_bits(w, 124, 1, 1)
    return _apply_pred(w, p.pred)


def _parse_pred_src(tok: str) -> tuple[int, int]:
    """Parse a source predicate token, returning (idx, neg_flag)."""
    neg = 0
    if tok.startswith("!"):
        neg = 1
        tok = tok[1:]
    # Uniform predicate source: same 3-bit idx, caller uses it as a UP.
    if tok.startswith("U"):
        tok = tok[1:]
    if tok == "PT":
        idx = 7
    else:
        m = re.fullmatch(r"P(\d)", tok.strip())
        if not m:
            raise UnsupportedInstruction(f"pred src {tok!r}")
        idx = int(m.group(1))
    return idx, neg


def _lop3_lut(p: ParsedInsn) -> int:
    """LOP3.LUT [Pd,] Rd, Ra, {R|UR|imm}, Rc, imm_lut, Ps

    Encoding:
      bits 0..15   : 0x7212 (R), 0x7c12 (UR), 0x7812 (imm) — PT-guard base
      bits 16..23  : Rd
      bits 24..31  : Ra
      bits 32..39  : Rb     / or bits 32..63 = imm32
      bits 64..71  : Rc
      bits 72..79  : LUT truth-table (8-bit immediate)
      bit 80       : 0
      bits 81..83  : Pd idx (7=PT when no explicit Pd)
      bits 84..86  : 0 (unused)
      bit 87       : Ps_negate flag (1 for !PT, which is the common case)
      bits 88..90  : Ps idx (7=PT)
      bit 91       : UR-B selector
      bits 92..95  : 0
    """
    ops = list(p.operands)
    # Detect whether the first operand is a predicate (Pd form).
    if ops and (ops[0] == "PT" or re.fullmatch(r"P\d", ops[0])):
        pd_idx = _parse_pred_dst(ops[0])
        ops = ops[1:]
    else:
        pd_idx = 7  # implicit PT
    if len(ops) != 6:
        raise UnsupportedInstruction(f"LOP3.LUT arity {len(ops)}")
    rd, _ = parse_reg(ops[0])
    ra, ra_reuse = parse_reg(ops[1])
    b_tok = ops[2]
    rc, rc_reuse = parse_reg(ops[3])
    lut = parse_imm(ops[4]) & 0xff
    ps_idx, ps_neg = _parse_pred_src(ops[5])

    w = 0
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 64, 8, rc)
    w = set_bits(w, 72, 8, lut)
    w = set_bits(w, 81, 3, pd_idx)
    w = set_bits(w, 87, 1, ps_neg)
    w = set_bits(w, 88, 3, ps_idx)

    if b_tok.startswith("UR") or b_tok.replace(".reuse", "").strip() == "URZ":
        urb, rb_reuse = parse_ureg(b_tok)
        w = set_bits(w, 0, 16, 0x7c12)
        w = set_bits(w, 32, 8, urb)
        w = set_bits(w, 91, 1, 1)
    elif b_tok.startswith("R") or b_tok.replace(".reuse", "").strip() == "RZ":
        rb, rb_reuse = parse_reg(b_tok)
        w = set_bits(w, 0, 16, 0x7212)
        w = set_bits(w, 32, 8, rb)
    else:
        imm = parse_imm(b_tok) & 0xffffffff
        w = set_bits(w, 0, 16, 0x7812)
        w = set_bits(w, 32, 32, imm)
        rb_reuse = False

    if ra_reuse:
        w = set_bits(w, 122, 1, 1)
    if rb_reuse:
        w = set_bits(w, 123, 1, 1)
    if rc_reuse:
        w = set_bits(w, 124, 1, 1)
    return _apply_pred(w, p.pred)


CMP_OP = {"F": 0, "LT": 1, "EQ": 2, "LE": 3, "GT": 4, "NE": 5, "GE": 6, "T": 7}
COMBINE_OP = {"AND": 0, "OR": 1, "XOR": 2}  # observed encoding tbd per usage


def _setp_family(p: ParsedInsn, *, is_uniform: bool) -> int:
    """Common handler for ISETP / UISETP.

    Mnemonic form: {I|UI}SETP.<CMP>[.U32].<COMB>[.EX]  Pd1, Pd2, A, B, Ps [, Pextra]

    Implements the common (non-.EX) subset observed in 70_blackwell_fp16_gemm:
      .AND combiner, with Pd2 = PT.
    """
    parts = p.mnemonic.split(".")
    # parts[0] = "ISETP" or "UISETP"
    if parts[0] not in ("ISETP", "UISETP"):
        raise UnsupportedInstruction(p.mnemonic)
    mods = parts[1:]
    cmp_op = None
    is_u32 = False
    comb = None
    has_ex = False
    for m in mods:
        if m in CMP_OP:
            cmp_op = CMP_OP[m]
        elif m == "U32":
            is_u32 = True
        elif m in COMBINE_OP:
            comb = m
        elif m == "EX":
            has_ex = True
        else:
            raise UnsupportedInstruction(f"SETP modifier {m!r}")
    if cmp_op is None:
        raise UnsupportedInstruction("SETP without cmp op")
    if comb not in ("AND", "OR"):
        raise UnsupportedInstruction(f"SETP combiner {comb!r}")
    expected_arity = 6 if has_ex else 5
    if len(p.operands) != expected_arity:
        raise UnsupportedInstruction(f"SETP arity {len(p.operands)}")

    pd1 = _parse_upred_dst(p.operands[0]) if is_uniform else _parse_pred_dst(p.operands[0])
    pd2 = _parse_upred_dst(p.operands[1]) if is_uniform else _parse_pred_dst(p.operands[1])
    a_tok = p.operands[2]
    b_tok = p.operands[3]
    ps_idx, ps_neg = _parse_pred_src(p.operands[4])
    pc_in_idx = 0
    pc_in_neg = 0
    if has_ex:
        pc_in_idx, pc_in_neg = _parse_pred_src(p.operands[5])

    if is_uniform:
        ra, ra_reuse = parse_ureg(a_tok)
    else:
        ra, ra_reuse = parse_reg(a_tok)

    w = 0
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 64, 8, 0x70)                # fixed
    if has_ex:
        # Pc_in at bits 68..70, with bit 71 = Pc_in negate flag.
        w = set_bits(w, 68, 3, pc_in_idx)
        w = set_bits(w, 71, 1, pc_in_neg)
    # byte 9: bits 76..79 = cmp_op, bit 73 = signed, bit 74 = .OR, bit 72 = .EX
    byte9 = ((cmp_op << 4) | (0 if is_u32 else 2)
             | (4 if comb == "OR" else 0) | (1 if has_ex else 0))
    w = set_bits(w, 72, 8, byte9)
    w = set_bits(w, 80, 1, 0)
    w = set_bits(w, 81, 3, pd1)
    w = set_bits(w, 84, 3, pd2)
    w = set_bits(w, 87, 3, ps_idx)
    w = set_bits(w, 90, 1, ps_neg)

    if is_uniform:
        ur_form_bit = 1
    else:
        ur_form_bit = 0

    if b_tok.startswith("UR") or b_tok.replace(".reuse", "").strip() == "URZ":
        urb, rb_reuse = parse_ureg(b_tok)
        w = set_bits(w, 0, 16, 0x728c if is_uniform else 0x7c0c)
        w = set_bits(w, 32, 8, urb)
        w = set_bits(w, 91, 1, 1)
    elif (not is_uniform) and (b_tok.startswith("R") or b_tok.replace(".reuse", "").strip() == "RZ"):
        rb, rb_reuse = parse_reg(b_tok)
        w = set_bits(w, 0, 16, 0x720c)
        w = set_bits(w, 32, 8, rb)
        w = set_bits(w, 91, 1, ur_form_bit)
    else:
        imm = parse_imm(b_tok) & 0xffffffff
        w = set_bits(w, 0, 16, 0x788c if is_uniform else 0x780c)
        w = set_bits(w, 32, 32, imm)
        rb_reuse = False
        w = set_bits(w, 91, 1, ur_form_bit)

    if ra_reuse:
        w = set_bits(w, 122, 1, 1)
    if rb_reuse:
        w = set_bits(w, 123, 1, 1)
    return _apply_pred(w, p.pred)


# Register the common cmp × type × combine (× .EX) variants the program uses.
for _cmp in ("EQ", "NE", "LT", "LE", "GT", "GE"):
    for _typ in ("", ".U32"):
        for _comb in (".AND", ".OR", ".AND.EX", ".OR.EX"):
            _name = f"ISETP.{_cmp}{_typ}{_comb}"
            _uname = f"U{_name}"
            def _mk(nm, is_u):
                def fn(p, ctx=None, _isu=is_u):
                    return _setp_family(p, is_uniform=_isu)
                fn.__name__ = f"enc_{nm.replace('.', '_')}"
                return fn
            _TABLE[_name] = _mk(_name, False)
            _TABLE[_uname] = _mk(_uname, True)


@register("FADD")
def enc_fadd(p: ParsedInsn) -> int:
    """FADD Rd, [-]Ra, [-]Rb|URb|float-imm"""
    if len(p.operands) != 3:
        raise UnsupportedInstruction(f"FADD arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    a_tok = p.operands[1]
    ra_neg = a_tok.startswith("-")
    if ra_neg:
        a_tok = a_tok[1:].strip()
    ra, ra_reuse = parse_reg(a_tok)
    b_tok = p.operands[2]
    rb_neg = b_tok.startswith("-")
    if rb_neg:
        b_tok = b_tok[1:].strip()

    w = 0
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, ra)
    if ra_neg:
        w = set_bits(w, 72, 1, 1)

    if b_tok.startswith("UR") or b_tok.replace(".reuse", "").strip() == "URZ":
        urb, rb_reuse = parse_ureg(b_tok)
        # FADD with -Ra + UR uses opcode 0x7e21 (bit 9 set); bit 72 stays.
        w = set_bits(w, 0, 16, 0x7e21 if ra_neg else 0x7c21)
        w = set_bits(w, 32, 8, urb)
        w = set_bits(w, 91, 1, 1)
        if rb_neg: w = set_bits(w, 63, 1, 1)
    elif b_tok.startswith("R") or b_tok.replace(".reuse", "").strip() == "RZ":
        rb, rb_reuse = parse_reg(b_tok)
        w = set_bits(w, 0, 16, 0x7221)
        w = set_bits(w, 32, 8, rb)
        if rb_neg: w = set_bits(w, 63, 1, 1)
    else:
        imm = _parse_float_imm(b_tok if not rb_neg else "-" + b_tok)
        w = set_bits(w, 0, 16, 0x7421)
        w = set_bits(w, 32, 32, imm)
        rb_reuse = False

    if ra_reuse:
        w = set_bits(w, 122, 1, 1)
    if rb_reuse:
        # FADD stores Rb.reuse at bit 124 (unusual — FFMA uses 123 for B;
        # FADD lacks a Rc operand so the B slot maps to the C-reuse position).
        w = set_bits(w, 124, 1, 1)
    return _apply_pred(w, p.pred)


@register("FADD.FTZ")
def enc_fadd_ftz(p: ParsedInsn) -> int:
    """FADD.FTZ — identical to FADD plus the flush-to-zero flag at bit 80."""
    w = enc_fadd(p)
    return w | (1 << 80)


@register("FSEL")
def enc_fsel(p: ParsedInsn) -> int:
    """FSEL Rd, Ra, {Rb|URb|float-imm}, Ps"""
    if len(p.operands) != 4:
        raise UnsupportedInstruction(f"FSEL arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    ra, ra_reuse = parse_reg(p.operands[1])
    b_tok = p.operands[2]
    ps_idx, ps_neg = _parse_pred_src(p.operands[3])

    w = 0
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 87, 3, ps_idx)
    w = set_bits(w, 90, 1, ps_neg)

    if b_tok.startswith("UR") or b_tok.replace(".reuse", "").strip() == "URZ":
        urb, rb_reuse = parse_ureg(b_tok)
        w = set_bits(w, 0, 16, 0x7c08)
        w = set_bits(w, 32, 8, urb)
        w = set_bits(w, 91, 1, 1)
    elif b_tok.startswith("R") or b_tok.replace(".reuse", "").strip() == "RZ":
        rb, rb_reuse = parse_reg(b_tok)
        w = set_bits(w, 0, 16, 0x7208)
        w = set_bits(w, 32, 8, rb)
    else:
        imm = _parse_float_imm(b_tok)
        w = set_bits(w, 0, 16, 0x7808)
        w = set_bits(w, 32, 32, imm)
        rb_reuse = False

    if ra_reuse:
        w = set_bits(w, 122, 1, 1)
    if rb_reuse:
        w = set_bits(w, 123, 1, 1)
    return _apply_pred(w, p.pred)


FSETP_CMP_OP = {"F": 0, "LT": 1, "EQ": 2, "LE": 3, "GT": 4, "NE": 5, "GE": 6, "T": 7,
                "NUM": 8, "LTU": 9, "EQU": 10, "LEU": 11, "GTU": 12, "NEU": 13, "GEU": 14}


def _fsetp_impl(p: ParsedInsn, cmp_op: int, comb: str, ftz: bool = False) -> int:
    """FSETP[.FTZ].<CMP>.<COMB> Pd1, Pd2, [|]Ra[|], {Rb|URb|float-imm}, Ps"""
    if len(p.operands) != 5:
        raise UnsupportedInstruction(f"FSETP arity {len(p.operands)}")
    pd1 = _parse_pred_dst(p.operands[0])
    pd2 = _parse_pred_dst(p.operands[1])
    a_tok = p.operands[2].strip()
    ra_abs = a_tok.startswith("|")
    if ra_abs:
        # |Rn|[.reuse] — strip the surrounding bars, keep any trailing modifiers.
        end = a_tok.index("|", 1)
        a_tok = a_tok[1:end] + a_tok[end + 1:]
    ra, ra_reuse = parse_reg(a_tok)
    b_tok = p.operands[3]
    ps_idx, ps_neg = _parse_pred_src(p.operands[4])

    w = 0
    w = set_bits(w, 24, 8, ra)
    # byte 9 bits 76..79 = cmp_op; bit 74 = OR-combine.
    byte9 = ((cmp_op & 0xf) << 4) | (4 if comb == "OR" else 0)
    w = set_bits(w, 72, 8, byte9)
    w = set_bits(w, 81, 3, pd1)
    w = set_bits(w, 84, 3, pd2)
    w = set_bits(w, 87, 3, ps_idx)
    w = set_bits(w, 90, 1, ps_neg)
    if ra_abs:
        # |Ra| absolute-value flag at bit 73.
        w = set_bits(w, 73, 1, 1)
    if ftz:
        # .FTZ (flush-to-zero) flag at bit 80, matching FADD.FTZ.
        w = set_bits(w, 80, 1, 1)

    if b_tok.startswith("UR") or b_tok.replace(".reuse", "").strip() == "URZ":
        urb, rb_reuse = parse_ureg(b_tok)
        w = set_bits(w, 0, 16, 0x7c0b)
        w = set_bits(w, 32, 8, urb)
        w = set_bits(w, 91, 1, 1)
    elif b_tok.startswith("R") or b_tok.replace(".reuse", "").strip() == "RZ":
        rb, rb_reuse = parse_reg(b_tok)
        w = set_bits(w, 0, 16, 0x720b)
        w = set_bits(w, 32, 8, rb)
    else:
        imm = _parse_float_imm(b_tok)
        w = set_bits(w, 0, 16, 0x780b)
        w = set_bits(w, 32, 32, imm)
        rb_reuse = False

    if ra_reuse:
        w = set_bits(w, 122, 1, 1)
    if rb_reuse:
        w = set_bits(w, 123, 1, 1)
    return _apply_pred(w, p.pred)


for _cmp, _code in FSETP_CMP_OP.items():
    for _comb in ("AND", "OR"):
        _name = f"FSETP.{_cmp}.{_comb}"
        _TABLE[_name] = (lambda p, ctx=None, _co=_code, _cb=_comb: _fsetp_impl(p, _co, _cb))
        _ftz_name = f"FSETP.{_cmp}.FTZ.{_comb}"
        _TABLE[_ftz_name] = (lambda p, ctx=None, _co=_code, _cb=_comb: _fsetp_impl(p, _co, _cb, ftz=True))


def _parse_f32x2_swiz(tok: str) -> tuple[str, str, bool]:
    """Return (bare_reg, swizzle, reuse). Swizzle is one of {'F32', 'F32x2.HI_LO', ''}."""
    reuse = False
    swiz = ""
    # Detect .reuse (before any swizzle suffix).
    if ".reuse" in tok:
        reuse = True
        tok = tok.replace(".reuse", "")
    # Detect trailing swizzle.
    for s in (".F32x2.HI_LO", ".F32"):
        if tok.endswith(s):
            swiz = s[1:]
            tok = tok[: -len(s)]
            break
    return tok, swiz, reuse


@register("FMUL2")
def enc_fmul2(p: ParsedInsn) -> int:
    """FMUL2 Rd, Ra.F32 [.reuse], Rb.F32x2.HI_LO — FP32×FP32×2 packed mul."""
    if len(p.operands) != 3:
        raise UnsupportedInstruction(f"FMUL2 arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    a_tok, a_swiz, ra_reuse = _parse_f32x2_swiz(p.operands[1])
    b_tok, b_swiz, rb_reuse = _parse_f32x2_swiz(p.operands[2])
    ra, _ = parse_reg(a_tok)
    rb, _ = parse_reg(b_tok)
    w = 0
    w = set_bits(w, 0, 16, 0x724a)
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 32, 8, rb)
    # FMUL2: Ra=F32 -> bit 82, Rb=F32 -> bit 88.
    if a_swiz == "F32":
        w = set_bits(w, 82, 1, 1)
    if b_swiz == "F32":
        w = set_bits(w, 88, 1, 1)
    if ra_reuse: w = set_bits(w, 122, 1, 1)
    if rb_reuse: w = set_bits(w, 123, 1, 1)
    return _apply_pred(w, p.pred)


@register("FADD2")
def enc_fadd2(p: ParsedInsn) -> int:
    """FADD2 Rd, Ra.swiz, Rb.swiz — packed FP32×2 add."""
    if len(p.operands) != 3:
        raise UnsupportedInstruction(f"FADD2 arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    a_tok, a_swiz, ra_reuse = _parse_f32x2_swiz(p.operands[1])
    b_tok, b_swiz, rb_reuse = _parse_f32x2_swiz(p.operands[2])
    ra, _ = parse_reg(a_tok)
    rb, _ = parse_reg(b_tok)
    w = 0
    w = set_bits(w, 0, 16, 0x724b)
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 32, 8, rb)
    if a_swiz == "F32":
        w = set_bits(w, 82, 1, 1)
    if b_swiz == "F32":
        w = set_bits(w, 85, 1, 1)
    if ra_reuse: w = set_bits(w, 122, 1, 1)
    # FADD2 is 2-source: B-reuse uses the C-reuse slot (bit 124).
    if rb_reuse: w = set_bits(w, 124, 1, 1)
    return _apply_pred(w, p.pred)


@register("FFMA2")
def enc_ffma2(p: ParsedInsn) -> int:
    """FFMA2 Rd, Ra.swiz, {Rb|URb}.swiz, Rc.swiz"""
    if len(p.operands) != 4:
        raise UnsupportedInstruction(f"FFMA2 arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    a_tok, a_swiz, ra_reuse = _parse_f32x2_swiz(p.operands[1])
    b_tok, b_swiz, rb_reuse = _parse_f32x2_swiz(p.operands[2])
    c_tok, c_swiz, rc_reuse = _parse_f32x2_swiz(p.operands[3])
    ra, _ = parse_reg(a_tok)
    rc, _ = parse_reg(c_tok)
    w = 0
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 64, 8, rc)
    if b_tok.startswith("UR") or b_tok == "URZ":
        urb, _ = parse_ureg(b_tok)
        w = set_bits(w, 0, 16, 0x7c49)
        w = set_bits(w, 32, 8, urb)
        w = set_bits(w, 91, 1, 1)
    else:
        rb, _ = parse_reg(b_tok)
        w = set_bits(w, 0, 16, 0x7249)
        w = set_bits(w, 32, 8, rb)
    if a_swiz == "F32":
        w = set_bits(w, 82, 1, 1)
    # FFMA2: Rb.F32 -> bit 88, Rc.F32 -> bit 85 (opposite from FADD2).
    if b_swiz == "F32":
        w = set_bits(w, 88, 1, 1)
    if c_swiz == "F32":
        w = set_bits(w, 85, 1, 1)
    if ra_reuse: w = set_bits(w, 122, 1, 1)
    if rb_reuse: w = set_bits(w, 123, 1, 1)
    if rc_reuse: w = set_bits(w, 124, 1, 1)
    return _apply_pred(w, p.pred)


@register("FMUL")
def enc_fmul(p: ParsedInsn) -> int:
    """FMUL Rd, Ra, {Rb | URb | imm32}"""
    if len(p.operands) != 3:
        raise UnsupportedInstruction(f"FMUL arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    ra, ra_reuse = parse_reg(p.operands[1])
    b_tok = p.operands[2]

    w = 0
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 86, 1, 1)                   # fixed FMUL/FFMA flag

    if b_tok.startswith("UR") or b_tok.replace(".reuse", "").strip() == "URZ":
        urb, rb_reuse = parse_ureg(b_tok)
        w = set_bits(w, 0, 16, 0x7c20)
        w = set_bits(w, 32, 8, urb)
        w = set_bits(w, 91, 1, 1)
    elif b_tok.startswith("R") or b_tok.replace(".reuse", "").strip() == "RZ":
        rb, rb_reuse = parse_reg(b_tok)
        w = set_bits(w, 0, 16, 0x7220)
        w = set_bits(w, 32, 8, rb)
    else:
        imm = _parse_float_imm(b_tok)
        w = set_bits(w, 0, 16, 0x7820)
        w = set_bits(w, 32, 32, imm)
        rb_reuse = False

    if ra_reuse:
        w = set_bits(w, 122, 1, 1)
    if rb_reuse:
        w = set_bits(w, 123, 1, 1)
    return _apply_pred(w, p.pred)


def _parse_float_imm(tok: str) -> int:
    """Parse a float immediate from SASS text into its 32-bit IEEE-754 bits.

    For FP ops, even integer-looking tokens are treated as floating-point
    values (SASS prints small ints like 8388608 literally — but the hardware
    encodes them as their FP32 bit pattern). Hex literals are treated as raw
    bit patterns.
    """
    import struct
    if tok.startswith("0x") or tok.startswith("-0x"):
        return parse_imm(tok) & 0xffffffff
    try:
        f = float(tok)
    except ValueError:
        raise UnsupportedInstruction(f"bad float imm {tok!r}")
    return int.from_bytes(struct.pack("<f", f), "little")


def _is_imm_tok(tok: str) -> bool:
    t = tok.strip()
    if t.startswith("-"):
        t = t[1:].strip()
    if t.startswith("0x") or t.startswith("0X") or re.fullmatch(r"\d+", t):
        return True
    try:
        float(t)
        return True
    except ValueError:
        return False


@register("FFMA")
def enc_ffma(p: ParsedInsn) -> int:
    """FFMA Rd, [-]Ra, Rb|URb|imm, Rc   OR   FFMA Rd, [-]Ra, Rb, imm

    When the 4th text operand is an immediate, the disassembler has moved
    the imm into the Rc text-slot for readability; the actual encoding
    places it at bits 32..63 (Rb slot) and the textual Rb is stored at
    bits 64..71 (Rc slot).
    """
    if len(p.operands) != 4:
        raise UnsupportedInstruction(f"FFMA arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])

    # Optional `-` on Ra.
    a_tok = p.operands[1]
    ra_neg = a_tok.startswith("-")
    if ra_neg:
        a_tok = a_tok[1:]
    ra, ra_reuse = parse_reg(a_tok)

    b_tok = p.operands[2]
    c_tok = p.operands[3]

    w = 0
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, ra)
    if ra_neg:
        w = set_bits(w, 72, 1, 1)

    if _is_imm_tok(c_tok):
        # Imm-at-C form (0x7423): bits 32..63 = imm32, bits 64..71 = Rb_text.
        rb_neg = b_tok.startswith("-")
        if rb_neg:
            b_tok = b_tok[1:].strip()
        rb_text, rb_reuse = parse_reg(b_tok)
        imm = _parse_float_imm(c_tok)
        w = set_bits(w, 0, 16, 0x7423)
        w = set_bits(w, 32, 32, imm)
        w = set_bits(w, 64, 8, rb_text)
        if rb_neg:
            # -Rb in imm-at-C FFMA form: bit 75 (byte 9 bit 3).
            w = set_bits(w, 75, 1, 1)
        rc_reuse = False
    else:
        rc_neg = c_tok.startswith("-")
        if rc_neg:
            c_tok = c_tok[1:].strip()
        rb_neg = b_tok.startswith("-")
        if rb_neg:
            b_tok = b_tok[1:].strip()
        # Rc may be a uniform register (URc-at-C form, opcode 0x7e23).
        if c_tok.startswith("UR") or c_tok.replace(".reuse", "").strip() == "URZ":
            urc, rc_reuse = parse_ureg(c_tok)
            rb, rb_reuse = parse_reg(b_tok)
            w = set_bits(w, 0, 16, 0x7e23)
            w = set_bits(w, 32, 8, urc)
            w = set_bits(w, 64, 8, rb)
            w = set_bits(w, 91, 1, 1)
            if rb_neg:
                w = set_bits(w, 72, 1, 1)
            if rc_neg:
                w = set_bits(w, 75, 1, 1)
        else:
            rc, rc_reuse = parse_reg(c_tok)
            w = set_bits(w, 64, 8, rc)
            if rc_neg:
                # -Rc negate flag at byte 9 bit 75.
                w = set_bits(w, 75, 1, 1)
            if b_tok.startswith("UR") or b_tok.replace(".reuse", "").strip() == "URZ":
                urb, rb_reuse = parse_ureg(b_tok)
                w = set_bits(w, 0, 16, 0x7c23)
                w = set_bits(w, 32, 8, urb)
                w = set_bits(w, 91, 1, 1)
            elif b_tok.startswith("R") or b_tok.replace(".reuse", "").strip() == "RZ":
                rb, rb_reuse = parse_reg(b_tok)
                w = set_bits(w, 0, 16, 0x7223)
                w = set_bits(w, 32, 8, rb)
            else:
                imm = _parse_float_imm(b_tok)
                w = set_bits(w, 0, 16, 0x7823)
                w = set_bits(w, 32, 32, imm)
                rb_reuse = False
            if rb_neg:
                # -Rb negate flag at bit 63 (same slot FADD uses for -Rb).
                w = set_bits(w, 63, 1, 1)

    if ra_reuse:
        w = set_bits(w, 122, 1, 1)
    if rb_reuse:
        w = set_bits(w, 123, 1, 1)
    if rc_reuse:
        w = set_bits(w, 124, 1, 1)
    return _apply_pred(w, p.pred)


MEM_RE = re.compile(r"\[\s*(.+?)\s*\]$")


def _parse_mem(tok: str) -> tuple[str, int]:
    """Parse a memory operand like "[R4+0x100]" or "[UR5]" or "[R0.64]".

    Returns (base_token, offset). base_token includes any `.64`/`.U32` etc.
    """
    m = MEM_RE.fullmatch(tok.strip())
    if not m:
        raise UnsupportedInstruction(f"bad mem {tok!r}")
    inside = m.group(1)
    # Split on last `+` or `-` at the top level.
    base = inside
    off = 0
    # Look for trailing `+imm` or `-imm`.
    mm = re.match(r"^(.*?)([+-])\s*(0x[0-9a-fA-F]+|\d+)\s*$", inside)
    if mm:
        base = mm.group(1).strip()
        sign = 1 if mm.group(2) == "+" else -1
        off = sign * int(mm.group(3), 0)
    return base, off


LOCAL_DTYPE = {
    "":    0x08,        # 32-bit default
    "U8":  0x00,
    "S8":  0x01,
    "U16": 0x04,
    "S16": 0x06,
    "32":  0x08,
    "64":  0x0a,
    "128": 0x0c,
}


def _ldl_impl(p: ParsedInsn, *, dtype: int, lu: bool) -> int:
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"LDL arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    base, off = _parse_mem(p.operands[1])
    if not base.startswith("R"):
        raise UnsupportedInstruction(f"LDL base {base!r}")
    ra, _ = parse_reg(base)
    w = 0
    w = set_bits(w, 0, 16, 0x7983)
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 40, 24, off & ((1 << 24) - 1))
    w = set_bits(w, 72, 8, dtype)
    w = set_bits(w, 80, 8, 0x30 if lu else 0x10)
    return _apply_pred(w, p.pred)


def _stl_impl(p: ParsedInsn, *, dtype: int) -> int:
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"STL arity {len(p.operands)}")
    base, off = _parse_mem(p.operands[0])
    if not base.startswith("R"):
        raise UnsupportedInstruction(f"STL base {base!r}")
    ra, _ = parse_reg(base)
    rb, _ = parse_reg(p.operands[1])
    w = 0
    w = set_bits(w, 0, 16, 0x7387)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 32, 8, rb)
    w = set_bits(w, 40, 24, off & ((1 << 24) - 1))
    w = set_bits(w, 72, 8, dtype)
    w = set_bits(w, 80, 8, 0x10)
    return _apply_pred(w, p.pred)


for _suf, _dt in LOCAL_DTYPE.items():
    _name = f"LDL.{_suf}" if _suf else "LDL"
    _TABLE[_name] = (lambda p, ctx=None, _d=_dt: _ldl_impl(p, dtype=_d, lu=False))
    _name2 = f"LDL.LU.{_suf}" if _suf else "LDL.LU"
    _TABLE[_name2] = (lambda p, ctx=None, _d=_dt: _ldl_impl(p, dtype=_d, lu=True))
    _name3 = f"STL.{_suf}" if _suf else "STL"
    _TABLE[_name3] = (lambda p, ctx=None, _d=_dt: _stl_impl(p, dtype=_d))


DESC_MEM_RE = re.compile(r"^desc\[(UR\d+|URZ)\]\[(.+)\]$")


def _parse_desc_mem(tok: str) -> tuple[str, str, int]:
    """Parse "desc[URx][Ra.64 (+off)?]" -> (URx, Ra_reg_text, offset)."""
    m = DESC_MEM_RE.fullmatch(tok.strip())
    if not m:
        raise UnsupportedInstruction(f"bad desc mem {tok!r}")
    urd = m.group(1)
    inner = m.group(2).strip()
    # inner is like "Ra.64" or "Ra.64+0xc" or "Ra.64-0x4"
    off = 0
    base = inner
    mm = re.match(r"^(.*?)([+-])\s*(0x[0-9a-fA-F]+|\d+)\s*$", inner)
    if mm:
        base = mm.group(1).strip()
        sign = 1 if mm.group(2) == "+" else -1
        off = sign * int(mm.group(3), 0)
    return urd, base, off


GLOBAL_DTYPE = {
    "":     0x19,    # 32-bit default
    "U8":   0x11,
    "S8":   0x13,
    "U16":  0x15,
    "S16":  0x17,
    "32":   0x19,
    "64":   0x1b,
    "128":  0x1d,
}


def _stg_impl(p: ParsedInsn, dtype: int) -> int:
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"STG arity {len(p.operands)}")
    urd_tok, base_tok, off = _parse_desc_mem(p.operands[0])
    base_reg = base_tok.split(".")[0]
    ra, _ = parse_reg(base_reg)
    urd, _ = parse_ureg(urd_tok)
    rb, _ = parse_reg(p.operands[1])

    w = 0
    w = set_bits(w, 0, 16, 0x7986)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 32, 8, rb)
    w = set_bits(w, 40, 24, off & ((1 << 24) - 1))
    w = set_bits(w, 64, 8, urd)
    w = set_bits(w, 72, 8, dtype)
    w = set_bits(w, 80, 16, 0x0c10)
    return _apply_pred(w, p.pred)


for _suf, _dt in GLOBAL_DTYPE.items():
    _name = f"STG.E.{_suf}" if _suf else "STG.E"
    _TABLE[_name] = (lambda p, ctx=None, _d=_dt: _stg_impl(p, _d))


def _st_e_impl(p: ParsedInsn, dtype: int) -> int:
    """ST.E[.size] desc[URd][Rbase.64], Rdata — same shape as STG.E, opcode 0x7985."""
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"ST.E arity {len(p.operands)}")
    urd_tok, base_tok, off = _parse_desc_mem(p.operands[0])
    base_reg = base_tok.split(".")[0]
    ra, _ = parse_reg(base_reg)
    urd, _ = parse_ureg(urd_tok)
    rb, _ = parse_reg(p.operands[1])
    w = 0
    w = set_bits(w, 0, 16, 0x7985)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 32, 8, rb)
    w = set_bits(w, 40, 24, off & ((1 << 24) - 1))
    w = set_bits(w, 64, 8, urd)
    w = set_bits(w, 72, 8, dtype)
    w = set_bits(w, 80, 16, 0x0c10)
    return _apply_pred(w, p.pred)


for _suf, _dt in GLOBAL_DTYPE.items():
    _name = f"ST.E.{_suf}" if _suf else "ST.E"
    _TABLE[_name] = (lambda p, ctx=None, _d=_dt: _st_e_impl(p, _d))


def _ldg_impl(p: ParsedInsn, dtype: int) -> int:
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"LDG arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    urd_tok, base_tok, off = _parse_desc_mem(p.operands[1])
    base_reg = base_tok.split(".")[0]
    ra, _ = parse_reg(base_reg)
    urd, _ = parse_ureg(urd_tok)
    w = 0
    w = set_bits(w, 0, 16, 0x7981)
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 32, 8, urd)
    w = set_bits(w, 40, 24, off & ((1 << 24) - 1))
    w = set_bits(w, 72, 8, dtype)
    w = set_bits(w, 80, 16, 0x0c1e)
    return _apply_pred(w, p.pred)


for _suf, _dt in GLOBAL_DTYPE.items():
    _name = f"LDG.E.{_suf}" if _suf else "LDG.E"
    _TABLE[_name] = (lambda p, ctx=None, _d=_dt: _ldg_impl(p, _d))


# Shared-memory dtype field (bits 72..79). Default = 32-bit (0x08).
SHARED_DTYPE = {
    "":     0x08,
    "U8":   0x00,
    "S8":   0x02,
    "U16":  0x04,
    "S16":  0x06,
    "32":   0x08,
    "64":   0x0a,
    "128":  0x0c,
}


def _lds_impl(p: ParsedInsn, dtype: int) -> int:
    """LDS[.dtype] Rd, [Rbase|URbase|Rbase+URa+imm]"""
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"LDS arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    ra, ur, off = _parse_syncs_addr(p.operands[1])

    w = 0
    w = set_bits(w, 0, 16, 0x7984)
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 72, 8, dtype)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 40, 24, off & ((1 << 24) - 1))

    if ur != 0xff:
        # Any UR in the address selects the UR-combined form (bit 91 set,
        # UR index at bits 32..39).
        w = set_bits(w, 32, 8, ur)
        w = set_bits(w, 91, 1, 1)
    return _apply_pred(w, p.pred)


def _sts_impl(p: ParsedInsn, dtype: int) -> int:
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"STS arity {len(p.operands)}")
    addr_tok = p.operands[0]
    # May be [R+off], [UR+off], or [R+UR+off] for mixed.
    ra, ur, off = _parse_syncs_addr(addr_tok)
    rb, _ = parse_reg(p.operands[1])

    w = 0
    w = set_bits(w, 72, 8, dtype)

    if ur != 0xff:
        # Any UR in the address selects the UR-form opcode 0x7988 + bit 91.
        w = set_bits(w, 0, 16, 0x7988)
        w = set_bits(w, 24, 8, ra)
        w = set_bits(w, 64, 8, ur)
        w = set_bits(w, 91, 1, 1)
    else:
        w = set_bits(w, 0, 16, 0x7388)
        w = set_bits(w, 24, 8, ra)

    w = set_bits(w, 32, 8, rb)
    w = set_bits(w, 40, 24, off & ((1 << 24) - 1))
    return _apply_pred(w, p.pred)


for _suffix, _dtype in SHARED_DTYPE.items():
    _name_lds = f"LDS.{_suffix}" if _suffix else "LDS"
    _name_sts = f"STS.{_suffix}" if _suffix else "STS"
    _TABLE[_name_lds] = (lambda p, ctx=None, _dt=_dtype: _lds_impl(p, _dt))
    _TABLE[_name_sts] = (lambda p, ctx=None, _dt=_dtype: _sts_impl(p, _dt))


def _f2fp_pack_ab_merge_c(p: ParsedInsn, *, byte9: int) -> int:
    if len(p.operands) != 4:
        raise UnsupportedInstruction("F2FP arity")
    rd, _ = parse_reg(p.operands[0])
    ra, ra_reuse = parse_reg(p.operands[1])
    rb, rb_reuse = parse_reg(p.operands[2])
    rc, rc_reuse = parse_reg(p.operands[3])
    w = 0
    w = set_bits(w, 0, 16, 0x723e)
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 32, 8, rb)
    w = set_bits(w, 64, 8, rc)
    w = set_bits(w, 72, 8, byte9)
    w = set_bits(w, 80, 16, 0x0480)
    if ra_reuse: w = set_bits(w, 122, 1, 1)
    if rb_reuse: w = set_bits(w, 123, 1, 1)
    if rc_reuse: w = set_bits(w, 124, 1, 1)
    return _apply_pred(w, p.pred)


@register("F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C")
def enc_f2fp_e4m3_pack_ab_merge_c(p: ParsedInsn) -> int:
    return _f2fp_pack_ab_merge_c(p, byte9=0x70)


@register("F2FP.SATFINITE.E5M2.F32.PACK_AB_MERGE_C")
def enc_f2fp_e5m2_pack_ab_merge_c(p: ParsedInsn) -> int:
    return _f2fp_pack_ab_merge_c(p, byte9=0x60)


def _f2fp_pack_ab(p: ParsedInsn, *, byte9: int) -> int:
    if len(p.operands) != 3:
        raise UnsupportedInstruction("F2FP pack arity")
    rd, _ = parse_reg(p.operands[0])
    ra, ra_reuse = parse_reg(p.operands[1])
    rb, rb_reuse = parse_reg(p.operands[2])
    w = 0
    w = set_bits(w, 0, 16, 0x723e)
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 32, 8, rb)
    w = set_bits(w, 64, 8, 0xff)
    w = set_bits(w, 72, 8, byte9)
    if ra_reuse: w = set_bits(w, 122, 1, 1)
    if rb_reuse: w = set_bits(w, 123, 1, 1)
    return _apply_pred(w, p.pred)


@register("F2FP.F16.F32.PACK_AB")
def enc_f2fp_f16_f32_pack_ab(p: ParsedInsn) -> int:
    return _f2fp_pack_ab(p, byte9=0x00)


@register("F2FP.BF16.F32.PACK_AB")
def enc_f2fp_bf16_f32_pack_ab(p: ParsedInsn) -> int:
    return _f2fp_pack_ab(p, byte9=0x10)


def _f2fp_unpack_b(p: ParsedInsn, *, byte9: int) -> int:
    if len(p.operands) != 2:
        raise UnsupportedInstruction("F2FP.UNPACK arity")
    rd, _ = parse_reg(p.operands[0])
    b_tok = p.operands[1]
    h1 = False
    if b_tok.endswith(".H1"):
        h1 = True
        b_tok = b_tok[:-3]
    rb, rb_reuse = parse_reg(b_tok)
    w = 0
    w = set_bits(w, 0, 16, 0x723e)
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, 0xff)
    w = set_bits(w, 32, 8, rb)
    w = set_bits(w, 64, 8, 0xff)
    w = set_bits(w, 72, 8, byte9)
    # byte 11: 0x02 default; .H1 swizzle sets bit 88 → 0x03.
    w = set_bits(w, 80, 16, 0x0300 if h1 else 0x0200)
    if rb_reuse: w = set_bits(w, 123, 1, 1)
    return _apply_pred(w, p.pred)


@register("F2FP.F16.E4M3.UNPACK_B")
def enc_f2fp_e4m3_unpack_b(p: ParsedInsn) -> int:
    return _f2fp_unpack_b(p, byte9=0x06)


@register("F2FP.F16.E5M2.UNPACK_B")
def enc_f2fp_e5m2_unpack_b(p: ParsedInsn) -> int:
    return _f2fp_unpack_b(p, byte9=0x04)


def _peel_hfp_swiz(tok: str) -> tuple[str, str, bool, bool]:
    """Parse "[-]Rn[.reuse][.H0_H0|.H1_H1]" or the UR analog.

    Returns (bare_token, swiz, reuse, neg) where swiz is "", "H0_H0", or "H1_H1".
    """
    neg = tok.startswith("-")
    if neg:
        tok = tok[1:].strip()
    swiz = ""
    for s in (".H0_H0", ".H1_H1"):
        if tok.endswith(s):
            swiz = s[1:]
            tok = tok[: -len(s)]
            break
    reuse = ".reuse" in tok
    if reuse:
        tok = tok.replace(".reuse", "")
    return tok, swiz, reuse, neg


def _hpair_impl(p: ParsedInsn, *, opcode_r: int, opcode_ur: int,
                byte9_base: int, byte10_base: int,
                bf16: bool = False, has_rc: bool = False) -> int:
    """Encoder shared by HADD2 / HMUL2 / HFMA2 and their .BF16_V2 variants.

    Common layout (FP16/BF16 pair ops):
      opcode        0x7230/0x7231/0x7232 (Ra+Rb reg form); +0x0c00 for UR Rb.
      Rd            bits 16..23
      Ra            bits 24..31
      Rb / URb      bits 32..39 (+ bit 91 for UR form)
      byte 7        swizzle/negate bits 56..63:
                      bit 60 = Rb H1_H1, bit 61 = Rb swizzle-present,
                      bit 62 = Ra H1_H1, bit 63 = Rb negate.
      byte 9 bit 3  = Ra has explicit swizzle (H0_H0 or H1_H1).
      byte 10 bit 5 = BF16_V2 variant flag (0x20).
      Rc            bits 64..71 (HFMA2 only).
      Rb reuse      bit 123, Rc reuse bit 124 for HFMA2; Rb reuse bit 123 for HADD2/HMUL2.
    """
    expected = 4 if has_rc else 3
    if len(p.operands) != expected:
        raise UnsupportedInstruction(f"{p.mnemonic} arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    a_tok, a_swiz, ra_reuse, ra_neg = _peel_hfp_swiz(p.operands[1])
    b_tok, b_swiz, rb_reuse, rb_neg = _peel_hfp_swiz(p.operands[2])

    # Only Ra/Rb forms covered here; Rb may be R or UR. Imm-Rb or imm-Rc are not
    # supported (they use different packed-imm encodings).
    ra, _ = parse_reg(a_tok)

    w = 0
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 72, 8, byte9_base)
    w = set_bits(w, 80, 8, byte10_base)

    if b_tok.startswith("UR") or b_tok == "URZ":
        urb, _ = parse_ureg(b_tok)
        w = set_bits(w, 0, 16, opcode_ur)
        w = set_bits(w, 32, 8, urb)
        w = set_bits(w, 91, 1, 1)
    elif b_tok.startswith("R") or b_tok == "RZ":
        rb, _ = parse_reg(b_tok)
        w = set_bits(w, 0, 16, opcode_r)
        w = set_bits(w, 32, 8, rb)
    else:
        raise UnsupportedInstruction(f"{p.mnemonic} Rb {b_tok!r}")

    # Byte-7 swizzle/negate nibble (Rb side only).
    byte7 = 0
    if b_swiz:
        byte7 |= 0x20                                       # bit 61 (Rb swizzle-present)
        if b_swiz == "H1_H1": byte7 |= 0x10                 # bit 60
    if rb_neg: byte7 |= 0x80                                # bit 63
    w = set_bits(w, 56, 8, byte7)

    # Ra-side swizzle is at bits 74-75 of byte 9:
    #   bit 75 = Ra has explicit swizzle
    #   bit 74 = Ra is H1_H1 (vs H0_H0)
    if a_swiz:
        w = set_bits(w, 75, 1, 1)
        if a_swiz == "H1_H1":
            w = set_bits(w, 74, 1, 1)
    if ra_neg:
        w = set_bits(w, 72, 1, 1)                           # bit 72: -Ra

    if has_rc:
        c_tok, c_swiz, rc_reuse, rc_neg = _peel_hfp_swiz(p.operands[3])
        rc, _ = parse_reg(c_tok)
        w = set_bits(w, 64, 8, rc)
        if c_swiz == "H1_H1":
            # Rc H1_H1 shares bit 82 (byte 10 bit 2) as the swizzle flag.
            w = set_bits(w, 82, 1, 1)
        # Rc negate lives at byte 10 bit 1 (bit 81).
        if rc_neg:
            w = set_bits(w, 81, 1, 1)
        if rc_reuse:
            w = set_bits(w, 124, 1, 1)

    if ra_reuse:
        w = set_bits(w, 122, 1, 1)
    if rb_reuse:
        w = set_bits(w, 123, 1, 1)
    return _apply_pred(w, p.pred)


@register("HADD2")
def enc_hadd2(p: ParsedInsn) -> int:
    return _hpair_impl(p, opcode_r=0x7230, opcode_ur=0x7c30,
                       byte9_base=0x08, byte10_base=0x00)


@register("HMUL2")
def enc_hmul2(p: ParsedInsn) -> int:
    return _hpair_impl(p, opcode_r=0x7232, opcode_ur=0x7c32,
                       byte9_base=0x00, byte10_base=0x00)


@register("HADD2.BF16_V2")
def enc_hadd2_bf16_v2(p: ParsedInsn) -> int:
    return _hpair_impl(p, opcode_r=0x7230, opcode_ur=0x7c30,
                       byte9_base=0x08, byte10_base=0x20, bf16=True)


@register("HMUL2.BF16_V2")
def enc_hmul2_bf16_v2(p: ParsedInsn) -> int:
    return _hpair_impl(p, opcode_r=0x7232, opcode_ur=0x7c32,
                       byte9_base=0x00, byte10_base=0x20, bf16=True)


@register("HFMA2.BF16_V2")
def enc_hfma2_bf16_v2(p: ParsedInsn) -> int:
    return _hpair_impl(p, opcode_r=0x7231, opcode_ur=0x7c31,
                       byte9_base=0x00, byte10_base=0x24, bf16=True,
                       has_rc=True)


@register("HADD2.F32")
def enc_hadd2_f32(p: ParsedInsn) -> int:
    """HADD2.F32 Rd, [-]Ra, Rb{.reuse}.{H0_H0|H1_H1}

    Observed only in the "splat a single FP16 half to both lanes of an FP32
    pair" idiom: Rd = (-Ra).hi + Rb.swizzle, with Ra always RZ.
    """
    if len(p.operands) != 3:
        raise UnsupportedInstruction(f"HADD2.F32 arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    a_tok = p.operands[1]
    ra_neg = a_tok.startswith("-")
    if ra_neg:
        a_tok = a_tok[1:].strip()
    ra, _ = parse_reg(a_tok)
    b_tok = p.operands[2]
    # Rb text is "Rn[.reuse]{.H0_H0|.H1_H1}". Peel off the swizzle first,
    # then detect .reuse on the Rn.
    swiz = ""
    if b_tok.endswith(".H0_H0"):
        swiz = "H0_H0"
        b_tok = b_tok[:-len(".H0_H0")]
    elif b_tok.endswith(".H1_H1"):
        swiz = "H1_H1"
        b_tok = b_tok[:-len(".H1_H1")]
    else:
        raise UnsupportedInstruction(f"HADD2.F32 Rb swiz {b_tok!r}")
    rb, rb_reuse = parse_reg(b_tok)
    if not ra_neg:
        raise UnsupportedInstruction("HADD2.F32 without Ra negate not supported")

    w = 0
    w = set_bits(w, 0, 16, 0x7230)
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 32, 8, rb)
    # byte 7: 0x20 for H0_H0, 0x30 for H1_H1 (bit 61 always, bit 60 for H1).
    byte7 = 0x20 | (0x10 if swiz == "H1_H1" else 0)
    w = set_bits(w, 56, 8, byte7)
    w = set_bits(w, 72, 8, 0x41)
    if rb_reuse:
        # Rb reuse bit at bit 124 for HADD2 (distinct from the usual 123).
        w = set_bits(w, 124, 1, 1)
    return _apply_pred(w, p.pred)


@register("FMNMX3")
def enc_fmnmx3(p: ParsedInsn) -> int:
    """FMNMX3 Rd, Ra, Rb, Rc, Ps — 3-input FP min-max with selection."""
    if len(p.operands) != 5:
        raise UnsupportedInstruction(f"FMNMX3 arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    ra, ra_reuse = parse_reg(p.operands[1])
    rb, rb_reuse = parse_reg(p.operands[2])
    rc, rc_reuse = parse_reg(p.operands[3])
    ps_idx, ps_neg = _parse_pred_src(p.operands[4])
    w = 0
    w = set_bits(w, 0, 16, 0x7276)
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 32, 8, rb)
    w = set_bits(w, 64, 8, rc)
    w = set_bits(w, 80, 8, 0x80)
    w = set_bits(w, 87, 3, ps_idx)
    w = set_bits(w, 90, 1, ps_neg)
    if ra_reuse: w = set_bits(w, 122, 1, 1)
    if rb_reuse: w = set_bits(w, 123, 1, 1)
    if rc_reuse: w = set_bits(w, 124, 1, 1)
    return _apply_pred(w, p.pred)


@register("FMNMX.NAN")
def enc_fmnmx_nan(p: ParsedInsn) -> int:
    """FMNMX.NAN Rd, Ra, {Rb|URb|imm}, Ps  — FP min/max with NaN propagation.

    Result = Ra < Rb ? (Ps ? Ra : Rb) : (Ps ? Rb : Ra) (MIN when Ps==!PT).
    """
    if len(p.operands) != 4:
        raise UnsupportedInstruction(f"FMNMX.NAN arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    a_tok = p.operands[1]
    ra_abs = a_tok.startswith("|") and a_tok.endswith("|")
    if ra_abs:
        a_tok = a_tok[1:-1].strip()
    ra, ra_reuse = parse_reg(a_tok)
    b_tok = p.operands[2]
    rb_abs = b_tok.startswith("|") and b_tok.endswith("|")
    if rb_abs:
        b_tok = b_tok[1:-1].strip()
    ps_idx, ps_neg = _parse_pred_src(p.operands[3])

    w = 0
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, ra)
    if ra_abs:
        w = set_bits(w, 73, 1, 1)
    if rb_abs:
        # |Rb| sets bit 62 (observed in FMNMX.NAN Rd, Ra, |Rb|, Ps).
        w = set_bits(w, 62, 1, 1)
    # byte 10 = 0x82 (observed fixed bits 80..87).
    w = set_bits(w, 80, 8, 0x82)
    w = set_bits(w, 87, 3, ps_idx)
    w = set_bits(w, 90, 1, ps_neg)

    if b_tok.startswith("UR") or b_tok.replace(".reuse", "").strip() == "URZ":
        urb, rb_reuse = parse_ureg(b_tok)
        w = set_bits(w, 0, 16, 0x7c09)
        w = set_bits(w, 32, 8, urb)
        w = set_bits(w, 91, 1, 1)
    elif b_tok.startswith("R") or b_tok.replace(".reuse", "").strip() == "RZ":
        rb, rb_reuse = parse_reg(b_tok)
        w = set_bits(w, 0, 16, 0x7209)
        w = set_bits(w, 32, 8, rb)
    else:
        imm = _parse_float_imm(b_tok)
        w = set_bits(w, 0, 16, 0x7809)
        w = set_bits(w, 32, 32, imm)
        rb_reuse = False

    if ra_reuse:
        w = set_bits(w, 122, 1, 1)
    if rb_reuse:
        w = set_bits(w, 123, 1, 1)
    return _apply_pred(w, p.pred)


def _lea_base(p: ParsedInsn, *, is_uniform: bool, hi_x: bool, sx32: bool = False) -> int:
    """LEA / LEA.HI.X / ULEA encoding.

    Plain LEA forms:
      LEA Rd, Ra, Rb|URb, shift
      LEA Rd, Pd, Ra, Rb|URb, shift
    LEA.HI.X form:
      LEA.HI.X Rd, Ra, Rb|URb, Rc, shift, Ps
    """
    ops = list(p.operands)
    if hi_x:
        if len(ops) != 6:
            raise UnsupportedInstruction(f"LEA.HI.X arity {len(ops)}")
        rd_tok, ra_tok, b_tok, c_tok, sh_tok, ps_tok = ops
        pd_idx = 7  # implicit PT
    else:
        if len(ops) == 4:
            rd_tok, ra_tok, b_tok, sh_tok = ops
            pd_idx = 7
            c_tok = None
            ps_tok = None
        elif len(ops) == 5:
            rd_tok, pd_tok, ra_tok, b_tok, sh_tok = ops
            pd_idx = _parse_upred_dst(pd_tok) if is_uniform else _parse_pred_dst(pd_tok)
            c_tok = None
            ps_tok = None
        else:
            raise UnsupportedInstruction(f"LEA arity {len(ops)}")

    parse_d = parse_ureg if is_uniform else parse_reg
    rd, _ = parse_d(rd_tok)
    ra, ra_reuse = parse_d(ra_tok)
    shift = parse_imm(sh_tok) & 0x1f

    # Base opcode selection by source-B class.
    base_r = 0x7291 if is_uniform else 0x7211
    base_ur = 0x7291 if is_uniform else 0x7c11

    w = 0
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, ra)
    # byte 9: bits 72..74 = flags, bits 75..79 = shift. Bit 74 = .X flag,
    # bit 73 = .SX32 flag.
    flags = 0
    if hi_x:
        flags |= (1 << 2)    # bit 74
    if sx32:
        flags |= (1 << 1)    # bit 73
    byte9 = (flags & 0x07) | (shift << 3)
    w = set_bits(w, 72, 8, byte9)

    is_ur_src = b_tok.startswith("UR") or b_tok.replace(".reuse", "").strip() == "URZ"
    is_reg_src = b_tok.startswith("R") or b_tok.replace(".reuse", "").strip() == "RZ"
    if is_ur_src:
        urb, rb_reuse = parse_ureg(b_tok)
        w = set_bits(w, 0, 16, base_ur)
        w = set_bits(w, 32, 8, urb)
    elif is_reg_src:
        rb, rb_reuse = parse_d(b_tok) if is_uniform else parse_reg(b_tok)
        w = set_bits(w, 0, 16, base_r)
        w = set_bits(w, 32, 8, rb)
    else:
        # Immediate at the B slot: opcode switches to the imm form.
        imm = parse_imm(b_tok) & 0xffffffff
        base_imm = 0x7891 if is_uniform else 0x7811
        w = set_bits(w, 0, 16, base_imm)
        w = set_bits(w, 32, 32, imm)
        rb_reuse = False
        if is_uniform:
            # ULEA imm form still has the UR-selector bit 91 set (fixed for
            # uniform encoding), matching the R/UR forms.
            w = set_bits(w, 91, 1, 1)

    # byte 11 low nibble: 0x0e (R src) / 0x0f (UR src) for plain LEA;
    # 0x08 (R src) / 0x08 (UR src) for LEA.HI.X (UR-selector is in bit 91).
    # Uniform-imm form also keeps the UR-selector at bit 91.
    ur_selector = is_ur_src or (is_uniform and not is_reg_src and not is_ur_src)
    if hi_x:
        w = set_bits(w, 88, 8, 0x08)
        w = set_bits(w, 91, 1, 1 if ur_selector else 0)
    else:
        w = set_bits(w, 88, 8, 0x0f if ur_selector else 0x07)
        # For plain LEA: byte 10 = 0x80 | (Pd_idx << 1).
        w = set_bits(w, 80, 8, 0x80 | (pd_idx << 1))

    if hi_x:
        rc, rc_reuse = parse_d(c_tok) if is_uniform else parse_reg(c_tok)
        w = set_bits(w, 64, 8, rc)
        ps_idx, ps_neg = _parse_pred_src(ps_tok)
        # byte 10 = 0x0f | (ps_idx_lsb << 7); bits 87-89 = ps_idx, bit 90 = ps_neg.
        # Start from 0x0f (bit 80 + bits 81-83 = 1111, i.e. flag + implicit PT Pd).
        base_b10 = 0x0f
        w = set_bits(w, 80, 8, base_b10)
        w = set_bits(w, 87, 3, ps_idx)
        w = set_bits(w, 90, 1, ps_neg)
    else:
        w = set_bits(w, 64, 8, 0xff)

    if ra_reuse:
        w = set_bits(w, 122, 1, 1)
    if rb_reuse:
        w = set_bits(w, 123, 1, 1)
    if hi_x and rc_reuse:
        w = set_bits(w, 124, 1, 1)
    return _apply_pred(w, p.pred)


@register("LEA")
def enc_lea(p: ParsedInsn) -> int:
    return _lea_base(p, is_uniform=False, hi_x=False)


@register("LEA.HI.X")
def enc_lea_hi_x(p: ParsedInsn) -> int:
    return _lea_base(p, is_uniform=False, hi_x=True)


@register("LEA.HI.X.SX32")
def enc_lea_hi_x_sx32(p: ParsedInsn) -> int:
    """LEA.HI.X.SX32 Rd, Ra, Rb|URb, shift, Ps (5 operands, Rc implicit RZ).

    Insert an RZ in the Rc position so we can reuse the LEA.HI.X helper.
    """
    if len(p.operands) != 5:
        raise UnsupportedInstruction(f"LEA.HI.X.SX32 arity {len(p.operands)}")
    ops = [p.operands[0], p.operands[1], p.operands[2], "RZ",
           p.operands[3], p.operands[4]]
    p6 = ParsedInsn(pred=p.pred, mnemonic=p.mnemonic, operands=ops)
    return _lea_base(p6, is_uniform=False, hi_x=True, sx32=True)


@register("ULEA")
def enc_ulea(p: ParsedInsn) -> int:
    return _lea_base(p, is_uniform=True, hi_x=False)


@register("ULEA.HI.X")
def enc_ulea_hi_x(p: ParsedInsn) -> int:
    return _lea_base(p, is_uniform=True, hi_x=True)


@register("UGETNEXTWORKID.BROADCAST")
def enc_ugetnextworkid_broadcast(p: ParsedInsn) -> int:
    """UGETNEXTWORKID.BROADCAST [URd], [URa] — broadcast the next work-id.

    The source [URa] is implicitly URd+1, so it doesn't appear in the encoding;
    only URd is stored at bits 24..31.
    """
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"UGETNEXTWORKID arity {len(p.operands)}")
    import re as _re
    m0 = _re.fullmatch(r"\[\s*(\S+)\s*\]", p.operands[0].strip())
    if not m0:
        raise UnsupportedInstruction(f"UGETNEXTWORKID operand[0] {p.operands[0]!r}")
    urd, _ = parse_ureg(m0.group(1))
    w = 0
    w = set_bits(w, 0, 16, 0x73ca)
    w = set_bits(w, 24, 8, urd)
    w = set_bits(w, 72, 8, 0x01)
    w = set_bits(w, 91, 1, 1)
    return _apply_pred(w, p.pred)


@register("FADD.RZ")
def enc_fadd_rz(p: ParsedInsn) -> int:
    """FADD.RZ — FADD with round-toward-zero rounding mode (bits 78..79 = 11)."""
    w = enc_fadd(p)
    return w | (0b11 << 78)


def _parse_branch_target(tok: str) -> str:
    """Parse "`(.L_x_N)" into the label name ".L_x_N"."""
    t = tok.strip()
    if t.startswith("`(") and t.endswith(")"):
        return t[2:-1]
    raise UnsupportedInstruction(f"unexpected branch target {tok!r}")


def _encode_bra_offset(w: int, field: int) -> int:
    """Splice a signed branch offset into a BRA/BRA.U/etc. encoding.

    The field is the (delta-16)/4 value, stored as:
      - low 8 bits  -> bits 16..23
      - bits 8..55  -> bits 34..81 (sign-extended to 48 bits)
    """
    low = field & 0xff
    high = (field >> 8) & ((1 << 48) - 1)    # 48-bit two's complement
    w = set_bits(w, 16, 8, low)
    w = set_bits(w, 34, 48, high)
    return w


def _parse_barrier(tok: str) -> int:
    m = re.fullmatch(r"B(\d+)", tok.strip())
    if not m:
        raise UnsupportedInstruction(f"barrier {tok!r}")
    return int(m.group(1))


def _bssy_like(p: ParsedInsn, ctx: Context | None, *, opcode: int,
               reconvergent: bool) -> int:
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"{p.mnemonic} arity {len(p.operands)}")
    if ctx is None or ctx.labels is None:
        raise UnsupportedInstruction(f"{p.mnemonic} without label context")
    b = _parse_barrier(p.operands[0])
    label = _parse_branch_target(p.operands[1])
    target = ctx.labels.get(label)
    if target is None:
        raise UnsupportedInstruction(f"{p.mnemonic} unknown label {label!r}")
    field = (target - ctx.pc - 16) & 0xffffffff

    w = 0
    w = set_bits(w, 0, 16, opcode)
    w = set_bits(w, 16, 8, b)
    w = set_bits(w, 32, 32, field)
    if reconvergent:
        w = set_bits(w, 72, 8, 0x02)
    w = set_bits(w, 80, 16, 0x0380)
    return _apply_pred(w, p.pred)


def _bsync_like(p: ParsedInsn, *, opcode: int, reconvergent: bool) -> int:
    if len(p.operands) != 1:
        raise UnsupportedInstruction(f"{p.mnemonic} arity {len(p.operands)}")
    b = _parse_barrier(p.operands[0])
    w = 0
    w = set_bits(w, 0, 16, opcode)
    w = set_bits(w, 16, 8, b)
    if reconvergent:
        w = set_bits(w, 72, 8, 0x02)
    w = set_bits(w, 80, 16, 0x0380)
    return _apply_pred(w, p.pred)


@register("BSSY")
def enc_bssy(p, ctx=None):
    return _bssy_like(p, ctx, opcode=0x7945, reconvergent=False)


@register("BSSY.RECONVERGENT")
def enc_bssy_r(p, ctx=None):
    return _bssy_like(p, ctx, opcode=0x7945, reconvergent=True)


@register("BSYNC")
def enc_bsync(p, ctx=None):
    return _bsync_like(p, opcode=0x7941, reconvergent=False)


@register("BSYNC.RECONVERGENT")
def enc_bsync_r(p, ctx=None):
    return _bsync_like(p, opcode=0x7941, reconvergent=True)


@register("BRA.U")
def enc_bra_u(p: ParsedInsn, ctx: Context | None = None) -> int:
    """BRA.U [!]UPx, `(.L_label) — uniform-predicated unconditional branch."""
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"BRA.U arity {len(p.operands)}")
    if ctx is None or ctx.labels is None:
        raise UnsupportedInstruction("BRA.U without label context")
    ps_idx, ps_neg = _parse_pred_src(p.operands[0])
    label = _parse_branch_target(p.operands[1])
    target = ctx.labels.get(label)
    if target is None:
        raise UnsupportedInstruction(f"BRA.U unknown label {label!r}")
    field = (target - ctx.pc - 16) // 4

    w = 0
    w = set_bits(w, 0, 16, 0x7547)
    w = _encode_bra_offset(w, field)
    w = set_bits(w, 32, 1, 1)                   # BRA.U-specific flag
    w = set_bits(w, 24, 3, ps_idx)
    w = set_bits(w, 27, 1, ps_neg)
    w = set_bits(w, 87, 1, 1)
    w = set_bits(w, 88, 2, 0b11)
    w = set_bits(w, 91, 1, 1)
    return _apply_pred(w, p.pred)


@register("CALL.ABS.NOINC")
def enc_call_abs_noinc(p: ParsedInsn, ctx: Context | None = None) -> int:
    """CALL.ABS.NOINC Rn — indirect call via 64-bit address in Rn."""
    if len(p.operands) != 1:
        raise UnsupportedInstruction(f"CALL.ABS.NOINC arity {len(p.operands)}")
    rn, _ = parse_reg(p.operands[0])
    w = 0
    w = set_bits(w, 0, 16, 0x7343)
    w = set_bits(w, 24, 8, rn)
    # Tail flags: bits 86..89 = 1111.
    w = set_bits(w, 86, 4, 0xF)
    return _apply_pred(w, p.pred)


@register("CALL.REL.NOINC")
def enc_call_rel_noinc(p: ParsedInsn, ctx: Context | None = None) -> int:
    """CALL.REL.NOINC `(label) — PC-relative call."""
    if len(p.operands) != 1:
        raise UnsupportedInstruction(f"CALL.REL.NOINC arity {len(p.operands)}")
    if ctx is None or ctx.labels is None:
        raise UnsupportedInstruction("CALL.REL.NOINC without label context")
    label = _parse_branch_target(p.operands[0])
    target = ctx.labels.get(label)
    if target is None:
        raise UnsupportedInstruction(f"CALL.REL.NOINC unknown label {label!r}")
    field = (target - ctx.pc - 16) // 4
    w = 0
    w = set_bits(w, 0, 16, 0x7944)
    w = _encode_bra_offset(w, field)
    w = set_bits(w, 86, 4, 0xF)
    return _apply_pred(w, p.pred)


@register("RET.REL.NODEC")
def enc_ret_rel_nodec(p: ParsedInsn, ctx: Context | None = None) -> int:
    """RET.REL.NODEC Rn `(label) — PC-relative return, base in Rn."""
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"RET.REL.NODEC arity {len(p.operands)}")
    if ctx is None or ctx.labels is None:
        raise UnsupportedInstruction("RET.REL.NODEC without label context")
    rn, _ = parse_reg(p.operands[0])
    label = _parse_branch_target(p.operands[1])
    target = ctx.labels.get(label)
    if target is None:
        raise UnsupportedInstruction(f"RET.REL.NODEC unknown label {label!r}")
    field = (target - ctx.pc - 16) // 4
    w = 0
    w = set_bits(w, 0, 16, 0x7950)
    w = set_bits(w, 24, 8, rn)
    w = _encode_bra_offset(w, field)
    # Tail flags: bits 80, 81, 86-89 set.
    w = set_bits(w, 80, 2, 0b11)
    w = set_bits(w, 86, 4, 0xF)
    return _apply_pred(w, p.pred)


@register("LEPC")
def enc_lepc(p: ParsedInsn, ctx: Context | None = None) -> int:
    """LEPC Rd, `(label) — load PC-of-label into Rd (disp is raw bytes)."""
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"LEPC arity {len(p.operands)}")
    if ctx is None or ctx.labels is None:
        raise UnsupportedInstruction("LEPC without label context")
    rd, _ = parse_reg(p.operands[0])
    label = _parse_branch_target(p.operands[1])
    target = ctx.labels.get(label)
    if target is None:
        raise UnsupportedInstruction(f"LEPC unknown label {label!r}")
    disp = target - ctx.pc - 16
    w = 0
    w = set_bits(w, 0, 16, 0x794e)
    w = set_bits(w, 16, 8, rd)
    # LEPC stores raw byte displacement: low 8 bits at bits 24..31,
    # high 48 bits (signed) at bits 32..79.
    w = set_bits(w, 24, 8, disp & 0xff)
    high = (disp >> 8) & ((1 << 48) - 1)
    w = set_bits(w, 32, 48, high)
    return _apply_pred(w, p.pred)


@register("BRA")
def enc_bra(p: ParsedInsn, ctx: Context | None = None) -> int:
    """BRA [Pd,] `(label) — unconditional branch, with optional pred dest."""
    if len(p.operands) not in (1, 2):
        raise UnsupportedInstruction(f"BRA arity {len(p.operands)}")
    if ctx is None or ctx.labels is None:
        raise UnsupportedInstruction("BRA without label context")
    if len(p.operands) == 2:
        pd = _parse_pred_dst(p.operands[0])
        label = _parse_branch_target(p.operands[1])
    else:
        pd = 7                                              # PT sentinel
        label = _parse_branch_target(p.operands[0])
    target = ctx.labels.get(label)
    if target is None:
        raise UnsupportedInstruction(f"BRA unknown label {label!r}")
    field = (target - ctx.pc - 16) // 4

    w = 0
    w = set_bits(w, 0, 16, 0x7947)
    w = _encode_bra_offset(w, field)
    w = set_bits(w, 87, 3, pd)                              # pred dst at 87..89
    return _apply_pred(w, p.pred)


SHF_BYTE9 = {
    "L.U32":     0x06,
    "L.U64.HI":  0x02,
    "R.U32":     0x12,
    "R.U64":     0x12,
    "R.U32.HI":  0x16,
    "R.S32.HI":  0x14,
}


def _shf_impl(p: ParsedInsn, *, is_uniform: bool) -> int:
    if len(p.operands) != 4:
        raise UnsupportedInstruction(f"SHF arity {len(p.operands)}")
    # mnemonic like "SHF.L.U32"; keep parts after the opcode base
    base = "USHF" if is_uniform else "SHF"
    mod = p.mnemonic[len(base) + 1:]
    if mod not in SHF_BYTE9:
        raise UnsupportedInstruction(f"SHF modifier {mod!r}")
    byte9 = SHF_BYTE9[mod]

    parse_d = parse_ureg if is_uniform else parse_reg
    rd, _ = parse_d(p.operands[0])
    ra, ra_reuse = parse_d(p.operands[1])
    b_tok = p.operands[2]
    rc, rc_reuse = parse_d(p.operands[3])

    w = 0
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 64, 8, rc)
    w = set_bits(w, 72, 8, byte9)
    if mod.endswith(".HI"):
        w = set_bits(w, 80, 1, 1)               # .HI flag

    base_r = 0x7299 if is_uniform else 0x7219
    base_ur = 0x7299 if is_uniform else 0x7c19
    base_imm = 0x7899 if is_uniform else 0x7819

    if b_tok.startswith("UR") or b_tok.replace(".reuse", "").strip() == "URZ":
        urb, rb_reuse = parse_ureg(b_tok)
        w = set_bits(w, 0, 16, base_ur)
        w = set_bits(w, 32, 8, urb)
        w = set_bits(w, 91, 1, 1)
    elif (not is_uniform) and (b_tok.startswith("R") or b_tok.replace(".reuse", "").strip() == "RZ"):
        rb, rb_reuse = parse_reg(b_tok)
        w = set_bits(w, 0, 16, base_r)
        w = set_bits(w, 32, 8, rb)
    else:
        imm = parse_imm(b_tok) & 0xffffffff
        w = set_bits(w, 0, 16, base_imm)
        w = set_bits(w, 32, 32, imm)
        rb_reuse = False
        if is_uniform:
            w = set_bits(w, 91, 1, 1)

    if ra_reuse:
        w = set_bits(w, 122, 1, 1)
    if rb_reuse:
        w = set_bits(w, 123, 1, 1)
    if rc_reuse:
        w = set_bits(w, 124, 1, 1)
    return _apply_pred(w, p.pred)


# Register all SHF / USHF variants we saw in this program.
for _nm in ("SHF.L.U32", "SHF.L.U64.HI", "SHF.R.U32", "SHF.R.U64",
           "SHF.R.U32.HI", "SHF.R.S32.HI"):
    _TABLE[_nm] = (lambda p, ctx=None, _n=_nm: _shf_impl(p, is_uniform=False))
for _nm in ("USHF.L.U32", "USHF.R.U32.HI", "USHF.R.S32.HI",
            "USHF.R.U64", "USHF.R.U32", "USHF.L.U64.HI"):
    _TABLE[_nm] = (lambda p, ctx=None, _n=_nm: _shf_impl(p, is_uniform=True))


@register("MOV.SPILL")
def enc_mov_spill(p: ParsedInsn) -> int:
    """MOV.SPILL Rd, URb — compiler spill form of MOV, bit 88 set vs plain MOV."""
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"MOV.SPILL arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    src = p.operands[1]
    w = 0
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 72, 4, 0xf)                 # write-mask
    w = set_bits(w, 88, 1, 1)                   # .SPILL flag

    if src.startswith("UR") or src.replace(".reuse", "").strip() == "URZ":
        urb, _ = parse_ureg(src)
        w = set_bits(w, 0, 16, 0x7c02)
        w = set_bits(w, 32, 8, urb)
        w = set_bits(w, 91, 1, 1)
    elif src.startswith("R") or src.replace(".reuse", "").strip() == "RZ":
        rb, _ = parse_reg(src)
        w = set_bits(w, 0, 16, 0x7202)
        w = set_bits(w, 32, 8, rb)
    else:
        raise UnsupportedInstruction(f"MOV.SPILL src {src!r}")
    return _apply_pred(w, p.pred)


@register("F2I.FTZ.U32.TRUNC.NTZ")
def enc_f2i_ftz_u32_trunc_ntz(p: ParsedInsn) -> int:
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"F2I arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    ra, _ = parse_reg(p.operands[1])
    w = 0
    w = set_bits(w, 0, 16, 0x7305)
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 32, 8, ra)
    w = set_bits(w, 72, 8, 0xf0)
    w = set_bits(w, 80, 8, 0x21)
    return _apply_pred(w, p.pred)


@register("F2I.U32.TRUNC.NTZ")
def enc_f2i_u32_trunc_ntz(p: ParsedInsn) -> int:
    """F2I.U32.TRUNC.NTZ — non-FTZ variant (byte 10 bit 80 clear)."""
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"F2I arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    ra, _ = parse_reg(p.operands[1])
    w = 0
    w = set_bits(w, 0, 16, 0x7305)
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 32, 8, ra)
    w = set_bits(w, 72, 8, 0xf0)
    w = set_bits(w, 80, 8, 0x20)
    return _apply_pred(w, p.pred)


@register("UFLO.U32")
def enc_uflo_u32(p: ParsedInsn) -> int:
    """UFLO.U32 URd, URa — find leading one on a uniform register."""
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"UFLO arity {len(p.operands)}")
    urd, _ = parse_ureg(p.operands[0])
    ura, _ = parse_ureg(p.operands[1])
    w = 0
    w = set_bits(w, 0, 16, 0x72bd)
    w = set_bits(w, 16, 8, urd)
    w = set_bits(w, 32, 8, ura)
    w = set_bits(w, 80, 8, 0x0e)                # bits 81..83 = PT sentinel
    w = set_bits(w, 91, 1, 1)                   # UR-form flag
    return _apply_pred(w, p.pred)


@register("DEPBAR")
def enc_depbar(p: ParsedInsn) -> int:
    """DEPBAR {b5,b4,b3,b2,b1,b0} — wait on a set of barrier indices.

    The comma-separated list inside the braces is split by the generic
    operand parser; we rejoin and parse out the integer indices here.
    """
    if not p.operands:
        raise UnsupportedInstruction(f"DEPBAR arity {len(p.operands)}")
    joined = ",".join(op.strip() for op in p.operands).strip()
    if not (joined.startswith("{") and joined.endswith("}")):
        raise UnsupportedInstruction(f"DEPBAR operand {joined!r}")
    inner = joined[1:-1]
    idxs = [int(x.strip()) for x in inner.split(",") if x.strip()]
    mask = 0
    for i in idxs:
        if 0 <= i <= 5:
            mask |= 1 << i
    w = 0
    w = set_bits(w, 0, 16, 0x791a)
    w = set_bits(w, 32, 8, mask & 0x3f)
    return _apply_pred(w, p.pred)


@register("NANOSLEEP")
def enc_nanosleep(p: ParsedInsn) -> int:
    """NANOSLEEP imm — sleep for `imm` nanoseconds."""
    if len(p.operands) != 1:
        raise UnsupportedInstruction(f"NANOSLEEP arity {len(p.operands)}")
    imm = parse_imm(p.operands[0]) & 0xffff
    w = 0
    w = set_bits(w, 0, 16, 0x795d)
    w = set_bits(w, 32, 16, imm)
    w = set_bits(w, 80, 16, 0x0380)
    return _apply_pred(w, p.pred)


@register("PREEXIT")
def enc_preexit(p: ParsedInsn) -> int:
    """PREEXIT — signal pre-exit to the warp scheduler."""
    if len(p.operands) != 0:
        raise UnsupportedInstruction(f"PREEXIT arity {len(p.operands)}")
    w = 0
    w = set_bits(w, 0, 16, 0x782d)
    return _apply_pred(w, p.pred)


@register("YIELD")
def enc_yield(p: ParsedInsn) -> int:
    """YIELD — release the warp scheduler slot this cycle."""
    if len(p.operands) != 0:
        raise UnsupportedInstruction(f"YIELD arity {len(p.operands)}")
    w = 0
    w = set_bits(w, 0, 16, 0x7946)
    w = set_bits(w, 80, 16, 0x0380)
    return _apply_pred(w, p.pred)


@register("UCGABAR_ARV")
def enc_ucgabar_arv(p: ParsedInsn) -> int:
    """UCGABAR_ARV — arrive on the uniform cluster-group barrier."""
    if len(p.operands) != 0:
        raise UnsupportedInstruction(f"UCGABAR_ARV arity {len(p.operands)}")
    w = 0
    w = set_bits(w, 0, 16, 0x79c7)
    w = set_bits(w, 91, 1, 1)                   # fixed tail bit
    return _apply_pred(w, p.pred)


@register("UCGABAR_WAIT")
def enc_ucgabar_wait(p: ParsedInsn) -> int:
    """UCGABAR_WAIT — wait on the uniform cluster-group barrier."""
    if len(p.operands) != 0:
        raise UnsupportedInstruction(f"UCGABAR_WAIT arity {len(p.operands)}")
    w = 0
    w = set_bits(w, 0, 16, 0x7dc7)
    w = set_bits(w, 91, 1, 1)
    return _apply_pred(w, p.pred)


@register("BAR.ARV")
def enc_bar_arv(p: ParsedInsn) -> int:
    """BAR.ARV imm_barrier, imm_count — arrive-only on a CTA-level barrier.

    Field layout (empirical):
      bits 42..53 = count   (12-bit thread-count the barrier is waiting for)
      bits 54..56 = barrier (3-bit barrier index)
      bit  77      = fixed .ARV mode flag (byte 9 = 0x20)
    """
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"BAR.ARV arity {len(p.operands)}")
    bar = parse_imm(p.operands[0]) & 0xf
    cnt = parse_imm(p.operands[1]) & 0xfff
    w = 0
    w = set_bits(w, 0, 16, 0x7b1d)
    w = set_bits(w, 42, 12, cnt)
    w = set_bits(w, 54, 4, bar)
    w = set_bits(w, 72, 8, 0x20)
    return _apply_pred(w, p.pred)


@register("DADD")
def enc_dadd(p: ParsedInsn) -> int:
    """DADD Rd, Ra, Rb — double-precision add (Rb at Rc slot: bits 64..71)."""
    if len(p.operands) != 3:
        raise UnsupportedInstruction(f"DADD arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    ra, _ = parse_reg(p.operands[1])
    rb, _ = parse_reg(p.operands[2])
    w = 0
    w = set_bits(w, 0, 16, 0x7229)
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 64, 8, rb)
    return _apply_pred(w, p.pred)


@register("F2F.F64.F32")
def enc_f2f_f64_f32(p: ParsedInsn) -> int:
    """F2F.F64.F32 Rd, Ra — F32→F64 float-to-float conversion."""
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"F2F arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    ra, _ = parse_reg(p.operands[1])
    w = 0
    w = set_bits(w, 0, 16, 0x7310)
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 32, 8, ra)
    w = set_bits(w, 72, 8, 0x18)
    w = set_bits(w, 80, 8, 0x20)
    return _apply_pred(w, p.pred)


@register("UVIRTCOUNT.DEALLOC.SMPOOL")
def enc_uvirtcount_dealloc_smpool(p: ParsedInsn) -> int:
    """UVIRTCOUNT.DEALLOC.SMPOOL imm — deallocate SM-pool virtual counters."""
    if len(p.operands) != 1:
        raise UnsupportedInstruction(f"UVIRTCOUNT arity {len(p.operands)}")
    imm = parse_imm(p.operands[0]) & 0xff
    w = 0
    w = set_bits(w, 0, 16, 0x784c)
    w = set_bits(w, 32, 8, imm)
    return _apply_pred(w, p.pred)


@register("USETMAXREG.TRY_ALLOC.CTAPOOL")
def enc_usetmaxreg_try_alloc_ctapool(p: ParsedInsn) -> int:
    """USETMAXREG.TRY_ALLOC.CTAPOOL UPd, imm — attempt register-pool alloc."""
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"USETMAXREG.TRY_ALLOC arity {len(p.operands)}")
    upd = _parse_upred_dst(p.operands[0])
    imm = parse_imm(p.operands[1]) & 0xffffffff
    w = 0
    w = set_bits(w, 0, 16, 0x79c8)
    w = set_bits(w, 32, 32, imm)
    w = set_bits(w, 72, 8, 0x06)                # TRY_ALLOC subop
    w = set_bits(w, 81, 3, upd)
    w = set_bits(w, 91, 1, 1)
    return _apply_pred(w, p.pred)


@register("R2UR.OR")
def enc_r2ur_or(p: ParsedInsn) -> int:
    """R2UR.OR Pd, URd, Ra — or-reduce a warp into a uniform register."""
    if len(p.operands) != 3:
        raise UnsupportedInstruction(f"R2UR.OR arity {len(p.operands)}")
    pd = _parse_pred_dst(p.operands[0])
    urd, _ = parse_ureg(p.operands[1])
    ra, ra_reuse = parse_reg(p.operands[2])
    w = 0
    w = set_bits(w, 0, 16, 0x72ca)
    w = set_bits(w, 16, 8, urd)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 80, 8, 0x14)
    w = set_bits(w, 81, 3, pd)
    if ra_reuse:
        w = set_bits(w, 122, 1, 1)
    return _apply_pred(w, p.pred)


@register("RPCMOV.32")
def enc_rpcmov_32(p: ParsedInsn) -> int:
    """RPCMOV.32 Rpc.LO|Rpc.HI, Ra — move a 32-bit half of the PC register."""
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"RPCMOV.32 arity {len(p.operands)}")
    dst = p.operands[0].strip()
    if dst not in ("Rpc.LO", "Rpc.HI"):
        raise UnsupportedInstruction(f"RPCMOV.32 dst {dst!r}")
    ra, _ = parse_reg(p.operands[1])
    w = 0
    w = set_bits(w, 0, 16, 0x7352)
    w = set_bits(w, 32, 8, ra)
    if dst == "Rpc.HI":
        w = set_bits(w, 16, 8, 0x01)
    return _apply_pred(w, p.pred)


@register("CCTL.IVALL")
def enc_cctl_ivall(p: ParsedInsn) -> int:
    """CCTL.IVALL — invalidate all L1 cache lines (no operand)."""
    if len(p.operands) != 0:
        raise UnsupportedInstruction(f"CCTL.IVALL arity {len(p.operands)}")
    w = 0
    w = set_bits(w, 0, 16, 0x798f)
    w = set_bits(w, 24, 8, 0xff)
    w = set_bits(w, 89, 1, 1)
    return _apply_pred(w, p.pred)


@register("FMNMX")
def enc_fmnmx(p: ParsedInsn) -> int:
    """FMNMX Rd, Ra, {Rb|URb|imm}, Ps — FP min/max (Ps=PT→max, !PT→min)."""
    if len(p.operands) != 4:
        raise UnsupportedInstruction(f"FMNMX arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    ra, ra_reuse = parse_reg(p.operands[1])
    b_tok = p.operands[2]
    ps_idx, ps_neg = _parse_pred_src(p.operands[3])

    w = 0
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 87, 3, ps_idx)
    w = set_bits(w, 90, 1, ps_neg)
    if b_tok.startswith("UR") or b_tok.replace(".reuse", "").strip() == "URZ":
        urb, rb_reuse = parse_ureg(b_tok)
        w = set_bits(w, 0, 16, 0x7c09)
        w = set_bits(w, 32, 8, urb)
        w = set_bits(w, 91, 1, 1)
    elif b_tok.startswith("R") or b_tok.replace(".reuse", "").strip() == "RZ":
        rb, rb_reuse = parse_reg(b_tok)
        w = set_bits(w, 0, 16, 0x7209)
        w = set_bits(w, 32, 8, rb)
    else:
        imm = _parse_float_imm(b_tok)
        w = set_bits(w, 0, 16, 0x7409)
        w = set_bits(w, 32, 32, imm)
        rb_reuse = False
    if ra_reuse: w = set_bits(w, 122, 1, 1)
    if rb_reuse: w = set_bits(w, 123, 1, 1)
    return _apply_pred(w, p.pred)


@register("P2R")
def enc_p2r(p: ParsedInsn) -> int:
    """P2R Rd, PR, Ra, imm — pack predicate bits into register byte."""
    if len(p.operands) != 4:
        raise UnsupportedInstruction(f"P2R arity {len(p.operands)}")
    if p.operands[1] != "PR":
        raise UnsupportedInstruction(f"P2R src1 {p.operands[1]!r}")
    rd, _ = parse_reg(p.operands[0])
    ra, _ = parse_reg(p.operands[2])
    imm = parse_imm(p.operands[3]) & 0xffffffff
    w = 0
    w = set_bits(w, 0, 16, 0x7803)
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 32, 32, imm)
    return _apply_pred(w, p.pred)


@register("IABS")
def enc_iabs(p: ParsedInsn) -> int:
    """IABS Rd, {Ra | URa} — integer absolute value."""
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"IABS arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    a_tok = p.operands[1]
    w = 0
    w = set_bits(w, 16, 8, rd)
    if a_tok.startswith("UR") or a_tok.replace(".reuse", "").strip() == "URZ":
        ura, ra_reuse = parse_ureg(a_tok)
        w = set_bits(w, 0, 16, 0x7c13)
        w = set_bits(w, 32, 8, ura)
        w = set_bits(w, 91, 1, 1)
    else:
        ra, ra_reuse = parse_reg(a_tok)
        w = set_bits(w, 0, 16, 0x7213)
        w = set_bits(w, 32, 8, ra)
    if ra_reuse:
        w = set_bits(w, 123, 1, 1)
    return _apply_pred(w, p.pred)


def _mufu_impl(p: ParsedInsn, *, byte9: int) -> int:
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"MUFU arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    ra, _ = parse_reg(p.operands[1])
    w = 0
    w = set_bits(w, 0, 16, 0x7308)
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 32, 8, ra)
    w = set_bits(w, 72, 8, byte9)
    return _apply_pred(w, p.pred)


@register("MUFU.RCP")
def enc_mufu_rcp(p: ParsedInsn) -> int:
    return _mufu_impl(p, byte9=0x10)


@register("MUFU.EX2")
def enc_mufu_ex2(p: ParsedInsn) -> int:
    return _mufu_impl(p, byte9=0x08)


@register("ELECT")
def enc_elect(p: ParsedInsn) -> int:
    """ELECT Pd, URd, Ps — elect one active lane. 3 operands."""
    if len(p.operands) != 3:
        raise UnsupportedInstruction(f"ELECT arity {len(p.operands)}")
    pd = _parse_pred_dst(p.operands[0])
    urd, _ = parse_ureg(p.operands[1])
    ps_idx, ps_neg = _parse_pred_src(p.operands[2])
    w = 0
    w = set_bits(w, 0, 16, 0x782f)
    w = set_bits(w, 16, 8, urd)
    # bits 81..83 = Pd, bit 87 = 1 (fixed), bits 88..89 = 1,1 (Ps=PT part)
    w = set_bits(w, 80, 8, 0x80 | (pd << 1))
    w = set_bits(w, 88, 8, 0x03)
    w = set_bits(w, 87, 3, ps_idx)
    w = set_bits(w, 90, 1, ps_neg)
    return _apply_pred(w, p.pred)


@register("WARPSYNC.ALL")
def enc_warpsync_all(p: ParsedInsn) -> int:
    if p.operands:
        raise UnsupportedInstruction("WARPSYNC.ALL has no operands")
    w = 0
    w = set_bits(w, 0, 16, 0x7948)
    w = set_bits(w, 80, 16, 0x0380)
    return _apply_pred(w, p.pred)


@register("IMAD.MOV")
def enc_imad_mov(p: ParsedInsn) -> int:
    """IMAD.MOV Rd, RZ, RZ, [-]Rc — signed move through IMAD."""
    if len(p.operands) != 4:
        raise UnsupportedInstruction(f"IMAD.MOV arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    ra, _ = parse_reg(p.operands[1])
    rb, _ = parse_reg(p.operands[2])
    c_tok = p.operands[3]
    rc_neg = c_tok.startswith("-")
    if rc_neg:
        c_tok = c_tok[1:].strip()
    rc, rc_reuse = parse_reg(c_tok)
    if ra != 255 or rb != 255:
        raise UnsupportedInstruction("IMAD.MOV expects Ra=RZ Rb=RZ")
    w = 0
    w = set_bits(w, 0, 16, 0x7224)
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, 0xff)
    w = set_bits(w, 32, 8, 0xff)
    w = set_bits(w, 64, 8, rc)
    # byte 9: 0x02 signed, plus bit 75 = Rc-negate.
    byte9 = 0x02 | (0x08 if rc_neg else 0)
    w = set_bits(w, 72, 8, byte9)
    w = set_bits(w, 80, 16, 0x078e)
    if rc_reuse:
        w = set_bits(w, 124, 1, 1)
    return _apply_pred(w, p.pred)


@register("UPLOP3.LUT")
def enc_uplop3_lut(p: ParsedInsn) -> int:
    """UPLOP3.LUT UPd1, UPd2, UPa, UPb, UPc, imm1, imm2

    Handles the common "UPd1, UPT, UPT, UPT, UPT, 0x??, 0x?" form — the
    uniform analog of PLOP3.LUT.
    """
    if len(p.operands) != 7:
        raise UnsupportedInstruction(f"UPLOP3.LUT arity {len(p.operands)}")
    pd1 = _parse_upred_dst(p.operands[0])
    pd2 = _parse_upred_dst(p.operands[1])
    pa = _parse_upred_dst(p.operands[2])
    pb = _parse_upred_dst(p.operands[3])
    pc = _parse_upred_dst(p.operands[4])
    imm1 = parse_imm(p.operands[5]) & 0xff
    imm2 = parse_imm(p.operands[6]) & 0xff
    if pd2 != 7 or pa != 7 or pb != 7 or pc != 7:
        raise UnsupportedInstruction("UPLOP3.LUT Pa/Pb/Pc/Pd2 must be UPT")
    # Only the specific (imm1=0x80,imm2=0x08) and (imm1=0x40,imm2=0x04) forms
    # are observed — these are the "AND-reduce" and "half-reduce" patterns.
    if (imm1, imm2) not in ((0x80, 0x08), (0x40, 0x04)):
        raise UnsupportedInstruction("UPLOP3.LUT imm combo")

    w = 0
    w = set_bits(w, 0, 16, 0x789c)
    w = set_bits(w, 16, 8, imm2)
    # byte 8 is always 0x70 (Pc=UPT with uniform flag + special bits).
    w = set_bits(w, 64, 8, 0x70)
    # byte 9: 0xe8 for imm pair (0x40,0x4), 0xf0 for (0x80,0x8).
    w = set_bits(w, 72, 8, 0xe8 if imm1 == 0x40 else 0xf0)
    w = set_bits(w, 80, 8, 0xf0 | (pd1 << 1))
    w = set_bits(w, 88, 8, 0x03)
    return _apply_pred(w, p.pred)


@register("PRMT")
def enc_prmt(p: ParsedInsn) -> int:
    """PRMT Rd, Ra, imm, Rc — byte permute."""
    if len(p.operands) != 4:
        raise UnsupportedInstruction(f"PRMT arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    ra, ra_reuse = parse_reg(p.operands[1])
    b_tok = p.operands[2]
    rc, rc_reuse = parse_reg(p.operands[3])

    w = 0
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 64, 8, rc)

    if b_tok.startswith("UR") or b_tok.replace(".reuse", "").strip() == "URZ":
        urb, rb_reuse = parse_ureg(b_tok)
        w = set_bits(w, 0, 16, 0x7c16)
        w = set_bits(w, 32, 8, urb)
        w = set_bits(w, 91, 1, 1)
    elif b_tok.startswith("R") or b_tok.replace(".reuse", "").strip() == "RZ":
        rb, rb_reuse = parse_reg(b_tok)
        w = set_bits(w, 0, 16, 0x7216)
        w = set_bits(w, 32, 8, rb)
    else:
        imm = parse_imm(b_tok) & 0xffffffff
        w = set_bits(w, 0, 16, 0x7816)
        w = set_bits(w, 32, 32, imm)
        rb_reuse = False

    if ra_reuse:
        w = set_bits(w, 122, 1, 1)
    if rb_reuse:
        w = set_bits(w, 123, 1, 1)
    if rc_reuse:
        w = set_bits(w, 124, 1, 1)
    return _apply_pred(w, p.pred)


@register("UPRMT")
def enc_uprmt(p: ParsedInsn) -> int:
    """UPRMT URd, URa, imm|URb, URc"""
    if len(p.operands) != 4:
        raise UnsupportedInstruction(f"UPRMT arity {len(p.operands)}")
    urd, _ = parse_ureg(p.operands[0])
    ura, ura_reuse = parse_ureg(p.operands[1])
    b_tok = p.operands[2]
    urc, urc_reuse = parse_ureg(p.operands[3])

    w = 0
    w = set_bits(w, 16, 8, urd)
    w = set_bits(w, 24, 8, ura)
    w = set_bits(w, 64, 8, urc)

    if b_tok.startswith("UR") or b_tok.replace(".reuse", "").strip() == "URZ":
        urb, rb_reuse = parse_ureg(b_tok)
        w = set_bits(w, 0, 16, 0x7296)
        w = set_bits(w, 32, 8, urb)
        w = set_bits(w, 91, 1, 1)
    else:
        imm = parse_imm(b_tok) & 0xffffffff
        w = set_bits(w, 0, 16, 0x7896)
        w = set_bits(w, 32, 32, imm)
        rb_reuse = False
        w = set_bits(w, 91, 1, 1)

    if ura_reuse:
        w = set_bits(w, 122, 1, 1)
    if rb_reuse:
        w = set_bits(w, 123, 1, 1)
    if urc_reuse:
        w = set_bits(w, 124, 1, 1)
    return _apply_pred(w, p.pred)


def _fp16_to_bits(tok: str) -> int:
    """Convert a printed FP16 immediate (e.g. '0.1171875' or '0') into its
    16-bit IEEE-754 half encoding."""
    if tok in ("0", "0x0"):
        return 0
    try:
        f = float(tok)
    except ValueError:
        raise UnsupportedInstruction(f"bad fp16 imm {tok!r}")
    # Convert via numpy if available, else manual.
    try:
        import numpy as np
        return int(np.float16(f).view(np.uint16))
    except Exception:
        import struct
        # Manual: cast float->half via struct pack/unpack of a half-precision
        # value; Python's struct supports 'e' for half.
        return int.from_bytes(struct.pack("<e", f), "little")


@register("HFMA2")
def enc_hfma2(p: ParsedInsn) -> int:
    """HFMA2 Rd, [-]Ra, Rb, imm_H, imm_L — packed FP16×2 FMA.

    Observed in 70_blackwell_fp16_gemm only in the "splat constant" idiom with
    Ra = -RZ, Rb = RZ: Rd = RZ*RZ + (imm_H, imm_L).
    """
    if len(p.operands) != 5:
        raise UnsupportedInstruction(f"HFMA2 arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    a_tok = p.operands[1]
    ra_neg = a_tok.startswith("-")
    if ra_neg:
        a_tok = a_tok[1:].strip()
    ra, _ = parse_reg(a_tok)
    rb, _ = parse_reg(p.operands[2])
    imm_h = _fp16_to_bits(p.operands[3]) & 0xffff
    imm_l = _fp16_to_bits(p.operands[4]) & 0xffff

    w = 0
    w = set_bits(w, 0, 16, 0x7431)
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 32, 16, imm_l)
    w = set_bits(w, 48, 16, imm_h)
    w = set_bits(w, 64, 8, rb)
    w = set_bits(w, 72, 8, 0x01)
    return _apply_pred(w, p.pred)


@register("R2P")
def enc_r2p(p: ParsedInsn) -> int:
    """R2P PR, Ra[.B0/.B1/.B2/.B3], imm32 — pack register bits to predicates."""
    if len(p.operands) != 3:
        raise UnsupportedInstruction(f"R2P arity {len(p.operands)}")
    if p.operands[0] != "PR":
        raise UnsupportedInstruction(f"R2P dst {p.operands[0]!r}")
    a_tok = p.operands[1]
    byte_sel = 0
    if "." in a_tok:
        a_base, _, swiz = a_tok.partition(".")
        a_tok = a_base
        if swiz == "B0":
            byte_sel = 0
        elif swiz == "B1":
            byte_sel = 1
        elif swiz == "B2":
            byte_sel = 2
        elif swiz == "B3":
            byte_sel = 3
        else:
            raise UnsupportedInstruction(f"R2P swiz {swiz!r}")
    ra, ra_reuse = parse_reg(a_tok)
    imm = parse_imm(p.operands[2]) & 0xffffffff

    w = 0
    w = set_bits(w, 0, 16, 0x7804)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 32, 32, imm)
    # Byte-select at bits 76..77 (observed: .B1 sets 0x10 in byte 9 = bit 76).
    w = set_bits(w, 76, 2, byte_sel)
    if ra_reuse:
        w = set_bits(w, 122, 1, 1)
    return _apply_pred(w, p.pred)


SHFL_SUBOP = {"IDX": 0, "UP": 1, "DOWN": 2, "BFLY": 3}


def _shfl_impl(p: ParsedInsn, *, subop: int) -> int:
    """SHFL.<subop> Pd, Rd, Ra, Rb|imm5, Rc|imm13 — warp shuffle.

    Layout (structural bits, ignoring predicate guard):
      opcode       bits 0..7    = 0x89
      byte 1       bits 8..15   = 0x75 base; bit 10 set if Rc is imm,
                                  bit 11 set if Rb is imm, bit 9 set if both.
      Rd           bits 16..23
      Ra           bits 24..31
      Rb           bits 32..39  = 0xff when Rb is imm
      imm_c        bits 40..52  (13-bit shuffle mask)
      imm_b        bits 53..57  (5-bit shift amount)
      subop        bits 58..59  (IDX=0, UP=1, DOWN=2, BFLY=3)
      Rc           bits 64..71
      bits 80..87  = 0x0e (fixed tail flag across all SHFL forms)
    """
    if len(p.operands) != 5:
        raise UnsupportedInstruction(f"SHFL arity {len(p.operands)}")
    if p.operands[0] != "PT":
        raise UnsupportedInstruction(f"SHFL Pd {p.operands[0]!r}")
    rd, _ = parse_reg(p.operands[1])
    ra, _ = parse_reg(p.operands[2])
    b_tok = p.operands[3]
    c_tok = p.operands[4]

    b_is_imm = not (b_tok.startswith("R") or b_tok == "RZ")
    c_is_imm = not (c_tok.startswith("R") or c_tok == "RZ")

    byte1 = 0x71
    if c_is_imm: byte1 |= 0x04                               # bit 10
    if b_is_imm: byte1 |= 0x08                               # bit 11
    if b_is_imm and c_is_imm: byte1 |= 0x02                  # bit 9

    w = 0
    w = set_bits(w, 0, 8, 0x89)
    w = set_bits(w, 8, 8, byte1)
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, ra)

    if b_is_imm:
        # When Rb slot is immediate, byte 4 stays zero; the imm lives at bits 53..57.
        w = set_bits(w, 53, 5, parse_imm(b_tok) & 0x1f)
    else:
        rb, _ = parse_reg(b_tok)
        w = set_bits(w, 32, 8, rb)

    if c_is_imm:
        # When Rc slot is immediate, byte 8 stays zero; the imm lives at bits 40..52.
        w = set_bits(w, 40, 13, parse_imm(c_tok) & 0x1fff)
    else:
        rc, _ = parse_reg(c_tok)
        w = set_bits(w, 64, 8, rc)

    w = set_bits(w, 58, 2, subop)
    w = set_bits(w, 80, 8, 0x0e)
    return _apply_pred(w, p.pred)


@register("SHFL.IDX")
def enc_shfl_idx(p: ParsedInsn) -> int:
    return _shfl_impl(p, subop=SHFL_SUBOP["IDX"])


@register("SHFL.UP")
def enc_shfl_up(p: ParsedInsn) -> int:
    return _shfl_impl(p, subop=SHFL_SUBOP["UP"])


@register("SHFL.DOWN")
def enc_shfl_down(p: ParsedInsn) -> int:
    return _shfl_impl(p, subop=SHFL_SUBOP["DOWN"])


@register("SHFL.BFLY")
def enc_shfl_bfly(p: ParsedInsn) -> int:
    return _shfl_impl(p, subop=SHFL_SUBOP["BFLY"])


@register("VOTEU.ANY")
def enc_voteu_any(p: ParsedInsn) -> int:
    """VOTEU.ANY URd, UPs, Ps  (or UPd, Ps — other form)."""
    if len(p.operands) == 2:
        # UPd, Ps form — similar to ELECT structurally.
        upd = _parse_upred_dst(p.operands[0])
        ps_idx, ps_neg = _parse_pred_src(p.operands[1])
        w = 0
        w = set_bits(w, 0, 16, 0x7886)
        w = set_bits(w, 16, 8, 0xff)             # URd = URZ (unused)
        w = set_bits(w, 72, 8, 0x01)
        # UPd at bits 81..83, bit 82 = ? bit 87 varies.
        w = set_bits(w, 81, 3, upd)
        w = set_bits(w, 87, 3, ps_idx)
        w = set_bits(w, 90, 1, ps_neg)
        return _apply_pred(w, p.pred)
    if len(p.operands) == 3:
        urd, _ = parse_ureg(p.operands[0])
        ups_idx, ups_neg = _parse_pred_src(p.operands[1])
        ps_idx, ps_neg = _parse_pred_src(p.operands[2])
        w = 0
        w = set_bits(w, 0, 16, 0x7886)
        w = set_bits(w, 16, 8, urd)
        w = set_bits(w, 72, 8, 0x01)
        # UPs at bits 81..83, bit 87 = 1 fixed.
        w = set_bits(w, 80, 8, 0x80)
        w = set_bits(w, 81, 3, ups_idx)
        w = set_bits(w, 88, 8, 0x03)
        w = set_bits(w, 87, 3, ps_idx)
        w = set_bits(w, 90, 1, ps_neg)
        return _apply_pred(w, p.pred)
    raise UnsupportedInstruction("VOTEU.ANY non-3-operand form")


@register("DEPBAR.LE")
def enc_depbar_le(p: ParsedInsn) -> int:
    """DEPBAR.LE SB<n>, imm  — wait until dependency barrier count ≤ imm."""
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"DEPBAR.LE arity {len(p.operands)}")
    m = re.fullmatch(r"SB(\d+)", p.operands[0].strip())
    if not m:
        raise UnsupportedInstruction(f"DEPBAR.LE SB {p.operands[0]!r}")
    sb = int(m.group(1))
    imm = parse_imm(p.operands[1]) & 0x3f
    w = 0
    w = set_bits(w, 0, 16, 0x791a)
    # bits 38..43 = imm (6 bits), bits 44..46 = SB idx, bit 47 = 1.
    w = set_bits(w, 38, 6, imm)
    w = set_bits(w, 44, 3, sb)
    w = set_bits(w, 47, 1, 1)
    return _apply_pred(w, p.pred)


@register("UP2UR")
def enc_up2ur(p: ParsedInsn) -> int:
    """UP2UR URd, UPR, URb, imm — uniform predicate to uniform register."""
    if len(p.operands) != 4:
        raise UnsupportedInstruction(f"UP2UR arity {len(p.operands)}")
    urd, _ = parse_ureg(p.operands[0])
    if p.operands[1] != "UPR":
        raise UnsupportedInstruction(f"UP2UR src1 {p.operands[1]!r}")
    urb, _ = parse_ureg(p.operands[2])
    imm = parse_imm(p.operands[3]) & 0xffffffff
    w = 0
    w = set_bits(w, 0, 16, 0x7883)
    w = set_bits(w, 16, 8, urd)
    w = set_bits(w, 24, 8, urb)
    w = set_bits(w, 32, 32, imm)
    w = set_bits(w, 91, 1, 1)
    return _apply_pred(w, p.pred)


@register("FENCE.VIEW.ASYNC.S")
def enc_fence_view_async_s(p: ParsedInsn) -> int:
    if p.operands:
        raise UnsupportedInstruction("FENCE.VIEW.ASYNC.S has no operands")
    w = 0
    w = set_bits(w, 0, 16, 0x73c6)
    return _apply_pred(w, p.pred)


@register("BPT.TRAP")
def enc_bpt_trap(p: ParsedInsn) -> int:
    if len(p.operands) != 1:
        raise UnsupportedInstruction(f"BPT.TRAP arity {len(p.operands)}")
    imm = parse_imm(p.operands[0]) & 0xffffffff
    w = 0
    w = set_bits(w, 0, 16, 0x795c)
    # 0x1 -> bytes 4-7 = 04 00 00 00 = 4. Store imm << 2.
    w = set_bits(w, 32, 32, (imm << 2) & 0xffffffff)
    w = set_bits(w, 80, 8, 0x30)
    return _apply_pred(w, p.pred)


@register("BAR.SYNC.DEFER_BLOCKING")
def enc_bar_sync_defer_blocking(p: ParsedInsn) -> int:
    """BAR.SYNC.DEFER_BLOCKING 0x<bar-id> [, 0x<count>]

    Only the 1-operand (bar-id) form is observed in FMHA. For `0x0`: bytes
    are fixed; for non-zero bar-id we extend the value field at bits 38+.
    """
    if len(p.operands) not in (1, 2):
        raise UnsupportedInstruction("BAR.SYNC.DEFER_BLOCKING arity")
    bar = parse_imm(p.operands[0]) & 0xff
    w = 0
    w = set_bits(w, 0, 16, 0x7b1d)
    if bar:
        # 1-operand form with nonzero bar encodes the id at bits 38..?
        # Observed only for 0x0 here; just put the raw value at bits 54..
        w = set_bits(w, 54, 8, bar)
    if len(p.operands) == 2:
        cnt = parse_imm(p.operands[1]) & 0xfff
        w = set_bits(w, 42, 12, cnt)
    w = set_bits(w, 80, 8, 0x01)
    return _apply_pred(w, p.pred)


@register("MEMBAR.ALL.CTA")
def enc_membar_all_cta(p: ParsedInsn) -> int:
    if p.operands:
        raise UnsupportedInstruction("MEMBAR.ALL.CTA has no operands")
    w = 0
    w = set_bits(w, 0, 16, 0x7992)
    w = set_bits(w, 72, 8, 0x80)
    return _apply_pred(w, p.pred)


@register("NANOSLEEP.SYNCS")
def enc_nanosleep_syncs(p: ParsedInsn) -> int:
    if len(p.operands) != 1:
        raise UnsupportedInstruction(f"NANOSLEEP.SYNCS arity {len(p.operands)}")
    imm = parse_imm(p.operands[0]) & 0xffffffff
    w = 0
    w = set_bits(w, 0, 16, 0x795d)
    w = set_bits(w, 32, 32, imm)
    w = set_bits(w, 80, 16, 0x0390)
    return _apply_pred(w, p.pred)


@register("VIMNMX")
def enc_vimnmx(p: ParsedInsn) -> int:
    """VIMNMX Rd, Pd1, Pd2, Ra, {Rb|URb|imm}, Ps — video integer min/max."""
    if len(p.operands) != 6:
        raise UnsupportedInstruction(f"VIMNMX arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    pd1 = _parse_pred_dst(p.operands[1])
    pd2 = _parse_pred_dst(p.operands[2])
    ra, ra_reuse = parse_reg(p.operands[3])
    b_tok = p.operands[4]
    ps_idx, ps_neg = _parse_pred_src(p.operands[5])

    w = 0
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 72, 8, 0x01)
    w = set_bits(w, 80, 16, 0x03fe)              # bit 91 set later for UR form
    w = set_bits(w, 81, 3, pd1)
    w = set_bits(w, 84, 3, pd2)
    w = set_bits(w, 87, 3, ps_idx)
    w = set_bits(w, 90, 1, ps_neg)

    if b_tok.startswith("UR") or b_tok.replace(".reuse", "").strip() == "URZ":
        urb, rb_reuse = parse_ureg(b_tok)
        w = set_bits(w, 0, 16, 0x7c48)
        w = set_bits(w, 32, 8, urb)
        w = set_bits(w, 91, 1, 1)
    elif b_tok.startswith("R") or b_tok.replace(".reuse", "").strip() == "RZ":
        rb, rb_reuse = parse_reg(b_tok)
        w = set_bits(w, 0, 16, 0x7248)
        w = set_bits(w, 32, 8, rb)
    else:
        imm = parse_imm(b_tok) & 0xffffffff
        w = set_bits(w, 0, 16, 0x7848)
        w = set_bits(w, 32, 32, imm)
        rb_reuse = False
    if ra_reuse: w = set_bits(w, 122, 1, 1)
    if rb_reuse: w = set_bits(w, 123, 1, 1)
    return _apply_pred(w, p.pred)


@register("VIADD")
def enc_viadd(p: ParsedInsn) -> int:
    """VIADD Rd, Ra, {imm32 | URb}"""
    if len(p.operands) != 3:
        raise UnsupportedInstruction(f"VIADD arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    ra, ra_reuse = parse_reg(p.operands[1])
    b_tok = p.operands[2]
    w = 0
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, ra)
    if b_tok.startswith("UR") or b_tok.replace(".reuse", "").strip() == "URZ":
        urb, rb_reuse = parse_ureg(b_tok)
        w = set_bits(w, 0, 16, 0x7c36)
        w = set_bits(w, 32, 8, urb)
        w = set_bits(w, 91, 1, 1)
        if rb_reuse: w = set_bits(w, 123, 1, 1)
    else:
        imm = parse_imm(b_tok) & 0xffffffff
        w = set_bits(w, 0, 16, 0x7836)
        w = set_bits(w, 32, 32, imm)
    if ra_reuse:
        w = set_bits(w, 122, 1, 1)
    return _apply_pred(w, p.pred)


@register("SEL")
def enc_sel(p: ParsedInsn) -> int:
    """SEL Rd, Ra, {Rb | URb | imm32}, Ps"""
    if len(p.operands) != 4:
        raise UnsupportedInstruction(f"SEL arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    ra, ra_reuse = parse_reg(p.operands[1])
    b_tok = p.operands[2]
    ps_idx, ps_neg = _parse_pred_src(p.operands[3])

    w = 0
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 87, 3, ps_idx)
    w = set_bits(w, 90, 1, ps_neg)

    if b_tok.startswith("UR") or b_tok.replace(".reuse", "").strip() == "URZ":
        urb, rb_reuse = parse_ureg(b_tok)
        w = set_bits(w, 0, 16, 0x7c07)
        w = set_bits(w, 32, 8, urb)
        w = set_bits(w, 91, 1, 1)
    elif b_tok.startswith("R") or b_tok.replace(".reuse", "").strip() == "RZ":
        rb, rb_reuse = parse_reg(b_tok)
        w = set_bits(w, 0, 16, 0x7207)
        w = set_bits(w, 32, 8, rb)
    else:
        imm = parse_imm(b_tok) & 0xffffffff
        w = set_bits(w, 0, 16, 0x7807)
        w = set_bits(w, 32, 32, imm)
        rb_reuse = False

    if ra_reuse:
        w = set_bits(w, 122, 1, 1)
    if rb_reuse:
        w = set_bits(w, 123, 1, 1)
    return _apply_pred(w, p.pred)


@register("USEL")
def enc_usel(p: ParsedInsn) -> int:
    """USEL URd, URa, {URb | imm32}, Ps

    Selects URa if Ps true else URb/imm. Uniform analogue of SEL.
    """
    if len(p.operands) != 4:
        raise UnsupportedInstruction(f"USEL arity {len(p.operands)}")
    urd, _ = parse_ureg(p.operands[0])
    ura, ura_reuse = parse_ureg(p.operands[1])
    b_tok = p.operands[2]
    ps_idx, ps_neg = _parse_pred_src(p.operands[3])

    w = 0
    w = set_bits(w, 16, 8, urd)
    w = set_bits(w, 24, 8, ura)
    w = set_bits(w, 87, 3, ps_idx)
    w = set_bits(w, 90, 1, ps_neg)
    w = set_bits(w, 91, 1, 1)                   # UR form flag

    if b_tok.startswith("UR") or b_tok.replace(".reuse", "").strip() == "URZ":
        urb, urb_reuse = parse_ureg(b_tok)
        w = set_bits(w, 0, 16, 0x7287)
        w = set_bits(w, 32, 8, urb)
    else:
        imm = parse_imm(b_tok) & 0xffffffff
        w = set_bits(w, 0, 16, 0x7887)
        w = set_bits(w, 32, 32, imm)
        urb_reuse = False

    if ura_reuse:
        w = set_bits(w, 122, 1, 1)
    if urb_reuse:
        w = set_bits(w, 123, 1, 1)
    return _apply_pred(w, p.pred)


@register("LOP3.LUT")
def enc_lop3_lut(p: ParsedInsn) -> int:
    return _lop3_lut(p)


@register("PLOP3.LUT")
def enc_plop3_lut(p: ParsedInsn) -> int:
    """PLOP3.LUT Pd1, Pd2, Pa, Pb, Pc, imm_lut, imm_lut2

    Only handles the common "Pd1, PT, PT, PT, UPn, 0x80, 0x8" pattern
    observed in the FP16 GEMM — the case where the compiler emits a 3-input
    predicate logic operation whose truth table reduces to a single bit
    (imm_lut=0x80, imm_lut2=0x08).
    """
    if len(p.operands) != 7:
        raise UnsupportedInstruction(f"PLOP3.LUT arity {len(p.operands)}")
    pd1 = _parse_pred_dst(p.operands[0])
    pd2 = _parse_pred_dst(p.operands[1])
    pa = _parse_pred_dst(p.operands[2])
    pb = _parse_pred_dst(p.operands[3])
    pc_tok = p.operands[4]
    imm1 = parse_imm(p.operands[5])
    imm2 = parse_imm(p.operands[6])

    if pd2 != 7 or pa != 7 or pb != 7:
        raise UnsupportedInstruction("PLOP3.LUT Pa/Pb/Pd2 must be PT")
    if (imm1, imm2) != (0x80, 0x08):
        raise UnsupportedInstruction("PLOP3.LUT imm must be (0x80, 0x8)")
    if pc_tok.startswith("UP"):
        pc_idx = _parse_upred_dst(pc_tok)
        pc_uniform = 1
    else:
        pc_idx = _parse_pred_dst(pc_tok)
        pc_uniform = 0

    w = 0
    w = set_bits(w, 0, 16, 0x781c)
    w = set_bits(w, 16, 8, imm2)
    if pc_uniform:
        w = set_bits(w, 64, 8, 0x08 | (pc_idx << 4))
    else:
        w = set_bits(w, 64, 8, 0x70 | (0 if pc_idx == 7 else pc_idx << 4))
    w = set_bits(w, 72, 8, 0xf0)
    w = set_bits(w, 80, 8, 0xf0 | (pd1 << 1))
    w = set_bits(w, 88, 8, 0x03)
    return _apply_pred(w, p.pred)


def _ulop3_lut(p: ParsedInsn) -> int:
    """ULOP3.LUT [UPd,] URd, URa, {UR|imm}, URc, imm_lut, UPs — uniform LOP3."""
    ops = list(p.operands)
    if ops and (ops[0] == "UPT" or re.fullmatch(r"UP\d", ops[0])):
        pd_idx = _parse_upred_dst(ops[0])
        ops = ops[1:]
    else:
        pd_idx = 7
    if len(ops) != 6:
        raise UnsupportedInstruction(f"ULOP3.LUT arity {len(ops)}")
    urd, _ = parse_ureg(ops[0])
    ura, ura_reuse = parse_ureg(ops[1])
    b_tok = ops[2]
    urc, urc_reuse = parse_ureg(ops[3])
    lut = parse_imm(ops[4]) & 0xff
    ps_idx, ps_neg = _parse_pred_src(ops[5])

    w = 0
    w = set_bits(w, 16, 8, urd)
    w = set_bits(w, 24, 8, ura)
    w = set_bits(w, 64, 8, urc)
    w = set_bits(w, 72, 8, lut)
    w = set_bits(w, 81, 3, pd_idx)
    w = set_bits(w, 87, 1, ps_neg)
    w = set_bits(w, 88, 3, ps_idx)
    w = set_bits(w, 91, 1, 1)                   # UR form

    if b_tok.startswith("UR") or b_tok.replace(".reuse", "").strip() == "URZ":
        urb, rb_reuse = parse_ureg(b_tok)
        w = set_bits(w, 0, 16, 0x7292)
        w = set_bits(w, 32, 8, urb)
    else:
        imm = parse_imm(b_tok) & 0xffffffff
        w = set_bits(w, 0, 16, 0x7892)
        w = set_bits(w, 32, 32, imm)
        rb_reuse = False

    if ura_reuse:
        w = set_bits(w, 122, 1, 1)
    if rb_reuse:
        w = set_bits(w, 123, 1, 1)
    if urc_reuse:
        w = set_bits(w, 124, 1, 1)
    return _apply_pred(w, p.pred)


@register("ULOP3.LUT")
def enc_ulop3_lut(p: ParsedInsn) -> int:
    return _ulop3_lut(p)


def _imad_like(p: ParsedInsn, *, base_r: int, base_ur: int, base_imm: int,
               byte9: int, is_uniform: bool) -> int:
    """Encode a generic IMAD / IMAD.U32 / UIMAD with 4 operands:
       Rd, Ra, {Rb | URb | imm32}, Rc

    base_r/base_ur/base_imm: bits 0..15 for the three source-B forms.
    byte9: value for bits 72..79 (varies with signed/unsigned, .X, etc).
    """
    if len(p.operands) != 4:
        raise UnsupportedInstruction(f"IMAD arity {len(p.operands)}")

    parse_d = parse_ureg if is_uniform else parse_reg
    rd, _ = parse_d(p.operands[0])
    ra, ra_reuse = parse_d(p.operands[1])
    b_tok = p.operands[2]
    c_tok = p.operands[3]

    # `-Rc` and `~Rc` both set the Rc-negate flag (bit 75 for reg-Rc, bit 63
    # in the UR-at-C swap form).
    rc_neg = c_tok.startswith("-") or c_tok.startswith("~")
    if rc_neg:
        c_tok = c_tok[1:].strip()

    # "UR-at-C" idiom: when Rc is a uniform register, the disassembler shows
    # the UR in the Rc position but the encoding stores it at the Rb slot and
    # flips opcode bit 9 (base |= 0x0200).
    c_is_ur = c_tok.startswith("UR") or c_tok.replace(".reuse", "").strip() == "URZ"

    w = 0
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, ra)
    # byte 9 bit 75 is the Rc-negate flag for all forms except the non-uniform
    # UR-at-C swap form (which uses bit 63 instead — handled below).
    in_swap_form = c_is_ur and not is_uniform
    w = set_bits(w, 72, 8, byte9 | (0x08 if (rc_neg and not in_swap_form) else 0))
    w = set_bits(w, 80, 16, 0x078e)              # fixed high-word pattern

    if c_is_ur and not is_uniform:
        # Swap: UR at bits 32..39, Rb_text at bits 64..71.
        urc, urc_reuse = parse_ureg(c_tok)
        # b_tok is the displayed Rb position (usually RZ for move idiom).
        rb_val, rb_reuse = parse_reg(b_tok)
        w = set_bits(w, 0, 16, base_ur | 0x0200)
        w = set_bits(w, 32, 8, urc)
        w = set_bits(w, 64, 8, rb_val)
        w = set_bits(w, 91, 1, 1)
        # -URc in UR-at-C form is encoded via bit 63.
        if rc_neg:
            w = set_bits(w, 63, 1, 1)
        rc_reuse = urc_reuse
    else:
        rc, rc_reuse = parse_d(c_tok)
        w = set_bits(w, 64, 8, rc)
        if b_tok.startswith("UR") or b_tok.replace(".reuse", "").strip() == "URZ":
            urb, rb_reuse = parse_ureg(b_tok)
            w = set_bits(w, 0, 16, base_ur)
            w = set_bits(w, 32, 8, urb)
            w = set_bits(w, 91, 1, 1)
        elif b_tok.startswith("R") or b_tok.replace(".reuse", "").strip() == "RZ":
            rb, rb_reuse = parse_reg(b_tok) if not is_uniform else parse_ureg(b_tok)
            w = set_bits(w, 0, 16, base_r)
            w = set_bits(w, 32, 8, rb)
            if is_uniform:
                w = set_bits(w, 91, 1, 1)
        else:
            imm = parse_imm(b_tok) & 0xffffffff
            w = set_bits(w, 0, 16, base_imm)
            w = set_bits(w, 32, 32, imm)
            rb_reuse = False
            if is_uniform:
                w = set_bits(w, 91, 1, 1)

    if ra_reuse:
        w = set_bits(w, 122, 1, 1)
    if rb_reuse:
        w = set_bits(w, 123, 1, 1)
    if rc_reuse:
        w = set_bits(w, 124, 1, 1)
    return _apply_pred(w, p.pred)


@register("IMAD")
def enc_imad(p: ParsedInsn) -> int:
    return _imad_like(p, base_r=0x7224, base_ur=0x7c24, base_imm=0x7824,
                      byte9=0x02, is_uniform=False)


@register("IMAD.U32")
def enc_imad_u32(p: ParsedInsn) -> int:
    return _imad_like(p, base_r=0x7224, base_ur=0x7c24, base_imm=0x7824,
                      byte9=0x00, is_uniform=False)


@register("UIMAD")
def enc_uimad(p: ParsedInsn) -> int:
    # UIMAD is signed by default? Observed byte 9 = 0x02 (matches IMAD).
    return _imad_like(p, base_r=0x72a4, base_ur=0x72a4, base_imm=0x78a4,
                      byte9=0x02, is_uniform=True)


@register("IMAD.SHL.U32")
def enc_imad_shl_u32(p: ParsedInsn) -> int:
    # Same layout as IMAD.U32 imm form.
    return _imad_like(p, base_r=0x7224, base_ur=0x7c24, base_imm=0x7824,
                      byte9=0x00, is_uniform=False)


@register("IMAD.WIDE.U32")
def enc_imad_wide_u32(p: ParsedInsn) -> int:
    # Optional 5th-operand form: Rd, Pcarry_out, Ra, Rb, Rc.
    if len(p.operands) == 5 and re.fullmatch(r"P[0-9T]", p.operands[1]):
        pc = _parse_pred_dst(p.operands[1])
        p4 = ParsedInsn(pred=p.pred, mnemonic=p.mnemonic,
                         operands=[p.operands[0]] + p.operands[2:])
        w = _imad_like(p4, base_r=0x7225, base_ur=0x7c25, base_imm=0x7825,
                       byte9=0x00, is_uniform=False)
        # Overwrite bits 80..95 default (0x078e) so bits 81..83 carry pc.
        w = set_bits(w, 80, 8, 0x80 | (pc << 1))
        return w
    return _imad_like(p, base_r=0x7225, base_ur=0x7c25, base_imm=0x7825,
                      byte9=0x00, is_uniform=False)


@register("IMAD.WIDE")
def enc_imad_wide(p: ParsedInsn) -> int:
    return _imad_like(p, base_r=0x7225, base_ur=0x7c25, base_imm=0x7825,
                      byte9=0x02, is_uniform=False)


@register("IMAD.IADD")
def enc_imad_iadd(p: ParsedInsn) -> int:
    # Observed as plain IMAD form with signed byte9, used as a degenerate
    # multiply-accumulate (Rb = 1) for fused add through the IMAD pipeline.
    return _imad_like(p, base_r=0x7224, base_ur=0x7c24, base_imm=0x7824,
                      byte9=0x02, is_uniform=False)


@register("UIMAD.WIDE.U32")
def enc_uimad_wide_u32(p: ParsedInsn) -> int:
    # Optional 5th-op form: UIMAD.WIDE.U32 URd, UPcarry_out, URa, URb, URc.
    if len(p.operands) == 5 and re.fullmatch(r"UP[0-9T]", p.operands[1]):
        pc = _parse_upred_dst(p.operands[1])
        p4 = ParsedInsn(pred=p.pred, mnemonic=p.mnemonic,
                         operands=[p.operands[0]] + p.operands[2:])
        w = _imad_like(p4, base_r=0x72a5, base_ur=0x72a5, base_imm=0x78a5,
                       byte9=0x00, is_uniform=True)
        w = set_bits(w, 80, 8, 0x80 | (pc << 1))
        return w
    return _imad_like(p, base_r=0x72a5, base_ur=0x72a5, base_imm=0x78a5,
                      byte9=0x00, is_uniform=True)


@register("UIMAD.WIDE")
def enc_uimad_wide(p: ParsedInsn) -> int:
    """UIMAD.WIDE — signed wide multiply-add on uniform registers."""
    if len(p.operands) == 5 and re.fullmatch(r"UP[0-9T]", p.operands[1]):
        pc = _parse_upred_dst(p.operands[1])
        p4 = ParsedInsn(pred=p.pred, mnemonic=p.mnemonic,
                         operands=[p.operands[0]] + p.operands[2:])
        w = _imad_like(p4, base_r=0x72a5, base_ur=0x72a5, base_imm=0x78a5,
                       byte9=0x02, is_uniform=True)
        w = set_bits(w, 80, 8, 0x80 | (pc << 1))
        return w
    return _imad_like(p, base_r=0x72a5, base_ur=0x72a5, base_imm=0x78a5,
                      byte9=0x02, is_uniform=True)


def _imad_wide_x(p: ParsedInsn, *, base_r: int, base_ur: int, base_imm: int,
                 is_uniform: bool, byte9: int) -> int:
    """IMAD.WIDE.X variant: base form plus .X flag (bit 74) and Pcarry-in."""
    if len(p.operands) != 5:
        raise UnsupportedInstruction(f"{p.mnemonic} arity {len(p.operands)}")
    pc_in_idx, _ = _parse_pred_src(p.operands[4])
    p4 = ParsedInsn(pred=p.pred, mnemonic=p.mnemonic, operands=p.operands[:4])
    w = _imad_like(p4, base_r=base_r, base_ur=base_ur, base_imm=base_imm,
                   byte9=byte9, is_uniform=is_uniform)
    ur_form = (w >> 91) & 1
    # Rewrite bits 80..95: 0x000e (PT dst sentinel) + Pc_in index at 87..89.
    w = set_bits(w, 80, 16, 0x000e)
    w = set_bits(w, 87, 3, pc_in_idx)
    if ur_form:
        w = set_bits(w, 91, 1, 1)
    return w


@register("IMAD.WIDE.U32.X")
def enc_imad_wide_u32_x(p: ParsedInsn) -> int:
    return _imad_wide_x(p, base_r=0x7225, base_ur=0x7c25, base_imm=0x7825,
                        is_uniform=False, byte9=0x04)


@register("IMAD.WIDE.X")
def enc_imad_wide_x(p: ParsedInsn) -> int:
    return _imad_wide_x(p, base_r=0x7225, base_ur=0x7c25, base_imm=0x7825,
                        is_uniform=False, byte9=0x06)


@register("UIMAD.WIDE.U32.X")
def enc_uimad_wide_u32_x(p: ParsedInsn) -> int:
    return _imad_wide_x(p, base_r=0x72a5, base_ur=0x72a5, base_imm=0x78a5,
                        is_uniform=True, byte9=0x04)


@register("UIMAD.WIDE.X")
def enc_uimad_wide_x(p: ParsedInsn) -> int:
    return _imad_wide_x(p, base_r=0x72a5, base_ur=0x72a5, base_imm=0x78a5,
                        is_uniform=True, byte9=0x06)


@register("IMAD.X")
def enc_imad_x(p: ParsedInsn) -> int:
    """IMAD.X Rd, Ra, Rb|URb|imm, Rc, Pc_in — carry-in IMAD.

    Layout: like IMAD but byte9 bit 74 set (.X flag), and Pcarry_in idx at
    bits 87..89 (replacing the implicit Ps=PT).
    """
    if len(p.operands) != 5:
        raise UnsupportedInstruction(f"IMAD.X arity {len(p.operands)}")
    pc_in_idx, _ = _parse_pred_src(p.operands[4])
    p4 = ParsedInsn(pred=p.pred, mnemonic=p.mnemonic, operands=p.operands[:4])
    w = _imad_like(p4, base_r=0x7224, base_ur=0x7c24, base_imm=0x7824,
                   byte9=0x06, is_uniform=False)
    # Preserve the UR-form flag (bit 91) that _imad_like may have set.
    ur_form = (w >> 91) & 1
    # Overwrite bits 80..95 with the .X pattern (byte 10 = 0x0e, byte 11 = 0).
    w = set_bits(w, 80, 16, 0x000e)
    w = set_bits(w, 87, 3, pc_in_idx)
    if ur_form:
        w = set_bits(w, 91, 1, 1)
    return w


@register("IMAD.HI.U32")
def enc_imad_hi_u32(p: ParsedInsn) -> int:
    # Optional 5-op form: IMAD.HI.U32 Rd, Pcarry_out, Ra, Rb, Rc.
    if len(p.operands) == 5 and re.fullmatch(r"P[0-9T]", p.operands[1]):
        pc = _parse_pred_dst(p.operands[1])
        p4 = ParsedInsn(pred=p.pred, mnemonic=p.mnemonic,
                         operands=[p.operands[0]] + p.operands[2:])
        w = _imad_like(p4, base_r=0x7227, base_ur=0x7c27, base_imm=0x7827,
                       byte9=0x00, is_uniform=False)
        w = set_bits(w, 80, 8, 0x80 | (pc << 1))
        return w
    return _imad_like(p, base_r=0x7227, base_ur=0x7c27, base_imm=0x7827,
                      byte9=0x00, is_uniform=False)


@register("IMAD.MOV.U32")
def enc_imad_mov_u32(p: ParsedInsn) -> int:
    """IMAD.MOV.U32 Rd, RZ, RZ, {Rc | imm32} — idiomatic move via IMAD."""
    if len(p.operands) != 4:
        raise UnsupportedInstruction(f"IMAD.MOV.U32 arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    ra, _ = parse_reg(p.operands[1])
    rb, _ = parse_reg(p.operands[2])
    src = p.operands[3]
    if ra != 255 or rb != 255:
        raise UnsupportedInstruction("IMAD.MOV.U32 expects Ra=RZ, Rb=RZ")

    w = 0
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, 0xff)                # Ra = RZ
    # Bits 80..95 = 0x078e; bit 91 (UR flag) is not set for reg/imm variants.
    w = set_bits(w, 80, 16, 0x078e)

    if src.startswith("R") or src.replace(".reuse", "").strip() == "RZ":
        rc, rc_reuse = parse_reg(src)
        w = set_bits(w, 0, 16, 0x7224)
        w = set_bits(w, 32, 8, 0xff)            # Rb = RZ
        w = set_bits(w, 64, 8, rc)
        if rc_reuse:
            w = set_bits(w, 124, 1, 1)
    else:
        imm = parse_imm(src) & 0xffffffff
        w = set_bits(w, 0, 16, 0x7424)
        w = set_bits(w, 32, 32, imm)
        w = set_bits(w, 64, 8, 0xff)            # Rc = RZ (unused)

    return _apply_pred(w, p.pred)


@register("IADD3")
def enc_iadd3(p: ParsedInsn) -> int:
    """IADD3 Rd, Pcarry1, Pcarry2, Ra, Rb|URb|imm, Rc

    Fields (empirical):
      bits 0..15  : 0x7210 (R-b), 0x7c10 (UR-b), 0x7810 (imm-b) with PT guard.
      bits 16..23 : Rd
      bits 24..31 : Ra
      bits 32..63 : Rb (8 bits) / URb (8 bits) / imm32
      bits 64..71 : Rc
      bits 72..79 : 0xe0
      bit 80      : 1 (fixed)
      bits 81..83 : Pcarry1 idx (0..7)
      bits 84..87 : 0xf (fixed)
      bits 88..90 : Pcarry2 idx (0..7)
      bit 91      : UR selector (1 for UR-b form)
      bits 92..95 : 0
      bits 96..103: 0
      bit 122     : Ra.reuse
      bit 123     : Rb.reuse
      bit 124     : Rc.reuse
    """
    if len(p.operands) != 6:
        raise UnsupportedInstruction(f"IADD3 arity {len(p.operands)}")
    rd, _ = parse_reg(p.operands[0])
    pc1 = _parse_pred_dst(p.operands[1])
    pc2 = _parse_pred_dst(p.operands[2])
    a_tok = p.operands[3]
    # IADD3 accepts both `-Ra` (arith negate) and `~Ra` (bitwise-not);
    # both set the same negate-flag bit at 72.
    ra_neg = a_tok.startswith("-") or a_tok.startswith("~")
    if ra_neg:
        a_tok = a_tok[1:].strip()
    ra, ra_reuse = parse_reg(a_tok)
    b_tok = p.operands[4]
    c_tok = p.operands[5]
    rc_neg = c_tok.startswith("-") or c_tok.startswith("~")
    if rc_neg:
        c_tok = c_tok[1:].strip()
    rc, rc_reuse = parse_reg(c_tok)

    w = 0
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 24, 8, ra)
    w = set_bits(w, 64, 8, rc)
    # byte 9 bits: bit 72 = -Ra, bit 75 = -Rc.
    w = set_bits(w, 72, 8, 0xe0 | (1 if ra_neg else 0) | (8 if rc_neg else 0))
    w = set_bits(w, 80, 1, 1)
    w = set_bits(w, 81, 3, pc1)
    w = set_bits(w, 84, 3, pc2)
    w = set_bits(w, 87, 1, 1)                   # fixed
    w = set_bits(w, 88, 3, 7)                   # Ps = PT (always in this data)

    # If B starts with `-` or `~`, it's a reg negate/bitwise-not flag.
    # Negative-literal immediates are left intact for parse_imm.
    raw_b = b_tok
    if b_tok.startswith("-") or b_tok.startswith("~"):
        stripped = b_tok[1:].strip()
        if stripped.startswith("UR") or stripped.startswith("R") or stripped in ("URZ", "RZ"):
            b_neg_flag = True
            b_tok = stripped
        else:
            b_neg_flag = False   # negative immediate — leave sign in parse_imm
    else:
        b_neg_flag = False

    if b_tok.startswith("UR") or b_tok.replace(".reuse", "").strip() == "URZ":
        urb, urb_reuse = parse_ureg(b_tok)
        w = set_bits(w, 0, 16, 0x7c10)
        w = set_bits(w, 32, 8, urb)
        w = set_bits(w, 91, 1, 1)
        rb_reuse = urb_reuse
        if b_neg_flag:
            w = set_bits(w, 63, 1, 1)
    elif b_tok.startswith("R") or b_tok.replace(".reuse", "").strip() == "RZ":
        rb, rb_reuse = parse_reg(b_tok)
        w = set_bits(w, 0, 16, 0x7210)
        w = set_bits(w, 32, 8, rb)
        if b_neg_flag:
            w = set_bits(w, 63, 1, 1)
    else:
        # Use the original (possibly signed) token.
        imm = parse_imm(raw_b) & 0xffffffff
        w = set_bits(w, 0, 16, 0x7810)
        w = set_bits(w, 32, 32, imm)
        rb_reuse = False

    if ra_reuse:
        w = set_bits(w, 122, 1, 1)
    if rb_reuse:
        w = set_bits(w, 123, 1, 1)
    if rc_reuse:
        w = set_bits(w, 124, 1, 1)
    return _apply_pred(w, p.pred)


@register("UMOV")
def enc_umov(p: ParsedInsn) -> int:
    """UMOV URd, URb | imm32"""
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"UMOV arity {len(p.operands)}")
    urd, _ = parse_ureg(p.operands[0])
    src = p.operands[1]

    w = 0
    w = set_bits(w, 16, 8, urd)

    if src.startswith("UR") or src.replace(".reuse", "").strip() == "URZ":
        urb, reuse = parse_ureg(src)
        w = set_bits(w, 0, 16, 0x7c82)
        w = set_bits(w, 32, 8, urb)
        w = set_bits(w, 91, 1, 1)
        if reuse:
            raise UnsupportedInstruction("UMOV UR.reuse")
    else:
        imm = parse_imm(src) & 0xffffffff
        w = set_bits(w, 0, 16, 0x7882)
        w = set_bits(w, 32, 32, imm)
    return _apply_pred(w, p.pred)


def _r2ur_impl(p: ParsedInsn, fill: bool) -> int:
    if len(p.operands) != 2:
        raise UnsupportedInstruction(f"R2UR arity {len(p.operands)}")
    urd, _ = parse_ureg(p.operands[0])
    rb, reuse = parse_reg(p.operands[1])
    w = 0
    w = set_bits(w, 0, 16, 0x72ca)          # opcode with PT guard
    w = set_bits(w, 16, 8, urd)
    w = set_bits(w, 24, 8, rb)
    # bits 80..95: 0x000e (non-FILL) or 0x004e (FILL)
    w |= (0x004e if fill else 0x000e) << 80
    if reuse:
        w = set_bits(w, 122, 1, 1)
    return _apply_pred(w, p.pred)


@register("R2UR")
def enc_r2ur(p: ParsedInsn) -> int:
    return _r2ur_impl(p, fill=False)


@register("R2UR.FILL")
def enc_r2ur_fill(p: ParsedInsn) -> int:
    return _r2ur_impl(p, fill=True)


@register("MOV")
def enc_mov(p: ParsedInsn) -> int:
    """MOV Rd, Rb / URb / imm32  [, mask]

    Observed forms (bits 0..15):
      0x7202  Rd, Rb
      0x7c02  Rd, URb
      0x7802  Rd, imm32
    Rd at bits 16..23.
    Src (R or UR) at bits 32..39.
    Imm32 at bits 32..63.
    Write-mask (default 0xf) at bits 72..75 (byte 9 low nibble).
    Source-class selector at bit 83 (set for UR source) — present only in
    the UR form.
    """
    if len(p.operands) not in (2, 3):
        raise UnsupportedInstruction(f"MOV arity {len(p.operands)}")
    dst = p.operands[0]
    src = p.operands[1]
    mask = 0xf
    if len(p.operands) == 3:
        mask = parse_imm(p.operands[2]) & 0xf

    rd, _ = parse_reg(dst)

    w = 0
    w = set_bits(w, 16, 8, rd)
    w = set_bits(w, 72, 4, mask)

    if src.startswith("UR") or src.replace(".reuse", "").strip() == "URZ":
        urb, reuse = parse_ureg(src)
        w = set_bits(w, 0, 16, 0x7c02)
        w = set_bits(w, 32, 8, urb)
        # UR-source selector lives in byte 11 bit 3, i.e. global bit 91.
        w = set_bits(w, 91, 1, 1)
        if reuse:
            raise UnsupportedInstruction("MOV UR.reuse")
    elif src.startswith("R") or src.replace(".reuse", "").strip() == "RZ":
        rb, reuse = parse_reg(src)
        w = set_bits(w, 0, 16, 0x7202)
        w = set_bits(w, 32, 8, rb)
        if reuse:
            # R-source (operand B) reuse flag = byte 15 bit 3 = global bit 123.
            w = set_bits(w, 123, 1, 1)
    else:
        # Immediate form
        imm = parse_imm(src) & 0xffffffff
        w = set_bits(w, 0, 16, 0x7802)
        w = set_bits(w, 32, 32, imm)

    return _apply_pred(w, p.pred)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("text", help="SASS instruction text")
    args = ap.parse_args()
    enc = Encoder()
    bits, mask = enc.encode(args.text)
    print(f"bits : {bits.to_bytes(16, 'little').hex()}")
    print(f"mask : {mask.to_bytes(16, 'little').hex()}")


if __name__ == "__main__":
    main()
