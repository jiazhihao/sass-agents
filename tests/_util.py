"""Shared helpers for the SASS encoder test suite.

Keeps nvdisasm plumbing, encode/compose helpers, and text-normalization in one
place so per-family test modules can focus on the sample lists.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).parent.resolve()
ENCODER_DIR = HERE.parent / "encoder"
if str(ENCODER_DIR) not in sys.path:
    sys.path.insert(0, str(ENCODER_DIR))

from sass_encoder import Encoder, Context  # noqa: E402  — re-exported below

__all__ = [
    "Encoder", "Context",
    "SCHED_MASK", "NVDISASM",
    "compose_with_cubin_sched",
    "compose_with_template_sched",
    "run_nvdisasm",
    "parse_nvdisasm_output",
    "canon_sass",
    "assert_byte_exact",
    "assert_roundtrip",
]


# ---------------------------------------------------------------------------
# Scheduling region (bits 105..121). Encoder does not own these — they pass
# through from the observed cubin (or from a known-good template).
# ---------------------------------------------------------------------------

SCHED_LO, SCHED_HI = 105, 121
SCHED_MASK = ((1 << (SCHED_HI - SCHED_LO + 1)) - 1) << SCHED_LO


# ---------------------------------------------------------------------------
# nvdisasm discovery
# ---------------------------------------------------------------------------

NVDISASM = shutil.which("nvdisasm")


requires_nvdisasm = pytest.mark.skipif(
    NVDISASM is None,
    reason="nvdisasm not on PATH — install CUDA toolkit to run round-trip tests",
)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def compose_with_cubin_sched(encoder: Encoder, sass_text: str,
                               cubin_bytes: bytes) -> bytes:
    """Encode `sass_text`, splice the scheduling region from `cubin_bytes`."""
    bits, mask = encoder.encode(sass_text, Context())
    c = int.from_bytes(cubin_bytes, "little")
    return ((bits & mask) | (c & ~mask)).to_bytes(16, "little")


def compose_with_template_sched(encoder: Encoder, sass_text: str,
                                  template_bytes: bytes) -> bytes:
    """Encode `sass_text`, splice in only the 17-bit scheduling word from
    `template_bytes`. Useful for mutated-operand tests where we don't have a
    cubin ground truth for the mutated form."""
    bits, mask = encoder.encode(sass_text, Context())
    t = int.from_bytes(template_bytes, "little")
    composed = (bits & mask) | (t & SCHED_MASK)
    return composed.to_bytes(16, "little")


# ---------------------------------------------------------------------------
# nvdisasm invocation + output parsing
# ---------------------------------------------------------------------------

def run_nvdisasm(raw: bytes) -> str:
    if NVDISASM is None:
        raise RuntimeError("nvdisasm not on PATH")
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(raw)
        path = f.name
    try:
        proc = subprocess.run(
            [NVDISASM, "--binary", "SM100a", path],
            capture_output=True, text=True, check=False,
        )
    finally:
        Path(path).unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"nvdisasm failed (exit {proc.returncode}):\n"
            f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
        )
    return proc.stdout


_HEADER_RE = re.compile(r"^\s*(?:\.|//|$)")
_PC_PREFIX_RE = re.compile(r"^/\*[0-9a-f]+\*/\s*")


def parse_nvdisasm_output(stdout: str) -> str:
    """Return the first real SASS instruction line from nvdisasm output."""
    for line in stdout.splitlines():
        s = line.strip()
        if not s or _HEADER_RE.match(s):
            continue
        m = _PC_PREFIX_RE.match(s)
        return s[m.end():] if m else s
    raise RuntimeError(f"no instruction line in nvdisasm output:\n{stdout}")


# ---------------------------------------------------------------------------
# Normalisation for round-trip comparison
# ---------------------------------------------------------------------------

_COMMENT_RE = re.compile(r"/\*.*?\*/")
_WS_RE = re.compile(r"\s+")


def canon_sass(s: str) -> str:
    """Normalise a SASS line for comparison: strip comments, `;`, collapse
    whitespace, unify `0x0<digit>` -> `0x<digit>`."""
    s = _COMMENT_RE.sub("", s).strip()
    s = s.rstrip(";").strip()
    s = _WS_RE.sub(" ", s)
    s = re.sub(r"0x0+([0-9a-fA-F])", r"0x\1", s)
    return s


# ---------------------------------------------------------------------------
# High-level assertion helpers
# ---------------------------------------------------------------------------

def assert_byte_exact(encoder: Encoder, sass_text: str, cubin_hex: str) -> None:
    """Byte-exact match of encoder output vs. cubin (scheduling OR'd in)."""
    cubin = bytes.fromhex(cubin_hex)
    composed = compose_with_cubin_sched(encoder, sass_text, cubin)
    assert composed == cubin, (
        f"byte mismatch for {sass_text!r}\n"
        f"  want: {cubin.hex()}\n"
        f"  got : {composed.hex()}\n"
        f"  diff: {(int.from_bytes(cubin,'little') ^ int.from_bytes(composed,'little')).to_bytes(16,'little').hex()}"
    )


def assert_roundtrip(encoder: Encoder, sass_text: str,
                      cubin_or_template_hex: str) -> None:
    """nvdisasm(encoder(sass_text)) must match sass_text."""
    template = bytes.fromhex(cubin_or_template_hex)
    composed = compose_with_template_sched(encoder, sass_text, template)
    disasm = parse_nvdisasm_output(run_nvdisasm(composed))
    assert canon_sass(disasm) == canon_sass(sass_text), (
        f"round-trip mismatch\n"
        f"  original : {canon_sass(sass_text)!r}\n"
        f"  disasm'd : {canon_sass(disasm)!r}\n"
        f"  bytes    : {composed.hex()}"
    )
