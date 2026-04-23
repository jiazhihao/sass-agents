"""Extract (sass text, 16-byte encoding) pairs from a .sass disassembly + .cubin.

The .sass file is the text produced by `cuobjdump --dump-sass`. It contains
.text.<mangled> sections, each with lines like:

        /*0050*/                   IMAD R0, R3, UR4, R0 ;

The .cubin is an ELF where each matching `.text.<mangled>` section's bytes are
the raw encodings (16 bytes per instruction on Blackwell).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

from elftools.elf.elffile import ELFFile


SECTION_HEADER_RE = re.compile(r"\.section\s+(\.text\.[^,\s]+)")
LABEL_RE = re.compile(r"^(\.text\.[^:]+):\s*$")
INSN_RE = re.compile(r"^\s*/\*([0-9a-f]+)\*/\s+(.*?)\s*;?\s*$")
LOCAL_LABEL_RE = re.compile(r"^(\.L_x_[A-Za-z0-9_]+):\s*$")


@dataclass
class Instruction:
    section: str
    offset: int          # byte offset within the .text section
    text: str            # full text after the /*offset*/ marker, sans trailing `;`
    encoding: bytes      # 16 bytes
    labels: dict[str, int] | None = None   # {label: byte_offset} for this section


def _strip_inline_comment(s: str) -> str:
    # SASS occasionally has inline /*...*/ comments after the insn offset marker.
    return re.sub(r"/\*[^*]*\*/", "", s).strip()


def parse_sass_sections(sass_path: Path) -> tuple[dict[str, list[tuple[int, str]]], dict[str, dict[str, int]]]:
    """Returns ({section_name: [(offset, text), ...]}, {section_name: {label: offset}}).

    Labels are local `.L_x_*` labels that attach to the next instruction
    offset. We track the offset of the last instruction we saw so a label
    line immediately following an instruction resolves to the *next*
    instruction's offset (= last + 16 on Blackwell).
    """
    insns: dict[str, list[tuple[int, str]]] = {}
    labels: dict[str, dict[str, int]] = {}
    current: str | None = None
    in_text_section = False
    last_off = 0
    pending_labels: list[str] = []

    with sass_path.open() as f:
        for line in f:
            m = SECTION_HEADER_RE.search(line)
            if m:
                current = m.group(1)
                in_text_section = True
                insns.setdefault(current, [])
                labels.setdefault(current, {})
                pending_labels.clear()
                last_off = 0
                continue
            if ".section" in line and "text" not in line:
                in_text_section = False
                current = None
                continue
            if not in_text_section or current is None:
                continue
            lm = LOCAL_LABEL_RE.match(line.strip())
            if lm:
                # Label attaches to the *next* instruction. If we've seen any
                # instruction yet, next insn is at last_off + 16.
                target = last_off + 16 if insns[current] else 0
                pending_labels.append(lm.group(1))
                labels[current][lm.group(1)] = target
                continue
            m = INSN_RE.match(line)
            if not m:
                continue
            off = int(m.group(1), 16)
            text = m.group(2).strip()
            if text.startswith(".byte") or text.startswith(".dword") or text.startswith(".align"):
                continue
            text = _strip_inline_comment(text)
            if not text:
                continue
            # Fix up any pending labels that should point at this offset.
            for lb in pending_labels:
                labels[current][lb] = off
            pending_labels.clear()
            insns[current].append((off, text))
            last_off = off
    return insns, labels


def load_text_sections(cubin_path: Path) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    with cubin_path.open("rb") as f:
        elf = ELFFile(f)
        for sec in elf.iter_sections():
            if sec.name.startswith(".text."):
                out[sec.name] = sec.data()
    return out


def iter_instructions(sass_path: Path, cubin_path: Path):
    """Yield Instruction objects correlating sass text with cubin bytes."""
    sass_sections, sass_labels = parse_sass_sections(sass_path)
    cubin_sections = load_text_sections(cubin_path)

    for name, insns in sass_sections.items():
        data = cubin_sections.get(name)
        if data is None:
            continue
        lbls = sass_labels.get(name, {})
        for off, text in insns:
            if off + 16 > len(data):
                continue
            yield Instruction(
                section=name,
                offset=off,
                text=text,
                encoding=bytes(data[off:off + 16]),
                labels=lbls,
            )


def main():
    if len(sys.argv) != 3:
        print("usage: extract.py <sass> <cubin>", file=sys.stderr)
        sys.exit(1)
    sass_path = Path(sys.argv[1])
    cubin_path = Path(sys.argv[2])
    n = 0
    for insn in iter_instructions(sass_path, cubin_path):
        hexenc = insn.encoding.hex()
        print(f"{insn.section}\t+{insn.offset:#06x}\t{hexenc}\t{insn.text}")
        n += 1
    print(f"# {n} instructions", file=sys.stderr)


if __name__ == "__main__":
    main()
