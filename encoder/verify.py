"""Verify that the encoder reproduces each instruction's 128-bit encoding.

For each SASS instruction in the given (.sass, .cubin) pair:
  1. Extract ground-truth 16-byte encoding from the cubin.
  2. Feed the sass text into the encoder -> (bits, struct_mask).
  3. Construct the "reconstructed" image:
        rec = (bits & struct_mask) | (cubin & ~struct_mask)
     i.e. the encoder owns the structural bits; scheduling bits (not in the
     sass text) are carried through from the cubin.
  4. Pass iff rec == cubin, byte for byte.

Reports counts, first few mismatches, and per-mnemonic pass/fail histogram.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from extract import iter_instructions   # noqa: E402
from sass_encoder import (               # noqa: E402
    Encoder,
    UnsupportedInstruction,
    parse,
    Context,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sass")
    ap.add_argument("cubin")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--show-mismatches", type=int, default=5)
    ap.add_argument("--only", default=None, help="Only verify this mnemonic")
    args = ap.parse_args()

    enc = Encoder()

    total = 0
    unsupported: Counter[str] = Counter()
    mismatch: Counter[str] = Counter()
    passed: Counter[str] = Counter()
    shown = 0

    for insn in iter_instructions(Path(args.sass), Path(args.cubin)):
        if args.limit is not None and total >= args.limit:
            break
        total += 1
        try:
            p = parse(insn.text)
        except Exception as e:
            unsupported[f"PARSE-FAIL:{insn.text[:40]}"] += 1
            continue
        if args.only and p.mnemonic != args.only:
            continue
        try:
            ctx = Context(pc=insn.offset, labels=insn.labels)
            bits, mask = enc.encode(insn.text, ctx)
        except UnsupportedInstruction:
            unsupported[p.mnemonic] += 1
            continue
        cubin_int = int.from_bytes(insn.encoding, "little")
        rec = (bits & mask) | (cubin_int & ~mask)
        if rec == cubin_int:
            passed[p.mnemonic] += 1
        else:
            mismatch[p.mnemonic] += 1
            if shown < args.show_mismatches:
                diff = rec ^ cubin_int
                print(f"MISMATCH  {insn.text}")
                print(f"  want : {insn.encoding.hex()}")
                print(f"  got  : {rec.to_bytes(16, 'little').hex()}")
                print(f"  diff : {diff.to_bytes(16, 'little').hex()}")
                print(f"  mask : {mask.to_bytes(16, 'little').hex()}")
                shown += 1

    pass_total = sum(passed.values())
    fail_total = sum(mismatch.values())
    unsup_total = sum(unsupported.values())

    print(f"\n=== SUMMARY ===")
    print(f"total instructions   : {total}")
    print(f"passed               : {pass_total}")
    print(f"mismatches           : {fail_total}")
    print(f"unsupported (no enc) : {unsup_total}")
    if unsupported:
        print("\nTop unsupported mnemonics:")
        for m, n in unsupported.most_common(20):
            print(f"  {n:6d}  {m}")
    if mismatch:
        print("\nMismatched mnemonics:")
        for m, n in mismatch.most_common(20):
            pw = passed.get(m, 0)
            print(f"  {n:6d} mismatch, {pw:6d} pass   {m}")

    # exit code: 0 only if *all* supported instructions passed AND none unsupported
    if fail_total == 0 and unsup_total == 0:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
