# SASS Instructions in `70_blackwell_fp16_gemm.sass`

Target architecture: **`sm_100a` (Blackwell)**. The file contains 153 unique
instruction mnemonics (mnemonic + modifiers). Entries are grouped by
functional category. For each instruction the first line gives the
operand/operation semantic in pseudo-code, the second line is a short
description.

Notation: `Rd`, `Ra`, `Rb`, `Rc` = 32-bit general registers; `URx` = 32-bit
uniform register; `Pn`, `UPn` = predicate / uniform predicate; `[addr]` =
memory; `c[b][o]` = constant bank `b` offset `o`.

---

## 1. Integer arithmetic / logic

### `IADD3`
`Rd = Ra + Rb + Rc`
Three-source 32-bit integer add. Workhorse for address arithmetic and
general integer math on Blackwell.

### `IADD3.X`
`Rd = Ra + Rb + Rc + CC_in`
Extended add that consumes a carry-in predicate; used as the high half of
64-bit additions paired with `IADD3`.

### `UIADD3` / `UIADD3.X`
Uniform-register variants of `IADD3`/`IADD3.X` that operate in the warp-uniform
datapath. Used for per-warp scalars (block/grid dims, base pointers).

### `VIADD`
`Rd = Ra + Rb` (with video/saturation semantics)
Legacy video-integer add used occasionally by the compiler for saturating
additions.

### `IMAD`
`Rd = Ra * Rb + Rc`
32-bit integer multiply-add. Core primitive for index math.

### `IMAD.IADD`
`Rd = Ra + Rb` encoded through the IMAD pipe (`Rc=0`, `Rb=1`).
Compiler convenience form; lets an add issue on the IMAD pipe.

### `IMAD.MOV.U32`
`Rd = Rb` (encoded as `0 * x + Rb`).
Canonical 32-bit immediate/register move on recent SM architectures.

### `IMAD.SHL.U32`
`Rd = Ra << imm` via IMAD.
IMAD-pipe shift-left that keeps encoding compact.

### `IMAD.U32`
Unsigned 32×32→32 multiply-add.

### `IMAD.WIDE.U32`
`{Rd+1,Rd} = Ra * Rb + {Rc+1,Rc}` (unsigned, 64-bit result)
Widening multiply-add used for 64-bit address computation.

### `IMAD.X`
Extended IMAD that incorporates a carry predicate — second half of
wide/64-bit sums.

### `UIMAD` / `UIMAD.WIDE.U32`
Uniform-pipe equivalents of `IMAD` and `IMAD.WIDE.U32`.

### `UFLO.U32`
`Rd = find_leading_one(URa)`
Uniform find-leading-one (count-leading-zeros style primitive) on
unsigned 32-bit value.

### `LOP3.LUT`
`Rd = LUT(Ra, Rb, Rc, imm8)`
Three-input bitwise operation defined by an 8-bit truth-table immediate
(covers AND/OR/XOR and arbitrary ternary logic in one op).

### `ULOP3.LUT` / `UPLOP3.LUT`
Uniform-register and uniform-predicate variants of `LOP3.LUT`.

### `PLOP3.LUT`
Predicate-register LOP3: `Pd = LUT(Pa, Pb, Pc, imm8)`.
Combines three predicates into one with an arbitrary truth table.

### `SHF.L.U32`
`Rd = (Rb:Ra) << imm` — low 32-bit funnel shift left, unsigned.

### `SHF.L.U64.HI`
High-half of a 64-bit left funnel shift.

### `SHF.R.U32.HI` / `SHF.R.S32.HI` / `SHF.R.U64`
Funnel shift right variants (unsigned/signed high half, 64-bit).

### `USHF.L.U32` / `USHF.R.U32.HI` / `USHF.R.S32.HI`
Uniform funnel shift variants.

### `LEA`
`Rd = (Ra << shift) + Rb`
Load-effective-address: scaled index + base. Ubiquitous for pointer math.

### `LEA.HI.X`
High-half LEA with carry; the 64-bit-pointer companion to `LEA`.

### `ULEA` / `ULEA.HI`
Uniform-pipe LEA variants.

### `SEL`
`Rd = P ? Ra : Rb`
Predicated select between two operands.

### `USEL`
Uniform-register SEL.

### `PRMT` / `UPRMT`
Byte-permute: `Rd = permute_bytes(Ra, Rb, selector)` and its uniform
variant. Used for byte/half packing such as building TMA descriptors.

### `P2R`
`Rd = pack_predicates()` — moves predicate bits into a general register.

### `R2P`
`P = unpack_bits(Ra)` — opposite of `P2R`.

### `R2UR` / `R2UR.FILL`
Move value from a thread register into a uniform register. `.FILL`
broadcasts/fills the uniform from a single source.

### `UP2UR`
Uniform-predicate to uniform-register conversion.

---

## 2. Integer compare / set-predicate

### `ISETP.{EQ,NE,GE,GT,LE}.{,U32}.{AND,OR}[.EX]`
`Pd, Pq = (Ra cmp Rb) combine Pc`
Integer set-predicate, signed or unsigned (`.U32`), combining with a
prior predicate using AND/OR. The `.EX` form consumes an input predicate
from an earlier `ISETP` — used to build 64-bit compares.

All variants present in this file:
`ISETP.EQ.U32.AND`, `ISETP.EQ.U32.AND.EX`, `ISETP.GE.AND`,
`ISETP.GE.OR`, `ISETP.GE.U32.AND`, `ISETP.GE.U32.AND.EX`,
`ISETP.GT.OR`, `ISETP.GT.U32.AND`, `ISETP.GT.U32.AND.EX`,
`ISETP.LE.AND`, `ISETP.NE.AND`, `ISETP.NE.U32.AND`,
`ISETP.NE.U32.AND.EX`, `ISETP.NE.U32.OR`.

### `UISETP.{…}` family
Uniform-pipe versions of the above (`UISETP.EQ.U32.AND`,
`UISETP.EQ.U32.OR.EX`, `UISETP.GE.AND`, `UISETP.GE.OR`,
`UISETP.GE.U32.AND`, `UISETP.GT.AND`, `UISETP.GT.U32.AND`,
`UISETP.LT.AND`, `UISETP.LT.U32.AND`, `UISETP.NE.AND`,
`UISETP.NE.U32.AND`, `UISETP.NE.U32.AND.EX`, `UISETP.NE.U32.OR`).

---

## 3. Floating-point arithmetic

### `FADD`
32-bit IEEE add: `Rd = Ra + Rb`.

### `FMUL`
32-bit IEEE multiply: `Rd = Ra * Rb`.

### `FFMA`
Fused multiply-add: `Rd = Ra * Rb + Rc` (single-rounding).

### `HADD2.F32`
`Rd = (half)(Ra_lo + Rb_lo) :: (half)(Ra_hi + Rb_hi)`
Packed half-precision add with F32 internal accumulation; produces two
FP16 lanes packed in one register.

### `HFMA2`
Packed FP16 fused multiply-add: two FP16 FMAs in one 32-bit register.

### `FSETP.{EQ,GT,NEU}.AND`
F32 set-predicate (equal / greater / not-equal-unordered) combined with
an input predicate via AND.

### `HSETP2.NEU.AND`
Packed FP16 set-predicate, NaN-aware not-equal, AND-combined.

### `DSETP.{GEU,LT}.AND`
FP64 set-predicate (unordered-greater-equal or less-than) AND-combined.

---

## 4. Conversions

### `F2F.F64.F32`
Convert FP32 → FP64.

### `F2FP.F16.F32.PACK_AB`
Convert two FP32 values to FP16 and pack into one 32-bit register (A in
lo, B in hi). Used heavily in the epilogue after TMEM reads.

### `F2I.U32.TRUNC.NTZ`
FP32 → unsigned 32-bit integer, truncating toward zero, NaN→0.

### `I2FP.F32.U32`
Unsigned int → FP32 conversion.

---

## 5. Moves / constants

### `MOV`
General register move (result of most trivial aliases).

### `MOV.SPILL`
Move used by the register allocator for spill/reload of local storage.

### `UMOV`
Uniform register move.

### `CS2R` / `CS2R.32`
Read a constant system register into a general register (64-bit or
32-bit form). Used to fetch `%clock`, `%globaltimer`, etc.

### `S2R`
Read a special register (e.g. `SR_TID.X`, `SR_CTAID.X`) into a thread
register.

### `S2UR`
Read a special register into a uniform register (cheaper for
block-uniform values).

---

## 6. Memory — generic / global

### `LDG.E`
`Rd = *global[addr]`
Load from generic/global memory (`.E` = 64-bit address / extended).

### `STG.E`
Store to generic/global memory.

### `STG.E.U16`
16-bit global store.

### `LDC`
`Rd = c[b][o]`
Load from constant memory.

### `LDC.64`
64-bit constant load.

### `LDCU` / `LDCU.64` / `LDCU.128`
Uniform-pipe constant load (32/64/128-bit). Result lands in a uniform
register.

---

## 7. Memory — shared / TMEM / scratch

### `LDS` / `LDS.128` / `LDS.U8`
Shared-memory loads at 32-bit / 128-bit / unsigned-byte granularity.

### `STS`
Shared-memory store.

### `STAS`
Store to the *async* shared-memory path — used to stage data for
subsequent async copy / MMA operations on Hopper/Blackwell.

### `LDTM.`
`Tensor Memory (TMEM) load` — Blackwell tcgen05 tensor-memory read.
Moves MMA accumulator tiles from the per-SM TMEM banks into registers.

### `CCTL.IVALL`
Cache-control: invalidate all L1 lines. Used for coherence barriers
around collective/tensor memory transitions.

---

## 8. Tensor-core / MMA (Blackwell `tcgen05`)

### `UTCHMMA.2CTA`
Half-precision (FP16/BF16) Matrix-Multiply-Accumulate issued from the
**uniform** pipe, targeting a paired-CTA (`.2CTA`) tcgen05 MMA unit.
Reads A/B operands from shared memory descriptors, accumulates into
TMEM. This is the main compute instruction of the kernel.

### `UTMACCTL.PF`
TMA (Tensor Memory Accelerator) control — issue a *prefetch* descriptor
to the TMA engine.

### `UTMACMDFLUSH`
Flush pending TMA commands; completes outstanding async copy issues.

### `UTMALDG.3D.2CTA`
Issue a 3D global→shared TMA load for a paired-CTA group.

### `UTMALDG.3D.MULTICAST.2CTA`
Same as above but **multicast** — one load fans out to multiple CTAs in
the cluster, reducing HBM traffic.

### `UTMASTG.3D`
Issue a 3D shared→global TMA store (epilogue write-back).

### `UTCBAR.2CTA.MULTICAST`
Tensor-core barrier across a 2-CTA group, multicast variant. Waits
until all participating CTAs reach the barrier.

### `UTCATOMSWS.2CTA.FIND_AND_SET.ALIGN`
Tensor-core atomic "find and set with alignment" across 2 CTAs; used by
tcgen05 to allocate/claim TMEM columns in a paired-CTA MMA.

### `UTCATOMSWS.AND`
Tensor-core atomic AND on the scoreboard/word-status register.

### `UVIRTCOUNT.DEALLOC.SMPOOL`
Decrement virtual-count of an SM pool allocation (tcgen05 TMEM column
de-alloc bookkeeping).

### `UCGABAR_ARV`
Arrive on a cluster-group (CGA) barrier.

### `UCGABAR_WAIT`
Wait on a cluster-group barrier.

---

## 9. Thread/warp synchronization

### `BAR.SYNC.DEFER_BLOCKING`
CTA-wide barrier with deferred blocking — thread waits only when it
needs the result, allowing overlap.

### `BAR.ARV`
Arrive on a named CTA barrier (no wait).

### `WARPSYNC.ALL`
Synchronize all active threads in the warp.

### `WARPSYNC.COLLECTIVE`
Warp-sync variant used inside collective operations (epilogue, reduce).

### `MEMBAR.ALL.CTA`
Memory-ordering fence at CTA scope across all memory spaces.

### `FENCE.VIEW.ASYNC.S`
Async-proxy memory fence that publishes prior async writes so
subsequent async reads observe them (shared-memory view).

### `DEPBAR.LE`
Dependency-barrier "less-or-equal" wait — stalls until outstanding
async ops reach a given counter value. Paired with TMA/MMA.

### `NANOSLEEP`
Sleep the thread for ~N nanoseconds (software yield).

### `SYNCS.ARRIVE.TRANS64.A1T0`
Tensor-core `arrive-on-transaction` (64-bit) with pattern `A1T0` —
signals completion of one unit on a transaction barrier.

### `SYNCS.ARRIVE.TRANS64.RED`
Arrive + reduce on a 64-bit transaction barrier.

### `SYNCS.ARRIVE.TRANS64.RED.A1T0`
Combined arrive + reduce + A1T0 pattern.

### `SYNCS.EXCH.64`
Exchange on a 64-bit sync word (atomic swap as a signaling primitive).

### `SYNCS.PHASECHK.TRANS64`
Check the phase bit of a 64-bit transaction barrier (non-blocking).

### `SYNCS.PHASECHK.TRANS64.TRYWAIT`
Phase-check with a try-wait — blocks until the barrier phase changes
(or returns immediately if already there).

---

## 10. Warp-collective / vote / shuffle

### `SHFL.IDX`
Warp shuffle by source lane index.

### `VOTEU.ALL`
Uniform warp vote — all-active predicate.

### `VOTEU.ANY`
Uniform warp vote — any-active predicate.

### `REDUX`
Warp-level reduction (sum/min/max across active lanes, result in each).

### `ELECT`
Elect one active lane in the warp; chosen lane gets `P=true`.

### `UGETNEXTWORKID.BROADCAST`
Producer/consumer work-stealing primitive: obtain next work-id and
broadcast it warp-wide (used by persistent-kernel schedulers).

---

## 11. Atomics

### `ATOMS.OR`
Shared-memory atomic OR.

### `ATOMS.XOR`
Shared-memory atomic XOR.

---

## 12. Control flow

### `BRA`
Unconditional relative branch.

### `BRA.U`
Uniform branch — all threads follow without divergence check.

### `BRA.DIV`
Divergent branch (lanes may take different paths).

### `BSSY`
Begin structured sync — push a reconvergence target on the SSY stack.

### `BSSY.RECONVERGENT`
`BSSY` flavored for reconvergent execution regions.

### `BSYNC`
Synchronize / reconverge at the SSY-stack target.

### `BSYNC.RECONVERGENT`
`BSYNC` for reconvergent regions.

### `CALL.ABS.NOINC`
Absolute-address call, PC not auto-incremented on return.

### `CALL.REL.NOINC`
Relative-address call.

### `RET.REL.NODEC`
Relative return without SSY-stack decrement.

### `EXIT`
Thread exits kernel.

### `NOP`
No operation.

### `BPT.TRAP`
Breakpoint trap — used by the tcgen05 guardrails to halt on misuse
(corresponds to the `__cuda_sm10x_tcgen05_guardrail_trap_*` symbols
seen in the debug frame).

### `LEPC`
Load effective PC into a register (for PC-relative addressing).

### `ENDCOLLECTIVE`
Marks the end of a collective region opened by a warp/cluster
collective primitive.

---

---

# Instruction Arguments (Operands)

The operands that appear in the file fall into the following categories. For
each category the list shows every concrete form observed in
`70_blackwell_fp16_gemm.sass`, plus notes on which instructions accept it.

## A. Thread (per-lane) general registers — `Rn`

- **Range observed:** `R0` .. `R69` (plus `RZ`, the read-as-zero / sink register).
- **Access widths:**
  - 32-bit scalar — default (`R5`, `R17`, …).
  - 64-bit pair — `R2.64`, `R4.64`, `R10.64` (low register of an even pair).
  - 128-bit quad — implicit on `LDS.128` / `LDCU.128`.
- **Sub-lane selectors:** `.H0_H0`, `.H1_H1` (halves of a 32-bit reg) for
  packed FP16; `.B0`/`.B1`/`.B2`/`.B3` for byte operations through `PRMT`.
- **Scheduling hint:** `.reuse` (e.g. `R0.reuse`) — marks the operand for
  the compiler's operand-reuse cache.
- **Used by:** almost every thread-pipe instruction — `IADD3`, `IMAD*`,
  `LEA*`, `SHF*`, `LOP3.LUT`, `SEL`, `PRMT`, `FADD`, `FMUL`, `FFMA`,
  `HFMA2`, `HADD2.F32`, `F2*`, `I2*`, `LDG.E`, `STG.E*`, `LDS*`, `STS`,
  `STAS`, `LDC*`, `LDTM`, `MOV*`, `S2R`, `CS2R*`, `SHFL.IDX`, `P2R`,
  `R2P`, `R2UR*`, `ATOMS.*`, etc.

## B. Warp-uniform registers — `URn`

- **Range observed:** `UR4` .. `UR79` (`UR0`..`UR3` reserved; `URZ` zero-sink).
- **Access widths:** 32-bit, 64-bit pair (`UR6`, `UR8` on `LDCU.64`),
  128-bit (`LDCU.128`).
- **Used by:** the entire uniform datapath — `UIADD3`, `UIMAD*`, `ULEA*`,
  `ULOP3.LUT`, `USHF*`, `USEL`, `UPRMT`, `UMOV`, `UFLO.U32`, `UISETP.*`,
  `LDCU*`, `S2UR`, `UP2UR`, `VOTEU.*`, `REDUX`, `UGETNEXTWORKID.BROADCAST`,
  and all `UT*` / `UCGABAR_*` tcgen05/TMA issue slots (e.g. `desc[URn]` is
  how TMA descriptors and global base pointers are passed).

## C. Predicate registers — `Pn`

- **Range observed:** `P0` .. `P6`, plus `PT` (always-true) and the `!`
  negation prefix (`!P0`, `!PT`).
- **Used for:**
  - instruction guard (leading `@P0` / `@!P2` on any instruction);
  - result of comparisons — `ISETP.*`, `FSETP.*`, `DSETP.*`, `HSETP2.*`
    produce up to two predicate outputs (`Pd, Pq`);
  - carry-in/out on wide arithmetic — `IADD3`, `IMAD.X`, `IADD3.X`,
    `LEA.HI.X`;
  - selector for `SEL`, `PLOP3.LUT`, and the `ELECT`/`VOTEU.*` family.

## D. Uniform predicate registers — `UPn`

- **Range observed:** `UP0` .. `UP6`, plus `UPT` and its negation `!UPT`.
- **Used by:** `UISETP.*`, `UPLOP3.LUT`, `UP2UR`, and as guards on
  uniform-pipe instructions.

## E. Convergence-barrier registers — `Bn`

- **Observed:** `B0`, `B1`, `B6`.
- **Used by:** `BSSY`, `BSSY.RECONVERGENT`, `BSYNC`, `BSYNC.RECONVERGENT`,
  `WARPSYNC.*`. The register names a structured-control-flow SSY stack slot.

## F. Special registers — `SR_*` (read via `S2R` / `S2UR`)

Observed in this kernel:
- `SR_TID.X`, `SR_TID.Y` — thread ID within the CTA.
- `SR_CTAID.X`, `SR_CTAID.Y` — CTA coordinates within the grid.
- `SR_LANEID` — lane index within the warp.
- `SR_C` — clusterID / cluster-context register (used by cluster
  collectives).

## G. Immediates

Integer immediates are written as hex (`0x4`, `0x1`, `0x40`, `0x200`, …);
341 distinct hex immediates appear.
- **Small shift / byte-selector immediates:** `0x0`–`0xff` (used by
  `IMAD.SHL.U32`, `SHF*`, `PRMT`, `LOP3.LUT` truth-table, `LEA` shift
  amount, `PLOP3.LUT`).
- **Address / offset immediates:** larger values (`0x200`, `0x400`,
  `0x600`, `0x800`, …) — mostly shared-memory byte offsets inside
  `[R0+…+0xNNN]` patterns.
- **Float immediates:** decimal literals such as `0.1171875`, `0.5`
  (observed in `FFMA`/`FMUL`).
- **64-bit immediates:** composed via `IMAD.WIDE.U32` + `SHF.L.U64.HI`
  rather than a direct 64-bit literal.

## H. Constant-memory operands — `c[bank][offset]`

- **Banks observed:** `c[0x0][…]` (kernel-parameter / driver bank) and
  `c[0x4][…]` (user/lifted constant bank).
- **Offsets observed:** 39 distinct offsets total. Low bank-0 offsets
  (`0x348`–`0x3d8`) hold kernel-argument words (M, N, K, pointers); the
  `0x88x`–`0xb3x` range holds compiler-lifted constants; bank-4 values
  (`0x40`–`0xc8`) are generated constants.
- **Access widths:** `LDC`/`LDCU` for 32-bit, `LDC.64`/`LDCU.64` for
  64-bit pointers, `LDCU.128` for 128-bit TMA descriptors.
- **Inline constant form:** any arithmetic instruction can take
  `c[0x0][off]` directly as an operand (observed on `IMAD`, `IADD3`,
  `ISETP.*`, etc.).

## I. Memory addressing modes

- **Global / generic** (`LDG.E`, `STG.E*`):
  `desc[URn][Rm.64]` — TMA-style descriptor held in a uniform register
  pair, paired-register 64-bit address in the thread regs.
- **Shared memory** (`LDS*`, `STS`, `STAS`, `LDTM`):
  `[Rn]`, `[Rn+imm]`, `[Rn+URm+imm]` — thread-reg base + optional
  uniform-reg stride + byte immediate. Observed offsets span
  `0x8`..`0x800`.
- **Uniform-pipe loads** (`LDCU*`, `UT*` issue): `[URn]`, `[URn+imm]`.
- **Constant memory**: `c[bank][offset]` (see §H).
- **Branch targets**: symbolic labels like `` `(.L_x_3) `` and
  `` `(.L_x_0) ``.

## J. Descriptor operands — `desc[URn]`

TMA-style memory descriptor handle. The referenced uniform register
(typically a 64-bit pair loaded from bank-0 via `LDCU.64`) holds the
packed tensor descriptor. Used as the first operand of `LDG.E`,
`STG.E*`, and as implicit input to every `UTMA*` / `UTC*` issue. The
second bracket holds the thread-register offset, e.g.
`LDG.E R2, desc[UR6][R2.64]`.

## K. Branch / label operands

- Relative PC labels such as `` `(.L_x_0) ``, `` `(.L_x_3) `` — consumed
  by `BRA`, `BRA.U`, `BRA.DIV`, `BSSY*`, `CALL.REL.NOINC`.
- Absolute targets (symbol + offset) — consumed by `CALL.ABS.NOINC`.

## L. Operand modifiers & qualifiers

Cross-cutting modifiers that decorate operands rather than being operands
themselves:

| Modifier     | Meaning                                              |
|--------------|------------------------------------------------------|
| `.reuse`     | Keep operand in the operand-reuse cache              |
| `.64`        | Operand is a 64-bit register pair                    |
| `.128`       | 128-bit access (load/store width)                    |
| `.H0_H0` / `.H1_H1` | Replicate low/high FP16 half across both lanes |
| `.B0`–`.B3`  | Byte selector (byte permute, sub-word ops)           |
| `!` prefix   | Negated predicate (`!P0`, `!PT`, `!UPT`)             |
| `PT` / `UPT` | Always-true predicate literal                        |
| `RZ` / `URZ` | Zero register / zero-uniform register                |

## M. Quick summary of argument-type usage per instruction class

| Instruction class                 | Typical arg tuple                                  |
|-----------------------------------|----------------------------------------------------|
| Integer arith (`IADD3`, `IMAD*`)  | `Rd[, Pcout], Ra, Rb, Rc[, Pcin]`                  |
| Uniform arith (`UIADD3`, `UIMAD`) | `URd[, UPcout], URa, URb, URc`                     |
| Compare (`ISETP.*`, `FSETP.*`)    | `Pd, Pq, Ra, Rb_or_imm_or_c[][], Pin`              |
| FP MAD (`FFMA`, `HFMA2`)          | `Rd, Ra, Rb, Rc` (optional `-`/`|.|` modifiers)    |
| Move (`MOV`, `IMAD.MOV.U32`)      | `Rd, src` where src = reg/imm/`c[][]`              |
| Global mem (`LDG.E`, `STG.E`)     | `Rd_or_Rs, desc[URn][Rm.64]`                       |
| Shared mem (`LDS`, `STS`)         | `Rd_or_Rs, [Rn+URm+imm]`                           |
| Constant load (`LDC*`, `LDCU*`)   | `Rd_or_URd, c[bank][offset]`                       |
| Tensor-core (`UTCHMMA.2CTA`)      | TMEM dest, A-desc(`URa`), B-desc(`URb`), ctrl imm  |
| TMA (`UTMALDG.3D.*`, `UTMASTG.3D`)| descriptor + coordinates in uniform regs           |
| Sync (`SYNCS.*`, `BAR.*`)         | barrier id (imm or reg), transaction count         |
| Control flow (`BRA`, `BSSY`)      | `Bn` reg (if structured) + label                   |
| Shuffle / vote (`SHFL.IDX`, `VOTEU.*`) | Rd/URd, Ra, lane spec, mask                   |

---

*Generated from the mnemonic set present in
`/home/zhihaoj2/sass_agents/build/70_blackwell_fp16_gemm.sass`
(target `sm_100a`, 123 526 source lines, 153 unique mnemonics).*
