# SASS Instruction Reference — Blackwell (sm_100a)

Scope: every unique SASS mnemonic observed in the 47 `.sass` files under `cutlass/build/sass/` (122 k – 200 k line cuobjdump dumps from CUTLASS Blackwell samples). Extraction counted **135 base mnemonics** and **422 mnemonic+suffix variants**. Counts shown are summed across all 47 files.

SASS is NVIDIA’s post-assembly (not officially documented). The semantics below are synthesized from: Volta–Blackwell ISA patterns visible in cuobjdump, public disassemblers (nv-cuda-disasm, NervanaSystems/maxas, turingas, cuasmrl), PTX–SASS correspondence, and NVIDIA PTX ISA / CUDA programming guide. Blackwell (sm_100) adds the Tensor Memory (TMEM) + 5th-gen Tensor Core (UMMA) + TMA + convergent-barrier families; those are called out individually below.

## 0. Notation

```
  @P0 / @!P0           regular-predicate guard (per thread)
  @UP0 / @!UP0         uniform-predicate guard (per warp; sourced from URn)
  @PT / @UPT           always-true predicate (instruction always executes)

  Rd,Ra,Rb,Rc          32-bit vector registers (per thread).  RZ = constant 0.
  Rd.64 / Rd.128       64-/128-bit register group (consecutive Rd,Rd+1,...)
  URd,URa,...          Uniform registers (one copy per warp). URZ = const 0.
  Pd, UPd              Predicate / uniform predicate destinations
  c[0x0][off]          Constant-bank load (bank 0, byte offset)
  desc[URx]            TMA / tensor-memory descriptor register
  gdesc[URx]           UMMA shared-memory matrix descriptor
  idesc[URx]           UMMA instruction descriptor
  tmem[URx + off]      Tensor-memory address
  [Rbase + off]        Generic memory address (global/local/shared depending on op)

Common suffixes
  .U16/.U32/.U64/.S16/.S32/.S64/.F16/.F32/.F64/.BF16/.E4M3/.E5M2/.E2M1/.E8
                       data type / element width
  .FTZ                 flush subnormals to zero (FP)
  .NAN                 NaN-propagating min/max
  .SAT / .SATFINITE    clamp result to representable / finite range
  .RN/.RM/.RP/.RZ      round nearest-even / toward −∞ / +∞ / toward-zero
  .X / .AND.EX         use carry-in / extended predicate chain
  .HI / .LO            upper / lower result half
  .WIDE                full-width (e.g. 32×32→64 multiply)
  .LUT                 4-bit lookup-table immediate
  .E                   "extended" 64-bit address form (global memory)
  .STRONG.{GPU,SYS}    memory-ordering scope for atomics/loads
  .LTC128B / .BYPASS   L2 cache class / bypass hint (LDGSTS)
  .RECONVERGENT / .RELIABLE
                       barrier-style behaviour for BSSY/BSYNC/BREAK
  .2CTA / .MULTICAST   cluster-scoped TMA / UMMA (Blackwell CTA-pair)
```

Operand order across the ISA: **destination first, then sources left-to-right** (e.g. `IMAD.WIDE Rd, Ra, Rb, Rc` computes `Rd = Ra*Rb + Rc`).

---

## 1. Integer ALU

### IADD3 / IADD3.X   ( `20 689` / `1 173` )
`IADD3 Rd, Ra, Rb, Rc` — 3-operand add: `Rd = Ra + Rb + Rc` (32-bit).
`.X` variant takes carry-in from a predicate chain (`Rd = Ra + Rb + Rc + CarryIn`, produces new carry in `Pd`).
Frequently paired with `IMAD` or itself to synthesize 64-bit adds.

### UIADD3 / UIADD3.X / UIADD3.64   ( `39 204` / `2 936` / `10` )
Uniform-register version of IADD3. `.64` produces a 64-bit result into a UR pair.

### IMAD (and variants)   (`10 331` + many suffixes)
`IMAD Rd, Ra, Rb, Rc` — integer multiply-add: `Rd = Ra*Rb + Rc` (low 32 bits).
Suffix matrix seen:
| Variant | Semantics |
|---------|-----------|
| `IMAD` | 32-bit signed mul-add, low half |
| `IMAD.U32` | same, unsigned operands |
| `IMAD.X` | mul-add with carry-in (64-bit synthesis helper) |
| `IMAD.HI.U32` | returns high 32 bits of `Ra*Rb + Rc` |
| `IMAD.WIDE` / `IMAD.WIDE.U32` | 32×32 → 64-bit mul then +Rc pair → 64-bit Rd pair (`Rd,Rd+1`) |
| `IMAD.WIDE.U32.X` / `UIMAD.WIDE.U32.X` | wide multiply-add with carry-in |
| `IMAD.MOV` / `IMAD.MOV.U32` | idiomatic 32-bit register move (`Ra=RZ, Rb=RZ` or `Rb=imm, Rc=0`) |
| `IMAD.SHL.U32` | synthesizes shift-left-by-constant via `Ra*imm + 0` |
| `IMAD.IADD` | degenerate mul-accumulate used as an add pipeline variant |

`UIMAD`, `UIMAD.WIDE`, `UIMAD.WIDE.U32`, `UIMAD.WIDE.U32.X` are the uniform-register forms.

### IABS   (`2 151`)
`IABS Rd, Ra` — signed absolute value.

### BREV   (`32`)
`BREV Rd, Ra` — bit-reverse 32-bit operand.

### FLO.U32.SH / UFLO.U32   (`32` / `151`)
Find leading-one (count of leading zeros minus bias). `.SH` places result as a shift count; outputs `0xffffffff` on `Ra == 0`.

### POPC / UPOPC   (`9` / `1`)
Population count — number of set bits in `Ra`.

### SEL / USEL / FSEL   (`6 295` / `7 194` / `11 189`)
Conditional select: `Rd = Pp ? Ra : Rb`. `FSEL` forwards the FP flags but the data is a raw bit-select (identical to SEL on GPU registers). `USEL` operates on URs.

### PRMT / UPRMT   (`5 447` / `1 345`)
`PRMT Rd, Ra, selector, Rb` — byte-permute: each output byte comes from a source byte of the concatenation `{Rb:Ra}` selected by a 4-bit nibble in `selector`. Used for shuffle-style reformatting and sign/zero extension.

### R2P / P2R   (`429` / `411`)
`R2P Pd, Rd_mask, Ra, Rb` — set predicate register file from GPR (unpack bits as predicates).
`P2R Rd, Pmask, Ra, imm` — pack predicate bits into a GPR.

### R2UR / R2UR.FILL / R2UR.OR / R2UR.BROADCAST   (`51 309` / `3 611` / `38` / `137`)
Register → Uniform Register transfer. Plain `R2UR` requires the vector operand to be identical across the warp (otherwise undefined). `.BROADCAST` takes a specific lane. `.OR` reduces the 32 lane values with OR. `.FILL` fills the UR regardless (no divergence check).

### S2R / S2UR   (`2 207` / `4 723`)
Special-register read. `S2R Rd, SRx` reads things like `SR_TID`, `SR_CTAID`, `SR_CLOCKLO`, `SR_LANEID`, `SR_NSMID`, `SR_TMEM_DEALLOC` … into a GPR. `S2UR` reads into a UR when the SR is warp-uniform.

### UP2UR / UMOV / MOV / MOV.SPILL   (`284` / `44 882` / `26 503` / `3 611`)
Register moves. `MOV Rd, Ra` copies a 32-bit value. `MOV.SPILL` is the compiler-emitted form when spilling to local memory (paired with `LDL`/`STL`). `UMOV` moves to a UR. `UP2UR UPd, Ra` packs a predicate into a UR bit.

### LEA / LEA.HI / LEA.HI.X / LEA.HI.SX32 / LEA.HI.X.SX32   (`4 748` / `373` / `2 004` / `6` / `17`)
`LEA Rd, Ra, Rb, imm5` — effective-address arithmetic: `Rd = (Ra << imm5) + Rb`. `.HI` returns the upper 32 bits of the 64-bit sum (used with a companion `LEA` to build 64-bit pointers). `.X` consumes a carry-in; `.SX32` sign-extends `Ra` to 64 bits before shifting. `ULEA`/`ULEA.HI`/`ULEA.HI.X`/`ULEA.HI.SX32`/`ULEA.HI.X.SX32` (`8 314` / `391` / `55` / `21` / `17`) are the uniform variants.

### SHF.{L,R}.{U32,S32,U64,S64}.{HI,LO}   and USHF mirror   (`SHF.R.U64 9 002`, `SHF.L.U32 8 581`, `SHF.R.U32.HI 3 512`, `SHF.R.S32.HI 1 334`, `SHF.L.U64.HI 141`, `SHF.R.S64 6`; `USHF.*` 4 901 / 2 192 / 782 / 275 / 15 / 2)
Funnel shift: `SHF.L.U64 Rd, Ra, imm, Rb` computes `(Rb:Ra) << imm` and returns the specified half. `.HI` / `.LO` selects half. Signed (`S32`/`S64`) sign-extends.

### LOP3.LUT / ULOP3.LUT / PLOP3.LUT / UPLOP3.LUT   (`41 556` / `4 859` / `3 124` / `2 183`)
`LOP3.LUT Rd, Ra, Rb, Rc, imm8, !Pd` — 3-input bitwise logic with an arbitrary truth table encoded in `imm8` (Karnaugh index of `(a,b,c)`). `PLOP3.LUT` does the same for predicates. `U*` variants are uniform.

### SHFL.{IDX,UP,BFLY}   (`1 665` / `640` / `290`)
Warp shuffle: `SHFL.IDX Pp, Rd, Ra, Rb, Rc` — lane `Rb` of `Ra`, with membership mask `Rc`, predicate out `Pp`. `.UP`/`.BFLY` use the CUDA shuffle modes (`__shfl_up_sync`, `__shfl_xor_sync`).

### VIADD / VIADDMNMX / VIADDMNMX.U32 / VIMNMX / VIMNMX.U32 / VIMNMX3   (`10 899` / `25` / `124` / `247` / `18` / `7`)
Video/integer SIMD ops. `VIADD` is a variant of integer add with byte/half-byte lane support that the compiler uses as a pipeline-balancing alternative to `IADD3`. `VIMNMX` is integer min/max; `VIADDMNMX` fuses add-then-min/max; `VIMNMX3` is 3-way.

### REDUX / CREDUX.MAX.F32.NAN / CREDUX.MIN / CREDUX.MAX.S32   (`270` / `16` / `2` / `2`)
Warp-wide reduction across lanes. `REDUX Rd, Ra, op` fuses a shuffle-reduce tree. `CREDUX.*` are the cluster/CGA-scoped reductions added on Blackwell.

---

## 2. Integer predicates

### ISETP.{cmp}[.U32].{AND,OR,XOR}[.EX]
Integer compare-and-set-predicate. Form: `ISETP.GE.AND Pd, Pq, Ra, Rb, Pr` which computes `Pd = (Ra cmp Rb) AND Pr`; `Pq` receives the negated result. Comparisons: `EQ, NE, LT, LE, GT, GE`. `.U32` = unsigned. `.EX` consumes another predicate (extended-precision chaining for 64-bit compares).

Observed: `ISETP.GE.AND 9 073`, `ISETP.NE.U32.AND 6 238`, `ISETP.GE.U32.AND 4 850`, `ISETP.LE.AND 3 612`, `ISETP.NE.AND 1 927`, `ISETP.GE.OR 1 623`, `ISETP.GT.U32.AND 1 607`, `ISETP.EQ.AND 960`, `ISETP.GT.AND 612`, `ISETP.LT.AND 550`, `ISETP.NE.U32.OR 503`, `ISETP.EQ.U32.AND 372`, `ISETP.LT.OR 309`, `ISETP.GE.U32.AND.EX 308`, `ISETP.NE.U32.AND.EX 778`, `ISETP.EQ.U32.AND.EX 26`, `ISETP.GE.AND.EX 1`, `ISETP.GT.AND.EX 1`, `ISETP.GT.U32.OR 82`, `ISETP.LE.U32.AND 147`, `ISETP.LT.U32.AND 12`, `ISETP.LE.OR 47`, `ISETP.LT.U32.OR 5`, `ISETP.LT.U32.AND.EX 4` `ISETP.NE.AND.EX 18`, `ISETP.EQ.U32.OR 13`, `ISETP.EQ.U32.OR.EX 3`.

`UISETP.*` (`6 284` NE.U32.AND plus 27 other variants) is the uniform counterpart: uniform registers in, **uniform** predicates out.

---

## 3. Floating-point ALU

### Scalar FP32
- **FADD / FADD.FTZ / FADD.RZ** (`3 297` / `321` / `41`) — `Rd = Ra + Rb`.
- **FMUL / FMUL.FTZ / FMUL.RZ** (`43 136` / `135` / `26`) — `Rd = Ra * Rb`.
- **FFMA / FFMA.RM / FFMA.RP / FFMA.RZ / FFMA.SAT** (`18 747` / `386` / `112` / `69` / `274`) — `Rd = Ra*Rb + Rc`, rounding and saturation controlled by suffix.
- **FMNMX / FMNMX.NAN / FMNMX3 / FMNMX3.NAN** (`136` / `680` / `6 202` / `96`) — FP min/max; `3` takes 3 inputs; `.NAN` propagates NaNs.
- **FSEL** (`11 189`) — FP conditional select (by predicate).
- **FCHK** (`274`) — FP validity check; sets `Pd` if `Ra` is denormal/NaN/inf (used in software divide).
- **FSETP.{cmp}[.FTZ].{AND,OR,XOR}** (`FSETP.GEU.AND 15 116`, `FSETP.NEU.AND 827`, `FSETP.NEU.FTZ.AND 346`, `FSETP.GEU.FTZ.AND 27`, `FSETP.GTU.FTZ.AND 165`, `FSETP.EQ.AND 125`, `FSETP.GT.AND 70`, `FSETP.NE.AND 6`) — FP compare→predicate, `U` variants are unordered (treat NaN as "true" for ≠, "false" for ≤/≥).

### Paired FP32 ("F32x2" and FP16x2) — Blackwell SIMD
- **FMUL2** (`14 512`), **FADD2** (`6 820`), **FFMA2** (`6 752`) — packed-FP32-pair multiply / add / fused-multiply-add, operating on a register pair `{R, R+1}` as two lanes. Enables 2-wide FP32 throughput per instruction.
- **HADD2 / HADD2.F32 / HADD2.BF16_V2** (`12` / `11 980` / `194`) — packed half-precision add. `HADD2.F32` is `fp16×2 + fp16×2 → fp32×2` accumulation; `.BF16_V2` operates on packed bf16×2.
- **HMUL2 / HMUL2.BF16_V2** (`486` / `66`) — packed half mul.
- **HFMA2 / HFMA2.BF16_V2** (`5 357` / `69`) — packed half FMA.
- **HSETP2.{NEU,NE,GEU}.AND[.BF16_V2]** (`33 / 8 / 4 / 3`) — packed half compare into two predicates.

### MUFU (Multi-function unit)   (`MUFU.EX2 14 327`, `MUFU.RCP 1 795`, `MUFU.RSQ 123`, `MUFU.SIN 8`, `MUFU.COS 8`, `MUFU.RCP64H 66`, `MUFU.RSQ64H 44`)
Transcendental approximations: `EX2` (2^x), `RCP` (1/x), `RSQ` (rsqrt), `SIN`, `COS`. `RCP64H` / `RSQ64H` are 64-bit refinement helpers.

### FP64
- **DADD / DMUL / DFMA / DFMA.RM / DFMA.RP** (`98` / `220` / `616` / `22` / `22`) — scalar FP64 add / mul / fma. Rounding suffix when explicit.
- **DSETP.{LT,LE,GT,GE,GTU,GEU,EQ,NE,MAX}.AND** (`DSETP.MAX.AND 87`, `DSETP.LT.AND 32`, `DSETP.GEU.AND 32`, `DSETP.NE.AND 33`, `DSETP.GTU.AND 22`, `DSETP.GT.AND 22`) — FP64 compare into predicate.

---

## 4. Type conversions

- **F2FP.*** (e.g. `F2FP.F16.F32.PACK_AB 17 707`, `F2FP.F16.E4M3.UNPACK_B 1 565`, `F2FP.BF16.F32.PACK_AB 1 139`, `F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C 1 221`, `F2FP.SATFINITE.E5M2.F32.PACK_AB_MERGE_C 458`, `F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C 96`, `F2FP.SATFINITE.BF16.F32.PACK_AB 36`, `F2FP.F16.E5M2.UNPACK_B 295`, `F2FP.BF16.E8.UNPACK_B 4`, `F2FP.SATFINITE.E8.F32.PACK_AB_MERGE_C.RP 4`).
  General form: `F2FP.<out>.<in>.{PACK_AB|UNPACK_B|PACK_AB_MERGE_C}[.SATFINITE][.<round>] Rd, Ra, Rb [,Rc]`.
  Converts between FP types with narrow float formats for MMA. `PACK_AB` takes two FP32 inputs and packs a pair of narrow-float outputs into `Rd`; `PACK_AB_MERGE_C` additionally merges into an existing packed `Rc`. `UNPACK_B` splits a packed source into an FP32 pair. Supports the Blackwell narrow formats `E4M3`, `E5M2`, `E2M1`, `E8` (microscaling).
- **F2F.F16.F32 / F2F.F32.F64 / F2F.F64.F32** (`120` / `22` / `75`) — scalar float→float conversions.
- **F2I.FTZ.U32.TRUNC.NTZ / F2I.U32.TRUNC.NTZ / F2I.S64.TRUNC / F2I.U64.TRUNC / F2I.TRUNC.NTZ / F2I.NTZ** (`1 161` / `120` / `41` / `26` / `3` / `4`) — float→integer. `.TRUNC` = round-toward-zero, `.NTZ` = don’t-flush on subnormals.
- **I2F.RP / I2F.U32.RP / I2F.S64 / I2F.U64.RP / I2F.F64 / I2F.U16** (`905` / `256` / `41` / `26` / `22` / `18`) — integer→float conversions.
- **I2FP.F32.S32 / I2FP.F32.U32 / I2FP.F32.U32.RZ** (`186` / `183` / `18`) — integer→packed-float (FP32) with optional rounding mode.

---

## 5. Control flow

### BRA / BRA.U / BRA.U.ANY / BRA.DIV / BRX / BRXU   (`32 998` / `4 845` / `115` / `26` / `112` / `12`)
Direct / uniform / divergent / indexed branch. `BRA.U` = warp-uniform (taken or not by all lanes). `.DIV` = may diverge across the warp. `BRX Rt` = indirect via register. `BRXU` = uniform indirect.

### CALL.ABS.NOINC / CALL.REL.NOINC   (`3 476` / `876`)
Absolute / PC-relative subroutine call that does **not** auto-push the return PC (pairs with `LEPC` that pre-loads it into a register).

### LEPC   (`3 476`)
`LEPC Rd` — load effective PC into Rd; used as a return-address anchor prior to `CALL.*.NOINC`.

### RET.REL.NODEC   (`632`)
Return from subroutine, PC-relative form, no stack decrement (matches the NOINC call convention).

### EXIT / PREEXIT   (`1 884` / `52`)
`EXIT` — terminate thread; `PREEXIT` — hint that a warp is about to exit (lets the SM reclaim resources early on Blackwell).

### NOP   (`8 487`)
No-operation (`NOP ;` — pipeline filler, also emitted to align control flow).

### YIELD   (`112`)
Warp scheduler yield hint (release issue slot).

### BPT.TRAP   (`510`)
Breakpoint/trap (raises exception).

### Barriers and sync primitives
- **BAR.SYNC.DEFER_BLOCKING** (`1 195`) — CTA-scoped barrier (with Blackwell deferred-blocking semantics).
- **BAR.ARV** (`77`) — barrier arrive (non-blocking).
- **BSSY / BSYNC** (`629` / `629`), **BSSY.RECONVERGENT / BSYNC.RECONVERGENT** (`3 997` / `3 997`), **BSSY.RELIABLE / BSYNC.RELIABLE** (`28` / `28`), **BREAK / BREAK.RELIABLE** (`6` / `28`) — Volta+ SIMT-convergence primitives. `BSSY Bn, label` installs a sync barrier; matching `BSYNC Bn` rejoins lanes; `BREAK` exits a sync scope. `.RECONVERGENT`/`.RELIABLE` variants tighten the guarantees (hardware-managed convergence barrier registers).
- **WARPSYNC.ALL / WARPSYNC.COLLECTIVE** (`539` / `30`) — force warp re-convergence on a lane mask.
- **ELECT** (`1 165`) — `ELECT Pd, Rd` selects one active lane of the warp; sets `Pd` for the winner and returns its lane id in `Rd` (used to serialize work like TMA issue).
- **VOTE.ANY / VOTEU.ANY / VOTEU.ALL** (`188` / `473` / `248`) — warp vote returning a uniform predicate.
- **DEPBAR / DEPBAR.LE** (`76` / `746`) — dependency barrier: wait until outstanding async transactions (loads, LDGSTS, TMAs) reach a specified counter value.
- **LDGDEPBAR** (`6`) — install a dependency counter for subsequent `LDG`s.
- **MEMBAR.ALL.CTA / MEMBAR.ALL.GPU / MEMBAR.SC.GPU / MEMBAR.SC.SYS** (`877` / `5` / `16` / `1`) — memory fences at CTA / device / system scope; `SC` = sequentially consistent.
- **FENCE.VIEW.ASYNC.S / FENCE.VIEW.ASYNC.T** (`1 958` / `239`) — async-view fence: makes prior writes from async-copy / TMA visible to subsequent shared-memory (`S`) or tensor-memory (`T`) reads.
- **ERRBAR / CGAERRBAR** (`22` / `22`) — error barrier at SM / cluster scope.
- **ENDCOLLECTIVE** (`30`) — closes a collective region begun by `WARPSYNC.COLLECTIVE`.
- **ACQBULK** (`166`) — bulk-acquire fence for async-commit groups.
- **NANOSLEEP / NANOSLEEP.SYNCS** (`140` / `6 500`) — sleep a nanosecond count. `.SYNCS` combines sleep with the new SYNCS queue wait.
- **PLOP3 / UPLOP3.LUT** (listed above) — predicate logic.

### SYNCS.* (Blackwell signal/wait queue)
`SYNCS` is the new CTA-pair signalling mechanism backing `tcgen05` / `cp.async.mbarrier`. Observed variants:

| Mnemonic | Semantics |
|---|---|
| `SYNCS.PHASECHK.TRANS64.TRYWAIT` (`13 524`) | 64-bit transaction-phase check; attempt to consume a ready phase without blocking and return a predicate. |
| `SYNCS.PHASECHK.TRANS64` (`6 583`) | Blocking phase check. |
| `SYNCS.EXCH.64` (`4 818`) | Atomic exchange on a mbarrier slot (32/64-bit token). |
| `SYNCS.ARRIVE.TRANS64` (`1 392`) | Arrive on a transaction-counting barrier. |
| `SYNCS.ARRIVE.TRANS64.RED.A1T0` (`1 902`) | Arrive with reduction, "arrive=1, thread=0" participation encoding. |
| `SYNCS.ARRIVE.TRANS64.A1T0` (`821`) | Same without reduction. |
| `SYNCS.ARRIVE.TRANS64.RED / .RED.A0TR / .RED.A0T1` (`69 / 303 / 158`) | Reduction arrivals with other participation encodings (all-arrive, arrive-0/txn-received, …). |

---

## 6. Memory — global / local / shared / constant

### Global (LD/ST across device memory)
- **LDG.E / LDG.E.U8 / LDG.E.U16 / LDG.E.64 / LDG.E.128 / LDG.E.STRONG.GPU / LDG.E.STRONG.SYS** (`18 842` / `1 512` / `6 130` / `190` / `30` / `4` / `4`) — global load. `.E` = 64-bit address; size suffix = payload; `.STRONG.{GPU,SYS}` = acquire-scope load.
- **LD.E / LD.E.STRONG.SYS** (`72` / `1`) — generic-space load with 64-bit address.
- **STG.E / STG.E.U8 / STG.E.U16 / STG.E.64 / STG.E.128** (`2 339` / `587` / `1 246` / `46` / `367`) — global stores.
- **ST.E / ST.E.U16 / ST.E.128** (`6 144` / `16` / `1 968`) — generic stores.
- **LDGSTS.E.LTC128B / LDGSTS.E.LTC128B.128 / LDGSTS.E.BYPASS.LTC128B.128** (`297` / `98` / `468`) — async copy global→shared (ampere `cp.async`), `.BYPASS` skips L1.

### Local memory (stack / spill)
- **LDL / LDL.64 / LDL.LU / LDL.LU.64** (`598` / `428` / `532` / `730`) — local loads; `.LU` = last-use (invalidate in cache). 
- **STL / STL.64 / STL.128 / STL.S16** (`3 268` / `1 243` / `22` / `38`) — local stores.
- **STAS** (`56`) — store to stack, small-object form.

### Shared memory
- **LDS / LDS.U8 / LDS.U16 / LDS.64 / LDS.128** (`4 319` / `1 721` / `243` / `201` / `1 802`) — shared load.
- **STS / STS.U8 / STS.U16 / STS.64 / STS.128** (`1 934` / `1 712` / `856` / `262` / `1 520`) — shared store.

### Constant memory
- **LDC / LDC.64 / LDC.U8 / LDC.U16** (`3 971` / `8 712` / `2` / `92`) — constant-bank load (`c[bank][offset]`).
- **LDCU / LDCU.64 / LDCU.128 / LDCU.U8 / LDCU.U16** (`6 565` / `8 293` / `733` / `30` / `6`) — uniform-register-destined constant load (reads into UR).

### Cache control
- **CCTL.E.C.LDCU.IV.DEEP / CCTL.E.PF2 / CCTL.IVALL** (`76` / `64` / `69`) — explicit cache-control (invalidate, prefetch, bypass).

### Atomics
- **ATOMS.OR / ATOMS.AND / ATOMS.XOR / ATOMS.CAST.SPIN** (`338` / `274` / `64` / `128`) — shared-memory atomics (logic + the spin-lock CAST helper used for mutex loops).
- **ATOMG.E.ADD.F32.FTZ.RN.STRONG.GPU** (`132`) — global FP32 atomic add with ordering.
- **ATOMG.E.ADD.STRONG.GPU / ATOMG.E.EXCH.STRONG.GPU / ATOMG.E.MAX.S32.STRONG.GPU / ATOMG.E.CAS.STRONG.GPU / ATOMG.E.CAS.STRONG.SYS / ATOMG.E.CAS.64.STRONG.SYS** (`8` / `76` / `4` / `1` / `8` / `11`) — global integer atomics with scope.
- **ATOM.E.ADD.STRONG.GPU** (`1`) — generic-space atomic add.
- **REDG.E.ADD.{,F64.RN,S32}.STRONG.GPU / REDG.E.MAX.S32.STRONG.GPU / REDG.E.MIN.STRONG.GPU** (`8 / 11 / 1 / 2 / 2`) — fire-and-forget reductions on global memory (no return value).

---

## 7. Tensor Memory (TMEM) — Blackwell 5th-gen Tensor Core / `tcgen05`

Tensor memory is a new SM-local SRAM partition holding MMA accumulator tiles. It is addressed through `tmem[URx + offset]` with UR bases supplied by the allocator.

- **USETMAXREG.TRY_ALLOC.CTAPOOL / USETMAXREG.DEALLOC.CTAPOOL** (`132` / `424`) — allocate / release a TMEM region for the CTA.
- **UVIRTCOUNT.DEALLOC.SMPOOL** (`48`) — decrement virtual tile counter after reclaim.
- **LDTM.x2 / x8 / x16 / x32 / x64 / x128** (`248` / `54` / `1 948` / `640` / `26` / `5`) — load N×32-bit elements per lane from tmem into vector GPRs.
- **LDTM.16dp256bit.x2 / .x4 / .x8 / .x16** (`6` / `92` / `36` / `2`) — 16-datapath, 256-bit-wide tmem loads (column-partitioned).
- **LDTM.16dp32bit_t0_t15.x16 / t16_t31.x16 / x32 variants** (`36` / `36` / `16` / `16`) — partitioned tmem loads for specific lane subsets (`t0..t15`, `t16..t31`).
- **STTM.x2 / x8 / x16 / x32 / 16dp128bit.x16** (`184` / `118` / `776` / `244` / `4`) — tmem stores (write accumulator back).
- **UTCBAR / UTCBAR.2CTA.MULTICAST / UTCBAR.MULTICAST** (`1 359` / `203` / `10`) — tensor-core barrier (wait for outstanding UMMA/UTMA); `.2CTA.MULTICAST` spans a CTA pair.
- **UTCHMMA / UTCHMMA.2CTA** (`5 116` / `207`) — 5th-gen half-precision MMA: `UTCHMMA gdesc[A], gdesc[B], tmem[C_in], tmem[C_out], idesc[I], UPp` accumulates `C_out += A·B` on fp16/bf16 inputs using an instruction descriptor in `idesc[]`. `.2CTA` = CTA-pair form consuming multicast data.
- **UTCQMMA / UTCQMMA.2CTA** (`186` / `156`) — quarter-precision (fp8 / mxfp4 / nvfp4) MMA variants.
- **UTCOMMA.4X / UTCOMMA.2CTA.4X** (`24` / `18`) — "output-MMA" tile-merge operation for scale-factored MMA.
- **UTCATOMSWS.FIND_AND_SET.ALIGN / UTCATOMSWS.2CTA.FIND_AND_SET.ALIGN / UTCATOMSWS.AND** (`210` / `64` / `137`) — tensor-core atomic with "find-and-set" semantics (scheduler-ticket allocation for async MMA issue).
- **UTCCP.T.S.4x32dp128bit / UTCCP.T.S.2CTA.4x32dp128bit / UTCCP.T.S.2CTA.128dp128bit** (`62` / `72` / `4`) — tensor-core cache copy: SMEM→TMEM gather used to stage operand tiles.

### TMA (Tensor Memory Accelerator, UTMA unit)
- **UTMALDG.2D / 3D / 4D / 5D [ / .2CTA / .MULTICAST / .MULTICAST.2CTA / .IM2COL]** (2D `60`, 3D `390`, 3D.2CTA `469`, 3D.MULTICAST `22`, 3D.MULTICAST.2CTA `34`, 4D `1 718`, 4D.2CTA `5`, 4D.MULTICAST `8`, 4D.MULTICAST.2CTA `18`, 5D `325`, 5D.IM2COL `16`, 2D.MULTICAST `12`, 2D.MULTICAST.2CTA `12`, 2D.2CTA `4`) — bulk tensor load global→shared/TMEM.
  Form: `UTMALDG.{nD}[.MULTICAST[.2CTA]] [URdst], [URcoords], desc[URtma]`. `.IM2COL` does implicit image-to-column reshaping for conv.
- **UTMASTG.2D / 3D / 4D / 5D / 5D.IM2COL** (`52` / `370` / `4` / `2 596` / `8`) — bulk tensor store.
- **UTMAREDG.3D.ADD / 4D.ADD / 5D.ADD** (`8` / `36` / `36`) — bulk tensor reduction (atomic-add-into) from SMEM→GMEM.
- **UTMAPF.L2.3D / .L2.4D** (`72` / `680`) — TMA prefetch into L2.
- **UTMACCTL.PF / UTMACCTL.IV** (`478` / `76`) — TMA cache control (prefetch / invalidate).
- **UTMACMDFLUSH** (`488`) — drain pending TMA command buffer.
- **ARRIVES.LDGSTSBAR.64.TRANSCNT / .ARVCNT** (`158` / `4`) — arrive on an mbarrier tracking LDGSTS or TMA transactions (transaction-count / arrival-count modes).

### LDSM / STSM (matrix distribution helpers)
- **LDSM.16.MT88.2 / MT88.4 / M88.2 / M88.4** (`16` / `278` / `16` / `16`) — `ldmatrix` equivalents: load 8×8 fragment(s) of 16-bit elements from shared memory into the register file laid out for an MMA operand. `MT` = transposed, `M` = native.
- **STSM.16.MT88.4 / M88.2** (`294` / `16`) — `stmatrix` equivalents.

### UBLKCP.S.G   (`3`)
Uniform block-copy shared→global (one-shot TMA fallback path).

### UGETNEXTWORKID.BROADCAST   (`37`)
Fetch next grid-work item (used by persistent kernel schedulers like CUTLASS's grid-dependency scheme), broadcasting the uniform result to the CTA.

### UCGABAR_ARV / UCGABAR_WAIT   (`46` / `46`)
Cluster-GA (CGA / thread-block cluster) barrier arrive / wait.

---

## 8. Register-file and spill plumbing

- **CS2R / CS2R.32** (`4 029` / `220`) — convergent-special-register read (`SR_CLOCKLO`, `SR_TMEM_ALLOC`, etc.) with warp-converging semantics.
- **RPCMOV.32** (`38`) — return-PC conditional move (used by the Blackwell call/return stack).
- **MOV.SPILL** (`3 611`) and **R2UR.FILL** (`3 611`) — compiler-emitted spill/fill pairs against a stack slot.

---

## 9. Quick operand-form cheat-sheet

```
  IMAD.WIDE.U32 R12, R12, UR10, RZ        ; R12:R13 = uR12 * uUR10 + 0  (64-bit)
  LDG.E.U16     R28, desc[UR14][R38.64]   ; load 16-bit; 64-bit virtual addr = desc+R38
  UTMALDG.3D    [UR48], [UR4], desc[UR8]  ; 3D TMA copy into smem at [UR48] from coords [UR4]
  UTCHMMA.2CTA  gdesc[UR38], gdesc[UR40], tmem[UR8], tmem[UR24], idesc[UR25], UP2
                ; A,B=gdesc matrix descriptors; C_in=tmem[UR8]; C_out=tmem[UR24];
                ; instruction descriptor idesc[UR25]; guard UP2
  SYNCS.PHASECHK.TRANS64.TRYWAIT  UPd, [UR6+0x40], UR4
                ; non-blocking phase check on an mbarrier at [UR6+0x40] with phase UR4
  F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C R4, R10, R11, R4
                ; pack (FP32 R10, FP32 R11) -> two e4m3 lanes, merge into existing R4
  LOP3.LUT      R5, R6, R7, R8, 0xE2, !PT ; bitwise (a&b)|c  (imm 0xE2)
  ISETP.GE.U32.AND  P0, PT, R3, UR5, PT   ; P0 = (uR3 >= UR5) ; !PT (second output) = negated
```

---

## 10. Summary histogram (top 30 most-used mnemonics across the 47 files)

```
  51 309  R2UR                       6 130  LDG.E.U16
  44 882  UMOV                       5 447  PRMT
  43 136  FMUL                       5 357  HFMA2
  41 556  LOP3.LUT                   5 116  UTCHMMA
  39 204  UIADD3                     4 901  USHF.L.U32
  32 998  BRA                        4 859  ULOP3.LUT
  26 503  MOV                        4 850  ISETP.GE.U32.AND
  20 689  IADD3                      4 845  BRA.U
  18 842  LDG.E                      4 818  SYNCS.EXCH.64
  18 747  FFMA                       4 748  LEA
  17 707  F2FP.F16.F32.PACK_AB       4 723  S2UR
  16 699  IMAD.MOV.U32               4 319  LDS
  15 116  FSETP.GEU.AND              4 029  CS2R
  14 512  FMUL2                      3 997  BSSY/BSYNC.RECONVERGENT
  14 327  MUFU.EX2                   3 971  LDC
  13 524  SYNCS.PHASECHK.TRANS64.TRYWAIT
  11 980  HADD2.F32
  11 930  IMAD.U32
  11 189  FSEL
  10 899  VIADD
  10 331  IMAD
   9 073  ISETP.GE.AND
   9 002  SHF.R.U64
   8 712  LDC.64
   8 581  SHF.L.U32
   8 487  NOP
   8 314  ULEA
   8 293  LDCU.64
   7 194  USEL
```

Caveats
- This is Blackwell (`sm_100a`) SASS as emitted by the ptxas that ships with the CUDA 13 toolchain used by CUTLASS. Mnemonics on earlier archs (Ampere/Hopper) share most of the integer/FP set but differ in the TMEM/UMMA/UTMA/SYNCS/UTC families.
- NVIDIA does not publish a SASS spec. Semantics above are reverse-engineered from PTX↔SASS correspondence plus public disassembler conventions; treat operand exact slots as "approximate" rather than authoritative.
- The full per-file 422-variant list is in `/tmp/all_sass_ops.txt` (sorted by occurrence count).
