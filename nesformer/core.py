"""The NES CPU (a 6502 without decimal mode), its 2 KB RAM and an NROM cartridge's PRG-ROM as a
circuit of stock Qwen3.5 layers (Gated DeltaNet heads + SwiGLU MLPs), built with Qwenputer's q35
compiler (chipformer/q35). spec.py has the trace format; cpu.py is the reference it must match.

Every position computes the instruction that the next token belongs to. Writes take their key
from t-1 (a conv tap), their write gate from the token at t, and their value from the token's
payload (registers, flags, latches, stored bytes) or from dims computed at t-1 below the writing
layer (the byte a push stores).

  L1  attn  PC anchor; counters n1 n2 n3 (final tokens of each length) and j (tokens of the
            record); A X Y SP; flags N V Z C I D; latches LO HI; the NMI flag
      mlp   thermometers; SP - 1, SP + 1; LO + Y
  L2  mlp   PC word w = anchor + n1 + 2 n2 + 3 n3, w + 1, w + 2; the refresh test
  L3  attn  PRG-ROM fetch: opcode and operand bytes at w (16 key heads, 128 KB of byte rows)
      mlp   opcode one-hots; operand bits; B1 + X, B1 + Y; branch flags
  L4  mlp   the effective address (by mode and record stage); the byte a push stores
  L5  attn  RAM (2 KB of bit rows, written by ST / RM / PSH) and PRG-ROM data bytes, at EA
      mlp   the operand M; adder (ADC / SBC), compares, BIT; address decode
  L6  mlp   the result byte, carry, overflow, flags of P records, jump targets
  L7  mlp   the output code (the exit): the next token
"""

import numpy as np

from chipformer.dsl.graph import Expr
from chipformer.q35.ir import A_BIT, A_REF, HD, REF_COLS, Chan, K, Target
from chipformer.q35.prims import Circ, chan, from_bits, sel, thead
from nesformer import spec as S
from nesformer.cpu import LENGTH, OPS, PAGE_PENALTY

TARGET = Target(key_heads=16, value_heads=32, intermediate=9216)  # Qwen3.5-4B's layer shapes
OUT_NAMES = list(S.FAMILIES) + S.CONTROL_NAMES
FAM_ID = {n: i for i, n in enumerate(OUT_NAMES)}
NFAM = (len(OUT_NAMES) - 1).bit_length()
NPAY = 16
OUT_W = 2.0**7
JMAX = 8
CODE_AMP = 128
KAPPA = 2.0**-20
BYTE_FAMS = ("A", "X", "Y", "ST", "RM", "S", "LO", "HI", "PSH", "PUL", "IN", "IOW", "DM")


def lits_of(bits, v):
    return [b if v >> i & 1 else 1 - b for i, b in enumerate(bits)]


def andpre(lits) -> Expr:
    return sum(lits, Expr()) - (len(lits) - 1)


def famp(t: int):
    return S.family(t) if 0 <= t < S.VOCAB else (None, 0)


class NesCore(Circ):
    def __init__(self, target: Target | None = None):
        super().__init__(target or TARGET)
        self.tables = {}
        self.vfeats = {}  # dim -> vectorized feature fn(fam, payload array)

    floor = 0

    def ffn(self, terms, name="ffn", into=None, min_level=0):
        return super().ffn(terms, name, into, max(min_level, self.floor))

    # --- token features ------------------------------------------------------------------------
    def feat(self, name: str, fn) -> Expr:
        """A token feature fn(fam, p) -> int, vectorizable over p (a numpy array)."""
        def scalar(t):
            fam, p = famp(t)
            return int(np.asarray(fn(fam, np.asarray(p))).item())
        e = self.token_feature(name, scalar, full=True)
        (d,) = e.terms
        self.vfeats[d] = fn
        return e

    # --- helpers (from Qwenputer's RV core) ----------------------------------------------------
    def zeros(self, n: int, name: str) -> list:
        return [self.feat(f"{name}[{i}]", lambda f, p: p * 0) for i in range(n)]

    def table(self, name, queries, keys, vheads, levels="bit", tiny=0.0):
        first, rest = vheads[0], vheads[1:]
        outs = self.head(name, {s: sel(self, e) for s, e in queries.items()},
                         {s: sel(self, e) for s, e in keys.items()}, first["v"], first.get("a", 0),
                         first["z"], first["outs"], tiny=tiny, beta=first.get("beta"),
                         into=first.get("into"),
                         vx=[{"v": x["v"], "a": x.get("a", 0), "z": x["z"], "outs": x["outs"],
                              "beta": x.get("beta"), "into": x.get("into")} for x in rest])
        node = self.nodes[-1]
        node.levels = levels
        self.tables[name] = node
        return outs

    def slots(self, name, vheads):
        """A key head of one-row tables (one per value head): data written from the token."""
        one = self.one
        outs = thead(self, name, {0: one}, {0: one}, vheads)
        self.tables[name] = self.nodes[-1]
        return outs

    def counter(self, name, reset, step):
        one = self.one
        ((x,),) = thead(self, name, {0: reset, 1: one - reset}, {0: one - reset, 1: one - reset},
                        [{"data": {2: reset}, "ref": one - reset,
                          "a": 110 * step + 238 * reset}], tiny=2.0**-90)
        self.tables[name] = self.nodes[-1]
        self.nodes[-1].levels = "decay"
        return x

    def thermo(self, x, reset, n, name):
        one = self.one
        return [self.node([(1, one - reset, 7 * one - 8 * 2**m * x),
                           (-1, one - reset, 6 * one - 8 * 2**m * x)], f"{name}>{m}")
                for m in range(n)]

    def int_bits(self, z: Expr, hi: int, name: str, n: int, lits=()) -> list:
        out = []
        for j in range(n):
            terms = []
            for theta in range(1, hi + 1):
                d = (theta >> j & 1) - ((theta - 1) >> j & 1)
                if d:
                    terms += self.t_ge(z, theta, 0, hi, d, lits=list(lits))
            out.append(self.node(terms, f"{name}.b{j}"))
        return out

    def fn_terms(self, z, lo: int, hi: int, f, lits=(), coef=1) -> list:
        """Unit terms of coef * f(z) (0 or 1) for an integer z in [lo, hi], AND lits."""
        terms = []
        if f(lo):
            terms += self.t_and(list(lits), coef) if lits else self.t_const(coef)
        for theta in range(lo + 1, hi + 1):
            d = f(theta) - f(theta - 1)
            if d:
                terms += self.t_ge(z, theta, lo, hi, d * coef, lits=list(lits))
        return terms

    def add8(self, z: Expr, hi: int, name: str):
        """Bits of z mod 256 and the carry [z >= 256], for an integer z in [0, hi <= 511]."""
        bits = [self.node(self.fn_terms(z, 0, hi, lambda v, j=j: v >> j & 1), f"{name}.b{j}")
                for j in range(8)]
        carry = self.node(self.fn_terms(z, 0, hi, lambda v: int(v >= 256)), f"{name}.c")
        return bits, carry

    # bit j of x + 1, x - 1 and x + c (c a 0/1 literal), as AND terms over x's bits
    def inc_terms(self, x, j, lits, coef=1):
        lo = list(x[:j])
        return (self.t_and(lits + [x[j]], coef) + self.t_and(lits + lo, coef)
                + self.t_and(lits + lo + [x[j]], -2 * coef))

    def dec_terms(self, x, j, lits, coef=1):
        lo = [1 - b for b in x[:j]]
        return (self.t_and(lits + [x[j]], coef) + self.t_and(lits + lo, coef)
                + self.t_and(lits + lo + [x[j]], -2 * coef))

    def addc_terms(self, x, c, j, lits, coef=1):
        lo = list(x[:j])
        return (self.t_and(lits + [x[j]], coef) + self.t_and(lits + [c] + lo, coef)
                + self.t_and(lits + [c] + lo + [x[j]], -2 * coef))

    def bit_terms(self, src, j, lits, coef=1):
        """coef * bit j of a source: ("bits", list) | ("inc", list) | ("dec", list) |
        ("addc", list, c) | ("const", int) | ("and", a, b) | ("or", a, b) | ("xor", a, b)."""
        kind = src[0]
        if kind == "bits":
            b = src[1][j]
            if isinstance(b, int):
                return self.t_and(lits, coef) if b else []
            return self.t_and(lits + [b], coef) if b.terms else []
        if kind == "const":
            return self.t_and(lits, coef) if src[1] >> j & 1 else []
        if kind == "inc":
            return self.inc_terms(src[1], j, lits, coef)
        if kind == "dec":
            return self.dec_terms(src[1], j, lits, coef)
        if kind == "addc":
            return self.addc_terms(src[1], src[2], j, lits, coef)
        a, b = src[1][j], src[2][j]
        if kind == "and":
            return self.t_and(lits + [a, b], coef)
        if kind == "or":
            return (self.t_and(lits + [a], coef) + self.t_and(lits + [b], coef)
                    + self.t_and(lits + [a, b], -coef))
        if kind == "xor":
            return self.t_and(lits + [a, 1 - b], coef) + self.t_and(lits + [1 - a, b], coef)
        raise ValueError(kind)

    def select(self, cases, n, name):
        """n bit nodes: sum over (lits, source) cases (mutually exclusive) of the source's bit."""
        out = []
        for j in range(n):
            terms = []
            for lits, src in cases:
                terms += self.bit_terms(src, j, list(lits))
            out.append(self.node(terms, f"{name}.b{j}"))
        return out


def build(target: Target | None = None) -> NesCore:
    c = NesCore(target)
    one, f = c.one, c.feat

    # ------------------------------------------------------------------ token features
    names = list(S.FAMILIES) + S.CONTROL_NAMES
    fam = {n: f(f"is{n}", lambda fm, p, n=n: (p * 0 + int(fm == n))) for n in names}

    def pbit(fams, i, shift=0):
        return lambda fm, p: (p >> (i + shift) & 1) if fm in fams else p * 0

    vb = [f(f"vb{i}", pbit(BYTE_FAMS, i)) for i in range(8)]  # the payload's low byte
    wb = [f(f"wb{i}", pbit(("ST", "RM"), i)) for i in range(8)]  # a stored byte
    inb = [f(f"in{i}", pbit(("IN",), i)) for i in range(8)]  # an I/O read's byte
    anc = [f(f"anc{i}", pbit(("J", "JI", "R"), i)) for i in range(15)]
    is_anchor = fam["J"] + fam["JI"] + fam["R"]
    lens = {L: f(f"len={L}", lambda fm, p, L=L: (vlen(fm, p) == L) * 1) for L in (1, 2, 3)}
    jreset = f("jreset", lambda fm, p: vfinal(fm, p) | (p * 0 + int(fm == "DM")))
    iow0 = f("iow0", lambda fm, p: ((p >> 14) == 0) * 1 if fm == "IOW" else p * 0)
    dmnext = [f(f"dmnext{i}", lambda fm, p, i=i: ((((p >> 8) + 1) & 255) >> i & 1)
                if fm == "DM" else p * 0) for i in range(8)]
    dmlast = f("dmlast", lambda fm, p: ((p >> 8) == 255).astype(int) * 1 if fm == "DM" else p * 0)

    def flagf(axy, rm, pbitn, extra=None):
        def fn(fm, p):
            if fm in ("A", "X", "Y") and axy is not None:
                return axy(p)
            if fm == "RM" and rm is not None:
                return rm(p)
            if fm == "P" and pbitn is not None:
                return p >> pbitn & 1
            if extra and fm in extra:
                return p * 0 + extra[fm]
            return p * 0
        return fn

    fN = f("fN", flagf(lambda p: p >> 7 & 1, lambda p: p >> 7 & 1, 5))
    fZ = f("fZ", flagf(lambda p: ((p & 255) == 0) * 1, lambda p: ((p & 255) == 0) * 1, 1))
    fC = f("fC", flagf(lambda p: p >> 8 & 1, lambda p: p >> 8 & 1, 0))
    fV = f("fV", flagf(lambda p: p >> 9 & 1, None, 4))
    fI = f("fI", flagf(None, None, 2, {"JI": 1}))
    fD = f("fD", flagf(None, None, 3))
    axy = fam["A"] + fam["X"] + fam["Y"]
    ramvh = []
    for x in range(2):
        def fn(fm, p, x=x):
            if fm == "ST":
                return ((p >> 8 & 1) == x) * 1
            if fm == "RM":
                return ((p >> 9 & 1) == x) * 1
            if fm == "PSH":
                return (((((p & 255) + 1) & 255) >> 7) == x) * 1
            return p * 0
        ramvh.append(f(f"ramvh{x}", fn))

    # ------------------------------------------------------------------ L1: state heads
    (anchor,) = c.slots("pc", [{"data": {2 + i: anc[i] for i in range(15)}, "ref": one,
                                "beta": is_anchor}])
    A, X = c.slots("regAX", [{"data": {2 + i: vb[i] for i in range(8)}, "ref": one,
                              "beta": fam["A"]},
                             {"data": {2 + i: vb[i] for i in range(8)}, "ref": one,
                              "beta": fam["X"]}])
    Y, SP = c.slots("regYS", [{"data": {2 + i: vb[i] for i in range(8)}, "ref": one,
                               "beta": fam["Y"]},
                              {"data": {2 + i: vb[i] for i in range(8)}, "ref": one,
                               "beta": fam["S"] + fam["PSH"] + fam["PUL"]}])
    LO, HI = c.slots("lat", [{"data": {2 + i: vb[i] for i in range(8)}, "ref": one,
                              "beta": fam["LO"]},
                             {"data": {2 + i: vb[i] for i in range(8)}, "ref": one,
                              "beta": fam["HI"]}])
    wNZC = axy + fam["RM"] + fam["P"]
    (Nf,), (Zf,) = c.slots("flNZ", [{"data": {2: fN}, "ref": one, "beta": wNZC},
                                     {"data": {2: fZ}, "ref": one, "beta": wNZC}])
    (Cf,), (Vf,) = c.slots("flCV", [{"data": {2: fC}, "ref": one, "beta": wNZC},
                                     {"data": {2: fV}, "ref": one, "beta": axy + fam["P"]}])
    (If,), (Df,) = c.slots("flID", [{"data": {2: fI}, "ref": one, "beta": fam["P"] + fam["JI"]},
                                     {"data": {2: fD}, "ref": one, "beta": fam["P"]}])
    ((irq,),) = c.slots("irq", [{"data": {2: fam["NMI"]}, "ref": one,
                                 "beta": fam["NMI"] + fam["JI"]}])
    x1 = c.counter("n1", is_anchor, lens[1])
    x2 = c.counter("n2", is_anchor, lens[2])
    x3 = c.counter("n3", is_anchor, lens[3])
    xj = c.counter("j", jreset, one - jreset)
    t1 = c.thermo(x1, is_anchor, 62, "n1")
    t2 = c.thermo(x2, is_anchor, 31, "n2")
    t3 = c.thermo(x3, is_anchor, 21, "n3")
    jt = c.thermo(xj, jreset, JMAX, "j")
    offset = sum(t1, Expr()) + 2 * sum(t2, Expr()) + 3 * sum(t3, Expr())

    def jeq(v):
        return (one if v == 0 else jt[v - 1]) - (jt[v] if v < JMAX else 0)

    j0 = jeq(0)
    Aint, Xint, Yint, SPint = (from_bits(r) for r in (A, X, Y, SP))
    spm1 = c.select([([one], ("dec", SP))], 8, "sp-1")
    spp1 = c.select([([one], ("inc", SP))], 8, "sp+1")
    ely, cly = c.add8(from_bits(LO) + Yint, 510, "lo+y")
    lo_ones = c.and_(list(LO), "lo==ff")

    # ------------------------------------------------------------------ L2: PC
    def add_small(abits, extra, name):
        L = from_bits(abits[:6]) + offset + extra
        hi = 63 + 62 + extra
        bits = [c.node(c.fn_terms(L, 0, hi, lambda v, j=j: v >> j & 1), f"{name}.b{j}")
                for j in range(6)]
        for j in range(6, 15):
            below = abits[6:j]
            terms = c.t_and([abits[j]])
            terms += c.t_ge(L, 64, 0, hi, 1, lits=below + [one - abits[j]])
            terms += c.t_ge(L, 64, 0, hi, -1, lits=below + [abits[j]])
            bits.append(c.node(terms, f"{name}.b{j}"))
        return bits

    w = add_small(anchor, 0, "w")
    w1 = add_small(anchor, 1, "w+1")
    w2 = add_small(anchor, 2, "w+2")
    at_max = c.node(c.t_ge(offset, S.OFF_MAX, 0, 62), "at_max")

    # ------------------------------------------------------------------ L3: fetch
    # address w (15 bits) holds the word (op, b1, b2, 0): key head w[14:11], value head w[10],
    # row pair (p, p + 64) with p = w[9:4], half w[3], word w[2:0] (Qwenputer's ROM layout)
    fdims = c.zeros(4, "rom.b")
    for h in range(16):
        hl = lits_of(w[11:15], h)
        q = {}
        for p in range(64):
            e = andpre(lits_of(w[4:10], p) + hl)
            q[p], q[p + 64] = e, e
        vhs = []
        for x in range(2):
            z, outs, into = {}, [], []
            for half in range(2):
                for k in range(8):
                    ze = K * andpre(lits_of([w[3]], half) + lits_of(w[0:3], k)
                                    + lits_of([w[10]], x))
                    for b in range(3):
                        col = 2 + 32 * half + 4 * k + b
                        z[col] = ze
                        outs.append((col, OUT_W))
                        into.append(fdims[b])
            vhs.append({"v": {}, "z": z, "outs": outs, "into": into})
        c.table(f"rom{h}", q, {}, vhs, levels="byte")
    B0, B1, B2 = (256 * d for d in fdims[:3])

    # ------------------------------------------------------------------ L3: decode
    op_is = {o: c.node(c.t_eq(B0, o, 0, 255, lits=[one - irq]), f"op{o:02X}") for o in OPS}

    def ops(pred) -> Expr:
        return sum((op_is[o] for o, (mn, mode, _) in OPS.items() if pred(mn, mode)), Expr())

    def mn_is(*mns):
        return ops(lambda mn, mode: mn in mns)

    illegal = one - irq - sum(op_is.values(), Expr())
    ln = {L: ops(lambda mn, mode, L=L: LENGTH[mode] == L) for L in (1, 2, 3)}
    len_bits = [ln[1] + ln[3], ln[2] + ln[3]]
    b1 = c.int_bits(B1, 255, "b1", 8)
    b2 = c.int_bits(B2, 255, "b2", 8)
    ezx, czx = c.add8(B1 + Xint, 510, "b1+x")
    ezy, czy = c.add8(B1 + Yint, 510, "b1+y")
    taken = c.node([t for mn, (fl, want) in {"BPL": (Nf, 0), "BMI": (Nf, 1), "BVC": (Vf, 0),
                                              "BVS": (Vf, 1), "BCC": (Cf, 0), "BCS": (Cf, 1),
                                              "BNE": (Zf, 0), "BEQ": (Zf, 1)}.items()
                    for t in c.t_and([mn_is(mn), fl if want else 1 - fl])], "taken")
    # branch target: PC + 2 + sext(b1); low byte by thresholds, the high byte +-1 below
    pc2_lo = from_bits(w2[0:8])
    btl, bcy = c.add8(pc2_lo + B1, 510, "btgt")
    pc2_hi = w2[8:15] + [one]

    # ------------------------------------------------------------------ L4: effective address
    data_op = {"ORA", "AND", "EOR", "ADC", "LDA", "CMP", "SBC", "STA", "STX", "STY", "LDX",
               "LDY", "CPX", "CPY", "BIT", "ASL", "ROL", "LSR", "ROR", "INC", "DEC"}

    def mode_set(m):
        return ops(lambda mn, mode: mode == m and mn in data_op)

    store_reg = [("STA", A), ("STX", X), ("STY", Y)]
    dmastage = fam["DM"] + iow0
    ram_page = [one if 0x01 >> i & 1 else Expr() for i in range(8)]
    cases_lo, cases_hi = [], []

    def ea(lits, lo, hi):
        cases_lo.append((lits, lo))
        cases_hi.append((lits, hi))

    zero = ("const", 0)
    nd = [1 - dmastage]
    ea([mode_set("imm")] + nd, ("bits", w1[0:8]), ("bits", w1[8:15] + [one]))
    ea([mode_set("zp")] + nd, ("bits", b1), zero)
    ea([mode_set("zpx")] + nd, ("bits", ezx), zero)
    ea([mode_set("zpy")] + nd, ("bits", ezy), zero)
    ea([mode_set("abs")] + nd, ("bits", b1), ("bits", b2))
    ea([mode_set("abx")] + nd, ("bits", ezx), ("addc", b2, czx))
    ea([mode_set("aby")] + nd, ("bits", ezy), ("addc", b2, czy))
    inx, iny = mode_set("inx"), mode_set("iny")
    ea([inx, j0], ("bits", ezx), zero)
    ea([inx, fam["LO"]], ("inc", ezx), zero)
    for st in (fam["HI"], fam["IN"]):
        ea([inx, st], ("bits", LO), ("bits", HI))
        ea([iny, st], ("bits", ely), ("addc", HI, cly))
    ea([iny, j0], ("bits", b1), zero)
    ea([iny, fam["LO"]], ("inc", b1), zero)
    jind = mn_is("JMP") - ops(lambda mn, mode: mn == "JMP" and mode == "abs")
    ea([jind, j0], ("bits", b1), ("bits", b2))
    ea([jind, fam["LO"]], ("inc", b1), ("bits", b2))
    stack_ops = mn_is("JSR", "RTS", "RTI", "PHA", "PHP", "PLA", "PLP")
    brk = mn_is("BRK")
    ea([stack_ops], ("bits", SP), ("bits", ram_page))
    ea([brk, 1 - jeq(3), 1 - jeq(4)], ("bits", SP), ("bits", ram_page))
    ea([brk, jeq(3)], ("const", 0xFE), ("const", 0xFF))
    ea([brk, jeq(4)], ("const", 0xFF), ("const", 0xFF))
    ea([irq, 1 - jeq(4), 1 - jeq(5)], ("bits", SP), ("bits", ram_page))
    ea([irq, jeq(4)], ("const", 0xFA), ("const", 0xFF))
    ea([irq, jeq(5)], ("const", 0xFB), ("const", 0xFF))
    for mn, r in store_reg:  # OAM DMA: page = the stored register, index 0 / next
        ea([iow0, mn_is(mn)], zero, ("bits", r))
        ea([fam["DM"], mn_is(mn)], ("bits", dmnext), ("bits", r))
    EA = c.select(cases_lo, 8, "ea.lo") + c.select(cases_hi, 8, "ea.hi")

    # the byte a push stores (written by the PSH token at t + 1)
    def p_bits(b):
        return [Cf, Zf, If, Df, one if b else Expr(), one, Vf, Nf]

    jsr, php, pha = mn_is("JSR"), mn_is("PHP"), mn_is("PHA")
    pushdata = c.select([
        ([pha], ("bits", A)), ([php], ("bits", p_bits(1))),
        ([jsr, j0], ("bits", pc2_hi)), ([jsr, jeq(1)], ("bits", w2[0:8])),
        ([brk, j0], ("bits", pc2_hi)), ([brk, jeq(1)], ("bits", w2[0:8])),
        ([brk, jeq(2)], ("bits", p_bits(1))),
        ([irq, jeq(1)], ("bits", w[8:15] + [one])), ([irq, jeq(2)], ("bits", w[0:8])),
        ([irq, jeq(3)], ("bits", p_bits(0)))], 8, "push")

    # ------------------------------------------------------------------ L5: RAM, ROM data
    ram_lits = [1 - EA[15], 1 - EA[14], 1 - EA[13]]
    mem = c.zeros(8, "mem")
    for h in range(8):  # byte a: key head a[10:8], value head a[7], row a[6:0]
        hl = lits_of(EA[8:11], h)
        keys, q = {}, {}
        for row in range(128):
            e = andpre(lits_of(EA[0:7], row) + hl + ram_lits)
            keys[row] = Chan(e, (0, 1, 0, 0))
            q[row] = e
        vhs = []
        for x in range(2):
            v, z, outs, into = {}, {}, [], []
            for i in range(8):
                v[2 + i] = chan(Chan(wb[i]), A_BIT)  # ST / RM: the payload's byte
                v[10 + i] = chan(Chan(pushdata[i], (0, 1, 0, 0)), A_BIT)  # PSH: from t-1
            for rc in REF_COLS:
                v[rc] = chan(one, A_REF)
            for col in range(2, 18):
                z[col] = K * (EA[7] if x else 1 - EA[7])
                outs.append((col, OUT_W))
                into.append(mem[(col - 2) % 8])
            vhs.append({"v": v, "z": z, "outs": outs, "into": into,
                        "beta": ramvh[x]})
        c.table(f"ram{h}", q, keys, vhs)
    # PRG data bytes: word wi = EA[14:2]: key head EA[14:13], value head EA[12], row pair EA[11:6],
    # half EA[5], word EA[4:2], byte lane EA[1:0] (picked by the output gate)
    romd = c.zeros(1, "romd")[0]
    for h in range(4):
        hl = lits_of(EA[13:15], h) + [EA[15]]
        q = {}
        for p in range(64):
            e = andpre(lits_of(EA[6:12], p) + hl)
            q[p], q[p + 64] = e, e
        vhs = []
        for x in range(2):
            z, outs, into = {}, [], []
            for half in range(2):
                for k in range(8):
                    for b in range(4):
                        col = 2 + 32 * half + 4 * k + b
                        z[col] = K * andpre(lits_of([EA[5]], half) + lits_of(EA[2:5], k)
                                            + lits_of([EA[12]], x) + lits_of(EA[0:2], b))
                        outs.append((col, OUT_W))
                        into.append(romd)
            vhs.append({"v": {}, "z": z, "outs": outs, "into": into})
        c.table(f"prg{h}", q, {}, vhs, levels="byte")

    # ------------------------------------------------------------------ L5: operand, ALU
    Mint = from_bits(mem) + 256 * romd + from_bits(inb)
    romb = c.int_bits(256 * romd, 255, "romb", 8)
    mb = [mem[i] + romb[i] + inb[i] for i in range(8)]
    adc, sbc = mn_is("ADC"), mn_is("SBC")
    za = Aint + Mint + Cf
    zs = Aint + 255 - Mint + Cf
    s_terms = [[] for _ in range(8)]
    for z, lit in ((za, adc), (zs, sbc)):
        for j in range(8):
            s_terms[j] += c.fn_terms(z, 0, 511, lambda v, j=j: v >> j & 1, lits=[lit])
    ssum = [c.node(s_terms[j], f"sum.b{j}") for j in range(8)]
    scar = c.node(c.t_ge(za, 256, 0, 511, lits=[adc]) + c.t_ge(zs, 256, 0, 511, lits=[sbc]),
                  "sum.c")
    sovf = c.node(c.t_ge(za, 128, 0, 511, lits=[adc, 1 - A[7]])
                  + c.t_ge(Mint, 128, 0, 255, -1, lits=[adc, 1 - A[7]])
                  + c.t_ge(Mint, 128, 0, 255, lits=[adc, A[7]])
                  + c.t_ge(za, 384, 0, 511, -1, lits=[adc, A[7]])
                  + c.t_ge(zs, 128, 0, 511, lits=[sbc, 1 - A[7]])
                  + c.t_lt(Mint, 128, 0, 255, -1, lits=[sbc, 1 - A[7]])
                  + c.t_lt(Mint, 128, 0, 255, lits=[sbc, A[7]])
                  + c.t_ge(zs, 384, 0, 511, -1, lits=[sbc, A[7]]), "sum.v")
    cmpC, cmpZ, cmpN = [], [], []
    for mn, R in (("CMP", Aint), ("CPX", Xint), ("CPY", Yint)):
        z, lit = R + 256 - Mint, mn_is(mn)
        cmpC += c.t_ge(z, 256, 1, 511, lits=[lit])
        cmpZ += c.t_eq(z, 256, 1, 511, lits=[lit])
        cmpN += c.t_in(z, 128, 256, 1, 511, lits=[lit]) + c.t_ge(z, 384, 1, 511, lits=[lit])
    cmpC, cmpZ, cmpN = (c.node(t, nm) for t, nm in ((cmpC, "cmp.c"), (cmpZ, "cmp.z"),
                                                    (cmpN, "cmp.n")))
    bit_op = mn_is("BIT")
    bit_and = [c.node(c.fn_terms(Mint, 0, 255, lambda v, i=i: v >> i & 1, lits=[bit_op, A[i]]),
                      f"bit.and{i}") for i in range(8)]
    in_ram = c.and_(ram_lits, "in_ram")
    in_io = one - in_ram - EA[15]
    is_ppu = [1 - EA[15], 1 - EA[14], EA[13]]
    is_apu = [1 - EA[15], EA[14], 1 - EA[13]] + [1 - b for b in EA[5:13]]
    reg = []
    for i in range(6):
        terms = []
        if i < 3:
            terms += c.t_and(is_ppu + [EA[i]]) + c.t_and(is_apu + [EA[i]])
        elif i < 5:
            terms += c.t_and(is_apu + [EA[i]])
        if i == 5:
            terms += c.t_and(is_ppu)
        if i in (3, 5):  # other ($4020-$7FFF): 40 = 0b101000
            terms += c.t_and([1 - EA[15], EA[14]]) + c.t_and(is_apu, -1)
        reg.append(c.node(terms, f"reg.b{i}"))
    is4014 = c.and_(lits_of(EA, 0x4014), "ea==4014")
    hb_carry = c.and_([bcy, 1 - b1[7]], "btgt.up")
    hb_borrow = c.and_([1 - bcy, b1[7]], "btgt.down")

    # ------------------------------------------------------------------ L6: results
    loads = mn_is("LDA", "LDX", "LDY")
    ind_nl = ops(lambda mn, mode: mode in ("inx", "iny") and mn not in ("LDA", "LDX", "LDY"))
    rts, rti, pla, plp = (mn_is(m) for m in ("RTS", "RTI", "PLA", "PLP"))

    def final_cases(pred):
        """Literal sets of the final stage of the opcodes matching pred."""
        d = ops(lambda mn, mode: pred(mn, mode) and mode not in ("inx", "iny"))
        i = ops(lambda mn, mode: pred(mn, mode) and mode in ("inx", "iny"))
        return [[d, 1 - dmastage], [i, fam["HI"]], [i, fam["IN"]]]

    vcases = [([loads], ("bits", mb)), ([ind_nl, j0], ("bits", mb)),
              ([ind_nl, fam["LO"]], ("bits", mb)), ([jind, j0], ("bits", mb)),
              ([rts, jeq(1)], ("bits", mb)), ([rti, jeq(3)], ("bits", mb)),
              ([brk, jeq(3)], ("bits", mb)), ([irq, jeq(4)], ("bits", mb)), ([pla], ("bits", mb)),
              ([dmastage], ("bits", mb))]
    for lits in final_cases(lambda mn, mode: mn in ("ADC", "SBC")):
        vcases.append((lits, ("bits", ssum)))
    for op_, kind in (("AND", "and"), ("ORA", "or"), ("EOR", "xor")):
        for lits in final_cases(lambda mn, mode, op_=op_: mn == op_):
            vcases.append((lits, (kind, A, mb)))
    for mn, r in store_reg:
        for lits in final_cases(lambda m, mode, mn=mn: m == mn):
            vcases.append((lits, ("bits", r)))
    for mn, r in (("TAX", A), ("TAY", A), ("TXA", X), ("TYA", Y), ("TSX", SP), ("TXS", X)):
        vcases.append(([mn_is(mn)], ("bits", r)))
    for mn, kind, r in (("INX", "inc", X), ("INY", "inc", Y), ("DEX", "dec", X),
                        ("DEY", "dec", Y)):
        vcases.append(([mn_is(mn)], (kind, r)))
    acc = {mn: ops(lambda m, mode, mn=mn: m == mn and mode == "acc")
           for mn in ("ASL", "LSR", "ROL", "ROR")}
    rmw = {mn: ops(lambda m, mode, mn=mn: m == mn and mode != "acc")
           for mn in ("ASL", "LSR", "ROL", "ROR", "INC", "DEC")}
    for src, lit in ((A, None), (mb, "rmw")):
        sets = acc if lit is None else rmw
        vcases.append(([sets["ASL"]], ("bits", [Expr()] + src[:7])))
        vcases.append(([sets["LSR"]], ("bits", src[1:] + [Expr()])))
        vcases.append(([sets["ROL"]], ("bits", [Cf] + src[:7])))
        vcases.append(([sets["ROR"]], ("bits", src[1:] + [Cf])))
    vcases.append(([rmw["INC"]], ("inc", mb)))
    vcases.append(([rmw["DEC"]], ("dec", mb)))
    val = c.select(vcases, 8, "val")

    shl = acc["ASL"] + acc["ROL"] + rmw["ASL"] + rmw["ROL"]
    shr = acc["LSR"] + acc["ROR"] + rmw["LSR"] + rmw["ROR"]
    old_c = (loads + mn_is("AND", "ORA", "EOR", "TAX", "TAY", "TXA", "TYA", "TSX", "INX", "INY",
                           "DEX", "DEY", "PLA") + rmw["INC"] + rmw["DEC"])
    cnew = c.node(c.t_and([adc + sbc, scar]) + c.t_and([old_c, Cf])
                  + c.t_and([acc["ASL"] + acc["ROL"], A[7]]) + c.t_and([acc["LSR"] + acc["ROR"],
                                                                        A[0]])
                  + c.t_and([rmw["ASL"] + rmw["ROL"], mb[7]])
                  + c.t_and([rmw["LSR"] + rmw["ROR"], mb[0]]), "cnew")
    assert shl.terms and shr.terms
    old_v = old_c - rmw["INC"] - rmw["DEC"] + sum(acc.values(), Expr())
    vnew = c.node(c.t_and([adc + sbc, sovf]) + c.t_and([old_v, Vf]), "vnew")
    cmp_any = mn_is("CMP", "CPX", "CPY")
    fl_ops = {mn: mn_is(mn) for mn in ("CLC", "SEC", "CLI", "SEI", "CLV", "CLD", "SED")}
    from_m = rti + plp

    def pflag(name, old, cmp_src, bit_src, op_set, op_clr, mbit):
        terms = c.t_and([cmp_any, cmp_src]) if cmp_src is not None else c.t_and([cmp_any, old])
        if bit_src == "bitZ":  # (A & M) == 0, folded in
            terms += c.t_and([bit_op] + [1 - b for b in bit_and])
        else:
            terms += (c.t_and([bit_op, bit_src]) if bit_src is not None
                      else c.t_and([bit_op, old]))
        terms += c.t_and([from_m, mb[mbit]])
        for mn, fl in fl_ops.items():
            if mn == op_set:
                terms += c.t_and([fl])
            elif mn != op_clr:
                terms += c.t_and([fl, old])
        return c.node(terms, f"pf.{name}")

    pf = [pflag("C", Cf, cmpC, None, "SEC", "CLC", 0), pflag("Z", Zf, cmpZ, "bitZ", None, None, 1),
          pflag("I", If, None, None, "SEI", "CLI", 2), pflag("D", Df, None, None, "SED", "CLD", 3),
          pflag("V", Vf, None, mb[6], None, "CLV", 6), pflag("N", Nf, cmpN, mb[7], None, None, 7)]
    pen = lambda m: ops(lambda mn, mode: mn in PAGE_PENALTY and mode == m)  # noqa: E731
    px = c.node(c.t_and([pen("abx"), czx]) + c.t_and([pen("aby"), czy])
                + c.t_and([pen("iny"), cly]), "px")
    # jump targets (15 bits below $8000) and bit 15 (the target is in the ROM)
    br = mn_is("BPL", "BMI", "BVC", "BVS", "BCC", "BCS", "BNE", "BEQ")
    jabs = ops(lambda mn, mode: mn in ("JMP", "JSR") and mode == "abs")
    lo_hi_mb = LO + mb[:7]
    tgt_cases = [([jabs], ("bits", b1 + b2[:7])), ([jind, fam["LO"]], ("bits", lo_hi_mb)),
                 ([rti, jeq(5)], ("bits", lo_hi_mb)), ([brk, jeq(4)], ("bits", lo_hi_mb)),
                 ([irq, jeq(5)], ("bits", lo_hi_mb)), ([br], ("bits", btl + [Expr()] * 7))]
    jtgt = []
    for j in range(15):
        terms = []
        for lits, src in tgt_cases:
            terms += c.bit_terms(src, j, lits)
        if j < 8:  # RTS: (mb << 8 | LO) + 1
            terms += c.inc_terms(LO, j, [rts, jeq(3)])
        else:
            terms += c.addc_terms(mb, lo_ones, j - 8, [rts, jeq(3)])
            # a branch's high byte: PC+2's, plus the low byte's carry, minus b1's sign
            terms += _updown_terms(c, pc2_hi, hb_carry, hb_borrow, j - 8, [br], with_base=True)
        jtgt.append(c.node(terms, f"jt.b{j}"))
    rom_terms = (c.t_and([jabs, b2[7]]) + c.t_and([jind, fam["LO"], mb[7]])
                 + c.t_and([rti, jeq(5), mb[7]]) + c.t_and([brk, jeq(4), mb[7]])
                 + c.t_and([irq, jeq(5), mb[7]]) + c.addc_terms(mb, lo_ones, 7, [rts, jeq(3)])
                 + _updown_terms(c, pc2_hi, hb_carry, hb_borrow, 7, [br], with_base=True))
    tgt_rom = c.node(rom_terms, "jt.rom")

    # ------------------------------------------------------------------ L7: the next token
    start = [j0, 1 - fam["DM"]]
    go = start + [1 - at_max]
    cases = []  # (lits, family, payload bits (list) or int)

    def case(lits, fam_name, payload=0):
        cases.append((list(lits), fam_name, payload))

    def bits_of(v, n):
        return [one if v >> i & 1 else Expr() for i in range(n)]

    L2 = list(len_bits)
    zeros3 = [Expr()] * 3
    pay = {
        "reg": lambda: val + [cnew, vnew, px] + L2,
        "ST": lambda: val + [EA[7]] + L2,
        "RM": lambda: val + [cnew, EA[7]] + L2,
        "P": lambda ln_=None: pf + [px] + (L2 if ln_ is None else bits_of(ln_, 2)),
    }
    case(start + [at_max], "R", w)
    case(go + [illegal], "HALT")
    # reads: (LO, HI,) IOR / IN, then the result
    rd_dest = {"A": ("LDA", "ADC", "SBC", "AND", "ORA", "EOR"), "X": ("LDX",), "Y": ("LDY",),
               "P": ("CMP", "CPX", "CPY", "BIT")}
    rd_all = sum(rd_dest.values(), ())

    def direct(mns):
        return ops(lambda mn, mode: mn in mns and mode not in ("inx", "iny"))

    def indir(mns):
        return ops(lambda mn, mode: mn in mns and mode in ("inx", "iny"))

    ind_all = indir(rd_all + ("STA",))
    case(go + [ind_all], "LO", val)
    case([ind_all, fam["LO"]], "HI", val)
    rmw_mem = sum(rmw.values(), Expr())
    case(go + [direct(rd_all) + rmw_mem, in_io], "IOR", reg)
    case([indir(rd_all), fam["HI"], in_io], "IOR", reg)
    for dest, mns in rd_dest.items():
        payload = pay["P"]() if dest == "P" else pay["reg"]()
        for lits in ([direct(mns), 1 - in_io] + go, [indir(mns), fam["HI"], 1 - in_io],
                     [ops(lambda mn, mode, mns=mns: mn in mns), fam["IN"]]):
            case(lits, dest, payload)
    # stores
    st_all = ("STA", "STX", "STY")
    for lits in (go + [direct(st_all)], [indir(st_all), fam["HI"]]):
        case(lits + [1 - in_io], "ST", pay["ST"]())
        case(lits + [in_io, 1 - is4014], "IOW", val + reg + L2)
        case(lits + [is4014], "IOW", val + reg + [Expr(), Expr()])
    case([iow0], "DM", mb + zeros3 * 2 + [Expr(), Expr()])
    case([fam["DM"], 1 - dmlast], "DM", mb + dmnext)
    case([fam["DM"], dmlast], "NT", L2)
    # read-modify-write
    case(go + [rmw_mem, 1 - in_io], "RM", pay["RM"]())
    case([rmw_mem, fam["IN"]], "RM", pay["RM"]())
    # one-token records
    for dest, mns in (("A", ("TXA", "TYA")), ("X", ("TAX", "TSX", "INX", "DEX")),
                      ("Y", ("TAY", "INY", "DEY"))):
        case(go + [mn_is(*mns)], dest, pay["reg"]())
    case(go + [sum(acc.values(), Expr())], "A", pay["reg"]())
    case(go + [mn_is("TXS")], "S", val + L2)
    case(go + [sum(fl_ops.values(), Expr())], "P", pay["P"]())
    case(go + [mn_is("NOP")], "NT", L2)
    # branches and jumps
    case(go + [br, taken, tgt_rom], "J", jtgt)
    case(go + [br, taken, 1 - tgt_rom], "HALT")
    case(go + [br, 1 - taken], "NT", L2)
    jmp_abs = ops(lambda mn, mode: mn == "JMP" and mode == "abs")
    case(go + [jmp_abs, b2[7]], "J", jtgt)
    case(go + [jmp_abs, 1 - b2[7]], "HALT")
    case(go + [jind, in_io], "HALT")
    case(go + [jind, 1 - in_io], "LO", val)
    case([jind, fam["LO"], tgt_rom], "J", jtgt)
    case([jind, fam["LO"], 1 - tgt_rom], "HALT")
    # the stack
    psh0 = spm1 + [Expr(), Expr()]
    case(go + [jsr, b2[7]], "PSH", psh0)
    case(go + [jsr, 1 - b2[7]], "HALT")
    case([jsr, jeq(1)], "PSH", psh0)
    case([jsr, jeq(2)], "J", jtgt)
    case(go + [rts + rti + pla + plp], "PUL", spp1)
    case([rts, jeq(1)], "LO", val)
    case([rts, jeq(2)], "PUL", spp1)
    case([rts, jeq(3), tgt_rom], "J", jtgt)
    case([rts, jeq(3), 1 - tgt_rom], "HALT")
    case([rti, jeq(1)], "P", pay["P"](0))
    case([rti, jeq(2)], "PUL", spp1)
    case([rti, jeq(3)], "LO", val)
    case([rti, jeq(4)], "PUL", spp1)
    case([rti, jeq(5), tgt_rom], "J", jtgt)
    case([rti, jeq(5), 1 - tgt_rom], "HALT")
    case(go + [pha + php], "PSH", spm1 + [one, Expr()])
    case([pla, jeq(1)], "A", pay["reg"]())
    case([plp, jeq(1)], "P", pay["P"]())
    for st, lits in (([brk], go), ([irq], [])):
        k = 0 if lits else 1  # NMI: the injected token is the record's first
        case(lits + st if lits else [irq, jeq(1)], "PSH", psh0)
        case(st + [jeq(1 + k)], "PSH", psh0)
        case(st + [jeq(2 + k)], "PSH", psh0)
        case(st + [jeq(3 + k)], "LO", val)
        case(st + [jeq(4 + k), tgt_rom], "JI", jtgt)
        case(st + [jeq(4 + k), 1 - tgt_rom], "HALT")

    fam_cases = [(lits, FAM_ID[fn]) for lits, fn, _ in cases]
    pay_cases = [(lits, pl) for lits, _, pl in cases if not isinstance(pl, int)]
    fam_bits = c.select_bits(fam_cases, NFAM, "exit.fam")
    pay_bits = c.select_bits(pay_cases, NPAY, "exit.pay")
    c.out_fam, c.out_pay = fam_bits, pay_bits
    exit_min = max(c.level(b) for b in fam_bits + pay_bits) - 1

    # the exit: clear the token's own code, write the predicted one (KAPPA * (2 bit - 1)); the
    # output head is tied to the embedding, whose code dims hold CODE_AMP * (2 bit - 1)
    c.code = [f(f"code{i}", lambda fm, p, i=i: _code_bit(fm, p, i)) for i in range(NFAM + NPAY)]
    for e in c.code:
        c.ffn([(-1, one, e), (1, one, -1 * e)], "code.clear", into=e)
    for i, e in enumerate(c.code):
        cs, bit = (fam_cases, i) if i < NFAM else (pay_cases, i - NFAM)
        terms = c.t_const(-KAPPA)
        for lits, v in cs:
            if isinstance(v, int):
                if v >> bit & 1:
                    terms += c.t_and(lits, 2 * KAPPA)
            elif bit < len(v) and v[bit].terms:
                terms += c.t_and([*lits, v[bit]], 2 * KAPPA)
        c.ffn(terms, f"exit.code{i}", into=e, min_level=exit_min)
    c.keep = [next(iter(e.terms)) for e in c.code]
    c.probe = dict(w=w, A=A, X=X, Y=Y, SP=SP, EA=EA, mem=mem, romd=romd, mb=mb, val=val,
                   jtgt=jtgt, pf=pf, op_is=op_is, B0=B0, B1=B1, B2=B2, j=jt, at_max=at_max,
                   pushdata=pushdata, cnew=cnew, vnew=vnew, px=px, reg=reg, tgt_rom=tgt_rom,
                   flags=dict(N=Nf, V=Vf, Z=Zf, C=Cf, I=If, D=Df), irq=irq, LO=LO, HI=HI)
    return c


def _updown_terms(c, x, up, down, k, lits, with_base=False):
    """Bit k of x + up - down (up, down: exclusive 0/1 literals), minus bit k of x itself
    unless with_base (then the whole bit)."""
    terms = c.t_and(lits + [x[k]]) if with_base and x[k].terms else []
    lo1 = list(x[:k])
    lo0 = [1 - b for b in x[:k]]
    # x + 1: bit flips where all lower bits are 1; x - 1: where all lower bits are 0
    terms += c.t_and(lits + [up] + lo1) + c.t_and(lits + [up] + lo1 + [x[k]], -2)
    terms += c.t_and(lits + [down] + lo0) + c.t_and(lits + [down] + lo0 + [x[k]], -2)
    return terms


def vlen(fm, p):
    """Vectorized spec.length_of: the instruction length a final token commits."""
    shift = {"A": 11, "X": 11, "Y": 11, "ST": 9, "RM": 10, "P": 7, "S": 8, "PSH": 8, "NT": 0,
             "IOW": 14}
    if fm in shift:
        return p >> shift[fm]
    return p * 0


def vfinal(fm, p):
    if fm in ("J", "JI", "R", "HALT", "GO"):
        return p * 0 + 1
    return (vlen(fm, p) > 0) * 1


def code_bits(fam: str, p: int) -> list[int]:
    fid = FAM_ID[fam]
    return [fid >> i & 1 for i in range(NFAM)] + [p >> i & 1 for i in range(NPAY)]


def _code_bit(fm, p, i):
    if fm is None:
        return p * 0
    if i < NFAM:
        return p * 0 + CODE_AMP * (2 * (FAM_ID[fm] >> i & 1) - 1)
    return CODE_AMP * (2 * (p >> (i - NFAM) & 1) - 1)


# --- snapshots (the state the host injects into the heads) --------------------------------------


def slot_row(bits: dict) -> np.ndarray:
    row = np.zeros(HD)
    row[list(REF_COLS)] = A_REF
    for col, b in bits.items():
        row[col] = A_BIT * b
    return row


def byte_bits(v: int, base: int = 2) -> dict:
    return {base + i: v >> i & 1 for i in range(8)}


def counter_rows(k: int) -> dict:
    r0 = np.zeros(HD)
    r0[2] = A_BIT * 2.0**-k
    r1 = np.zeros(HD)
    r1[list(REF_COLS)] = A_REF
    return {0: r0, 1: r1}


def byte_rows(words: np.ndarray, n_heads: int) -> list:
    """Qwenputer's byte-table layout: per key head, both value heads' rows of 4-byte words."""
    per = len(words) // n_heads
    out = []
    for h in range(n_heads):
        vh = []
        for x in range(2):
            blk = words[h * per + x * (per // 2):h * per + (x + 1) * (per // 2)]
            rows = np.zeros((128, HD))
            rows[:, list(REF_COLS)] = A_REF
            wl = np.arange(len(blk))
            pp, half, k = wl >> 4, wl >> 3 & 1, wl & 7
            rr = pp + 64 * half
            for b in range(4):
                rows[rr, 2 + 32 * half + 4 * k + b] = blk >> (8 * b) & 0xFF
            vh.append({i: rows[i] for i in range(128)})
        out.append(vh)
    return out


_ROM_CACHE: dict = {}


def rom_tables(prg: bytes) -> dict:
    key = hash(prg)
    if _ROM_CACHE.get("key") != key:
        p = np.frombuffer(prg, dtype=np.uint8).astype(np.int64)
        fetch = p | np.roll(p, -1) << 8 | np.roll(p, -2) << 16  # (op, b1, b2, 0) per address
        data = p[0::4] | p[1::4] << 8 | p[2::4] << 16 | p[3::4] << 24
        _ROM_CACHE.update(key=key, fetch=byte_rows(fetch, 16), data=byte_rows(data, 4))
    return _ROM_CACHE


def state_rows(cpu) -> dict:
    """The heads' initial state rows for a reference CPU's state (at a record boundary):
    table name -> one {row: vector} dict per value head."""
    out = {}
    out["pc"] = [{0: slot_row({2 + i: cpu.anchor >> i & 1 for i in range(15)})}]
    out["regAX"] = [{0: slot_row(byte_bits(cpu.A))}, {0: slot_row(byte_bits(cpu.X))}]
    out["regYS"] = [{0: slot_row(byte_bits(cpu.Y))}, {0: slot_row(byte_bits(cpu.SP))}]
    out["lat"] = [{0: slot_row({})}, {0: slot_row({})}]
    fl = cpu.flags
    out["flNZ"] = [{0: slot_row({2: fl["N"]})}, {0: slot_row({2: fl["Z"]})}]
    out["flCV"] = [{0: slot_row({2: fl["C"]})}, {0: slot_row({2: fl["V"]})}]
    out["flID"] = [{0: slot_row({2: fl["I"]})}, {0: slot_row({2: fl["D"]})}]
    out["irq"] = [{0: slot_row({})}]
    for i, nm in enumerate(("n1", "n2", "n3")):
        out[nm] = [counter_rows(cpu.n[i])]
    out["j"] = [counter_rows(0)]
    tabs = rom_tables(bytes(cpu.prg))
    for h in range(16):
        out[f"rom{h}"] = tabs["fetch"][h]
    for h in range(4):
        out[f"prg{h}"] = tabs["data"][h]
    for h in range(8):
        vhs = []
        for x in range(2):
            vhs.append({r: slot_row(byte_bits(cpu.ram[h << 8 | x << 7 | r])) for r in range(128)})
        out[f"ram{h}"] = vhs
    return out


def snapshot(c: NesCore, cpu) -> None:
    """Load a reference CPU's state into the circuit's heads (as at a record boundary)."""
    for name, vhs in state_rows(cpu).items():
        c.tables[name].init = vhs


def decode_code(fid: int, p: int) -> int:
    if not 0 <= fid < len(OUT_NAMES):
        return -1
    nm = OUT_NAMES[fid]
    if nm in S.CONTROLS:
        return S.tok(nm)
    return S.tok(nm, p) if p < S.FAMILIES[nm][1] else -1


def predictions(c: NesCore, mat: np.ndarray) -> list[int]:
    """The predicted next token at every position (from the exit's select dims)."""
    def bits(es):
        return [np.rint(sum(mat[:, d] * float(w_) for d, w_ in e.terms.items()))
                if e.terms else np.zeros(mat.shape[0]) for e in es]

    fb, pb = bits(c.out_fam), bits(c.out_pay)
    out = []
    for t in range(mat.shape[0]):
        fid = int(sum(int(fb[i][t]) << i for i in range(NFAM)))
        p = int(sum(int(pb[i][t]) << i for i in range(NPAY)))
        out.append(decode_code(fid, p))
    return out
