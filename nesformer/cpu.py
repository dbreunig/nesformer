"""The reference 6502 (2A03: no decimal mode) that writes the trace the circuit must reproduce.

``Cpu.record()`` is a generator: it yields the tokens of one record. When it yields an IOR token
the driver sends back the byte the host read (and the driver logs the IN token); every other
yield gets None. The state the trace commits (registers, flags, RAM, the PC anchor and length
counters) lives here; the PRG-ROM is read-only.
"""

from dataclasses import dataclass, field

from nesformer import spec as S

# mnemonic, mode, base cycles; modes: imp acc imm zp zpx zpy abs abx aby inx iny rel ind
OPS: dict[int, tuple[str, str, int]] = {}


def _ops(mn, *pairs):
    for op, mode, cyc in pairs:
        OPS[op] = (mn, mode, cyc)


_READ = [("imm", 2), ("zp", 3), ("zpx", 4), ("abs", 4), ("abx", 4), ("aby", 4), ("inx", 6),
         ("iny", 5)]
for _mn, _base in (("ORA", 0x00), ("AND", 0x20), ("EOR", 0x40), ("ADC", 0x60), ("LDA", 0xA0),
                   ("CMP", 0xC0), ("SBC", 0xE0)):
    _ops(_mn, *[(_base + off, m, c) for off, (m, c) in zip(
        (0x09, 0x05, 0x15, 0x0D, 0x1D, 0x19, 0x01, 0x11), _READ)])
_ops("STA", (0x85, "zp", 3), (0x95, "zpx", 4), (0x8D, "abs", 4), (0x9D, "abx", 5),
     (0x99, "aby", 5), (0x81, "inx", 6), (0x91, "iny", 6))
for _mn, _base in (("ASL", 0x00), ("ROL", 0x20), ("LSR", 0x40), ("ROR", 0x60)):
    _ops(_mn, (_base + 0x0A, "acc", 2), (_base + 0x06, "zp", 5), (_base + 0x16, "zpx", 6),
         (_base + 0x0E, "abs", 6), (_base + 0x1E, "abx", 7))
_ops("INC", (0xE6, "zp", 5), (0xF6, "zpx", 6), (0xEE, "abs", 6), (0xFE, "abx", 7))
_ops("DEC", (0xC6, "zp", 5), (0xD6, "zpx", 6), (0xCE, "abs", 6), (0xDE, "abx", 7))
_ops("LDX", (0xA2, "imm", 2), (0xA6, "zp", 3), (0xB6, "zpy", 4), (0xAE, "abs", 4),
     (0xBE, "aby", 4))
_ops("LDY", (0xA0, "imm", 2), (0xA4, "zp", 3), (0xB4, "zpx", 4), (0xAC, "abs", 4),
     (0xBC, "abx", 4))
_ops("STX", (0x86, "zp", 3), (0x96, "zpy", 4), (0x8E, "abs", 4))
_ops("STY", (0x84, "zp", 3), (0x94, "zpx", 4), (0x8C, "abs", 4))
_ops("CPX", (0xE0, "imm", 2), (0xE4, "zp", 3), (0xEC, "abs", 4))
_ops("CPY", (0xC0, "imm", 2), (0xC4, "zp", 3), (0xCC, "abs", 4))
_ops("BIT", (0x24, "zp", 3), (0x2C, "abs", 4))
for _mn, _op in (("BPL", 0x10), ("BMI", 0x30), ("BVC", 0x50), ("BVS", 0x70), ("BCC", 0x90),
                 ("BCS", 0xB0), ("BNE", 0xD0), ("BEQ", 0xF0)):
    _ops(_mn, (_op, "rel", 2))
_ops("JMP", (0x4C, "abs", 3), (0x6C, "ind", 5))
_ops("JSR", (0x20, "abs", 6))
for _mn, _op, _c in (("BRK", 0x00, 7), ("RTI", 0x40, 6), ("RTS", 0x60, 6), ("PHP", 0x08, 3),
                     ("PLP", 0x28, 4), ("PHA", 0x48, 3), ("PLA", 0x68, 4), ("CLC", 0x18, 2),
                     ("SEC", 0x38, 2), ("CLI", 0x58, 2), ("SEI", 0x78, 2), ("CLV", 0xB8, 2),
                     ("CLD", 0xD8, 2), ("SED", 0xF8, 2), ("TXA", 0x8A, 2), ("TYA", 0x98, 2),
                     ("TXS", 0x9A, 2), ("TAY", 0xA8, 2), ("TAX", 0xAA, 2), ("TSX", 0xBA, 2),
                     ("DEY", 0x88, 2), ("INY", 0xC8, 2), ("DEX", 0xCA, 2), ("INX", 0xE8, 2),
                     ("NOP", 0xEA, 2)):
    _ops(_mn, (_op, "imp", _c))

LENGTH = {"imp": 1, "acc": 1, "imm": 2, "zp": 2, "zpx": 2, "zpy": 2, "rel": 2, "inx": 2,
          "iny": 2, "abs": 3, "abx": 3, "aby": 3, "ind": 3}
PAGE_PENALTY = {"ORA", "AND", "EOR", "ADC", "LDA", "CMP", "SBC", "LDX", "LDY"}
BRANCH_FLAG = {"BPL": ("N", 0), "BMI": ("N", 1), "BVC": ("V", 0), "BVS": ("V", 1),
               "BCC": ("C", 0), "BCS": ("C", 1), "BNE": ("Z", 0), "BEQ": ("Z", 1)}
RMW = {"ASL", "ROL", "LSR", "ROR", "INC", "DEC"}
STORES = {"STA": "A", "STX": "X", "STY": "Y"}
LOADS = {"LDA": "A", "LDX": "X", "LDY": "Y"}
COMPARES = {"CMP": "A", "CPX": "X", "CPY": "Y"}
FLAG_OPS = {"CLC": ("C", 0), "SEC": ("C", 1), "CLI": ("I", 0), "SEI": ("I", 1),
            "CLV": ("V", 0), "CLD": ("D", 0), "SED": ("D", 1)}
TRANSFERS = {"TAX": ("A", "X"), "TAY": ("A", "Y"), "TXA": ("X", "A"), "TYA": ("Y", "A"),
             "TSX": ("SP", "X")}
INCDEC_REG = {"INX": ("X", 1), "INY": ("Y", 1), "DEX": ("X", -1), "DEY": ("Y", -1)}


class Halt(Exception):
    pass


@dataclass
class Cpu:
    prg: bytes  # 32 KB, mapped at $8000
    A: int = 0
    X: int = 0
    Y: int = 0
    SP: int = 0xFD
    flags: dict = field(default_factory=lambda: dict(N=0, V=0, D=0, I=1, Z=0, C=0))
    anchor: int = 0
    n: list = field(default_factory=lambda: [0, 0, 0])  # final tokens of length 1, 2, 3
    ram: bytearray = field(default_factory=lambda: bytearray(2048))
    halted: bool = False
    steps: int = 0

    @classmethod
    def boot(cls, prg: bytes, pc: int | None = None) -> "Cpu":
        assert len(prg) == 0x8000
        c = cls(prg)
        if pc is None:
            pc = prg[0x7FFC] | prg[0x7FFD] << 8
        assert pc >= 0x8000, f"reset vector {pc:04X} outside the ROM"
        c.anchor = pc - 0x8000
        return c

    # --- state helpers -----------------------------------------------------------------------
    @property
    def offset(self) -> int:
        return self.n[0] + 2 * self.n[1] + 3 * self.n[2]

    @property
    def pc(self) -> int:
        return 0x8000 | ((self.anchor + self.offset) & 0x7FFF)

    def p_byte(self, b: int) -> int:
        f = self.flags
        return (f["N"] << 7 | f["V"] << 6 | 1 << 5 | b << 4 | f["D"] << 3 | f["I"] << 2
                | f["Z"] << 1 | f["C"])

    def commit(self, ln: int) -> None:
        self.n[ln - 1] += 1

    def jump(self, target: int) -> int:
        self.anchor, self.n = target - 0x8000, [0, 0, 0]
        return target - 0x8000

    def rom(self, a: int) -> int:
        return self.prg[a & 0x7FFF]

    # --- the data port (one byte per position) -------------------------------------------------
    def read(self, ea: int):
        """Generator: the byte at ea; an I/O address yields IOR and receives the host's byte."""
        if ea < 0x2000:
            return self.ram[ea & 0x7FF]
        if ea >= 0x8000:
            return self.prg[ea & 0x7FFF]
        v = yield S.tok("IOR", S.reg_index(ea))
        assert v is not None and 0 <= v < 256, "the driver must send the IN byte"
        return v

    # --- one record ------------------------------------------------------------------------
    def record(self):
        if self.halted:
            return
        if self.offset >= S.OFF_MAX:
            w = (self.anchor + self.offset) & 0x7FFF
            self.jump(0x8000 + w)
            yield S.tok("R", w)
            return
        pc = self.pc
        op, b1, b2 = self.rom(pc), self.rom(pc + 1), self.rom(pc + 2)
        if op not in OPS:
            self.halted = True
            yield S.tok("HALT")
            return
        mn, mode, _ = OPS[op]
        ln = LENGTH[mode]
        f = self.flags
        self.steps += 1

        # effective address of the data access (and the pointer prefix of indirect modes)
        px = 0
        ea = None
        if mode == "imm":
            ea = (pc + 1) & 0xFFFF
        elif mode == "zp":
            ea = b1
        elif mode == "zpx":
            ea = (b1 + self.X) & 0xFF
        elif mode == "zpy":
            ea = (b1 + self.Y) & 0xFF
        elif mode == "abs":
            ea = b1 | b2 << 8
        elif mode in ("abx", "aby"):
            r = self.X if mode == "abx" else self.Y
            ea = ((b1 | b2 << 8) + r) & 0xFFFF
            px = int(b1 + r > 0xFF)
        elif mode == "inx":
            lo = self.ram[(b1 + self.X) & 0xFF]
            yield S.tok("LO", lo)
            hi = self.ram[(b1 + self.X + 1) & 0xFF]
            yield S.tok("HI", hi)
            ea = lo | hi << 8
        elif mode == "iny":
            lo = self.ram[b1]
            yield S.tok("LO", lo)
            hi = self.ram[(b1 + 1) & 0xFF]
            yield S.tok("HI", hi)
            ea = ((lo | hi << 8) + self.Y) & 0xFFFF
            px = int(lo + self.Y > 0xFF)
        if mn not in PAGE_PENALTY:
            px = 0

        def reg_tok(r, value, c=None, v=None):
            self.commit(ln)
            if r == "SP":
                self.SP = value
            else:
                setattr(self, r, value)
                f["N"], f["Z"] = value >> 7, int(value == 0)
            if c is not None:
                f["C"] = c
            if v is not None:
                f["V"] = v
            return S.t_reg(r, value, f["C"], f["V"], px, ln)

        def p_tok(ln_=ln):
            if ln_:
                self.commit(ln_)
            return S.t_p(f, px, ln_)

        # --- reads ---------------------------------------------------------------------------
        if mn in LOADS or mn in ("ADC", "SBC", "AND", "ORA", "EOR") or mn in COMPARES \
                or mn == "BIT":
            m = yield from self.read(ea)
            if mn in LOADS:
                yield reg_tok(LOADS[mn], m)
            elif mn in ("ADC", "SBC"):
                mm = m if mn == "ADC" else m ^ 0xFF
                s = self.A + mm + f["C"]
                v = int(((self.A ^ s) & (mm ^ s) & 0x80) != 0)
                yield reg_tok("A", s & 0xFF, int(s > 0xFF), v)
            elif mn in ("AND", "ORA", "EOR"):
                r = {"AND": self.A & m, "ORA": self.A | m, "EOR": self.A ^ m}[mn]
                yield reg_tok("A", r)
            elif mn in COMPARES:
                rv = getattr(self, COMPARES[mn])
                t = (rv - m) & 0xFF
                f["C"], f["Z"], f["N"] = int(rv >= m), int(rv == m), t >> 7
                yield p_tok()
            else:  # BIT
                f["Z"], f["N"], f["V"] = int(self.A & m == 0), m >> 7, m >> 6 & 1
                yield p_tok()
            return

        # --- stores ----------------------------------------------------------------------------
        if mn in STORES:
            v = getattr(self, STORES[mn])
            if ea == 0x4014:  # OAM DMA: the register write, the page's 256 bytes, NT
                yield S.tok("IOW", v | 0x14 << 8)
                for i in range(256):
                    a = v << 8 | i
                    byte = self.ram[a & 0x7FF] if a < 0x2000 else self.prg[a & 0x7FFF] \
                        if a >= 0x8000 else 0
                    yield S.tok("DM", byte | i << 8)
                self.commit(ln)
                yield S.tok("NT", ln)
                return
            if 0x2000 <= ea < 0x8000:
                self.commit(ln)
                yield S.tok("IOW", v | S.reg_index(ea) << 8 | ln << 14)
                return
            if ea < 0x2000:
                self.ram[ea & 0x7FF] = v
            self.commit(ln)
            yield S.tok("ST", v | (ea >> 7 & 1) << 8 | ln << 9)
            return

        # --- read-modify-write -----------------------------------------------------------------
        if mn in RMW:
            m = self.A if mode == "acc" else (yield from self.read(ea))
            c = f["C"]
            if mn == "ASL":
                r, c = (m << 1) & 0xFF, m >> 7
            elif mn == "LSR":
                r, c = m >> 1, m & 1
            elif mn == "ROL":
                r, c = ((m << 1) | c) & 0xFF, m >> 7
            elif mn == "ROR":
                r, c = (m >> 1) | c << 7, m & 1
            elif mn == "INC":
                r = (m + 1) & 0xFF
            else:
                r = (m - 1) & 0xFF
            if mode == "acc":
                yield reg_tok("A", r, c)
                return
            if ea < 0x2000:
                self.ram[ea & 0x7FF] = r
            f["N"], f["Z"], f["C"] = r >> 7, int(r == 0), c
            self.commit(ln)
            yield S.tok("RM", r | c << 8 | (ea >> 7 & 1) << 9 | ln << 10)
            return

        # --- control flow ----------------------------------------------------------------------
        if mn in BRANCH_FLAG:
            fl, want = BRANCH_FLAG[mn]
            if f[fl] == want:
                target = (pc + 2 + (b1 - 256 if b1 & 0x80 else b1)) & 0xFFFF
                if target < 0x8000:
                    self.halted = True
                    yield S.tok("HALT")
                    return
                yield S.tok("J", self.jump(target))
            else:
                self.commit(ln)
                yield S.tok("NT", ln)
            return
        if mn == "JMP":
            if mode == "abs":
                target = b1 | b2 << 8
            else:
                ptr = b1 | b2 << 8
                if 0x2000 <= ptr < 0x8000:  # (an I/O pointer is not supported)
                    self.halted = True
                    yield S.tok("HALT")
                    return
                lo = yield from self.read(ptr)
                yield S.tok("LO", lo)
                hi = yield from self.read((ptr & 0xFF00) | ((ptr + 1) & 0xFF))
                target = lo | hi << 8
            if target < 0x8000:
                self.halted = True
                yield S.tok("HALT")
                return
            yield S.tok("J", self.jump(target))
            return
        if mn == "JSR":
            target = b1 | b2 << 8
            if target < 0x8000:
                self.halted = True
                yield S.tok("HALT")
                return
            ret = (pc + 2) & 0xFFFF
            yield self.push(ret >> 8, 0)
            yield self.push(ret & 0xFF, 0)
            yield S.tok("J", self.jump(target))
            return
        if mn in ("RTS", "RTI"):
            if mn == "RTI":
                p = yield self.pull()
                self.set_p(self.ram[0x100 | self.SP])
                yield S.t_p(f, 0, 0)
            yield self.pull()
            lo = self.ram[0x100 | self.SP]
            yield S.tok("LO", lo)
            yield self.pull()
            hi = self.ram[0x100 | self.SP]
            target = (lo | hi << 8) + (1 if mn == "RTS" else 0)
            target &= 0xFFFF
            if target < 0x8000:
                self.halted = True
                yield S.tok("HALT")
                return
            yield S.tok("J", self.jump(target))
            return
        if mn == "BRK":
            ret = (pc + 2) & 0xFFFF
            yield from self.interrupt(ret, 0xFFFE, 1)
            return
        if mn in ("PHA", "PHP"):
            data = self.A if mn == "PHA" else self.p_byte(1)
            self.commit(ln)
            yield self.push(data, ln)
            return
        if mn in ("PLA", "PLP"):
            yield self.pull()
            m = self.ram[0x100 | self.SP]
            if mn == "PLA":
                yield reg_tok("A", m)
            else:
                self.set_p(m)
                yield p_tok()
            return
        if mn in FLAG_OPS:
            fl, val = FLAG_OPS[mn]
            f[fl] = val
            yield p_tok()
            return
        if mn in TRANSFERS:
            src, dst = TRANSFERS[mn]
            yield reg_tok(dst, getattr(self, src))
            return
        if mn == "TXS":
            self.SP = self.X
            self.commit(ln)
            yield S.tok("S", self.X | ln << 8)
            return
        if mn in INCDEC_REG:
            r, d = INCDEC_REG[mn]
            yield reg_tok(r, (getattr(self, r) + d) & 0xFF)
            return
        if mn == "NOP":
            self.commit(ln)
            yield S.tok("NT", ln)
            return
        raise AssertionError(mn)

    def push(self, data: int, ln: int) -> int:
        self.ram[0x100 | self.SP] = data
        self.SP = (self.SP - 1) & 0xFF
        return S.tok("PSH", self.SP | ln << 8)

    def pull(self) -> int:
        self.SP = (self.SP + 1) & 0xFF
        return S.tok("PUL", self.SP)

    def set_p(self, m: int) -> None:
        f = self.flags
        f["N"], f["V"], f["D"], f["I"], f["Z"], f["C"] = (m >> 7, m >> 6 & 1, m >> 3 & 1,
                                                          m >> 2 & 1, m >> 1 & 1, m & 1)

    def interrupt(self, ret: int, vector: int, b: int):
        yield self.push(ret >> 8, 0)
        yield self.push(ret & 0xFF, 0)
        yield self.push(self.p_byte(b), 0)
        lo = self.rom(vector)
        yield S.tok("LO", lo)
        target = lo | self.rom(vector + 1) << 8
        if target < 0x8000:
            self.halted = True
            yield S.tok("HALT")
            return
        self.flags["I"] = 1
        yield S.tok("JI", self.jump(target))

    def nmi_record(self):
        """The record after an injected NMI token."""
        yield from self.interrupt(self.pc, 0xFFFA, 0)
