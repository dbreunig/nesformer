"""Tokens and records of the NES CPU trace.

The model emits one record per 6502 instruction; machine state changes only at a record's tokens
as listed below. The PC is implied, as in Qwenputer's RV32 core: PC = 0x8000 + anchor + n1 +
2 n2 + 3 n3, where the anchor is the payload of the latest J / JI / R token and n_L counts the
final tokens of length L since then. Before an instruction, when that offset has reached
OFF_MAX, an R token refreshes the anchor.

Token families (payload bit layout, LSB first):

  A X Y   final   value(8) C V PX LEN(2)   the register = value; N, Z from the value; C, V set
  ST      final   value(8) A7 LEN(2)      RAM[EA] = value (EA computed in-circuit, no flags;
                                          A7 = EA bit 7, which picks the RAM value head)
  RM      final   value(8) C A7 LEN(2)    read-modify-write: RAM[EA] = value; N, Z, C
  P       final   C Z I D V N PX LEN(2)   all six flags (LEN = 0: inside RTI, not final)
  S       final   value(8) LEN(2)         SP = value (TXS)
  NT      final   LEN(2)                  nothing changes (NOP, branch not taken)
  J       final   target(15)              anchor = target (PC = 0x8000 + target)
  JI      final   target(15)              the same, and I = 1 (the end of NMI / BRK)
  R               anchor(15)              anchor refresh; no instruction executes
  LO HI           byte(8)                 latch a byte (indirect pointers, RTS / RTI / JMP ind)
  PSH             sp(8) LEN(2)            RAM[0x100 + SP] = data (in-circuit), SP = sp; final
                                          when LEN > 0 (PHA / PHP)
  PUL             sp(8)                   SP = sp (the pulled byte is read at the new SP)
  IOR             reg(6)                  an I/O read; the host answers with IN
  IN              byte(8)                 injected by the host: the value read
  IOW     final   value(8) reg(6) LEN(2)  an I/O write (PPU, APU, controller registers); a
                                          write to $4014 has LEN 0 and starts OAM DMA:
  DM              byte(8) index(8)        byte `index` of the page; after the 256th, NT(LEN)
                                          ends the record
  NMI     control injected by the host before a record: the interrupt sequence follows
  HALT    control the machine stopped (an unsupported opcode or a jump outside the ROM)
  GO      control the first token of a run (after the state snapshot)

I/O register index: $2000-$3FFF -> 32 + (addr & 7); $4000-$401F -> addr & 31; $4020-$7FFF -> 40.
"""

FAMILY_SIZES = [
    ("A", 1 << 13), ("X", 1 << 13), ("Y", 1 << 13), ("ST", 1 << 11), ("RM", 1 << 12),
    ("P", 1 << 9), ("S", 1 << 10), ("NT", 4), ("J", 1 << 15), ("JI", 1 << 15), ("R", 1 << 15),
    ("LO", 256), ("HI", 256), ("PSH", 1 << 10), ("PUL", 256), ("IOR", 64), ("IN", 256),
    ("IOW", 1 << 16), ("DM", 1 << 16),
]
FAMILIES = {}
_base = 0
for _n, _s in FAMILY_SIZES:
    FAMILIES[_n] = (_base, _s)
    _base += _s
CTRL = _base
CONTROL_NAMES = ["NMI", "HALT", "GO"]
CONTROLS = {n: CTRL + i for i, n in enumerate(CONTROL_NAMES)}
VOCAB = CTRL + len(CONTROL_NAMES)
PAD = VOCAB  # the zero prefix token of a run (an all-zero embedding row)

OFF_MAX = 60  # refresh the anchor before an instruction once the offset reaches this
ROM_BASE = 0x8000
IO_OTHER = 40


def family(t: int) -> tuple[str, int]:
    if t >= CTRL:
        if t >= VOCAB:
            raise ValueError(f"not a token: {t}")
        return CONTROL_NAMES[t - CTRL], 0
    for name, (base, size) in FAMILIES.items():
        if base <= t < base + size:
            return name, t - base
    raise ValueError(f"not a token: {t}")


def tok(name: str, payload: int = 0) -> int:
    if name in CONTROLS:
        return CONTROLS[name]
    base, size = FAMILIES[name]
    assert 0 <= payload < size, (name, payload)
    return base + payload


def reg_index(ea: int) -> int:
    if 0x2000 <= ea < 0x4000:
        return 32 + (ea & 7)
    if 0x4000 <= ea < 0x4020:
        return ea & 31
    return IO_OTHER


# payload builders -------------------------------------------------------------------------

def t_reg(r: str, value: int, c: int, v: int, px: int, ln: int) -> int:
    return tok(r, value | c << 8 | v << 9 | px << 10 | ln << 11)


def t_p(flags: dict, px: int, ln: int) -> int:
    p = (flags["C"] | flags["Z"] << 1 | flags["I"] << 2 | flags["D"] << 3 | flags["V"] << 4
         | flags["N"] << 5)
    return tok("P", p | px << 6 | ln << 7)


def length_of(t: int) -> int:
    """The instruction length a final token commits (0: not a final instruction token)."""
    fam, p = family(t)
    if fam in ("A", "X", "Y"):
        return p >> 11
    if fam in ("S", "PSH"):
        return p >> 8
    if fam == "ST":
        return p >> 9
    if fam == "RM":
        return p >> 10
    if fam == "P":
        return p >> 7
    if fam == "NT":
        return p
    if fam == "IOW":
        return p >> 14
    return 0


def is_final(t: int) -> bool:
    """The token ends a record (the next token starts the next record)."""
    fam = family(t)[0]
    return fam in ("J", "JI", "R", "HALT", "GO") or length_of(t) > 0


def is_anchor(t: int) -> bool:
    return family(t)[0] in ("J", "JI", "R")


def name(t: int) -> str:
    fam, p = family(t)
    if fam in CONTROLS:
        return fam
    if fam in ("A", "X", "Y"):
        return f"{fam}(${p & 255:02X} c{p >> 8 & 1} v{p >> 9 & 1} px{p >> 10 & 1} L{p >> 11})"
    if fam == "P":
        return f"P({p & 63:06b} px{p >> 6 & 1} L{p >> 7})"
    if fam in ("J", "JI", "R"):
        return f"{fam}(${0x8000 + p:04X})"
    if fam == "IOW":
        r = p >> 8 & 63
        reg = f"${0x2000 + r - 32:04X}" if 32 <= r < 40 else f"${0x4000 + r:04X}" if r < 32 \
            else "$4020+"
        return f"IOW({reg}=${p & 255:02X} L{p >> 14})"
    if fam == "DM":
        return f"DM({p >> 8}:${p & 255:02X})"
    if fam == "ST":
        return f"ST(${p & 255:02X} a7={p >> 8 & 1} L{p >> 9})"
    if fam == "RM":
        return f"RM(${p & 255:02X} c{p >> 8 & 1} a7={p >> 9 & 1} L{p >> 10})"
    if fam in ("S", "PSH"):
        return f"{fam}(${p & 255:02X}" + (f" L{p >> 8})" if p >> 8 else ")")
    if fam in ("PUL", "LO", "HI", "IN"):
        return f"{fam}(${p:02X})"
    if fam == "NT":
        return f"NT(L{p})"
    if fam == "IOR":
        reg = f"${0x2000 + p - 32:04X}" if 32 <= p < 40 else f"${0x4000 + p:04X}" if p < 32 \
            else "$4020+"
        return f"IOR({reg})"
    return f"{fam}({p})"
