"""The host: everything on the NES outside the CPU core and its RAM.

The host reads the trace token by token (``feed``), tracks the PC and the CPU cycle count from
it, runs a scanline-accurate PPU, answers I/O reads (``io_read``), and asks for an NMI at vblank
(``wants_nmi``). It never sees the CPU's registers or RAM except through the trace: OAM DMA
bytes arrive as DM tokens. Rendering is per scanline (mid-line effects are approximated).
"""

from dataclasses import dataclass, field

from nesformer import spec as S
from nesformer.cpu import OPS

DOTS_PER_LINE = 341
LINES = 262
FRAME_DOTS = DOTS_PER_LINE * LINES

# The 2C02 palette (RGB), as commonly used by emulators.
PALETTE = [
    (84, 84, 84), (0, 30, 116), (8, 16, 144), (48, 0, 136), (68, 0, 100), (92, 0, 48),
    (84, 4, 0), (60, 24, 0), (32, 42, 0), (8, 58, 0), (0, 64, 0), (0, 60, 0), (0, 50, 60),
    (0, 0, 0), (0, 0, 0), (0, 0, 0),
    (152, 150, 152), (8, 76, 196), (48, 50, 236), (92, 30, 228), (136, 20, 176),
    (160, 20, 100), (152, 34, 32), (120, 60, 0), (84, 90, 0), (40, 114, 0), (8, 124, 0),
    (0, 118, 40), (0, 102, 120), (0, 0, 0), (0, 0, 0), (0, 0, 0),
    (236, 238, 236), (76, 154, 236), (120, 124, 236), (176, 98, 236), (228, 84, 236),
    (236, 88, 180), (236, 106, 100), (212, 136, 32), (160, 170, 0), (116, 196, 0),
    (76, 208, 32), (56, 204, 108), (56, 180, 204), (60, 60, 60), (0, 0, 0), (0, 0, 0),
    (236, 238, 236), (168, 204, 236), (188, 188, 236), (212, 178, 236), (236, 174, 236),
    (236, 174, 212), (236, 180, 176), (228, 196, 144), (204, 210, 120), (180, 222, 120),
    (168, 226, 144), (152, 226, 180), (160, 214, 228), (160, 162, 160), (0, 0, 0), (0, 0, 0),
]

# per-frame PPU events: (dot within the frame, action, scanline)
EVENTS = ([(line * DOTS_PER_LINE + 256, "render", line) for line in range(240)]
          + [(241 * DOTS_PER_LINE + 1, "vblank", 0), (261 * DOTS_PER_LINE + 1, "prerender", 0),
             (FRAME_DOTS, "frame", 0)])

BUTTONS = {"a": 0x01, "b": 0x02, "select": 0x04, "start": 0x08, "up": 0x10, "down": 0x20,
           "left": 0x40, "right": 0x80}


@dataclass
class Cart:
    prg: bytes  # 32 KB (a 16 KB PRG is mirrored)
    chr: bytearray  # 8 KB (CHR-RAM when the cart has none)
    vertical: bool  # nametable mirroring
    chr_ram: bool

    @classmethod
    def load(cls, data: bytes) -> "Cart":
        if data[:4] != b"NES\x1a":
            raise ValueError("not an iNES file")
        nprg, nchr, f6, f7 = data[4], data[5], data[6], data[7]
        mapper = (f6 >> 4) | (f7 & 0xF0)
        if mapper != 0:
            raise ValueError(f"mapper {mapper} is not supported (NROM / mapper 0 only)")
        if nprg not in (1, 2):
            raise ValueError(f"{nprg} PRG banks: NROM has 1 or 2")
        off = 16 + (512 if f6 & 4 else 0)
        prg = data[off:off + 16384 * nprg]
        if nprg == 1:
            prg = prg * 2
        chr_ = data[off + 16384 * nprg:off + 16384 * nprg + 8192 * nchr]
        return cls(bytes(prg), bytearray(chr_ if nchr else bytes(8192)), bool(f6 & 1), nchr == 0)


@dataclass
class Host:
    cart: Cart
    buttons: object = None  # frame -> button mask (int), or None
    cycles: int = 7  # CPU cycles since power-on (the reset sequence takes 7)
    pc: int = 0
    frame: int = 0
    frames: list = field(default_factory=list)  # finished frames (lists of 240 rows of indices)
    keep_frames: bool = True
    skip: bool = True  # fast-forward identical wait loops (see maybe_skip)

    def __post_init__(self):
        c = self.cart
        self.pc = c.prg[0x7FFC] | c.prg[0x7FFD] << 8
        self.vram = bytearray(2048)
        self.pal = bytearray(32)
        self.oam = bytearray(256)
        self.ctrl = self.mask = self.status = self.oamaddr = 0
        self.v = self.t = self.x = self.w = 0
        self.buf = 0
        self.frame_base = 0  # absolute PPU dot where the current frame began
        self.ev_i = 0  # the next event of this frame (EVENTS)
        self.nmi_pending = False
        self.joy = self.shift = self.strobe = 0
        self.rec_start = True  # the next token starts a record
        self.rec_cycles = 0
        self.rec_op = None
        self.in_nmi = False
        self.fb = [bytearray(256) for _ in range(240)]
        self.halted = False
        self.io_log = []
        self.iter_toks = []  # tokens since the latest J
        self.prev_body = None
        self.prev_target = None
        self.iter_cycle0 = 0
        self.skipped_cycles = 0
        self.skipped_iters = 0
        self.ev_seq = 0  # PPU events processed so far
        self.iter_ev0 = 0

    # --- trace --------------------------------------------------------------------------------
    def feed(self, t: int) -> None:
        fam, p = S.family(t)
        if fam == "GO":
            return
        self.iter_toks.append(t)
        if self.rec_start:
            self.rec_start = False
            self.rec_dma = False
            self.rec_cycles = self.cycles
            self.in_nmi = fam == "NMI"
            self.rec_op = None if fam in ("NMI", "R") else self.cart.prg[self.pc & 0x7FFF]
        if fam == "IOW":
            self.io_write(p >> 8 & 63, p & 255)
            self.rec_dma |= (p >> 8 & 63) == 0x14
        elif fam == "DM":
            self.oam[(self.oamaddr + (p >> 8)) & 255] = p & 255
        elif fam == "HALT":
            self.halted = True
        if S.is_final(t):
            self.finish(t, fam, p)

    def finish(self, t, fam, p):
        self.rec_start = True
        if fam == "R":
            self.pc = 0x8000 + p
            return
        if fam == "HALT":
            return
        if self.in_nmi:
            cyc = 7
        else:
            mn, mode, cyc = OPS[self.rec_op]
            if fam in ("A", "X", "Y"):
                cyc += p >> 10 & 1
            elif fam == "P":
                cyc += p >> 6 & 1
            if self.rec_dma:
                cyc += 513
            if mode == "rel" and fam == "J":
                cyc += 1 + int(((self.pc + 2) & 0xFF00) != ((0x8000 + p) & 0xFF00))
        if fam in ("J", "JI"):
            self.pc = 0x8000 + p
        else:
            self.pc = (self.pc + S.length_of(t)) & 0xFFFF
        self.cycles += cyc
        self.catch_up(self.cycles * 3)
        if fam == "J" and not self.in_nmi:
            self.maybe_skip(p)
        elif fam in ("JI", "R"):
            self.prev_body = None
            self.iter_toks = []
            self.iter_cycle0 = self.cycles
            self.iter_ev0 = self.ev_seq

    SKIP_SAFE = {"P", "A", "X", "Y", "NT", "J", "LO", "HI", "IOR", "IN"}

    def maybe_skip(self, target: int) -> None:
        """At a J: if the loop body since the previous J (same target) repeated exactly and
        changes nothing but registers (no RAM or I/O writes; I/O reads only of $2002), every
        further iteration is identical until the PPU next changes state. Advance the clock by
        whole iterations to just before that, so the iterations still run (the trace is the
        full trace with identical iterations removed)."""
        body = tuple(self.iter_toks)
        period = self.cycles - self.iter_cycle0
        same = body == self.prev_body and target == self.prev_target
        quiet = self.ev_seq == self.iter_ev0 and not self.nmi_pending  # no PPU event meanwhile
        self.prev_body, self.prev_target = body, target
        self.iter_toks = []
        self.iter_cycle0 = self.cycles
        self.iter_ev0 = self.ev_seq
        if not (self.skip and same and quiet and period > 0):
            return
        reads = [S.family(t)[1] for t in body if S.family(t)[0] == "IOR"]
        if any(S.family(t)[0] not in self.SKIP_SAFE for t in body) or any(r != 34 for r in reads):
            return
        if reads:  # polling $2002: until the next PPU event
            ev_dot = self.frame_base + EVENTS[self.ev_i][0]
        elif self.ctrl & 0x80:  # waiting for the NMI
            ev_dot = self.frame_base + 241 * DOTS_PER_LINE + 1
            if self.ev_i > 240:
                ev_dot += FRAME_DOTS
        else:
            return
        k = (-(-ev_dot // 3) - self.cycles) // period - 1
        if k > 0:
            self.cycles += k * period
            self.iter_cycle0 = self.cycles
            self.skipped_cycles += k * period
            self.skipped_iters += k
            self.catch_up(self.cycles * 3)
            self.iter_ev0 = self.ev_seq

    def wants_nmi(self) -> bool:
        """At a record boundary: inject an NMI now?"""
        if self.nmi_pending and self.rec_start and not self.halted:
            self.nmi_pending = False
            return True
        return False

    def access_cycle(self) -> int:
        """The cycle of the current instruction's bus access (its last cycle)."""
        if self.rec_op is None:
            return self.rec_cycles
        return self.rec_cycles + OPS[self.rec_op][2] - 1

    # --- I/O ----------------------------------------------------------------------------------
    def io_read(self, reg: int) -> int:
        self.catch_up(self.access_cycle() * 3)
        if reg >= 32 and reg < 40:
            r = reg - 32
            if r == 2:
                v = self.status | (self.buf & 0x1F)
                self.status &= 0x7F
                self.w = 0
                return v
            if r == 4:
                return self.oam[self.oamaddr]
            if r == 7:
                a = self.v & 0x3FFF
                v = self.buf
                self.buf = self.vread(a)
                if a >= 0x3F00:
                    v = self.pal_read(a)
                    self.buf = self.vram[self.nt_index(a - 0x1000)]
                self.v = (self.v + (32 if self.ctrl & 4 else 1)) & 0x7FFF
                return v
            return self.buf & 0xFF
        if reg == 0x16:
            bit = self.shift & 1
            if not self.strobe:
                self.shift = (self.shift >> 1) | 0x80
            return 0x40 | bit
        if reg == 0x17:
            return 0x40
        return 0

    def io_write(self, reg: int, v: int) -> None:
        self.catch_up(self.access_cycle() * 3)
        self.io_log.append((self.cycles, reg, v))
        if 32 <= reg < 40:
            r = reg - 32
            if r == 0:
                if v & 0x80 and not self.ctrl & 0x80 and self.status & 0x80:
                    self.nmi_pending = True
                self.ctrl = v
                self.t = (self.t & ~0x0C00) | (v & 3) << 10
            elif r == 1:
                self.mask = v
            elif r == 3:
                self.oamaddr = v
            elif r == 4:
                self.oam[self.oamaddr] = v
                self.oamaddr = (self.oamaddr + 1) & 255
            elif r == 5:
                if self.w == 0:
                    self.t = (self.t & ~0x1F) | v >> 3
                    self.x = v & 7
                else:
                    self.t = (self.t & ~0x73E0) | (v & 7) << 12 | (v & 0xF8) << 2
                self.w ^= 1
            elif r == 6:
                if self.w == 0:
                    self.t = (self.t & 0x00FF) | (v & 0x3F) << 8
                else:
                    self.t = (self.t & 0x7F00) | v
                    self.v = self.t
                self.w ^= 1
            elif r == 7:
                self.vwrite(self.v & 0x3FFF, v)
                self.v = (self.v + (32 if self.ctrl & 4 else 1)) & 0x7FFF
        elif reg == 0x16:
            self.strobe = v & 1
            if self.strobe:
                self.shift = self.current_buttons()
        elif reg == 0x14:
            pass  # the DM tokens already filled OAM

    def current_buttons(self) -> int:
        if self.buttons is None:
            return 0
        return self.buttons(self.frame) if callable(self.buttons) else int(self.buttons)

    # --- VRAM -----------------------------------------------------------------------------------
    def nt_index(self, a: int) -> int:
        a &= 0x0FFF
        if self.cart.vertical:
            return a & 0x07FF
        return (a & 0x03FF) | (a & 0x0800) >> 1

    def pal_index(self, a: int) -> int:
        a &= 0x1F
        if a in (0x10, 0x14, 0x18, 0x1C):
            a -= 0x10
        return a

    def pal_read(self, a: int) -> int:
        return self.pal[self.pal_index(a)]

    def vread(self, a: int) -> int:
        a &= 0x3FFF
        if a < 0x2000:
            return self.cart.chr[a]
        if a < 0x3F00:
            return self.vram[self.nt_index(a)]
        return self.pal_read(a)

    def vwrite(self, a: int, v: int) -> None:
        a &= 0x3FFF
        if a < 0x2000:
            if self.cart.chr_ram:
                self.cart.chr[a] = v
        elif a < 0x3F00:
            self.vram[self.nt_index(a)] = v
        else:
            self.pal[self.pal_index(a)] = v & 0x3F

    # --- PPU timing ---------------------------------------------------------------------------
    def rendering(self) -> bool:
        return bool(self.mask & 0x18)

    def catch_up(self, target: int) -> None:
        """Run the PPU's per-scanline events up to absolute dot ``target``."""
        while True:
            d, act, line = EVENTS[self.ev_i]
            if self.frame_base + d > target:
                return
            if act == "render":
                self.render_line(line)
            elif act == "vblank":
                self.status |= 0x80
                if self.ctrl & 0x80:
                    self.nmi_pending = True
            elif act == "prerender":
                self.status &= 0x1F
                if self.rendering():  # dots 280-304: vertical scroll bits from t
                    self.v = (self.v & ~0x7BE0) | (self.t & 0x7BE0)
            self.ev_i += 1
            self.ev_seq += 1
            if act == "frame":
                self.end_frame()
                self.frame_base += FRAME_DOTS
                self.ev_i = 0

    def end_frame(self) -> None:
        if self.keep_frames:
            self.frames.append([bytes(r) for r in self.fb])
        self.frame += 1

    # --- rendering ----------------------------------------------------------------------------
    def render_line(self, line: int) -> None:
        row = self.fb[line]
        pal = self.pal
        chr_ = self.cart.chr
        if not self.rendering():
            bgc = pal[0] & 0x3F
            for i in range(256):
                row[i] = bgc
            return
        show_bg, show_sp = self.mask & 0x08, self.mask & 0x10
        bg = [0] * 256  # 2-bit pattern value
        bgp = [0] * 256
        if show_bg:
            v = self.v
            fine_y = v >> 12 & 7
            base = (self.ctrl & 0x10) << 8
            px = -self.x
            for _ in range(33):
                nt = 0x2000 | (v & 0x0FFF)
                tile = self.vram[self.nt_index(nt)]
                at = self.vram[self.nt_index(0x23C0 | (v & 0x0C00) | (v >> 4 & 0x38) | (v >> 2 & 7))]
                shift = (v >> 4 & 4) | (v & 2)
                palno = at >> shift & 3
                a = base + tile * 16 + fine_y
                lo, hi = chr_[a], chr_[a + 8]
                for b in range(8):
                    xx = px + b
                    if 0 <= xx < 256:
                        c = (lo >> (7 - b) & 1) | (hi >> (7 - b) & 1) << 1
                        bg[xx] = c
                        bgp[xx] = palno
                px += 8
                if (v & 0x1F) == 31:
                    v = (v & ~0x1F) ^ 0x0400
                else:
                    v += 1
            if not self.mask & 0x02:
                for i in range(8):
                    bg[i] = 0
        sp = [0] * 256
        spp = [0] * 256
        spfront = [0] * 256
        if show_sp:
            h = 16 if self.ctrl & 0x20 else 8
            found = 0
            sprite0_x = []
            for n in range(64):
                y = self.oam[4 * n]
                r = line - (y + 1)
                if not 0 <= r < h:
                    continue
                found += 1
                if found > 8:
                    self.status |= 0x20
                    break
                tile, attr, x0 = self.oam[4 * n + 1], self.oam[4 * n + 2], self.oam[4 * n + 3]
                if attr & 0x80:
                    r = h - 1 - r
                if h == 8:
                    a = ((self.ctrl & 0x08) << 9) + tile * 16 + r
                else:
                    a = (tile & 1) * 0x1000 + (tile & 0xFE) * 16 + (r & 7) + (16 if r >= 8 else 0)
                lo, hi = chr_[a], chr_[a + 8]
                for b in range(8):
                    xx = x0 + b
                    if xx > 255:
                        break
                    bit = b if attr & 0x40 else 7 - b
                    c = (lo >> bit & 1) | (hi >> bit & 1) << 1
                    if not c or sp[xx]:
                        continue
                    if xx < 8 and not self.mask & 0x04:
                        continue
                    sp[xx] = c
                    spp[xx] = attr & 3
                    spfront[xx] = not attr & 0x20
                    if n == 0 and bg[xx] and xx != 255:
                        sprite0_x.append(xx)
            if sprite0_x and show_bg:
                self.status |= 0x40
        for i in range(256):
            if sp[i] and (spfront[i] or not bg[i]):
                row[i] = pal[0x10 | spp[i] << 2 | sp[i]] & 0x3F
            elif bg[i]:
                row[i] = pal[bgp[i] << 2 | bg[i]] & 0x3F
            else:
                row[i] = pal[0] & 0x3F
        # end of the line: fine Y++ (dot 256), horizontal bits from t (dot 257)
        v = self.v
        if (v & 0x7000) != 0x7000:
            v += 0x1000
        else:
            v &= ~0x7000
            cy = (v & 0x03E0) >> 5
            if cy == 29:
                cy = 0
                v ^= 0x0800
            elif cy == 31:
                cy = 0
            else:
                cy += 1
            v = (v & ~0x03E0) | cy << 5
        self.v = (v & ~0x041F) | (self.t & 0x041F)


def frame_rgb(frame) -> bytes:
    """A frame (240 rows of palette indices) as packed RGB bytes."""
    out = bytearray()
    for row in frame:
        for c in row:
            out += bytes(PALETTE[c])
    return bytes(out)


def parse_buttons(spec: str):
    """'start@60-70,right@100-400,a@120-125+200-205' -> frame -> mask."""
    ranges = []
    for part in filter(None, (s.strip() for s in spec.split(","))):
        names, _, frames = part.partition("@")
        mask = 0
        for nm in names.split("+"):
            mask |= BUTTONS[nm.lower()]
        for fr in frames.split("+"):
            a, _, b = fr.partition("-")
            ranges.append((int(a), int(b or a), mask))

    def fn(frame):
        m = 0
        for a, b, mask in ranges:
            if a <= frame <= b:
                m |= mask
        return m
    return fn
