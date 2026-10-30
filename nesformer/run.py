"""Drivers: run a token source (the reference CPU or the model) against the host.

The host injects two kinds of token: NMI at a record boundary when the PPU raised one, and IN
right after the source emits IOR. Everything else comes from the source.
"""

from nesformer import spec as S
from nesformer.cpu import Cpu
from nesformer.host import Cart, Host

IOR_BASE = S.FAMILIES["IOR"][0]


def run_reference(cart: Cart, frames: int = 60, buttons=None, max_tokens: int | None = None,
                  keep_frames: bool = True, on_frame=None, pc: int | None = None,
                  skip: bool = True):
    """Run the reference CPU; returns (tokens, host, cpu)."""
    host = Host(cart, buttons=buttons, keep_frames=keep_frames, skip=skip)
    cpu = Cpu.boot(cart.prg, pc)
    host.pc = cpu.pc
    toks: list[int] = [S.tok("GO")]
    seen = 0
    while not cpu.halted and host.frame < frames:
        if max_tokens is not None and len(toks) >= max_tokens:
            break
        if host.wants_nmi():
            toks.append(S.tok("NMI"))
            host.feed(toks[-1])
            gen = cpu.nmi_record()
        else:
            gen = cpu.record()
        send = None
        while True:
            try:
                t = gen.send(send)
            except StopIteration:
                break
            toks.append(t)
            host.feed(t)
            send = None
            if IOR_BASE <= t < IOR_BASE + 64:
                send = host.io_read(t - IOR_BASE)
                toks.append(S.tok("IN", send))
                host.feed(toks[-1])
        if on_frame and host.frame > seen:
            seen = host.frame
            on_frame(host)
    return toks, host, cpu


def injected_positions(toks: list[int]) -> set[int]:
    """Indices of tokens the host injected (IN after IOR, NMI): not predicted by the source."""
    nmi = S.tok("NMI")
    out = set()
    for i, t in enumerate(toks):
        if t == nmi or S.family(t)[0] == "IN":
            out.add(i)
    return out


def run_model(engine, cart: Cart, frames: int = 60, buttons=None, max_tokens: int | None = None,
              keep_frames: bool = True, on_frame=None, on_token=None, pc: int | None = None,
              expect: list[int] | None = None, skip: bool = True):
    """Free-run the weights: the model predicts every token except the host's NMI / IN
    injections. With ``expect`` (a reference trace), stop at the first difference and report
    it. Returns (tokens, host)."""
    from nesformer import core

    host = Host(cart, buttons=buttons, keep_frames=keep_frames, skip=skip)
    boot = Cpu.boot(cart.prg, pc)
    host.pc = boot.pc
    engine.load_state(core.state_rows(boot))
    toks = [S.tok("GO")]
    pred = engine.step(toks[0])
    seen = 0
    halt, nmi = S.tok("HALT"), S.tok("NMI")
    while host.frame < frames and not host.halted:
        if max_tokens is not None and len(toks) >= max_tokens:
            break
        t = nmi if host.wants_nmi() else pred
        for tt in _with_injection(t, host):
            toks.append(tt)
            if expect is not None and (len(toks) > len(expect) or expect[len(toks) - 1] != tt):
                want = expect[len(toks) - 1] if len(toks) <= len(expect) else None
                raise AssertionError(f"token {len(toks) - 1}: model {S.name(tt)}, reference "
                                     f"{S.name(want) if want is not None else 'end'}")
            if tt != t:  # an injected IN: the host already saw it
                pass
            pred = engine.step(tt)
            if on_token:
                on_token(tt)
        if t == halt:
            break
        if on_frame and host.frame > seen:
            seen = host.frame
            on_frame(host)
    return toks, host


def _with_injection(t, host):
    """Feed t to the host; after an IOR also yield the host's IN answer."""
    host.feed(t)
    yield t
    if IOR_BASE <= t < IOR_BASE + 64:
        v = host.io_read(t - IOR_BASE)
        inj = S.tok("IN", v)
        host.feed(inj)
        yield inj
