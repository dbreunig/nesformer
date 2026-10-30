"""Teacher-forced check: the circuit's exact IR evaluation against the reference trace."""

import numpy as np

from chipformer.q35 import ir
from nesformer import core
from nesformer import spec as S
from nesformer.cpu import Cpu
from nesformer.run import injected_positions


def check_trace(c, prg: bytes, toks: list[int], pc=None, chunk: int = 4000, exact: bool = True,
                verbose: bool = False):
    """Evaluate the circuit on toks (from boot) and compare its prediction at every position
    with the next token (positions before an injected token are skipped). Returns
    (positions checked, mismatches list)."""
    core.snapshot(c, Cpu.boot(prg, pc))
    inj = injected_positions(toks)
    carry: dict = {}
    bad, checked = [], 0
    for s in range(0, len(toks), chunk):
        part = toks[s:s + chunk]
        mat = ir.evaluate(c, part, check=exact, carry=carry)
        pred = core.predictions(c, mat)
        for i, p in enumerate(pred):
            g = s + i
            if g + 1 >= len(toks) or (g + 1) in inj:
                continue
            checked += 1
            if p != toks[g + 1]:
                bad.append((g, S.name(toks[g]), S.name(toks[g + 1]),
                            S.name(p) if p >= 0 else p))
        if verbose:
            print(f"  {s + len(part):,} tokens, {len(bad)} mismatches", flush=True)
        if bad and len(bad) > 20:
            break
    return checked, bad
