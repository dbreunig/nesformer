"""Fast checks (a few minutes): python -m pytest -q tests"""

from pathlib import Path

import pytest

from nesformer import core
from nesformer import spec as S
from nesformer.check import check_trace
from nesformer.cpu import OPS, Cpu
from nesformer.host import Cart, parse_buttons
from nesformer.run import run_model, run_reference

ROOT = Path(__file__).resolve().parent.parent
ROMS = ROOT / "roms"
WEIGHTS = ROOT / "weights"
BUTTONS = "start@90-95+200-205,right@100-400,a@130-135+160-165"


def cart(name):
    return Cart.load((ROMS / name).read_bytes())


def test_reference_matches_nestest_log():
    """The reference CPU's registers before every instruction equal nestest.log's, up to the
    point where nestest jumps into RAM (code must run from the cartridge ROM)."""
    prg = cart("nestest.nes").prg
    cpu = Cpu.boot(prg, 0xC000)
    n = 0
    for line in (ROMS / "nestest.log").read_text().splitlines():
        pc = int(line[:4], 16)
        if pc < 0x8000 or int(line[6:8], 16) not in OPS:
            break
        while cpu.offset >= S.OFF_MAX:
            list(cpu.record())
        want = [int(line[line.index(k + ":") + len(k) + 1:][:2], 16)
                for k in ("A", "X", "Y", "P", "SP")]
        assert (cpu.pc, [cpu.A, cpu.X, cpu.Y, cpu.p_byte(0), cpu.SP]) == (pc, want), line
        g = cpu.record()
        t = next(g)
        try:
            while True:
                t = g.send(0xFF if S.family(t)[0] == "IOR" else None)
        except StopIteration:
            pass
        n += 1
    assert n == 3348


@pytest.fixture(scope="module")
def circ():
    return core.build()


@pytest.mark.parametrize("name,pc,n", [("nestest.nes", 0xC000, 5000),
                                        ("full_game.nes", None, 6000),
                                        ("thwaite.nes", None, 6000)])
def test_circuit_teacher_forced(circ, name, pc, n):
    """The exact IR (with its bf16 checks) predicts every reference token."""
    c = cart(name)
    toks, _, _ = run_reference(c, frames=10**6, pc=pc, max_tokens=n,
                               buttons=parse_buttons(BUTTONS))
    checked, bad = check_trace(circ, c.prg, toks, pc=pc)
    assert checked > 0 and not bad, bad[:5]


@pytest.fixture(scope="module")
def engine():
    if not (WEIGHTS / "model.safetensors").exists():
        pytest.skip("no weights (python -m nesformer build)")
    from nesformer.fastengine import FastEngine
    return FastEngine(str(WEIGHTS))


@pytest.mark.parametrize("name,pc,n", [("nestest.nes", 0xC000, 10**6),
                                        ("full_game.nes", None, 30000),
                                        ("thwaite.nes", None, 30000)])
def test_weights_free_run(engine, name, pc, n):
    """The checkpoint's weights generate the reference trace token for token."""
    c = cart(name)
    btn = parse_buttons(BUTTONS)
    ref, _, _ = run_reference(c, frames=10**6, pc=pc, max_tokens=n, buttons=btn)
    toks, _ = run_model(engine, c, frames=10**6, pc=pc, max_tokens=len(ref), expect=ref,
                        buttons=btn)
    assert toks == ref


def test_numpy_engine_agrees():
    if not (WEIGHTS / "model.safetensors").exists():
        pytest.skip("no weights")
    from nesformer.engine import Engine
    c = cart("full_game.nes")
    ref, _, _ = run_reference(c, frames=10**6, max_tokens=1500)
    toks, _ = run_model(Engine(str(WEIGHTS)), c, frames=10**6, max_tokens=len(ref), expect=ref)
    assert toks == ref


@pytest.mark.parametrize("name", ["thwaite.nes", "full_game.nes"])
def test_skip_changes_nothing_visible(name):
    c = cart(name)
    btn = parse_buttons(BUTTONS)
    _, full, _ = run_reference(c, 120, btn, skip=False)
    _, fast, _ = run_reference(c, 120, btn, skip=True)
    assert fast.skipped_iters > 0
    assert full.frames == fast.frames and full.io_log == fast.io_log
