"""nesformer: an NES CPU compiled into the weights of a Qwen3.5 model.

    python -m nesformer build [--out weights]                compile the circuit to a checkpoint
    python -m nesformer run GAME.nes [--frames 120] [--gif out.gif] [--buttons ...] [--show]
    python -m nesformer check GAME.nes [--tokens 50000]      the weights vs the reference CPU
    python -m nesformer trace GAME.nes [--tokens 40]         print the model's token stream
    python -m nesformer info [weights]

The model generates the CPU's execution trace token by token; the host (host.py) plays the PPU,
controllers and NMI timing around it and draws the frames.
"""

import argparse
import sys
import time
from pathlib import Path

DEFAULT_WEIGHTS = str(Path(__file__).resolve().parent.parent / "weights")


def load_cart(path):
    from nesformer.host import Cart
    try:
        return Cart.load(Path(path).read_bytes())
    except ValueError as e:
        sys.exit(f"{path}: {e}")


def make_engine(weights: str, kind: str):
    if not (Path(weights) / "model.safetensors").exists():
        sys.exit(f"no weights in {weights}: run `python -m nesformer build --out {weights}`")
    if kind == "numpy":
        from nesformer.engine import Engine
        return Engine(weights)
    if kind == "hf":
        from nesformer.hfcheck import HFEngine
        return HFEngine(weights)
    from nesformer.fastengine import FastEngine
    return FastEngine(weights)


def ansi_frame(frame, width=128) -> str:
    """A frame as terminal half-block characters (24-bit color), downscaled 2x."""
    from nesformer.host import PALETTE
    step = 256 // width
    lines = []
    for y in range(0, 240, 2 * step):
        top, bot = frame[y], frame[min(y + step, 239)]
        cells = []
        for x in range(0, 256, step):
            a, b = PALETTE[top[x]], PALETTE[bot[x]]
            cells.append(f"\x1b[38;2;{a[0]};{a[1]};{a[2]}m\x1b[48;2;{b[0]};{b[1]};{b[2]}m▀")
        lines.append("".join(cells) + "\x1b[0m")
    return "\n".join(lines)


def save_frames(frames, gif: str | None, png_dir: str | None, every: int, scale: int):
    from PIL import Image

    from nesformer.host import frame_rgb
    ims = [Image.frombytes("RGB", (256, 240), frame_rgb(f)) for f in frames[::every]]
    if scale != 1:
        ims = [im.resize((256 * scale, 240 * scale), Image.NEAREST) for im in ims]
    if png_dir:
        Path(png_dir).mkdir(parents=True, exist_ok=True)
        for i, im in enumerate(ims):
            im.save(Path(png_dir) / f"frame{i * every:05d}.png")
    if gif and ims:
        ims[0].save(gif, save_all=True, append_images=ims[1:], loop=0,
                    duration=max(17, round(1000 * every / 60)))


def cmd_build(a):
    from nesformer import weights
    t0 = time.time()
    info = weights.export(a.out)
    print(f"wrote {info['dir']}: {info['layers']} layers ({info['circuit']['layers']} circuit + 1 "
          f"zero attention), hidden {info['hidden']}, {info['params']:,} parameters, "
          f"{info['nonzero']:,} nonzero ({time.time() - t0:.0f}s)")


def cmd_run(a):
    from nesformer.host import parse_buttons
    from nesformer.run import run_model
    cart = load_cart(a.rom)
    eng = make_engine(a.weights, a.engine)
    buttons = parse_buttons(a.buttons) if a.buttons else None
    t0 = time.time()
    last = [t0]

    def on_frame(host):
        now = time.time()
        if a.show:
            sys.stdout.write("\x1b[H" + ansi_frame(host.frames[-1]) + "\n")
        msg = (f"frame {host.frame}/{a.frames}  {eng.steps:,} tokens  "
               f"{eng.steps / (now - t0):,.0f} tok/s  {now - last[0]:.1f}s/frame")
        print(msg if not a.show else msg + "\x1b[K", flush=True)
        last[0] = now

    if a.show:
        sys.stdout.write("\x1b[2J")
    toks, host = run_model(eng, cart, frames=a.frames, buttons=buttons, on_frame=on_frame,
                           skip=not a.no_skip, max_tokens=a.max_tokens)
    dt = time.time() - t0
    print(f"{len(toks):,} tokens in {dt:.0f}s ({len(toks) / dt:,.0f} tok/s), {host.frame} frames, "
          f"{host.cycles:,} CPU cycles ({host.cycles / 1789773:.1f}s of NES time)"
          + (", HALT" if host.halted else ""))
    if host.halted:
        from nesformer import spec as S
        print("the machine halted (an unsupported opcode, or a jump outside the cartridge ROM) "
              f"near PC ${host.pc:04X}; last tokens: "
              + " ".join(S.name(t) for t in toks[-6:]))
    out = a.gif or (None if a.png else Path(a.rom).with_suffix(".gif").name)
    if host.frames:
        save_frames(host.frames, out, a.png, a.every, a.scale)
        if out:
            print(f"saved {out}")
        if a.png:
            print(f"saved frames to {a.png}/")
    if a.trace:
        Path(a.trace).write_text("\n".join(str(t) for t in toks))


def cmd_check(a):
    from nesformer.host import parse_buttons
    from nesformer.run import run_model, run_reference
    cart = load_cart(a.rom)
    buttons = parse_buttons(a.buttons) if a.buttons else None
    ref, rh, cpu = run_reference(cart, frames=a.frames, buttons=buttons, max_tokens=a.tokens,
                                 skip=not a.no_skip)
    eng = make_engine(a.weights, a.engine)
    t0 = time.time()
    try:
        toks, host = run_model(eng, cart, frames=a.frames, buttons=buttons, max_tokens=len(ref),
                               expect=ref, skip=not a.no_skip)
    except AssertionError as e:
        sys.exit(f"MISMATCH: {e}")
    dt = time.time() - t0
    same = host.frames == rh.frames
    print(f"OK: the weights generated {len(toks):,} tokens identical to the reference CPU's trace "
          f"({host.frame} frames, frames identical: {same}; {len(toks) / dt:,.0f} tok/s)")


def cmd_trace(a):
    from nesformer import spec as S
    from nesformer.run import run_model
    cart = load_cart(a.rom)
    eng = make_engine(a.weights, a.engine)
    toks, host = run_model(eng, cart, frames=10**9, max_tokens=a.tokens, skip=False)
    print(" ".join(S.name(t) for t in toks))


def cmd_info(a):
    import json
    meta = json.loads((Path(a.weights) / "nesformer.json").read_text())
    cfg = json.loads((Path(a.weights) / "config.json").read_text())
    print(json.dumps({"machine": meta["machine"], "architecture": cfg["architectures"],
                      "hidden_size": cfg["hidden_size"], "layers": cfg["layer_types"],
                      "vocab": cfg["vocab_size"], "circuit": meta["circuit"]}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nesformer", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="compile the circuit into a Qwen3.5 checkpoint")
    b.add_argument("--out", default=DEFAULT_WEIGHTS)
    b.set_defaults(fn=cmd_build)
    for name, fn, hlp in (("run", cmd_run, "run a ROM on the weights"),
                          ("check", cmd_check, "free-run the weights against the reference CPU"),
                          ("trace", cmd_trace, "print the model's token stream")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("rom")
        p.add_argument("--weights", default=DEFAULT_WEIGHTS)
        p.add_argument("--engine", choices=["fast", "numpy", "hf"], default="fast",
                       help="fast: numba-compiled forward pass (default); numpy: plain numpy; "
                            "hf: stock transformers Qwen3_5ForCausalLM (needs torch, ~10 tok/s)")
        p.add_argument("--buttons", default="",
                       help="controller script, e.g. 'start@60-65,right@80-400,a@100-104+150-154' "
                            "(buttons a b select start up down left right; @frame ranges)")
        p.add_argument("--no-skip", action="store_true",
                       help="run every iteration of idle wait loops (slower; same frames)")
        p.set_defaults(fn=fn)
        if name == "run":
            p.add_argument("--frames", type=int, default=120)
            p.add_argument("--max-tokens", type=int, default=None)
            p.add_argument("--gif", default=None, help="animated GIF path (default: ROM name)")
            p.add_argument("--png", default=None, help="also save frames as PNGs in this folder")
            p.add_argument("--every", type=int, default=1, help="keep every n-th frame")
            p.add_argument("--scale", type=int, default=2)
            p.add_argument("--show", action="store_true", help="draw frames in the terminal")
            p.add_argument("--trace", default=None, help="save the token ids to this file")
        elif name == "check":
            p.add_argument("--tokens", type=int, default=50000)
            p.add_argument("--frames", type=int, default=10**9)
        else:
            p.add_argument("--tokens", type=int, default=40)
    i = sub.add_parser("info", help="describe a checkpoint")
    i.add_argument("weights", nargs="?", default=DEFAULT_WEIGHTS)
    i.set_defaults(fn=cmd_info)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
