"""Compile the circuit into a stock Qwen3_5ForCausalLM checkpoint (config.json + safetensors).

The emitter follows Qwenputer's chipformer/q35/emit.py and rv/tiny.py, rewritten in numpy so it
needs no torch. The residual layout of a chip token is

    [ballast x m] [IR dims 1..n-1] [unused]

with every RMSNorm dividing by a known power of 2 (the ballast dominates), so the weights carry
the inverse scale. The chip layers are linear-attention (Gated DeltaNet) layers; one all-zero
full-attention layer comes last because HF's cache counts positions there. The output head is
tied to the embedding: a token's row holds CODE_AMP * (2 bit - 1) on the output-code dims, and
the final norm passes only those dims, so the argmax is the token whose code bits all agree with
what the exit wrote.
"""

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chipformer.dsl.graph import ONE, Expr
from chipformer.q35.ir import FFN, HD, Head, K, bf16
from nesformer import core
from nesformer import spec as S

BASE = {  # Qwen3.5 text config fields we do not size ourselves
    "hidden_act": "silu", "linear_conv_kernel_dim": 4, "linear_key_head_dim": HD,
    "linear_value_head_dim": HD, "num_attention_heads": 8, "num_key_value_heads": 2,
    "head_dim": 256, "rms_norm_eps": 1e-6, "attn_output_gate": True,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000000,
                        "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10],
                        "mrope_interleaved": True},
    "max_position_embeddings": 1 << 30, "tie_word_embeddings": True,
}
CODE_GAIN = 2.0**20


@dataclass
class Layout:
    hidden: int
    m: int
    log2_ballast: int = 40

    @property
    def ballast(self) -> float:
        return 2.0**self.log2_ballast

    @property
    def scale(self) -> float:
        mean = self.m * self.ballast**2 / self.hidden
        log4 = math.log(mean, 4)
        assert abs(log4 - round(log4)) < 1e-12, "m * B^2 / hidden must be a power of 4"
        return 2.0 ** -round(log4)

    def dim(self, d: int) -> int:
        return self.m + d


def pick_layout(n_dims: int) -> Layout:
    best = None
    for m in range(3, 16):
        k = 0
        while m * 4**k < m + n_dims or (m * 4**k) % 64:
            k += 1
        h = m * 4**k
        if best is None or h < best.hidden:
            best = Layout(hidden=h, m=m)
    return best


def bf16_parts(x: float, n: int) -> list[float]:
    parts = []
    for _ in range(n):
        if x == 0:
            break
        p = float(bf16(np.array([x]))[0])
        parts.append(p)
        x -= p
    if x != 0:
        raise ValueError(f"constant needs more than {n} bf16 parts")
    return parts


def row(e: Expr, lay: Layout, out: np.ndarray, w: float = 1.0) -> None:
    c = lay.scale
    for d, coef in e.terms.items():
        if d == ONE:
            for i, p in enumerate(bf16_parts(w * float(coef) / (lay.ballast * c), lay.m)):
                out[i] += p
        else:
            out[lay.dim(d)] += w * float(coef) / c


def check_bf16(name: str, t: np.ndarray) -> None:
    t = np.asarray(t, np.float32)
    bad = np.argwhere(((t.view(np.uint32) & 0xFFFF) != 0) | ~np.isfinite(t))
    if len(bad):
        i = tuple(bad[0])
        raise ValueError(f"{name}: weight {t[i]!r} at {i} is not a bf16 value")


def emit_layers(c, lay: Layout, tc: dict) -> dict:
    h, nk, nv = tc["hidden_size"], tc["linear_num_key_heads"], tc["linear_num_value_heads"]
    inter = tc["intermediate_size"]
    r = nv // nk
    keep = set(c.keep)
    exit_level = max(n.level for n in c.nodes if n.name.startswith("exit."))
    exit_clear = [d for d in range(1, len(c.dims)) if d not in keep]
    one = Expr({ONE: 1})
    out = {}
    for L in range(c.n_layers()):
        p = f"model.layers.{L}."
        f32 = np.float32
        qkv = np.zeros((2 * nk * HD + nv * HD, h), f32)
        convw = np.zeros((2 * nk * HD + nv * HD, 1, tc["linear_conv_kernel_dim"]), f32)
        zw = np.zeros((nv * HD, h), f32)
        bw = np.zeros((nv, h), f32)
        aw = np.zeros((nv, h), f32)
        ow = np.zeros((h, nv * HD), f32)
        gate = np.zeros((inter, h), f32)
        up = np.zeros((inter, h), f32)
        down = np.zeros((h, inter), f32)
        heads = [n for n in c.nodes if isinstance(n, Head) and n.level == 2 * L + 1]
        assert len(heads) <= nk
        for j, n in enumerate(heads):
            vhs = n.vheads()
            assert len(vhs) <= r, (n.name, len(vhs), r)
            for base, chans in ((j * HD, n.q), (nk * HD + j * HD, n.k)):
                for i, ch in chans.items():
                    row(ch.expr, lay, qkv[base + i])
                    for lag, w in enumerate(ch.taps):
                        convw[base + i, 0, -1 - lag] = w
            for x, vhd in enumerate(vhs):
                vh = j * r + x
                base = 2 * nk * HD + vh * HD
                for i, ch in vhd.v.items():
                    row(ch.expr, lay, qkv[base + i])
                    for lag, w in enumerate(ch.taps):
                        convw[base + i, 0, -1 - lag] = w
                beta = Expr({ONE: 128}) if vhd.beta is None else 256 * vhd.beta - 128 * one
                row(beta, lay, bw[vh])
                row(vhd.a, lay, aw[vh])
                for col, e in vhd.z.items():
                    row(e, lay, zw[vh * HD + col])
                for col, d, w in vhd.out:
                    ow[lay.dim(d), vh * HD + col] += w
        units = []
        for n in c.nodes:
            if isinstance(n, FFN) and n.level == 2 * L + 2:
                units += [(coef, a, b, n.out) for coef, a, b in n.terms]
        if exit_level == 2 * L + 2:  # the exit clears every chip dim but the output code
            units += [(-1, Expr({o: 1}), one, o) for o in exit_clear]
        assert len(units) + (lay.m if exit_level == 2 * L + 2 else 0) <= inter, (L, len(units))
        for u, (coef, a, b, d) in enumerate(units):
            row(b, lay, gate[u], K)
            row(a, lay, up[u], 1 / K)
            down[lay.dim(d), u] = float(coef)
        if exit_level == 2 * L + 2:  # ... and the ballast: -B per ballast dim
            for i in range(lay.m):
                u = len(units) + i
                row(Expr({ONE: K}), lay, gate[u])
                up[u, i] = (lay.ballast / K) / (lay.ballast * lay.scale)
                down[i, u] = -1.0
        layer = {
            "input_layernorm.weight": np.zeros(h, f32),
            "post_attention_layernorm.weight": np.zeros(h, f32),
            "linear_attn.in_proj_qkv.weight": qkv,
            "linear_attn.in_proj_z.weight": zw,
            "linear_attn.in_proj_a.weight": aw,
            "linear_attn.in_proj_b.weight": bw,
            "linear_attn.conv1d.weight": convw,
            "linear_attn.dt_bias": np.full(nv, -110.0, f32),
            "linear_attn.A_log": np.zeros(nv, f32),
            "linear_attn.norm.weight": np.ones(HD, f32),
            "linear_attn.out_proj.weight": ow,
            "mlp.gate_proj.weight": gate,
            "mlp.up_proj.weight": up,
            "mlp.down_proj.weight": down,
        }
        for k, t in layer.items():
            check_bf16(p + k, t)
            out[p + k] = t
    return out


def embedding_matrix(c, lay: Layout, n_vocab: int) -> np.ndarray:
    e = np.zeros((n_vocab, lay.hidden), np.float32)
    for name, (base, size) in S.FAMILIES.items():
        p = np.arange(size)
        e[base:base + size, :lay.m] = lay.ballast
        for d, fn in c.vfeats.items():
            e[base:base + size, lay.dim(d)] = np.broadcast_to(np.asarray(fn(name, p)), (size,))
    for name, t in S.CONTROLS.items():
        e[t, :lay.m] = lay.ballast
        for d, fn in c.vfeats.items():
            e[t, lay.dim(d)] = float(np.asarray(fn(name, np.asarray(0))))
    # spot check against the IR's scalar features
    rng = np.random.default_rng(0)
    for t in rng.integers(0, S.VOCAB, 300).tolist() + [S.VOCAB - 1]:
        for d in c.token_fns:
            assert e[t, lay.dim(d)] == float(c.feature_value(d, t)), (t, c.dims[d].name)
    check_bf16("embedding", e)
    return e


def full_attention_zeros(tc: dict, L: int) -> dict:
    """The last layer: a full-attention layer with zero weights (shapes as transformers')."""
    h, nh, nkv, hd = tc["hidden_size"], tc["num_attention_heads"], tc["num_key_value_heads"], \
        tc["head_dim"]
    inter = tc["intermediate_size"]
    p = f"model.layers.{L}."
    z = np.zeros
    return {p + "input_layernorm.weight": z(h, np.float32),
            p + "post_attention_layernorm.weight": z(h, np.float32),
            p + "self_attn.q_proj.weight": z((nh * hd * 2, h), np.float32),
            p + "self_attn.k_proj.weight": z((nkv * hd, h), np.float32),
            p + "self_attn.v_proj.weight": z((nkv * hd, h), np.float32),
            p + "self_attn.o_proj.weight": z((h, nh * hd), np.float32),
            p + "self_attn.q_norm.weight": z(hd, np.float32),
            p + "self_attn.k_norm.weight": z(hd, np.float32),
            p + "mlp.gate_proj.weight": z((inter, h), np.float32),
            p + "mlp.up_proj.weight": z((inter, h), np.float32),
            p + "mlp.down_proj.weight": z((h, inter), np.float32)}


def build_checkpoint(c=None):
    """(config dict, state dict of float32 arrays that are all exact bf16 values, circuit)."""
    c = c or core.build()
    lay = pick_layout(len(c.dims))
    n_chip = c.n_layers()
    n_vocab = -(-(S.VOCAB + 1) // 64) * 64
    tc = dict(BASE, hidden_size=lay.hidden, vocab_size=n_vocab, num_hidden_layers=n_chip + 1,
              layer_types=["linear_attention"] * n_chip + ["full_attention"],
              linear_num_key_heads=c.target.key_heads,
              linear_num_value_heads=c.target.value_heads,
              intermediate_size=c.target.intermediate, eos_token_id=S.tok("HALT"),
              pad_token_id=S.PAD, bos_token_id=S.PAD)
    sd = emit_layers(c, lay, tc)
    sd.update(full_attention_zeros(tc, n_chip))
    sd["model.embed_tokens.weight"] = embedding_matrix(c, lay, n_vocab)
    norm = np.full(lay.hidden, -1.0, np.float32)  # (1 + w) = 0 except on the output code
    for d in c.keep:
        norm[lay.dim(d)] = CODE_GAIN
    sd["model.norm.weight"] = norm
    config = {"architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text",
              "torch_dtype": "bfloat16", **tc}
    meta = {"machine": "nesformer: NES CPU (6502 / 2A03) + 2 KB RAM + NROM PRG-ROM",
            "vocab": S.VOCAB, "pad": S.PAD, "families": {k: list(v) for k, v in
                                                         S.FAMILIES.items()},
            "controls": S.CONTROLS, "ballast_dims": lay.m, "log2_ballast": lay.log2_ballast,
            "code_dims": [lay.dim(d) for d in c.keep], "nfam": core.NFAM, "npay": core.NPAY,
            "out_names": core.OUT_NAMES, "chip_layers": n_chip, "circuit": c.stats(),
            "heads": head_layout(c)}
    return config, sd, meta, c


def head_layout(c) -> dict:
    """Where each state table lives (layer, key head, value heads): the snapshot's address."""
    out = {}
    for L in range(c.n_layers()):
        heads = [n for n in c.nodes if isinstance(n, Head) and n.level == 2 * L + 1]
        for j, n in enumerate(heads):
            out[n.name] = [L, j, len(n.vheads())]
    return out


# --- safetensors (bf16) without torch -----------------------------------------------------------

def save_safetensors(path: Path, tensors: dict, metadata: dict | None = None) -> None:
    header, offset, blobs = {}, 0, []
    for name in sorted(tensors):
        t = np.ascontiguousarray(tensors[name], dtype=np.float32)
        u = t.view(np.uint32)
        assert ((u & 0xFFFF) == 0).all(), f"{name} is not exactly bf16"
        b = (u >> 16).astype("<u2").tobytes()
        header[name] = {"dtype": "BF16", "shape": list(t.shape),
                        "data_offsets": [offset, offset + len(b)]}
        offset += len(b)
        blobs.append(b)
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        for b in blobs:
            fh.write(b)


def load_safetensors(path: Path) -> dict:
    """float32 arrays (bf16 / f32 / f16 tensors)."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        data = fh.read()
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        a, b = info["data_offsets"]
        raw = data[a:b]
        if info["dtype"] == "BF16":
            t = (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32)
        elif info["dtype"] == "F32":
            t = np.frombuffer(raw, "<f4").copy()
        elif info["dtype"] == "F16":
            t = np.frombuffer(raw, "<f2").astype(np.float32)
        else:
            raise ValueError(info["dtype"])
        out[name] = t.reshape(info["shape"])
    return out


def export(out_dir: str) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config, sd, meta, c = build_checkpoint()
    save_safetensors(out / "model.safetensors", sd, {"format": "pt"})
    (out / "config.json").write_text(json.dumps(config, indent=1))
    (out / "generation_config.json").write_text(json.dumps(
        {"eos_token_id": S.tok("HALT"), "pad_token_id": S.PAD, "do_sample": False}, indent=1))
    (out / "nesformer.json").write_text(json.dumps(meta, indent=1))
    n = sum(t.size for t in sd.values())
    nz = sum(int(np.count_nonzero(t)) for t in sd.values())
    return {"dir": str(out), "params": n, "nonzero": nz, "hidden": config["hidden_size"],
            "layers": config["num_hidden_layers"], "circuit": c.stats()}
