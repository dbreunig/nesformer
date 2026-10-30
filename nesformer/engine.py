"""Run the checkpoint's weights: Qwen3.5's forward pass, one token at a time, on the CPU.

This computes exactly what Qwen3_5ForCausalLM computes in bf16 (RMSNorm with a 1 + w gain, the
qkv / z / a / b projections, the causal conv + silu, l2-normalized q and k, the gated delta rule
in fp32, the gated RMSNorm, out_proj, the SwiGLU MLP, the final norm and the tied output head),
rounding to bf16 where the bf16 model does. It only skips work that is exactly zero: weight rows
and columns that are all zero, heads with no weights, and state rows a one-hot key or query
does not touch. hfcheck.py runs the same weights through stock transformers instead.
"""

import json
import struct
from pathlib import Path

import numpy as np
import scipy.sparse as sp

F32 = np.float32


def bf16r(x: np.ndarray) -> np.ndarray:
    """Round float32 to the nearest bf16 (ties to even), as float32."""
    x = np.ascontiguousarray(x, dtype=F32)
    u = x.view(np.uint32)
    u = (u + (0x7FFF + ((u >> 16) & 1))) & 0xFFFF0000
    return u.astype(np.uint32).view(F32)


def silu(x):
    with np.errstate(over="ignore"):
        return (x / (F32(1) + np.exp(-x))).astype(F32)


def rmsnorm(x, gain, eps):
    var = np.mean(x * x, dtype=F32)
    return bf16r(x * F32(1.0 / np.sqrt(var + F32(eps))) * gain)


class Weights:
    """Lazy bf16 safetensors reader (memory-mapped)."""

    def __init__(self, path: Path):
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            self.header = json.loads(fh.read(n))
        self.base = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def raw(self, name):
        info = self.header[name]
        a, b = info["data_offsets"]
        assert info["dtype"] == "BF16", info["dtype"]
        return self.mm[self.base + a:self.base + b].view("<u2").reshape(info["shape"])

    def get(self, name) -> np.ndarray:
        return (self.raw(name).astype(np.uint32) << 16).view(F32)


class Layer:
    def __init__(self, W: Weights, L: int, cfg: dict):
        p = f"model.layers.{L}."
        self.nk, self.nv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
        self.hd = cfg["linear_key_head_dim"]
        self.eps = cfg["rms_norm_eps"]
        hd, nk = self.hd, self.nk
        self.g_in = F32(1) + W.get(p + "input_layernorm.weight")
        self.g_post = F32(1) + W.get(p + "post_attention_layernorm.weight")
        qkv = W.get(p + "linear_attn.in_proj_qkv.weight")
        self.ch = np.nonzero(np.abs(qkv).sum(1))[0]
        self.Wqkv = sp.csr_matrix(qkv[self.ch])
        self.conv = W.get(p + "linear_attn.conv1d.weight")[self.ch, 0, :]  # [n, 4]
        self.hist = np.zeros((4, len(self.ch)), F32)
        kd = nk * hd
        isq, isk = self.ch < kd, (self.ch >= kd) & (self.ch < 2 * kd)
        isv = self.ch >= 2 * kd
        self.q_idx = (self.ch[isq] // hd, self.ch[isq] % hd, np.nonzero(isq)[0])
        self.k_idx = ((self.ch[isk] - kd) // hd, (self.ch[isk] - kd) % hd, np.nonzero(isk)[0])
        self.v_idx = ((self.ch[isv] - 2 * kd) // hd, (self.ch[isv] - 2 * kd) % hd,
                      np.nonzero(isv)[0])
        zw = W.get(p + "linear_attn.in_proj_z.weight")
        self.zch = np.nonzero(np.abs(zw).sum(1))[0]
        self.Wz = sp.csr_matrix(zw[self.zch])
        ow = W.get(p + "linear_attn.out_proj.weight")
        self.ocols = np.nonzero(np.abs(ow).sum(0))[0]
        self.Wo = sp.csr_matrix(ow[:, self.ocols])
        r = self.nv // nk
        used = set((self.v_idx[0]).tolist()) | set((self.ocols // hd).tolist()) \
            | set((self.zch // hd).tolist())
        self.vh = np.array(sorted(used), dtype=np.int64)  # active value heads
        self.kh = self.vh // r
        self.Wa = W.get(p + "linear_attn.in_proj_a.weight")[self.vh]
        self.Wb = W.get(p + "linear_attn.in_proj_b.weight")[self.vh]
        self.dt_bias = W.get(p + "linear_attn.dt_bias")[self.vh]
        self.A = np.exp(W.get(p + "linear_attn.A_log")[self.vh]).astype(F32)
        self.norm_w = W.get(p + "linear_attn.norm.weight")
        self.S = np.zeros((self.nv, hd, hd), F32)
        g = W.get(p + "mlp.gate_proj.weight")
        u = W.get(p + "mlp.up_proj.weight")
        self.units = np.nonzero(np.abs(g).sum(1) + np.abs(u).sum(1))[0]
        self.Wg = sp.csr_matrix(g[self.units])
        self.Wu = sp.csr_matrix(u[self.units])
        d = W.get(p + "mlp.down_proj.weight")
        self.Wd = sp.csr_matrix(d[:, self.units])
        self.qscale = F32(hd ** 0.5)

    def reset(self):
        self.hist[:] = 0
        self.S[:] = 0

    def __call__(self, x):
        hd = self.hd
        xn = rmsnorm(x, self.g_in, self.eps)
        # projections and the causal conv (kernel 4) + silu
        qkv = bf16r(self.Wqkv @ xn)
        self.hist[:-1] = self.hist[1:]
        self.hist[-1] = qkv
        cv = bf16r(silu(bf16r((self.conv.T * self.hist).sum(0))))
        q = np.zeros((self.nk, hd), F32)
        k = np.zeros((self.nk, hd), F32)
        v = np.zeros((self.nv, hd), F32)
        q[self.q_idx[0], self.q_idx[1]] = cv[self.q_idx[2]]
        k[self.k_idx[0], self.k_idx[1]] = cv[self.k_idx[2]]
        v[self.v_idx[0], self.v_idx[1]] = cv[self.v_idx[2]]
        z = np.zeros(self.nv * hd, F32)
        z[self.zch] = bf16r(self.Wz @ xn)
        z = z.reshape(self.nv, hd)
        b = bf16r(self.Wb @ xn)
        a = bf16r(self.Wa @ xn)
        with np.errstate(over="ignore"):
            beta = bf16r(F32(1) / (F32(1) + np.exp(-b, dtype=F32)))
        with np.errstate(over="ignore", under="ignore"):
            xa = (a + self.dt_bias).astype(F32)
            sp_ = np.where(xa > 20, xa, np.log1p(np.exp(xa, dtype=F32))).astype(F32)
            decay = np.exp(-self.A * sp_, dtype=F32)
        # the gated delta rule (fp32), for the active value heads
        Q, K, V = q[self.kh], k[self.kh], v[self.vh]
        Q = Q * (F32(1) / np.sqrt((Q * Q).sum(1, keepdims=True) + F32(1e-6)))
        K = K * (F32(1) / np.sqrt((K * K).sum(1, keepdims=True) + F32(1e-6)))
        Q = (Q / self.qscale).astype(F32)
        S = self.S
        dec = np.nonzero(decay != 1)[0]
        for i in dec:
            S[self.vh[i]] *= decay[i]
        ii, rr = np.nonzero(K)
        if len(ii):
            hv = self.vh[ii]
            kvals = K[ii, rr][:, None]
            kv = np.zeros_like(V)
            np.add.at(kv, ii, kvals * S[hv, rr, :])
            delta = (V - kv) * beta[:, None]
            S[hv, rr, :] += kvals * delta[ii]
        o = np.zeros_like(V)
        ii, rr = np.nonzero(Q)
        if len(ii):
            np.add.at(o, ii, Q[ii, rr][:, None] * S[self.vh[ii], rr, :])
        o = bf16r(o)
        # gated RMSNorm per value head, then out_proj
        var = np.mean(o * o, axis=1, keepdims=True, dtype=F32)
        y = bf16r(bf16r(o * (F32(1) / np.sqrt(var + F32(self.eps)))) * self.norm_w)
        y = bf16r(y * silu(z[self.vh]))
        full = np.zeros((self.nv, hd), F32)
        full[self.vh] = y
        x = bf16r(x + bf16r(self.Wo @ full.reshape(-1)[self.ocols]))
        # SwiGLU MLP
        xn = rmsnorm(x, self.g_post, self.eps)
        act = bf16r(bf16r(silu(bf16r(self.Wg @ xn))) * bf16r(self.Wu @ xn))
        return bf16r(x + bf16r(self.Wd @ act))


class Engine:
    """step(token) -> the model's greedy next token."""

    def __init__(self, model_dir: str):
        d = Path(model_dir)
        self.cfg = json.loads((d / "config.json").read_text())
        self.meta = json.loads((d / "nesformer.json").read_text())
        W = Weights(d / "model.safetensors")
        types = self.cfg["layer_types"]
        self.layers = [Layer(W, L, self.cfg) for L, t in enumerate(types)
                       if t == "linear_attention"]
        for L, t in enumerate(types):  # the full-attention layer must be all zero
            if t != "linear_attention":
                for name in W.header:
                    if name.startswith(f"model.layers.{L}.") and "norm" not in name:
                        assert not W.raw(name).any(), name
        self.E = W.raw("model.embed_tokens.weight")  # bf16 bits, memory-mapped
        self.gain = F32(1) + W.get("model.norm.weight")
        self.code = np.nonzero(self.gain)[0]
        self.Ecode = (self.E[:, self.code].astype(np.uint32) << 16).view(F32)
        self.eps = self.cfg["rms_norm_eps"]
        self.nfam, self.npay = self.meta["nfam"], self.meta["npay"]
        self.exact_head = False
        self.steps = 0

    def load_state(self, rows: dict) -> None:
        """Write a snapshot (core.state_rows) into the linear-attention states."""
        for layer in self.layers:
            layer.reset()
        for name, (L, j, nvh) in self.meta["heads"].items():
            S = self.layers[L].S
            for x in range(nvh):
                for r, vec in rows[name][x].items():
                    S[2 * j + x, r] = np.asarray(vec, F32)

    def embed(self, t: int) -> np.ndarray:
        return (self.E[t].astype(np.uint32) << 16).view(F32).copy()

    def step(self, t: int) -> int:
        x = self.embed(t)
        for layer in self.layers:
            x = layer(x)
        self.steps += 1
        y = rmsnorm(x, self.gain, self.eps)[self.code]
        if not self.exact_head:
            tok = self.decode(y > 0)
            if tok >= 0:
                return tok
        logits = bf16r(self.Ecode @ y)
        return int(np.argmax(logits))

    def decode(self, bits) -> int:
        from nesformer import core
        fid = int(sum(int(b) << i for i, b in enumerate(bits[:self.nfam])))
        p = int(sum(int(b) << i for i, b in enumerate(bits[self.nfam:])))
        return core.decode_code(fid, p)
