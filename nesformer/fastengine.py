"""engine.py's forward pass compiled with numba: the same operations in the same precision
(fp32 compute, bf16 rounding where the bf16 model rounds), without per-op interpreter overhead.
check.py and the tests compare it against engine.py and the reference trace."""

import numpy as np
from numba import njit

from nesformer.engine import Engine, F32, Layer

SQRT_HD = np.float32(128 ** 0.5)  # q / sqrt(128), as torch divides


@njit(cache=True)
def _rb(a):
    """Round a float32 array to bf16 in place (ties to even)."""
    u = a.reshape(-1).view(np.uint32)
    for i in range(u.shape[0]):
        w = u[i]
        u[i] = (w + np.uint32(0x7FFF) + ((w >> np.uint32(16)) & np.uint32(1))) \
            & np.uint32(0xFFFF0000)


@njit(cache=True, inline="always")
def _silu(v):
    if v >= np.float32(30.0):
        return v
    if v <= np.float32(-200.0):
        return np.float32(0.0) * v
    return np.float32(v / (np.float32(1.0) + np.float32(np.exp(-v))))


@njit(cache=True)
def _rmsnorm(x, gain, eps, out):
    s = np.float32(0.0)
    for i in range(x.shape[0]):
        s += x[i] * x[i]
    var = s / np.float32(x.shape[0])
    inv = np.float32(1.0) / np.float32(np.sqrt(var + np.float32(eps)))
    for i in range(x.shape[0]):
        out[i] = x[i] * inv * gain[i]
    _rb(out)


@njit(cache=True)
def _csr(data, rows, colptr, xv, out):
    """out = W @ xv for W stored by columns (CSC), skipping zero inputs. Each output still sums
    its terms in increasing column order, as a row-major dot product would."""
    for r in range(out.shape[0]):
        out[r] = 0.0
    for c in range(colptr.shape[0] - 1):
        xc = xv[c]
        if xc != np.float32(0.0):
            for j in range(colptr[c], colptr[c + 1]):
                out[rows[j]] += data[j] * xc
    _rb(out)


@njit(cache=True)
def layer_step(x, eps, g_in, g_post,
               has_attn, qd, qi, qp, conv, hist, ch_kind, ch_head, ch_dim,
               zd, zi, zp, z_vh, z_dim, ad, ai, ap, bd, bi, bp, dt_bias, A, vh, kh,
               norm_w, S, od, oi, op, ocol_vh, ocol_dim,
               gd, gi, gp, ud, ui, up, dd, di, dp, n_units):
    h = x.shape[0]
    xn = np.empty(h, np.float32)
    one = np.float32(1.0)
    if has_attn:
        _rmsnorm(x, g_in, eps, xn)
        n = conv.shape[0]
        qkv = np.empty(n, np.float32)
        _csr(qd, qi, qp, xn, qkv)
        vals = np.empty(n, np.float32)
        for c in range(n):
            hist[0, c] = hist[1, c]
            hist[1, c] = hist[2, c]
            hist[2, c] = hist[3, c]
            hist[3, c] = qkv[c]
            acc = np.float32(0.0)
            for t in range(4):
                acc += conv[c, t] * hist[t, c]
            vals[c] = acc
        _rb(vals)
        for c in range(n):
            vals[c] = _silu(vals[c])
        _rb(vals)
        nvh_all = S.shape[0]
        q = np.zeros((16, 128), np.float32)
        k = np.zeros((16, 128), np.float32)
        v = np.zeros((nvh_all, 128), np.float32)
        for c in range(n):
            kind = ch_kind[c]
            if kind == 0:
                q[ch_head[c], ch_dim[c]] = vals[c]
            elif kind == 1:
                k[ch_head[c], ch_dim[c]] = vals[c]
            else:
                v[ch_head[c], ch_dim[c]] = vals[c]
        z = np.zeros((nvh_all, 128), np.float32)
        zz = np.empty(z_vh.shape[0], np.float32)
        _csr(zd, zi, zp, xn, zz)
        for c in range(zz.shape[0]):
            z[z_vh[c], z_dim[c]] = zz[c]
        na = vh.shape[0]
        aa = np.empty(na, np.float32)
        bb = np.empty(na, np.float32)
        _csr(ad, ai, ap, xn, aa)
        _csr(bd, bi, bp, xn, bb)
        betas = np.empty(na, np.float32)
        for i in range(na):
            betas[i] = one / (one + np.float32(np.exp(-bb[i])))
        _rb(betas)
        y = np.zeros((nvh_all, 128), np.float32)
        o = np.empty(128, np.float32)
        t1 = np.empty(128, np.float32)
        kv = np.empty(128, np.float32)
        for i in range(na):
            hv, hk = vh[i], kh[i]
            beta = betas[i]
            xa = aa[i] + dt_bias[i]
            if xa > np.float32(20.0):
                spv = xa
            else:
                spv = np.float32(np.log1p(np.float32(np.exp(xa))))
            decay = np.float32(np.exp(-A[i] * spv))
            sq = np.float32(0.0)
            sk = np.float32(0.0)
            for d in range(128):
                sq += q[hk, d] * q[hk, d]
                sk += k[hk, d] * k[hk, d]
            iq = one / np.float32(np.sqrt(sq + np.float32(1e-6)))
            ik = one / np.float32(np.sqrt(sk + np.float32(1e-6)))
            if decay != one:
                for r in range(128):
                    for d in range(128):
                        S[hv, r, d] *= decay
            # the delta rule over the key's nonzero rows
            for d in range(128):
                kv[d] = 0.0
            for r in range(128):
                kr = k[hk, r] * ik
                if kr != np.float32(0.0):
                    for d in range(128):
                        kv[d] += S[hv, r, d] * kr
            for r in range(128):
                kr = k[hk, r] * ik
                if kr != np.float32(0.0):
                    for d in range(128):
                        S[hv, r, d] += kr * ((v[hv, d] - kv[d]) * beta)
            for d in range(128):
                o[d] = 0.0
            for r in range(128):
                qr = np.float32(np.float32(q[hk, r] * iq) / SQRT_HD)
                if qr != np.float32(0.0):
                    for d in range(128):
                        o[d] += qr * S[hv, r, d]
            _rb(o)
            s2 = np.float32(0.0)
            for d in range(128):
                s2 += o[d] * o[d]
            inv = one / np.float32(np.sqrt(s2 / np.float32(128.0) + np.float32(eps)))
            for d in range(128):
                t1[d] = o[d] * inv
            _rb(t1)
            for d in range(128):
                t1[d] = t1[d] * norm_w[d]
            _rb(t1)
            for d in range(128):
                t1[d] = t1[d] * _silu(z[hv, d])
            _rb(t1)
            for d in range(128):
                y[hv, d] = t1[d]
        yo = np.empty(ocol_vh.shape[0], np.float32)
        for c in range(ocol_vh.shape[0]):
            yo[c] = y[ocol_vh[c], ocol_dim[c]]
        att = np.empty(h, np.float32)
        _csr(od, oi, op, yo, att)
        for i in range(h):
            x[i] = x[i] + att[i]
        _rb(x)
    _rmsnorm(x, g_post, eps, xn)
    nu = n_units
    g = np.empty(nu, np.float32)
    u = np.empty(nu, np.float32)
    _csr(gd, gi, gp, xn, g)
    _csr(ud, ui, up, xn, u)
    for j in range(nu):
        g[j] = _silu(g[j])
    _rb(g)
    for j in range(nu):
        g[j] = g[j] * u[j]
    _rb(g)
    mo = np.empty(h, np.float32)
    _csr(dd, di, dp, g, mo)
    for i in range(h):
        x[i] = x[i] + mo[i]
    _rb(x)


def _csr_arrays(m):
    """(data, row indices, column pointers) of W in CSC form, rows sorted within columns;
    the output length is carried by the caller."""
    m = m.tocsc()
    m.sort_indices()
    return (m.data.astype(np.float32), m.indices.astype(np.int64), m.indptr.astype(np.int64))


class FastLayer:
    def __init__(self, L: Layer):
        self.L = L
        hd = L.hd
        kd = L.nk * hd
        ch = L.ch
        kind = np.where(ch < kd, 0, np.where(ch < 2 * kd, 1, 2)).astype(np.int64)
        rel = np.where(kind == 0, ch, np.where(kind == 1, ch - kd, ch - 2 * kd))
        self.ch_kind, self.ch_head, self.ch_dim = kind, (rel // hd).astype(np.int64), \
            (rel % hd).astype(np.int64)
        self.q = _csr_arrays(L.Wqkv)
        self.z = _csr_arrays(L.Wz)
        self.z_vh, self.z_dim = (L.zch // hd).astype(np.int64), (L.zch % hd).astype(np.int64)
        import scipy.sparse as sp
        self.a = _csr_arrays(sp.csr_matrix(L.Wa))
        self.b = _csr_arrays(sp.csr_matrix(L.Wb))
        self.o = _csr_arrays(L.Wo)
        self.ocol_vh = (L.ocols // hd).astype(np.int64)
        self.ocol_dim = (L.ocols % hd).astype(np.int64)
        self.g, self.u, self.d = _csr_arrays(L.Wg), _csr_arrays(L.Wu), _csr_arrays(L.Wd)
        self.conv = np.ascontiguousarray(L.conv, dtype=np.float32)
        self.hist = np.zeros((4, len(ch)), np.float32)
        self.has_attn = len(ch) > 0
        assert L.nk == 16, "the compiled kernel assumes 16 key heads"

    def __call__(self, x):
        L = self.L
        layer_step(x, np.float32(L.eps), L.g_in, L.g_post, self.has_attn,
                   *self.q, self.conv, self.hist, self.ch_kind, self.ch_head, self.ch_dim,
                   *self.z, self.z_vh, self.z_dim, *self.a, *self.b, L.dt_bias, L.A,
                   L.vh.astype(np.int64), L.kh.astype(np.int64), L.norm_w, L.S,
                   *self.o, self.ocol_vh, self.ocol_dim, *self.g, *self.u, *self.d,
                   len(L.units))
        return x


class FastEngine(Engine):
    def __init__(self, model_dir: str):
        super().__init__(model_dir)
        self.fast = [FastLayer(L) for L in self.layers]

    def load_state(self, rows):
        super().load_state(rows)
        for f in self.fast:
            f.hist[:] = 0

    def step(self, t: int) -> int:
        x = self.embed(t)
        for f in self.fast:
            f(x)
        self.steps += 1
        from nesformer.engine import rmsnorm
        y = rmsnorm(x, self.gain, self.eps)[self.code]
        if not self.exact_head:
            tok = self.decode(y > 0)
            if tok >= 0:
                return tok
        from nesformer.engine import bf16r
        return int(np.argmax(bf16r(self.Ecode @ y)))
