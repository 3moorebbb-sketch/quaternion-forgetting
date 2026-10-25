"""
Split-digits continual-learning engine (NumPy, no autograd).

Data: scikit-learn 8x8 digits, five binary tasks learned in sequence: 0/1, 2/3, 4/5, 6/7, 8/9.
Readout: 'shared' = one binary logit for every task (domain-incremental);
         'split'  = one binary logit per task, chosen by task id (task-incremental).
Networks (two hidden layers, tanh / split-tanh):
  qs3    quaternion, weight A*q with unit q on S3, Riemannian updates (exp map), A >= 0.01
  qfree  quaternion, weight w in R^4, Euclidean updates; same initial function as qs3 (w = A*q)
  real28 real MLP 64-28-28, about as many parameters as the quaternion nets
  real64 real MLP 64-64-64, same width (64 real activations) as the quaternion nets
Optimizers: sgd, adam, mep (= Adam without first-moment bias correction; see /home/claude/mep/equiv.py)
"""
import numpy as np
from sklearn.datasets import load_digits

# ---------------------------------------------------------------- data
def make_tasks(split_seed=0, test_frac=0.3):
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    rng = np.random.default_rng(split_seed)
    tasks = []
    for t in range(5):
        a, b = 2 * t, 2 * t + 1
        tr, te = [], []
        for c in (a, b):
            idx = np.where(y == c)[0]; rng.shuffle(idx)
            k = int(round(len(idx) * test_frac)); te += list(idx[:k]); tr += list(idx[k:])
        tr, te = np.array(tr), np.array(te)
        tasks.append(dict(Xtr=X[tr], ytr=(y[tr] == b).astype(float), Xte=X[te], yte=(y[te] == b).astype(float)))
    return tasks

# ---------------------------------------------------------------- quaternion helpers
def left_mat(w):
    """w: (..., 4) -> (..., 4, 4) real matrix of left Hamilton multiplication w (x) x."""
    a, b, c, d = w[..., 0], w[..., 1], w[..., 2], w[..., 3]
    return np.stack([np.stack([a, -b, -c, -d], -1), np.stack([b, a, -d, c], -1),
                     np.stack([c, d, a, -b], -1), np.stack([d, -c, b, a], -1)], -2)

def left_mat_grad(G):
    """G: (..., 4, 4) gradient w.r.t. left_mat(w) -> (..., 4) gradient w.r.t. w."""
    ga = G[..., 0, 0] + G[..., 1, 1] + G[..., 2, 2] + G[..., 3, 3]
    gb = -G[..., 0, 1] + G[..., 1, 0] - G[..., 2, 3] + G[..., 3, 2]
    gc = -G[..., 0, 2] + G[..., 1, 3] + G[..., 2, 0] - G[..., 3, 1]
    gd = -G[..., 0, 3] - G[..., 1, 2] + G[..., 2, 1] + G[..., 3, 0]
    return np.stack([ga, gb, gc, gd], -1)

def shoemake(rng, shape):
    u1, u2, u3 = rng.random(shape), rng.random(shape) * 2 * np.pi, rng.random(shape) * 2 * np.pi
    r1, r2 = np.sqrt(1 - u1), np.sqrt(u1)
    return np.stack([r1 * np.sin(u2), r1 * np.cos(u2), r2 * np.sin(u3), r2 * np.cos(u3)], -1)

# ---------------------------------------------------------------- networks
# Params: dict name -> array. Every network exposes init / forward / backward / step-geometry.
class QuatNet:
    """Layers (in quaternions -> out quaternions): 16 -> 16 -> 16 -> heads.
    Head: one output quaternion per logit; the logit is its i component."""
    def __init__(self, unit, n_out, hq=16):
        self.unit, self.n_out, self.dims = unit, n_out, [(16, hq), (hq, hq), (hq, n_out)]
    def init(self, rng):
        P = {}
        for l, (i, o) in enumerate(self.dims):
            q = shoemake(rng, (o, i))
            A = np.maximum(0.15, np.abs(rng.standard_normal((o, i)) * np.sqrt(2.0 / (4 * i))))
            if self.unit: P[f'q{l}'], P[f'A{l}'] = q, A
            else: P[f'w{l}'] = q * A[..., None]
            P[f'b{l}'] = np.zeros((o, 4))
        return P
    def weights(self, P, l):
        return P[f'q{l}'] * P[f'A{l}'][..., None] if self.unit else P[f'w{l}']
    def forward(self, P, X):
        h = X.reshape(len(X), 16, 4); cache = [h]
        for l, (i, o) in enumerate(self.dims):
            W = left_mat(self.weights(P, l)).transpose(0, 2, 1, 3).reshape(4 * o, 4 * i)   # real block matrix
            z = (h.reshape(len(h), 4 * i) @ W.T).reshape(len(h), o, 4) + P[f'b{l}']
            h = np.tanh(z) if l < 2 else z
            cache.append(h)
        return h[:, :, 1], cache                                  # logits (n, n_out), i component
    def features(self, P, X):
        return self.forward(P, X)[1][2].reshape(len(X), -1)       # last hidden layer, 64 reals
    def backward(self, P, cache, dlogit):
        G = {}; d = np.zeros_like(cache[-1]); d[:, :, 1] = dlogit
        for l in (2, 1, 0):
            hin, hout = cache[l], cache[l + 1]
            if l < 2: d = d * (1 - hout ** 2)
            G[f'b{l}'] = d.sum(0)
            i, o = self.dims[l]; n = len(d)
            gW = d.reshape(n, 4 * o).T @ hin.reshape(n, 4 * i)                 # (4o, 4i)
            gw = left_mat_grad(gW.reshape(o, 4, i, 4).transpose(0, 2, 1, 3))
            W = left_mat(self.weights(P, l)).transpose(0, 2, 1, 3).reshape(4 * o, 4 * i)
            dprev = (d.reshape(n, 4 * o) @ W).reshape(n, i, 4)
            if self.unit:
                q, A = P[f'q{l}'], P[f'A{l}']
                G[f'q{l}'] = gw * A[..., None]; G[f'A{l}'] = (gw * q).sum(-1)
            else: G[f'w{l}'] = gw
            d = dprev
        return G
    def project(self, P, G):                                     # tangent-space projection on S3
        if not self.unit: return G
        for l in range(3):
            q = P[f'q{l}']; g = G[f'q{l}']; G[f'q{l}'] = g - (g * q).sum(-1, keepdims=True) * q
        return G
    def apply(self, P, S, lr):                                   # S = descent direction (already projected)
        for k, s in S.items():
            if self.unit and k[0] == 'q':
                q = P[k]; n = np.linalg.norm(s, axis=-1, keepdims=True)
                ang = lr * n; safe = np.where(n > 1e-12, n, 1.0)
                nq = np.cos(ang) * q - np.sin(ang) * s / safe
                nq = np.where(n > 1e-12, nq, q); P[k] = nq / np.linalg.norm(nq, axis=-1, keepdims=True)
            elif self.unit and k[0] == 'A':
                P[k] = np.maximum(0.01, P[k] - lr * s)
            else:
                P[k] = P[k] - lr * s

class RealNet:
    def __init__(self, H, n_out):
        self.H, self.n_out, self.unit = H, n_out, False
    def init(self, rng):
        H = self.H
        return {'W0': rng.standard_normal((H, 64)) * np.sqrt(1 / 64), 'b0': np.zeros(H),
                'W1': rng.standard_normal((H, H)) * np.sqrt(1 / H), 'b1': np.zeros(H),
                'W2': rng.standard_normal((self.n_out, H)) * np.sqrt(1 / H), 'b2': np.zeros(self.n_out)}
    def forward(self, P, X):
        h0 = X; h1 = np.tanh(h0 @ P['W0'].T + P['b0']); h2 = np.tanh(h1 @ P['W1'].T + P['b1'])
        return h2 @ P['W2'].T + P['b2'], [h0, h1, h2]
    def features(self, P, X): return self.forward(P, X)[1][2]
    def backward(self, P, cache, dlogit):
        h0, h1, h2 = cache; G = {}
        G['W2'] = dlogit.T @ h2; G['b2'] = dlogit.sum(0)
        d2 = (dlogit @ P['W2']) * (1 - h2 ** 2); G['W1'] = d2.T @ h1; G['b1'] = d2.sum(0)
        d1 = (d2 @ P['W1']) * (1 - h1 ** 2); G['W0'] = d1.T @ h0; G['b0'] = d1.sum(0)
        return G
    def project(self, P, G): return G
    def apply(self, P, S, lr):
        for k, s in S.items(): P[k] = P[k] - lr * s

def make_net(arch, readout):
    n_out = 1 if readout == 'shared' else 5
    return {'qs3': lambda: QuatNet(True, n_out), 'qfree': lambda: QuatNet(False, n_out),
            'real28': lambda: RealNet(28, n_out), 'real64': lambda: RealNet(64, n_out)}[arch]()

def n_params(net, P): return sum(v.size for v in P.values())

# ---------------------------------------------------------------- optimizers
class Opt:
    """sgd | adam | mep. Adam-type moments are computed on the (tangent-projected) gradient,
    exactly as the toy engine trains Task A; the step then goes through the network's own geometry."""
    def __init__(self, kind, lr):
        self.kind, self.lr, self.t, self.m, self.v = kind, lr, 0, {}, {}
    def step(self, net, P, G):
        G = net.project(P, G)
        if self.kind == 'sgd': return net.apply(P, G, self.lr)
        self.t += 1; S = {}
        for k, g in G.items():
            m = self.m.get(k, 0) * 0.9 + 0.1 * g; v = self.v.get(k, 0) * 0.999 + 0.001 * g * g
            self.m[k], self.v[k] = m, v
            vh = v / (1 - 0.999 ** self.t)
            mh = m / (1 - 0.9 ** self.t) if self.kind == 'adam' else m      # mep: no first-moment correction
            S[k] = mh / (np.sqrt(vh) + 1e-8)
        net.apply(P, S, self.lr)

# ---------------------------------------------------------------- training
def bce_grad(logit, y):
    p = 1 / (1 + np.exp(-logit)); return (p - y) / len(y)

def logits_for(net, P, X, task, readout):
    L, cache = net.forward(P, X)
    return (L[:, 0] if readout == 'shared' else L[:, task]), L, cache

def accuracy(net, P, X, y, task, readout):
    z = logits_for(net, P, X, task, readout)[0]
    return 100.0 * np.mean((z >= 0) == (y == 1))

def run(arch, readout, opt_kind, lr, seed, tasks, epochs=20, batch=32, probe=True):
    rng = np.random.default_rng(seed * 7919 + 1)
    net = make_net(arch, readout); P = net.init(rng); opt = Opt(opt_kind, lr)
    order = np.random.default_rng(seed * 104729 + 3)
    R = np.zeros((5, 5))
    for t, T in enumerate(tasks):
        n = len(T['ytr'])
        for ep in range(epochs):
            perm = order.permutation(n)
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                z, L, cache = logits_for(net, P, T['Xtr'][idx], t, readout)
                dL = np.zeros_like(L); col = 0 if readout == 'shared' else t
                dL[:, col] = bce_grad(z, T['ytr'][idx])
                opt.step(net, P, net.backward(P, cache, dL))
        for j in range(5): R[t, j] = accuracy(net, P, tasks[j]['Xte'], tasks[j]['yte'], j, readout)
    acc = R[4].mean(); fgt = np.mean([R[j:, j].max() - R[4, j] for j in range(4)]); la = np.mean(np.diag(R))
    out = dict(arch=arch, readout=readout, opt=opt_kind, lr=lr, seed=seed, acc=acc, fgt=fgt, la=la,
               R=R.round(2).tolist(), P=n_params(net, P), finite=bool(np.isfinite(acc)))
    if probe:   # linear probe per task on final last-hidden-layer features (fit on train, score on test)
        from sklearn.linear_model import LogisticRegression
        pa = []
        for j, T in enumerate(tasks):
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(net.features(P, T['Xtr']), T['ytr'])
            pa.append(100 * clf.score(net.features(P, T['Xte']), T['yte']))
        out['probe'] = float(np.mean(pa)); out['probe_old'] = float(np.mean(pa[:4]))
        out['head_old'] = float(np.mean(R[4, :4]))
    return out
