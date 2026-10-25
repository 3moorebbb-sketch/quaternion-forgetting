// ============================================================================
// Q-CL ENGINE — continual learning on a 2-task toy problem with pluggable
// architectures and six Task-B update rules. Pure JS, no DOM; the same file
// runs in Node (harness), in the page, and in the page's benchmark Worker.
// Everything is driven by seeded RNGs so (seed, config) reproduces exactly.
//
// Architectures
//   qs3    quaternion MLP, weights A·q with q ∈ S³ (Riemannian updates)   292 params
//   qfree  same Hamilton-product network, weights w ∈ ℝ⁴ (Euclidean)       244 params
//          — identical initial function to qs3 (w = A·q); only the geometry differs
//   qs3sep / qfreesep  same, but 6→2 output quaternions: head A = out₀·i,
//          head B = out₁·j, so each task has private last-layer weights        326 / 272 params
//   real14 real tanh MLP 2→14→14→2, parameter-matched                     282 params
//   real24 real tanh MLP 2→24→24→2, width-matched (24 real activations)   722 params
// ============================================================================
var QCL = (function () {
  'use strict';

  // ---------- RNG ----------
  function makeRng(seed) {
    var a = (seed >>> 0) || 1;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function randn(rng) {
    var u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // ---------- Tasks ----------
  var R2 = 0.45, LOGIT_SCALE = 3.5;
  function labelA(x1, x2) { return (x1 * x1 + x2 * x2) < R2 ? 1 : 0; }
  function labelB(x1, x2) { return (x1 - x2) > 0 ? 1 : 0; }
  function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

  // Every architecture exposes:
  //   P, init(rng), forward(th,x1,x2) -> {zA, zB, ...cache}, backward(th, fw, head, c, g)
  //   (adds c · ∂out_head/∂θ to g, where z_head = LOGIT_SCALE · out_head),
  //   tangentProject(th, v), applyStep(th, v, lr), anchorGrad(th, ref, lam, g), quatBlocks

  // ======================= qs3 / qfree: quaternion networks =======================
  // sepHeads: two output quaternion neurons, head A reads out₀·i and head B reads out₁·j,
  // so each task gets private last-layer weights (as a real MLP's two output rows do).
  function makeQuaternionArch(constrained, sepHeads) {
    var LAYERS = [[1, 6], [6, 6], [6, sepHeads ? 2 : 1]], LAYOUT = [], P = 0, li;
    var IA = 1, IB = sepHeads ? 6 : 2;
    for (li = 0; li < LAYERS.length; li++) {
      var inS = LAYERS[li][0], outS = LAYERS[li][1], n = inS * outS;
      if (constrained) { LAYOUT.push({ inS: inS, outS: outS, n: n, q: P, A: P + 4 * n, b: P + 5 * n }); P += 5 * n + 4 * outS; }
      else { LAYOUT.push({ inS: inS, outS: outS, n: n, q: P, A: -1, b: P + 4 * n }); P += 4 * n + 4 * outS; }
    }
    var NQ = 0, QB = [];
    for (li = 0; li < LAYOUT.length; li++) { NQ += LAYOUT[li].n; for (var w0 = 0; w0 < LAYOUT[li].n; w0++) QB.push(LAYOUT[li].q + 4 * w0); }

    function init(rng) {
      var th = new Float64Array(P);
      for (var l = 0; l < LAYOUT.length; l++) {
        var L = LAYOUT[l];
        for (var w = 0; w < L.n; w++) {
          // Shoemake (1992) uniform unit quaternion
          var u1 = rng(), u2 = rng() * 2 * Math.PI, u3 = rng() * 2 * Math.PI;
          var r1 = Math.sqrt(1 - u1), r2 = Math.sqrt(u1), o = L.q + 4 * w;
          th[o] = r1 * Math.sin(u2); th[o + 1] = r1 * Math.cos(u2);
          th[o + 2] = r2 * Math.sin(u3); th[o + 3] = r2 * Math.cos(u3);
          // He-normal magnitude (fan-in counts 4 real components), floored
          var A = Math.max(0.15, Math.abs(randn(rng) * Math.sqrt(2 / (L.inS * 4))));
          if (constrained) th[L.A + w] = A;
          else { th[o] *= A; th[o + 1] *= A; th[o + 2] *= A; th[o + 3] *= A; }
        }
      }
      return th;
    }

    // Hidden layers: split tanh. Output: linear quaternion; head A reads i, head B reads j.
    function forward(th, x1, x2) {
      var inputs = [new Float64Array([0, x1, x2, 0])], acts = [];
      for (var l = 0; l < LAYOUT.length; l++) {
        var L = LAYOUT[l], X = inputs[l], Y = new Float64Array(4 * L.outS);
        for (var o = 0; o < L.outS; o++) {
          var bo = L.b + 4 * o, s0 = th[bo], s1 = th[bo + 1], s2 = th[bo + 2], s3 = th[bo + 3];
          for (var i = 0; i < L.inS; i++) {
            var w = o * L.inS + i, qo = L.q + 4 * w, A = constrained ? th[L.A + w] : 1;
            var a0 = th[qo], a1 = th[qo + 1], a2 = th[qo + 2], a3 = th[qo + 3];
            var x0 = X[4 * i], xi = X[4 * i + 1], xj = X[4 * i + 2], xk = X[4 * i + 3];
            s0 += A * (a0 * x0 - a1 * xi - a2 * xj - a3 * xk);
            s1 += A * (a0 * xi + a1 * x0 + a2 * xk - a3 * xj);
            s2 += A * (a0 * xj - a1 * xk + a2 * x0 + a3 * xi);
            s3 += A * (a0 * xk + a1 * xj - a2 * xi + a3 * x0);
          }
          if (l < LAYOUT.length - 1) { s0 = Math.tanh(s0); s1 = Math.tanh(s1); s2 = Math.tanh(s2); s3 = Math.tanh(s3); }
          Y[4 * o] = s0; Y[4 * o + 1] = s1; Y[4 * o + 2] = s2; Y[4 * o + 3] = s3;
        }
        acts.push(Y); inputs.push(Y);
      }
      var out = inputs[LAYOUT.length];
      return { zA: LOGIT_SCALE * out[IA], zB: LOGIT_SCALE * out[IB], inputs: inputs, acts: acts };
    }

    function backward(th, fw, head, c, g) {
      var delta = new Float64Array(4 * LAYERS[LAYERS.length - 1][1]); delta[head === 'A' ? IA : IB] = c;
      for (var l = LAYOUT.length - 1; l >= 0; l--) {
        var L = LAYOUT[l], X = fw.inputs[l], Y = fw.acts[l], next = new Float64Array(4 * L.inS);
        for (var o = 0; o < L.outS; o++) {
          var d0 = delta[4 * o], d1 = delta[4 * o + 1], d2 = delta[4 * o + 2], d3 = delta[4 * o + 3];
          if (l < LAYOUT.length - 1) {
            d0 *= 1 - Y[4 * o] * Y[4 * o]; d1 *= 1 - Y[4 * o + 1] * Y[4 * o + 1];
            d2 *= 1 - Y[4 * o + 2] * Y[4 * o + 2]; d3 *= 1 - Y[4 * o + 3] * Y[4 * o + 3];
          }
          var bo = L.b + 4 * o;
          g[bo] += d0; g[bo + 1] += d1; g[bo + 2] += d2; g[bo + 3] += d3;
          for (var i = 0; i < L.inS; i++) {
            var w = o * L.inS + i, qo = L.q + 4 * w, A = constrained ? th[L.A + w] : 1;
            var a0 = th[qo], a1 = th[qo + 1], a2 = th[qo + 2], a3 = th[qo + 3];
            var x0 = X[4 * i], xi = X[4 * i + 1], xj = X[4 * i + 2], xk = X[4 * i + 3];
            if (constrained) {
              // dL/dA = <d, q x>
              var p0 = a0 * x0 - a1 * xi - a2 * xj - a3 * xk, p1 = a0 * xi + a1 * x0 + a2 * xk - a3 * xj;
              var p2 = a0 * xj - a1 * xk + a2 * x0 + a3 * xi, p3 = a0 * xk + a1 * xj - a2 * xi + a3 * x0;
              g[L.A + w] += d0 * p0 + d1 * p1 + d2 * p2 + d3 * p3;
            }
            // dL/dq = A * d * conj(x)
            g[qo]     += A * (d0 * x0 + d1 * xi + d2 * xj + d3 * xk);
            g[qo + 1] += A * (-d0 * xi + d1 * x0 - d2 * xk + d3 * xj);
            g[qo + 2] += A * (-d0 * xj + d1 * xk + d2 * x0 - d3 * xi);
            g[qo + 3] += A * (-d0 * xk - d1 * xj + d2 * xi + d3 * x0);
            // dL/dx = A * conj(q) * d
            next[4 * i]     += A * (a0 * d0 + a1 * d1 + a2 * d2 + a3 * d3);
            next[4 * i + 1] += A * (a0 * d1 - a1 * d0 - a2 * d3 + a3 * d2);
            next[4 * i + 2] += A * (a0 * d2 + a1 * d3 - a2 * d0 - a3 * d1);
            next[4 * i + 3] += A * (a0 * d3 - a1 * d2 + a2 * d1 - a3 * d0);
          }
        }
        delta = next;
      }
    }

    // Project the quaternion blocks of a vector onto T_q S³ at the current point.
    function tangentProject(th, v) {
      if (!constrained) return;
      for (var l = 0; l < LAYOUT.length; l++) {
        var L = LAYOUT[l];
        for (var w = 0; w < L.n; w++) {
          var o = L.q + 4 * w;
          var d = th[o] * v[o] + th[o + 1] * v[o + 1] + th[o + 2] * v[o + 2] + th[o + 3] * v[o + 3];
          v[o] -= d * th[o]; v[o + 1] -= d * th[o + 1]; v[o + 2] -= d * th[o + 2]; v[o + 3] -= d * th[o + 3];
        }
      }
    }

    // qs3: exponential map on S³ for q (after re-projecting), Euclidean for A (floored) and b.
    // qfree: plain Euclidean step on everything.
    function applyStep(th, v, lr) {
      if (!constrained) { for (var e = 0; e < P; e++) th[e] -= lr * v[e]; return; }
      tangentProject(th, v);
      for (var l = 0; l < LAYOUT.length; l++) {
        var L = LAYOUT[l];
        for (var w = 0; w < L.n; w++) {
          var o = L.q + 4 * w, vx = v[o], vy = v[o + 1], vz = v[o + 2], vw = v[o + 3];
          var nv = Math.sqrt(vx * vx + vy * vy + vz * vz + vw * vw);
          if (nv > 1e-12) {
            var c = Math.cos(lr * nv), s = Math.sin(lr * nv) / nv;
            var q0 = c * th[o] - s * vx, q1 = c * th[o + 1] - s * vy, q2 = c * th[o + 2] - s * vz, q3 = c * th[o + 3] - s * vw;
            var nq = Math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
            th[o] = q0 / nq; th[o + 1] = q1 / nq; th[o + 2] = q2 / nq; th[o + 3] = q3 / nq;
          }
          th[L.A + w] = Math.max(0.01, th[L.A + w] - lr * v[L.A + w]);
        }
        for (var k = L.b; k < L.b + 4 * L.outS; k++) th[k] -= lr * v[k];
      }
    }

    // qs3: λ·[d_S³(q,q*)² + ‖A−A*‖² + ‖b−b*‖²].  qfree: λ·‖θ−θ*‖².
    function anchorGrad(th, ref, lam, g) {
      if (!constrained) { for (var e = 0; e < P; e++) g[e] += lam * 2 * (th[e] - ref[e]); return; }
      for (var l = 0; l < LAYOUT.length; l++) {
        var L = LAYOUT[l];
        for (var w = 0; w < L.n; w++) {
          var o = L.q + 4 * w;
          var inner = th[o] * ref[o] + th[o + 1] * ref[o + 1] + th[o + 2] * ref[o + 2] + th[o + 3] * ref[o + 3];
          var sg = inner >= 0 ? 1 : -1, ai = Math.min(Math.abs(inner), 1 - 1e-9);
          var d = 2 * Math.acos(ai);                // geodesic distance on SO(3) cover
          var sh = Math.sqrt(1 - ai * ai);
          // ∂(d²)/∂q (Euclidean); tangent projection happens in applyStep
          var f = sh > 1e-9 ? -4 * d / sh : -4;
          for (var c = 0; c < 4; c++) g[o + c] += lam * f * sg * ref[o + c];
          g[L.A + w] += lam * 2 * (th[L.A + w] - ref[L.A + w]);
        }
        for (var k = L.b; k < L.b + 4 * L.outS; k++) g[k] += lam * 2 * (th[k] - ref[k]);
      }
    }

    return { P: P, NQ: NQ, quatBlocks: QB, unit: constrained, LAYOUT: LAYOUT, sepHeads: !!sepHeads, init: init, forward: forward, backward: backward,
             tangentProject: tangentProject, applyStep: applyStep, anchorGrad: anchorGrad };
  }

  // ======================= real tanh MLP 2 → H → H → 2 =======================
  function makeRealArch(H) {
    var W1 = 0, b1 = W1 + 2 * H, W2 = b1 + H, b2 = W2 + H * H, W3 = b2 + H, b3 = W3 + 2 * H, P = b3 + 2;
    function init(rng) {
      // LeCun-normal weights (standard for tanh), zero biases
      var th = new Float64Array(P), i;
      for (i = 0; i < 2 * H; i++) th[W1 + i] = randn(rng) * Math.sqrt(1 / 2);
      for (i = 0; i < H * H; i++) th[W2 + i] = randn(rng) * Math.sqrt(1 / H);
      for (i = 0; i < 2 * H; i++) th[W3 + i] = randn(rng) * Math.sqrt(1 / H);
      return th;
    }
    function forward(th, x1, x2) {
      var h1 = new Float64Array(H), h2 = new Float64Array(H), j, k;
      for (j = 0; j < H; j++) h1[j] = Math.tanh(th[W1 + 2 * j] * x1 + th[W1 + 2 * j + 1] * x2 + th[b1 + j]);
      for (j = 0; j < H; j++) { var s = th[b2 + j], r = W2 + j * H; for (k = 0; k < H; k++) s += th[r + k] * h1[k]; h2[j] = Math.tanh(s); }
      var o0 = th[b3], o1 = th[b3 + 1];
      for (k = 0; k < H; k++) { o0 += th[W3 + k] * h2[k]; o1 += th[W3 + H + k] * h2[k]; }
      return { zA: LOGIT_SCALE * o0, zB: LOGIT_SCALE * o1, x1: x1, x2: x2, h1: h1, h2: h2 };
    }
    function backward(th, fw, head, c, g) {
      var row = head === 'A' ? 0 : 1, d2 = new Float64Array(H), d1 = new Float64Array(H), j, k;
      g[b3 + row] += c;
      for (k = 0; k < H; k++) { g[W3 + row * H + k] += c * fw.h2[k]; d2[k] = c * th[W3 + row * H + k] * (1 - fw.h2[k] * fw.h2[k]); }
      for (j = 0; j < H; j++) {
        var r = W2 + j * H; g[b2 + j] += d2[j];
        for (k = 0; k < H; k++) { g[r + k] += d2[j] * fw.h1[k]; d1[k] += d2[j] * th[r + k]; }
      }
      for (k = 0; k < H; k++) {
        var dk = d1[k] * (1 - fw.h1[k] * fw.h1[k]);
        g[W1 + 2 * k] += dk * fw.x1; g[W1 + 2 * k + 1] += dk * fw.x2; g[b1 + k] += dk;
      }
    }
    return { P: P, NQ: 0, quatBlocks: [], unit: false, init: init, forward: forward, backward: backward,
             tangentProject: function () {},
             applyStep: function (th, v, lr) { for (var e = 0; e < P; e++) th[e] -= lr * v[e]; },
             anchorGrad: function (th, ref, lam, g) { for (var e = 0; e < P; e++) g[e] += lam * 2 * (th[e] - ref[e]); } };
  }

  var ARCHS = {
    qs3: makeQuaternionArch(true),
    qfree: makeQuaternionArch(false),
    qs3sep: makeQuaternionArch(true, true),
    qfreesep: makeQuaternionArch(false, true),
    real14: makeRealArch(14),
    real24: makeRealArch(24)
  };
  var ARCH_INFO = {
    qs3: { name: 'Quaternion · S³', short: 'Quat S³', desc: 'A·q, q on the unit 3-sphere, Riemannian updates' },
    qfree: { name: 'Quaternion · free', short: 'Quat free', desc: 'same Hamilton network and init, w ∈ ℝ⁴, Euclidean updates' },
    qs3sep: { name: 'Quaternion · S³ · split heads', short: 'Quat S³ split', desc: 'as S³, but each task reads its own output quaternion' },
    qfreesep: { name: 'Quaternion · free · split heads', short: 'Quat free split', desc: 'as free, but each task reads its own output quaternion' },
    real14: { name: 'Real MLP · 14', short: 'Real 14', desc: '2→14→14→2 tanh, parameter-matched' },
    real24: { name: 'Real MLP · 24', short: 'Real 24', desc: '2→24→24→2 tanh, width-matched' }
  };
  for (var an in ARCHS) ARCH_INFO[an].P = ARCHS[an].P;
  function arch(name) { var a = ARCHS[name || 'qs3']; if (!a) throw new Error('unknown arch ' + name); return a; }

  // Add the gradient of mean BCE on one head over a batch, times weight.
  function addBCEGrad(ar, th, pts, head, weight, g) {
    var n = pts.length;
    for (var k = 0; k < n; k++) {
      var p = pts[k], fw = ar.forward(th, p[0], p[1]);
      var z = head === 'A' ? fw.zA : fw.zB;
      ar.backward(th, fw, head, weight * LOGIT_SCALE * (sigmoid(z) - p[2]) / n, g);
    }
  }

  // ---------- Evaluation on a fixed 50×50 grid (both heads, one forward each) ----------
  var GRID = [];
  (function () {
    for (var gy = 0; gy < 50; gy++) for (var gx = 0; gx < 50; gx++) {
      var x1 = gx / 49 * 2 - 1, x2 = 1 - gy / 49 * 2;
      GRID.push([x1, x2, labelA(x1, x2), labelB(x1, x2)]);
    }
  })();
  var MAJORITY_FLOOR = (function () { var c = 0; for (var i = 0; i < GRID.length; i++) c += GRID[i][2]; return 100 * (GRID.length - c) / GRID.length; })();

  function evaluate(th, ar) {
    ar = ar || ARCHS.qs3;
    var ca = 0, cb = 0;
    for (var i = 0; i < GRID.length; i++) {
      var p = GRID[i], fw = ar.forward(th, p[0], p[1]);
      if ((fw.zA >= 0 ? 1 : 0) === p[2]) ca++;
      if ((fw.zB >= 0 ? 1 : 0) === p[3]) cb++;
    }
    return [100 * ca / GRID.length, 100 * cb / GRID.length];
  }

  function sampleBatch(rng, n, labelFn) {
    var b = [];
    for (var i = 0; i < n; i++) { var x1 = rng() * 2 - 1, x2 = rng() * 2 - 1; b.push([x1, x2, labelFn(x1, x2)]); }
    return b;
  }

  // ---------- Task A pre-training: (Riemannian) Adam, convergence guard ----------
  // Stepwise so the page can yield between chunks; trainTaskA() just drains it.
  // Restarts from a fresh init (max 20 attempts) until the fixed grid reads >= 95%.
  function makeTrainer(rng, ar) {
    ar = ar || ARCHS.qs3;
    var P = ar.P, T = { attempts: 0, s: 0, acc: 0, done: false, th: null, m: null, v: null };
    function restart() {
      T.attempts++; T.th = ar.init(rng); T.s = 0;
      T.m = new Float64Array(P); T.v = new Float64Array(P);
    }
    function finishAttempt() {
      T.acc = evaluate(T.th, ar)[0];
      if (T.acc >= 95 || T.attempts >= 20) T.done = true; else restart();
    }
    restart();
    T.step = function (n) {
      var b1 = 0.9, b2 = 0.999;
      for (var c = 0; c < n && !T.done; c++) {
        var s = ++T.s, th = T.th, m = T.m, v = T.v;
        var g = new Float64Array(P);
        addBCEGrad(ar, th, sampleBatch(rng, 32, labelA), 'A', 1, g);
        ar.tangentProject(th, g);
        var st = new Float64Array(P);
        for (var i = 0; i < P; i++) {
          m[i] = b1 * m[i] + (1 - b1) * g[i]; v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
          st[i] = (m[i] / (1 - Math.pow(b1, s))) / (Math.sqrt(v[i] / (1 - Math.pow(b2, s))) + 1e-8);
        }
        ar.applyStep(th, st, 0.035);
        if (s % 60 === 0) {
          T.acc = evaluate(th, ar)[0];
          if (T.acc >= 95 && s >= 180) { finishAttempt(); continue; }
        }
        if (s >= 1500) finishAttempt();
      }
      return T.done;
    };
    T.result = function () { return { th: T.th, acc: T.acc, attempts: T.attempts }; };
    return T;
  }
  function trainTaskA(rng, ar) {
    var T = makeTrainer(rng, ar);
    while (!T.step(100)) {}
    return T.result();
  }

  // ---------- Episodic memory shared by every memory-based method ----------
  // M Task-A inputs drawn uniformly at checkpoint time. Replay uses their labels,
  // output anchoring uses the checkpoint's logits on them, T-OGD uses the
  // checkpoint's output gradients on them. Same points, same storage budget.
  function makeMemory(th, rng, M, ar) {
    ar = ar || ARCHS.qs3;
    var pts = [];
    for (var i = 0; i < M; i++) {
      var x1 = rng() * 2 - 1, x2 = rng() * 2 - 1;
      pts.push([x1, x2, labelA(x1, x2), ar.forward(th, x1, x2).zA]);
    }
    return pts;
  }

  // Symmetric eigendecomposition (cyclic Jacobi).
  function jacobiEigen(S, n) {
    var a = S.slice(), V = new Float64Array(n * n);
    for (var i = 0; i < n; i++) V[i * n + i] = 1;
    for (var sweep = 0; sweep < 60; sweep++) {
      var off = 0;
      for (var p = 0; p < n; p++) for (var q = p + 1; q < n; q++) off += a[p * n + q] * a[p * n + q];
      if (off < 1e-22) break;
      for (p = 0; p < n; p++) for (q = p + 1; q < n; q++) {
        var apq = a[p * n + q];
        if (Math.abs(apq) < 1e-300) continue;
        var theta = (a[q * n + q] - a[p * n + p]) / (2 * apq);
        var t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        var c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (var k = 0; k < n; k++) {
          var akp = a[k * n + p], akq = a[k * n + q];
          a[k * n + p] = c * akp - s * akq; a[k * n + q] = s * akp + c * akq;
        }
        for (k = 0; k < n; k++) {
          var apk = a[p * n + k], aqk = a[q * n + k];
          a[p * n + k] = c * apk - s * aqk; a[q * n + k] = s * apk + c * aqk;
        }
        for (k = 0; k < n; k++) {
          var vkp = V[k * n + p], vkq = V[k * n + q];
          V[k * n + p] = c * vkp - s * vkq; V[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
    var vals = []; for (i = 0; i < n; i++) vals.push(a[i * n + i]);
    return { vals: vals, V: V };
  }

  // OGD basis: (tangent-space) output gradients ∇θ zA(x_m) on the memory points,
  // reduced to the top-K principal directions (SVD via the M×M Gram matrix).
  function buildOGDBasis(th, mem, K, ar) {
    ar = ar || ARCHS.qs3;
    var P = ar.P, M = mem.length, J = [];
    for (var m = 0; m < M; m++) {
      var g = new Float64Array(P), fw = ar.forward(th, mem[m][0], mem[m][1]);
      ar.backward(th, fw, 'A', LOGIT_SCALE, g);
      ar.tangentProject(th, g);
      J.push(g);
    }
    var G = new Float64Array(M * M);
    for (var i = 0; i < M; i++) for (var j = i; j < M; j++) {
      var s = 0; for (var k = 0; k < P; k++) s += J[i][k] * J[j][k];
      G[i * M + j] = G[j * M + i] = s;
    }
    var eg = jacobiEigen(G, M), order = [];
    for (i = 0; i < M; i++) order.push(i);
    order.sort(function (a, b) { return eg.vals[b] - eg.vals[a]; });
    var total = 0; for (i = 0; i < M; i++) total += Math.max(0, eg.vals[i]);
    var basis = [], captured = 0;
    for (var r = 0; r < Math.min(K, M); r++) {
      var idx = order[r], lam = eg.vals[idx];
      if (lam <= 1e-10 * eg.vals[order[0]]) break;
      var u = new Float64Array(P);
      for (m = 0; m < M; m++) { var c = eg.V[m * M + idx]; if (c !== 0) for (k = 0; k < P; k++) u[k] += c * J[m][k]; }
      var nu = 0; for (k = 0; k < P; k++) nu += u[k] * u[k]; nu = Math.sqrt(nu);
      for (k = 0; k < P; k++) u[k] /= nu;
      basis.push(u); captured += lam;
    }
    return { basis: basis, energy: total > 0 ? captured / total : 0, rank: order.filter(function (i) { return eg.vals[i] > 1e-10 * eg.vals[order[0]]; }).length };
  }

  // ---------- Task-B update rules ----------
  // Every method sees the same Task-B batch each step. Memory methods additionally
  // draw the same memory minibatch indices (from a shared RNG stream).
  var METHODS = {
    sgd: { name: 'Vanilla SGD', knob: null },
    wanchor: { name: 'Weight anchor', knob: 'λ', note: 'λ·‖θ−θ*‖² (geodesic on S³ for qs3)' },
    oanchor: { name: 'Output anchor', knob: 'μ', note: 'μ·mean_m (z_A(x_m) − z_A*(x_m))²' },
    replay: { name: 'Replay', knob: 'ρ', note: 'ρ·BCE_A on stored labelled points' },
    togd: { name: 'Tangent OGD', knob: 'K', note: 'project ∇ onto ⊥ span(top-K ∇θ z_A), basis frozen at checkpoint' },
    togdr: { name: 'Tangent OGD, refreshed', knob: 'every', note: 'same projection, basis recomputed at the current θ every N steps' }
  };

  function stepMethod(ar, method, th, ctx, batchB, memIdx, cfg, lr) {
    var P = ar.P, g = new Float64Array(P), k, j;
    addBCEGrad(ar, th, batchB, 'B', 1, g);
    if (method === 'wanchor') {
      ar.anchorGrad(th, ctx.ref, cfg.lambda, g);
    } else if (method === 'oanchor') {
      var mu = cfg.mu, n = memIdx.length;
      for (j = 0; j < n; j++) {
        var p = ctx.mem[memIdx[j]], fw = ar.forward(th, p[0], p[1]);
        ar.backward(th, fw, 'A', mu * 2 * (fw.zA - p[3]) * LOGIT_SCALE / n, g);
      }
    } else if (method === 'replay') {
      var pts = []; for (j = 0; j < memIdx.length; j++) pts.push(ctx.mem[memIdx[j]]);
      addBCEGrad(ar, th, pts, 'A', cfg.rho, g);
    }
    ar.tangentProject(th, g);
    if (method === 'togd' || method === 'togdr') {
      var B = method === 'togd' ? ctx.basis : ctx.basisR;
      for (var b = 0; b < B.length; b++) {
        var u = B[b], dot = 0;
        for (k = 0; k < P; k++) dot += g[k] * u[k];
        for (k = 0; k < P; k++) g[k] -= dot * u[k];
      }
    }
    ar.applyStep(th, g, lr);
  }

  // ---------- One full experiment ----------
  // cfg: {arch, seed, lr, steps, batch, memSize, memBatch, lambda, mu, rho, K, refresh, evalEvery, methods}
  var DEFAULTS = { arch: 'qs3', seed: 1, lr: 0.035, steps: 1000, batch: 20, memSize: 100, memBatch: 20,
                   lambda: 0.05, mu: 0.1, rho: 1, K: 15, refresh: 10, evalEvery: 50,
                   methods: ['sgd', 'wanchor', 'oanchor', 'replay', 'togd', 'togdr'] };

  function setup(cfg, ckpt) {
    cfg = Object.assign({}, DEFAULTS, cfg || {});
    var ar = arch(cfg.arch);
    var rngA = makeRng(cfg.seed * 7919 + 1);
    var A = ckpt || trainTaskA(rngA, ar);
    var rngM = makeRng(cfg.seed * 7919 + 2);
    var mem = makeMemory(A.th, rngM, cfg.memSize, ar);
    // Static and refreshed OGD may use different K when per-method settings are given
    var Ks = (cfg.per && cfg.per.togd && cfg.per.togd.K) || cfg.K, Kr = (cfg.per && cfg.per.togdr && cfg.per.togdr.K) || cfg.K;
    var ogd = buildOGDBasis(A.th, mem, Ks, ar), ogdR = Kr === Ks ? ogd : buildOGDBasis(A.th, mem, Kr, ar);
    var models = {};
    for (var i = 0; i < cfg.methods.length; i++) models[cfg.methods[i]] = new Float64Array(A.th);
    return { cfg: cfg, ar: ar, ckpt: A, ctx: { ref: new Float64Array(A.th), mem: mem, basis: ogd.basis, basisR: ogdR.basis, ogd: ogd },
             models: models, step: 0, rngB: makeRng(cfg.seed * 104729 + 3), history: [] };
  }

  function advance(run, nSteps) {
    var cfg = run.cfg, ar = run.ar;
    for (var t = 0; t < nSteps && run.step < cfg.steps; t++) {
      var batchB = sampleBatch(run.rngB, cfg.batch, labelB), memIdx = [];
      for (var j = 0; j < cfg.memBatch; j++) memIdx.push(Math.floor(run.rngB() * run.ctx.mem.length));
      for (var i = 0; i < cfg.methods.length; i++) {
        var mth = cfg.methods[i];
        stepMethod(ar, mth, run.models[mth], run.ctx, batchB, memIdx, cfg.per && cfg.per[mth] ? Object.assign({}, cfg, cfg.per[mth]) : cfg,
                   cfg.per && cfg.per[mth] && cfg.per[mth].lr ? cfg.per[mth].lr : cfg.lr);
      }
      run.step++;
      // Refreshed OGD: re-linearise around its *current* parameters on the same memory inputs
      var pr = (cfg.per && cfg.per.togdr) || {}, every = pr.refresh || cfg.refresh;
      if (run.models.togdr && run.step % every === 0 && run.step < cfg.steps)
        run.ctx.basisR = buildOGDBasis(run.models.togdr, run.ctx.mem, pr.K || cfg.K, ar).basis;
      if (run.step % cfg.evalEvery === 0 || run.step === cfg.steps) record(run);
    }
    return run.step >= cfg.steps;
  }

  function record(run) {
    var row = { step: run.step };
    for (var i = 0; i < run.cfg.methods.length; i++) { var m = run.cfg.methods[i]; row[m] = evaluate(run.models[m], run.ar); }
    run.history.push(row);
    return row;
  }

  function runExperiment(cfg, ckpt) {
    var run = setup(cfg, ckpt); record(run);
    advance(run, run.cfg.steps);
    return run;
  }

  var Q = ARCHS.qs3;
  return { makeRng: makeRng, labelA: labelA, labelB: labelB, sigmoid: sigmoid, evaluate: evaluate,
           forward: function (th, x1, x2, an) { return arch(an).forward(th, x1, x2); },
           trainTaskA: trainTaskA, makeTrainer: makeTrainer, setup: setup, advance: advance, record: record, runExperiment: runExperiment,
           buildOGDBasis: buildOGDBasis, makeMemory: makeMemory, arch: arch, ARCHS: ARCHS, ARCH_INFO: ARCH_INFO,
           METHODS: METHODS, DEFAULTS: DEFAULTS, MAJORITY_FLOOR: MAJORITY_FLOOR,
           LAYOUT: Q.LAYOUT, P: Q.P, NQ: Q.NQ };
})();
if (typeof module !== 'undefined') module.exports = QCL;
