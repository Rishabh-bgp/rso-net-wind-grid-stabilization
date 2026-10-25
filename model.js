/* Pure-JS forward pass for the exported PyTorch models, incl. a statevector simulator
   for the variational quantum circuit. Works in the browser and in Node (for tests). */
(function (root) {
  const sig = x => 1 / (1 + Math.exp(-x));
  const softplus = x => x > 20 ? x : Math.log1p(Math.exp(x));
  function linear(x, W, b) { const o = new Array(W.length); for (let i = 0; i < W.length; i++) { let s = b ? b[i] : 0; const r = W[i]; for (let j = 0; j < r.length; j++) s += r[j] * x[j]; o[i] = s; } return o; }
  const relu = v => v.map(x => Array.isArray(x) ? relu(x) : (x > 0 ? x : 0));
  // seq: T x C ; W: [out][in][k]; causal pad (left) = pad, symmetric pad when sym
  function conv1d(seq, W, b, k, dil, padL, padR) {
    const T = seq.length, C = seq[0].length, out = [], Tout = T + padL + padR - dil * (k - 1);
    for (let t = 0; t < Tout; t++) { const o = new Array(W.length);
      for (let oc = 0; oc < W.length; oc++) { let s = b[oc];
        for (let j = 0; j < k; j++) { const ti = t + j * dil - padL; if (ti < 0 || ti >= T) continue;
          const x = seq[ti]; for (let c = 0; c < C; c++) s += W[oc][c][j] * x[c]; }
        o[oc] = s; } out.push(o); }
    return out;
  }
  function lstmDir(seq, p, sfx, reverse) {
    const Wi = p['weight_ih_l0' + sfx], Wh = p['weight_hh_l0' + sfx], bi = p['bias_ih_l0' + sfx], bh = p['bias_hh_l0' + sfx];
    const H = Wh[0].length; let h = new Array(H).fill(0), c = new Array(H).fill(0); const outs = new Array(seq.length);
    for (let s = 0; s < seq.length; s++) { const t = reverse ? seq.length - 1 - s : s;
      const a = linear(seq[t], Wi, bi), g2 = linear(h, Wh, bh);
      for (let i = 0; i < a.length; i++) a[i] += g2[i];
      for (let i = 0; i < H; i++) { const ig = sig(a[i]), fg = sig(a[H + i]), gg = Math.tanh(a[2 * H + i]), og = sig(a[3 * H + i]);
        c[i] = fg * c[i] + ig * gg; h[i] = og * Math.tanh(c[i]); }
      outs[t] = h.slice(); }
    return { outs, h };
  }
  function mha(seq, p, heads) {
    const E = seq[0].length, d = E / heads, W = p.in_proj_weight, b = p.in_proj_bias;
    const qkv = seq.map(x => linear(x, W, b)); const T = seq.length; const out = [];
    for (let t = 0; t < T; t++) { const o = new Array(E).fill(0);
      for (let hd = 0; hd < heads; hd++) { const off = hd * d; const sc = new Array(T); let mx = -Infinity;
        for (let u = 0; u < T; u++) { let s = 0; for (let i = 0; i < d; i++) s += qkv[t][off + i] * qkv[u][E + off + i]; sc[u] = s / Math.sqrt(d); if (sc[u] > mx) mx = sc[u]; }
        let z = 0; for (let u = 0; u < T; u++) { sc[u] = Math.exp(sc[u] - mx); z += sc[u]; }
        for (let u = 0; u < T; u++) { const w = sc[u] / z; for (let i = 0; i < d; i++) o[off + i] += w * qkv[u][2 * E + off + i]; } }
      out.push(linear(o, p['out_proj.weight'], p['out_proj.bias'])); }
    return out;
  }
  // ---- statevector simulator (wire 0 = most significant bit, PennyLane convention) ----
  function vqc(angles, weights) {
    const n = angles.length, N = 1 << n; let re = new Float64Array(N), im = new Float64Array(N); re[0] = 1;
    const bit = w => 1 << (n - 1 - w);
    function ry(w, th) { const c = Math.cos(th / 2), s = Math.sin(th / 2), m = bit(w);
      for (let i = 0; i < N; i++) if (!(i & m)) { const j = i | m; const a0 = re[i], b0 = im[i], a1 = re[j], b1 = im[j];
        re[i] = c * a0 - s * a1; im[i] = c * b0 - s * b1; re[j] = s * a0 + c * a1; im[j] = s * b0 + c * b1; } }
    function rz(w, th) { const c = Math.cos(th / 2), s = Math.sin(th / 2), m = bit(w);
      for (let i = 0; i < N; i++) { const sg = (i & m) ? 1 : -1; const a = re[i], b = im[i]; re[i] = a * c - b * s * sg; im[i] = b * c + a * s * sg; } }
    function cnot(cw, tw) { const mc = bit(cw), mt = bit(tw);
      for (let i = 0; i < N; i++) if ((i & mc) && !(i & mt)) { const j = i | mt; let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
    for (let w = 0; w < n; w++) ry(w, angles[w]);
    for (const L of weights) { for (let w = 0; w < n; w++) { ry(w, L[w][0]); rz(w, L[w][1]); } for (let w = 0; w < n; w++) cnot(w, (w + 1) % n); }
    const ez = new Array(n).fill(0);
    for (let i = 0; i < N; i++) { const p = re[i] * re[i] + im[i] * im[i]; for (let w = 0; w < n; w++) ez[w] += (i & bit(w)) ? -p : p; }
    return ez;
  }
  function qblock(x, P, pre, wkey) { const a = linear(x, P[pre + '.weight'], P[pre + '.bias']).map(v => Math.tanh(v) * Math.PI / 2); return vqc(a, P[wkey]); }
  function sub(P, prefix) { const o = {}; for (const k in P) if (k.startsWith(prefix)) o[k.slice(prefix.length)] = P[k]; return o; }
  const T = seq => seq; 
  const MODELS = {
    hvqc(P, cfg, x) {
      let h = relu(conv1d(x, P['conv.0.weight'], P['conv.0.bias'], 3, 1, 1, 1));
      h = lstmDir(h, sub(P, 'lstm.'), '', false).h;
      const z = qblock(h, P, 'q.pre', 'q.q.weights');
      return linear(relu(linear(z, P['head.0.weight'], P['head.0.bias'])), P['head.2.weight'], P['head.2.bias'])[0];
    },
    cbma(P, cfg, x) {
      let h = relu(conv1d(x, P['cnn.0.weight'], P['cnn.0.bias'], 3, 1, 1, 1));
      const L = sub(P, 'bilstm.'), f = lstmDir(h, L, '', false).outs, b = lstmDir(h, L, '_reverse', true).outs;
      h = f.map((v, i) => v.concat(b[i]));
      const a = mha(h, sub(P, 'att.'), cfg.heads); const c = a[0].map((_, i) => a.reduce((s, r) => s + r[i], 0) / a.length);
      // MPS
      const cores = P['tn.cores'], D = P['tn.left'].length; let v = P['tn.left'].slice();
      for (let t = 0; t < x.length; t++) { const ang = sig(linear(x[t], P['tn.proj.weight'], P['tn.proj.bias'])[0]) * Math.PI / 2;
        const ph = [Math.cos(ang), Math.sin(ang)], nv = new Array(D).fill(0);
        for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) nv[j] += v[i] * (ph[0] * cores[t][0][i][j] + ph[1] * cores[t][1][i][j]);
        const nr = Math.sqrt(nv.reduce((s, q) => s + q * q, 0)) + 1e-8; v = nv.map(q => q / nr); }
      const tn = linear(v, P['tn.out.weight'], P['tn.out.bias']);
      const z = qblock(c, P, 'q.pre', 'q.q.weights');
      return linear(relu(linear(c.concat(tn, z), P['head.0.weight'], P['head.0.bias'])), P['head.2.weight'], P['head.2.bias'])[0];
    },
    hcq(P, cfg, x) {
      const H = cfg.hidden; let h = new Array(H).fill(0), c = new Array(H).fill(0);
      for (const xt of x) { const z = xt.concat(h); const g = linear(z, P['cell.cls.weight'], P['cell.cls.bias']);
        const a = linear(z, P['cell.pre.weight'], P['cell.pre.bias']).map(v => Math.tanh(v) * Math.PI / 2);
        const G = [0, 1, 2, 3].map(k => { const e = linear(vqc(a, P[`cell.vqc.${k}.weights`]), P[`cell.post.${k}.weight`], P[`cell.post.${k}.bias`]); return g.slice(k * H, (k + 1) * H).map((v, i) => v + e[i]); });
        for (let i = 0; i < H; i++) { c[i] = sig(G[1][i]) * c[i] + sig(G[0][i]) * Math.tanh(G[3][i]); h[i] = sig(G[2][i]) * Math.tanh(c[i]); } }
      return linear(h, P['head.weight'], P['head.bias'])[0];
    },
    rso(P, cfg, x, all) {
      let h = x; cfg.dilations.forEach((d, i) => { h = relu(conv1d(h, P[`tcn.${2 * i}.conv.weight`], P[`tcn.${2 * i}.conv.bias`], cfg.k, d, (cfg.k - 1) * d, 0)); });
      const a = mha(h, sub(P, 'att.'), cfg.heads); const q = linear(a[a.length - 1], P['head.weight'], P['head.bias']);
      const qs = [q[0], q[0] + softplus(q[1])]; qs.push(qs[1] + softplus(q[2]));
      return all ? qs : qs[1];
    },
  };
  // exported: standardise window, run model, de-standardise delta, add last speed
  function forecast(M, rawWindow, all) {
    const xs = rawWindow.map(r => r.map((v, i) => (v - M.mu[i]) / M.sd[i]));
    const out = MODELS[M.kind](M.params, M.cfg, xs, all); const last = rawWindow[rawWindow.length - 1][0];
    const f = d => last + d * M.ys + M.ym;
    return Array.isArray(out) ? out.map(f) : f(out);
  }
  function features(ws, dirDeg, rated) {
    const p = powerCurve(ws, rated); const r = dirDeg * Math.PI / 180;
    return [ws, Math.sin(r), Math.cos(r), p / 1000, 0.95 * p / 1000];
  }
  function powerCurve(v, rated = 3600, ci = 3, rv = 12.5, co = 25) { if (v > co) return 0; const f = Math.min(Math.max((v - ci) / (rv - ci), 0), 1); return rated * f ** 3; }
  const api = { forecast, features, powerCurve, vqc };
  if (typeof module !== 'undefined') module.exports = api; else root.WindModel = api;
})(this);
