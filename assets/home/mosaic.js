// Live retinal mosaic for the hero.
// Two ganglion-cell mosaics (OFF and ON) with elliptical difference-of-Gaussians
// receptive fields watch a dark disc that drifts, loom-scales and follows the cursor.
// OFF cells respond to darkening (with a transient), ON cells to the light returning.
// A spike raster underneath shows Poisson spikes from a sample of cells.
(() => {
  const hero = document.querySelector(".hero");
  const cv = document.getElementById("mosaic");
  const rcv = document.getElementById("raster");
  if (!cv || !rcv) return;
  const ctx = cv.getContext("2d");
  const rctx = rcv.getContext("2d");
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  let W = 0, H = 0, dpr = 1, cells = [], rows = [], colors = {};
  const rand = mulberry32(7);

  function mulberry32(a) {
    return () => {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const gauss = () => {
    const u = 1 - rand(), v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };

  function readColors() {
    const cs = getComputedStyle(document.documentElement);
    const g = (n) => cs.getPropertyValue(n).trim();
    colors = { off: g("--uv"), on: g("--green"), stim: g("--stim"), ring: g("--stim-ring"), ink3: g("--ink-3"), paper: g("--paper") };
  }

  // hexagonal lattice with jitter; each cell gets a slightly irregular elliptical RF
  function lattice(spacing, type, offset) {
    const out = [];
    const dy = spacing * Math.sqrt(3) / 2;
    for (let r = -1, y = offset.y; y < H + spacing; r++, y += dy) {
      for (let x = offset.x + (r & 1 ? spacing / 2 : 0) - spacing; x < W + spacing; x += spacing) {
        const s = spacing * 0.42;
        out.push({
          type,
          x: x + gauss() * spacing * 0.08,
          y: y + gauss() * spacing * 0.08,
          sx: s * (1 + gauss() * 0.1),
          sy: s * (0.78 + gauss() * 0.08),
          th: -0.45 + gauss() * 0.35,
          drive: 0, slow: 0, r: 0,
        });
      }
    }
    return out;
  }

  function build() {
    const rect = cv.getBoundingClientRect();
    W = rect.width; H = rect.height;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = W * dpr; cv.height = H * dpr;
    const rr = rcv.getBoundingClientRect();
    rcv.width = rr.width * dpr; rcv.height = rr.height * dpr;

    const spacing = Math.max(34, Math.min(54, W / 28));
    cells = [
      ...lattice(spacing * 1.12, "on", { x: spacing * 0.31, y: spacing * 0.2 }),
      ...lattice(spacing, "off", { x: 0, y: 0 }),
    ];
    for (const c of cells) { c.cos = Math.cos(c.th); c.sin = Math.sin(c.th); }

    // every cell keeps its own recent spike train, pre-filled with baseline firing
    for (const c of cells) {
      c.spikes = [];
      for (let ts = t - HISTORY; ts < t; ts += 1 / 60) if (rand() < BASE_RATE / 60) c.spikes.push(ts);
    }
    rows = [];
    cv.dataset.cells = cells.length;
  }

  // ---- raster: the N cells nearest the object, like an electrode array that follows it ----
  const N_ROWS = 24, HISTORY = 12, BASE_RATE = 1.5;
  let lastSelect = -1;
  function selectRows(t) {
    if (t - lastSelect < 0.2 && rows.length) return;
    lastSelect = t;
    const ranked = cells
      .map((c) => [(c.x - stim.x) ** 2 + (c.y - stim.y) ** 2, c])
      .sort((a, b) => a[0] - b[0])
      .map((e) => e[1]);
    if (!rows.length) {
      rows = ranked.slice(0, N_ROWS).map((cell) => ({ cell }));
      rows.sort((a, b) => a.cell.y - b.cell.y);
      document.getElementById("raster-n").textContent = rows.length;
      return;
    }
    // hysteresis: a row keeps its cell while it stays among the nearest 1.5N, so rows
    // only change when the object moves away, and they change one slot at a time
    const keep = new Set(ranked.slice(0, Math.round(N_ROWS * 1.5)));
    const shown = new Set(rows.map((r) => r.cell));
    const fresh = ranked.slice(0, N_ROWS).filter((c) => !shown.has(c));
    for (const row of rows) if (!keep.has(row.cell) && fresh.length) row.cell = fresh.shift();
  }

  // ---- stimulus ----
  const stim = { x: 0, y: 0, R: 30 };
  const pointer = { x: 0, y: 0, last: -1e9 };
  hero.addEventListener("pointermove", (e) => {
    const r = cv.getBoundingClientRect();
    pointer.x = e.clientX - r.left; pointer.y = e.clientY - r.top; pointer.last = performance.now();
  });
  hero.addEventListener("pointerleave", () => (pointer.last = -1e9));

  function autopilot(t) {
    // slow Lissajous wander over the right part of the hero (centre-bottom on narrow screens)
    const narrow = W < 760;
    const cx = narrow ? W * 0.5 : W * 0.7, cy = narrow ? H * 0.72 : H * 0.48;
    const ax = narrow ? W * 0.35 : W * 0.2, ay = narrow ? H * 0.12 : H * 0.28;
    return { x: cx + ax * Math.sin(t * 0.21) + 18 * Math.sin(t * 0.83), y: cy + ay * Math.sin(t * 0.33 + 1.1) + 12 * Math.cos(t * 0.71) };
  }

  // Gaussian approximation of a uniform disc (variance R^2/4 per axis) convolved with a
  // Gaussian RF: overlap = R^2/(2V) * exp(-d^2 / 2V), clipped at 1.
  function dog(c, px, py, R) {
    const dx = px - c.x, dy = py - c.y;
    const u = dx * c.cos + dy * c.sin, v = -dx * c.sin + dy * c.cos;
    const vd = (R * R) / 4;
    const cen = (sx, sy) => {
      const Vx = sx * sx + vd, Vy = sy * sy + vd;
      const m = (u * u) / Vx + (v * v) / Vy;
      return Math.min(1, (R * R) / (2 * Math.sqrt(Vx * Vy))) * Math.exp(-m / 2);
    };
    return cen(c.sx, c.sy) - 0.75 * cen(c.sx * 2.1, c.sy * 2.1);
  }

  function step(t, dt) {
    const now = performance.now();
    const target = now - pointer.last < 2500 ? pointer : autopilot(t);
    const k = 1 - Math.exp(-dt * 4.5);
    stim.x += (target.x - stim.x) * k;
    stim.y += (target.y - stim.y) * k;
    const base = Math.max(20, Math.min(40, W / 36));
    stim.R = base * (1 + 0.75 * (0.5 + 0.5 * Math.sin(t * 0.55)) ** 2); // looming / receding

    const kf = 1 - Math.exp(-dt * 6), ks = 1 - Math.exp(-dt * 1.4);
    for (const c of cells) {
      if (Math.abs(c.x - stim.x) > stim.R * 3 + c.sx * 6 && c.drive < 0.01 && c.slow < 0.01 && c.r < 0.01) {
        c.drive *= 1 - kf; c.slow *= 1 - ks; c.r *= 1 - kf; continue;
      }
      const d = dog(c, stim.x, stim.y, stim.R);
      c.drive += (d - c.drive) * kf;
      c.slow += (c.drive - c.slow) * ks;
      const resp = c.type === "off" ? c.drive + 1.4 * (c.drive - c.slow) : 2.6 * (c.slow - c.drive);
      c.r = Math.max(0, Math.min(1, resp * 2.1));
    }

    for (const c of cells) {
      const rate = BASE_RATE + 70 * c.r; // Hz
      if (rand() < rate * dt) c.spikes.push(t);
      while (c.spikes.length && t - c.spikes[0] > HISTORY) c.spikes.shift();
    }
    selectRows(t);
  }

  function draw(t) {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    // stimulus
    ctx.globalAlpha = 0.9;
    ctx.fillStyle = colors.stim;
    ctx.beginPath();
    ctx.arc(stim.x, stim.y, stim.R, 0, Math.PI * 2);
    ctx.fill();
    if (colors.ring && colors.ring !== "transparent") {
      ctx.globalAlpha = 1; ctx.strokeStyle = colors.ring; ctx.lineWidth = 1; ctx.stroke();
    }
    ctx.globalAlpha = 1;

    for (const c of cells) {
      const col = c.type === "off" ? colors.off : colors.on;
      const a = c.r;
      ctx.beginPath();
      ctx.ellipse(c.x, c.y, c.sx * 1.1, c.sy * 1.1, c.th, 0, Math.PI * 2);
      ctx.globalAlpha = (c.type === "off" ? 0.32 : 0.13) + 0.6 * a;
      ctx.strokeStyle = col;
      ctx.lineWidth = 1 + a * 0.8;
      ctx.stroke();
      if (a > 0.02) {
        ctx.globalAlpha = a * 0.5;
        ctx.fillStyle = col;
        ctx.fill();
      }
      if (a > 0.35) {
        ctx.beginPath();
        ctx.ellipse(c.x, c.y, c.sx * 2.3, c.sy * 2.3, c.th, 0, Math.PI * 2);
        ctx.globalAlpha = (a - 0.35) * 0.5;
        ctx.setLineDash([2, 4]);
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
    // recorded cells: a dot and a small ring, like electrode sites
    for (const row of rows) {
      const col = row.cell.type === "off" ? colors.off : colors.on;
      ctx.globalAlpha = 0.95;
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.arc(row.cell.x, row.cell.y, 2, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 0.5;
      ctx.strokeStyle = col;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(row.cell.x, row.cell.y, 5, 0, Math.PI * 2);
      ctx.stroke();
    }

    // raster
    const rw = rcv.width / dpr, rh = rcv.height / dpr;
    rctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    rctx.clearRect(0, 0, rw, rh);
    const speed = 120; // px per second
    const rowH = rh / Math.max(1, rows.length);
    rows.forEach((row, i) => {
      rctx.fillStyle = row.cell.type === "off" ? colors.off : colors.on;
      for (const s of row.cell.spikes) {
        const x = rw - (t - s) * speed;
        if (x < 0) continue;
        rctx.globalAlpha = Math.min(1, x / (rw * 0.35));
        rctx.fillRect(x, i * rowH + rowH * 0.15, 1.3, rowH * 0.7);
      }
    });
    rctx.globalAlpha = 1;
  }

  // ---- loop ----
  let running = false, raf = 0, t = 0, prev = 0;
  function frame(now) {
    const dt = Math.min(0.05, (now - prev) / 1000 || 0.016);
    prev = now; t += dt;
    step(t, dt);
    draw(t);
    if (running) raf = requestAnimationFrame(frame);
  }
  function start() { if (running || reduced) return; running = true; prev = performance.now(); raf = requestAnimationFrame(frame); }
  function stop() { running = false; cancelAnimationFrame(raf); }

  function staticFrame() {
    // one settled frame for reduced motion: simulate a few seconds without drawing
    const p = autopilot(3);
    stim.x = p.x; stim.y = p.y;
    for (let i = 0; i < 180; i++) { t += 1 / 60; step(t, 1 / 60); }
    draw(t);
  }

  function init() {
    readColors();
    build();
    const p = autopilot(0);
    stim.x = p.x; stim.y = p.y;
    if (reduced) staticFrame(); else draw(t);
  }

  init();
  let resizeTimer;
  addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { init(); }, 150); });
  addEventListener("themechange", () => { readColors(); if (reduced) draw(t); });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { readColors(); if (reduced) draw(t); });

  let visible = false;
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; visible && !document.hidden ? start() : stop(); }).observe(hero);
  document.addEventListener("visibilitychange", () => (document.hidden || !visible ? stop() : start()));
})();
