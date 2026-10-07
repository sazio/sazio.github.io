(() => {
  const S = window.SITE;
  const $ = (sel, root = document) => root.querySelector(sel);
  const fill = (key) => $(`[data-fill="${key}"]`);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const linkChips = (links = []) =>
    links.length ? `<div class="links">${links.map((l) => `<a href="${esc(l.url)}">${esc(l.label)} ↗</a>`).join("")}</div>` : "";

  // ---- hero ----
  const social = [
    ["Scholar", S.links.scholar], ["GitHub", S.links.github], ["X", S.links.x],
    ["Bluesky", S.links.bluesky], ["LinkedIn", S.links.linkedin], ["Email", `mailto:${S.links.email}`],
  ];
  fill("social").innerHTML = social.map(([n, u]) => `<li><a href="${esc(u)}">${n}</a></li>`).join("");
  const mail = fill("email");
  mail.href = `mailto:${S.links.email}`;
  mail.textContent = S.links.email;

  // ---- research glyphs (small SVG diagrams, one per pillar) ----
  const glyphs = {
    // second-order kernel W_ij = a_i b_j + b_i a_j, drawn as a grid of discs
    statistics() {
      const n = 7, a = [], b = [];
      for (let k = 0; k < n; k++) {
        const x = k - 3;
        a.push(Math.cos(x * 0.95) * Math.exp(-(x * x) / 7));
        b.push(Math.sin(x * 0.95) * Math.exp(-(x * x) / 7));
      }
      let out = "";
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) {
          const w = a[i] * b[j] + b[i] * a[j] + 0.6 * a[i] * a[j];
          const r = Math.min(6, Math.abs(w) * 7 + 0.4);
          const cx = 14 + j * 12, cy = 14 + i * 12;
          out += w >= 0
            ? `<circle cx="${cx}" cy="${cy}" r="${r.toFixed(2)}" fill="var(--ink)"/>`
            : `<circle cx="${cx}" cy="${cy}" r="${r.toFixed(2)}" fill="none" stroke="var(--ink)" stroke-width="1"/>`;
        }
      return out;
    },
    // a receptive field and its orbit under dilation
    symmetry() {
      let out = "";
      [1, 1.42, 2.0, 2.8].forEach((s, k) => {
        out += `<ellipse cx="50" cy="50" rx="${(11 * s).toFixed(1)}" ry="${(7.5 * s).toFixed(1)}" transform="rotate(-25 50 50)" fill="none" stroke="var(--uv)" stroke-width="${1.6 - k * 0.25}" opacity="${1 - k * 0.22}" ${k ? 'stroke-dasharray="3 3"' : ""}/>`;
      });
      return out + `<circle cx="50" cy="50" r="3" fill="var(--uv)"/>`;
    },
    // metric-tensor glyphs G(x) over a 2-D stimulus space
    information() {
      let out = "";
      for (let i = 0; i < 5; i++)
        for (let j = 0; j < 5; j++) {
          const x = 14 + j * 18, y = 14 + i * 18;
          const u = (x - 50) / 40, v = (y - 50) / 40;
          const th = (Math.atan2(v + 0.3, u - 0.2) * 180) / Math.PI + 90;
          const big = 4 + 4.5 * Math.exp(-((u + 0.25) ** 2 + (v - 0.15) ** 2) * 2.2);
          out += `<ellipse cx="${x}" cy="${y}" rx="${big.toFixed(2)}" ry="${(big * 0.42).toFixed(2)}" transform="rotate(${th.toFixed(1)} ${x} ${y})" fill="color-mix(in srgb, var(--green) 25%, transparent)" stroke="var(--green)" stroke-width="1"/>`;
        }
      return out;
    },
  };

  fill("pillars").innerHTML = S.pillars
    .map(
      (p, i) => `
    <article class="pillar reveal" data-key="${p.key}" style="transition-delay:${i * 90}ms">
      <div class="pillar-top">
        <span class="num">0${i + 1}</span>
        <svg class="pillar-glyph" viewBox="0 0 100 100" aria-hidden="true">${glyphs[p.key]()}</svg>
        <h3>${esc(p.name)}</h3>
        <p class="q">${esc(p.question)}</p>
      </div>
      <div class="pillar-body">
        <h4>${esc(p.title)}</h4>
        <p>${esc(p.body)}</p>
        <p class="venues">${esc(p.venues)}</p>
        ${linkChips(p.links)}
      </div>
    </article>`
    )
    .join("");

  // ---- publications ----
  const me = /S\. Azeglio\*?/g;
  fill("pubs").innerHTML = S.publications
    .map(
      (p) => `
    <li class="pub reveal" data-tag="${p.tag}" data-selected="${!!p.selected}" ${p.selected ? "" : "hidden"}>
      <span class="yr">${p.year}</span>
      <span class="dot" title="${esc(p.tag)}"></span>
      <div>
        <h3>${esc(p.title)}</h3>
        <p class="au">${esc(p.authors).replace(me, (m) => `<b>${m}</b>`)}</p>
        <p class="ve">${esc(p.venue)}${p.highlight ? `<span class="hl">★ ${esc(p.highlight)}</span>` : ""}</p>
      </div>
      ${linkChips(p.links)}
    </li>`
    )
    .join("");

  document.querySelectorAll(".pub-controls button").forEach((btn) =>
    btn.addEventListener("click", () => {
      const all = btn.dataset.filter === "all";
      document.querySelectorAll(".pub-controls button").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
      document.querySelectorAll(".pub").forEach((li) => {
        li.hidden = !all && li.dataset.selected !== "true";
        li.classList.add("in");
      });
    })
  );

  // ---- workshops, news, writing ----
  fill("workshops").innerHTML = S.workshops
    .map(
      (w) => `
    <li class="reveal"><span class="when">${w.year}</span>
      <span class="what"><span class="where" data-v="${esc(w.venue)}">${esc(w.venue)}</span>${w.url ? `<a href="${esc(w.url)}">${esc(w.name)}</a>` : esc(w.name)}</span></li>`
    )
    .join("");

  const fmt = (d) => new Date(d + "T12:00:00").toLocaleDateString("en-GB", { month: "short", year: "numeric" });
  const NEWS_SHOWN = 7;
  fill("news").innerHTML = S.news
    .map((n, i) => `<li ${i >= NEWS_SHOWN ? "hidden" : ""}><span class="when">${fmt(n.date)}</span>${n.url ? `<a href="${esc(n.url)}">${esc(n.text)}</a>` : `<span>${esc(n.text)}</span>`}</li>`)
    .join("");
  const more = $('[data-more="news"]');
  more.addEventListener("click", () => {
    const open = more.dataset.open === "1";
    document.querySelectorAll(".news li").forEach((li, i) => (li.hidden = open && i >= NEWS_SHOWN));
    more.dataset.open = open ? "0" : "1";
    more.textContent = open ? "Show older" : "Show fewer";
  });

  fill("writing").innerHTML = S.writing
    .map((w) => `<li><span class="when">${w.year}</span><a href="${esc(w.url)}">${esc(w.title)}</a></li>`)
    .join("");

  // ---- reveal on scroll ----
  const io = new IntersectionObserver(
    (entries) => entries.forEach((e) => e.isIntersecting && (e.target.classList.add("in"), io.unobserve(e.target))),
    { rootMargin: "0px 0px -8% 0px" }
  );
  document.querySelectorAll(".reveal").forEach((el) => io.observe(el));

})();
