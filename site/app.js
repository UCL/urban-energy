/* NEPI Atlas — fully static client.
 *
 * Data contract (written by stats/atlas_export.py):
 *   tiles/{lad,lsoa,oa}.pmtiles   — three zoom levels, short property names
 *   data/aggregates.json          — England panel + LAD rank table
 *   data/meta.json                — frozen band thresholds, constants, headlines
 *   data/pc/{OUTCODE}.json        — postcode → [OA21CD, lon, lat]
 *
 * All slider maths is client-side arithmetic on baked attributes; there is no
 * model and no backend. Unit-level (LAD/LSOA) slider responses scale the
 * unit's median by the ratio of its linear sums — exact when the EV intensity
 * ratio is constant within the unit, a close approximation otherwise.
 */

"use strict";

const EPC = {
  A: "#008054", B: "#19b459", C: "#8dce46", D: "#ffd500",
  E: "#fcaa65", F: "#ef8023", G: "#e9153b",
};
const LETTERS = "ABCDEFG";
/* Dark text on the pale mid-bands, white elsewhere (contrast). */
const chipText = (l) => ("CDE".includes(l) ? "#1d2733" : "#ffffff");

/* The quantity views use sequential ramps (light = little, dark = much),
 * never the A–G letters. ColorBrewer 7-class Blues and YlOrBr: perceptually
 * stepped, colourblind-safe choropleth ramps. */
const ACCESS_RAMP = [
  "#eff3ff", "#c6dbef", "#9ecae1", "#6baed6",
  "#4292c6", "#2171b5", "#084594",
];
const ENERGY_RAMP = [
  "#ffffd4", "#fee391", "#fec44f", "#fe9929",
  "#ec7014", "#cc4c02", "#8c2d04",
];
const ENGLAND = [[-6.4, 49.8], [1.8, 55.9]];

const state = { metric: "rate", uf: 0, uh: 0, uv: 0, selected: null };
let meta, agg, map, hp;

/* ---------- helpers ---------- */

const $ = (sel) => document.querySelector(sel);

function letterFor(value, thresholds, best) {
  if (!isFinite(value)) return null;
  let idx = 0;
  for (const t of thresholds) if (value >= t) idx += 1;
  return best === "high" ? LETTERS[thresholds.length - idx] : LETTERS[idx];
}

/* Step expression: value expression → band colour from an ascending ramp. */
function colourExpr(valueExpr, thresholds, ramp) {
  const expr = ["step", valueExpr, ramp[0]];
  thresholds.forEach((t, i) => expr.push(t, ramp[i + 1]));
  return expr;
}

/* Ascending ramp for the active view. Only the rate carries the A–G
 * certificate colours; energy and access are descriptive quantities. */
function rampFor(metric) {
  if (metric === "rate") return [...LETTERS].map((l) => EPC[l]).reverse();
  if (metric === "energy") return ENERGY_RAMP;
  return ACCESS_RAMP;
}

/* Per-OA value for the active view, as a MapLibre expression. The grade is
 * access per kWh of TOTAL energy, so all three levers re-grade the rate
 * view; energy and access show the status quo. */
function oaValueExpr() {
  if (state.metric === "rate") {
    const { uf, uh, uv } = state;
    const fb = ["+", 1 - uf, ["*", uf, ["get", "f"]]];
    const hb = 1 - uh + uh * hp;
    const tb = ["+", 1 - uv, ["*", uv, ["get", "v"]]];
    const total = [
      "+", ["*", ["get", "g"], fb, hb], ["get", "e"], ["*", ["get", "t"], tb],
    ];
    return ["/", ["get", "ac"], total];
  }
  if (state.metric === "energy")
    return ["+", ["get", "g"], ["get", "e"], ["get", "t"]];
  return ["get", "aw"];
}

/* Per-unit (LAD/LSOA) counterpart via the linear sums. */
function unitValueExpr() {
  if (state.metric === "rate") {
    const { uf, uh, uv } = state;
    const hb = 1 - uh + uh * hp;
    const gasP = [
      "*", ["+", ["*", 1 - uf, ["get", "sg"]], ["*", uf, ["get", "sgf"]]], hb,
    ];
    const travP = ["+", ["*", 1 - uv, ["get", "st"]], ["*", uv, ["get", "sv"]]];
    const now = ["+", ["get", "sg"], ["get", "se"], ["get", "st"]];
    const adj = ["+", gasP, ["get", "se"], travP];
    return ["*", ["get", "mr"], ["/", now, adj]];
  }
  if (state.metric === "energy") return ["get", "me"];
  return ["get", "ma"];
}

function thresholdsFor(metric) {
  return meta.bands.thresholds[metric];
}

/* Describe the active view; the levers belong to the graded rate view. */
function updateLeverAvailability() {
  $("#levers").hidden = state.metric !== "rate";
  $("#viewdesc").textContent = {
    rate:
      "The NEPI grade: amenities reachable per kilowatt-hour of total " +
      "energy, home plus car travel. All three levers re-grade the map.",
    energy:
      "Status quo energy per dwelling, home plus car travel, " +
      "shown as plain values rather than grades.",
    access:
      "Amenities reachable on foot, shown as plain counts; " +
      "reach by car is part of the rate view.",
  }[state.metric];
  renderLegend();
}

/* Legend: A–G letters for the rate; value swatches for the quantities. */
function renderLegend() {
  if (state.metric === "rate") {
    $("#legend").innerHTML = [...LETTERS]
      .map((l) => `<span style="background:${EPC[l]};color:${chipText(l)}">${l}</span>`)
      .join("");
    $("#legendnote").textContent =
      "A best — G worst · bands fixed at the 2021 national distribution";
    return;
  }
  const ramp = rampFor(state.metric);
  const cuts = [0, ...thresholdsFor(state.metric)];
  const label =
    state.metric === "energy"
      ? (c) => `${Math.round(c / 1000)}k`
      : (c) => `${Math.round(c)}`;
  $("#legend").innerHTML = cuts
    .map((c, i) => {
      const fg = i < 4 ? "#1d2733" : "#ffffff";
      return `<span style="background:${ramp[i]};color:${fg}">${label(c)}</span>`;
    })
    .join("");
  $("#legendnote").textContent =
    state.metric === "energy"
      ? "kWh per dwelling per year, from the value shown upward · scale fixed at the 2021 distribution"
      : "amenities within a 1.6 km walk, from the count shown upward · scale fixed at the 2021 distribution";
}

/* Light the preset matching the current lever settings; none on custom positions. */
function syncPresetButtons() {
  const cur = ["uf", "uh", "uv"].map((k) => Math.round(state[k] * 100)).join(",");
  document
    .querySelectorAll(".presets button")
    .forEach((b) => b.classList.toggle("on", b.dataset.preset === cur));
}

function repaint() {
  /* Panel and card first: their updates must never be blocked by map state. */
  syncPresetButtons();
  renderNational();
  renderCard();
  if (!map || !map.getLayer("oa-fill")) return;
  const t = thresholdsFor(state.metric);
  const ramp = rampFor(state.metric);
  map.setPaintProperty("oa-fill", "fill-color", colourExpr(oaValueExpr(), t, ramp));
  for (const level of ["lad", "lsoa"])
    map.setPaintProperty(
      `${level}-fill`, "fill-color", colourExpr(unitValueExpr(), t, ramp)
    );
}

/* ---------- national panel ---------- */

function englandEnergyTWh(uf, uh, uv) {
  const e = agg.england;
  const hb = 1 - uh + uh * hp;
  const gas = ((1 - uf) * e.sg + uf * e.sgf) * hb;
  const travel = (1 - uv) * e.st + uv * e.sv;
  return (gas + e.se + travel) / 1000;
}

/* Energy per household (kWh/yr) for a summed group at the current lever
 * settings. The shipped sums are GWh/yr, hence the 1e6. */
function groupKWhPerHH(g) {
  const hb = 1 - state.uh + state.uh * hp;
  const gas = ((1 - state.uf) * g.sg + state.uf * g.sgf) * hb;
  const travel = (1 - state.uv) * g.st + state.uv * g.sv;
  return ((gas + g.se + travel) * 1e6) / g.hh;
}

function shareBar(el, counts, hh) {
  el.innerHTML = "";
  for (const l of LETTERS) {
    const seg = document.createElement("div");
    seg.style.background = EPC[l];
    seg.style.width = `${(counts[l] / hh) * 100}%`;
    seg.title = `${l}: ${((counts[l] / hh) * 100).toFixed(1)}% of households`;
    el.appendChild(seg);
  }
}

function renderNational() {
  const e = agg.england;
  const now = englandEnergyTWh(0, 0, 0);
  const adj = englandEnergyTWh(state.uf, state.uh, state.uv);
  const mrAdj = e.mr * (now / adj);
  $("#n-median").innerHTML = chip(letterFor(mrAdj, thresholdsFor("rate"), "high"));
  $("#n-median-sub").textContent = `${mrAdj.toFixed(2)} amenities per kWh`;
  $("#n-energy").textContent = `${adj.toFixed(0)} TWh`;
  $("#n-energy-sub").textContent = `status quo ${now.toFixed(0)} TWh`;
  $("#n-saving").textContent = `${fmt(((now - adj) * 1e9) / e.hh)} kWh`;
  $("#n-saving-sub").textContent =
    `${(now - adj).toFixed(0)} TWh · ${(((now - adj) / now) * 100).toFixed(0)}% of status quo`;
  const t = agg.england.types;
  if (t) {
    const perD = groupKWhPerHH(t.Detached);
    const perF = groupKWhPerHH(t.Flat);
    $("#n-lockin").textContent = `+${fmt(perD - perF)} kWh`;
    $("#n-lockin-sub").textContent =
      `${(perD / perF).toFixed(2)}× vs flats`;
  } else {
    $("#n-lockin").textContent = "–";
    $("#n-lockin-sub").textContent = "reload to fetch updated data";
  }
  shareBar($("#bar-current"), Object.fromEntries([...LETTERS].map((l) => [l, e["c" + l]])), e.hh);
  shareBar($("#bar-potential"), Object.fromEntries([...LETTERS].map((l) => [l, e["p" + l]])), e.hh);
}

function renderHeadline() {
  const h = meta.headlines;
  $("#headline").textContent =
    `Across ${h.sampleN} neighbourhoods, flats reach about ` +
    `${Math.round(Number(h.walkAmen))}× the amenities of detached areas on foot and gain ` +
    `about ${h.rate}× the access per kilowatt-hour of driving. The levers below ` +
    `apply insulation, heat pumps and electric vehicles at any uptake.`;
}

/* ---------- card ---------- */

function chip(letter) {
  const bg = letter ? EPC[letter] : "#9aa3ad";
  const fg = letter ? chipText(letter) : "#ffffff";
  return `<span class="chip" style="background:${bg};color:${fg}">${letter ?? "–"}</span>`;
}

function fmt(n, dp = 0) {
  return Number(n).toLocaleString("en-GB", {
    maximumFractionDigits: dp, minimumFractionDigits: 0,
  });
}

/* One table row: quantity name, then a graded chip per scenario column. */
function gradeRow(name, letters, note) {
  const cells = letters.map((l) => `<td>${chip(l)}</td>`).join("");
  return `<tr><td>${name}${note ? `<span class="locked">${note}</span>` : ""}</td>${cells}</tr>`;
}

const GRADE_HEAD = `<tr><th></th><th>Status quo</th><th>Adjusted</th><th>Full rollout</th></tr>`;

function renderCard() {
  const sel = state.selected;
  if (!sel) return;
  if (sel.kind === "oa") renderOACard(sel.props);
  else renderUnitCard(sel.props, sel.levelName);
}

const DT_LABEL = {
  Flat: "flats", Terraced: "terraced houses",
  Semi: "semi-detached houses", Detached: "detached houses",
};

function renderOACard(p) {
  const tb = 1 - state.uv + state.uv * p.v;
  const fb = 1 - state.uf + state.uf * p.f;
  const hb = 1 - state.uh + state.uh * hp;
  const homeAdj = p.g * fb * hb + p.e;
  const travelAdj = p.t * tb;
  const rateAdj = p.ac / (homeAdj + travelAdj);
  const lRate = letterFor(rateAdj, thresholdsFor("rate"), "high");
  const flags = [];
  if (p.fx & 1) flags.push("no EPC coverage (fabric potential assumed nil)");
  if (p.fx & 2) flags.push("metered energy may be under-recorded here");
  $("#card").hidden = false;
  $("#card").innerHTML = `
    <h2>Output Area ${p.id}</h2>
    <p class="sub">Mostly ${DT_LABEL[p.dt] ?? p.dt} · ${fmt(p.hh)} households · household size ${fmt(p.hs, 2)} · floor area ${fmt(p.fa)} m²</p>
    <table class="cardtable">
      ${GRADE_HEAD}
      ${gradeRow("NEPI grade", [p.lr, lRate, p.lrp])}
    </table>
    <p class="sub">A better rate than ${fmt(p.p)}% of England's households.</p>
    <h3>Energy and the rate</h3>
    <table class="cardtable numbers">
      <tr><th></th><th>Status quo</th><th>Adjusted</th></tr>
      <tr><td>Rate, amenities per kWh</td><td>${fmt(p.r, 3)}</td><td>${fmt(rateAdj, 3)}</td></tr>
      <tr><td>Home energy, kWh/yr</td><td>${fmt(p.g + p.e)}</td><td>${fmt(homeAdj)}</td></tr>
      <tr><td>Car-travel energy, kWh/yr</td><td>${fmt(p.t)}</td><td>${fmt(travelAdj)}</td></tr>
      <tr class="total"><td>Total energy, kWh/yr</td><td>${fmt(p.g + p.e + p.t)}</td><td>${fmt(homeAdj + travelAdj)}</td></tr>
    </table>
    <h3>Access, unchanged by the levers</h3>
    <table class="cardtable numbers">
      <tr><td>Amenities on foot, 1.6 km</td><td>${fmt(p.aw)}</td></tr>
      <tr><td>England median on foot</td><td>${fmt(agg.england.ma)}</td></tr>
      <tr><td>Within its own car catchment</td><td>${fmt(p.ac)}</td></tr>
    </table>
    ${flags.length ? `<p class="flagnote">⚠ ${flags.join("; ")}</p>` : ""}`;
}

function renderUnitCard(p, levelName) {
  const travP = (1 - state.uv) * p.st + state.uv * p.sv;
  const gasP = ((1 - state.uf) * p.sg + state.uf * p.sgf) * (1 - state.uh + state.uh * hp);
  const totalNow = p.sg + p.se + p.st;
  const totalAdj = gasP + p.se + travP;
  const mrAdj = p.mr * (totalNow / totalAdj);
  const meAdj = p.me * (totalAdj / totalNow);
  const counts = Object.fromEntries([...LETTERS].map((l) => [l, p["c" + l]]));
  $("#card").hidden = false;
  $("#card").innerHTML = `
    <h2>${p.nm}</h2>
    <p class="sub">${levelName} · ${fmt(p.hh)} households in ${fmt(p.n)} Output Areas · better median rate than ${fmt(p.p)}% of peers</p>
    <table class="cardtable">
      <tr><th></th><th>Status quo</th><th>Adjusted</th></tr>
      ${gradeRow("NEPI grade (median)", [
        letterFor(p.mr, thresholdsFor("rate"), "high"),
        letterFor(mrAdj, thresholdsFor("rate"), "high"),
      ])}
    </table>
    <h3>Medians here</h3>
    <table class="cardtable numbers">
      <tr><th></th><th>Status quo</th><th>Adjusted</th></tr>
      <tr><td>Rate, amenities per kWh</td><td>${fmt(p.mr, 3)}</td><td>${fmt(mrAdj, 3)}</td></tr>
      <tr><td>Energy, kWh/dwelling/yr</td><td>${fmt(p.me)}</td><td>${fmt(meAdj)}</td></tr>
    </table>
    <h3>Access, unchanged by the levers</h3>
    <table class="cardtable numbers">
      <tr><td>Amenities on foot, median</td><td>${fmt(p.ma)}</td></tr>
      <tr><td>England median on foot</td><td>${fmt(agg.england.ma)}</td></tr>
    </table>
    <h3>Rate grades of households here, status quo</h3>
    <div class="bar" id="unit-bar"></div>
    <p class="sub">Zoom in for street-level Output Areas.</p>`;
  shareBar($("#unit-bar"), counts, p.hh);
}

/* ---------- search ---------- */

async function searchPostcode(raw) {
  const key = raw.replace(/\s+/g, "").toUpperCase();
  if (key.length < 5) return;
  const outcode = key.slice(0, -3);
  try {
    const shard = await (await fetch(`data/pc/${outcode}.json`)).json();
    const hit = shard[key];
    if (!hit) return;
    map.flyTo({ center: [hit[1], hit[2]], zoom: 12.5 });
  } catch {
    /* unknown outcode: ignore */
  }
}

/* ---------- boot ---------- */

async function init() {
  [meta, agg] = await Promise.all([
    (await fetch("data/meta.json")).json(),
    (await fetch("data/aggregates.json")).json(),
  ]);
  hp = meta.constants.boiler_eff / meta.constants.cop;

  const protocol = new pmtiles.Protocol();
  maplibregl.addProtocol("pmtiles", protocol.tile);

  map = new maplibregl.Map({
    container: "map",
    style: {
      version: 8,
      /* Basemap: Carto Positron raster (OSM-derived) for orientation — the
       * one external runtime dependency; swap for self-hosted OS Zoomstack
       * at full launch (dissemination/launch_checklist.md). */
      sources: {
        basemap: {
          type: "raster",
          tiles: ["a", "b", "c", "d"].map(
            (s) => `https://${s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png`
          ),
          tileSize: 256,
          attribution:
            '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors © <a href="https://carto.com/attributions">CARTO</a>',
        },
        labels: {
          type: "raster",
          tiles: ["a", "b", "c", "d"].map(
            (s) => `https://${s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}.png`
          ),
          tileSize: 256,
        },
      },
      layers: [
        { id: "bg", type: "background", paint: { "background-color": "#eef1f4" } },
        { id: "basemap", type: "raster", source: "basemap" },
      ],
    },
    bounds: ENGLAND,
    fitBoundsOptions: { padding: 20 },
    attributionControl: { compact: true },
  });
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }));

  map.on("load", () => {
    const levels = [
      ["lad", 0, 8.4], ["lsoa", 8.4, 10.6], ["oa", 10.6, 24],
    ];
    for (const [name, minz, maxz] of levels) {
      map.addSource(name, { type: "vector", url: `pmtiles://tiles/${name}.pmtiles` });
      map.addLayer({
        id: `${name}-fill`, type: "fill", source: name, "source-layer": name,
        minzoom: minz, maxzoom: maxz,
        paint: { "fill-color": "#ccc", "fill-opacity": 0.75 },
      });
      map.addLayer({
        id: `${name}-line`, type: "line", source: name, "source-layer": name,
        minzoom: minz, maxzoom: maxz,
        paint: { "line-color": "#ffffff", "line-width": 0.4 },
      });
      /* Selection outline: a filtered line layer, empty until a click. */
      map.addLayer({
        id: `${name}-selected`, type: "line", source: name, "source-layer": name,
        minzoom: minz, maxzoom: maxz,
        filter: ["==", ["get", "id"], ""],
        paint: {
          "line-color": "#1d2733",
          "line-width": 2.5,
          "line-opacity": 0.95,
        },
      });
      map.on("click", `${name}-fill`, (ev) => {
        const p = ev.features[0].properties;
        state.selected =
          name === "oa"
            ? { kind: "oa", props: p }
            : { kind: "unit", props: p, levelName: name === "lad" ? "Local authority" : "LSOA" };
        for (const [lvl] of levels)
          map.setFilter(`${lvl}-selected`, [
            "==", ["get", "id"], lvl === name ? p.id : "",
          ]);
        renderCard();
      });
      map.on("mouseenter", `${name}-fill`, () => (map.getCanvas().style.cursor = "pointer"));
      map.on("mouseleave", `${name}-fill`, () => (map.getCanvas().style.cursor = ""));
    }
    /* Place names and road labels render above the fills for orientation. */
    map.addLayer({ id: "labels", type: "raster", source: "labels" });
    repaint();
  });

  /* sliders + presets */
  for (const k of ["uf", "uh", "uv"]) {
    $(`#${k}`).addEventListener("input", (ev) => {
      state[k] = Number(ev.target.value) / 100;
      $(`#o-${k}`).textContent = `${ev.target.value}%`;
      repaint();
    });
  }
  document.querySelectorAll(".presets button").forEach((b) =>
    b.addEventListener("click", () => {
      const [uf, uhv, uv] = b.dataset.preset.split(",").map(Number);
      for (const [k, v] of [["uf", uf], ["uh", uhv], ["uv", uv]]) {
        $(`#${k}`).value = v;
        $(`#o-${k}`).textContent = `${v}%`;
        state[k] = v / 100;
      }
      repaint();
    })
  );

  /* metric buttons */
  document.querySelectorAll(".metrics button").forEach((b) =>
    b.addEventListener("click", () => {
      document.querySelectorAll(".metrics button").forEach((x) => x.classList.remove("on"));
      b.classList.add("on");
      state.metric = b.dataset.metric;
      updateLeverAvailability();
      repaint();
    })
  );
  updateLeverAvailability();

  $("#search").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") searchPostcode(ev.target.value);
  });

  initPanelModes();
  renderHeadline();
  renderNational();
}

/* The panel opens on the About argument for first-time visitors and on the
 * data view thereafter. The About text is fetched from about.html so the
 * argument has a single source. */
function setPanelMode(mode) {
  const about = mode === "about";
  $("#aboutpane").hidden = !about;
  for (const sel of ["#intro", "#national", "#controls", "#card"])
    $(sel).style.display = about ? "none" : "";
  if (!about && !state.selected) $("#card").hidden = true;
  document
    .querySelectorAll("#panel-tabs button")
    .forEach((b) => b.classList.toggle("on", b.dataset.mode === mode));
  try {
    localStorage.setItem("nepiPanelMode", mode);
  } catch {
    /* private-mode browsers: no persistence */
  }
}

async function initPanelModes() {
  try {
    const html = await (await fetch("about.html")).text();
    const doc = new DOMParser().parseFromString(html, "text/html");
    $("#aboutpane").innerHTML = doc.querySelector(".prose").innerHTML;
  } catch {
    $("#aboutpane").innerHTML =
      '<p>See the <a href="about.html">About page</a>.</p>';
  }
  document
    .querySelectorAll("#panel-tabs button")
    .forEach((b) => b.addEventListener("click", () => setPanelMode(b.dataset.mode)));
  $("#nav-about").addEventListener("click", (ev) => {
    ev.preventDefault();
    setPanelMode("about");
  });
  let mode = "about";
  try {
    if (localStorage.getItem("nepiPanelMode")) mode = "explore";
  } catch {
    /* default to about */
  }
  setPanelMode(mode);
}

init();
