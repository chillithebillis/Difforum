/**
 * Difforum · Director - multi-track timeline editor
 * ==================================================
 * Scenes (prompt + mood), Camera (visual move picker) and Energy (denoise
 * curve) on one timeline, with an animated camera preview computed by the
 * same Python engine that renders (POST /difforum/preview).
 *
 * The node's hidden `timeline` STRING widget stays the source of truth, so
 * workflows serialise exactly as ComfyUI expects.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_ID = "Difforum_Director";
const SETUP_ID = "Difforum_Setup";

// Fallback catalogue, replaced by GET /difforum/catalog when the server answers.
let CATALOG = {
    moves: [
        "still", "zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down",
        "roll_cw", "roll_ccw", "dolly_in", "dolly_out", "orbit_left", "orbit_right",
        "tilt_up", "tilt_down", "rise", "crane_up", "dolly_zoom", "spiral", "vortex",
        "sway", "breathe", "drift", "handheld", "shake",
    ].map((id) => ({ id, label: id.replace(/_/g, " "), group: "basic", hint: "", depth: false })),
    moods: [
        { id: "calm", color: "#4a90d9", strength: 0.44 },
        { id: "build", color: "#e0a020", strength: 0.52 },
        { id: "tense", color: "#e2703a", strength: 0.57 },
        { id: "climax", color: "#a45ec4", strength: 0.63 },
        { id: "resolve", color: "#37a86b", strength: 0.44 },
        { id: "dream", color: "#1f9c8c", strength: 0.48 },
    ],
    easings: ["linear", "ease_in", "ease_out", "ease_in_out", "step"],
    reactions: [{ id: "none", hint: "" }, { id: "beat_pulse", hint: "" }, { id: "onset_shake", hint: "" },
                { id: "bass_speed", hint: "" }, { id: "mid_sway", hint: "" }],
};
let catalogLoaded = null;
function loadCatalog() {
    if (!catalogLoaded) {
        catalogLoaded = api.fetchApi("/difforum/catalog")
            .then((r) => (r.ok ? r.json() : null))
            .then((c) => { if (c && c.moves) CATALOG = c; })
            .catch(() => {});
    }
    return catalogLoaded;
}
const moodOf = (id) => CATALOG.moods.find((m) => m.id === id) || CATALOG.moods[0];
const moveOf = (id) => CATALOG.moves.find((m) => m.id === id) || CATALOG.moves[0];

const LENSES = [[0, "auto (mood)"], [16, "16° ultra tele"], [24, "24° tele"], [32, "32° long"],
                [40, "40° normal"], [50, "50° wide"], [65, "65° very wide"], [85, "85° ultra wide"],
                [110, "110° fisheye"]];
const REACT_LABEL = { none: "no audio", beat_pulse: "pulse on beat", onset_shake: "shake on hits",
                      bass_speed: "bass drives speed", mid_sway: "mids rock roll" };

// ---------------------------------------------------------------------------
// move glyphs (drawn, so they scale and theme cleanly)
// ---------------------------------------------------------------------------
function arrow(ctx, x0, y0, x1, y1, h = 4) {
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    const a = Math.atan2(y1 - y0, x1 - x0);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - h * Math.cos(a - 0.5), y1 - h * Math.sin(a - 0.5));
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - h * Math.cos(a + 0.5), y1 - h * Math.sin(a + 0.5));
    ctx.stroke();
}
function arc(ctx, cx, cy, r, a0, a1, head = true) {
    ctx.beginPath(); ctx.arc(cx, cy, r, a0, a1, a1 < a0); ctx.stroke();
    if (!head) return;
    const ex = cx + r * Math.cos(a1), ey = cy + r * Math.sin(a1);
    const t = a1 + (a1 > a0 ? Math.PI / 2 : -Math.PI / 2);
    arrow(ctx, ex - 2 * Math.cos(t), ey - 2 * Math.sin(t), ex, ey, 4);
}
function drawGlyph(ctx, id, x, y, s, color) {
    ctx.save();
    ctx.translate(x, y);
    ctx.scale(s / 24, s / 24);
    ctx.strokeStyle = color; ctx.fillStyle = color;
    ctx.lineWidth = 1.8; ctx.lineCap = "round"; ctx.lineJoin = "round";
    const R = (w, h) => ctx.strokeRect(12 - w / 2, 12 - h / 2, w, h);
    switch (id) {
        case "still": R(14, 10); ctx.beginPath(); ctx.arc(12, 12, 1.5, 0, 7); ctx.fill(); break;
        case "zoom_in": R(18, 13); R(8, 6); arrow(ctx, 5, 6, 8, 9, 3); arrow(ctx, 19, 18, 16, 15, 3); break;
        case "zoom_out": R(8, 6); R(18, 13); arrow(ctx, 8, 9, 4, 5, 3); arrow(ctx, 16, 15, 20, 19, 3); break;
        case "pan_left": R(12, 10); arrow(ctx, 20, 12, 3, 12); break;
        case "pan_right": R(12, 10); arrow(ctx, 4, 12, 21, 12); break;
        case "pan_up": R(12, 10); arrow(ctx, 12, 21, 12, 3); break;
        case "pan_down": R(12, 10); arrow(ctx, 12, 3, 12, 21); break;
        case "roll_cw": arc(ctx, 12, 12, 8, -2.4, 1.6); break;
        case "roll_ccw": arc(ctx, 12, 12, 8, 1.6, -2.4); break;
        case "dolly_in":
            ctx.beginPath(); ctx.moveTo(3, 20); ctx.lineTo(10, 9); ctx.moveTo(21, 20); ctx.lineTo(14, 9); ctx.stroke();
            arrow(ctx, 12, 21, 12, 7); break;
        case "dolly_out":
            ctx.beginPath(); ctx.moveTo(3, 20); ctx.lineTo(10, 9); ctx.moveTo(21, 20); ctx.lineTo(14, 9); ctx.stroke();
            arrow(ctx, 12, 8, 12, 21); break;
        case "orbit_left":
            ctx.beginPath(); ctx.ellipse(12, 13, 9, 4, 0, 0, 7); ctx.stroke();
            ctx.beginPath(); ctx.arc(12, 9, 2.4, 0, 7); ctx.fill(); arrow(ctx, 8, 17, 4, 15, 3); break;
        case "orbit_right":
            ctx.beginPath(); ctx.ellipse(12, 13, 9, 4, 0, 0, 7); ctx.stroke();
            ctx.beginPath(); ctx.arc(12, 9, 2.4, 0, 7); ctx.fill(); arrow(ctx, 16, 17, 20, 15, 3); break;
        case "tilt_up": arc(ctx, 6, 12, 10, 0.7, -0.7); break;
        case "tilt_down": arc(ctx, 6, 12, 10, -0.7, 0.7); break;
        case "rise": case "crane_up":
            ctx.beginPath(); ctx.moveTo(4, 21); ctx.lineTo(20, 21); ctx.stroke();
            arrow(ctx, 12, 19, 12, 4);
            if (id === "crane_up") arc(ctx, 12, 8, 5, -0.2, 1.2); break;
        case "dolly_zoom":
            ctx.beginPath(); ctx.moveTo(3, 5); ctx.lineTo(12, 12); ctx.lineTo(3, 19); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(21, 8); ctx.lineTo(14, 12); ctx.lineTo(21, 16); ctx.stroke(); break;
        case "spiral": case "vortex": {
            ctx.beginPath();
            for (let t = 0; t < 12.5; t += 0.2) {
                const r = 1 + t * 0.75, px = 12 + r * Math.cos(t), py = 12 + r * Math.sin(t);
                t === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
            }
            ctx.stroke();
            if (id === "vortex") arrow(ctx, 17, 17, 21, 21, 3); else arrow(ctx, 16, 8, 13, 11, 3);
            break;
        }
        case "sway": arc(ctx, 12, 24, 14, -2.2, -0.9); arc(ctx, 12, 24, 14, -0.9, -2.2); break;
        case "breathe": R(16, 11); ctx.setLineDash([2, 2]); R(10, 7); ctx.setLineDash([]); break;
        case "drift":
            ctx.beginPath(); ctx.moveTo(3, 14);
            ctx.bezierCurveTo(8, 4, 12, 22, 21, 9); ctx.stroke(); break;
        case "handheld":
            ctx.beginPath(); ctx.moveTo(3, 12);
            for (let i = 1; i <= 9; i++) ctx.lineTo(3 + i * 2, 12 + (i % 2 ? -2 : 2) * (i % 3 ? 0.6 : 1));
            ctx.stroke(); break;
        case "shake":
            ctx.beginPath(); ctx.moveTo(3, 12);
            for (let i = 1; i <= 9; i++) ctx.lineTo(3 + i * 2, 12 + (i % 2 ? -5 : 5));
            ctx.stroke(); break;
        default: R(14, 10);
    }
    ctx.restore();
}

// ---------------------------------------------------------------------------
// styles
// ---------------------------------------------------------------------------
function injectStyles() {
    if (document.getElementById("dfx-css")) return;
    const el = document.createElement("style");
    el.id = "dfx-css";
    el.textContent = `
.dfx { --bg:#141414; --panel:#1c1c1c; --line:#2c2c2c; --txt:#d4d4d4; --dim:#7a7a7a; --acc:#4f8ef7;
  font: 11px/1.3 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif; color: var(--txt);
  background: var(--bg); border-radius: 8px; display: flex; flex-direction: column; height: 100%;
  box-sizing: border-box; overflow: hidden; pointer-events: none; user-select: none; }
.dfx > * { pointer-events: auto; }
.dfx-bar { display: flex; align-items: center; gap: 6px; padding: 6px 8px; background: var(--panel);
  border-bottom: 1px solid var(--line); }
.dfx-btn { background: #2a2a2a; border: 1px solid #3c3c3c; color: var(--txt); border-radius: 5px;
  padding: 3px 9px; cursor: pointer; font: inherit; white-space: nowrap; }
.dfx-btn:hover { background: #343434; }
.dfx-btn.on, .dfx-btn.pri { background: #22406f; border-color: var(--acc); }
.dfx-time { font-variant-numeric: tabular-nums; color: var(--dim); min-width: 96px; }
.dfx-sp { flex: 1; }
.dfx-main { display: flex; gap: 0; flex: 0 0 250px; }
.dfx-prev { width: 38%; min-width: 180px; max-width: 320px; padding: 8px; box-sizing: border-box;
  border-right: 1px solid var(--line); display: flex; flex-direction: column; gap: 6px; }
.dfx-prev canvas { width: 100%; background: #0b0b0b; border-radius: 5px; }
.dfx-cap { color: var(--dim); font-size: 10px; min-height: 26px; overflow: hidden; }
.dfx-tl { flex: 1; position: relative; }
.dfx-tl canvas { position: absolute; inset: 0; width: 100%; height: 100%; cursor: default; }
.dfx-ins { border-top: 1px solid var(--line); background: var(--panel); padding: 8px; display: flex; flex: 1 1 auto;
  flex-direction: column; gap: 7px; min-height: 84px; overflow-y: auto; }
.dfx-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.dfx-lbl { color: var(--dim); font-size: 10px; text-transform: uppercase; letter-spacing: .05em; }
.dfx-chip { padding: 3px 9px; border-radius: 12px; cursor: pointer; border: 1px solid transparent;
  font-size: 10px; color: #fff; opacity: .55; }
.dfx-chip.on { opacity: 1; border-color: #fff; }
.dfx-moves { display: grid; grid-template-columns: repeat(auto-fill, minmax(58px, 1fr)); gap: 3px; width: 100%; }
.dfx-mv { display: flex; flex-direction: column; align-items: center; gap: 1px; padding: 3px 2px;
  border-radius: 6px; background: #232323; border: 1px solid #2f2f2f; cursor: pointer; position: relative; }
.dfx-mv:hover { background: #2c2c2c; }
.dfx-mv.on { border-color: var(--acc); background: #1f2f4a; }
.dfx-mv span { font-size: 9px; color: #bdbdbd; text-align: center; }
.dfx-mv i { position: absolute; top: 2px; right: 4px; font-style: normal; font-size: 8px; color: #e0a020; }
.dfx-grp { font-size: 9px; color: var(--dim); grid-column: 1 / -1; margin-top: 2px; }
.dfx-sl { display: flex; align-items: center; gap: 6px; min-width: 150px; flex: 1; }
.dfx-sl input[type=range] { flex: 1; accent-color: var(--acc); }
.dfx-sl b { font-weight: 500; min-width: 30px; text-align: right; font-variant-numeric: tabular-nums; }
.dfx select, .dfx textarea { background: #242424; color: var(--txt); border: 1px solid #3a3a3a;
  border-radius: 5px; padding: 3px 6px; font: inherit; }
.dfx textarea { width: 100%; box-sizing: border-box; resize: vertical; min-height: 44px; user-select: text; }
.dfx-warn { color: #e0a020; font-size: 10px; padding: 0 8px 6px; }
.dfx-hint { color: var(--dim); font-size: 10px; }
.dfx-head b { font-weight: 600; font-size: 12px; }
`;
    document.head.appendChild(el);
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
};
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

function emptyTimeline() {
    return { version: 2, scenes: [], camera: [], energy: [], guidance: [] };
}
function normalise(v) {
    if (Array.isArray(v)) {               // 0.x list -> v2
        const t = emptyTimeline();
        for (const s of v) {
            const start = s.frame_start ?? s.start ?? 0;
            t.scenes.push({ start, mood: s.mood || "calm", prompt: s.prompt || "" });
            t.camera.push({ start, move: s.camera || "still", speed: 1, intensity: 1,
                            lens: s.lens || 0, ease: s.ease || "ease_in_out", react: "none" });
        }
        return t;
    }
    const t = Object.assign(emptyTimeline(), v || {});
    for (const k of ["scenes", "camera", "energy", "guidance"]) if (!Array.isArray(t[k])) t[k] = [];
    return t;
}

/** frames / fps / size: read the upstream Setup node when there is one. */
function readSetup(node) {
    const out = { frames: 120, fps: 24, width: 768, height: 432, known: false };
    try {
        const slot = node.inputs?.findIndex((i) => i.name === "params") ?? -1;
        const src = slot >= 0 && node.getInputNode ? node.getInputNode(slot) : null;
        if (src && src.type === SETUP_ID) {
            const w = (n) => src.widgets?.find((x) => x.name === n)?.value;
            let fps = Number(w("fps") ?? 24);
            const target = String(w("target") ?? "");
            if (target.startsWith("MiniMax")) fps = 24;
            const dur = Number(w("duration") ?? 5);
            out.fps = fps;
            out.frames = Math.max(1, Math.round(w("duration_mode") === "frames" ? dur : dur * fps));
            const le = Number(w("long_edge") ?? 768);
            const asp = String(w("aspect") ?? "16:9");
            const m = asp.match(/([\d.]+):([\d.]+)/);
            const [aw, ah] = m ? [Number(m[1]), Number(m[2])] : [16, 9];
            out.width = aw >= ah ? le : Math.round(le * aw / ah);
            out.height = aw >= ah ? Math.round(le * ah / aw) : le;
            out.known = true;
        }
    } catch (_) { /* graph not ready */ }
    return out;
}

// ---------------------------------------------------------------------------
// the editor
// ---------------------------------------------------------------------------
function buildEditor(node, tw) {
    let tl = emptyTimeline();
    let setup = readSetup(node);
    let sel = null;                 // {track:"scenes"|"camera", i} | {track:"energy", i}
    let drag = null;
    let playing = false, playhead = 0, lastT = 0;
    let preview = null, previewErr = "";
    let previewTimer = null;

    const root = el("div", "dfx");
    // toolbar
    const bar = el("div", "dfx-bar");
    const playBtn = el("button", "dfx-btn", "▶");
    const time = el("span", "dfx-time", "");
    const addScene = el("button", "dfx-btn", "+ Scene");
    const addCam = el("button", "dfx-btn", "+ Camera");
    const autoE = el("button", "dfx-btn", "Energy: auto");
    const del = el("button", "dfx-btn", "Delete");
    bar.append(playBtn, time, el("span", "dfx-sp"), addScene, addCam, autoE, del);
    for (const b of [playBtn, addScene, addCam, autoE, del]) b.type = "button";

    // preview + tracks
    const main = el("div", "dfx-main");
    const prev = el("div", "dfx-prev");
    const pc = el("canvas");
    const cap = el("div", "dfx-cap");
    prev.append(pc, cap);
    const tlBox = el("div", "dfx-tl");
    const tc = el("canvas");
    tlBox.append(tc);
    main.append(prev, tlBox);

    const warn = el("div", "dfx-warn");
    const ins = el("div", "dfx-ins");
    root.append(bar, main, warn, ins);

    // ---- data -----------------------------------------------------------
    function load() {
        try { tl = normalise(JSON.parse(tw.value || "{}")); } catch { tl = emptyTimeline(); }
        sortAll();
    }
    function sortAll() {
        tl.scenes.sort((a, b) => a.start - b.start);
        tl.camera.sort((a, b) => a.start - b.start);
        tl.energy.sort((a, b) => a[0] - b[0]);
    }
    function save() {
        sortAll();
        tw.value = JSON.stringify(tl);
        if (tw.callback) try { tw.callback(tw.value); } catch (_) { /* noop */ }
        node.setDirtyCanvas?.(true, true);
        schedulePreview();
    }
    const N = () => setup.frames;
    const secs = (f) => (f / setup.fps).toFixed(2) + "s";

    function schedulePreview() {
        clearTimeout(previewTimer);
        previewTimer = setTimeout(fetchPreview, 220);
    }
    async function fetchPreview() {
        const w = (n) => node.widgets?.find((x) => x.name === n)?.value;
        try {
            const r = await api.fetchApi("/difforum/preview", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    timeline: tl, frames: N(), fps: setup.fps, width: setup.width, height: setup.height,
                    mode: w("camera_mode"), transition: w("transition"), camera_scale: w("camera_scale"),
                    energy_bias: w("energy_bias"), variation: w("variation"), variation_seed: w("variation_seed"),
                }),
            });
            const j = await r.json();
            if (j.error) { previewErr = j.error; } else { preview = j; previewErr = ""; }
        } catch (e) { previewErr = "preview offline (server restart needed?)"; }
        warn.textContent = previewErr || (preview?.warnings || []).map((x) => "⚠ " + x).join("   ");
        draw();
    }

    // ---- geometry --------------------------------------------------------
    const RULER = 18, LANE = 34, PAD = 8;
    let tlH = 160;
    const lanes = () => ({
        scenes: [RULER + 4, LANE],
        camera: [RULER + 8 + LANE, LANE],
        energy: [RULER + 12 + 2 * LANE, Math.max(40, tlH - (RULER + 12 + 2 * LANE) - 6)],
    });
    function fx(frame, W) { return PAD + (frame / Math.max(1, N())) * (W - 2 * PAD); }
    function frameAt(x, W) { return clamp(Math.round(((x - PAD) / (W - 2 * PAD)) * N()), 0, N() - 1); }
    function blockEnd(list, i) { return i + 1 < list.length ? list[i + 1].start : N(); }
    function energyCurve() {
        if (tl.energy.length) return tl.energy;
        if (preview?.strength) {
            const st = preview.strength_step || 1;
            return preview.strength.map((v, k) => [k * st, v]);
        }
        return tl.scenes.map((s) => [s.start, moodOf(s.mood).strength ?? 0.5]);
    }

    // ---- drawing ---------------------------------------------------------
    function sizeCanvas(c) {
        const dpr = window.devicePixelRatio || 1;
        const r = c.getBoundingClientRect();
        const w = Math.max(10, Math.round(r.width * dpr)), h = Math.max(10, Math.round(r.height * dpr));
        if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
        const ctx = c.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        return [ctx, r.width, r.height];
    }

    function drawTracks() {
        const [ctx, W, H] = sizeCanvas(tc);
        tlH = H;
        ctx.clearRect(0, 0, W, H);
        const L = lanes();
        // ruler
        ctx.fillStyle = "#1a1a1a"; ctx.fillRect(0, 0, W, RULER);
        ctx.fillStyle = "#6c6c6c"; ctx.font = "9px sans-serif";
        const total = N() / setup.fps;
        const stepS = total > 40 ? 5 : total > 16 ? 2 : total > 6 ? 1 : 0.5;
        for (let s = 0; s <= total + 1e-6; s += stepS) {
            const x = fx(s * setup.fps, W);
            ctx.fillRect(x, RULER - 6, 1, 6);
            ctx.fillText(s % 1 ? s.toFixed(1) + "s" : s + "s", x + 2, 11);
        }
        // lane backgrounds + labels
        for (const [name, [y, h]] of Object.entries(L)) {
            ctx.fillStyle = "#181818"; ctx.fillRect(PAD, y, W - 2 * PAD, h);
            if (name === "energy" || !tl[name]?.length) {
                ctx.fillStyle = "#4a4a4a"; ctx.font = "9px sans-serif";
                ctx.fillText(name === "energy" ? "ENERGY (denoise)" : `${name.toUpperCase()} · double-click to add`,
                             PAD + 4, name === "energy" ? y + h - 4 : y + h / 2 + 3);
            }
        }
        // scenes
        tl.scenes.forEach((s, i) => {
            const [y, h] = L.scenes;
            const x0 = fx(s.start, W), x1 = fx(blockEnd(tl.scenes, i), W);
            const m = moodOf(s.mood);
            ctx.fillStyle = m.color + "cc";
            roundRect(ctx, x0 + 1, y + 2, Math.max(4, x1 - x0 - 2), h - 4, 5, true);
            if (sel?.track === "scenes" && sel.i === i) outline(ctx, x0 + 1, y + 2, x1 - x0 - 2, h - 4);
            ctx.fillStyle = "#fff"; ctx.font = "600 10px sans-serif";
            clipText(ctx, s.prompt || "(empty prompt)", x0 + 6, y + 15, x1 - x0 - 12);
            ctx.fillStyle = "#ffffffaa"; ctx.font = "9px sans-serif";
            clipText(ctx, s.mood, x0 + 6, y + 27, x1 - x0 - 12);
            grip(ctx, x0, y, h);
        });
        // camera
        tl.camera.forEach((c, i) => {
            const [y, h] = L.camera;
            const x0 = fx(c.start, W), x1 = fx(blockEnd(tl.camera, i), W);
            const mv = moveOf(c.move);
            ctx.fillStyle = mv.depth ? "#3a3222" : "#23324a";
            roundRect(ctx, x0 + 1, y + 2, Math.max(4, x1 - x0 - 2), h - 4, 5, true);
            if (sel?.track === "camera" && sel.i === i) outline(ctx, x0 + 1, y + 2, x1 - x0 - 2, h - 4);
            drawGlyph(ctx, c.move, x0 + 5, y + 6, 22, "#e6e6e6");
            ctx.fillStyle = "#e6e6e6"; ctx.font = "600 10px sans-serif";
            clipText(ctx, mv.label, x0 + 31, y + 15, x1 - x0 - 36);
            ctx.fillStyle = "#ffffff88"; ctx.font = "9px sans-serif";
            const sub = `x${(+c.speed).toFixed(2)} · ${c.ease.replace(/_/g, " ")}` +
                        (c.react && c.react !== "none" ? " · ♪" : "");
            clipText(ctx, sub, x0 + 31, y + 27, x1 - x0 - 36);
            grip(ctx, x0, y, h);
        });
        // energy
        {
            const [y, h] = L.energy;
            const pts = energyCurve();
            const auto = !tl.energy.length;
            const vy = (v) => y + h - 4 - clamp(v, 0, 1) * (h - 8);
            ctx.strokeStyle = "#2a2a2a";
            for (const q of [0.25, 0.5, 0.75]) { ctx.beginPath(); ctx.moveTo(PAD, vy(q)); ctx.lineTo(W - PAD, vy(q)); ctx.stroke(); }
            if (pts.length) {
                ctx.strokeStyle = auto ? "#6f6f6f" : "#f0b53a";
                ctx.setLineDash(auto ? [4, 3] : []);
                ctx.lineWidth = 1.6;
                ctx.beginPath();
                ctx.moveTo(PAD, vy(pts[0][1]));
                for (const [f, v] of pts) ctx.lineTo(fx(f, W), vy(v));
                ctx.lineTo(W - PAD, vy(pts[pts.length - 1][1]));
                ctx.stroke();
                ctx.setLineDash([]); ctx.lineWidth = 1;
                if (!auto) tl.energy.forEach(([f, v], i) => {
                    ctx.fillStyle = sel?.track === "energy" && sel.i === i ? "#fff" : "#f0b53a";
                    ctx.beginPath(); ctx.arc(fx(f, W), vy(v), 4, 0, 7); ctx.fill();
                });
            }
            ctx.fillStyle = "#5a5a5a"; ctx.font = "9px sans-serif";
            ctx.fillText(auto ? "auto from moods · click to draw" : "click: add · drag: move · alt-click: remove",
                         W - PAD - 190, y + 10);
        }
        // playhead
        const px = fx(playhead, W);
        ctx.fillStyle = "#4f8ef7"; ctx.fillRect(px, 0, 1.5, H);
        ctx.beginPath(); ctx.moveTo(px - 4, 0); ctx.lineTo(px + 5.5, 0); ctx.lineTo(px + 0.75, 6); ctx.fill();
    }

    function drawPreview() {
        const r = pc.getBoundingClientRect();
        pc.style.height = Math.round(r.width * (setup.height / setup.width)) + "px";
        const [ctx, W, H] = sizeCanvas(pc);
        ctx.fillStyle = "#0b0b0b"; ctx.fillRect(0, 0, W, H);
        const sx = W / setup.width, sy = H / setup.height;
        // the "world": a grid of the first frame, carried by the camera
        let a = [1, 0, 0, 0, 1, 0];
        if (preview?.affines?.length) {
            let best = preview.affines[0];
            for (const row of preview.affines) { if (row[0] <= playhead) best = row; else break; }
            a = best.slice(1);
        }
        // canvas = scale(view) . affine(camera) ; a = [a00 a01 a02 a10 a11 a12]
        const dpr = window.devicePixelRatio || 1;
        ctx.save();
        ctx.setTransform(a[0] * sx * dpr, a[3] * sy * dpr, a[1] * sx * dpr,
                         a[4] * sy * dpr, a[2] * sx * dpr, a[5] * sy * dpr);
        const gw = setup.width, gh = setup.height;
        const g = ctx.createLinearGradient(0, 0, gw, gh);
        g.addColorStop(0, "#1e3a5f"); g.addColorStop(0.5, "#2b1e4a"); g.addColorStop(1, "#4a2a1e");
        ctx.fillStyle = g; ctx.fillRect(0, 0, gw, gh);
        ctx.strokeStyle = "#ffffff30"; ctx.lineWidth = Math.max(1, gw / 300);
        for (let i = 0; i <= 12; i++) {
            ctx.beginPath(); ctx.moveTo((i / 12) * gw, 0); ctx.lineTo((i / 12) * gw, gh); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(0, (i / 12) * gh); ctx.lineTo(gw, (i / 12) * gh); ctx.stroke();
        }
        ctx.strokeStyle = "#ffffffa0"; ctx.lineWidth = Math.max(1.5, gw / 200);
        ctx.beginPath(); ctx.arc(gw / 2, gh / 2, gh * 0.18, 0, 7); ctx.stroke();
        ctx.restore();
        // frame guides
        ctx.strokeStyle = "#ffffff40"; ctx.strokeRect(0.5, 0.5, W - 1, H - 1);
        ctx.strokeStyle = "#ffffff18";
        ctx.beginPath(); ctx.moveTo(W / 3, 0); ctx.lineTo(W / 3, H); ctx.moveTo(2 * W / 3, 0); ctx.lineTo(2 * W / 3, H);
        ctx.moveTo(0, H / 3); ctx.lineTo(W, H / 3); ctx.moveTo(0, 2 * H / 3); ctx.lineTo(W, 2 * H / 3); ctx.stroke();

        const sc = [...tl.scenes].reverse().find((s) => s.start <= playhead);
        const cm = [...tl.camera].reverse().find((c) => c.start <= playhead);
        cap.textContent = `${cm ? moveOf(cm.move).label : "still"} - ${sc ? sc.prompt : ""}`;
        time.textContent = `${secs(playhead)} / ${secs(N())}  (${playhead}f)`;
    }

    function roundRect(ctx, x, y, w, h, r, fill) {
        ctx.beginPath();
        ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
        ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
        fill ? ctx.fill() : ctx.stroke();
    }
    function outline(ctx, x, y, w, h) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; roundRect(ctx, x, y, w, h, 5, false); ctx.lineWidth = 1; }
    function grip(ctx, x, y, h) { ctx.fillStyle = "#ffffff30"; ctx.fillRect(x + 1, y + 6, 3, h - 12); }
    function clipText(ctx, t, x, y, maxW) {
        if (maxW < 8) return;
        let s = String(t);
        while (s.length > 1 && ctx.measureText(s).width > maxW) s = s.slice(0, -2) + "…";
        ctx.fillText(s, x, y);
    }
    function draw() { drawTracks(); drawPreview(); }

    // ---- hit testing & mouse -------------------------------------------
    function hit(x, y, W) {
        const L = lanes();
        for (const track of ["scenes", "camera"]) {
            const [ly, lh] = L[track];
            if (y < ly || y > ly + lh) continue;
            const list = tl[track];
            for (let i = list.length - 1; i >= 0; i--) {
                const x0 = fx(list[i].start, W), x1 = fx(blockEnd(list, i), W);
                if (x >= x0 && x <= x1) return { track, i, edge: x - x0 < 7 && i > 0 };
            }
            return { track, i: -1 };
        }
        const [ey, eh] = L.energy;
        if (y >= ey && y <= ey + eh) {
            const vy = (v) => ey + eh - 4 - clamp(v, 0, 1) * (eh - 8);
            for (let i = 0; i < tl.energy.length; i++) {
                const [f, v] = tl.energy[i];
                if (Math.hypot(fx(f, W) - x, vy(v) - y) < 7) return { track: "energy", i };
            }
            return { track: "energy", i: -1, value: clamp((ey + eh - 4 - y) / (eh - 8), 0, 1) };
        }
        if (y < RULER) return { track: "ruler" };
        return null;
    }

    tc.addEventListener("pointerdown", (e) => {
        const r = tc.getBoundingClientRect();
        const x = e.clientX - r.left, y = e.clientY - r.top, W = r.width;
        const h = hit(x, y, W);
        const f = frameAt(x, W);
        tc.setPointerCapture(e.pointerId);
        if (!h || h.track === "ruler") { playhead = f; drag = { kind: "scrub" }; draw(); return; }
        if (h.track === "energy") {
            if (h.i >= 0 && e.altKey) { tl.energy.splice(h.i, 1); sel = null; save(); renderInspector(); return; }
            if (h.i < 0) {
                if (!tl.energy.length) tl.energy = energyCurve().map(([ff, v]) => [ff, +(+v).toFixed(3)])
                    .filter((_, k, arr) => arr.length < 24 || k % Math.ceil(arr.length / 12) === 0);
                tl.energy.push([f, +h.value.toFixed(3)]);
                sortAll();
                h.i = tl.energy.findIndex((p) => p[0] === f);
            }
            sel = { track: "energy", i: h.i };
            drag = { kind: "energy", i: h.i };
            renderInspector(); draw(); return;
        }
        if (h.i < 0) { sel = null; renderInspector(); playhead = f; draw(); return; }
        sel = { track: h.track, i: h.i };
        drag = { kind: h.edge ? "edge" : "move", track: h.track, i: h.i, f0: f,
                 start0: tl[h.track][h.i].start };
        playhead = tl[h.track][h.i].start;
        renderInspector(); draw();
    });
    tc.addEventListener("pointermove", (e) => {
        if (!drag) return;
        const r = tc.getBoundingClientRect();
        const W = r.width, f = frameAt(e.clientX - r.left, W);
        if (drag.kind === "scrub") { playhead = f; draw(); return; }
        if (drag.kind === "energy") {
            const [ey, eh] = lanes().energy;
            const v = clamp((ey + eh - 4 - (e.clientY - r.top)) / (eh - 8), 0.02, 0.98);
            tl.energy[drag.i] = [f, +v.toFixed(3)];
            draw(); return;
        }
        const list = tl[drag.track];
        const i = drag.i;
        const lo = i > 0 ? list[i - 1].start + 2 : 0;
        const hi = i + 1 < list.length ? list[i + 1].start - 2 : N() - 2;
        if (i === 0) return;                       // the first block always starts at 0
        list[i].start = clamp(drag.start0 + (f - drag.f0), lo, hi);
        playhead = list[i].start;
        draw();
    });
    tc.addEventListener("pointerup", () => {
        if (drag && drag.kind !== "scrub") save();
        drag = null;
        renderInspector();
    });
    tc.addEventListener("dblclick", (e) => {
        const r = tc.getBoundingClientRect();
        const h = hit(e.clientX - r.left, e.clientY - r.top, r.width);
        const f = frameAt(e.clientX - r.left, r.width);
        if (h?.track === "scenes") addBlock("scenes", f);
        else if (h?.track === "camera") addBlock("camera", f);
    });

    function addBlock(track, f) {
        f = clamp(Math.round(f), 0, N() - 2);
        const list = tl[track];
        const start = list.length ? f : 0;
        if (list.some((b) => Math.abs(b.start - start) < 2)) return;   // one block per spot
        let blk;
        if (track === "scenes") {
            const prevS = [...list].reverse().find((s) => s.start <= f);
            blk = { start, mood: prevS?.mood || "calm", prompt: prevS?.prompt || "" };
        } else {
            const prevC = [...list].reverse().find((c) => c.start <= f);
            blk = { start, move: "zoom_in", speed: prevC?.speed ?? 1, intensity: prevC?.intensity ?? 1,
                    lens: 0, ease: "ease_in_out", react: "none" };
        }
        list.push(blk);
        sortAll();
        sel = { track, i: list.indexOf(blk) };
        save(); renderInspector(); draw();
    }

    // ---- inspector ---------------------------------------------------------
    function slider(label, value, min, max, step, onInput) {
        const wrap = el("label", "dfx-sl");
        const lab = el("span", "dfx-lbl", label);
        const r = el("input"); r.type = "range"; r.min = min; r.max = max; r.step = step; r.value = value;
        const b = el("b", null, (+value).toFixed(2));
        r.addEventListener("input", () => { b.textContent = (+r.value).toFixed(2); onInput(+r.value); draw(); });
        r.addEventListener("change", save);
        wrap.append(lab, r, b);
        return wrap;
    }
    function select(options, value, onChange) {
        const s = el("select");
        for (const [v, t] of options) s.add(new Option(t, v));
        s.value = value;
        s.addEventListener("change", () => { onChange(s.value); save(); renderInspector(); draw(); });
        return s;
    }

    function renderInspector() {
        ins.replaceChildren();
        if (!sel || (sel.track !== "energy" && !tl[sel.track]?.[sel.i])) {
            ins.append(el("div", "dfx-hint",
                "Click a block to edit it · double-click a lane to add a block · drag a block's left edge to retime · " +
                "click the ruler to scrub · ▶ plays the camera the renderer will use."));
            return;
        }
        if (sel.track === "energy") {
            const p = tl.energy[sel.i];
            if (!p) return;
            ins.append(el("div", "dfx-hint",
                `Energy point at ${secs(p[0])}: ${p[1].toFixed(2)} denoise. Higher = the model re-imagines more ` +
                "each frame (more morphing); lower = steadier. Alt-click removes a point."));
            return;
        }
        if (sel.track === "scenes") {
            const s = tl.scenes[sel.i];
            const r1 = el("div", "dfx-row");
            r1.append(el("span", "dfx-lbl", `Scene @ ${secs(s.start)}`));
            for (const m of CATALOG.moods) {
                const c = el("span", "dfx-chip" + (m.id === s.mood ? " on" : ""), m.id);
                c.style.background = m.color;
                c.addEventListener("click", () => { s.mood = m.id; save(); renderInspector(); draw(); });
                r1.append(c);
            }
            const ta = el("textarea");
            ta.placeholder = "What is on screen in this scene…";
            ta.value = s.prompt;
            ta.addEventListener("input", () => { s.prompt = ta.value; draw(); });
            ta.addEventListener("change", save);
            ta.addEventListener("keydown", (e) => e.stopPropagation());
            ins.append(r1, ta);
            return;
        }
        // camera block
        const c = tl.camera[sel.i];
        const grid = el("div", "dfx-moves");
        const groups = { basic: "FLAT - works in 2D", space: "SPACE - real parallax with 3D + depth", fx: "MOTION FX" };
        for (const [gid, gname] of Object.entries(groups)) {
            grid.append(el("div", "dfx-grp", gname));
            for (const mv of CATALOG.moves.filter((m) => (m.group || "basic") === gid)) {
                const b = el("div", "dfx-mv" + (mv.id === c.move ? " on" : ""));
                b.title = mv.hint || mv.label;
                const cv = el("canvas"); cv.width = 44; cv.height = 44; cv.style.width = "22px"; cv.style.height = "22px";
                const g = cv.getContext("2d"); g.scale(2, 2); drawGlyph(g, mv.id, 0, 0, 22, mv.id === c.move ? "#fff" : "#bdbdbd");
                b.append(cv, el("span", null, mv.label));
                if (mv.depth) b.append(el("i", null, "3D"));
                b.addEventListener("click", () => { c.move = mv.id; save(); renderInspector(); draw(); });
                grid.append(b);
            }
        }
        const r2 = el("div", "dfx-row");
        r2.append(
            slider("Speed", c.speed, 0.1, 4, 0.05, (v) => { c.speed = v; }),
            slider("Amount", c.intensity, 0, 3, 0.05, (v) => { c.intensity = v; }),
        );
        const r3 = el("div", "dfx-row");
        r3.append(
            el("span", "dfx-lbl", "Lens"), select(LENSES.map(([v, t]) => [String(v), t]), String(c.lens || 0),
                (v) => { c.lens = Number(v); }),
            el("span", "dfx-lbl", "Ease in"), select(CATALOG.easings.map((e) => [e, e.replace(/_/g, " ")]), c.ease,
                (v) => { c.ease = v; }),
            el("span", "dfx-lbl", "Audio"), select(CATALOG.reactions.map((r) => [r.id, REACT_LABEL[r.id] || r.id]),
                c.react || "none", (v) => { c.react = v; }),
        );
        const head = el("div", "dfx-head");
        head.append(el("b", null, `${moveOf(c.move).label} @ ${secs(c.start)}`),
                    el("span", "dfx-hint", "  " + (moveOf(c.move).hint || "")));
        ins.append(head, grid, r2, r3);
    }

    // ---- toolbar -----------------------------------------------------------
    addScene.onclick = () => addBlock("scenes", playhead);
    addCam.onclick = () => addBlock("camera", playhead);
    autoE.onclick = () => { tl.energy = []; sel = null; save(); renderInspector(); draw(); };
    del.onclick = () => {
        if (!sel) return;
        if (sel.track === "energy") tl.energy.splice(sel.i, 1);
        else if (tl[sel.track].length > 1) {
            tl[sel.track].splice(sel.i, 1);
            if (tl[sel.track][0]) tl[sel.track][0].start = 0;
        }
        sel = null; save(); renderInspector(); draw();
    };
    playBtn.onclick = () => {
        playing = !playing;
        playBtn.textContent = playing ? "❚❚" : "▶";
        playBtn.classList.toggle("on", playing);
        if (playing) {
            if (playhead >= N() - 1) playhead = 0;
            phF = playhead; lastT = performance.now(); requestAnimationFrame(tick);
        }
    };
    let phF = 0;
    function tick(t) {
        if (!playing || !root.isConnected) { playing = false; playBtn.textContent = "▶"; return; }
        phF = (playhead === Math.floor(phF) ? phF : playhead) + ((t - lastT) / 1000) * setup.fps;
        lastT = t;
        if (phF >= N() - 1) phF = 0;
        playhead = Math.floor(phF);
        draw();
        requestAnimationFrame(tick);
    }

    // keep in sync with Setup / widget changes
    const syncTimer = setInterval(() => {
        if (!root.isConnected) return;
        const s = readSetup(node);
        if (s.frames !== setup.frames || s.fps !== setup.fps || s.width !== setup.width || s.height !== setup.height) {
            setup = s; schedulePreview(); draw();
        }
    }, 800);
    const ro = new ResizeObserver(() => draw());
    ro.observe(root);

    loadCatalog().then(() => { renderInspector(); draw(); });
    load();
    renderInspector();
    schedulePreview();

    return {
        root,
        reload: () => { load(); sel = null; renderInspector(); schedulePreview(); draw(); },
        refresh: () => schedulePreview(),
        dispose: () => { playing = false; clearInterval(syncTimer); ro.disconnect(); },
    };
}

// ---------------------------------------------------------------------------
app.registerExtension({
    name: "Difforum.DirectorTimeline",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onCreated?.apply(this, arguments);
            injectStyles();
            const tw = this.widgets?.find((w) => w.name === "timeline");
            if (!tw) return;
            tw.computeSize = () => [0, -4];
            tw.hidden = true;
            if (tw.inputEl) tw.inputEl.style.display = "none";
            if (tw.element) tw.element.style.display = "none";

            const ed = buildEditor(this, tw);
            this.addDOMWidget("difforum_timeline", "div", ed.root, {
                serialize: false,
                hideOnZoom: false,
                getHeight: () => 640,
                getMinHeight: () => 560,
            });
            // other widgets (camera_mode, transition...) refresh the preview
            for (const w of this.widgets || []) {
                if (w === tw || w.name === "difforum_timeline") continue;
                const cb = w.callback;
                w.callback = function () { const r = cb?.apply(this, arguments); ed.refresh(); return r; };
            }
            const onDraw = this.onDrawForeground;
            this.onDrawForeground = function (...a) {
                if (tw.inputEl) tw.inputEl.style.display = "none";
                if (tw.element) tw.element.style.display = "none";
                return onDraw?.apply(this, a);
            };
            this.__dfx = ed;
            this.setSize([Math.max(this.size[0], 800), Math.max(this.size[1], 880)]);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            setTimeout(() => this.__dfx?.reload(), 30);
        };
        const onConn = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const r = onConn?.apply(this, arguments);
            setTimeout(() => this.__dfx?.refresh(), 30);
            return r;
        };
        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this.__dfx?.dispose();
            return onRemoved?.apply(this, arguments);
        };
    },
});
