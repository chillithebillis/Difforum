/**
 * Difforum · Film Director — visual timeline widget
 * =================================================
 * Draws a drag-and-drop shot timeline on top of the node's `timeline` string
 * widget. The string widget stays the source of truth, so the workflow
 * serialises exactly as ComfyUI expects; the timeline only edits it.
 *
 * Pointer handling note: the overlay root is `pointer-events: none` and only
 * its interactive children opt back in. Without that, the DOM element covers
 * the numeric widgets above it on the node and their arrows stop responding.
 */

import { app } from "../../scripts/app.js";

const NODE_ID = "DifforumFilmDirector";

const MOODS = {
    calm:    { color: "#4a90d9", label: "Calm" },
    build:   { color: "#e0a020", label: "Build" },
    tense:   { color: "#e2703a", label: "Tense" },
    climax:  { color: "#a45ec4", label: "Climax" },
    resolve: { color: "#37a86b", label: "Resolve" },
    dream:   { color: "#1f9c8c", label: "Dream" },
};
const MOOD_KEYS = Object.keys(MOODS);

const CAMERAS = [
    "still", "zoom_in", "zoom_out", "dolly_in", "dolly_out",
    "pan_left", "pan_right", "pan_up", "pan_down",
    "orbit_left", "orbit_right", "roll_cw", "roll_ccw",
    "spiral", "sway", "dolly_zoom", "rise", "shake",
];

const EASINGS = ["ease_in_out", "ease_in", "ease_out", "linear", "step"];

// Field of view in degrees. `auto` lets the mood preset choose.
const LENSES = [
    { v: 0,   label: "auto" },
    { v: 16,  label: "16° ultra tele" },
    { v: 24,  label: "24° tele" },
    { v: 32,  label: "32° long" },
    { v: 40,  label: "40° normal" },
    { v: 50,  label: "50° wide" },
    { v: 65,  label: "65° very wide" },
    { v: 85,  label: "85° ultra wide" },
    { v: 110, label: "110° fisheye" },
];

const FALLBACK = [
    { frame_start: 0,   mood: "calm",    camera: "dolly_in",    prompt: "ancient forest, golden hour, wide establishing shot" },
    { frame_start: 60,  mood: "build",   camera: "orbit_right", prompt: "roots and moss, glowing details, warm rim light" },
    { frame_start: 120, mood: "climax",  camera: "spiral",      prompt: "explosion of light, cosmic transformation" },
    { frame_start: 180, mood: "resolve", camera: "zoom_out",    prompt: "embers becoming stars, calm, vast scale" },
];

// ---------------------------------------------------------------------------

function injectStyles() {
    if (document.getElementById("difforum-director-css")) return;
    const el = document.createElement("style");
    el.id = "difforum-director-css";
    el.textContent = `
.dfd { font: 11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       color: #c8c8c8; background: #1b1b1b; border-radius: 6px;
       display: flex; flex-direction: column; box-sizing: border-box;
       height: 100%; max-height: 100%; overflow: hidden;
       pointer-events: none; }
.dfd-bar, .dfd-tl, .dfd-ed { pointer-events: auto; }

.dfd-bar { display: flex; align-items: center; gap: 5px; padding: 4px 7px;
           background: #262626; border-bottom: 1px solid #333; flex: 0 0 auto; }
.dfd-btn { background: #383838; border: 1px solid #525252; color: #ddd;
           border-radius: 4px; padding: 2px 9px; cursor: pointer; font-size: 11px;
           white-space: nowrap; font-family: inherit; }
.dfd-btn:hover { background: #474747; }
.dfd-btn.pri { background: #24559e; border-color: #3a76cf; }
.dfd-btn.pri:hover { background: #2d66bb; }
.dfd-btn.del { background: #5c2323; border-color: #8a3333; color: #ffa0a0; }
.dfd-btn.del:hover { background: #742c2c; }
.dfd-stat { margin-left: auto; color: #6a6a6a; font-size: 10px; white-space: nowrap; }

.dfd-tl { position: relative; flex: 1 1 auto; min-height: 60px;
          background: #151515; overflow: hidden; }
.dfd-ruler { position: absolute; inset: 0 0 auto 0; height: 14px;
             background: #202020; border-bottom: 1px solid #303030; }
.dfd-tick { position: absolute; top: 0; height: 14px; border-left: 1px solid #3a3a3a;
            padding-left: 3px; font-size: 9px; color: #575757; line-height: 14px; }
.dfd-track { position: absolute; inset: 14px 0 0 0; }
.dfd-scene { position: absolute; top: 4px; bottom: 4px; border-radius: 4px;
             border: 1px solid; box-sizing: border-box; cursor: grab;
             padding: 3px 11px 3px 6px; overflow: hidden; display: flex;
             flex-direction: column; justify-content: center; gap: 1px; }
.dfd-scene:hover { filter: brightness(1.2); }
.dfd-scene.sel { box-shadow: inset 0 0 0 1px #fff; }
.dfd-t1 { font-size: 10px; font-weight: 600; white-space: nowrap;
          overflow: hidden; text-overflow: ellipsis; }
.dfd-t2 { font-size: 9px; color: #ffffff8c; white-space: nowrap;
          overflow: hidden; text-overflow: ellipsis; }
.dfd-grip { position: absolute; right: 0; top: 0; bottom: 0; width: 8px;
            cursor: ew-resize; background: #ffffff14; }
.dfd-grip:hover { background: #ffffff38; }

.dfd-ed { padding: 5px 7px 6px; background: #202020;
          border-top: 1px solid #2d2d2d; flex: 0 0 auto;
          display: flex; flex-direction: column; gap: 4px; }
.dfd-ed[hidden] { display: none; }
.dfd-row { display: grid; gap: 5px; align-items: end; }
.dfd-row.r1 { grid-template-columns: 1fr auto; }
.dfd-row.r2 { grid-template-columns: 1fr 1fr 1fr 1fr; }
.dfd-f { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.dfd-f label { font-size: 9px; color: #6a6a6a; text-transform: uppercase;
               letter-spacing: .04em; }
.dfd-f input, .dfd-f select { background: #2d2d2d; border: 1px solid #464646;
    color: #ddd; border-radius: 3px; padding: 3px 5px; font-size: 11px;
    width: 100%; box-sizing: border-box; outline: none; font-family: inherit; }
.dfd-f input:focus, .dfd-f select:focus { border-color: #4a90d9; }
`;
    document.head.appendChild(el);
}

// ---------------------------------------------------------------------------

function buildTimeline(node, textWidget) {
    let scenes = [];
    let sel = null;
    let drag = null;

    const root = document.createElement("div");
    root.className = "dfd";

    // toolbar
    const bar = document.createElement("div");
    bar.className = "dfd-bar";
    const addBtn = mkBtn("+ Shot", "pri");
    const spreadBtn = mkBtn("Spread");
    const stat = document.createElement("span");
    stat.className = "dfd-stat";
    bar.append(addBtn, spreadBtn, stat);

    // timeline
    const tl = document.createElement("div");
    tl.className = "dfd-tl";
    const ruler = document.createElement("div");
    ruler.className = "dfd-ruler";
    const track = document.createElement("div");
    track.className = "dfd-track";
    tl.append(ruler, track);

    // editor
    const ed = document.createElement("div");
    ed.className = "dfd-ed";
    ed.hidden = true;

    const row1 = document.createElement("div");
    row1.className = "dfd-row r1";
    const [promptF, promptI] = mkField("Prompt", "input");
    promptI.placeholder = "describe this shot...";
    const delBtn = mkBtn("Delete", "del");
    row1.append(promptF, delBtn);

    const row2 = document.createElement("div");
    row2.className = "dfd-row r2";
    const [moodF, moodS] = mkField("Mood", "select");
    for (const k of MOOD_KEYS) moodS.add(new Option(MOODS[k].label, k));
    const [camF, camS] = mkField("Move", "select");
    for (const c of CAMERAS) camS.add(new Option(c.replace(/_/g, " "), c));
    const [lensF, lensS] = mkField("Lens", "select");
    for (const l of LENSES) lensS.add(new Option(l.label, String(l.v)));
    const [easeF, easeS] = mkField("Ease in", "select");
    for (const e of EASINGS) easeS.add(new Option(e.replace(/_/g, " "), e));
    row2.append(moodF, camF, lensF, easeF);

    ed.append(row1, row2);
    root.append(bar, tl, ed);

    function mkBtn(text, cls) {
        const b = document.createElement("button");
        b.className = "dfd-btn" + (cls ? " " + cls : "");
        b.textContent = text;
        b.type = "button";
        return b;
    }
    function mkField(label, tag) {
        const w = document.createElement("div");
        w.className = "dfd-f";
        const l = document.createElement("label");
        l.textContent = label;
        const c = document.createElement(tag);
        if (tag === "input") c.type = "text";
        w.append(l, c);
        return [w, c];
    }
    function totalFrames() {
        const last = scenes.length ? Math.max(...scenes.map((s) => s.frame_start)) : 0;
        return Math.max(120, Math.ceil((last + 60) / 60) * 60);
    }
    const pct = (f) => (f / totalFrames()) * 100;

    function load() {
        try {
            const v = JSON.parse(textWidget.value);
            scenes = Array.isArray(v) && v.length ? v : structuredClone(FALLBACK);
        } catch {
            scenes = structuredClone(FALLBACK);
        }
        scenes.sort((a, b) => a.frame_start - b.frame_start);
    }
    function save() {
        scenes.sort((a, b) => a.frame_start - b.frame_start);
        node.__dfdLastWritten = JSON.stringify(scenes, null, 1);
        textWidget.value = node.__dfdLastWritten;
        node.setDirtyCanvas?.(true, true);
    }

    function render() {
        const total = totalFrames();

        ruler.innerHTML = "";
        const step = total <= 240 ? 60 : total <= 600 ? 120 : 240;
        for (let f = 0; f <= total; f += step) {
            const t = document.createElement("div");
            t.className = "dfd-tick";
            t.style.left = pct(f) + "%";
            t.textContent = f;
            ruler.appendChild(t);
        }

        track.innerHTML = "";
        scenes.forEach((sc, i) => {
            const mood = MOODS[sc.mood] || MOODS.calm;
            const next = i + 1 < scenes.length ? scenes[i + 1].frame_start : total;
            const left = pct(sc.frame_start);
            const width = Math.max(pct(next) - left - 0.4, 2);

            const el = document.createElement("div");
            el.className = "dfd-scene" + (i === sel ? " sel" : "");
            el.style.left = left + "%";
            el.style.width = width + "%";
            el.style.background = mood.color + "2e";
            el.style.borderColor = mood.color + "aa";

            const lensTxt = sc.lens ? ` · ${sc.lens}°` : "";
            const t1 = document.createElement("div");
            t1.className = "dfd-t1";
            t1.style.color = mood.color;
            t1.textContent = `${sc.frame_start}f · ${(sc.camera || "still").replace(/_/g, " ")}${lensTxt}`;

            const t2 = document.createElement("div");
            t2.className = "dfd-t2";
            t2.textContent = sc.prompt || "—";

            const grip = document.createElement("div");
            grip.className = "dfd-grip";
            el.append(t1, t2, grip);

            el.addEventListener("pointerdown", (e) => {
                e.stopPropagation();
                e.preventDefault();
                select(i);
                if (i === 0 && e.target !== grip) return;   // frame 0 stays pinned
                drag = {
                    mode: e.target === grip ? "resize" : "move",
                    i,
                    x0: e.clientX,
                    f0: sc.frame_start,
                    nf0: i + 1 < scenes.length ? scenes[i + 1].frame_start : null,
                    w: track.getBoundingClientRect().width,
                };
            });

            track.appendChild(el);
        });

        stat.textContent = `${scenes.length} shots · ${total}f`;
    }

    function select(i) {
        sel = i;
        const sc = scenes[i];
        if (!sc) { ed.hidden = true; return; }
        ed.hidden = false;
        promptI.value = sc.prompt || "";
        moodS.value = MOODS[sc.mood] ? sc.mood : "calm";
        camS.value = CAMERAS.includes(sc.camera) ? sc.camera : "still";
        lensS.value = String(sc.lens || 0);
        easeS.value = EASINGS.includes(sc.ease) ? sc.ease : "ease_in_out";
        render();
    }

    addBtn.onclick = () => {
        const last = scenes.length ? scenes[scenes.length - 1].frame_start : -60;
        scenes.push({
            frame_start: last + 60,
            mood: MOOD_KEYS[scenes.length % MOOD_KEYS.length],
            camera: "still",
            prompt: "",
        });
        save(); select(scenes.length - 1); render();
    };

    spreadBtn.onclick = () => {
        const n = scenes.length;
        if (n < 2) return;
        const span = scenes[n - 1].frame_start;
        const step = Math.max(12, Math.round(span / (n - 1) / 6) * 6);
        scenes.forEach((s, i) => { s.frame_start = i * step; });
        save(); render();
    };

    delBtn.onclick = () => {
        if (sel == null || scenes.length <= 1) return;
        scenes.splice(sel, 1);
        sel = null; ed.hidden = true;
        save(); render();
    };

    const set = (key, val) => {
        if (sel == null) return;
        if (val === null || val === undefined || val === "" || val === 0) {
            delete scenes[sel][key];
        } else {
            scenes[sel][key] = val;
        }
        save(); render();
    };

    promptI.oninput = () => { if (sel != null) { scenes[sel].prompt = promptI.value; save(); render(); } };
    moodS.onchange = () => set("mood", moodS.value);
    camS.onchange  = () => set("camera", camS.value);
    lensS.onchange = () => set("lens", parseInt(lensS.value, 10));
    easeS.onchange = () => set("ease", easeS.value);

    const onMove = (e) => {
        if (!drag) return;
        const d = Math.round(((e.clientX - drag.x0) / drag.w) * totalFrames());
        if (drag.mode === "move") {
            const prev = drag.i > 0 ? scenes[drag.i - 1].frame_start + 6 : 0;
            const next = drag.i + 1 < scenes.length ? scenes[drag.i + 1].frame_start - 6 : 1e9;
            scenes[drag.i].frame_start = Math.max(prev, Math.min(next, drag.f0 + d));
        } else if (drag.nf0 != null) {
            const after = drag.i + 2 < scenes.length ? scenes[drag.i + 2].frame_start - 6 : 1e9;
            scenes[drag.i + 1].frame_start =
                Math.max(drag.f0 + 6, Math.min(after, drag.nf0 + d));
        }
        render();
    };
    const onUp = () => { if (drag) { drag = null; save(); render(); } };

    document.addEventListener("pointermove", onMove);
    document.addEventListener("pointerup", onUp);

    load();
    render();

    return {
        root,
        reload: () => { load(); sel = null; ed.hidden = true; render(); },
        dispose: () => {
            document.removeEventListener("pointermove", onMove);
            document.removeEventListener("pointerup", onUp);
        },
    };
}

// ---------------------------------------------------------------------------

app.registerExtension({
    name: "Difforum.FilmDirector",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onCreated?.apply(this, arguments);
            injectStyles();

            const tw = this.widgets?.find((w) => w.name === "timeline");
            if (!tw) return;

            // Collapse the raw JSON widget to zero height but keep it serialising.
            // [0, -4] is ComfyUI's own convention for a hidden widget slot.
            tw.computeSize = () => [0, -4];
            tw.hidden = true;
            const hideInput = () => {
                if (tw.inputEl) {
                    tw.inputEl.style.display = "none";
                    tw.inputEl.style.pointerEvents = "none";
                }
            };
            hideInput();

            const ui = buildTimeline(this, tw);

            this.addDOMWidget("difforum_timeline_ui", "div", ui.root, {
                serialize: false,
                hideOnZoom: false,
                getHeight: () => 210,
            });

            // ComfyUI re-shows the textarea on redraw, so pin it closed.
            const onDraw = nodeType.prototype.onDrawForeground;
            this.onDrawForeground = function (...args) {
                hideInput();
                return onDraw?.apply(this, args);
            };

            this.__dfdReload = ui.reload;
            this.__dfdDispose = ui.dispose;
            this.setSize([600, 470]);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            setTimeout(() => this.__dfdReload?.(), 40);
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this.__dfdDispose?.();
            return onRemoved?.apply(this, arguments);
        };
    },
});
