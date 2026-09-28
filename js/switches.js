/**
 * Difforum · Workflow Switches - one switch per group, plus "locate".
 *
 * A group turned off is MUTED when it holds an output node (Previz, Render:
 * nothing downstream needs it) and BYPASSED otherwise (Live Preview, Fill
 * Reveal, Restyle, Look Mix, Upscale: data passes straight through), so the
 * rest of the graph keeps running. The state lives in the nodes' own modes,
 * so it survives save/load and stays in sync with Ctrl+M / Ctrl+B.
 */

import { app } from "../../scripts/app.js";

const NODE_ID = "Difforum_Switches";
const CSS = `
.dfs { font: 12px/1.3 system-ui, sans-serif; color: #ddd; display: flex; flex-direction: column; gap: 4px;
       padding: 6px; box-sizing: border-box; height: 100%; overflow: auto; }
.dfs-row { display: flex; align-items: center; gap: 8px; padding: 5px 6px; border-radius: 6px;
           background: #1e1e1e; cursor: pointer; user-select: none; }
.dfs-row:hover { background: #262626; }
.dfs-row.locked { cursor: default; }
.dfs-sw { width: 30px; height: 16px; border-radius: 9px; background: #3a3a3a; position: relative; flex: none; }
.dfs-sw::after { content: ""; position: absolute; top: 2px; left: 2px; width: 12px; height: 12px;
                 border-radius: 50%; background: #999; transition: left .12s; }
.dfs-row.on .dfs-sw { background: var(--gc, #4f8ef7); }
.dfs-row.on .dfs-sw::after { left: 16px; background: #fff; }
.dfs-row.locked .dfs-sw { visibility: hidden; }
.dfs-t { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.dfs-row:not(.on):not(.locked) .dfs-t { color: #777; text-decoration: line-through; }
.dfs-m { color: #777; font-size: 10px; flex: none; }
.dfs-go { flex: none; border: 1px solid #3c3c3c; background: #2a2a2a; color: #ccc; border-radius: 4px;
          padding: 0 6px; cursor: pointer; }
.dfs-go:hover { background: #3a3a3a; }
.dfs-empty { color: #777; padding: 6px; }
`;

function injectStyles() {
    if (document.getElementById("dfs-css")) return;
    const s = document.createElement("style");
    s.id = "dfs-css";
    s.textContent = CSS;
    document.head.append(s);
}

const groupsOf = (g) => g?._groups || g?.groups || [];
const nodesOf = (g) => g?._nodes || g?.nodes || [];
const boundsOf = (gr) => {
    const b = gr._bounding || gr.bounding || [gr.pos?.[0] ?? 0, gr.pos?.[1] ?? 0, gr.size?.[0] ?? 0, gr.size?.[1] ?? 0];
    return [b[0], b[1], b[2], b[3]];
};
const isOutput = (n) => !!n.constructor?.nodeData?.output_node;

function membersOf(graph, gr, self) {
    const [x, y, w, h] = boundsOf(gr);
    return nodesOf(graph).filter((n) => {
        if (n === self || n.type === NODE_ID || n.isVirtualNode) return false;
        const cx = n.pos[0] + (n.size?.[0] ?? 0) / 2, cy = n.pos[1] + (n.size?.[1] ?? 0) / 2;
        return cx >= x && cx <= x + w && cy >= y && cy <= y + h;
    });
}

/** How a group turns off: the Switches node's `modes` map wins, else mute when it saves something. */
function kindOf(node, gr, members) {
    const m = node.properties?.modes?.[gr.title];
    if (m === "mute" || m === "bypass") return m;
    return members.some(isOutput) ? "mute" : "bypass";
}

function stateOf(node, gr, members) {
    const kind = kindOf(node, gr, members);
    const off = kind === "mute" ? 2 : 4;
    // off only when every member carries this group's off-mode; a single node bypassed on
    // purpose (e.g. an optional LoRA) does not turn the whole group off
    return { on: !members.length || !members.every((n) => n.mode === off), kind };
}

function linkById(graph, id) {
    const L = graph.links;
    return L?.get ? L.get(id) : L?.[id];
}

/** Output nodes fed (directly or not) by any of `members`. */
function downstreamOutputs(graph, members) {
    const seen = new Set(members.map((n) => n.id)), out = [], stack = [...members];
    while (stack.length) {
        const n = stack.pop();
        for (const o of n.outputs || []) {
            for (const id of o.links || []) {
                const l = linkById(graph, id);
                const t = l && graph.getNodeById(l.target_id);
                if (!t || seen.has(t.id)) continue;
                seen.add(t.id);
                if (isOutput(t)) out.push(t);
                stack.push(t);
            }
        }
    }
    return out;
}

function setGroup(node, gr, members, on) {
    const graph = node.graph || app.graph;
    const kind = kindOf(node, gr, members) === "mute" ? 2 : 4;
    node.properties = node.properties || {};
    const auto = node.properties.auto_muted || (node.properties.auto_muted = {});
    const saved = node.properties.saved_modes || (node.properties.saved_modes = {});
    if (on) {
        const prev = saved[gr.title] || {};
        for (const n of members) n.mode = prev[n.id] ?? 0;
        delete saved[gr.title];
    } else {
        saved[gr.title] = Object.fromEntries(members.map((n) => [n.id, n.mode]));
        for (const n of members) n.mode = kind;
    }
    if (!on && kind === 2) {
        // outputs elsewhere that depend on this group would fail validation: mute them too
        const extra = downstreamOutputs(graph, members).filter((n) => n.mode === 0);
        for (const n of extra) n.mode = 2;
        auto[gr.title] = extra.map((n) => n.id);
    } else if (on && auto[gr.title]) {
        for (const id of auto[gr.title]) { const n = graph.getNodeById(id); if (n && n.mode === 2) n.mode = 0; }
        delete auto[gr.title];
    }
    app.graph?.setDirtyCanvas?.(true, true);
}

function locate(gr) {
    const c = app.canvas;
    if (!c?.ds) return;
    const [x, y, w, h] = boundsOf(gr);
    const el = c.canvas;
    const dpr = window.devicePixelRatio || 1;
    const cw = (el?.width || 1200) / dpr, ch = (el?.height || 800) / dpr;
    const scale = Math.max(0.1, Math.min(1.2, cw / (w + 120), ch / (h + 120)));
    c.ds.scale = scale;
    c.ds.offset[0] = cw / (2 * scale) - (x + w / 2);
    c.ds.offset[1] = ch / (2 * scale) - (y + h / 2);
    c.setDirty(true, true);
}

function buildPanel(node) {
    const root = document.createElement("div");
    root.className = "dfs";
    let sig = "";

    function render(force) {
        const graph = node.graph || app.graph;
        const locked = new Set(node.properties?.locked || []);
        const rows = groupsOf(graph)
            .map((gr) => ({ gr, members: membersOf(graph, gr, node) }));   // graph order = flow order
        const nextSig = rows.map(({ gr, members }) =>
            `${gr.title}|${stateOf(node, gr, members).on}|${members.length}|${locked.has(gr.title)}`).join(";");
        if (!force && nextSig === sig) return;
        sig = nextSig;
        root.replaceChildren();
        if (!rows.length) {
            const e = document.createElement("div");
            e.className = "dfs-empty";
            e.textContent = "No groups in this workflow. Add groups (Ctrl+G) and they appear here.";
            root.append(e);
            return;
        }
        for (const { gr, members } of rows) {
            const st = stateOf(node, gr, members);
            const lock = locked.has(gr.title);
            const row = document.createElement("div");
            row.className = "dfs-row" + (st.on ? " on" : "") + (lock ? " locked" : "");
            row.style.setProperty("--gc", gr.color || "#4f8ef7");
            row.title = lock ? "Always on" :
                (st.kind === "mute" ? "Off = muted (holds an output)" : "Off = bypassed (data passes through)");
            const sw = document.createElement("span"); sw.className = "dfs-sw";
            const t = document.createElement("span"); t.className = "dfs-t"; t.textContent = gr.title || "(group)";
            const m = document.createElement("span"); m.className = "dfs-m";
            m.textContent = lock ? `${members.length}` : `${members.length} · ${st.kind}`;
            const go = document.createElement("button"); go.className = "dfs-go"; go.type = "button";
            go.textContent = "⌖"; go.title = "Show this group";
            go.addEventListener("click", (e) => { e.stopPropagation(); locate(gr); });
            row.append(sw, t, m, go);
            if (!lock) row.addEventListener("click", () => { setGroup(node, gr, members, !st.on); render(true); });
            root.append(row);
        }
        const want = 12 + rows.length * 34;
        if (Math.abs((node.__dfsH || 0) - want) > 1) {
            node.__dfsH = want;
            node.setSize([Math.max(node.size[0], 300), Math.max(node.size[1], want + 40)]);
            node.setDirtyCanvas?.(true, true);
        }
    }

    const timer = setInterval(() => { if (root.isConnected) render(false); }, 700);
    setTimeout(() => render(true), 50);
    return { root, render, dispose: () => clearInterval(timer) };
}

app.registerExtension({
    name: "Difforum.WorkflowSwitches",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) return;
        nodeType.prototype.isVirtualNode = true;      // UI only: never sent to the server
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onCreated?.apply(this, arguments);
            injectStyles();
            this.properties = this.properties || {};
            const p = buildPanel(this);
            this.__dfs = p;
            this.addDOMWidget("difforum_switches", "div", p.root, {
                serialize: false, hideOnZoom: false,
                getHeight: () => this.__dfsH || 120, getMinHeight: () => 60,
            });
            this.setSize([Math.max(this.size[0], 300), Math.max(this.size[1], 160)]);
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            setTimeout(() => this.__dfs?.render(true), 60);
        };
        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this.__dfs?.dispose();
            return onRemoved?.apply(this, arguments);
        };
    },
});
