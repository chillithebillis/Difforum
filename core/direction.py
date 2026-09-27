"""
Timeline direction: the data model behind the Director's visual timeline.

A timeline has independent tracks, so camera cuts do not have to line up with
prompt changes:

    scenes    [{start, prompt, mood}]                        what is on screen
    camera    [{start, move, speed, intensity, lens, ease, react}]  the camera
    energy    [[frame, strength], ...]                       denoise envelope
    guidance  [[frame, cfg], ...]            (optional)      cfg envelope

`build_direction` turns it into everything a renderer needs - camera axis
curves, lens, strength and cfg per frame, prompt keyframes and a plain-language
camera description - with no ComfyUI or torch dependency, so it is fully
testable and can back a live preview endpoint.

The 0.x Film Director format (a JSON list of scenes that each carried a camera)
is still accepted and converted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from .camera_keys import DEFAULT_LENS, EASINGS, keys_list_to_axis_values
from .camera_presets import CAMERA_PRESETS, MOVE_INFO, needs_depth

MOODS = {
    #           denoise  speed  amplitude  lens  ease
    "calm":    {"strength": 0.44, "speed": 0.6, "intensity": 0.5, "lens": 46.0, "ease": "ease_in_out"},
    "build":   {"strength": 0.52, "speed": 1.0, "intensity": 0.8, "lens": 40.0, "ease": "ease_in_out"},
    "tense":   {"strength": 0.57, "speed": 1.2, "intensity": 1.0, "lens": 32.0, "ease": "ease_in"},
    "climax":  {"strength": 0.63, "speed": 1.6, "intensity": 1.3, "lens": 62.0, "ease": "ease_in"},
    "resolve": {"strength": 0.44, "speed": 0.5, "intensity": 0.45, "lens": 44.0, "ease": "ease_out"},
    "dream":   {"strength": 0.48, "speed": 0.4, "intensity": 0.6, "lens": 52.0, "ease": "ease_in_out"},
}

MOOD_COLORS = {
    "calm": "#4a90d9", "build": "#e0a020", "tense": "#e2703a",
    "climax": "#a45ec4", "resolve": "#37a86b", "dream": "#1f9c8c",
}

# audio reaction per camera block (applied when an audio analysis is connected)
REACTIONS = {
    "none": "No audio reaction",
    "beat_pulse": "Zoom punches on every beat",
    "onset_shake": "Shakes on transients",
    "bass_speed": "Bass drives the move's speed",
    "mid_sway": "Mids rock the roll",
}

# Render looks. One choice drives both render paths:
#   * the Feedback Sampler reads the engine keys (colour lock, detail, grain)
#     and `energy` (added to the Director's denoise curve);
#   * prompt-driven video models (MiniMax H3, LTX...) get `prompt` appended to
#     the camera text, so the same look is asked for in words.
LOOKS = {
    "cinematic": dict(
        color_coherence=0.85, color_mode="lab", sharpen=0.25, noise=0.025, energy=0.0,
        prompt="Cinematic live-action footage, natural motion blur, stable geometry, subtle film grain."),
    "documentary": dict(
        color_coherence=0.90, color_mode="lab", sharpen=0.15, noise=0.015, energy=-0.03,
        prompt="Documentary realism, natural light, true-to-life textures, observational camera."),
    "deforum_morph": dict(
        color_coherence=0.65, color_mode="lab", sharpen=0.30, noise=0.040, energy=0.08,
        prompt="Deforum-style AI animation: the image keeps morphing and re-imagining itself as the "
               "camera travels, shapes melting into new forms, painterly detail, dreamlike zoom tunnel."),
    "animatediff_dream": dict(
        color_coherence=0.75, color_mode="lab", sharpen=0.15, noise=0.030, energy=0.05,
        prompt="AnimateDiff-style animation: fluid, dreamy morphing between illustrated frames, soft "
               "painterly textures, gentle temporal shimmer, stylised anime-inspired rendering."),
    "psychedelic": dict(
        color_coherence=0.60, color_mode="rgb", sharpen=0.20, noise=0.045, energy=0.10,
        prompt="Psychedelic visuals: saturated shifting colours, kaleidoscopic symmetry, liquid "
               "morphing patterns, hypnotic flow."),
    "music_video": dict(
        color_coherence=0.75, color_mode="lab", sharpen=0.35, noise=0.035, energy=0.03,
        prompt="90s music video shot on VHS: chroma bleed, tape noise, slightly unstable image, "
               "punchy contrast."),
    "stop_motion": dict(
        color_coherence=0.85, color_mode="lab", sharpen=0.30, noise=0.020, energy=0.0,
        prompt="Stop-motion animation: tactile handmade materials, stepped motion at 12 frames per "
               "second, miniature set lighting."),
    "hand_drawn": dict(
        color_coherence=0.80, color_mode="lab", sharpen=0.20, noise=0.020, energy=0.04,
        prompt="Hand-drawn 2D animation: visible line work and brush texture, boiling lines, limited "
               "palette."),
}
ENGINE_LOOK_KEYS = ("color_coherence", "color_mode", "sharpen", "noise")


def default_timeline(frames: int = 120) -> dict:
    """A safe starting timeline: only moves that work without a depth map."""
    q = max(1, frames // 4)
    return {
        "version": 2,
        "scenes": [
            {"start": 0, "mood": "calm", "prompt": "misty ancient forest at dawn, volumetric light, wide shot"},
            {"start": q, "mood": "build", "prompt": "glowing roots and moss, bioluminescent details, macro"},
            {"start": 2 * q, "mood": "climax", "prompt": "explosion of light, cosmic transformation, high contrast"},
            {"start": 3 * q, "mood": "resolve", "prompt": "embers becoming stars, calm, vast cosmic scale"},
        ],
        "camera": [
            {"start": 0, "move": "zoom_in", "speed": 0.8, "intensity": 0.8, "ease": "ease_in_out"},
            {"start": q, "move": "pan_right", "speed": 1.0, "intensity": 0.8, "ease": "ease_in_out"},
            {"start": 2 * q, "move": "spiral", "speed": 1.4, "intensity": 1.1, "ease": "ease_in"},
            {"start": 3 * q, "move": "zoom_out", "speed": 0.6, "intensity": 0.5, "ease": "ease_out"},
        ],
        "energy": [],
        "guidance": [],
    }


@dataclass
class Direction:
    frames: int
    fps: float
    mode: str
    axes: dict[str, list[float]]
    lens: list[float]
    strength: list[float]
    cfg: list[float] | None
    prompts: list[tuple[int, str]]
    scenes: list[dict]
    camera_blocks: list[dict]
    camera_text: str
    warnings: list[str] = field(default_factory=list)
    summary: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def _num(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def parse_timeline(raw) -> dict:
    """Timeline JSON (v2 dict or 0.x scene list, or a `frame | mood | move |
    prompt` table) -> normalised v2 dict."""
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {"version": 2, "scenes": [], "camera": [], "energy": [], "guidance": []}
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            raw = _parse_table(text)

    if isinstance(raw, list):              # 0.x format
        scenes, camera = [], []
        for item in raw:
            if not isinstance(item, dict):
                continue
            start = int(_num(item.get("frame_start", item.get("start", 0)), 0))
            mood = str(item.get("mood", "calm")).lower()
            scenes.append({"start": start, "mood": mood, "prompt": str(item.get("prompt", ""))})
            m = MOODS.get(mood, MOODS["calm"])
            camera.append({
                "start": start, "move": str(item.get("camera", "still")).lower(),
                "speed": m["speed"], "intensity": m["intensity"],
                "lens": _num(item.get("lens"), 0.0) or 0.0,
                "ease": item.get("ease") or m["ease"],
            })
        raw = {"version": 2, "scenes": scenes, "camera": camera, "energy": [], "guidance": []}

    if not isinstance(raw, dict):
        raw = {}

    scenes = []
    for s in raw.get("scenes", []) or []:
        if isinstance(s, dict):
            scenes.append({
                "start": max(0, int(_num(s.get("start", 0), 0))),
                "mood": str(s.get("mood", "calm")).lower(),
                "prompt": " ".join(str(s.get("prompt", "")).split()),
            })
    camera = []
    for c in raw.get("camera", []) or []:
        if isinstance(c, dict):
            move = str(c.get("move", "still")).lower()
            camera.append({
                "start": max(0, int(_num(c.get("start", 0), 0))),
                "move": move if move in CAMERA_PRESETS else "still",
                "speed": _num(c.get("speed"), 1.0),
                "intensity": _num(c.get("intensity"), 1.0),
                "lens": _num(c.get("lens"), 0.0),
                "ease": c.get("ease") if c.get("ease") in EASINGS else "ease_in_out",
                "react": c.get("react") if c.get("react") in REACTIONS else "none",
            })

    def env(key):
        pts = []
        for p in raw.get(key, []) or []:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                pts.append((max(0, int(_num(p[0], 0))), _num(p[1], 0.5)))
            elif isinstance(p, dict):
                pts.append((max(0, int(_num(p.get("frame", 0), 0))), _num(p.get("value"), 0.5)))
        return sorted(pts)

    return {
        "version": 2,
        "scenes": sorted(scenes, key=lambda s: s["start"]),
        "camera": sorted(camera, key=lambda c: c["start"]),
        "energy": env("energy"),
        "guidance": env("guidance"),
    }


def _parse_table(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        try:
            frame = int(parts[0])
        except ValueError:
            continue
        out.append({"frame_start": frame, "mood": parts[1],
                    "camera": parts[2] if len(parts) > 2 else "still",
                    "prompt": parts[3] if len(parts) > 3 else ""})
    return out


# ---------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------

def _jitter(seed: int, index: int, channel: str) -> float:
    h = hashlib.sha1(f"{seed}:{index}:{channel}".encode()).digest()
    return (int.from_bytes(h[:4], "big") / 0xFFFFFFFF) * 2.0 - 1.0


def _envelope(points: list[tuple[int, float]], n: int, smooth: bool = True) -> list[float]:
    """Piecewise curve through (frame, value) points, held flat at the ends."""
    if not points:
        return []
    pts = sorted(points)
    out = []
    j = 0
    for f in range(n):
        while j + 1 < len(pts) and pts[j + 1][0] <= f:
            j += 1
        f0, v0 = pts[j]
        if f <= pts[0][0]:
            out.append(pts[0][1])
            continue
        if j + 1 >= len(pts):
            out.append(v0)
            continue
        f1, v1 = pts[j + 1]
        u = (f - f0) / max(1, f1 - f0)
        if smooth:
            u = u * u * (3.0 - 2.0 * u)
        out.append(v0 + (v1 - v0) * u)
    return out


def mood_energy(scenes: list[dict], strength_bias: float = 0.0) -> list[tuple[int, float]]:
    """Energy points implied by the scene moods (what the 'auto' button draws)."""
    pts = []
    for s in scenes:
        m = MOODS.get(s["mood"], MOODS["calm"])
        pts.append((s["start"], round(min(0.95, max(0.05, m["strength"] + strength_bias)), 3)))
    return pts


def _react(axes: dict, blocks: list[dict], n: int, curves: dict | None, warnings: list[str]):
    if not any(b.get("react", "none") != "none" for b in blocks):
        return
    if not curves:
        warnings.append("audio reactions are set on camera blocks but no audio is connected")
        return

    def curve(name):
        c = list(curves.get(name, []))
        return (c + [c[-1] if c else 0.0] * n)[:n] if c else [0.0] * n

    beat, onset, low, mid = curve("beat"), curve("onset"), curve("low"), curve("mid")
    for i, b in enumerate(blocks):
        kind = b.get("react", "none")
        if kind == "none":
            continue
        start = min(b["start"], n)
        end = min(blocks[i + 1]["start"], n) if i + 1 < len(blocks) else n
        amt = float(b.get("intensity", 1.0))
        for f in range(start, end):
            if kind == "beat_pulse":
                axes["zoom"][f] *= 1.0 + 0.035 * amt * beat[f]
            elif kind == "onset_shake":
                sign = 1.0 if (f % 2) else -1.0
                axes["translation_x"][f] += sign * 4.0 * amt * onset[f]
                axes["translation_y"][f] -= sign * 3.0 * amt * onset[f]
            elif kind == "bass_speed":
                k = 0.3 + 1.4 * low[f]
                for ax in ("translation_x", "translation_y", "translation_z",
                           "rotation_3d_x", "rotation_3d_y", "rotation_3d_z"):
                    axes[ax][f] *= k
                axes["zoom"][f] = 1.0 + (axes["zoom"][f] - 1.0) * k
            elif kind == "mid_sway":
                axes["rotation_3d_z"][f] += 0.6 * amt * (mid[f] - 0.5)


def build_direction(
    timeline,
    frames: int,
    fps: float = 24.0,
    mode: str = "2d",
    camera_scale: float = 1.0,
    strength_bias: float = 0.0,
    blend: float = 1.0,
    variation: float = 0.0,
    variation_seed: int = 0,
    audio_curves: dict | None = None,
    has_depth: bool | None = None,
) -> Direction:
    tl = parse_timeline(timeline)
    n = max(1, int(frames))
    fps = float(fps)
    warnings: list[str] = []
    var = max(0.0, min(1.0, float(variation)))

    scenes = [s for s in tl["scenes"] if s["start"] < n]
    blocks = [b for b in tl["camera"] if b["start"] < n]
    dropped = len(tl["scenes"]) - len(scenes) + len(tl["camera"]) - len(blocks)
    if dropped:
        warnings.append(f"{dropped} block(s) start after the clip ends ({n} frames) and were ignored")
    if not blocks:
        blocks = [{"start": 0, "move": "still", "speed": 1.0, "intensity": 1.0,
                   "lens": 0.0, "ease": "ease_in_out", "react": "none"}]

    def mood_at(frame):
        cur = "calm"
        for s in scenes:
            if s["start"] <= frame:
                cur = s["mood"]
        return MOODS.get(cur, MOODS["calm"])

    keys = []
    for i, b in enumerate(blocks):
        m = mood_at(b["start"])
        js = 1.0 + var * 0.35 * _jitter(variation_seed, i, "speed")
        ja = 1.0 + var * 0.35 * _jitter(variation_seed, i, "intensity")
        jl = var * 12.0 * _jitter(variation_seed, i, "lens")
        lens = b["lens"] if b["lens"] and b["lens"] > 0 else m["lens"]
        keys.append({
            "frame": b["start"], "move": b["move"],
            "speed": round(b["speed"] * camera_scale * js, 4),
            "intensity": round(b["intensity"] * camera_scale * ja, 4),
            "lens": round(min(140.0, max(8.0, lens + jl)), 2),
            "ease": b["ease"],
        })
    axes, lens, _summary = keys_list_to_axis_values(keys, n, fps, extra_vars=audio_curves, blend=blend)
    _react(axes, blocks, n, audio_curves, warnings)

    depth_moves = sorted({MOVE_INFO[b["move"]][0] for b in blocks if needs_depth(b["move"])})
    if depth_moves and (mode == "2d"):
        warnings.append(
            f"{', '.join(depth_moves)}: 3D moves are approximated in 2d mode "
            "(dolly -> zoom, orbit -> pan). Set camera_mode 3d + a depth map for real parallax.")
    elif depth_moves and has_depth is False:
        warnings.append(
            f"{', '.join(depth_moves)}: pseudo-3D until a depth map reaches the sampler.")

    energy_pts = tl["energy"] or mood_energy(scenes, strength_bias)
    if tl["energy"] and strength_bias:
        energy_pts = [(f, min(0.95, max(0.05, v + strength_bias))) for f, v in energy_pts]
    strength = _envelope(energy_pts, n) or [0.5] * n
    if var:
        strength = [min(0.95, max(0.05, v + var * 0.06 * _jitter(variation_seed, 0, "strength")))
                    for v in strength]
    cfg = _envelope(tl["guidance"], n) if tl["guidance"] else None

    prompts = [(s["start"], s["prompt"]) for s in scenes if s["prompt"]]
    if not prompts:
        prompts = [(0, "")]
    elif prompts[0][0] != 0:
        prompts.insert(0, (0, prompts[0][1]))

    summary = []
    for s in scenes:
        summary.append(f"  scene {s['start'] / fps:6.2f}s  {s['mood']:<8} {s['prompt'][:60]}")
    for b, k in zip(blocks, keys):
        summary.append(
            f"  camera {b['start'] / fps:5.2f}s  {MOVE_INFO[b['move']][0]:<11} "
            f"x{k['speed']:<5g} amp {k['intensity']:<5g} lens {k['lens']:g}deg  {k['ease']}"
            + (f"  react:{b['react']}" if b.get("react", "none") != "none" else ""))

    return Direction(
        frames=n, fps=fps, mode=mode, axes=axes, lens=lens, strength=strength, cfg=cfg,
        prompts=prompts, scenes=scenes, camera_blocks=blocks,
        camera_text=describe_blocks(blocks, n, fps), warnings=warnings, summary=summary,
    )


# ---------------------------------------------------------------------------
# camera -> words (for prompt-driven video models: H3, LTX, Seedance...)
# ---------------------------------------------------------------------------

_PHRASE = {
    "still": "holds a locked-off frame",
    "zoom_in": "zooms in", "zoom_out": "zooms out",
    "dolly_in": "pushes in", "dolly_out": "pulls back",
    "pan_left": "pans left", "pan_right": "pans right",
    "pan_up": "tilts up", "pan_down": "tilts down",
    "tilt_up": "tilts up", "tilt_down": "tilts down",
    "orbit_left": "orbits left around the subject", "orbit_right": "orbits right around the subject",
    "roll_cw": "rolls clockwise", "roll_ccw": "rolls counter-clockwise",
    "spiral": "spirals inward, rolling as it pushes in",
    "vortex": "spirals outward, rolling as it pulls back",
    "sway": "sways gently side to side",
    "dolly_zoom": "performs a dolly zoom (vertigo effect)",
    "rise": "rises smoothly", "crane_up": "cranes up, tilting down to keep the subject",
    "shake": "shakes hard", "handheld": "floats with a subtle handheld feel",
    "drift": "drifts slowly and weightlessly", "breathe": "breathes with a gentle push and pull",
}


def _speed_word(speed: float, intensity: float) -> str:
    e = float(speed) * float(intensity)
    if e < 0.45:
        return "very slowly"
    if e < 0.8:
        return "slowly"
    if e < 1.3:
        return ""
    if e < 2.0:
        return "quickly"
    return "rapidly"


def describe_blocks(blocks: list[dict], frames: int, fps: float) -> str:
    """One flowing sentence of camera direction, in shot order."""
    parts = []
    for i, b in enumerate(blocks):
        phrase = _PHRASE.get(b["move"], b["move"].replace("_", " "))
        sw = _speed_word(b.get("speed", 1.0), b.get("intensity", 1.0))
        if sw and b["move"] not in ("still", "shake", "handheld"):
            phrase = f"{phrase} {sw}"
        if i == 0:
            parts.append(f"The camera {phrase}")
        elif b.get("ease") == "step":
            parts.append(f"then cuts and {phrase}")
        elif i == len(blocks) - 1 and i > 1:
            parts.append(f"and finally {phrase}")
        else:
            parts.append(f"then {phrase}")
    return (", ".join(parts) + ".") if parts else "The camera holds a locked-off frame."


def describe_timed(blocks: list[dict], frames: int, fps: float) -> str:
    lines = []
    for i, b in enumerate(blocks):
        end = blocks[i + 1]["start"] if i + 1 < len(blocks) else frames
        phrase = _PHRASE.get(b["move"], b["move"])
        sw = _speed_word(b.get("speed", 1.0), b.get("intensity", 1.0))
        lines.append(f"[{b['start'] / fps:.1f}s-{end / fps:.1f}s] camera {phrase}"
                     + (f" {sw}" if sw else ""))
    return "\n".join(lines)


def ui_catalog() -> dict:
    """Everything the timeline widget needs to draw its pickers - served to the
    frontend so the move list lives in exactly one place (Python)."""
    return {
        "moves": [
            {"id": k, "label": MOVE_INFO[k][0], "group": MOVE_INFO[k][1],
             "hint": MOVE_INFO[k][2], "depth": needs_depth(k)}
            for k in CAMERA_PRESETS
        ],
        "moods": [{"id": k, "color": MOOD_COLORS.get(k, "#888"), **v} for k, v in MOODS.items()],
        "easings": list(EASINGS),
        "reactions": [{"id": k, "hint": v} for k, v in REACTIONS.items()],
        "looks": [{"id": k, "prompt": v["prompt"]} for k, v in LOOKS.items()],
        "default_lens": DEFAULT_LENS,
    }


@dataclass
class DirectionBundle:
    """What travels on a DIFFORUM_DIRECTION wire: one object instead of five."""

    params: dict
    camera: object                 # core.camera.CameraTrack
    strength: object               # core.schedule.Schedule
    cfg: object | None             # core.schedule.Schedule | None
    prompts: object | None         # core.prompt.PromptTrack | None
    direction: Direction
    look_name: str = "cinematic"

    @property
    def look(self) -> dict:
        """Feedback engine settings of the chosen look."""
        lk = LOOKS.get(self.look_name, LOOKS["cinematic"])
        return {k: lk[k] for k in ENGINE_LOOK_KEYS}

    @property
    def look_prompt(self) -> str:
        return LOOKS.get(self.look_name, LOOKS["cinematic"])["prompt"]


def blocks_in_range(blocks: list[dict], start: int, end: int) -> list[dict]:
    """Camera blocks overlapping [start, end), re-timed so the range starts at 0."""
    out = []
    for i, b in enumerate(blocks):
        b_end = blocks[i + 1]["start"] if i + 1 < len(blocks) else 10**9
        if b_end <= start or b["start"] >= end:
            continue
        nb = dict(b)
        nb["start"] = max(0, b["start"] - start)
        out.append(nb)
    return out


# normalisers: per-frame magnitude that reads as a "normal speed" move
_TRACK_AXES = (
    ("zoom", 0.012, "zoom_in", "zoom_out"),
    ("tz", 1.0, "dolly_out", "dolly_in"),
    ("ry", 0.5, "orbit_right", "orbit_left"),
    ("rx", 0.35, "tilt_up", "tilt_down"),
    ("rz", 0.45, "roll_cw", "roll_ccw"),
    ("tx", 1.8, "pan_right", "pan_left"),
    ("ty", 1.8, "pan_down", "pan_up"),
)


def describe_track(camera, fps: float, window_seconds: float = 1.0) -> tuple[str, str]:
    """Plain-language description of any camera track (not only Director
    blocks): windows of ~1 s are classified by their dominant motion and runs
    of the same move are merged. Returns (sentence, timed lines)."""
    import math

    n = len(camera.deltas)
    win = max(1, int(round(window_seconds * fps)))
    blocks: list[dict] = []
    for s in range(0, n, win):
        e = min(n, s + win)
        acc = {k: 0.0 for k, *_ in _TRACK_AXES}
        for f in range(s, e):
            d = camera.deltas[f]
            acc["zoom"] += math.log(max(1e-6, float(camera.zoom[f])))
            acc["tz"] += float(d[2][3])
            acc["tx"] += float(d[0][3])
            acc["ty"] += float(d[1][3])
            acc["rz"] += math.degrees(math.atan2(float(d[1][0]), float(d[0][0])))
            acc["ry"] += math.degrees(math.asin(max(-1.0, min(1.0, float(d[0][2])))))
            acc["rx"] += math.degrees(math.atan2(-float(d[1][2]), float(d[2][2])))
        L = e - s
        best, best_v = "still", 0.25
        speed = 1.0
        for key, norm, pos, neg in _TRACK_AXES:
            if camera.mode == "2d" and key in ("tz", "ry", "rx"):
                continue
            v = acc[key] / L / norm
            if abs(v) > best_v:
                best, best_v, speed = (pos if v > 0 else neg), abs(v), abs(v)
        zv = acc["zoom"] / L / 0.012
        if best in ("roll_cw", "roll_ccw") and abs(zv) > 0.5:
            best = "spiral" if zv > 0 else "vortex"
        if blocks and blocks[-1]["move"] == best:
            continue
        blocks.append({"start": s, "move": best, "speed": speed, "intensity": 1.0,
                       "ease": "ease_in_out"})
    return describe_blocks(blocks, n, fps), describe_timed(blocks, n, fps)
