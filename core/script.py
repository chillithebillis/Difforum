"""
Shot scripts -> Director timeline.

Lets the timeline come from outside the editor: a text written by hand or by
an LLM node, a CSV exported from a spreadsheet, a file in ComfyUI's input
folder. Everything lands in the same v2 timeline dict the Director draws.

Accepted forms (mixed freely, one beat per line):

    0s    | calm  | dolly_in slow small | misty forest at dawn, light shafts
    4.5s  | build | orbit_left fast     | glowing roots and moss
    00:09 | key: the light breaks
    @12s  the mist lifts and the camera rises          (time + prompt)
    a lone tree on a hill                               (no time: spread evenly)

- time: `4s`, `4.5s`, `00:04`, `1:02.5`, `f96` / `96` (frames)
- the fields after the time are recognised by content: a mood name
  (calm, build, tense, climax, resolve, dream), a camera move (`dolly_in`,
  `dolly in`, `push in`...) with optional modifiers (`slow`, `fast`, `small`,
  `large`, `x1.3` speed, `amp 0.8`, `35mm`, an easing, an audio reaction,
  `energy 0.6`), `key: label`, and the rest is the prompt.
- a CSV with a header row (`time,mood,camera,prompt,key,energy`) and a JSON
  timeline (what the Director saves) are accepted as they are.
"""

from __future__ import annotations

import csv
import io
import json
import re

from .camera_presets import CAMERA_PRESETS
from .direction import MOODS, REACTIONS

EASES = ("linear", "ease_in", "ease_out", "ease_in_out", "step")
ALIASES = {
    "push_in": "dolly_in", "push": "dolly_in", "pull_out": "dolly_out", "pull_back": "dolly_out",
    "truck_left": "pan_left", "truck_right": "pan_right", "pedestal_up": "pan_up", "pedestal_down": "pan_down",
    "arc_left": "orbit_left", "arc_right": "orbit_right", "static": "still", "hold": "still", "locked": "still",
    "crane": "crane_up", "boom_up": "crane_up", "vertigo": "dolly_zoom", "hand_held": "handheld",
}
SPEED_WORDS = {"very_slow": 0.4, "slow": 0.6, "slowly": 0.6, "medium": 1.0, "fast": 1.5, "quick": 1.5,
               "very_fast": 2.0}
AMP_WORDS = {"subtle": 0.4, "small": 0.5, "gentle": 0.6, "big": 1.4, "large": 1.4, "huge": 1.8, "strong": 1.4}

_KEY = re.compile(r"^(?:key|mark|beat)\s*[:\-]\s*(.*)$", re.I)


def parse_time(token: str, fps: float) -> int | None:
    t = token.strip().lower().replace(" ", "")
    if not t:
        return None
    if t.startswith("@"):
        t = t[1:]
    if re.fullmatch(r"f\d+", t):
        return int(t[1:])
    m = re.fullmatch(r"(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)", t)          # h:mm:ss / mm:ss(.x)
    if m:
        h = int(m.group(1) or 0)
        return round((h * 3600 + int(m.group(2)) * 60 + float(m.group(3))) * fps)
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(s|sec|secs|seconds)", t)
    if m:
        return round(float(m.group(1)) * fps)
    m = re.fullmatch(r"(\d+)(f|fr|frames?)?", t)
    if m:
        return int(m.group(1))
    return None


def _norm(word: str) -> str:
    return re.sub(r"[\s\-]+", "_", word.strip().lower())


def parse_camera(text: str):
    """'dolly in slow small 35mm ease_out' -> camera block fields, or None."""
    t = _norm(text)
    move = None
    for cand in sorted(set(CAMERA_PRESETS) | set(ALIASES), key=len, reverse=True):
        if t == cand or t.startswith(cand + "_"):
            move = ALIASES.get(cand, cand)
            rest = t[len(cand):].strip("_")
            break
    if move is None or move not in CAMERA_PRESETS:
        return None
    block = {"move": move}
    words = [w for w in rest.split("_") if w]
    i = 0
    while i < len(words):
        w = words[i]
        two = "_".join(words[i:i + 2])
        nxt = words[i + 1] if i + 1 < len(words) else ""
        if two in SPEED_WORDS:
            block["speed"] = SPEED_WORDS[two]
            i += 2
            continue
        if two in EASES or two in REACTIONS:
            block["ease" if two in EASES else "react"] = two
            i += 2
            continue
        if w in SPEED_WORDS:
            block["speed"] = SPEED_WORDS[w]
        elif w in AMP_WORDS:
            block["intensity"] = AMP_WORDS[w]
        elif w in EASES:
            block["ease"] = w
        elif w in REACTIONS:
            block["react"] = w
        elif re.fullmatch(r"x\d+(\.\d+)?", w):
            block["speed"] = float(w[1:])
        elif re.fullmatch(r"\d+mm", w):
            block["lens"] = float(w[:-2])
        elif w in ("speed", "amp", "amplitude", "intensity", "lens") and re.fullmatch(r"\d+(\.\d+)?", nxt):
            key = {"amp": "intensity", "amplitude": "intensity"}.get(w, w)
            block[key] = float(nxt)
            i += 1
        i += 1
    return block


def _energy(text: str):
    m = re.search(r"\benergy\s*[:=]?\s*(\d*\.?\d+)", text, re.I)
    return float(m.group(1)) if m else None


def _empty():
    return {"version": 2, "scenes": [], "camera": [], "keys": [], "energy": [], "guidance": []}


def _add_beat(tl, start, fields, notes, line_no):
    mood, cam, prompt_parts = None, None, []
    for f in fields:
        f = f.strip()
        if not f:
            continue
        km = _KEY.match(f)
        if km:
            tl["keys"].append({"start": start, "label": km.group(1).strip()})
            continue
        if _norm(f) in MOODS:
            mood = _norm(f)
            continue
        e = _energy(f)
        if e is not None and re.fullmatch(r"\s*energy\s*[:=]?\s*\d*\.?\d+\s*", f, re.I):
            tl["energy"].append([start, max(0.0, min(1.0, e))])
            continue
        c = parse_camera(f) if cam is None else None
        if c is not None:
            cam = c
            if e is not None:
                tl["energy"].append([start, max(0.0, min(1.0, e))])
            continue
        prompt_parts.append(f)
    prompt = ", ".join(prompt_parts)
    if prompt or mood:
        tl["scenes"].append({"start": start, "mood": mood or "calm", "prompt": prompt})
    if cam is not None:
        m = MOODS.get(mood or "", {})
        for k in ("speed", "intensity", "ease"):
            if k not in cam and k in m:
                cam[k] = m[k]
        tl["camera"].append({"start": start, **cam})
    if not (prompt or mood or cam) and not any(k["start"] == start for k in tl["keys"]):
        notes.append(f"line {line_no}: nothing recognised")


def _from_csv(text, fps):
    rows = list(csv.DictReader(io.StringIO(text)))
    tl = _empty()
    notes = []
    for i, r in enumerate(rows, 2):
        r = {(k or "").strip().lower(): (v or "").strip() for k, v in r.items()}
        start = parse_time(r.get("time") or r.get("start") or r.get("frame") or "", fps)
        if start is None:
            notes.append(f"row {i}: no time")
            continue
        fields = [r.get("mood", ""), r.get("camera", "") or r.get("move", ""), r.get("prompt", "")]
        if r.get("key") or r.get("label"):
            fields.append("key: " + (r.get("key") or r.get("label")))
        if r.get("energy"):
            fields.append("energy " + r["energy"])
        _add_beat(tl, start, fields, notes, i)
    return tl, notes


def script_to_timeline(text: str, fps: float = 24.0, frames: int | None = None):
    """Shot script (text / CSV / JSON) -> (timeline dict, notes)."""
    from .direction import parse_timeline

    src = str(text or "").strip()
    if not src:
        return _empty(), ["empty script"]
    if src[0] in "[{":
        try:
            return parse_timeline(json.loads(src)), []
        except json.JSONDecodeError:
            pass
    src = src.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    first = src.splitlines()[0].lower()
    if "," in first and any(h in first for h in ("time", "start", "frame")) and "|" not in first:
        tl, notes = _from_csv(src, fps)
    else:
        tl, notes, untimed = _empty(), [], []
        for no, raw in enumerate(src.splitlines(), 1):
            line = raw.split("#", 1)[0].strip().lstrip("-*•").strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("|")]
            start = parse_time(parts[0], fps)
            if start is None and len(parts) == 1:          # '@12s the prompt' / '4s: prompt'
                m = re.match(r"^(@?\s*[\d:.]+\s*(?:s|sec|f)?|f\d+)\s*[:\-–]?\s+(.+)$", line, re.I)
                if m and parse_time(m.group(1), fps) is not None:
                    start, parts = parse_time(m.group(1), fps), [m.group(1)] + [m.group(2)]
            if start is None:
                untimed.append((no, parts))
                continue
            _add_beat(tl, start, parts[1:], notes, no)
        if untimed:
            n = int(frames or 0) or max(1, round(len(untimed) * 4 * fps))
            if tl["scenes"] or tl["camera"]:
                notes.append(f"{len(untimed)} line(s) without a time were ignored")
            else:
                step = n / len(untimed)
                for k, (no, parts) in enumerate(untimed):
                    _add_beat(tl, round(k * step), parts, notes, no)
                notes.append(f"no times: {len(untimed)} beats spread evenly over the clip")
    if frames:
        late = [s for s in tl["scenes"] + tl["camera"] + tl["keys"] if s["start"] >= frames]
        if late:
            notes.append(f"{len(late)} beat(s) start after the clip ends ({frames} frames)")
    return parse_timeline(tl), notes


def merge_timelines(drawn: dict, external: dict, mode: str) -> dict:
    """How an external timeline meets the drawn one."""
    if mode.startswith("replace"):
        out = external
    elif mode.startswith("text"):            # scenes, keys (+ energy if given) from outside, camera drawn
        out = {**drawn, "scenes": external["scenes"], "keys": external["keys"] or drawn["keys"],
               "energy": external["energy"] or drawn["energy"]}
    elif mode.startswith("camera"):
        out = {**drawn, "camera": external["camera"]}
    else:                                    # add: both, external wins on the same frame
        out = dict(drawn)
        for key in ("scenes", "camera", "keys"):
            ext_frames = {b["start"] for b in external[key]}
            out[key] = sorted([b for b in drawn[key] if b["start"] not in ext_frames] + external[key],
                              key=lambda b: b["start"])
        ext_e = {p[0] for p in external["energy"]}
        out["energy"] = sorted([p for p in drawn["energy"] if p[0] not in ext_e] + external["energy"])
    return {**out, "version": 2}


EXTERNAL_MODES = ("replace", "text only (keep drawn camera)", "camera only (keep drawn text)",
                  "add to drawn")

LLM_GUIDE = (
    "Write a shot script for a {seconds:.1f}-second shot, one beat per line:\n"
    "TIME | MOOD | CAMERA | PROMPT\n"
    "TIME: seconds like 0s, 2.5s. MOOD: calm, build, tense, climax, resolve or dream.\n"
    "CAMERA: one of {moves}, optionally followed by slow/fast, small/large, a lens like 35mm.\n"
    "PROMPT: what is on screen, in one visual sentence.\n"
    "Add lines like `6s | key: the light breaks` for moments that must land on a beat.\n"
    "Answer with the lines only."
)
