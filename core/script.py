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

- time: `4s`, `4.5s`, `00:04`, `1:02.5`, `f96` / `96` (frames), or a range
  `0-35`, `0s-1.5s` (a range ends where the next one starts)
- the fields after the time are recognised by content: a mood name
  (calm, build, tense, climax, resolve, dream), a camera move (`dolly_in`,
  `dolly in`, `push in`...) with optional modifiers (`slow`, `fast`, `small`,
  `large`, `x1.3` speed, `amp 0.8`, `35mm` / `fov 46`, an easing, an audio
  reaction, `energy 0.6`), `key: label`, `sound: what is heard`, and the rest
  is the prompt.
- a CSV with a header row (`time,mood,camera,prompt,key,energy`) and a JSON
  timeline (what the Director saves) are accepted as they are.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import unicodedata

from .camera_presets import CAMERA_PRESETS
from .direction import MOODS, REACTIONS

EASES = ("linear", "ease_in", "ease_out", "ease_in_out", "step")
ALIASES = {
    "push_in": "dolly_in", "push": "dolly_in", "pull_out": "dolly_out", "pull_back": "dolly_out",
    "truck_left": "pan_left", "truck_right": "pan_right", "pedestal_up": "pan_up", "pedestal_down": "pan_down",
    "track_left": "pan_left", "track_right": "pan_right", "whip_pan_left": "pan_left",
    "whip_pan_right": "pan_right", "slide_left": "pan_left", "slide_right": "pan_right",
    "arc_left": "orbit_left", "arc_right": "orbit_right", "static": "still", "hold": "still", "locked": "still",
    "locked_off": "still", "crane": "crane_up", "boom_up": "crane_up", "vertigo": "dolly_zoom",
    "hand_held": "handheld", "dutch_angle": "roll_cw", "dutch_tilt": "roll_cw",
    # pt / es
    "aproximar": "dolly_in", "aproxima": "dolly_in", "afastar": "dolly_out", "afasta": "dolly_out",
    "pan_esquerda": "pan_left", "pan_direita": "pan_right", "pan_cima": "pan_up", "pan_baixo": "pan_down",
    "pan_izquierda": "pan_left", "pan_derecha": "pan_right", "pan_arriba": "pan_up", "pan_abajo": "pan_down",
    "orbita_esquerda": "orbit_left", "orbita_direita": "orbit_right", "orbitar_esquerda": "orbit_left",
    "orbitar_direita": "orbit_right", "girar": "roll_cw", "giro": "roll_cw", "subir": "crane_up",
    "sobe": "crane_up", "grua": "crane_up", "parado": "still", "parada": "still", "fixo": "still",
    "fixa": "still", "estatico": "still", "estatica": "still", "espiral": "spiral", "camera_na_mao": "handheld",
    "tremer": "shake", "respirar": "breathe", "deriva": "drift", "na_mao": "handheld",
    "zoom_dentro": "zoom_in", "zoom_fora": "zoom_out", "zoom_adentro": "zoom_in", "zoom_afuera": "zoom_out",
    "girar_direita": "roll_cw", "girar_esquerda": "roll_ccw", "girar_derecha": "roll_cw",
    "girar_izquierda": "roll_ccw", "inclinar_cima": "tilt_up", "inclinar_baixo": "tilt_down",
    "descer": "pan_down", "desce": "pan_down", "livre": "free", "libre": "free",
}
WHIP = ("whip_pan_left", "whip_pan_right")
SPEED_WORDS = {"very_slow": 0.4, "slow": 0.6, "slowly": 0.6, "medium": 1.0, "fast": 1.5, "quick": 1.5,
               "quickly": 1.5, "very_fast": 2.0, "rapid": 2.0,
               "muito_lento": 0.4, "lento": 0.6, "lenta": 0.6, "devagar": 0.6, "lentamente": 0.6,
               "rapido": 1.5, "rapida": 1.5, "muito_rapido": 2.0}
AMP_WORDS = {"subtle": 0.4, "small": 0.5, "slight": 0.5, "gentle": 0.6, "big": 1.4, "large": 1.4, "huge": 1.8,
             "strong": 1.4, "sutil": 0.4, "pequeno": 0.5, "pequena": 0.5, "leve": 0.5, "suave": 0.6,
             "grande": 1.4, "forte": 1.4, "enorme": 1.8}
MOOD_ALIASES = {"calmo": "calm", "calma": "calm", "tranquilo": "calm", "tranquila": "calm", "tenso": "tense",
                "tensa": "tense", "tension": "tense", "climax": "climax", "auge": "climax", "pico": "climax",
                "sonho": "dream", "onirico": "dream", "dreamy": "dream", "resolucao": "resolve",
                "desfecho": "resolve", "resolution": "resolve", "crescendo": "build", "construcao": "build",
                "building": "build", "subida": "build"}
FILLER = {"and", "with", "the", "a", "at", "very", "camera", "move", "shot", "e", "com", "de", "da", "do",
          "a", "o", "para", "y", "con", "la", "el"}
CAMERA_HINTS = ("pan", "zoom", "dolly", "tilt", "orbit", "truck", "crane", "whip", "dutch", "tracking",
                "handheld", "steadicam", "rack", "pedestal", "roll", "arc")
LABELS = {"mood": "mood", "humor": "mood", "camera": "camera", "cam": "camera", "movement": "camera",
          "movimento": "camera", "prompt": "prompt", "scene": "prompt", "cena": "prompt",
          "description": "prompt", "descricao": "prompt", "visual": "prompt", "time": "time", "tempo": "time",
          "frames": "time", "frame": "time"}

_KEY = re.compile(r"^(?:key|mark|beat|marca)\s*[:\-]\s*(.*)$", re.I)
_SOUND = re.compile(r"^(?:sound|audio|sfx|som)\s*[:\-]\s*(.*)$", re.I)
_RANGE = re.compile(r"^(.+?)\s*(?:-|–|—|→|->|\bto\b|\bao?\b|\baté\b)\s*(.+)$", re.I)
_LABEL = re.compile(r"^([A-Za-zÀ-ÿ]+)\s*[:=]\s*(.+)$")
_CURVE = re.compile(r"^\s*(energy|energia|guidance|cfg)\s*[:=]?\s*(\d*\.?\d+)\s*$", re.I)


def _plain(text: str) -> str:
    """Lower case, no accents."""
    return "".join(c for c in unicodedata.normalize("NFKD", str(text).lower()) if not unicodedata.combining(c))


def parse_start(token: str, fps: float) -> int | None:
    """A time, or the start of a range: `0-35`, `0s-1.5s`, `f0 to f35`."""
    t = parse_time(token, fps)
    if t is not None:
        return t
    m = _RANGE.match(token.strip())
    if m and parse_time(m.group(2), fps) is not None:
        return parse_time(m.group(1), fps)
    return None


def parse_time(token: str, fps: float) -> int | None:
    t = token.strip().lower().replace(" ", "")
    if not t:
        return None
    if t.startswith("@"):
        t = t[1:]
    t = re.sub(r"^(?:do|de|from)?(?:frames?|quadros?)(?=\d)", "f", t)      # 'frame 35', 'do frame 35'
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
    return re.sub(r"[\s\-]+", "_", _plain(word).strip())


def parse_camera(text: str, strict: bool = True):
    """'dolly in slow small 35mm ease_out' -> camera block fields, or None.

    `strict` (unlabelled fields): every word after the move has to be a known
    modifier, so a prompt like 'rise of the machines' is not read as a camera.
    """
    text = re.sub(r"(^|[\s:=])-(?=\.?\d)", r"\1~", str(text))          # keep minus signs through _norm
    t = _norm(re.sub(r"[,;()]+", " ", text))
    t = "_".join(w for w in t.split("_") if w and w not in FILLER)       # 'zoom para dentro' -> zoom_dentro
    move = None
    names = sorted(set(CAMERA_PRESETS) | set(ALIASES), key=len, reverse=True)
    for t in (t, re.sub(r"^([a-z]+?)(?:es|s)_", r"\1_", t)):                 # 'pushes in' -> push_in
        for cand in names:
            if t == cand or t.startswith(cand + "_"):
                move, alias = ALIASES.get(cand, cand), cand
                rest = t[len(cand):].strip("_")
                break
        if move:
            break
    if move is None or move not in CAMERA_PRESETS:
        return None
    block = {"move": move}
    if alias in WHIP:
        block["speed"] = 2.0
    if move == "free":
        block.update(dx=0.0, dy=0.0, zoom=1.0, roll=0.0)
    words = [w for w in rest.split("_") if w]
    unknown = []
    i = 0
    while i < len(words):
        w = words[i]
        two = "_".join(words[i:i + 2])
        nxt = words[i + 1] if i + 1 < len(words) else ""
        num = re.fullmatch(r"-?\d*\.?\d+", nxt.replace("~", "-"))
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
        elif re.fullmatch(r"\d+mm", w):            # full-frame focal length -> horizontal field of view
            block["lens"] = round(math.degrees(2 * math.atan(36.0 / (2 * float(w[:-2])))), 1)
        elif re.fullmatch(r"\d+deg", w):
            block["lens"] = float(w[:-3])
        elif w in ("speed", "amp", "amplitude", "intensity", "lens", "fov", "dx", "dy", "zoom", "roll") and num:
            key = {"amp": "intensity", "amplitude": "intensity", "fov": "lens"}.get(w, w)
            block[key] = float(num.group(0))
            i += 1
        elif w not in FILLER:
            unknown.append(w)
        i += 1
    if unknown and strict:
        return None
    if unknown:
        block["_unknown"] = unknown
    return block


def _curve(text: str):
    m = _CURVE.match(_plain(text))
    if not m:
        return None
    name = "energy" if m.group(1) in ("energy", "energia") else "guidance"
    return name, float(m.group(2))


def _empty():
    return {"version": 2, "scenes": [], "camera": [], "keys": [], "energy": [], "guidance": []}


def _looks_like_camera(text: str) -> bool:
    words = _norm(text).split("_")
    return len(words) <= 5 and any(w.startswith(h) for w in words for h in CAMERA_HINTS)


def _add_beat(tl, start, fields, notes, line_no):
    mood, cam, sound, prompt_parts = None, None, "", []
    noted, marks = len(notes), len(tl["keys"]) + len(tl["energy"]) + len(tl["guidance"])
    plain = 0                                   # unlabelled columns seen so far (mood, camera, prompt)
    columns = sum(1 for f in fields if f.strip() and not (_SOUND.match(f.strip()) or _KEY.match(f.strip())
                                                         or _curve(f)))
    for f in fields:
        f = f.strip().strip("*_`").strip()
        if not f:
            continue
        sm = _SOUND.match(f)
        if sm:
            sound = sm.group(1).strip()
            continue
        km = _KEY.match(f)
        if km:
            tl["keys"].append({"start": start, "label": km.group(1).strip()})
            continue
        cv = _curve(f)
        if cv:
            v = max(0.0, min(1.0, cv[1])) if cv[0] == "energy" else max(0.0, min(30.0, cv[1]))
            tl[cv[0]].append([start, v])
            continue
        label = None
        lm = _LABEL.match(f)
        if lm and _plain(lm.group(1)) in LABELS:
            label, f = LABELS[_plain(lm.group(1))], lm.group(2).strip()
            if label == "time":
                continue
        word = _norm(f)
        word = MOOD_ALIASES.get(word, word)
        plain += label is None
        if label in (None, "mood") and word in MOODS:
            mood = word
            continue
        if label == "mood":
            notes.append(f"line {line_no}: mood {f!r} not recognised (calm, build, tense, climax, resolve, dream)")
            continue
        if label != "prompt" and cam is None:
            c = parse_camera(f, strict=label != "camera")
            if c is not None:
                extra = c.pop("_unknown", None)
                if extra:
                    notes.append(f"line {line_no}: camera words not understood: {' '.join(extra)}")
                cam = c
                continue
            # 'time | mood | CAMERA | prompt': a short second column is a camera the parser does not know
            column = label is None and mood and plain == 2 and columns >= 3 and len(f.split()) <= 4
            if label == "camera" or column or _looks_like_camera(f):
                notes.append(f"line {line_no}: camera {f!r} is not a known move; the camera was left as it is")
                continue
        prompt_parts.append(f)
    prompt = ", ".join(prompt_parts)
    if prompt or mood or sound:
        tl["scenes"].append({"start": start, "mood": mood or "calm", "prompt": prompt,
                             **({"sound": sound} if sound else {})})
    if cam is not None:
        m = MOODS.get(mood or "", {})
        for k in ("speed", "intensity", "ease"):
            if k not in cam and k in m and cam["move"] != "free":
                cam[k] = m[k]
        tl["camera"].append({"start": start, **cam})
    if not (prompt or mood or cam or sound) and len(notes) == noted \
            and len(tl["keys"]) + len(tl["energy"]) + len(tl["guidance"]) == marks:
        notes.append(f"line {line_no}: nothing recognised")


def _from_csv(text, fps):
    rows = list(csv.DictReader(io.StringIO(text)))
    tl = _empty()
    notes = []
    for i, r in enumerate(rows, 2):
        r = {(k or "").strip().lower(): (v or "").strip() for k, v in r.items()}
        start = parse_start(r.get("time") or r.get("start") or r.get("frame") or r.get("frames") or "", fps)
        if start is None:
            notes.append(f"row {i}: no time")
            continue
        fields = [r.get("mood", ""), r.get("camera", "") or r.get("move", ""), r.get("prompt", "")]
        if r.get("key") or r.get("label"):
            fields.append("key: " + (r.get("key") or r.get("label")))
        if r.get("energy"):
            fields.append("energy " + r["energy"])
        if r.get("sound"):
            fields.append("sound: " + r["sound"])
        _add_beat(tl, start, fields, notes, i)
    return tl, notes


_HEADER_CELLS = {"time", "frames", "frame", "start", "mood", "camera", "prompt", "sound", "key", "energy",
                 "scene", "tempo", "cena", "som", "humor"}


def _clean_line(raw: str) -> str:
    """One beat out of however an LLM formatted it: bullets, numbering, bold, table pipes."""
    line = raw.split("#", 1)[0].strip()
    line = re.sub(r"^\s*(?:[-*•>]+|\d+[.)])\s+", "", line)          # '- ', '1. ', '2) '
    line = line.replace("**", "").replace("__", "").replace("`", "")
    line = re.sub(r"^(?:shot|plano|cena|scene|beat)\s*\d+\s*[:.\-–]\s*", "", line, flags=re.I)
    return line.strip().strip("|").strip()


def _unwrap(src: str):
    """Fenced block or embedded JSON out of an LLM answer."""
    m = re.search(r"```[a-zA-Z]*\s*\n(.*?)```", src, re.S)
    if m:
        src = m.group(1).strip()
    if src[:1] in "[{":
        try:
            return json.loads(src), src
        except json.JSONDecodeError:
            pass
    else:
        j = re.search(r"\{.*\"(?:scenes|camera)\".*\}", src, re.S)
        if j:
            try:
                return json.loads(j.group(0)), src
            except json.JSONDecodeError:
                pass
    return None, src


def script_to_timeline(text: str, fps: float = 24.0, frames: int | None = None):
    """Shot script (text / CSV / JSON, as typed or as an LLM answers) -> (timeline dict, notes)."""
    from .direction import parse_timeline

    src = str(text or "").strip()
    if not src:
        return _empty(), ["empty script"]
    data, src = _unwrap(src)
    if data is not None:
        return parse_timeline(data), []
    first = src.splitlines()[0].lower()
    if "," in first and any(h in first for h in ("time", "start", "frame")) and "|" not in first:
        tl, notes = _from_csv(src, fps)
    else:
        tl, notes, untimed, prose = _empty(), [], [], 0
        for no, raw in enumerate(src.splitlines(), 1):
            line = _clean_line(raw)
            if not line or re.fullmatch(r"[\s|:\-]+", line):                 # blank, table rule
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) > 1 and all(_plain(p).strip("*: ") in _HEADER_CELLS for p in parts if p):
                continue                                                      # header row
            start = parse_start(parts[0], fps)
            if start is None and len(parts) == 1:          # '@12s the prompt' / '4s: prompt'
                m = re.match(r"^(@?\s*[\d:.]+\s*(?:s|sec|f)?|f\d+)\s*[:\-–]?\s+(.+)$", line, re.I)
                if m and parse_time(m.group(1), fps) is not None:
                    start, parts = parse_time(m.group(1), fps), [m.group(1)] + [m.group(2)]
            if start is None:
                if len(parts) == 1 and (line.endswith((":", "!", "?")) or len(line.split()) > 25):
                    prose += 1                                                # the LLM talking, not a beat
                    continue
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


def _camera_words(c: dict) -> str:
    out = [c.get("move", "still")]
    if c.get("move") == "free":
        out += [f"dx {float(c.get('dx', 0)):g}", f"dy {float(c.get('dy', 0)):g}",
                f"zoom {float(c.get('zoom', 1)):g}", f"roll {float(c.get('roll', 0)):g}"]
        if c.get("ease", "ease_in_out") != "ease_in_out":
            out.append(c["ease"])
        return " ".join(out)
    if abs(float(c.get("speed", 1.0)) - 1.0) > 1e-6:
        out.append(f"x{float(c['speed']):g}")
    if abs(float(c.get("intensity", 1.0)) - 1.0) > 1e-6:
        out.append(f"amp {float(c['intensity']):g}")
    if float(c.get("lens", 0.0) or 0.0) > 0:
        out.append(f"fov {float(c['lens']):g}")
    if c.get("ease", "ease_in_out") != "ease_in_out":
        out.append(c["ease"])
    if c.get("react", "none") != "none":
        out.append(c["react"])
    return " ".join(out)


def _prompt_words(prompt: str) -> str:
    """A prompt the parser reads back as a prompt: no column separator, labelled when ambiguous."""
    text = prompt.replace("|", "/")
    word = MOOD_ALIASES.get(_norm(text), _norm(text))
    if word in MOODS or parse_camera(text) is not None or _KEY.match(text) or _SOUND.match(text) \
            or _curve(text) or _looks_like_camera(text) or _LABEL.match(text):
        return "prompt: " + text
    return text


def timeline_to_script(timeline, frames: int | None = None) -> str:
    """The timeline as editable text, one line per frame range:
    `0-35 | calm | zoom_in x0.8 | prompt | sound: ...` (what script_to_timeline reads back)."""
    from .direction import parse_timeline

    tl = parse_timeline(timeline)
    starts = sorted({b["start"] for b in tl["scenes"] + tl["camera"]})
    end = int(frames) if frames else (max(starts) + 48 if starts else 48)
    scenes = {s["start"]: s for s in tl["scenes"]}
    cams = {c["start"]: c for c in tl["camera"]}
    lines = ["# FRAMES | MOOD | CAMERA | PROMPT | sound: ...   (ranges end where the next one starts)"]
    for i, f in enumerate(starts):
        nxt = starts[i + 1] if i + 1 < len(starts) else end
        parts = [f"{f}-{max(f, nxt - 1)}"]
        sc, cam = scenes.get(f), cams.get(f)
        if sc:
            parts.append(sc["mood"])
        if cam:
            parts.append(_camera_words(cam))
        if sc:
            if sc["prompt"]:
                parts.append(_prompt_words(sc["prompt"]))
            if sc.get("sound"):
                parts.append("sound: " + sc["sound"])
        lines.append(" | ".join(parts))
    for k in tl["keys"]:
        lines.append(f"{k['start']} | key: {k['label']}")
    for f, v in tl["energy"]:
        lines.append(f"{f} | energy {v:g}")
    for f, v in tl["guidance"]:
        lines.append(f"{f} | guidance {v:g}")
    return "\n".join(lines)


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
    "PROMPT: what is on screen, in one visual sentence. Optionally add `| sound: what is heard`.\n"
    "Add lines like `6s | key: the light breaks` for moments that must land on a beat.\n"
    "Answer with the lines only."
)
