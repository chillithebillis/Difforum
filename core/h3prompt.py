"""
The Director timeline written as a MiniMax H3 prompt in the model's native
structure (integrated_multimodal_description / overall_soundscape /
non_diegetic_music, or the six full-reference sections).

Scenes become what is on screen, camera blocks become H3 camera vocabulary
(push in, truck, arc, pedestal...) with amplitude and speed, keys become the
events that happen at those moments, and the look opens [Shot 1] as the
style. One continuous take by default; `cuts=True` turns every scene change
into a new [Shot N] with its cut time.
"""

from __future__ import annotations

H3_MOVE = {
    "still": "holds a static shot",
    "zoom_in": "zooms in", "zoom_out": "zooms out",
    "dolly_in": "pushes in", "dolly_out": "pulls out",
    "pan_left": "trucks left", "pan_right": "trucks right",
    "pan_up": "pedestals up", "pan_down": "pedestals down",
    "tilt_up": "tilts up", "tilt_down": "tilts down",
    "orbit_left": "arcs left around the subject", "orbit_right": "arcs right around the subject",
    "roll_cw": "rolls clockwise", "roll_ccw": "rolls counterclockwise",
    "spiral": "pushes in while rolling clockwise",
    "vortex": "pulls out while rolling counterclockwise",
    "sway": "sways gently from side to side",
    "dolly_zoom": "pushes in while zooming out in a dolly zoom",
    "rise": "pedestals up", "crane_up": "pedestals up while tilting down to keep the subject framed",
    "shake": "shakes strongly", "handheld": "shakes slightly like a handheld camera",
    "drift": "drifts slowly", "breathe": "pushes in and pulls out gently",
}
_NO_MODIFIERS = {"still", "shake", "handheld", "sway", "breathe", "drift"}

MODES = ("T2VA (text only)", "I2VA (first frame)", "FL2VA (first + last frame)",
         "reference (ref2va / guides)")


def camera_clause(block: dict) -> str:
    move = block.get("move", "still")
    phrase = H3_MOVE.get(move, move.replace("_", " "))
    if move in _NO_MODIFIERS:
        return phrase
    amp, spd = float(block.get("intensity", 1.0)), float(block.get("speed", 1.0))
    mods = []
    if amp < 0.7:
        mods.append("with small amplitude")
    elif amp > 1.3:
        mods.append("with large amplitude")
    if spd < 0.75:
        mods.append("at slow speed")
    elif spd > 1.35:
        mods.append("at fast speed")
    return " ".join([phrase, *mods])


def _sentence(text: str) -> str:
    t = " ".join(str(text).split()).strip().rstrip(".,;")
    return (t[:1].upper() + t[1:] + ".") if t else ""


def _lower_first(text: str) -> str:
    t = " ".join(str(text).split()).strip().rstrip(".,;")
    return t[:1].lower() + t[1:] if t else ""


def _stamp(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:06.3f}"


def _active(items: list[dict], f: int) -> dict | None:
    cur = None
    for it in items:
        if it["start"] <= f:
            cur = it
        else:
            break
    return cur


def _beats(direction, start: int, end: int):
    """Everything that changes inside [start, end): (frame, kind, payload)."""
    beats = []
    for kind, items in (("scene", direction.scenes), ("camera", direction.camera_blocks)):
        items = sorted(items, key=lambda b: b["start"])
        first = _active(items, start)
        if first is not None:
            beats.append((start, kind, first))
        for it in items:
            if start < it["start"] < end:
                beats.append((it["start"], kind, it))
    for k in sorted(direction.keys, key=lambda k: k["start"]):
        if start <= k["start"] < end and k.get("label", "").strip():
            beats.append((k["start"], "key", k))
    order = {"scene": 0, "camera": 1, "key": 2}
    return sorted(beats, key=lambda b: (b[0], order[b[1]]))


def body(direction, style: str, lead: str = "", start: int = 0, end: int | None = None,
         cuts: bool = False, first_frame_ref: bool = False) -> tuple[str, int]:
    """[Shot 1] ... text and the number of shots."""
    fps = float(direction.fps)
    end = int(direction.frames if end is None else end)
    beats = _beats(direction, start, end)
    shots, cur, n_shots = [], [], 0
    if first_frame_ref:
        opening = "[Shot 1] The shot begins from <Picture 1>."
    else:
        opening = f"[Shot 1] {_sentence(style)}" + (f" {_sentence(lead)}" if lead.strip() else "")
    for i, (f, kind, item) in enumerate(beats):
        t = (f - start) / fps
        if kind == "scene":
            prompt = item.get("prompt", "").strip()
            if not cur:                       # opening of the video
                n_shots = 1
                cur.append(opening)
                if prompt:
                    cur.append(_sentence(prompt))
            elif cuts:
                shots.append(" ".join(cur))
                n_shots += 1
                cur = [f"[Shot {n_shots}] At {_stamp(t)}, the camera cuts to {_lower_first(prompt)}."]
            elif prompt:
                cur.append(f"The scene then transforms into {_lower_first(prompt)}.")
        elif kind == "camera":
            if not cur:
                n_shots = 1
                cur.append(opening)
            again = any(b[1] == "camera" and b[0] < f for b in beats[:i])
            cur.append(f"The camera {'then ' if again else ''}{camera_clause(item)}.")
        else:
            cur.append(_sentence(item["label"]))
    if cur:
        shots.append(" ".join(cur))
    if not shots:
        shots, n_shots = [opening + " The camera holds a static shot."], 1
    return " ".join(shots), max(1, n_shots)


def h3_prompt(direction, look_prompt: str, mode: str = MODES[0], lead: str = "", soundscape: str = "",
              music: str = "", start: int = 0, end: int | None = None, cuts: bool = False) -> str:
    fps = float(direction.fps)
    end = int(direction.frames if end is None else end)
    duration = max(0.0, (end - 1 - start) / fps)
    style = look_prompt or "Cinematic live-action footage."
    sound = soundscape.strip() or "Natural ambient sound of the environment, soft and continuous."
    score = music.strip() or "N/A"
    ref = mode.startswith("reference")
    text, n = body(direction, style, lead, start, end, cuts, first_frame_ref=ref)
    if ref:
        subj = "<Picture 1> is the first frame of [Shot 1]"
        subj += f", showing {_lower_first(lead)}." if lead.strip() else "."
        return "\n".join([
            "subject_definitions:", subj, "",
            "summary:",
            f"[keyframe completion] The target video is a {duration:.1f}-second "
            f"{'continuous shot' if n == 1 else f'{n}-shot sequence'} that begins from <Picture 1> and follows "
            "the planned camera path through the guide keyframes.", "",
            "retention_analysis:",
            "<Picture 1> ([Shot 1] first frame): fully_preserved - its composition, subjects, lighting and palette "
            "open the video.", "",
            "detailed_description:", style, text, "",
            "overall_soundscape:", sound, "",
            "non_diegetic_music:", score,
        ])
    lines = []
    if mode.startswith("I2VA"):
        lines += ["For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) "
                  "is fully referenced.", ""]
    elif mode.startswith("FL2VA"):
        lines += ["How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the "
                  f"0.00-second mark of the target video; Picture 2 (from Shot {n}) aligns with the "
                  f"{duration:.2f}-second mark of the target video.", ""]
    lines += [f"integrated_multimodal_description: {text}", "",
              f"overall_soundscape: {sound}", "",
              f"non_diegetic_music: {score}"]
    return "\n".join(lines)
