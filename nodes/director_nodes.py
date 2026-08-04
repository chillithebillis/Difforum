"""
Difforum · Film Director
========================
Visual timeline orchestrator: build a shot list on a drag-and-drop timeline
and get every Difforum schedule generated automatically.

Outputs plug straight into the existing nodes:
    camera_shots      -> Difforum · Camera Shots (director)  [shots]
    prompt_schedule   -> Difforum · Prompt Schedule (travel)  [prompts]
    strength_schedule -> Difforum · Schedule                  [schedule]
"""

import json

CATEGORY = "Difforum"

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

# Camera presets accepted by core.camera_presets.CAMERA_PRESETS
CAMERA_PRESETS = (
    "still", "zoom_in", "zoom_out", "dolly_in", "dolly_out",
    "pan_left", "pan_right", "pan_up", "pan_down",
    "orbit_left", "orbit_right", "roll_cw", "roll_ccw",
    "spiral", "sway", "dolly_zoom", "rise", "shake",
)

# mood -> camera rate, amplitude, denoise, and the lens the mood tends to want.
# A calm establishing beat reads wider; a tense one compresses onto the subject.
MOOD_PRESETS = {
    "calm":    {"strength": 0.44, "speed": 0.6, "intensity": 0.5,  "lens": 46.0, "ease": "ease_in_out"},
    "build":   {"strength": 0.52, "speed": 1.0, "intensity": 0.8,  "lens": 40.0, "ease": "ease_in_out"},
    "tense":   {"strength": 0.57, "speed": 1.2, "intensity": 1.0,  "lens": 32.0, "ease": "ease_in"},
    "climax":  {"strength": 0.63, "speed": 1.6, "intensity": 1.3,  "lens": 62.0, "ease": "ease_in"},
    "resolve": {"strength": 0.44, "speed": 0.5, "intensity": 0.45, "lens": 44.0, "ease": "ease_out"},
    "dream":   {"strength": 0.48, "speed": 0.4, "intensity": 0.6,  "lens": 52.0, "ease": "ease_in_out"},
}

# style -> feedback sampler / grade recommendations (reported in grade_info)
STYLE_GRADE = {
    "cinematic":   {"coherence": 0.85, "mode": "lab", "sharpen": 0.25, "noise": 0.025},
    "documentary": {"coherence": 0.90, "mode": "lab", "sharpen": 0.15, "noise": 0.015},
    "music_video": {"coherence": 0.75, "mode": "lab", "sharpen": 0.35, "noise": 0.035},
    "psychedelic": {"coherence": 0.60, "mode": "rgb", "sharpen": 0.20, "noise": 0.045},
}

DEFAULT_TIMELINE = json.dumps([
    {"frame_start": 0,   "mood": "calm",    "camera": "dolly_in",
     "prompt": "ancient forest, golden hour, wide establishing shot, epic scale"},
    {"frame_start": 60,  "mood": "build",   "camera": "orbit_right",
     "prompt": "roots and moss, glowing details, warm rim light, shallow focus"},
    {"frame_start": 120, "mood": "climax",  "camera": "spiral",
     "prompt": "explosion of light, cosmic transformation, high contrast"},
    {"frame_start": 180, "mood": "resolve", "camera": "zoom_out",
     "prompt": "embers becoming stars, calm, vast cosmic scale"},
])


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class DifforumFilmDirector:
    """Direct the whole clip on one visual timeline: scenes carry a mood, a
    camera preset and a prompt; the node emits the matching Difforum
    schedules."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "params": ("DIFFORUM_PARAMS",),
                # Custom DOM widget (js/director.js). Falls back to a plain
                # multiline JSON box if the JS did not load.
                "timeline": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_TIMELINE,
                }),
                "style_preset": (list(STYLE_GRADE), {"default": "cinematic"}),
                "strength_bias": ("FLOAT", {
                    "default": 0.0, "min": -0.25, "max": 0.25, "step": 0.01,
                }),
                "camera_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05,
                }),
                # 0 reproduces the timeline exactly. Above 0 the mood presets are
                # nudged by a hash of (seed, scene index), so the same seed always
                # gives the same variant and coming back to 0 restores the original.
                "variation": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "variation_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("camera_keys", "prompt_schedule", "strength_schedule",
                    "camera_shots", "info")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Visual film director: drag scenes on a timeline, pick a mood and a "
        "camera preset per scene, and get camera / prompt / strength schedules."
    )

    # ------------------------------------------------------------------
    @staticmethod
    def _jitter(seed: int, index: int, channel: str) -> float:
        """Deterministic value in [-1, 1] for one scene/channel."""
        import hashlib
        h = hashlib.sha1(f"{seed}:{index}:{channel}".encode()).digest()
        return (int.from_bytes(h[:4], "big") / 0xFFFFFFFF) * 2.0 - 1.0

    def run(self, params, timeline, style_preset, strength_bias, camera_scale,
            variation=0.0, variation_seed=0):
        max_frames = int(params.get("max_frames", 240))
        fps = float(params.get("fps", 24))

        scenes = self._parse(timeline)
        if not scenes:
            return (
                "0: still 1.0 1.0 40 linear",
                "0: empty timeline",
                "0:(0.5)",
                "0: still 1.0 1.0",
                "[Film Director] timeline is empty - add at least one scene.",
            )

        grade = STYLE_GRADE.get(style_preset, STYLE_GRADE["cinematic"])
        var = max(0.0, min(1.0, float(variation)))

        key_lines, shot_lines, prompt_lines, strength_parts, summary = [], [], [], [], []

        for idx, scene in enumerate(scenes):
            frame = max(0, int(scene.get("frame_start", 0)))
            if frame >= max_frames:
                continue

            mood = scene.get("mood", "calm")
            preset = MOOD_PRESETS.get(mood, MOOD_PRESETS["calm"])

            camera = scene.get("camera", "still")
            if camera not in CAMERA_PRESETS:
                camera = "still"

            # Variation rides on top of the mood, never replaces it: at 0 the
            # multipliers are exactly 1 and the timeline comes out as written.
            j_speed = 1.0 + var * 0.35 * self._jitter(variation_seed, idx, "speed")
            j_amp = 1.0 + var * 0.35 * self._jitter(variation_seed, idx, "intensity")
            j_str = var * 0.06 * self._jitter(variation_seed, idx, "strength")
            j_lens = var * 12.0 * self._jitter(variation_seed, idx, "lens")

            speed = round(preset["speed"] * camera_scale * j_speed, 3)
            intensity = round(preset["intensity"] * camera_scale * j_amp, 3)
            strength = round(
                min(0.95, max(0.05, preset["strength"] + strength_bias + j_str)), 3
            )
            base_lens = scene.get("lens") or preset["lens"]
            lens = round(min(140.0, max(12.0, base_lens + j_lens)), 1)
            ease = scene.get("ease") or preset["ease"]

            end = (
                int(scenes[idx + 1]["frame_start"])
                if idx + 1 < len(scenes)
                else max_frames
            )
            end = min(end, max_frames)

            # Camera Keys: "frame: move speed intensity lens ease" (blended)
            key_lines.append(
                f"{frame}: {camera} {speed:g} {intensity:g} {lens:g} {ease}"
            )
            # Camera Shots: same move as a hard cut, for the older node
            shot_lines.append(f"{frame}: {camera} {speed:g} {intensity:g}")

            # Prompt Schedule format: one "frame: prompt" per line
            prompt = " ".join(str(scene.get("prompt", "")).split())
            if prompt:
                prompt_lines.append(f"{frame}: {prompt}")

            # Deforum keyframe string, comma separated
            strength_parts.append(f"{frame}:({strength})")

            dur = max(0, end - frame)
            preview = prompt[:42] + ("..." if len(prompt) > 42 else "")
            summary.append(
                f"  {frame:>4}-{max(frame, end - 1):<4}  {dur / fps:5.1f}s  "
                f"{mood:<8} {camera:<13} sp{speed:<6g} in{intensity:<6g} "
                f"{lens:>4g}d  str{strength:<6g} {ease:<12} {preview or '-'}"
            )

        # both camera formats need a key at frame 0
        if not key_lines or not key_lines[0].startswith("0:"):
            key_lines.insert(0, "0: still 1 1 40 linear")
        if not shot_lines or not shot_lines[0].startswith("0:"):
            shot_lines.insert(0, "0: still 1 1")
        if not prompt_lines:
            prompt_lines.append("0: untitled scene")
        if not strength_parts:
            strength_parts.append("0:(0.5)")

        var_line = (
            f"  variation={var:g} (seed {variation_seed}) - "
            + ("timeline exactly as written" if var == 0.0
               else "moods nudged; return to 0 to restore")
        )

        info = "\n".join([
            "[Difforum - Film Director]",
            f"  style={style_preset}  frames={max_frames}  fps={fps:g}  "
            f"({max_frames / fps:.1f}s)",
            f"  scenes={len(key_lines)}  camera_scale={camera_scale:g}  "
            f"strength_bias={strength_bias:+g}",
            var_line,
            "",
            "  Feedback Sampler recommendation:",
            f"    color_mode={grade['mode']}  color_coherence={grade['coherence']}",
            f"    sharpen={grade['sharpen']}  noise={grade['noise']}  "
            f"seed_mode=fixed  border=reflection",
            "",
            "  wire camera_keys -> Camera Keys (blend + lens) for crane moves,",
            "  or camera_shots -> Camera Shots (director) for hard cuts.",
            "",
            "  frames        dur  mood     camera        rate   amp    lens  denoise  ease         prompt",
            *summary,
        ])

        return (
            "\n".join(key_lines),
            "\n".join(prompt_lines),
            ", ".join(strength_parts),
            "\n".join(shot_lines),
            info,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _parse(timeline):
        """Accept the JSON emitted by the timeline widget; tolerate a hand
        written `frame | mood | camera | prompt` table as a fallback."""
        if isinstance(timeline, (list, tuple)):
            raw = list(timeline)
        else:
            text = str(timeline or "").strip()
            if not text:
                return []
            try:
                raw = json.loads(text)
            except json.JSONDecodeError:
                raw = DifforumFilmDirector._parse_table(text)
            if not isinstance(raw, list):
                return []

        scenes = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                frame = int(item.get("frame_start", 0))
            except (TypeError, ValueError):
                continue
            scenes.append({
                "frame_start": frame,
                "mood": str(item.get("mood", "calm")).lower(),
                "camera": str(item.get("camera", "still")).lower(),
                "prompt": str(item.get("prompt", "")),
                # optional per-scene overrides from the timeline widget;
                # absent or 0 means "let the mood preset decide"
                "ease": (str(item["ease"]).lower() if item.get("ease") else None),
                "lens": (float(item["lens"]) if item.get("lens") else None),
            })
        scenes.sort(key=lambda s: s["frame_start"])
        return scenes

    @staticmethod
    def _parse_table(text):
        """`0 | calm | dolly_in | forest at dawn` per line."""
        out = []
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 2:
                continue
            try:
                frame = int(parts[0])
            except ValueError:
                continue
            out.append({
                "frame_start": frame,
                "mood": parts[1] if len(parts) > 1 else "calm",
                "camera": parts[2] if len(parts) > 2 else "still",
                "prompt": parts[3] if len(parts) > 3 else "",
            })
        return out


NODE_CLASS_MAPPINGS = {
    "DifforumFilmDirector": DifforumFilmDirector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DifforumFilmDirector": "Difforum · Film Director",
}
