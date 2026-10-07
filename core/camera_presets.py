"""
Camera move presets for Difforum - intuitive camera control without typing
schedule strings. A preset + speed + intensity expands into the per-axis
schedule strings the camera engine already understands.

Per-frame values are increments (they accumulate through the feedback loop),
so a constant like translation_x = "0:(2)" means "pan a bit every frame".
"""

from __future__ import annotations

CAMERA_PRESETS = (
    "still", "zoom_in", "zoom_out", "dolly_in", "dolly_out",
    "pan_left", "pan_right", "pan_up", "pan_down",
    "orbit_left", "orbit_right", "roll_cw", "roll_ccw",
    "spiral", "sway", "dolly_zoom", "rise", "shake",
    "tilt_up", "tilt_down", "crane_up", "handheld", "drift", "breathe", "vortex",
    "free",
)

# UI metadata for the Director timeline: label, group, one-line description.
MOVE_INFO = {
    "still":      ("Still", "basic", "Locked-off frame; the diffusion still evolves."),
    "free":       ("Free pose", "basic", "Drag, scroll and shift-drag the preview: the block ends on that framing."),
    "zoom_in":    ("Zoom in", "basic", "Lens magnifies toward the centre."),
    "zoom_out":   ("Zoom out", "basic", "Lens pulls back from the centre."),
    "pan_left":   ("Pan left", "basic", "Frame slides left."),
    "pan_right":  ("Pan right", "basic", "Frame slides right."),
    "pan_up":     ("Pan up", "basic", "Frame slides up."),
    "pan_down":   ("Pan down", "basic", "Frame slides down."),
    "roll_cw":    ("Roll CW", "basic", "Rotates clockwise around the view axis."),
    "roll_ccw":   ("Roll CCW", "basic", "Rotates counter-clockwise."),
    "dolly_in":   ("Dolly in", "space", "Camera travels forward into the scene (parallax with depth)."),
    "dolly_out":  ("Dolly out", "space", "Camera travels backward."),
    "orbit_left": ("Orbit left", "space", "Camera circles the subject to the left."),
    "orbit_right": ("Orbit right", "space", "Camera circles the subject to the right."),
    "tilt_up":    ("Tilt up", "space", "Camera pivots upward."),
    "tilt_down":  ("Tilt down", "space", "Camera pivots downward."),
    "rise":       ("Rise", "space", "Camera lifts while easing forward."),
    "crane_up":   ("Crane up", "space", "Rises and tilts down to keep the subject."),
    "dolly_zoom": ("Dolly zoom", "space", "Vertigo: push in while the lens widens."),
    "spiral":     ("Spiral", "fx", "Roll plus zoom - the classic Deforum tunnel."),
    "vortex":     ("Vortex", "fx", "Roll while pulling back - an outward spiral."),
    "sway":       ("Sway", "fx", "Slow side-to-side yaw."),
    "breathe":    ("Breathe", "fx", "Zoom gently pulses in and out."),
    "drift":      ("Drift", "fx", "Lazy wandering float, no fixed direction."),
    "handheld":   ("Handheld", "fx", "Small organic operator shake."),
    "shake":      ("Shake", "fx", "Hard jitter, for impacts."),
}

_AXES = (
    "translation_x", "translation_y", "translation_z",
    "rotation_3d_x", "rotation_3d_y", "rotation_3d_z", "zoom",
)

# The 2D affine warp can only express in-plane translation, a roll about the
# view axis, and a scale. Everything else describes motion through space and
# needs a depth map to mean anything.
#
# This matters more than it looks: `build_camera(mode="2d")` zeroes the other
# axes outright, and the Feedback Sampler silently falls back to the 2D warp
# whenever `depth` is not connected - even with mode set to "3d". A shot list
# built from orbit/dolly moves will then render as a completely still frame,
# with nothing in the logs to say why.
AXES_2D = ("translation_x", "translation_y", "rotation_3d_z", "zoom")


def preset_axes(preset: str) -> set[str]:
    """Which axes a preset actually drives (at speed=intensity=1)."""
    exprs = preset_schedules(preset, 1.0, 1.0)
    return {
        ax for ax, e in exprs.items()
        if e not in ("0:(0)", "0:(1.0)")
    }


def needs_depth(preset: str) -> bool:
    """True if the preset does nothing without a depth map."""
    used = preset_axes(preset)
    return bool(used) and not (used & set(AXES_2D))


def flat_presets() -> tuple[str, ...]:
    """Presets that work with the plain 2D warp, no depth needed."""
    return tuple(p for p in CAMERA_PRESETS if not needs_depth(p))


def preset_schedules(preset: str, speed: float = 1.0, intensity: float = 1.0) -> dict[str, str]:
    """Return the 7 axis schedule strings for a preset at given speed/intensity."""
    if preset not in CAMERA_PRESETS:
        raise ValueError(f"unknown preset {preset!r}, pick from {CAMERA_PRESETS}")
    s, i = float(speed), float(intensity)
    out = {ax: "0:(0)" for ax in _AXES[:-1]}
    out["zoom"] = "0:(1.0)"

    def c(v):  # constant per-frame increment
        return f"0:({v:.4g})"

    def osc(amp, period):  # oscillator around 0
        p = max(2.0, period)
        return f"0:({amp:.4g}*sin(2*pi*t/{p:.4g}))"

    if preset in ("still", "free"):          # free: the Director writes the axes from the block's pose
        pass
    elif preset == "zoom_in":
        out["zoom"] = c(1.0 + 0.015 * i * s)
    elif preset == "zoom_out":
        out["zoom"] = c(1.0 - 0.012 * i * s)
    elif preset == "dolly_in":
        out["translation_z"] = c(-1.5 * i * s)
    elif preset == "dolly_out":
        out["translation_z"] = c(1.5 * i * s)
    elif preset == "pan_left":
        out["translation_x"] = c(-2.0 * i * s)
    elif preset == "pan_right":
        out["translation_x"] = c(2.0 * i * s)
    elif preset == "pan_up":
        out["translation_y"] = c(-2.0 * i * s)
    elif preset == "pan_down":
        out["translation_y"] = c(2.0 * i * s)
    elif preset == "orbit_left":
        out["rotation_3d_y"] = c(-0.6 * i * s)
    elif preset == "orbit_right":
        out["rotation_3d_y"] = c(0.6 * i * s)
    elif preset == "roll_cw":
        out["rotation_3d_z"] = c(0.5 * i * s)
    elif preset == "roll_ccw":
        out["rotation_3d_z"] = c(-0.5 * i * s)
    elif preset == "spiral":
        out["rotation_3d_z"] = c(0.6 * i * s)
        out["zoom"] = c(1.0 + 0.012 * i * s)
    elif preset == "sway":
        out["rotation_3d_y"] = osc(0.8 * i, 120.0 / s)
    elif preset == "dolly_zoom":  # vertigo
        out["translation_z"] = c(-1.2 * i * s)
        out["zoom"] = c(1.0 + 0.012 * i * s)
    elif preset == "rise":
        out["translation_y"] = c(-1.5 * i * s)
        out["translation_z"] = c(-0.4 * i * s)
    elif preset == "shake":
        out["translation_x"] = osc(2.5 * i, 6.0 / s)
        out["translation_y"] = osc(2.0 * i, 5.0 / s)
    elif preset == "tilt_up":
        out["rotation_3d_x"] = c(0.4 * i * s)
    elif preset == "tilt_down":
        out["rotation_3d_x"] = c(-0.4 * i * s)
    elif preset == "crane_up":
        out["translation_y"] = c(-1.6 * i * s)
        out["rotation_3d_x"] = c(-0.15 * i * s)
    elif preset == "handheld":
        p1, p2, p3 = 37.0 / s, 53.0 / s, 71.0 / s
        out["translation_x"] = (f"0:({0.6 * i:.4g}*sin(2*pi*t/{p1:.4g})"
                                f"+{0.35 * i:.4g}*sin(2*pi*t/{p3:.4g}+1.3))")
        out["translation_y"] = f"0:({0.45 * i:.4g}*sin(2*pi*t/{p2:.4g}+0.7))"
        out["rotation_3d_z"] = f"0:({0.08 * i:.4g}*sin(2*pi*t/{p3:.4g}+2.1))"
    elif preset == "drift":
        p1, p2 = 180.0 / s, 240.0 / s
        out["translation_x"] = f"0:({0.9 * i:.4g}*sin(2*pi*t/{p1:.4g}))"
        out["translation_y"] = f"0:({0.6 * i:.4g}*sin(2*pi*t/{p2:.4g}+1.1))"
        out["rotation_3d_z"] = f"0:({0.1 * i:.4g}*sin(2*pi*t/{p2:.4g}+0.4))"
    elif preset == "breathe":
        p = max(2.0, 96.0 / s)
        out["zoom"] = f"0:(1.0 + {0.006 * i:.4g}*sin(2*pi*t/{p:.4g}))"
    elif preset == "vortex":
        out["rotation_3d_z"] = c(-0.6 * i * s)
        out["zoom"] = c(1.0 - 0.01 * i * s)

    return out
