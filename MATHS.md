# Maths in Difforum - schedule cookbook

Every schedule field takes keyframes of math expressions, Deforum-style:

```
0:(expr), 60:(expr), 120:(expr)
```

Variables: `t` or `f` (frame), `s` (seconds), `fps`, `max_f` (last frame),
`pi`, `e`, `tau`, plus any audio curve (`amp`, `low`, `mid`, `high`, `onset`,
`beat`) when an Audio Analyzer is connected. Functions: `sin cos tan asin acos
atan atan2 sinh cosh tanh abs sqrt exp log log10 pow floor ceil round min max
sign clamp clip lerp smoothstep`. Python ternaries work. Evaluated by a safe
AST walker, never `eval()`.

Camera fields are **per-frame increments** (they accumulate); strength/cfg
fields are absolute values. Easing between keyframes: `linear`, `ease_in`,
`ease_out`, `ease_in_out`, `step`.

## Copy-paste recipes

| Want | Expression |
|---|---|
| Constant | `0:(0.5)` |
| Ramp between keyframes | `0:(0.3), 90:(0.7)` (easing does the rest) |
| Smooth bob / sway | `0:(0.5 + 0.2*sin(2*pi*t/60))` (period 60 frames) |
| Bounce (always positive) | `0:(abs(sin(2*pi*t/30)))` |
| Ping-pong loop A to B | `0:(lerp(0.3, 0.8, 0.5 + 0.5*sin(2*pi*t/max_f)))` |
| Pulse every 24 frames | `0:(1.0 + 0.05*(1 if t % 24 < 3 else 0))` |
| Hold, then start moving | `0:(0 if t < 48 else (t - 48) * 0.5)` |
| Gentle start (no jerk) | `0:(smoothstep(0, 30, t) * 2.0)` |
| Accelerating zoom | `0:(1.0 + 0.0004*t)` (per-frame increment grows) |
| Seconds-based timing | `0:(0.5 + 0.3*sin(2*pi*s/4))` (4-second cycle) |
| Bass pumps the zoom | `0:(1.0 + 0.06*low)` |
| Beat kicks the strength | `0:(0.35 + 0.4*beat)` |
| Quiet parts calm, drops go wild | `0:(0.3 + 0.5*amp*amp)` |
| Spin that follows the highs | `0:(0.2 + 1.5*high)` |
| Clamp anything into range | `0:(clamp(2*low, 0.0, 0.9))` |

## Prefer not to write math?

The intuitive layer generates these for you: **Camera Move** (one preset +
speed + intensity), **Camera Shots** (a shot list like `0: dolly_in`,
`48: orbit_right`), **Audio Schedule** (band + base + amount, no expression),
and **Prompt Scenes** (one box per scene). Check any curve before rendering
with **Schedule Plot** and the camera with **Camera Path Preview**.
