# Difforum prompt pack

A curated set of prompt-travel presets to explore quality and motion. Each one
is ready to paste into **Difforum - Prompt Schedule (travel)** (the `prompts`
box). The frame numbers assume `max_frames` around 48; scale them to your clip
length. Settings under each preset are starting points, not rules.

How to read a preset:

- **Travel** goes into the Prompt Schedule node (`frame: prompt` per line).
- **Camera** values go into the Camera node (advanced) fields, or pick the
  closest Camera Move preset.
- **Strength** is the per-frame denoise (Schedule node). Lower = more coherent
  and smoother, higher = more change per frame.
- **Symmetry** refers to the Feedback Sampler's `symmetry` option.

## Quality settings (read this first)

The checkpoint is most of the quality. Base SD1.5 (`v1-5-pruned-emaonly`) looks
flat; use a strong finetune instead. Good drop-in options:

- **SDXL** (best detail): Juggernaut XL, RealVisXL. Render at **1024x576** (16:9).
- **SD1.5**: DreamShaper 8. Render at 768x432 or 768x768.
- **Flux** (top quality, slower): flux1-schnell, 4 to 8 steps, cfg 1.

Sampler that holds detail across the feedback loop:

- **steps 26 to 30**, **sampler `dpmpp_2m`**, **scheduler `karras`**
- **cfg** 6 to 7 (SDXL) or 7 to 8 (SD1.5)
- Append this to the positive prompt:
  `, highly detailed, sharp focus, intricate, cinematic lighting, vivid colors, masterpiece`

Negative prompt that works for most of these:

```
blurry, low quality, watermark, text, jpeg artifacts, deformed, washed out, oversaturated, plain background, low contrast
```

---

## 1. Cosmic Voyage

A flight through deep space. Big, clean, great for an intro.

```
0: a vast purple nebula, distant stars, volumetric cosmic dust, deep space, cinematic
16: a swirling spiral galaxy seen edge on, glowing core, star clusters
32: the bright accretion disk around a black hole, gravitational lensing, intense light
47: a newborn star igniting, lens flare, brilliant rays piercing the dark
```

- Camera: mode `3d`, zoom `0:(1.03)`, translation_z `0:(2)`, rotation_z `0:(0.2)`
- Strength `0:(0.45)`, cfg 7.0, color_mode lab

## 2. Organic Bloom

Growth and life. Morphs read beautifully here.

```
0: a single glowing seed in dark soil, macro, dramatic light
14: green vines unfurling, tendrils reaching, dewdrops, lush
28: a field of bioluminescent flowers blooming, vivid petals, fireflies
47: an infinite fractal garden, intricate botanical patterns, golden hour
```

- Camera: mode `2d`, zoom `0:(1.02)`, rotation_z `0:(0.1)`
- Strength `0:(0.4)`, cfg 7.5

## 3. Liquid Metal Dreams

Reflective, flowing surfaces. Pairs well with Echo Trails.

```
0: flowing chrome liquid, mirror surface, studio light, abstract
16: rippling mercury waves, silver reflections, smooth caustics
32: molten gold pouring, glowing hot metal, sparks
47: a frozen explosion of liquid metal, sharp reflective shards, dramatic
```

- Camera: mode `2d`, zoom `0:(1.015)`, rotation_z `0:(0.3)`
- Strength `0:(0.5)`, add Echo Trails (decay 0.6, mix 0.4)

## 4. Sacred Geometry (kaleidoscope)

Built for the in-loop symmetry. Mesmerizing, hypnotic.

```
0: ornate symmetric mandala, stained glass, iridescent, intricate fractal detail
16: glowing sacred geometry, golden ratio spirals, luminous lines
32: a rotating fractal flower of life, jewel tones, deep detail
47: a radiant cosmic mandala, blooming symmetry, vivid colors
```

- Camera: mode `2d`, rotation_z `0:(0.7)`, zoom `0:(1.012)`
- Strength `0:(0.42)`, **Symmetry `kaleidoscope`, segments 6**, Echo Trails on
- See `difforum_mesmerize_kaleidoscope.json`

## 5. Neon City Flythrough

Cyberpunk motion. Strong forward camera sells it.

```
0: a rain slicked neon street at night, reflections, cyberpunk, cinematic
16: flying between towering skyscrapers, holograms, dense neon signs
32: a futuristic megacity skyline, flying cars, glowing grid
47: orbit above the city at night, city lights from above, atmosphere
```

- Camera: mode `3d`, translation_z `0:(3)`, zoom `0:(1.02)`, rotation_x `0:(0.1)`
- Strength `0:(0.45)`, cfg 7.0

## 6. Elemental Shift

Four elements morphing into each other. High contrast, dramatic.

```
0: a wall of roaring fire, embers, intense orange glow, dramatic
12: crashing ocean waves, deep blue water, foam, spray
24: a frozen ice cavern, blue crystals, glittering frost
36: a swirling storm, lightning, dark clouds, electric
47: pure elemental energy, fire water ice and wind merging, epic
```

- Camera: mode `2d`, zoom `0:(1.02)`, rotation_z `0:(0.4*sin(2*pi*t/48))`
- Strength `0:(0.5)`, cfg 7.5

## 7. Psychedelic Tunnel

The classic Deforum feel, modernized.

```
0: psychedelic cosmic tunnel, vivid swirling colors, fractal walls
16: an endless kaleidoscopic corridor, neon spirals, deep perspective
32: melting rainbow geometry, liquid light, trippy
47: a vortex of intricate fractal patterns, hypnotic, vibrant
```

- Camera: mode `2d`, zoom `0:(1.04)`, rotation_z `0:(0.6)`
- Strength `0:(0.5)`, optional Symmetry `kaleidoscope` segments 8

## 8. Underwater Abyss

Calm, deep, glowing. Slow camera.

```
0: a vibrant coral reef, schools of fish, sun rays through water
16: descending into the deep, bioluminescent jellyfish, dark blue
32: an ancient sunken temple, glowing runes, drifting particles
47: a colossal bioluminescent leviathan in the abyss, awe, atmosphere
```

- Camera: mode `3d`, translation_z `0:(1.5)`, translation_y `0:(-0.5)`, zoom `0:(1.01)`
- Strength `0:(0.4)`, cfg 7.0

## 9. Audio-reactive Pulse (music video)

Drive zoom and strength from the audio. Connect Audio Analyzer to the schedules.

```
0: abstract liquid light show, vivid colors, glowing particles, reactive
16: pulsing energy waves, neon bloom, rhythmic patterns
32: explosive bursts of color, fractal bloom, dynamic
47: a vortex of sound made visible, vibrant, kinetic
```

- Zoom schedule: `0:(1.0 + 0.6*low)` (bass pumps the zoom)
- Strength schedule: `0:(0.35 + 0.4*beat)` (beats add change)
- cfg 7.0, see `difforum_audio_reactive_video.json`

## 10. Dreamscape Morph

Surreal, painterly, slow drift. Great for an ambient loop.

```
0: a surreal floating island in a pastel sky, dreamlike, soft light
16: melting clocks over an endless desert, surrealism, long shadows
32: a staircase spiraling into the clouds, impossible architecture
47: a giant moon over a mirror ocean, serene, otherworldly
```

- Camera: mode `2d`, zoom `0:(1.01)`, translation_x `0:(0.5)`
- Strength `0:(0.38)`, cfg 7.5, color_mode lab

---

## Quick recipe tips

- **Smoother motion:** lower strength (0.35 to 0.45) and keep `color_mode = lab`.
- **More change per beat / scene:** raise strength toward 0.55, or shorten the
  gaps between prompt keyframes.
- **Mesmerizing loops:** turn on Feedback Sampler `symmetry` and add Echo Trails.
- **Seamless loop:** match the last prompt back toward the first, and ping-pong
  the frames on export.
- **Higher quality:** swap the SD1.5 checkpoint for SDXL or Flux. The sampler is
  model-agnostic, the prompts above carry over.
