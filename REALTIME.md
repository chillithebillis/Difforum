# Difforum Live - realtime / live-performance

Uma versão **tempo-real** do Difforum, nativa: tipo o StreamDiffusion do
TouchDesigner, mas dentro do ComfyUI e com a alma do Deforum (câmera matemática,
áudio-reatividade, prompt travel, simetria) dirigindo um loop de difusão rápido,
com preview ao vivo e saída para software de VJ.

## O que já existe: Difforum Live Sampler (nativo)

O **Difforum Live Sampler** é o motor realtime próprio do Difforum. O ComfyUI
normal reexecuta o grafo a cada "Queue Prompt" (não é streaming), então o Live
Sampler resolve isso com um **loop interno** num único node:

- modelo **residente** na VRAM (carrega uma vez, não recompila o grafo por frame);
- a cada tick: **warp** (câmera) → **simetria/caleidoscópio** → **re-difusão de
  1-2 steps** (Turbo/LCM) → **color match** → realimenta;
- **preview ao vivo dentro do node** enquanto roda (manda cada frame pra UI por
  WebSocket via `ProgressBar`) - você dá Queue uma vez e assiste gerar;
- **sinks opcionais**: `stream_dir` grava os frames numa pasta em tempo real (pra
  OBS/Resolume/ffmpeg consumirem) e `spout_name` envia por **Spout** se você tiver
  `SpoutGL` (degrada pra no-op se não tiver, nunca quebra);
- no fim retorna o batch de frames (dá pra salvar em MP4 também).

Isso é "câmera matemática → difusão → tela, ao vivo" sem depender de runtime
externo. Pareie com um modelo **SD-Turbo / SDXL-Turbo / LCM** (1-2 steps, cfg ~1)
e resolução baixa (512) pra ganhar FPS. Template: `difforum_realtime_live.json`.

### Espelho mágico: webcam ao vivo

Setando `live_source` o Live Sampler vira um estilizador de **câmera ao vivo**
(o efeito TouchDesigner): cada tick puxa um frame novo, blenda com o feedback
(`source_blend`), aplica o caleidoscópio e difunde. Valores de `live_source`:

- `""` (vazio) = feedback generativo puro (padrão);
- `"0"` / `"1"` = índice de webcam (via OpenCV);
- um caminho de arquivo = estiliza um vídeo (em loop).

`source_blend` 0.9 = segue a câmera com um leve rastro; 1.0 = estiliza limpo;
baixo = mais feedback/trails. Precisa de `opencv-python` (já vem na maioria das
instalações ComfyUI). Validado de ponta a ponta com vídeo e webcam.

### Parâmetros que importam

| Campo | O que faz |
|---|---|
| `duration_frames` | quantos frames a sessão ao vivo roda |
| `steps` / `cfg` | 1-2 e ~1.0 com Turbo/LCM = realtime |
| `strength` | denoise por frame: baixo = coerente/suave, alto = mais mudança |
| `symmetry` | espelho/caleidoscópio dentro do loop (custo ~zero, hipnótico) |
| `target_fps` | trava o FPS (0 = o mais rápido possível) |
| `live_source` | webcam (`"0"`) ou vídeo (caminho) para o modo espelho ao vivo |
| `source_blend` | quanto da câmera vs feedback (0.9 = câmera com leve rastro) |
| `stream_dir` / `spout_name` | saída ao vivo pra apps de VJ |
| `live_preview` | liga o preview ao vivo no node |

## Controle ao vivo

A câmera (presets + intensidade), o prompt travel e a reatividade de áudio já
são Difforum. Para knobs/faders físicos, mapeie **MIDI/OSC**
([RyanOnTheInside](https://github.com/ryanontheinside/ComfyUI_RyanOnTheInside))
para os parâmetros do Live Sampler.

## Desempenho esperado (referência)

| GPU | Modelo | ~FPS @512 |
|---|---|---|
| RTX 3060 12GB | SD1.5 LCM/Turbo, 1 step | ~6-15 |
| RTX 4090 | SD1.5-Turbo + TensorRT | 30-60+ |
| RTX 4090 | SDXL-Turbo | ~10-20 |

Aceleração extra (opcional): **TensorRT**
([ComfyUI_TensorRT](https://github.com/comfyanonymous/ComfyUI_TensorRT)) compila
o modelo em engine (3-10 min, uma vez) e **TAESD** (VAE minúsculo) deixa o decode
quase instantâneo. É o que leva a FPS alto de verdade.

## Roadmap

1. ~~Entrada de webcam~~ - **feito** (`live_source` no Live Sampler).
2. **Sink NDI** nativo (além de Spout/folder) pra Resolume/OBS sem plugin extra.
3. **Live param bus** - MIDI/OSC mapeado direto nos campos do Live Sampler.
4. **A/B prompt crossfade** num fader, estilo VJ.

## Interop opcional

Se você precisar especificamente de **streaming web/WebRTC** (mandar pra um
navegador ou pra nuvem), dá pra hospedar os mesmos nodes Difforum sob o
[ComfyStream](https://github.com/yondonfu/comfystream); e pra **FPS máximo** num
palco, o [StreamDiffusion](https://github.com/pschroedl/ComfyUI-StreamDiffusion)
+ TensorRT é o pipeline mais rápido. São caminhos complementares - o Live Sampler
nativo cobre o caso "rodar ao vivo no meu ComfyUI" sem nada disso.
