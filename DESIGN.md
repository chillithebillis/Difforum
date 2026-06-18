# Deforum 2 para ComfyUI - Documento de Design

> Análise do cenário (junho/2026) e arquitetura proposta para um sucessor moderno
> do Deforum dentro do ComfyUI.

---

## 1. Estado atual do Deforum

| Projeto | Estado | Tecnologia | Problema |
|---|---|---|---|
| Deforum A1111 (clássico) | Legado | SD1.5/SDXL img2img feedback loop, depth MiDaS/AdaBins, schedules matemáticos | Flicker, sem coerência temporal real, ecossistema A1111 morrendo |
| [XmYx/deforum-comfy-nodes](https://github.com/XmYx/deforum-comfy-nodes) | Ativo (push 06/2026), 199★, 29 issues | Port do deforum clássico; loop via auto-queue (hack contra o DAG do Comfy) | Mesmo paradigma 2023; UX confusa; preso a SD1.5-era |
| Deforum Studio (deforum.studio) | Comercial/cloud | Presets áudio-reativos, render em nuvem | Fechado, pago, fora do ComfyUI |
| Parseq | Ativo | Scheduler visual externo (JSON de keyframes) | Só scheduling - continua útil como integração |

**O que faz o Deforum ser "Deforum"** (a alma que devemos preservar):
1. **Schedules com expressões matemáticas** - `0:(0), 60:(2.5*sin(2*pi*t/30))` para qualquer parâmetro.
2. **Câmera 2D/3D real** - translation/rotation/zoom warpando o frame anterior via depth.
3. **Feedback loop** - cada frame nasce do anterior (o "look" psicodélico de morphing contínuo).
4. **Cadence** - difundir 1 a cada N frames e interpolar o resto.
5. **Áudio-reatividade** - schedules dirigidos por música.

**O que envelheceu mal:** flicker frame-a-frame, drift de cor, depth ruim (MiDaS),
interpolação de cadence primitiva, ausência de modelo temporal.

---

## 2. Tecnologias modernas a incorporar (2025–2026)

### Geração de vídeo
- **Wan 2.2** (14B MoE / 5B): melhor open-source para controle. Variantes-chave:
  - **FLF2V (First-Last Frame)**: gera vídeo entre dois keyframes → perfeito para "cadence 2.0".
  - **VACE**: aceita frames de controle (depth/flow/pose/scribble) + imagens de referência +
    inpainting temporal → é o "img2img temporal" que o Deforum nunca teve.
  - **Lightning LoRAs (4-step)** + GGUF para low-VRAM.
- **LTX-Video**: alternativa muito rápida para preview/iteração.
- AnimateDiff: ainda útil como modo retrô/low-VRAM, mas não é mais o estado da arte.

### Percepção / warp
- **Depth Anything V2** (ou DepthCrafter para depth temporalmente consistente) - substitui MiDaS.
- **RIFE 4.x / FILM / GIMM-VFI** - interpolação para cadence e pós.
- Optical flow (RAFT/UniMatch) para blending anti-flicker e máscaras de oclusão.

### Áudio-reatividade
- [ComfyUI_Yvann-Nodes](https://github.com/yvann-ba/ComfyUI_Yvann-Nodes) e
  [RyanOnTheInside](https://github.com/ryanontheinside/ComfyUI_RyanOnTheInside) (everything-reactivity).
- [FizzNodes](https://github.com/FizzleDorf/ComfyUI_FizzNodes) / AudioScheduler - referência de API de scheduling no Comfy.
- Análise via librosa: amplitude, onsets, bandas de frequência, BPM.

### Consistência e aceleração
- IPAdapter / Flux Redux para travar estilo entre keyframes.
- Color matching LAB + referência ancorada (anti-drift).
- TeaCache, SageAttention, torch.compile.

---

## 3. Conceito central do Deforum 2

> **O warp deixa de ser o frame final e vira condição para um modelo de vídeo.**

O Deforum original warpa o frame anterior e roda img2img - daí o flicker.
O Deforum 2 mantém o motor de câmera matemático, mas usa os frames warpados como
**guias estruturais** (depth/flow) para o Wan VACE preencher com coerência temporal nativa.

### Dois modos de operação

**Modo A - Clássico+ (low VRAM, look retrô melhorado)**
Feedback loop tradicional (SD1.5/SDXL/Flux img2img) com melhorias modernas:
depth Anything V2 no warp, blending por optical flow, color coherence LAB,
máscara de oclusão para re-difundir só o que o warp expôs.

**Modo B - Híbrido Vídeo (o verdadeiro "Deforum 2")**
1. Motor de câmera gera a trajetória e os guide frames warpados (a cada N frames = "super-cadence", ex. N=16–32).
2. Keyframes são difundidos nos pontos âncora (com IPAdapter para estilo).
3. **Wan 2.2 VACE/FLF2V preenche os intervalos**, condicionado por depth/flow dos guide frames.
4. Resultado: movimento de câmera 100% deforum (matemático, áudio-reativo) com qualidade
   e suavidade de modelo de vídeo nativo. Duração ilimitada por encadeamento de segmentos
   (overlap de frames entre segmentos, técnica dos video-extenders).

---

## 4. Nodes propostos (schema)

### Tipos customizados
- `D2_SCHEDULE` - curva avaliada por frame (float[]) com metadados.
- `D2_CAMERA` - lista de matrizes 4×4 (pose por frame) + intrínsecos.
- `D2_AUDIO` - features de áudio alinhadas ao fps (amplitude, onsets, bandas, BPM grid).
- `D2_GUIDES` - pacote de guias por frame: {warped_rgb, depth, flow, occlusion_mask}.

### Núcleo (Fase 1 - sem GPU)
| Node | Inputs | Outputs | Função |
|---|---|---|---|
| `D2 Anim Setup` | width, height, fps, max_frames, seed_schedule, mode | `D2_PARAMS` | Config global da animação |
| `D2 Schedule` | string keyframes, `D2_PARAMS`, [`D2_AUDIO`] | `D2_SCHEDULE`, FLOAT[] | Parser `0:(v), n:(expr)` com `t, fps, audio.*`, interpolação linear/bezier/step |
| `D2 Prompt Schedule` | string multilinha, CLIP, `D2_PARAMS` | CONDITIONING[] | Prompt travel com blend de conditioning |
| `D2 Audio Analyzer` | AUDIO, `D2_PARAMS`, bandas, smoothing | `D2_AUDIO`, curvas FLOAT[] | librosa → curvas por frame |
| `D2 Camera` | schedules tx/ty/tz/rx/ry/rz/zoom/fov | `D2_CAMERA` | Trajetória de câmera |
| `D2 Parseq Import` | JSON | `D2_SCHEDULE`s | Compatibilidade Parseq |

### Motor de movimento (Fase 2)
| Node | Inputs | Outputs | Função |
|---|---|---|---|
| `D2 Depth` | IMAGE, modelo (DA-V2 S/B/L, DepthCrafter) | DEPTH | Estimativa de profundidade |
| `D2 Warp 3D` | IMAGE, DEPTH, `D2_CAMERA`, frame_idx | IMAGE, MASK (oclusão) | Reprojeção 3D do frame |
| `D2 Feedback Sampler` | MODEL, VAE, COND[], `D2_CAMERA`, schedules (strength, cfg, noise), cadence, color_coherence, init IMAGE | IMAGE[] | **Node monolítico** com o loop interno (Modo A) |
| `D2 Color Coherence` | IMAGE[], ref, modo (LAB/HSV/none), força | IMAGE[] | Anti-drift de cor |

### Ponte de vídeo (Fase 3)
| Node | Inputs | Outputs | Função |
|---|---|---|---|
| `D2 Guide Builder` | keyframes IMAGE[], `D2_CAMERA`, DEPTH | `D2_GUIDES` | Warps + depth + flow para os intervalos |
| `D2 VACE Fill` | WAN MODEL, `D2_GUIDES`, keyframes, ref IMAGE, overlap | IMAGE[] | Preenche segmentos via VACE/FLF2V, encadeia com overlap |
| `D2 Interpolate` | IMAGE[], fator, engine (RIFE 4.x/FILM) | IMAGE[] | Pós-cadence |

### Decisão de arquitetura: o loop
ComfyUI é um DAG sem loops nativos. Opções avaliadas:
- ❌ Auto-queue (abordagem XmYx): frágil, estado global, UX ruim.
- ✅ **Node monolítico com loop interno** (como samplers do AnimateDiff): robusto, cacheável,
  permite barra de progresso e preview por frame via callbacks.
- 🔶 Futuro: migrar para subgraph/loop nativo quando a execution model do Comfy estabilizar isso.

---

## 5. Roadmap de implementação

1. **Fase 1 - Schedule Engine** (puro Python, testável sem GPU)
   Parser de expressões (numexpr/AST seguro), curvas, áudio, Parseq import. É a fundação de tudo.
2. **Fase 2 - Modo Clássico+**
   Depth Anything V2 + Warp 3D + Feedback Sampler + color coherence. Já entrega um "deforum melhor que o original".
3. **Fase 3 - Modo Híbrido Vídeo**
   Guide Builder + VACE Fill com Wan 2.2 (+ Lightning LoRA p/ velocidade, GGUF p/ low-VRAM).
4. **Fase 4 - UX**
   Widget JS de preview de curvas no node, presets áudio-reativos, exemplos de workflow.

## 6. Requisitos de hardware (referência)

| Modo | VRAM mínima | Confortável |
|---|---|---|
| Clássico+ (SD1.5) | 6 GB | 8 GB |
| Clássico+ (SDXL/Flux) | 10 GB | 16 GB |
| Híbrido (Wan 2.2 5B / GGUF) | 8–10 GB | 12 GB |
| Híbrido (Wan 2.2 14B fp8) | 16 GB | 24 GB |

---

## 7. O esquema HÍBRIDO em detalhe (o "Deforum 2" de verdade)

### Grafo de nodes

```
                    ┌─────────────────┐
  Anim Setup ──────►│  Difforum Camera │  schedules tx/ty/tz/rot/zoom (+audio)
       │            └────────┬────────┘
       │                     │ DIFFORUM_CAMERA (poses 4x4 por frame)
       │                     ▼
  Model Profile ───►┌─────────────────┐◄──── keyframe IMAGE(s) (anchors)
  (VRAM 12-32GB)    │ Difforum Hybrid  │◄──── DIFFORUM_SCHEDULE (strength/cfg/motion)
       │            │     Render      │◄──── COND (prompt travel)
       └───────────►│                 │◄──── WanVideoWrapper MODEL/VAE
                    └────────┬────────┘
                             │ IMAGE[] (vídeo)
                  ┌──────────┴──────────┐
                  ▼                     ▼
            RIFE/GIMM-VFI         Video Combine
            (cadence 2.0)         (mp4 + áudio)
```

### Loop interno do Hybrid Render (por segmento)

A animação é cortada em **segmentos** de `segment_frames` (do perfil), com
`overlap` frames compartilhados para encadear sem corte visível:

1. **Âncora**: difunde (ou recebe) o keyframe inicial do segmento. IPAdapter/
   Redux trava o estilo para não driftar entre segmentos.
2. **Guias de câmera**: para cada frame do segmento, warpa a âncora pela `pose`
   acumulada da `DIFFORUM_CAMERA` usando depth (Depth Anything V2) → gera
   `warped_rgb + depth + máscara de oclusão`. É aqui que mora o controle de
   câmera estilo Deforum.
3. **Preenchimento VACE**: o Wan 2.2 recebe os guias como controle estrutural +
   a âncora como referência e **gera o segmento com coerência temporal nativa**
   (sem o flicker do feedback img2img clássico).
4. **Encadeia**: os últimos `overlap` frames viram a âncora do próximo segmento
   (técnica dos video-extenders) → duração ilimitada.
5. **Schedules por frame**: strength, cfg, motion_scale e prompt são lidos das
   curvas Difforum a cada frame - tudo áudio-reativo.

### Modos de motor (família selecionável no Model Profile)

- **wan22** (recomendado): híbrido completo VACE/FLF2V. O melhor resultado.
- **sd15_animatediff**: o "look AnimateDiff" leve; câmera vira context+motion.
- **ltxv**: previews ultrarrápidos para iterar schedules antes do render final.
- **sdxl**: modo Clássico+ (feedback loop por frame, look deforum puro).

### Tiers de VRAM (resolvidos automaticamente - `Difforum Model Profile`)

| VRAM | Modelo (família wan22) | Quant | Resolução | Segmento | Offload |
|---|---|---|---|---|---|
| 12 GB | Wan 2.2 **5B** | GGUF Q5_K_M | 640×640 | 49f | sequential |
| 16 GB | Wan 2.2 **14B** | GGUF Q4_K_M | 832×480 | 65f | model |
| 24 GB | Wan 2.2 **14B** | GGUF Q5_K_M | 1280×720 | 81f | model |
| 32 GB | Wan 2.2 **14B** | fp8 | 1280×720 | 81f | none |

`quality=fast` usa **Lightning LoRA (4 steps)**; `balanced` 6 steps; `quality`
sobe para 20 steps sem distill. O node auto-detecta a VRAM via torch e cai no
tier certo (abaixo de 12 GB faz clamp pro 5B em vez de quebrar).

### Status desta fase

✅ `Difforum Camera` (motor de câmera 2D/3D, poses acumuladas, áudio-reativo)
✅ `Difforum Model Profile` (tiering 12-32GB, GGUF-aware, auto-detect VRAM)
✅ `Difforum Model Catalog` (modelos clássicos treináveis + receitas)
✅ `Difforum Warp (2D/3D)` (warp afim + reprojeção por depth + máscara de oclusão)
✅ `Difforum Feedback Sampler` (Modo Clássico+: warp→re-difunde→cor, vídeo end-to-end)
✅ `Difforum Guide Builder` (ponte híbrida: guias warpados → Wan VACE existente)

### Por que Guide Builder em vez de um Hybrid Render monolítico

Acoplar à API interna do WanVideoWrapper quebra a cada versão dele. O Guide
Builder gera batches de IMAGE/MASK padrão que plugam em qualquer grafo VACE do
Wan que o usuário já tenha - robusto e à prova de versão. A câmera continua 100%
Difforum; o Wan faz só o preenchimento temporal.
