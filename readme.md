# MVUtensils - a faster, cleaner motion-estimation toolkit for VapourSynth

MVUtensils (namespace `mvu`) is a large refactoring and cleanup of the original
VapourSynth [mvtools](https://github.com/dubhater/vapoursynth-mvtools) port. It keeps the same
overall workflow (build a *super* clip, estimate motion vectors with *Analyse*, then feed those
vectors to filters like *Degrain* or *Flow*) but fixes a number of long-standing bugs, is faster,
and adds full high-bit-depth and float support.

## Goals

* Fix long-standing correctness bugs — most notably the right/bottom border not being processed.
* Be faster through more cache-friendly algorithms and wider SIMD (SSE2/AVX2/AVX-512 where it helps).
* Greatly reduce memory usage in `Super`, `Analyse`, `Degrain` and the `Flow*` filters.
* Full float (32-bit) support in every filter except the `Depan*` family.
* A smaller, more consistent API: arrays instead of cryptic per-direction arguments, and consistent
  frame-property propagation.

## Notes

* Supported pixel formats are GRAY and YUV at 8–16 bit integer or 32 bit float. The bit depth and
  subsampling are taken from the input/super clip; masks derive their format from the vector clip.
* The attached frame properties use the prefix `MVUtensils` by default. Every function accepts a
  `prefix` argument to change it, which lets two independent MVUtensils graphs coexist on one clip.
* Motion vectors are stored as frame properties: `<prefix>AnalysisVectors` (an int array where the
  low 32 bits hold *x* and the high 32 bits hold *y*) and `<prefix>AnalysisSAD` (the per-block SAD).

## Porting from MVTools

MVUtensils is API-compatible in spirit but not verbatim. The headline differences:

| MVTools | MVUtensils |
| --- | --- |
| namespace `mv` | namespace `mvu` |
| `Super(clip, pel=2)` (blksize implicit) | `Super(clip, blksize=8, overlap=4, pel=2)` — **blksize and overlap are mandatory** |
| `hpad` / `vpad` | `pad=[h, v]` |
| `levels` | `alllevels=True/False` |
| `blksize`/`blksizev`, `overlap`/`overlapv` | `blksize=[h, v]`, `overlap=[h, v]` (a single value applies to both axes) |
| `Analyse(isb=False, delta=1)` (forward) | `Analyse(delta=-1)` — **negative delta = forward, positive = backward** |
| `dct=0` / `dct=5` | `satd=False` / `satd=True` |
| `search`/`pelsearch` modes 0–7 | modes 0–5 (old modes 0 and 1 dropped, everything shifted by −2) |
| `rfilter` 2 / 4 | `rfilter` 1 / 2 (modes 1 and 3 dropped) |
| `lambda`, `global` | `mvlambda`, `globalmv` (avoid Python keywords) |
| `Degrain1(clip, super, mvbw, mvfw, ...)` | `Degrain(clip, super, [mvbw, mvfw], ...)` — vectors in a list |
| `thsad` + `thsadc` | `thsad=[luma, chroma]` |
| `limit` + `limitc` (int) | `limit=[luma, chroma]` (float; non-finite or > max = no limit) |
| `Flow(mode=...)`, `BlockFPS`, `Finest`, `search_coarse`, `divide`, `scbehavior` | removed |
| `FlowFPS(mask=1/2)` | `FlowFPS(extramask=False/True)` |
| `Mask(kind=0/1/2)` | `VectorLengthMask` / `SADMask` / `OcclusionMask` |

A typical denoise, before and after:

```py
# MVTools
super  = core.mv.Super(clip, pel=2)
mvbw = core.mv.Analyse(super, isb=True,  delta=1)
mvfw = core.mv.Analyse(super, isb=False, delta=1)
out  = core.mv.Degrain1(clip, super, mvbw, mvfw)

# MVUtensils
super  = core.mvu.Super(clip, blksize=8, overlap=4, pel=2)  # blksize/overlap mandatory
mvbw = core.mvu.Analyse(super, delta=1)    # positive delta = backward
mvfw = core.mvu.Analyse(super, delta=-1)   # negative delta = forward (was isb=False)
out  = core.mvu.Degrain(clip, super, [mvbw, mvfw])  # vectors passed as a list
```

## Table of Contents

* [Quick start](#quick-start)
* [Common parameters](#common-parameters)
* [Super](#super)
* [Analyse](#analyse)
* [AnalyseMany](#analysemany)
* [Recalculate](#recalculate)
* [Degrain](#degrain)
* [Compensate](#compensate)
* [Flow](#flow)
* [FlowInter](#flowinter)
* [FlowFPS](#flowfps)
* [FlowBlur](#flowblur)
* [VectorLengthMask / SADMask / OcclusionMask](#vectorlengthmask--sadmask--occlusionmask)
* [SCDetection](#scdetection)
* [Depan functions](#depan-functions)

## Quick start

The vector estimators (`Analyse`, `AnalyseMany`, `Recalculate`) consume a *super* clip and emit a
*vector* clip; the consumers (`Degrain`, `Compensate`, `Flow*`, masks) take the super and the
vector clip(s). `AnalyseMany` is the easy way to produce the list of forward/backward vectors that
`Degrain`, `FlowInter`, `FlowFPS` and `FlowBlur` expect.

```py
# Temporal denoise (radius 3 = 6 vector clips)
super = core.mvu.Super(clip, blksize=8, overlap=4)
out = core.mvu.Degrain(clip, super, core.mvu.AnalyseMany(super, radius=3))

# Motion-interpolated frame doubling
super = core.mvu.Super(clip, blksize=8, overlap=4)
out = core.mvu.FlowFPS(clip, super, core.mvu.AnalyseMany(super), num=clip.fps_num*2, den=clip.fps_den)
```

Only `Analyse` uses the full hierarchical pyramid (`Super(..., alllevels=True)`, the default). For
`Recalculate` and every consumer filter a single level is enough, so build their super with
`alllevels=False` to save time and memory.

## Common parameters

`prefix` is accepted by every function; `thscd1`/`thscd2` are accepted by the consumer filters
(`Compensate`, `Degrain`, `Flow`, `FlowInter`, `FlowFPS`, `FlowBlur`, the mask functions and
`SCDetection`). They behave identically everywhere and, to avoid repetition, are listed here once
and omitted from the per-function tables below.

| Parameter | Type | Options (Default) | Description |
| --- | --- | --- | --- |
| thscd1 | int | (400) | Scene-change SAD threshold. A block whose SAD exceeds this is considered "changed". The value is defined for an 8×8 luma block at 8-bit and is scaled internally to the actual block size, chroma usage and bit depth. |
| thscd2 | int | 0–255 (130) | Percentage of changed blocks above which the whole frame is treated as a scene change. On a scene change the consumer leaves the frame unprocessed (passes the source through). |
| prefix | str | ("MVUtensils") | Prefix of the frame properties read/written by this function. Must match between the producer and consumer of a vector/super clip. |

## Super

Prepares a clip for motion estimation: it pads the frame, optionally generates sub-pixel
(`pel`) planes, and builds the hierarchical pyramid used by `Analyse`.

```py
core.mvu.Super(clip clip, int[] blksize, int[] overlap[, int[] pad=[16, 16], int pel=2, int sharp=2, int rfilter=1, bint alllevels=True, clip pelclip=None, str prefix="MVUtensils"])
```

| Parameter | Type | Options (Default) | Description |
| --- | --- | --- | --- |
| clip | 8–16 bit integer or 32 bit float, GRAY/YUV | | Clip to prepare. |
| blksize | int[] | (required) | Block size `[h, v]` (a single value sets both). Used to pad the frame so the right/bottom edges are fully covered. Must match the block size you intend to use in `Analyse`. |
| overlap | int[] | (required) | Block overlap `[h, v]`, must be ≤ blksize/2. Used together with `blksize` for edge padding. |
| pad | int[] | ([16, 16]) | Border padding `[h, v]` in pixels. One value applies to both axes. |
| pel | int | 1, 2, 4 (2) | Sub-pixel accuracy: 1 = full-pixel, 2 = half-pixel, 4 = quarter-pixel. Higher needs more memory and time. |
| sharp | int | 0–2 (2) | Sub-pixel interpolation for `pel` > 1: 0 = bilinear, 1 = bicubic, 2 = Wiener (sharpest). |
| rfilter | int | 0–2 (1) | Pyramid downscale filter: 0 = simple average, 1 = bilinear, 2 = cubic. |
| alllevels | bint | (True) | Generate every hierarchical level. Only `Analyse` uses levels above 0; for everything else pass `False` to save memory. |
| pelclip | clip | (None) | Optional externally-supplied sub-pixel clip instead of generating one internally. |

> **Porting:** `blksize`/`overlap` were optional in mvtools and are now **mandatory** because the
> super clip pads itself to make edge blocks valid. `hpad`/`vpad` became `pad=[h, v]`, and `levels`
> became the boolean `alllevels`.

## Analyse

Estimates a field of motion vectors for one temporal direction/distance.

```py
core.mvu.Analyse(clip super[, int[] blksize=<from super>, int[] overlap=<from super>, int levels=0, int search=2, int searchparam=2, int pelsearch=<pel>, int mvlambda=<auto>, bint chroma=True, int delta=1, bint truemotion=True, int lsad=<auto>, int plevel=<auto>, bint globalmv=<auto>, int pnew=<auto>, int pzero=<pnew>, int pglobal=0, int badsad=10000, int badrange=24, bint meander=True, bint trymany=False, bint fields=False, bint tff=False, bint satd=False, str prefix="MVUtensils"])
```

| Parameter | Type | Options (Default) | Description |
| --- | --- | --- | --- |
| super | clip | (required) | Super clip from `Super` (built with `alllevels=True`). |
| blksize | int[] | (super's value) | Block size `[h, v]`. Smaller = more accurate but slower. |
| overlap | int[] | (super's value) | Block overlap `[h, v]`, ≤ blksize/2. More overlap = smoother field, slower. |
| levels | int | (0 = all) | Number of hierarchical levels to use. 0 uses all available. |
| search | int | 0–5 (2) | Search algorithm: 0 = logarithmic/diamond, 1 = exhaustive, 2 = hexagon, 3 = uneven multi-hexagon (UMH), 4 = horizontal, 5 = vertical. |
| searchparam | int | (2) | Search radius/step for the chosen `search`. |
| pelsearch | int | (super's pel) | Refinement search radius at the finest (sub-pixel) level. |
| mvlambda | int | (auto) | Motion-coherence penalty; higher favours smooth vector fields. Defaults scale with block size when `truemotion=True`, else 0. |
| chroma | bint | (True) | Include chroma planes in the SAD/SATD metric. |
| delta | int | (1) | Temporal distance **and direction** of the reference frame. **Positive = backward (past), negative = forward (future).** |
| truemotion | bint | (True) | Preset that flips the defaults of `mvlambda`, `lsad`, `pnew`, `plevel` and `globalmv` toward coherent ("true") motion vs. lowest-SAD vectors. |
| lsad | int | (auto) | SAD level above which `mvlambda` is reduced (so bad predictors aren't over-trusted). Default 1200 with `truemotion`, else 400. |
| plevel | int | 0–2 (auto) | How `mvlambda` scales with hierarchical level: 0 = constant, 1 = linear, 2 = quadratic. |
| globalmv | bint | (auto) | Estimate a global (pan) motion vector as an extra predictor. |
| pnew | int | (auto) | Penalty (relative to 256) added to the SAD of a newly found vector vs. the predictor. Default 50 with `truemotion`, else 0. |
| pzero | int | (pnew) | Penalty for accepting the zero vector. |
| pglobal | int | (0) | Penalty for the global-motion predictor. |
| badsad | int | (10000) | SAD above which a block gets a second, wider search. |
| badrange | int | (24) | Radius of that wider search. |
| meander | bint | (True) | Scan block rows alternately left-to-right / right-to-left for better predictor reuse. |
| trymany | bint | (False) | Try multiple predictors at coarse levels too (slower, occasionally better). |
| fields | bint | (False) | Treat the clip as field-based. |
| tff | bint | (False) | Top field first (only relevant with `fields=True`). |
| satd | bint | (False) | Use SATD instead of plain SAD as the block metric. Equivalent to mvtools `dct=5`. |

> **Porting:** `isb` is gone — direction is now the sign of `delta` (negative = forward).
> `dct` became the boolean `satd` (`dct=0`→`satd=False`, `dct=5`→`satd=True`). `search`/`pelsearch`
> modes lost the old 0 and 1, so subtract 2 from your old value. `lambda`→`mvlambda`,
> `global`→`globalmv`. The `search_coarse` and `divide` arguments were removed.

## AnalyseMany

Convenience wrapper that produces the whole list of vector clips `Degrain`, `FlowInter`, `FlowFPS`
and `FlowBlur` expect, in the right order. Takes every `Analyse` argument plus `radius`.

```py
core.mvu.AnalyseMany(clip super[, <all Analyse args>, int delta=1, int radius=1, str prefix="MVUtensils"])
```

| Parameter | Type | Options (Default) | Description |
| --- | --- | --- | --- |
| radius | int | (1) | Number of forward/backward vector pairs to produce. |
| delta | int | (1) | Step size between successive vectors (always positive here). |

With `delta=1, radius=2` the result is `[Analyse(delta=1), Analyse(delta=-1), Analyse(delta=2), Analyse(delta=-2)]`.

```py
vectors = core.mvu.AnalyseMany(super, radius=3)   # 6 clips, ready for Degrain(..., radius 3)
```

## Recalculate

Re-estimates an existing vector field at (typically) a finer block size, refining the vectors you
already have instead of searching from scratch. Pair it with a halved `blksize`/`overlap`.

```py
core.mvu.Recalculate(clip super, clip vectors[, int thsad=200, bint smooth=True, int[] blksize=<from super>, int search=2, int searchparam=2, int mvlambda=<auto>, bint chroma=True, bint truemotion=True, int pnew=<auto>, int[] overlap=<from super>, bint meander=True, bint fields=False, bint tff=False, bint satd=False, str prefix="MVUtensils"])
```

| Parameter | Type | Options (Default) | Description |
| --- | --- | --- | --- |
| super | clip | (required) | Super clip. Only one level is needed, so build it with `alllevels=False`. |
| vectors | clip | (required) | Vector clip to refine. |
| thsad | int | (200) | Blocks whose SAD is below this keep their vector; worse blocks are re-searched. |
| smooth | bint | (True) | Interpolate the new (finer) vector field from neighbours (True) or take the nearest old vector (False). `smooth=False` roughly matches the old `divide=1` behaviour, `smooth=True` ≈ `divide=2`. |
| blksize | int[] | (super's value) | Finer block size `[h, v]`. Usually half of the original. |
| overlap | int[] | (super's value) | Finer overlap `[h, v]`. |

Other parameters (`search`, `searchparam`, `mvlambda`, `chroma`, `truemotion`, `pnew`, `meander`,
`fields`, `tff`, `satd`) behave exactly as in [Analyse](#analyse).

> **Porting:** `Recalculate` will raise an error if the chosen `blksize`/`overlap` can't cover the
> whole frame (unlike `Super`, which pads). Halving `blksize`+`overlap` and reusing the existing
> super usually works; unusual splits may need a fresh super.

## Degrain

Motion-compensated temporal denoiser. Averages each block with its motion-compensated counterparts
from `radius` previous and `radius` following frames, weighted by how well they match.

`Degrain` auto-selects the radius from the number of vector clips; `Degrain1`…`Degrain6` are explicit
variants that take the same arguments.

```py
core.mvu.Degrain(clip clip, clip super, clip[] vectors[, int[] thsad=[400, 400], int[] planes=[0, 1, 2], float[] limit=[inf, inf], int thscd1=400, int thscd2=130, str prefix="MVUtensils"])
```

| Parameter | Type | Options (Default) | Description |
| --- | --- | --- | --- |
| clip | 8–16 bit integer or 32 bit float, GRAY/YUV | | Clip to denoise. |
| super | clip | (required) | Super clip (only level 0 is needed, so `alllevels=False`). |
| vectors | clip[] | (required) | Vector clips in `AnalyseMany` order: `[bw1, fw1, bw2, fw2, …]`. Their count selects the radius. |
| thsad | int[] | ([400, 400]) | SAD `[luma, chroma]` at which a reference block's weight reaches zero. Higher = stronger denoising. Chroma defaults to the luma value. |
| planes | int[] | ([0, 1, 2]) | Which planes to process; unprocessed planes are copied. |
| limit | float[] | ([inf, inf]) | Maximum absolute change per pixel `[luma, chroma]`. Non-finite (`inf`/`nan`) or a value above the format maximum disables limiting. |

> **Porting:** the per-direction `mvbw*/mvfw*` arguments are now the single `vectors` list, the
> `thsad`/`thsadc` pair became `thsad=[luma, chroma]`, and `limit`/`limitc` became the float
> `limit=[luma, chroma]` (defaulting to no limit instead of 255).

```py
# MVTools:    core.mv.Degrain2(clip, super, mvbw1, mvfw1, mvbw2, mvfw2, thsad=400, thsadc=300)
# MVUtensils:
v = core.mvu.AnalyseMany(super, radius=2)
out = core.mvu.Degrain(clip, super, v, thsad=[400, 300])
```

## Compensate

Builds a single motion-compensated frame: each block is copied from the reference frame at its
motion vector (useful for building your own temporal filters).

```py
core.mvu.Compensate(clip clip, clip super, clip vectors[, int thsad=10000, bint fields=False, float time=100.0, int thscd1=400, int thscd2=130, bint tff=False, str prefix="MVUtensils"])
```

| Parameter | Type | Options (Default) | Description |
| --- | --- | --- | --- |
| clip | 8–16 bit integer or 32 bit float, GRAY/YUV | | Clip to compensate. |
| super | clip | (required) | Super clip. |
| vectors | clip | (required) | A single vector clip (one direction). |
| thsad | int | (10000) | Blocks whose SAD exceeds this are taken from the source instead of the compensated reference. |
| time | float | 0–100 (100.0) | Temporal position of the compensation, as a percentage toward the reference frame. |
| fields | bint | (False) | Field-based processing. |
| tff | bint | (False) | Top field first (only relevant with `fields=True`). |

> **Porting:** the `scbehavior` argument was removed.

## Flow

Pixel-accurate motion compensation: instead of copying whole blocks it warps the reference frame
using a per-pixel vector field interpolated from the block vectors.

```py
core.mvu.Flow(clip clip, clip super, clip vectors[, float time=100.0, bint fields=False, int thscd1=400, int thscd2=130, bint tff=False, str prefix="MVUtensils"])
```

| Parameter | Type | Options (Default) | Description |
| --- | --- | --- | --- |
| clip | 8–16 bit integer or 32 bit float, GRAY/YUV | | Clip to warp. |
| super | clip | (required) | Super clip. |
| vectors | clip | (required) | A single vector clip. |
| time | float | 0–100 (100.0) | How far toward the reference frame to warp, in percent. |
| fields | bint | (False) | Field-based processing. |
| tff | bint | (False) | Top field first (only relevant with `fields=True`). |

> **Porting:** the `mode` argument was removed (only the former `mode=0` remains).

## FlowInter

Interpolates a new frame *between* two existing frames by warping both halves of the motion field
toward the requested time.

```py
core.mvu.FlowInter(clip clip, clip super, clip[] vectors[, float time=50.0, float ml=100.0, bint blend=True, int thscd1=400, int thscd2=130, str prefix="MVUtensils"])
```

| Parameter | Type | Options (Default) | Description |
| --- | --- | --- | --- |
| clip | 8–16 bit integer or 32 bit float, GRAY/YUV | | Source clip. |
| super | clip | (required) | Super clip. |
| vectors | clip[] | (required) | `[mvbw, mvfw]` (e.g. from `AnalyseMany`). |
| time | float | 0–100 (50.0) | Position of the interpolated frame between the two source frames (50 = midpoint). |
| ml | float | (100.0) | Mask scale: the motion length that maps to full occlusion masking. Lower = stronger masking. |
| blend | bint | (True) | Blend occluded regions instead of hard-selecting one direction. |

> **Porting:** the `mvbw`/`mvfw` arguments became `vectors=[mvbw, mvfw]`.

## FlowFPS

Motion-compensated frame-rate conversion, building each output frame with `FlowInter`-style warping.

```py
core.mvu.FlowFPS(clip clip, clip super, clip[] vectors[, int num=25, int den=1, bint extramask=True, float ml=100.0, bint blend=True, int thscd1=400, int thscd2=130, str prefix="MVUtensils"])
```

| Parameter | Type | Options (Default) | Description |
| --- | --- | --- | --- |
| clip | 8–16 bit integer or 32 bit float, GRAY/YUV | | Source clip. |
| super | clip | (required) | Super clip. |
| vectors | clip[] | (required) | `[mvbw, mvfw]`. |
| num | int | (25) | Output frame-rate numerator. |
| den | int | (1) | Output frame-rate denominator. |
| extramask | bint | (True) | Use a second occlusion-mask pass. `extramask=False` matches the old `mask=1`; `extramask=True` (default) matches `mask=2`. |
| ml | float | (100.0) | Occlusion mask scale (see `FlowInter`). |
| blend | bint | (True) | Blend occluded regions. |

> **Porting:** the `mask` integer became the boolean `extramask`, and `mvbw`/`mvfw` became
> `vectors=[mvbw, mvfw]`.

## FlowBlur

Creates motion blur by smearing each pixel along its motion vector.

```py
core.mvu.FlowBlur(clip clip, clip super, clip[] vectors[, float blur=50.0, int prec=1, int thscd1=400, int thscd2=130, str prefix="MVUtensils"])
```

| Parameter | Type | Options (Default) | Description |
| --- | --- | --- | --- |
| clip | 8–16 bit integer or 32 bit float, GRAY/YUV | | Source clip. |
| super | clip | (required) | Super clip. |
| vectors | clip[] | (required) | `[mvbw, mvfw]`. |
| blur | float | 0–200 (50.0) | Blur strength as a percentage of the motion-vector length. |
| prec | int | (1) | Sampling step precision along the motion path; lower = more samples = smoother and slower. |

> **Porting:** the `mvbw`/`mvfw` arguments became `vectors=[mvbw, mvfw]`.

## VectorLengthMask / SADMask / OcclusionMask

These three functions replace the old `Mask(kind=…)`. Each produces a **full-range grayscale** mask
derived from a vector clip; the output bit depth/format is taken from the vector clip (float masks
are clamped to 0–1).

* **VectorLengthMask** — brightness proportional to motion-vector magnitude (old `kind=0`).
* **SADMask** — brightness proportional to per-block SAD (old `kind=1`).
* **OcclusionMask** — brightness proportional to occlusion / divergence of the field (old `kind=2`).

```py
core.mvu.VectorLengthMask(clip vectors[, float ml=100.0, float gamma=1.0, float time=100.0, float scval=0.0, int thscd1=400, int thscd2=130, str prefix="MVUtensils"])
core.mvu.SADMask(clip vectors[, ...same...])
core.mvu.OcclusionMask(clip vectors[, ...same...])
```

| Parameter | Type | Options (Default) | Description |
| --- | --- | --- | --- |
| vectors | clip | (required) | Vector clip; its format determines the mask format. |
| ml | float | (100.0) | Scale: the motion length / SAD that maps to the maximum mask value. Lower = stronger mask. Must be > 0. |
| gamma | float | (1.0) | Gamma curve applied to the mask. Must be ≥ 0. |
| time | float | 0–100 (100.0) | Temporal position used when projecting the vector field. |
| scval | float | (0.0) | The exact value written for frames detected as scene changes. For float you may pass any value (e.g. `inf`/`nan`), which can break the 0–1 guarantee. |

> **Porting:** `Mask(kind=…)` is split into the three named functions, the pointless `clip`
> argument was removed, and `ysc` became the float `scval`. Output is always a single grayscale
> plane rather than the original's UV trickery.

## SCDetection

Marks scene-change frames (using the vector clip's SAD statistics) by setting the
`_SceneChangePrev`/`_SceneChangeNext` frame properties; the pixels are passed through unchanged.

```py
core.mvu.SCDetection(clip clip, clip vectors[, int thscd1=400, int thscd2=130, str prefix="MVUtensils"])
```

| Parameter | Type | Options (Default) | Description |
| --- | --- | --- | --- |
| clip | clip | (required) | Clip to tag. |
| vectors | clip | (required) | Vector clip used for the decision. |

## Depan functions

The global-motion (camera pan/zoom/rotation) tools are carried over largely unchanged from mvtools,
with fixed edge handling and minor speedups. They are the only filters **without** float support.
Refer to the [mvtools2 documentation](https://github.com/pinterf/mvtools/blob/mvtools-pfmod/Documentation/mvtools2.html)
for the detailed meaning of their parameters.

```py
core.mvu.DepanAnalyse(clip clip, clip vectors[, clip mask, int zoom, int rot, float pixaspect, float error, int info, float wrong, float zerow, int thscd1, int thscd2, int fields, int tff])
core.mvu.DepanEstimate(clip clip[, float trust, int winx, int winy, int wleft, int wtop, int dxmax, int dymax, float zoommax, float stab, float pixaspect, int info, int show, int fields, int tff])
core.mvu.DepanCompensate(clip clip, clip data[, float offset, int subpixel, float pixaspect, int matchfields, int mirror, int blur, int info, int fields, int tff])
core.mvu.DepanStabilise(clip clip, clip data[, float cutoff, float damping, float initzoom, int addzoom, int prev, int next, int mirror, int blur, float dxmax, float dymax, float zoommax, float rotmax, int subpixel, float pixaspect, int fitlast, float tzoom, int info, int method, int fields])
```

All `Depan*` arguments except the leading clips are optional; their meanings and defaults match the
mvtools `MDepan*` filters in the linked reference.
