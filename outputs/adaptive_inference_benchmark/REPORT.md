# Adaptive prediction benchmark — 2026-10-06

Hardware: RTX 5070 Ti (16 GB), Intel i7-9700KF (8 physical cores), 60 GB RAM. Existing desktop applications remained open. The pinned inference environment was not upgraded. This is a throughput benchmark, not a guarantee for other hardware.

## Implementation

The GUI defaults to v8-m-mr4 and offers v8-s-mr4. Each spawned process owns its detector, seeded U-Net and GB9 models, reused across images/batches. GPU concurrency increases only within memory/core bounds, and warm service measurements stop growth when throughput no longer improves. Unknown GPU memory and MPS use one process. CPU uses bounded four-thread groups. Failures release all worker contexts, retry at lower concurrency, then use CPU with the same selected model. Ordinary input/checksum errors are surfaced. Cancellation drains submitted images without dispatching more.

Sequential improvements share image decoding when EXIF orientation permits and cache runtime metadata. FP32, tile batch size one, DPI-dependent tiles, overlap, confidence, whole-image pass, merge and refinement remain unchanged. Earlier tile-batching experiments changed individual outputs and were not adopted. Selecting S is an explicit model change, not a transparent optimization of M.

The initial stage profile found detection/slicing/merging consumed 94.2% of wall time; U-Net inference was 0.5%, niceness classification 0.5%, sidecar writing 0.6%. See [stage breakdown](../pipeline_benchmark/REPORT.md).

## Measurement scope

Before-code snapshots and GPU/CPU pickles are in `before/`. The three-plate comparison uses S counts 0, 20 and 48 (M: 1, 18 and 48), all 600 DPI. Baseline GPU comparison uses its warmed second repetition. CPU baseline was one sequential pass, with the first plate cold; after runs separately report cold and warmed batches. After batch wall time includes child result transfer and benchmark pickle writes, excludes GUI rendering, import, export and source-sidecar writes. Cold process/model startup can outweigh throughput gains for very small batches.

CPU per-image thread tuning on the middle plate:

| Threads | S seconds | M seconds |
|---|---:|---:|
| 1 | 54.18 | 159.82 |
| 2 | 30.01 | 91.40 |
| 4 | 19.46 | 56.25 |
| 8 | 26.43 | 86.39 |

Four threads were retained. Thread-count changes can alter last-bit detector scores; four-thread optimized outputs are compared against the original four-thread CPU results. CPU and GPU are separate numerical backends and are not claimed bit-identical.

## Before / after results

Seconds per image are batch wall time divided by image count (throughput, not single-image latency). GPU after is the mean of warmed repetitions 3–5; CPU after is repetition 1.

| Device / model | Before s/image | After s/image | Speedup |
|---|---:|---:|---:|
| GPU M | 2.001 | 0.987 | 2.03× |
| GPU S | 1.771 | 0.672 | 2.64× |
| CPU M | 51.905 | 43.685 | 1.19× |
| CPU S | 20.279 | 18.405 | 1.10× |

CPU baseline includes a cold first image, so its ratios are approximate rather than a fully warmed controlled comparison. The original warmed nonempty plates averaged 20.737 s for S and 51.348 s for M. Cold after batches took 58.34 s (S) and 141.76 s (M), versus original three-image totals of 60.84 s and 155.72 s excluding initialization. Warm after totals were 55.21 s and 131.06 s.

Fresh GPU after batches took 10.46 s for S and 10.41 s for M, including process/model startup and concurrency ramp. Warm three-worker batches averaged 2.02 s for S and 2.96 s for M. These overheads matter for small runs; process/model reuse amortizes them across bulk work.

All three matched plates retained identical final counts, centres, packed masks, areas, detector scores, niceness probabilities and decisions on each device/model. All CPU compared intermediate outputs were also exact. Raw comparisons: `gpu_final/measurements.json`, `integrated_cpu4/measurements.json`, `integrated_cpu_m/measurements.json`.

## Full S reference audit

The production scheduler processed all 150 plates in **116.56 seconds**, including cold startup, adaptive exploration and worker shutdown (0.777 s/image). It tried four workers and settled at three. All 150 counts matched, with 2,706 detections, 2,705 classified females and 1,439 nice decisions. Every compared final centre, score, area, probability and decision matched the frozen CSVs at 1e-12 tolerance; all sidecars round-tripped without loss. Two females had tiny intermediate-feature differences, so strict feature equality failed while `matches_reference_outputs` passed. This is not a claim of bitwise identity. Raw results: `full_parity/summary.json`, `differences.json`, `execution.json`.

## Portability and limits

Physical execution was tested on this NVIDIA GPU and x86 CPU. Low-VRAM limits, worker crashes, allocation failures, accelerator-to-CPU fallback, cancellation and model attribution are exercised with injected regression tests; other GPUs, Apple MPS and operating systems were not physically benchmarked. The scheduler does not promise an exact global optimum, use multiple GPU devices, or guarantee execution when a single CPU image exceeds available RAM. RAM and GPU headroom estimates are conservative but cannot prevent another process allocating memory concurrently; recovery handles supported failures.

The full frozen-reference audit covers S on the 150 available 600-DPI test plates. M parity checks compare before/after M inference on representative empty/sparse/dense plates. This does not substitute for a new model-quality evaluation across all scanners or prove M universally more accurate than S.

## Reproduce

```bash
.venv/bin/python scripts/benchmark_adaptive_pipeline.py --repeats 6
.venv/bin/python scripts/benchmark_adaptive_pipeline.py --device cpu --threads 4 --repeats 2
.venv/bin/python scripts/benchmark_cpu_threads.py
.venv/bin/python scripts/verify_adaptive_pipeline.py
QT_QPA_PLATFORM=offscreen .venv/bin/python scripts/verify_adaptive_gui.py
QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests
```

The frozen-reference audit retains a nonzero exit code for any intermediate-feature mismatch, even if `matches_reference_outputs` passes. Check both fields and the recorded differences rather than interpreting the exit code as final-output parity alone.

## GUI and regression verification

All **88 tests passed** (final run 6.24 s). A real M GUI prediction on a disposable plate copy produced 18 detections, survived sidecar reload and Excel export with M provenance, and remained unchanged when selecting S. The tests also cover out-of-order completion, editing/undo responsiveness, cancellation, backup creation and device recovery. Independent source review found no remaining actionable issues after the regression fixes.
