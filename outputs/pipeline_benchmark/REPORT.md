# Deployed bulk prediction benchmark — 2026-10-06

Benchmarked the actual `InferencePipeline.predict` path plus sidecar saving. Production inference code, models, reference outputs and source-image sidecars were not modified. All variants are temporary patches in the benchmark process.

## Method

Six 5WPI TIFF plates selected across the detection-count distribution: 0, 6, 13, 20, 28 and 48 females (115 total). All are 600 DPI, with 320-pixel slices and 20% overlap. Model, confidence, full-image prediction, precision, merging, refiner and classifier are unchanged. RTX 5070 Ti. Two measured repetitions per variant after warm-up; variant order reverses on repetition two. A separate one-repetition follow-up records internal YOLO timings and matched per-object differences.

Existing desktop applications and the GUI were left running, so GPU contention and normal timing noise are possible. This is warmed inference, not a cold-storage benchmark. Sidecars were written to a temporary directory, not alongside input images; storage-specific latency is not measured. GUI repaint, event dispatch, initial file import, and export are excluded. The six-image sample does not establish quality parity across all plates, DPI values or formats.

## Mean wall time per image

| Stage | Seconds | Share |
|---|---:|---:|
| Detection: decoding, slicing, YOLO, mask conversion and merge | 1.7663 | 94.2% |
| Second image decode / BGR conversion | 0.0415 | 2.2% |
| Crop preparation | 0.0026 | 0.1% |
| U-Net forward pass | 0.0096 | 0.5% |
| Refined components, features and mask encoding | 0.0113 | 0.6% |
| Five-model niceness classifier | 0.0087 | 0.5% |
| Other: DPI, transfers, tensor preparation, records, runtime metadata | 0.0228 | 1.2% |
| Sidecar serialization and write | 0.0121 | 0.6% |
| **Total** | **1.8748** | **100%** |

Model initialization: 1.15 s once (excludes Python startup/imports). First empty plate: 2.39 s, including cold inference work. Subsequent plates reuse model instances.

Nested detector timings must not be added again to the stage table. The primary run averages 1.611 s in all inference calls (about 1.562 s for tiles and 0.049 s for the standard full-image pass), 0.013 s in prediction conversion, and 0.007 s in merging. Each sampled plate has 132 tiles plus one full-image prediction.

Internal Ultralytics timers from the follow-up run:

| Variant | Preprocessing | Network inference | Postprocessing |
|---|---:|---:|---:|
| baseline | 0.258 s | 1.223 s | 0.047 s |
| decode_once | 0.260 s | 1.224 s | 0.044 s |
| batch4 | 0.249 s | 0.369 s | 0.030 s |
| batch8 | 0.252 s | 0.336 s | 0.028 s |

These are nested instrumented library timings, not isolated kernel throughput; all per-image calls are summed. Batching reduces repeated small inference calls. CPU feature extraction and the niceness classifier are already minor costs.

## Candidate speedups

| Variant | Mean seconds | Speedup | Time saved |
|---|---:|---:|---:|
| baseline | 1.875 | 1.00× | 0.0% |
| decode_once | 1.745 | 1.07× | 6.9% |
| batch4 | 0.845 | 2.22× | 54.9% |
| batch8 | 0.800 | 2.34× | 57.4% |
| batch16 | 0.827 | 2.27× | 55.9% |

Every batch variant also includes shared TIFF decoding. Compare it against `decode_once` to isolate the batching benefit. SAHI normally reads the path for slicing, standard full-image prediction, and result construction; the refiner reads it again. Passing one decoded RGB image through these stages removes repeated decoding.

**Shared decoding is the conservative first change:** every tested final contour, detector score, centre, packed mask, area, niceness probability and decision matched baseline. Strict comparison of all intermediate feature floats failed on two primary runs for both baseline and shared decoding; final outputs remained identical. The follow-up showed zero final differences. This experiment only accepts identity EXIF orientation; production integration must preserve the separate reader semantics for rotated images and other formats and rerun the full saved-reference audit.

**Tile batching offers about 2.2–2.3× throughput, but is not lossless.** All sampled counts and nice decisions matched; some centres, masks, areas and probabilities changed. Per-object matching in the follow-up gives:

| Variant | Changed masks / 115 | Max centre shift | Max area difference | Max relative area difference | Max probability difference |
|---|---:|---:|---:|---:|---:|
| batch4 | 1 | 0.5 px | 0.000988 mm² | 0.373% | 0.0000510 |
| batch8 | 2 | 0.5 px | 0.001111 mm² | 0.373% | 0.0008608 |

Batch 8 changed a detector score by as much as 0.00515. Unchanged decisions on this sample do not guarantee unchanged decisions for objects near either threshold. Batch 16 was slower than batch 8 in the primary run. Keep the current tile-by-tile configuration for strict reproducibility. Before promoting batching, compare all saved per-object outputs and evaluate count/segmentation quality on the established validation data. Do not silently change overlap, confidence, DPI scaling, precision, model, or standard full-image prediction to obtain speed.

## Reproduce

```bash
.venv/bin/python scripts/benchmark_pipeline.py
.venv/bin/python scripts/benchmark_pipeline.py --repeats 1 --variants baseline decode_once batch4 batch8 --output outputs/pipeline_benchmark/detail
```

Primary data: `summary.json`, `measurements.json`, `run.log`. Follow-up data: `detail/summary.json`, `detail/measurements.json`, `detail.log`. The final script additionally records internal YOLO times and matched numerical differences; those fields were added after the primary run.

Runtime: torch 2.11.0+cu128, ultralytics 8.4.115, sahi 0.12.5, numpy 2.5.3, opencv-python 5.0.0.93.
