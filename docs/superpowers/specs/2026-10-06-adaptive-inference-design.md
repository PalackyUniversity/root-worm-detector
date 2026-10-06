# Adaptive bulk inference and selectable detectors

## Requested outcome

Integrate image-level parallel prediction into the GUI, maximize measured throughput within available GPU and host memory, provide robust CPU fallback, optimize CPU and sequential execution, and deliver before/after benchmarks. Add trained M and S detectors with M selected by default. Preserve each detector's prediction behavior; optimization is not permission to change thresholds, precision, overlap, or merge rules.

## Existing evidence and model choice

The current GUI uses v8-s-mr4 best.pt at confidence 0.35. The frozen 150-plate reference contains 2,706 detections, 2,705 classified females, and 1,439 nice females at threshold 0.3. Three independent GPU processes plus shared decoding achieved approximately 2.53x sequential throughput and matched all final reference outputs at CSV precision. Small intermediate edge-feature roundoff also occurs in sequential runs.

Deploy v8-m-mr4 best.pt as M and retain v8-s-mr4 best.pt as S. Their per-run selection.json files both select confidence 0.35. M has validation count MAE 2.106; S has 2.362. In the existing lab comparison, S has lower MAE (1.260 versus M's 1.341); do not label M as universally more accurate. Bundle/checksum M alongside S. Keep the existing seeded U-Net, GB-9 ensemble and niceness threshold 0.3.

## Architecture

Use a Qt coordinator thread for scheduling and a bounded persistent pool of spawned inference processes. Each process owns its models and never touches Qt widgets. Jobs contain path, model ID and immutable execution settings; completed results are keyed to the original image and run ID. Keep the pool reusable while its model/device/thread configuration is unchanged. Shutdown releases models, processes, and accelerator memory. Preserve ui -> logic -> config dependencies.

A logic-layer execution policy detects available accelerators, free device memory, usable CPU cores and available host RAM. Begin conservatively with one worker, measure actual peak memory, and increase workers only within reserved memory headroom and CPU/RAM limits. Use completed real jobs for warm-up/calibration; never discard useful predictions. Favor measured batch throughput over launching the maximum number of processes. A short batch should not pay for unnecessary calibration or unused workers. No fixed RTX-specific worker count.

## Device compatibility and recovery

Prefer a functioning accelerator supported by the installed PyTorch/Ultralytics combination. CUDA includes ROCm only where that installed runtime actually supports it. Probe MPS when present; unsupported accelerator/runtime combinations use CPU. Do not claim native acceleration for every GPU vendor from the NVIDIA machine used here. Record actual device in result provenance and show CPU fallback in GUI status.

Reserve free memory for the desktop and concurrent applications; use free rather than nominal VRAM. On recoverable accelerator initialization or allocation failure, stop dispatch, retain already completed results, release failed workers, and retry unfinished jobs with fewer workers. Retry at one GPU worker before CPU fallback where appropriate. Bound retries and never repeat a successful save. A broken accelerator context requires a fresh process. Missing/corrupt weights, malformed input images and ordinary code errors must remain actionable errors rather than being hidden as CPU fallback. Never silently replace M with S.

For very dense images, bound outline-refiner memory. Any change to refiner batching must be compared with the unchanged refiner on the same device; if exact final-output parity cannot be maintained, prefer CPU retry of that image rather than silently modifying its measurements. RAM limits must also bound prefetch and outstanding result buffers.

## Sequential and CPU execution

Decode TIFF pixels once where reader semantics are identical; preserve existing orientation/color handling for other formats. Keep DPI-aware slice sizes, 20% overlap, standard full-image prediction, GREEDYNMM/IOS, tile batch size one, FP32, refiner threshold and classifier unchanged. Cache model/runtime metadata per worker instead of recomputing immutable information per image where profiling supports it.

Benchmark CPU layouts including one multithreaded worker and multiple workers sharing a total core budget. Set Torch/OpenCV/BLAS thread counts per process to avoid oversubscription and leave the UI responsive. Choose a bounded configuration based on usable cores and host RAM, with one-worker fallback. Retain only optimizations whose measured benefit and per-model output checks justify them; no quantization, FP16, model substitutions or unvalidated tile batching.

## GUI and persistence

Add a clearly labeled M/S selector beside prediction controls. M is the initial default; changing selection applies to future predictions and is disabled during an active batch. A separate explicit re-predict action allows rerunning completed images with the selected model; preserve a backup of replaced saved annotations. Switching the selector alone never deletes results or manual edits. Existing S sidecars remain recognized and visibly identified as S instead of being mislabeled or invalidated merely because M is now default.

Each result/sidecar records model ID, detector checksum, confidence, pipeline ID and actual execution device. Compatibility checks accept the recognized M and S pipelines with the correct metadata. Export preserves per-image provenance; mixed-model images must not masquerade as one model. Keep manual outline classification using the shared classifier.

Results may complete out of order. Apply each to its own image on the GUI thread, preserve undo and selection on other images, and keep browsing and editing completed images responsive. Mark only dispatched images as processing. Cancellation stops further dispatch, safely finishes/drains in-flight jobs, and retains completed results. Closing while active follows the same cancellation/shutdown path and leaves no child processes behind. Atomic sidecar writes remain intact.

## Validation and benchmarks

Capture the current sequential S baseline and an unchanged sequential M baseline before integration. Benchmark each model on the same plates/settings for GPU and CPU, reporting startup separately from warmed batch throughput, worker/thread settings, GPU/host memory, and per-stage timings. Include representative detection densities and 600/720/1200 DPI input where available; distinguish measured hardware from simulated fallback tests.

Compare optimized S against the 150-plate frozen outputs. Compare optimized M against newly captured unchanged sequential M outputs, not S outputs. Compare CPU optimizations against an unchanged CPU baseline; CPU/GPU equality is a separate measured question. Report counts, matched centres/scores, refined masks/areas, classifier probabilities and decisions, and intermediate roundoff separately.

Test low/unknown VRAM, zero accelerator availability, initialization failure, out-of-memory at startup and mid-image, broken workers, CPU retry exhaustion, cancellation, close during work, out-of-order results, model switches, retained S sidecars, manual edits/undo and export provenance. Use injected failures for GPUs not physically available. Run the existing GUI and persistence tests plus real GPU/CPU smoke runs.

## Scope and limits

Do not upgrade inference libraries, change training/model selection, retrain models, or alter scientific thresholds. Preserve unrelated local changes. Benchmarks on this RTX 5070 Ti and i7-9700KF cannot prove performance or native acceleration on every GPU/CPU; robust capability checks and tested fallback behavior provide portability. There is no claim of universal optimal concurrency.
