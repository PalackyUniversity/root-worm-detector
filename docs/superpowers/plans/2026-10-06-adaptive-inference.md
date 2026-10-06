# Adaptive inference implementation plan

Goal: resource-aware parallel GUI inference, CPU recovery/tuning, selectable M/S and verified before/after benchmarks.
Spec: ../specs/2026-10-06-adaptive-inference-design.md
Execution: inline, following the user's repeated instruction to proceed. Preserve this working checkout's substantial existing changes; no reset, dependency upgrades or unrelated commits.

- [x] Model registry and sequential inference: test M/S identity, legacy S compatibility and single-decode orientation handling; add M checksum/weights, explicit model argument, cache runtime metadata, configurable CPU threads. Keep tile inference and refinement semantics.
- [x] Adaptive engine: test memory caps, retry classification, cancellation and result attribution with injectable executors; implement independent spawned workers, bounded scheduling, measured scaling, GPU memory backoff and CPU fallback without model changes.
- [x] GUI/persistence: test selector defaults, out-of-order completion, cancel/close and sidecar provenance; integrate coordinator signals, retain editing/undo behavior, explicit rerun with backups, export model provenance.
- [x] Benchmark/verification: capture GPU M original baseline; CPU baselines already saved. Compare sequential/parallel M/S, CPU thread layouts, original saved CPU outputs, 150-plate S reference. Exercise low-memory/backend failures in tests. Write measured report and usage docs.
- [x] Independent final review and targeted corrections; run complete GUI unit suite and report actual limitations.

Review focus: corrupt inputs must not trigger silent fallback; no manual edit overwrites on selector change; cancellation drains only dispatched jobs; failed jobs not lost or duplicated during pool replacement; CPU/GPU outputs compared against the appropriate device/model reference.

Execution notes: CPU starts with resource-bounded four-thread process groups to avoid spending an entire slow plate ramping. Legacy direct prediction API defaults remain S for research-reference compatibility; the GUI explicitly selects M. No precision/tile-batch changes adopted. Physical tests cover NVIDIA CUDA and CPU; other hardware recovery is simulated. Results: outputs/adaptive_inference_benchmark/REPORT.md.
