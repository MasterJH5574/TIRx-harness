# Kernel optimization manual

Technical guidance for recognizing and implementing common TIRx-kernel
optimizations.

| Layer | Path | Role |
|-------|------|------|
| **Technique guides** | this folder | How to apply a named technique well |
| **Empirically strong patterns** | [perf_checklist/](perf_checklist/) | Structural and source patterns that have repeatedly performed well |

## Technique guides

| Guide | Teaches |
|-------|---------|
| [swp-edge-cutting.md](swp-edge-cutting.md) | Cut software-pipeline period via per-iter DAG edge cutting |
| [playbook.md](playbook.md) | Implementation detail for recurring, empirically strong moves |
| [kernel-catalog.md](kernel-catalog.md) | Which bundled canonical kernel already uses technique X |

## Empirically strong patterns

These patterns have repeatedly worked well in prior TIRx kernels. Use them as
strong defaults and audit prompts, then keep only the choices that fit the
workload and win in measurement; they are not universal requirements.

| Section | File |
|---------|------|
| 1 Persistent kernel structure | [perf_checklist/section_1_persistent_kernel_structure.md](perf_checklist/section_1_persistent_kernel_structure.md) |
| 2 Static source patterns | [perf_checklist/section_2_static_source_patterns.md](perf_checklist/section_2_static_source_patterns.md) |
| 3 Scalar dataflow | [perf_checklist/section_3_scalar_dataflow.md](perf_checklist/section_3_scalar_dataflow.md) |
| 4 Dead work | [perf_checklist/section_4_dead_work.md](perf_checklist/section_4_dead_work.md) |
| 5 SMEM | [perf_checklist/section_5_smem.md](perf_checklist/section_5_smem.md) |
| 6 Register pressure | [perf_checklist/section_6_register_pressure.md](perf_checklist/section_6_register_pressure.md) |

Use `$tirx-profile-kernel` to validate the expected bottleneck change and
`$tirx-debug-kernel` when synchronization or shared state changes.
