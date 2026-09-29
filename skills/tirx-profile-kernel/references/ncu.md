# NCU Kernel Profiling

Profile TIRx kernels with `ncu` and interpret the results. Use an explicit
launch contract: application argv, working directory, environment, and the
kernel symbol or regex to capture. User-provided values are authoritative.
When `$tirx-profile-kernel` is active and a field is omitted, inspect the
current task for its existing benchmark, runner, environment, and artifact
convention; do not invent a repository layout or artifact filename.

Obtain kernel symbols from the compiled module or
`tirx_harness.dump_kernel.dump_module()`. If the symbol is unknown, capture once
without `-k`, inspect the launched kernels, then repeat with an exact filter.

## When to Use

NCU answers questions that cannot be inferred reliably from source: real stall
reasons, achieved occupancy, `launch__registers_per_thread`, tensor-core
utilization, and DRAM/L2 traffic. It is especially useful for:

- Perf regression after a "small" change (setmaxnreg tune, BUF_DEPTH tweak, scalar param) — NCU pinpoints spill / occupancy cap / stall-reason shift faster than re-reading the IR
- Warpgroup count or barrier-protocol change — verify the stall breakdown shifted the way you expected
- Before a memory-hierarchy refactor — confirm `dram__throughput` + L2 hit rate actually indicate memory-bound before investing in SMEM reuse or TMA multicast

When SOL/MemoryWorkload reports a high estimated speedup from shared-bank
conflicts or uncoalesced global memory, use the source-attribution capture below
before selecting a swizzle or coalescing change. An aggregate percentage does
not identify the responsible buffer.

For latency-bound kernels with low SM and DRAM utilization and high
`No Eligible`, use the source-hotspot flow below to localize the cause.

## Launch contract

Resolve every field before capture:

| Input | Meaning |
|---|---|
| `<kernel-regex>` | Exact kernel name or NCU regex |
| `<launch-command> [args...]` | Deterministic command that reaches the target launch |
| `cwd` | Explicit working directory for that command |
| `env` | Required environment, including any switch that disables another profiler |
| `<report-base>` | Caller-selected output path without `.ncu-rep` |

Only one CUPTI subscriber can be active. If the launch normally starts another
profiler, use that integration's explicit disable hook before running NCU.
Timing printed during replay is not a benchmark result.

Use `--launch-count 1` when the command repeats the same deterministic target
kernel; otherwise NCU replays every matching launch and multiplies profile
time without adding signal.

```bash
ncu -k <kernel-regex> --launch-count 1 --set full \
    -f -o <report-base> <launch-command> [args...]
```

## Source-Line Attribution (shared bank conflicts + uncoalesced global)

When the SOL or MemoryWorkloadAnalysis section reports a high est-speedup from
excessive shared-load/store wavefronts (bank conflicts) or excessive global
sectors (uncoalesced access), the aggregate number does not say *which* buffer
or access is at fault. Capture the per-source-line tables to localize it.

**1. Profile with line info + source import.** Ask the caller's compiler
integration to enable line info and retain the generated CUDA/cubin artifacts.
Add `--import-source yes` to embed available source in the report. Without line
info, attribution reaches only SASS.

```bash
ncu -k <kernel-regex> --launch-count 1 --set full --import-source yes \
    -f -o <report-base> <launch-command> [args...]
```

**2. Rank the offending instructions (SASS view).** Only the SASS-correlated
`smsp__sass_*` counters are attributed per-instruction. Pass ONE metric per run
(each `--metrics` adds a column; the awk reads only `$NF`). The same template,
swapping the metric:

```bash
ncu -i <report-base>.ncu-rep --page source --print-source sass \
    --metrics <METRIC> 2>/dev/null \
  | grep -E '^0x' | sed -E 's/[[:space:]]+/ /g' \
  | awk '{m=$NF+0; if(m>0){inst=$2; for(i=3;i<NF;i++) inst=inst" "$i; print m"\t"inst}}' | sort -rn | head
```

| Symptom | `<METRIC>` |
|---|---|
| Shared bank conflicts, LDSM matrix load (dominant — swizzled MMA operand) | `smsp__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm` |
| Shared bank conflicts, STSM matrix store | `smsp__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_stsm` |
| Shared bank conflicts, cp.async (LDGSTS) | `smsp__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts` |
| Uncoalesced global load (sectors/req ≫ coalesced 4) | `smsp__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ld` |
| Uncoalesced global store | `smsp__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_st` |
| Uncoalesced global (low bytes/sector) | `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld` |

The SASS operand names the access — `LDS.U16` on bf16 = scalar un-vectorized
shared load (vectorize/swizzle); `LDG.E` strided = uncoalesced global load.

Caveats:
- Confirm a metric on a busy GPU with `ncu --query-metrics --chip gb100` (reads the B200 metric DB without a device; the plain `--query-metrics` needs a free GPU).
- Plain `LDS`/`STS` (non-LDSM) bank conflicts have no per-source `smsp__sass_*` counter on this build — only device-level `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld` / `_op_st` (Details page), then localize by SASS operand.

**3. Map to the CUDA source line.** The report embeds the resolved `.cu`
(verify the file named in the report). Open the report source
page in the GUI — `ncu-ui <report-base>.ncu-rep`, Source page — to read the
metric against each `.cu` line. CLI `--print-source cuda` shows the interleaved
source but does not print the per-line metric in a parseable column (GUI only);
for a CLI-only mapping, match the SASS operand (buffer name + offset) against
the caller-provided generated CUDA artifact.

## Source Hotspot (stall sampling → line → knob)

When the kernel is slower than its work should cost and SOL says latency-bound
(SM and DRAM both low, "No Eligible" high) with no single aggregate cause:
localize by warp-stall sampling, then trace the hot construct to its source
knob.

**1. Capture with line info** (`--set full` includes the sampling sections):

```bash
ncu -k <kernel-regex> --launch-count 1 --set full \
    -f -o <report-base> <launch-command> [args...]
```

**2. Rank instructions by stall samples.**

```bash
ncu --import <report-base>.ncu-rep --page source --csv > <source-csv>
# skip the 1-line banner; per-SASS-instruction columns include
# "Warp Stall Sampling (All Samples)" and "Instructions Executed"
```

Set aside the idle-warp artifacts first — `NANOSLEEP` (mbarrier spin), `EXIT`,
and `S2R` right after a barrier collect huge sample counts but are the
*waiters*, not the cause. The real hotspot is the top-stalled cluster of
ordinary instructions; a backward `BRA` inside it with a huge exec count = a
hot rolled loop.

**3. Map the cluster to source.** Run `nvdisasm --print-line-info` on the
caller-provided cubin; it emits `//## File ... line N` per SASS region. Read
the corresponding generated CUDA artifact at those lines.

**4. Trace to the producing construct and toggle it.** Every property of the
hot region's generated code is produced by something in the caller's kernel
source or compiler configuration — an explicit kwarg/annotation, an API
choice, or a default. Find it and A/B-toggle it in an isolated candidate before
hand-restructuring.

## Reference Docs

For full details, use NVIDIA's official
[Nsight Compute documentation](https://docs.nvidia.com/nsight-compute/).

## Quick Start

```bash
# Basic overview (SOL, occupancy, launch stats)
ncu -k <kernel-regex> --set basic <launch-command> [args...]

# Detailed analysis (+ memory/compute workload, source counters, roofline)
ncu -k <kernel-regex> --set detailed <launch-command> [args...]

# Specific metrics only
ncu -k <kernel-regex> \
    --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \
    <launch-command> [args...]

# Save a caller-named report for GUI inspection
ncu -k <kernel-regex> -o <report-base> <launch-command> [args...]
ncu-ui <report-base>.ncu-rep
```

## Key Flags

| Flag | Purpose | Default |
|---|---|---|
| `-k <kernel-regex>` | Filter to the caller-selected kernel (short for `--kernel-name`) | all kernels |
| `--set S` | Section set: `basic`, `detailed`, `full`, `roofline` | `basic` |
| `--section S` | Specific section (repeatable) | — |
| `--metrics M1,M2,...` | Specific metrics (comma-separated) | — |
| `-c N` / `--launch-count N` | Capture N kernel launches | all |
| `-s N` / `--launch-skip N` | Skip N matching launches before capture | 0 |
| `-o FILE` | Save `.ncu-rep` for GUI | — |
| `--csv` | Machine-readable output | off |
| `--page {details,raw,source}` | Output page (`raw` = flat metric table) | `details` |
| `--clock-control {base,none}` | Lock clocks for reproducibility | `base` |
| `--cache-control {all,none}` | Flush caches between replay passes | `all` |
| `--replay-mode M` | `kernel` (default), `application`, `range` | `kernel` |
| `--target-processes all` | Profile child processes too | `all` |

## Section Sets

Discover what's available: `ncu --list-sets`, `ncu --list-sections`.

| Set | Sections included | Approx. metrics | When to use |
|---|---|---|---|
| `basic` | SpeedOfLight, Occupancy, LaunchStats, WorkloadDistribution | ~213 | First look, quick iteration |
| `detailed` | + MemoryWorkloadAnalysis, ComputeWorkloadAnalysis, SourceCounters, SpeedOfLight_RooflineChart, Tile | ~996 | Investigating a bottleneck |
| `full` | + InstructionStats, SchedulerStats, WarpStateStats, PmSampling, MemoryWorkloadAnalysis_Tables, all roofline charts | ~8054 | Deep dive (slow — many replay passes) |
| `roofline` | SpeedOfLight + all roofline chart variants | ~6679 | Compute vs memory bound |

## Replay Modes

ncu collects metrics in multiple passes over the kernel. The replay mode determines how those passes execute.

| Mode | How it works | Best for |
|---|---|---|
| `kernel` (default) | Saves kernel memory, replays kernel N times | Most cases. Overhead grows with kernel's written memory. |
| `application` | Re-runs entire program N times | Kernels with host dependencies, or when memory save/restore is too expensive |

For TIRx kernels, the default `kernel` replay is almost always correct. Use `application` only if the kernel has host-side control flow dependencies (rare).

**Important:** ncu flushes caches between replay passes by default (`--cache-control all`). This means each metric pass sees cold caches. If you want to measure warm-cache behavior (e.g., persistent kernels reusing L2), use `--cache-control none` — but only with `--replay-mode application` for correct memory state.

---

## Reading NCU Metric Names

Metrics follow the pattern: `unit__quantity.rollup.submetric`

**Units:** `sm` (streaming multiprocessor), `smsp` (SM sub-partition), `dram` (HBM), `l1tex` (L1/texture), `lts` (L2 slice), `fe` (frontend)

**Rollups:** `.sum`, `.avg`, `.min`, `.max` — aggregation across unit instances

**Submetrics:**

| Suffix | Meaning |
|---|---|
| `.per_cycle_elapsed` | Per GPU clock cycle (including idle) |
| `.per_cycle_active` | Per cycle the unit was active |
| `.pct_of_peak_sustained_elapsed` | % of theoretical max throughput (over total time) |
| `.pct_of_peak_sustained_active` | % of theoretical max (over active time only) |
| `.per_second` | Rate in operations/second |

**Which suffix to use:**
- For **utilization/bottleneck analysis**, use `.pct_of_peak_sustained_elapsed` — this tells you what fraction of the GPU's total capacity is used, including idle time.
- `.pct_of_peak_sustained_active` excludes idle time, so it'll read higher for kernels with low occupancy. Useful for understanding efficiency *when the unit is actually working*.

---

## How to Read NCU Results

### Step 1: Speed of Light — Where's the Bottleneck?

The SOL (Speed of Light) section in `--set basic` shows:

| Metric | What it tells you |
|---|---|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | % of peak compute used |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | % of peak HBM bandwidth used |
| `gpu__time_duration.sum` | Absolute kernel time (nsec) |

**Decision tree:**
- SM high, DRAM low → **Compute bound**. Reduce instruction count, improve ILP, check tensor core utilization.
- SM low, DRAM high → **Memory bound**. Reduce traffic, improve locality, increase reuse. If the est-speedup is attributed to shared bank conflicts or uncoalesced global, localize it with Source-Line Attribution before picking a fix.
- Both low → **Latency bound**. Stalls, poor occupancy, or synchronization bottleneck. Drill into warp stalls.
- Both high → **Near peak**. Micro-optimizations only.

### Step 2: Tensor Core Utilization (GEMM-specific)

```
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed
```

The most important single metric for GEMM. Measures what fraction of time tensor cores are doing useful work.

| Value | Interpretation | Action |
|---|---|---|
| >80% | Excellent | Kernel is well-tuned |
| 50-80% | Good | Check if data delivery (TMA/SMEM) is the bottleneck |
| 20-50% | Significant stalls | Check warp stalls, pipeline drain, barrier overhead |
| <20% | Broken | Pipeline not working, wrong tile sizes, or scheduling bug |

If low: check `dram__throughput` and `lts__throughput` — are tensor cores starved for data?

### Step 3: Occupancy

```
sm__warps_active.avg.pct_of_peak_sustained_active    — Achieved occupancy
launch__occupancy_limit_registers                     — Occupancy cap from registers
launch__occupancy_limit_shared_mem                    — Occupancy cap from SMEM
launch__occupancy_limit_warps                         — Occupancy cap from block size
launch__occupancy_limit_barriers                      — Occupancy cap from barrier count
```

The `launch__occupancy_limit_*` metrics show what's capping occupancy. The lowest one is the binding constraint.

**For persistent GEMM kernels:** low occupancy (1-2 CTAs/SM) is expected and by design. The kernel hides latency through software pipelining (multi-stage TMA prefetch), not through warp-level parallelism. Don't optimize for higher occupancy here.

### Step 4: Memory Traffic

```
dram__bytes_read.sum      — Total HBM bytes read
dram__bytes_write.sum     — Total HBM bytes written
lts__t_bytes_lookup_hit.sum   — L2 bytes served from cache
lts__t_bytes_lookup_miss.sum  — L2 bytes that missed (went to HBM)
```

**Compute traffic amplification:**
- GEMM(M, N, K) theoretical minimum: read = `M*K*sizeof(A) + N*K*sizeof(B)`, write = `M*N*sizeof(D)`
- For FP8 with block scale factors, add `(M*K + N*K) / 128 * sizeof(sf)`
- `actual / minimum` = traffic amplification. Values >1.5x indicate poor tiling or scheduling.

**L2 hit rate** = `hit / (hit + miss)`. High hit rate → data reuse working (TMA multicast, tile reuse across CTA groups). Low hit rate → tiles reading cold data from HBM every time.

### Step 5: Launch Config Sanity Check

```
launch__grid_size                       — Grid dimensions
launch__block_size                      — Threads per CTA
launch__registers_per_thread            — Register count
launch__shared_mem_per_block_allocated  — SMEM bytes per CTA
```

Verify against expectations:
- **Grid size**: `ceil(M/BLK_M) * ceil(N/BLK_N)` for non-persistent, or SM count for persistent kernels.
- **Block size**: `num_warpgroups * 128` (each warpgroup = 4 warps = 128 threads).
- **Registers**: Blackwell MMA kernels typically 200-256. Higher → possible register spill or unnecessary live vars.
- **SMEM**: Should match pipeline buffer size. Much higher → check compiler-inserted padding.

### Step 6: Warp Stalls (when latency-bound)

Use `--set detailed` to get WarpStateStats, then check top stall reasons:

| Stall metric (`smsp__warps_issue_stalled_<reason>`) | Meaning | Fix |
|---|---|---|
| `long_scoreboard` | Waiting for global memory (HBM/L2) | Prefetch earlier, increase pipeline depth |
| `short_scoreboard` | Waiting for SMEM or math result | Check SMEM bank conflicts (localize via Source-Line Attribution), improve ILP |
| `barrier` | Waiting on `__syncthreads()` / mbarrier | Reduce barrier frequency, overlap work |
| `membar` | Waiting on memory fence (`__threadfence`) | Minimize fence scope |
| `mio_throttle` | Memory I/O back-pressure | Reduce concurrent memory requests |
| `math_pipe_throttle` | Math pipe back-pressure | Reduce instruction density |
| `lg_throttle` | Local/global memory back-pressure | Reduce concurrent GMEM accesses |
| `not_selected` | Eligible but not issued (scheduler full) | Not a problem — means other warps ran |
| `wait` | Generic wait state | Check async operations, pipeline stalls |
| `imc_miss` | Instruction cache miss | Kernel too large, poor code locality |

---

## Common Workflows

### A/B Comparison

```bash
# Capture baseline and candidate with identical flags and inputs
ncu -k <kernel-regex> --set basic <baseline-launch-command> [args...]
ncu -k <kernel-regex> --set basic <candidate-launch-command> [args...]

# Or targeted metrics for quick comparison
M=gpu__time_duration.sum,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed
ncu -k <kernel-regex> --metrics $M <baseline-launch-command> [args...]
ncu -k <kernel-regex> --metrics $M <candidate-launch-command> [args...]
```

Store the reports wherever the caller's artifact contract specifies.

### Diagnosing a Regression

1. `--set basic` on both versions → confirm `gpu__time_duration` regression.
2. SM throughput dropped? → compute regression. Check tensor core util, instruction count (`--section SourceCounters`).
3. DRAM throughput up? → more memory traffic. Check `dram__bytes_read/write`, L2 hit rate.
4. Neither? → latency regression. Run `--set detailed`, check WarpStateStats for new stall patterns.

### Sweeping Problem Sizes

Ask the caller's launch adapter to produce one deterministic command per shape,
then profile each with the same kernel filter and metric set. Keep the launch
environment and all non-shape inputs fixed.

Tensor core utilization dropping at small K → too much pipeline startup/drain overhead relative to steady-state compute.

### Saving for GUI Analysis

```bash
ncu -k <kernel-regex> --set detailed -o <report-base> \
    <launch-command> [args...]
ncu-ui <report-base>.ncu-rep
# GUI gives roofline charts, source correlation, memory hierarchy diagrams
```

---

## Quick Metric Reference

### Compute
| Metric | Description |
|---|---|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | Overall SM utilization |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed` | Tensor core utilization |
| `sm__inst_executed.sum` | Total instructions executed across all SMs |
| `sm__inst_executed_pipe_tensor_op_hmma.sum` | Tensor core instructions executed |

### Memory
| Metric | Description |
|---|---|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | HBM bandwidth utilization |
| `dram__bytes_read.sum` | Total bytes read from HBM |
| `dram__bytes_write.sum` | Total bytes written to HBM |
| `l1tex__throughput.avg.pct_of_peak_sustained_elapsed` | L1/TEX cache throughput |
| `lts__throughput.avg.pct_of_peak_sustained_elapsed` | L2 cache throughput |
| `lts__t_bytes_lookup_hit.sum` | L2 cache hit bytes |
| `lts__t_bytes_lookup_miss.sum` | L2 cache miss bytes |

### Occupancy & Launch
| Metric | Description |
|---|---|
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Achieved occupancy |
| `launch__occupancy_limit_registers` | Occupancy cap: registers |
| `launch__occupancy_limit_shared_mem` | Occupancy cap: shared memory |
| `launch__occupancy_limit_warps` | Occupancy cap: block size |
| `launch__occupancy_limit_barriers` | Occupancy cap: barrier count |
| `launch__grid_size` | Grid dimensions |
| `launch__block_size` | Threads per CTA |
| `launch__registers_per_thread` | Registers per thread |
| `launch__shared_mem_per_block_allocated` | SMEM per CTA (bytes) |

### Stalls (--set detailed)
| Metric | Description |
|---|---|
| `smsp__warps_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_active` | Waiting for GMEM/L2 |
| `smsp__warps_issue_stalled_short_scoreboard.avg.pct_of_peak_sustained_active` | Waiting for SMEM/math |
| `smsp__warps_issue_stalled_barrier.avg.pct_of_peak_sustained_active` | Waiting on barrier |
| `smsp__warps_issue_stalled_membar.avg.pct_of_peak_sustained_active` | Waiting on memory fence |
| `smsp__warps_issue_stalled_mio_throttle.avg.pct_of_peak_sustained_active` | Memory I/O back-pressure |
| `smsp__warps_issue_stalled_math_pipe_throttle.avg.pct_of_peak_sustained_active` | Math pipe back-pressure |
| `smsp__warps_issue_stalled_lg_throttle.avg.pct_of_peak_sustained_active` | Local/global back-pressure |
| `smsp__warps_issue_stalled_wait.avg.pct_of_peak_sustained_active` | Generic wait |
| `smsp__warps_issue_stalled_imc_miss.avg.pct_of_peak_sustained_active` | Instruction cache miss |
| `smsp__warps_issue_stalled_not_selected.avg.pct_of_peak_sustained_active` | Eligible but not issued |
