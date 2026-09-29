# External source dependencies

| Submodule | Used for |
| --- | --- |
| `tvm-rust-ext` | Building the Rust frontend of `tirx_harness`; installed tools do not read this checkout |

Keep the committed submodule revision when initializing it.

`tirx-kernels` is a separate kernel/DSL repository, prepared by the installer or
pip environment. Benchmark implementations and data live directly in
`evolution/benchmark/flashinfer_bench_evolve/`. Downloaded wiki reference repositories live under the wiki skill's
`references/repos/`; they are not additional runtime submodules here.
