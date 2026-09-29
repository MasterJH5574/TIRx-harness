# GPU engineering blogs

The bundled text snapshots are grouped by publisher. Root-relative images,
styles, and other site assets referenced by an article remain external:

- `colfax/`: CUTLASS/CuTe, Hopper and Blackwell GEMM, TMA, layouts, software
  pipelining, attention, and low-precision formats. Upstream collection:
  https://research.colfax-intl.com/
- `thunderkittens/`: ThunderKittens, attention, megakernels, Blackwell,
  persistent scheduling, and multi-GPU work. Upstream collection:
  https://hazyresearch.stanford.edu/blog

Search by concept or opcode:

```bash
rg -n 'tcgen05|Tensor Memory|CTA pair' references/blogs
rg -n 'persistent|software pipelin|TMA' references/blogs
```

Use these posts to understand a pattern, then verify hardware claims against
the live manuals and exact TIRx usage against the installed TIRx source and
canonical kernel package.
