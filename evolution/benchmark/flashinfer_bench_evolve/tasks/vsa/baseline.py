#!/usr/bin/env python3
"""FlashInfer BSA (``vsa_blackwell`` / ``vsa_blackwell_blk64``) baseline for VSA."""

from __future__ import annotations

import math

import torch
from flashinfer.cute_dsl.sparse import bsa_attn_blk64_fwd, bsa_attn_fwd


def _load_blk64_extension():
    """Load the blk64 C++ extension where flashinfer still ships one.

    flashinfer PR #4612 (in the nightlies from 0.6.18.dev20260902 on) replaced
    the C++/CUTLASS blk64 kernel with a CuTe-DSL implementation that compiles
    like the blk128 route, so there is no extension and nothing to load. On
    0.6.16/0.6.17, ``flashinfer.cute_dsl.sparse.blk64.loader`` walks up from
    its own file for a source checkout's ``3rdparty/cutlass``; the pip wheel
    ships CUTLASS under ``flashinfer/data/cutlass`` instead, so the walk fails
    with "Could not locate 3rdparty/cutlass". Point it at the packaged copy
    before the ninja build (~50 s once, then cached under
    ~/.cache/flashinfer/blk64_ext)."""

    try:
        from flashinfer.cute_dsl.sparse.blk64 import loader
    except ImportError:  # CuTe-DSL blk64: no extension to build
        return None

    try:
        loader._get_cutlass_root()
    except RuntimeError:
        from flashinfer.jit.env import CUTLASS_INCLUDE_DIRS

        loader._get_cutlass_root = lambda: CUTLASS_INCLUDE_DIRS[0].parent
    return loader.load_blk64_ext()


def _kernel_args(q, k, v, q2k_indices, kv_block_lens, block_size, sm_scale):
    """Turn the contract's metadata into the kernel's arguments.

    The BSA kernels take batch-major ``[1, S, H, D]`` tensors, the block list
    as int32 ``[1, H, MB, topk]`` and per-row block counts ``[1, H, MB]``;
    building those is the wrapper's ``plan()`` work, not the attention. The
    call itself mirrors ``BlockSparseAttentionWrapper.run()`` in flashinfer
    0.6.16 (``backend="vsa_blackwell"`` / ``"vsa_blackwell_blk64"``), the
    public-API arm flashinfer PR #4593 compares against: per-row counts,
    ``block_sizes=None`` and ``return_lse=True``, with the wrapper's
    ``block_sparse_num`` (2 / 1; ignored once counts are given). Loading the
    blk64 extension happens here too; the blk128 CuTe-DSL compile is keyed
    per shape class and happens on the first call."""

    block_size = int(block_size)
    scale = float(sm_scale) if sm_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    q2k = q2k_indices.to(torch.int32).unsqueeze(0).contiguous()
    counts = torch.full(q2k.shape[:3], q2k.shape[-1], dtype=torch.int32, device=q.device)
    batched = (q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
    if block_size == 64:
        _load_blk64_extension()
        # With counts and no block lengths, bsa_attn_blk64_fwd fills the
        # lengths with 64s itself (the wrapper's path), which masks the
        # phantom blocks it pads each row's list with. FastWan's partial
        # blocks (the PR's direct partial-block API rows) pass their lengths.
        block_sizes = None if kv_block_lens is None else kv_block_lens.to(torch.int32).contiguous()
        return (bsa_attn_blk64_fwd, *batched, q2k, 1, block_sizes, counts, scale)
    if block_size != 128:
        raise ValueError("BSA kernels support 64- or 128-token blocks only")
    if kv_block_lens is not None:
        raise ValueError("kv_block_lens is supported for 64-token blocks only")
    # The library default ``allow_empty_block_nums=True`` is the variant the
    # wrapper compiles. Every row here selects a block, so ``False`` would be
    # correct too and runs 1.7-2x faster, but it is not the wrapper's kernel.
    return (bsa_attn_fwd, *batched, q2k, 2, None, counts, scale)


def _launch(kernel, q, k, v, q2k_block_index, block_sparse_num, block_sizes, counts, scale):
    output, _lse = kernel(
        q,
        k,
        v,
        q2k_block_index,
        block_sparse_num=block_sparse_num,
        block_sizes=block_sizes,
        q2k_block_nums=counts,
        softmax_scale=scale,
        return_lse=True,
    )
    return output[0]


@torch.no_grad()
def prepare(q, k, v, q2k_indices, kv_block_lens, block_size, sm_scale):
    """Capture the kernel call in a CUDA graph outside timing.

    The harness times the CUPTI span of one call, first kernel start to last
    kernel end. blk128 is a single launch, but the blk64 extension issues 15
    (layout copies around the attention kernel), and eager launching leaves
    host-dependent gaps between them inside that span: 30-80 us over the
    kernel-sum on the S<=4096 rows, more than the kernels themselves at
    S=1024. Replaying a captured graph removes the gaps, so the timed span
    is the kernels alone. PR #4593's tables are synchronized end-to-end
    wall-clock times of the public API (host overhead included; its blk64
    baseline reads a flat ~250 us), so they are not the same quantity as any
    GPU span this harness reports. The output is bitwise equal to the eager
    call.

    The graph holds raw device pointers to everything the call reads, and the
    harness drops its own copy of the inputs once this returns, so the kernel
    arguments (including the block list and counts built here) are returned
    with it and stay alive as long as the graph does."""

    args = _kernel_args(q, k, v, q2k_indices, kv_block_lens, block_size, sm_scale)
    _launch(*args)  # extension load / CuTe-DSL compile and workspace allocation
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _launch(*args)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = _launch(*args)
    return graph, output, args


@torch.no_grad()
def run_prepared(graph, output, args):
    """Kernel-only dispatch: replay the graph captured by :func:`prepare`.

    ``args`` is not read here; it keeps the captured tensors alive."""

    graph.replay()
    return output


@torch.no_grad()
def run(q, k, v, q2k_indices, kv_block_lens, block_size, sm_scale):
    """Call FlashInfer's BSA block-sparse attention kernel for the block size (eager)."""

    return _launch(*_kernel_args(q, k, v, q2k_indices, kv_block_lens, block_size, sm_scale))
