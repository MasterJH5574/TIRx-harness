#!/usr/bin/env python3
"""cuBLAS baseline for the fp16 GEMM floor task."""

from __future__ import annotations

import torch


@torch.no_grad()
def prepare(a, b):
    """Allocate the output once so timing covers only the GEMM dispatch."""

    output = torch.empty((a.shape[0], b.shape[0]), dtype=a.dtype, device=a.device)
    return a, b, output


@torch.no_grad()
def run_prepared(a, b, output):
    torch.matmul(a, b.T, out=output)
    return output


@torch.no_grad()
def run(a, b):
    """Run the complete public task contract."""

    return torch.matmul(a, b.T)
