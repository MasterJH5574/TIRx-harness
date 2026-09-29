::::::::::::::::::::::::::: {.wp-block-group .is-layout-flow .wp-block-group-is-layout-flow role="main" style="margin-top:var(--wp--preset--spacing--50)"}
::: {.wp-block-group .has-global-padding .is-layout-constrained .wp-block-group-is-layout-constrained}
# Delivering 1 PFLOP/s of Performance with FP8 FlashAttention-2 {#delivering-1-pflops-of-performance-with-fp8-flashattention-2 .wp-block-post-title style="margin-bottom:var(--wp--preset--spacing--40);"}
:::

:::::::: {.entry-content .wp-block-post-content .has-global-padding .is-layout-constrained .wp-block-post-content-is-layout-constrained}
We recently released an
[update](https://github.com/ColfaxResearch/cutlass-kernels/tree/master/src/fmha-pipeline){data-type="link"
data-id="https://github.com/ColfaxResearch/cutlass-kernels/tree/master/src/fmha-pipeline"}
to our FlashAttention-2 forward pass implementation on NVIDIA Hopper^™^
architecture that incorporates a number of new optimizations and
improvements, including a software pipelining scheme and FP8 support. In
this article, we will explain a challenge with achieving *layout
conformance* of register fragments for WGMMA instructions that we
encountered in the process of fusing back-to-back mixed-precision GEMMs
with FP8 operands and FP32 accumulator. After explaining why this
problem surfaces for FP8 but not FP16 precision operands, we'll describe
a performant method to achieve layout conformance through using a
concise combination of two NVIDIA^®^ CUDA^®^ intrinsics -- namely, the
byte permute and shuffle sync instructions. Our solution takes some
inspiration from a [blog
post](https://blog.research.google/2024/01/mixed-input-matrix-multiplication.html){data-type="link"
data-id="https://blog.research.google/2024/01/mixed-input-matrix-multiplication.html"}
of Manish Gupta that studied a similar problem in the context of
mixed-input GEMMs. Furthermore, we'll discuss another complication that
arises with the majorness of the V matrix for FP8 WGMMA. Finally, we
will present some FLOPs/s benchmarks collected from runs with synthetic
data on the NVIDIA^®^ H100 Tensor Core GPU with SXM5 board form factor.
In particular, for head dimension 256 and large sequence length, we
obtain over *1 petaflop/s* of performance. This is consistent with other
claims on FP8 fused attention performance in the same regime, e.g. by
[HippoAttention](https://blog.hippoml.com/petaflops-inference-era-1-pflops-attention-and-preliminary-end-to-end-results-21f682cf2ed1){data-type="link"
data-id="https://blog.hippoml.com/petaflops-inference-era-1-pflops-attention-and-preliminary-end-to-end-results-21f682cf2ed1"}.

## Recollections on FlashAttention-2 {#recollections-on-flashattention-2 .wp-block-heading}

FlashAttention-2 is a *memory-aware* algorithm for multi-head attention
(MHA) that was
[introduced](https://arxiv.org/abs/2307.08691){data-type="link"
data-id="https://arxiv.org/abs/2307.08691"} by Tri Dao last year,
building on earlier [work](https://arxiv.org/abs/2205.14135) of Dao and
his collaborators. It serves as blueprint for implementing MHA as a
*fused* CUDA kernel (FMHA), in which intermediate steps of the attention
computation

![O = \\mathrm{softmax}\\left(\\frac{1}{\\sqrt{d}} Q K\^T \\right) V =
\\mathrm{softmax}(S) V = P
V](https://s0.wp.com/latex.php?latex=O+%3D+%5Cmathrm%7Bsoftmax%7D%5Cleft%28%5Cfrac%7B1%7D%7B%5Csqrt%7Bd%7D%7D+Q+K%5ET+%5Cright%29+V+%3D+%5Cmathrm%7Bsoftmax%7D%28S%29+V+%3D+P+V&bg=ffffff&fg=000&s=0&c=20201002){.latex
decoding="async"
srcset="https://s0.wp.com/latex.php?latex=O+%3D+%5Cmathrm%7Bsoftmax%7D%5Cleft%28%5Cfrac%7B1%7D%7B%5Csqrt%7Bd%7D%7D+Q+K%5ET+%5Cright%29+V+%3D+%5Cmathrm%7Bsoftmax%7D%28S%29+V+%3D+P+V&bg=ffffff&fg=000&s=0&c=20201002&zoom=2.25 2x"}

aren't read back to HBM (i.e., global memory), but rather kept in shared
memory or in registers. The key idea of FlashAttention-2, as well as the
original FlashAttention, is to leverage tiling of the input tensors Q,
K, V in conjunction with the *online-softmax* algorithm, which allows
one to circumvent keeping the entire S matrix live for the softmax
calculation.

Apart from the algorithm itself, there are a number of interesting
challenges in terms of optimization to get a performant implementation
on modern GPUs, in particular those built on NVIDIA Hopper architecture
like the H100 GPU. For example, to make optimal use of the Tensor Cores
on an H100 GPU for carrying out the matrix multiplications in FMHA, it's
important to target the new Hopper-specific WGMMA instructions as well
as the Tensor Memory Accelerator (TMA) for loads from global to shared
memory. We described how to accomplish this using tools from NVIDIA's
open-source CUTLASS library in our
[paper](https://arxiv.org/abs/2312.11918) from last December. There, we
chose to work with half-precision FP16 input tensors Q, K, V for the
head-to-head performance comparison with Dao's base implementation
written for NVIDIA Ampere architecture. However, in practice we want to
transition to even lower precision types, in keeping with the philosophy
of model quantization. Therefore, as a first step we need to understand
how to accommodate FP8 input tensors in the design of the FMHA kernel.

In what follows, when we discuss invoking WGMMA (that is,
`wgmma.mma_async` in PTX) to multiply matrices such as Q**·**K^T^ or
P**·**V, we will implicitly be referring to tiles of these matrices. For
example, we could choose 64×64 tiles.

## The challenge with FP8: layout conformance of WGMMA register fragments {#the-challenge-with-fp8-layout-conformance-of-wgmma-register-fragments .wp-block-heading}

WGMMA is warpgroup-level, so involves 128 threads (4 warps) to carry out
the MMA. When invoking a WGMMA instruction, accumulation occurs in those
threads' registers. It is then paramount to understand the thread
ownership pattern, i.e. register fragment layout, of the entries of the
result matrix, as prescribed by WGMMA. For example, we have the
following layout for a slice of the FP32 accumulator tile, which is
extracted from [Figure
119](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#wgmma-64n16-d){data-type="link"
data-id="https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#wgmma-64n16-d"}
in the PTX documentation:

![](https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/fig-1-fp32-wgmma.png?resize=920%2C124&ssl=1){.wp-image-9466
recalc-dims="1" decoding="async" width="920" height="124"
style="width: 920px;"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/fig-1-fp32-wgmma.png?w=2080&ssl=1 2080w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/fig-1-fp32-wgmma.png?resize=480%2C65&ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/fig-1-fp32-wgmma.png?resize=960%2C129&ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/fig-1-fp32-wgmma.png?resize=768%2C103&ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/fig-1-fp32-wgmma.png?resize=1536%2C207&ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/fig-1-fp32-wgmma.png?resize=2048%2C276&ssl=1 2048w"
sizes="(max-width: 920px) 100vw, 920px"}

In Figure 1, we took the subset of the 64×N output tile consisting of
the two rows 0 and 8 and columns 0-15. Figure 1 then indicates that
these entries are evenly divided among threads 0-3. Moreover, the
entries are arrayed contiguously in each thread's registers according to
the *d* index. This is the register fragment layout that would appear as
part of the `wgmma.mma_async` calls for computing either Q·K^T^ or P·V
in FMHA, for instance, and regardless of the precision type of the
operands given FP32 accumulator.

When invoking WGMMA to do a matrix multiplication A·B, you have the
option of arranging for the operand A to either be in shared memory or
in registers (by contrast, operand B must always be in shared memory).
For the second GEMM P·V in FMHA, we want to use the option of keeping
the first operand in registers for performance reasons, avoiding
unnecessary writes and reads with shared memory. In order to compute a
correct result, we then need to match the FP32 accumulator register
fragment layout to either the FP16 or FP8 operand A register fragment
layout. To be more precise, we should first downcast the entries held in
registers in place to the desired precision format, and then potentially
do some register movement, both within and across threads.

This sounds complicated, and it turns out to involve some non-trivial
maneuvers in the case of FP8, but let us first point out that one
doesn't need to do any register movement at all in the case of FP16.
This is because the layouts in question are then *identical*: just
compare [Figure
119](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#wgmma-64n16-d){data-type="link"
data-id="https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#wgmma-64n16-d"}
and [Figure
118](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#wgmma-64n16-a){data-type="link"
data-id="https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#wgmma-64n16-a"}
in the PTX documentation. In fact, this makes it easy to take our FP16
FMHA kernel and allow for the input tensors Q and K to be FP8 while
still fixing the input tensor V to be FP16. We call this the "FP8
hybrid" FMHA kernel, which is akin to the FP8 version of FMHA in the
[Triton
tutorial](https://github.com/openai/triton/blob/53d868113a706988394134ca1f7f85cb3016cc81/python/tutorials/06-fused-attention.py#L605){data-type="link"
data-id="https://github.com/openai/triton/blob/53d868113a706988394134ca1f7f85cb3016cc81/python/tutorials/06-fused-attention.py#L605"}.

Let's now consider the case at hand, with all three tensors Q, K, V in
FP8 precision. Consider the following extract from [Figure
122](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#wgmma-64n32-a)
in the PTX documentation:

![](https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/fig-2-fp8-wgmma.png?resize=920%2C124&ssl=1){.wp-image-9461
recalc-dims="1" decoding="async" width="920" height="124"
style="width: 920px;"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/fig-2-fp8-wgmma.png?w=2080&ssl=1 2080w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/fig-2-fp8-wgmma.png?resize=480%2C65&ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/fig-2-fp8-wgmma.png?resize=960%2C129&ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/fig-2-fp8-wgmma.png?resize=768%2C103&ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/fig-2-fp8-wgmma.png?resize=1536%2C207&ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/fig-2-fp8-wgmma.png?resize=2048%2C276&ssl=1 2048w"
sizes="(max-width: 920px) 100vw, 920px"}

This is the thread ownership pattern, or *layout conformance*
requirement, that must be satisfied prior to invoking an FP8 WGMMA. For
example, according to Figure 2, we need to arrange that thread 0 has
these 8 entries in its registers in order:

`T0d0, T0d1, T1d0, T1d1, T0d2, T0d3, T1d2, T1d3.`

In particular, observe that we will need to exchange data among threads
to achieve layout conformance.

## The Solution: Byte Permute and Shuffle Sync {#the-solution-byte-permute-and-shuffle-sync .wp-block-heading}

Our solution in code is exposed on our repo via
[`ReorgCFp8toAFp8`](https://github.com/ColfaxResearch/cutlass-kernels/blob/master/src/fmha-pipeline/reg2reg.h#L45){data-type="link"
data-id="https://github.com/ColfaxResearch/cutlass-kernels/blob/master/src/fmha-pipeline/reg2reg.h#L45"},
which we invoke in the consumer path of the pipelined implementation
immediately
[before](https://github.com/ColfaxResearch/cutlass-kernels/blob/master/src/fmha-pipeline/fmha_consumer.h#L57){data-type="link"
data-id="https://github.com/ColfaxResearch/cutlass-kernels/blob/master/src/fmha-pipeline/fmha_consumer.h#L57"}
the second `gemm` call. Invoking that method on the CUTLASS Tensor
`tSrSPrec` (the FP8 downcasted register fragment Tensor) accomplishes
the following data movement, going from top to bottom:

![](https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/figure-3-data-movement.png?resize=950%2C293&ssl=1){.wp-image-9453
recalc-dims="1" loading="lazy" decoding="async" width="950" height="293"
style="width: 950px;"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/figure-3-data-movement.png?w=2150&ssl=1 2150w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/figure-3-data-movement.png?resize=480%2C148&ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/figure-3-data-movement.png?resize=960%2C296&ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/figure-3-data-movement.png?resize=768%2C237&ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/figure-3-data-movement.png?resize=1536%2C474&ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/figure-3-data-movement.png?resize=2048%2C632&ssl=1 2048w"
sizes="auto, (max-width: 950px) 100vw, 950px"}

In Figure 3, each box represents two entries from Figure 1 of the same
thread and color combined. Keeping in mind that the entries have been
downcasted to FP8, each box is then 16 bits wide. We then accomplish the
register data movement in code by invoking a combination of the
following two CUDA intrinsics:

1.  Between the first and second rows, and between the third and fourth
    rows, we don't have to exchange data between threads, only
    internally within a thread. To do this, we use
    [`__byte_perm`](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__INT.html#group__CUDA__MATH__INTRINSIC__INT_1g0b8b8156fe619205bf8d65acd8f29131):
    given two 32-bit unsigned integers `x` and `y`, `__byte_perm(x,y,s)`
    returns 4 bytes from the 8 input bytes as specified by the selector
    `s`.\
    For example, we can do a swap using `__byte_perm` as follows. From
    the top, for a given thread let `upper` be the first 4 bytes (those
    in light and dark blue) and let `lower` be the last 4 bytes (those
    in light and dark yellow). Then for threads 1 and 2, we swap as
    indicated by calling `__byte_perm` with the following selectors:\
    \
    `auto upper0 = __byte_perm(upper, lower, 0x7654);`\
    `auto lower0 = __byte_perm(upper, lower, 0x3210);`\
    \
2.  Between the second and third row, we exchange data among threads
    using
    [`__shfl_sync`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions).
    Observe that the upper and lower blocks of 4 bytes are each
    exchanged among themselves. Moreover, the shuffling of the upper
    blocks differs from that of the lower blocks, and both shuffles
    depend on the thread index mod 4. We account for this using two
    [pre-defined
    arrays](https://github.com/ColfaxResearch/cutlass-kernels/blob/master/src/fmha-pipeline/reg2reg.h#L50)
    to call `__shfl_sync` with the correct
    [`srcLane`](https://github.com/ColfaxResearch/cutlass-kernels/blob/master/src/fmha-pipeline/reg2reg.h#L98)
    parameter. Specifically, we define\
    \
    `int upper_map[4] = {0,3,1,2};`\
    `int lower_map[4] = {1,2,0,3};`\
    \
    and then for the register movement we invoke\
    \
    `upper0 = __shfl_sync(uint32_t(-1), upper0, upper_map[threadIdx.x%4], 4);`\
    `lower0 = __shfl_sync(uint32_t(-1), lower0, lower_map[threadIdx.x%4], 4); `\
    \

## An added wrinkle: transposing the V matrix offline {#an-added-wrinkle-transposing-the-v-matrix-offline .wp-block-heading}

We've been thinking about Q, K, V as matrices for the attention formula,
but for the attention layer the Q, K, V are actually 4-dimensional
tensors, with dimensions given by the batch size B, sequence length S,
numbers of heads H, and head dimension D. For our FP16 FMHA kernel, we
conformed to a certain convention regarding how these tensors are
arrayed in memory. Namely, we supposed that they are packed in the (B,
S, 3, H, D) format. Then when loading tiles from global to shared memory
via TMA, this convention forces the 2-dimensional Q, K, V tiles to be
contiguous in the head dimension and strided in the sequence length
dimension.

Given a gemm call to multiply A·B^T^ for an (M×K)-matrix A and an
(N×K)-matrix B, we say that the A resp. B operand is *mn-major* if it is
contiguous in the M resp. N dimension (or outer dimension), and
*k-major* if is instead contiguous in the K-dimension (or inner
dimension). Then given the (B, S, 3, H, D) convention, for the first
GEMM Q·K^T^ the 2nd operand is k-major, whereas for the second GEMM P·V
the 2nd operand is mn-major. Fortuitously, for FP16 precision this
distinction turns out to be immaterial since WGMMA accepts both k-major
or mn-major for its 2nd operand. However, this is no longer the case for
FP8 precision. Therefore, we need to either transpose the V tensor as a
preprocessing step before calling our FP8 FMHA kernel (but not the FP8
hybrid version!), or otherwise handle the transpose elsewhere, such as
fusing it to the epilogue of the projection that creates the V tensor.

## FLOPS benchmarks {#flops-benchmarks .wp-block-heading}

![](https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/flops-benchmarks-h100-sxm5.png?resize=800%2C737&ssl=1){.wp-image-9658
recalc-dims="1" loading="lazy" decoding="async" width="800" height="737"
style="width: 800px;"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/flops-benchmarks-h100-sxm5.png?w=2040&ssl=1 2040w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/flops-benchmarks-h100-sxm5.png?resize=293%2C270&ssl=1 293w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/flops-benchmarks-h100-sxm5.png?resize=586%2C540&ssl=1 586w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/flops-benchmarks-h100-sxm5.png?resize=768%2C708&ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/flops-benchmarks-h100-sxm5.png?resize=1536%2C1416&ssl=1 1536w"
sizes="auto, (max-width: 800px) 100vw, 800px"}

Figure 4 shows the FLOPs/s improvements we see from moving to lower
precision types. Runs were conducted with synthetic data drawn from a
normal distribution with mean 0 and variance 1. To interpret the figure
correctly, recall from above that **FP8 Hybrid** refers to the FMHA
kernel in which the Q and K tensors are FP8 and V is FP16, and note that
the TFLOPs/s reported for **FP8** don't include the cost of transposing
V since that is a pre-processing step that is expected to be accounted
for elsewhere.

In our experiments, we have tuned the number of pipeline stages as well
as the tile sizes used in WGMMA independently for each kernel: we have
configurable parameters QBLKSIZE and KBLKSIZE for tiling along the
sequence length of Q and K, V respectively (note that we never divide
along the head dimension). We see that the largest improvement going
from **FP8 Hybrid** to **FP8** occurs in the case of head dimension 256.
This happens because keeping V as FP8 reduces pressure on the shared
memory, enabling us to set both QBLKSIZE and KBLKSIZE to 128.

Apart from the set of parameters chosen for Figure 4, we can also find
parameters that showcase over 1 petaflop/s of performance through
increasing the sequence length. For example, for sequence length 8448 =
64\*132 we have:

::: wp-block-syntaxhighlighter-code
``` {.brush: .plain; .gutter: .false; .highlight: .[1,2,3,4,5,6,7,8,9,10]; .title: .; .notranslate title=""}

dgxuser@PM-DGX-H100:~/kernels$ ./fmha_fp8_pipe_128x128xCTA256 \
--batch-size=4 --seq-length=8448 --head-size=256 --iterations=1000
Using device 0: NVIDIA H100 80GB HBM3  (SM90, 132 SMs)
M = 8448
N = 8448
K = 256
QBLK = 128
KBLK = 128
L = 32 : 8 * 4
CUTE_FMHA:     [1028661.0]Gflop/s  (2.2735)ms
```
:::

Or if we let sequence length be 16896 = 2\*8448, then we have:

::: wp-block-syntaxhighlighter-code
``` {.brush: .plain; .gutter: .false; .highlight: .[1,2,3,4,5,6,7,8,9,10]; .title: .; .notranslate title=""}

dgxuser@PM-DGX-H100:~/kernels$ ./fmha_fp8_pipe_128x128xCTA256 \
--batch-size=2 --seq-length=16896 --head-size=256 --iterations=1000
Using device 0: NVIDIA H100 80GB HBM3  (SM90, 132 SMs)
M = 16896
N = 16896
K = 256
QBLK = 128
KBLK = 128
L = 16 : 8 * 2
CUTE_FMHA:     [1057474.8]Gflop/s  (4.4230)ms
```
:::

For these examples, multiples of 132 were chosen to account for [wave
quantization](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#wave-quant)
effects.

Finally, for ease of reproducibility we give our autotuned parameters
for the different kernels:

![](https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/params.png?resize=800%2C287&ssl=1){.wp-image-9661
recalc-dims="1" loading="lazy" decoding="async" width="800" height="287"
style="width: 800px;"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/params.png?w=1952&ssl=1 1952w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/params.png?resize=480%2C172&ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/params.png?resize=960%2C344&ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/params.png?resize=768%2C275&ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/params.png?resize=1536%2C551&ssl=1 1536w"
sizes="auto, (max-width: 800px) 100vw, 800px"}

In Table 1, NOPIPE refers to replacing the software pipelining by our
original copy/gemm overlapping scheme described in section §6 of our
[paper](https://arxiv.org/abs/2312.11918){data-type="link"
data-id="https://arxiv.org/abs/2312.11918"}. Also, with NOPIPE we don't
use the QINRMEM option, while with the default pipelined version
(STAGES=3) we do use QINRMEM.

## Addendum: accuracy loss with FP8 {#addendum-accuracy-loss-with-fp8 .wp-block-heading}

As always with moving to lower precision types, the concomitant effects
on accuracy loss have to be understood and managed correctly. We leave
this important topic to future work. For now, we can report the
root-mean-square error taken between the output matrix and a reference
calculation in Table 2 (with the same synthetic data and other
parameters fixed as in Figure 4):

![](https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/rmse.png?resize=600%2C155&ssl=1){.wp-image-9668
recalc-dims="1" loading="lazy" decoding="async" width="600" height="155"
style="width: 600px;"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/rmse.png?w=1366&ssl=1 1366w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/rmse.png?resize=480%2C124&ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/rmse.png?resize=960%2C249&ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/02/rmse.png?resize=768%2C199&ssl=1 768w"
sizes="auto, (max-width: 600px) 100vw, 600px"}

In Table 2, the reference calculation is done on the Q, K, V tensors
generated in the given precision type and then upcasted to FP32.

## Acknowledgments {#acknowledgments .wp-block-heading}

We would like to thank Tri Dao for helpful comments on an earlier draft
and Ying Zhang from the PyTorch team for pointing out the wave
quantization optimization to us.

*Edit* (03/06/2024): Included more thorough autotuning for auxiliary
parameters and updated FLOPS numbers based on this.\
*Edit* (03/10/2024): Added reference to similar work by HippoAttention
on FP8 fused attention.

![](/wp-content/uploads/2023/10/PDF_32.png){.download-link}
[FP8-FMHA-blog.pdf](https://research.colfax-intl.com/download/fp8-fmha-blog/?tmstv=1771876180){#download-link-9542
.download-link e-disable-page-transition="true" rel="nofollow"
redirect="false"}

::::: {.sharedaddy .sd-sharing-enabled}
:::: {.robots-nocontent .sd-block .sd-social .sd-social-icon-text .sd-sharing}
### Share this: {#share-this .sd-title}

::: sd-content
- [[Share on LinkedIn (Opens in new window)]{#sharing-linkedin-9130
  hidden=""}
  LinkedIn](https://research.colfax-intl.com/adding-fp8-to-flashattention/?share=linkedin){.share-linkedin
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-linkedin-9130" target="_blank"
  aria-labelledby="sharing-linkedin-9130"}
- [[Share on X (Opens in new window)]{#sharing-twitter-9130 hidden=""}
  X](https://research.colfax-intl.com/adding-fp8-to-flashattention/?share=twitter){.share-twitter
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-twitter-9130" target="_blank"
  aria-labelledby="sharing-twitter-9130"}
- [[Share on Facebook (Opens in new window)]{#sharing-facebook-9130
  hidden=""}
  Facebook](https://research.colfax-intl.com/adding-fp8-to-flashattention/?share=facebook){.share-facebook
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-facebook-9130" target="_blank"
  aria-labelledby="sharing-facebook-9130"}
- [[Share on Reddit (Opens in new window)]{#sharing-reddit-9130
  hidden=""}
  Reddit](https://research.colfax-intl.com/adding-fp8-to-flashattention/?share=reddit){.share-reddit
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-reddit-9130" target="_blank"
  aria-labelledby="sharing-reddit-9130"}
-
:::
::::
:::::
::::::::

::::::: {.wp-block-group .has-global-padding .is-layout-constrained .wp-block-group-is-layout-constrained style="margin-top:48px;margin-bottom:48px;padding-top:5px;padding-bottom:5px"}

------------------------------------------------------------------------

### Discover more from Colfax Research {#discover-more-from-colfax-research .wp-block-heading .has-text-align-center style="margin-top:4px;margin-bottom:10px"}

Subscribe to get the latest posts sent to your email.

:::::: {.wp-block-group .has-global-padding .is-layout-constrained .wp-container-core-group-is-layout-b821fca1 .wp-block-group-is-layout-constrained}
::::: {.wp-block-jetpack-subscriptions__supports-newline .wp-block-jetpack-subscriptions}
:::: {.wp-block-jetpack-subscriptions__container .is-not-subscriber}
::: wp-block-jetpack-subscriptions__form-elements
Type your email...

Subscribe
:::
::::
:::::
::::::
:::::::

:::::::::: wp-block-template-part
::: {.wp-block-spacer style="height:0" aria-hidden="true"}
:::

:::::::: {.wp-block-group .has-global-padding .is-layout-constrained .wp-block-group-is-layout-constrained style="margin-top:var(--wp--preset--spacing--70)"}
::::::: {.wp-block-columns .alignwide .has-small-font-size .is-layout-flex .wp-container-core-columns-is-layout-7495e5c1 .wp-block-columns-is-layout-flex style="margin-top:var(--wp--preset--spacing--30)"}
:::::: {.wp-block-column .is-layout-flow .wp-container-core-column-is-layout-47e5a185 .wp-block-column-is-layout-flow}
::::: {.wp-block-group .is-layout-flex .wp-container-core-group-is-layout-f0ee7b9b .wp-block-group-is-layout-flex}
Posted

::: wp-block-post-date
February 29, 2024
:::

in

::: {.taxonomy-category .wp-block-post-terms}
[Deep
Learning](https://research.colfax-intl.com/category/papers/deep-learning/){rel="tag"}[,
]{.wp-block-post-terms__separator}[Publications](https://research.colfax-intl.com/category/papers/){rel="tag"}
:::
:::::
::::::
:::::::
::::::::
::::::::::

:::::: {.section .wp-block-template-part}
::::: {.wp-block-group .has-global-padding .is-layout-constrained .wp-container-core-group-is-layout-a666d811 .wp-block-group-is-layout-constrained style="padding-top:var(--wp--preset--spacing--40);padding-right:var(--wp--preset--spacing--40);padding-bottom:var(--wp--preset--spacing--40);padding-left:var(--wp--preset--spacing--40)"}
:::: wp-block-comments
## Comments {#comments .wp-block-heading}

::: {#respond .comment-respond .wp-block-post-comments-form}
### Leave a Reply [[Cancel reply](/adding-fp8-to-flashattention/#respond){#cancel-comment-reply-link rel="nofollow" style="display:none;"}]{.small} {#reply-title .comment-reply-title}

[Your email address will not be published.]{#email-notes} [Required
fields are marked [\*]{.required}]{.required-field-message}

Comment [\*]{.required}

Name

Email

Website

Save my name, email, and website in this browser for the next time I
comment.

Δ
:::
::::
:::::
::::::
:::::::::::::::::::::::::::
