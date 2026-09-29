::::::::::::::::::::::::: {.wp-block-group .is-layout-flow .wp-block-group-is-layout-flow role="main" style="margin-top:var(--wp--preset--spacing--50)"}
::: {.wp-block-group .has-global-padding .is-layout-constrained .wp-block-group-is-layout-constrained}
# FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision {#flashattention-3-fast-and-accurate-attention-with-asynchrony-and-low-precision .wp-block-post-title style="margin-bottom:var(--wp--preset--spacing--40);"}
:::

:::::: {.entry-content .wp-block-post-content .has-global-padding .is-layout-constrained .wp-block-post-content-is-layout-constrained}
#### Jay Shah^\*1^, Ganesh Bikshandi^\*1^, Ying Zhang^2^, Vijay Thakkar^3,4^, Pradeep Ramani^3^, Tri Dao^5,6^ {#jay-shah1-ganesh-bikshandi1-ying-zhang2-vijay-thakkar34-pradeep-ramani3-tri-dao56 .wp-block-heading .has-small-font-size}

#### ^1^Colfax Research, ^2^Meta, ^3^NVIDIA, ^4^Georgia Tech, ^5^Princeton University, ^6^Together AI {#colfax-research-2meta-3nvidia-4georgia-tech-5princeton-university-6together-ai .wp-block-heading .has-small-font-size}

*Cross-posted from* {<https://www.together.ai/blog/flashattention-3>,
<https://pytorch.org/blog/flashattention-3/>,
<https://tridao.me/blog/2024/flash3/>}

Attention, as a core layer of the ubiquitous Transformer architecture,
is a bottleneck for large language models and long-context applications.
FlashAttention (and FlashAttention-2) pioneered an approach to speed up
attention on GPUs by minimizing memory reads/writes, and is now used by
most
[libraries](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html){data-type="link"
data-id="https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html"}
to accelerate Transformer training and inference. This has contributed
to a massive increase in LLM context length in the last two years, from
2-4K (GPT-3, OPT) to 128K (GPT-4), or even 1M ([Llama
3](https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k){data-type="link"
data-id="https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k"}).

However, despite its success, FlashAttention has yet to take advantage
of new capabilities in modern hardware, with FlashAttention-2 achieving
only 35% utilization of theoretical max FLOPs on the H100 GPU. In this
blogpost, we describe three main techniques to speed up attention on
Hopper GPUs: exploiting asynchrony of the Tensor Cores and TMA to (1)
overlap overall computation and data movement via warp-specialization
and (2) interleave block-wise matmul and softmax operations, and (3)
incoherent processing that leverages hardware support for FP8
low-precision.

We're excited to release FlashAttention-3 that incorporates these
techniques. It's 1.5-2.0x faster than FlashAttention-2 with FP16, up to
740 TFLOPS, i.e., 75% utilization of H100 theoretical max FLOPS. With
FP8, FlashAttention-3 reaches close to 1.2 PFLOPS, with 2.6x smaller
error than baseline FP8 attention.

The improvements from FlashAttention-3 will result in:

1.  **More efficient GPU Utilization**: The new technique uses up to 75%
    of an H100 GPU's maximum capabilities, up from just 35% before. This
    results in significantly (1.5-2x) faster than previous versions for
    training and running of large language models (LLMs).
2.  **Better performance with lower precision**: FlashAttention-3 can
    work with lower precision numbers (FP8) while maintaining accuracy.
    This allows for even faster processing and potentially lower memory
    usage, which could lead to cost savings and improved efficiency for
    customers running large-scale AI operations.
3.  **Ability to use longer context in LLMs**: By speeding up the
    attention mechanism, FlashAttention-3 enables AI models to work with
    much longer pieces of text more efficiently. This could allow for
    applications that can understand and generate longer, more complex
    content without slowing down.

FlashAttention-3 is available at:
<https://github.com/Dao-AILab/flash-attention>.

![](/wp-content/uploads/2023/10/PDF_32.png){.download-link}
[Paper](https://research.colfax-intl.com/download/fa3-paper/?tmstv=1771876173){#download-link-11545
.download-link e-disable-page-transition="true" rel="nofollow"
redirect="false"}

## FlashAttention Recap {#flashattention-recap .wp-block-heading}

[FlashAttention](https://arxiv.org/abs/2205.14135){data-type="link"
data-id="https://arxiv.org/abs/2205.14135"} is an algorithm that
reorders the attention computation and leverages tiling and
recomputation to significantly speed it up and reduce memory usage from
quadratic to linear in sequence length. We use tiling to load blocks of
inputs from HBM (GPU memory) to SRAM (fast cache), perform attention
with respect to that block, and update the output in HBM. By not writing
the large intermediate attention matrices to HBM, we reduce the amount
of memory reads/writes, which brings 2-4x wallclock time speedup.

Here we show a diagram of FlashAttention forward pass: with tiling and
softmax rescaling, we operate by blocks and avoid having to read/write
from HBM, while obtaining the correct output with no approximation.

<figure class="wp-block-image size-large is-resized">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/flashattention.png?resize=960%2C485&amp;ssl=1"
class="wp-image-11533" style="width:800px" data-recalc-dims="1"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/flashattention.png?resize=960%2C485&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/flashattention.png?resize=480%2C243&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/flashattention.png?resize=768%2C388&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/flashattention.png?resize=1536%2C777&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/flashattention.png?w=1600&amp;ssl=1 1600w"
sizes="(max-width: 960px) 100vw, 960px" width="960" height="485" />
</figure>

## New hardware features on Hopper GPUs -- WGMMA, TMA, FP8 {#new-hardware-features-on-hopper-gpus-wgmma-tma-fp8 .wp-block-heading}

While FlashAttention-2 can achieve up to 70% theoretical max FLOPS on
Ampere (A100) GPUs, it does not yet take advantage of new features on
Hopper GPUs to maximize performance. We describe some of the new
Hopper-specific features here, and why they are important.

1\. **WGMMA (Warpgroup Matrix Multiply-Accumulate)**. This new feature
makes use of the new Tensor Cores on Hopper, with much higher
throughput^[1](#81189d53-d31b-41bf-8c9a-f5ae49431ccb){#81189d53-d31b-41bf-8c9a-f5ae49431ccb-link}^
than the older mma.sync instruction in Ampere (image from the [H100
white
paper](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper?ncid=no-ncid)).

<figure class="wp-block-image size-full is-resized">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/a100_vs_h100.png?resize=802%2C422&amp;ssl=1"
class="wp-image-11534" style="width:700px" data-recalc-dims="1"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/a100_vs_h100.png?w=802&amp;ssl=1 802w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/a100_vs_h100.png?resize=480%2C253&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/a100_vs_h100.png?resize=768%2C404&amp;ssl=1 768w"
sizes="(max-width: 802px) 100vw, 802px" width="802" height="422" />
</figure>

2\. **TMA (Tensor Memory Accelerator)**. This is a special hardware unit
that accelerates the transfer of data between global memory and shared
memory, taking care of all index calculation and out-of-bound
predication. This frees up registers, which is a valuable resource to
increase tile size and efficiency.

<figure class="wp-block-image size-full is-resized">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/tma.png?resize=916%2C372&amp;ssl=1"
class="wp-image-11535" style="width:600px" data-recalc-dims="1"
loading="lazy" decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/tma.png?w=916&amp;ssl=1 916w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/tma.png?resize=480%2C195&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/tma.png?resize=768%2C312&amp;ssl=1 768w"
sizes="auto, (max-width: 916px) 100vw, 916px" width="916"
height="372" />
</figure>

3\. **Low-precision with FP8**. This doubles the Tensor Core throughput
(e.g. 989 TFLOPS with FP16 and 1978 TFLOPS with FP8), but trades off
accuracy by using fewer bits to represent floating point numbers.

FlashAttention-3 makes use of all of these new features of Hopper, using
powerful abstractions from NVIDIA's [CUTLASS
library](https://github.com/NVIDIA/cutlass){data-type="link"
data-id="https://github.com/NVIDIA/cutlass"}.

By rewriting FlashAttention to use these new features, we can already
significantly speed it up (e.g., from 350 TFLOPS in FlashAttention-2
FP16 forward pass to around 540-570 TFLOPS). However, the asynchronous
nature of the new instructions on Hopper (WGMMA and TMA) opens up
additional algorithmic opportunities to overlap operations and thereby
extract even greater performance. For this blogpost, we'll explain two
such techniques specific to attention. The generic technique of warp
specialization, with separate producer and consumer warps doing TMA and
WGMMA, is [well-covered
elsewhere](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md#warp-specialization){data-type="link"
data-id="https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md#warp-specialization"}
in the context of GEMM and works the same here.

## Asynchrony: Overlapping GEMM and Softmax {#asynchrony-overlapping-gemm-and-softmax .wp-block-heading}

### Why overlap? {#why-overlap .wp-block-heading}

Attention has GEMMs (those matmuls between Q and K and between attention
probability P and V) and softmax as its two main operations. Why do we
need to overlap them? Isn't most of the FLOPS in the GEMMs anyway? As
long as the GEMMs are fast (e.g., computed using WGMMA instructions),
shouldn't the GPU [be going
brrrr](https://horace.io/brrr_intro.html){data-type="link"
data-id="https://horace.io/brrr_intro.html"}?

The problem is that non-matmul operations are much slower than matmul
operations on modern accelerators. Special functions such as exponential
(for the softmax) have even lower throughput than floating point
multiply-add; they are evaluated by the multi-function unit, a unit
separate from floating point multiply-add or matrix multiply-add. As an
example, the H100 GPU SXM5 has 989 TFLOPS of FP16 matrix multiply, but
only 3.9 TFLOPS (256x less throughput) for special
functions^[2](#ec290db0-c6c5-4d50-aa3d-cb39554711fd){#ec290db0-c6c5-4d50-aa3d-cb39554711fd-link}^!
For head dimension 128, there are 512x more matmul FLOPS than
exponential, which means that exponential can take 50% of the time
compared to matmul. The situation is even worse for FP8, where the
matmul FLOPS are twice as fast yet exponential FLOPS stay the same
speed. Ideally we want matmul and softmax to operate in parallel. While
the Tensor Cores are busy with matmul, the multi-function units should
be calculating exponential!

### Inter-warpgroup overlapping with pingpong scheduling {#inter-warpgroup-overlapping-with-pingpong-scheduling .wp-block-heading}

The first and easiest way to overlap GEMM and softmax is to do nothing
at all! The warp schedulers already try to schedule warps so that if
some warps are blocked (e.g., waiting for GEMM results), other warps can
run. That is, the warp schedulers do some of this overlapping for us,
for free.

However, we can improve on this by doing some of the scheduling
manually. As an example, if we have 2 warpgroups (labeled 1 and 2 --
each warpgroup is a group of 4 warps), we can use synchronization
barriers (`bar.sync`) so that warpgroup 1 first does its GEMMs (e.g.,
GEMM1 of one iteration and GEMM0 of the next iteration), and then
warpgroup 2 does its GEMMs while warpgroup 1 does its softmax, and so
on. This "pingpong" schedule is illustrated in the figure below, where
the same color denotes the same iteration.

<figure class="wp-block-image size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/pingpong.png?resize=960%2C208&amp;ssl=1"
class="wp-image-11536" data-recalc-dims="1" loading="lazy"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/pingpong.png?resize=960%2C208&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/pingpong.png?resize=480%2C104&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/pingpong.png?resize=768%2C167&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/pingpong.png?resize=1536%2C333&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/pingpong.png?w=1600&amp;ssl=1 1600w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="208" />
</figure>

This would allow us to perform the softmax in the shadow of the GEMMs of
the other warpgroup. Of course, this figure is just a caricature; in
practice the scheduling is not really this clean. Nevertheless, pingpong
scheduling can improve FP16 attention forward pass from around 570
TFLOPS to 620 TFLOPS (head dim 128, seqlen 8K).

### Intra-warpgroup overlapping of GEMM and Softmax {#intra-warpgroup-overlapping-of-gemm-and-softmax .wp-block-heading}

Even within one warpgroup, we can have some part of softmax running
while the GEMMs of that warpgroup is running. This is illustrated in
this figure, where the same color denotes the same iteration.

<figure class="wp-block-image size-full">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/2pipeline.png?resize=652%2C140&amp;ssl=1"
class="wp-image-11537" data-recalc-dims="1" loading="lazy"
decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/2pipeline.png?w=652&amp;ssl=1 652w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/2pipeline.png?resize=480%2C103&amp;ssl=1 480w"
sizes="auto, (max-width: 652px) 100vw, 652px" width="652"
height="140" />
</figure>

This pipelining increases throughput from around 620 TFLOPS to around
640-660 TFLOPS for FP16 attention forward, at the cost of higher
register pressure. We need more registers to hold both accumulators of
the GEMMs, and the input/output of softmax. Overall, we find this
technique to offer a favorable tradeoff.

## Low-precision: reduce quantization error with incoherent processing {#low-precision-reduce-quantization-error-with-incoherent-processing .wp-block-heading}

LLM activation can have
[outliers](https://arxiv.org/abs/2208.07339){data-type="link"
data-id="https://arxiv.org/abs/2208.07339"} with much larger magnitude
than the rest of the features. These outliers make it difficult to
quantize, producing much larger quantization errors. We leverage
incoherent processing, a technique used in the quantization literature
(e.g. from [QuIP](https://arxiv.org/abs/2307.13304){data-type="link"
data-id="https://arxiv.org/abs/2307.13304"}) that multiplies the query
and key with a random orthogonal matrix to "spread out" the outliers and
reduce quantization error. In particular, we use the Hadamard transform
(with random signs), which can be done per attention head in O(d log d)
instead of O(d\^2) time, where d is the head dimension. Since the
Hadamard transform is memory-bandwidth bound, it can be fused with
previous operations such as rotary embedding (also memory-bandwidth
bound) "for free".

In our experiment where Q, K, V are generated from a standard normal
distribution but 0.1% of the entries have large magnitudes (to simulate
outliers), we found that incoherent processing can reduce the
quantization error by 2.6x. We show numerical error comparison in the
table below. Please see the paper for details.

<figure class="wp-block-image size-large is-resized">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/error_comparison.png?resize=960%2C116&amp;ssl=1"
class="wp-image-11538" style="width:760px" data-recalc-dims="1"
loading="lazy" decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/error_comparison.png?resize=960%2C116&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/error_comparison.png?resize=480%2C58&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/error_comparison.png?resize=768%2C93&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/error_comparison.png?resize=1536%2C185&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/error_comparison.png?w=1600&amp;ssl=1 1600w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="116" />
</figure>

## Attention benchmark {#attention-benchmark .wp-block-heading}

We show some results with FlashAttention-3, and compare it to
FlashAttention-2, as well as the implementation in Triton and cuDNN
(both of which already use new hardware features of Hopper GPUs). For
FP16, we see about 1.6x-1.8x speedup over FlashAttention-2:

<figure class="wp-block-image size-large is-resized">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/fp16_benchmark-1.png?resize=461%2C540&amp;ssl=1"
class="wp-image-11541" style="width:760px" data-recalc-dims="1"
loading="lazy" decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/fp16_benchmark-1.png?resize=461%2C540&amp;ssl=1 461w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/fp16_benchmark-1.png?resize=230%2C270&amp;ssl=1 230w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/fp16_benchmark-1.png?resize=768%2C900&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/fp16_benchmark-1.png?resize=1310%2C1536&amp;ssl=1 1310w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/fp16_benchmark-1.png?w=1364&amp;ssl=1 1364w"
sizes="auto, (max-width: 461px) 100vw, 461px" width="461"
height="540" />
</figure>

<figure class="wp-block-image size-large is-resized">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/fp16_backward.png?resize=960%2C378&amp;ssl=1"
class="wp-image-11542" style="width:760px" data-recalc-dims="1"
loading="lazy" decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/fp16_backward.png?resize=960%2C378&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/fp16_backward.png?resize=480%2C189&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/fp16_backward.png?resize=768%2C302&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/fp16_backward.png?w=1036&amp;ssl=1 1036w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="378" />
</figure>

For FP8, we can reach close to 1.2 PFLOPS!

<figure class="wp-block-image size-large is-resized">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/fp8_benchmark.png?resize=467%2C540&amp;ssl=1"
class="wp-image-11539" style="width:760px" data-recalc-dims="1"
loading="lazy" decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/fp8_benchmark.png?resize=467%2C540&amp;ssl=1 467w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/fp8_benchmark.png?resize=234%2C270&amp;ssl=1 234w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/fp8_benchmark.png?resize=768%2C887&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/fp8_benchmark.png?resize=1330%2C1536&amp;ssl=1 1330w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/07/fp8_benchmark.png?w=1385&amp;ssl=1 1385w"
sizes="auto, (max-width: 467px) 100vw, 467px" width="467"
height="540" />
</figure>

## Discussion {#discussion .wp-block-heading}

This blogpost highlights some of the optimizations for FlashAttention
available on Hopper GPUs. Other optimizations (e.g., variable length
sequences, persistent kernel, and in-kernel transpose for FP8) are
covered in the paper.

We have seen that designing algorithms that take advantage of the
hardware they run on can bring significant efficiency gains and unlock
new model capabilities such as long context. We look forward to future
work on optimization for LLM inference, as well as generalizing our
techniques to other hardware architectures.

We also look forward to FlashAttention-3 being integrated in a future
release of PyTorch.

1.  [Without the `wgmma` instruction, the older `mma.sync` instruction
    can only reach about ⅔ the peak throughput of Hopper Tensor Cores:
    <https://arxiv.org/abs/2402.13499v1>
    [↩︎](#81189d53-d31b-41bf-8c9a-f5ae49431ccb-link){aria-label="Jump to footnote reference 1"}]{#81189d53-d31b-41bf-8c9a-f5ae49431ccb}
2.  [The CUDA programming guide specifies that the throughput for
    special functions is 16 operations per streaming multiprocessor (SM)
    per clock cycle. We multiply 16 by 132 SMs and 1830 Mhz (clock speed
    used to calculate 989 TFLOPS of FP16 matmul) to get 3.9 TFLOPS.
    [↩︎](#ec290db0-c6c5-4d50-aa3d-cb39554711fd-link){aria-label="Jump to footnote reference 2"}]{#ec290db0-c6c5-4d50-aa3d-cb39554711fd}

::::: {.sharedaddy .sd-sharing-enabled}
:::: {.robots-nocontent .sd-block .sd-social .sd-social-icon-text .sd-sharing}
### Share this: {#share-this .sd-title}

::: sd-content
- [[Share on LinkedIn (Opens in new window)]{#sharing-linkedin-11531
  hidden=""}
  LinkedIn](https://research.colfax-intl.com/flashattention-3-fast-and-accurate-attention-with-asynchrony-and-low-precision/?share=linkedin){.share-linkedin
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-linkedin-11531" target="_blank"
  aria-labelledby="sharing-linkedin-11531"}
- [[Share on X (Opens in new window)]{#sharing-twitter-11531 hidden=""}
  X](https://research.colfax-intl.com/flashattention-3-fast-and-accurate-attention-with-asynchrony-and-low-precision/?share=twitter){.share-twitter
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-twitter-11531" target="_blank"
  aria-labelledby="sharing-twitter-11531"}
- [[Share on Facebook (Opens in new window)]{#sharing-facebook-11531
  hidden=""}
  Facebook](https://research.colfax-intl.com/flashattention-3-fast-and-accurate-attention-with-asynchrony-and-low-precision/?share=facebook){.share-facebook
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-facebook-11531" target="_blank"
  aria-labelledby="sharing-facebook-11531"}
- [[Share on Reddit (Opens in new window)]{#sharing-reddit-11531
  hidden=""}
  Reddit](https://research.colfax-intl.com/flashattention-3-fast-and-accurate-attention-with-asynchrony-and-low-precision/?share=reddit){.share-reddit
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-reddit-11531" target="_blank"
  aria-labelledby="sharing-reddit-11531"}
-
:::
::::
:::::
::::::

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
July 11, 2024
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
### Leave a Reply [[Cancel reply](/flashattention-3-fast-and-accurate-attention-with-asynchrony-and-low-precision/#respond){#cancel-comment-reply-link rel="nofollow" style="display:none;"}]{.small} {#reply-title .comment-reply-title}

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
:::::::::::::::::::::::::
