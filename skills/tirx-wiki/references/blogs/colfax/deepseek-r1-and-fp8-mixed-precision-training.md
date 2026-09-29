::::::::::::::::::::::: {.wp-block-group .is-layout-flow .wp-block-group-is-layout-flow role="main" style="margin-top:var(--wp--preset--spacing--50)"}
::: {.wp-block-group .has-global-padding .is-layout-constrained .wp-block-group-is-layout-constrained}
# DeepSeek-R1 and FP8 Mixed-Precision Training {#deepseek-r1-and-fp8-mixed-precision-training .wp-block-post-title style="margin-bottom:var(--wp--preset--spacing--40);"}
:::

:::::: {.entry-content .wp-block-post-content .has-global-padding .is-layout-constrained .wp-block-post-content-is-layout-constrained}
[DeepSeek](https://www.deepseek.com/) has shocked the world with the
release of their reasoning model
[DeepSeek-R1](https://arxiv.org/abs/2501.12948). Similar to OpenAI's o1
and Google Gemini's Flash Thinking, the R1 model aims to improve the
quality of its replies by generating a "[chain of
thought](https://arxiv.org/abs/2201.11903){data-type="link"
data-id="https://arxiv.org/abs/2201.11903"}" before responding to a
prompt. The excitement around R1 stems from it achieving parity with o1
on several industry-standard benchmarks, including math, coding, and
English and Chinese language understanding, while also being open-source
and available through the DeepSeek API at a [fraction of the
cost](https://api-docs.deepseek.com/quick_start/pricing).

DeepSeek's technical reports cover a wide swath of performance
optimization techniques that enabled their breakthrough results on
efficient LLM training and inference. Many of these techniques were
already used to train DeepSeek-V3, a model comparable to Anthropic's
Claude Sonnet and OpenAI's GPT-4o, from which the R1 model was obtained
via fine-tuning and reinforcement learning. In this blog post, we'll
focus, in particular, on DeepSeek's FP8 mixed-precision training
strategy **for the base DeepSeek-V3 model**, described in section 3.3 of
the [DeepSeek-V3
paper](https://arxiv.org/abs/2412.19437v1){data-type="link"
data-id="https://arxiv.org/abs/2412.19437v1"} and in the figure below
(Figure 6 of that paper).

As always, a core bottleneck is matrix multiplication (aka "matmul" or
"GEMM"), indicated by the yellow boxes in the diagram. As the figure
shows, model weights are stored in FP8 and all matrix multiplications
are performed in FP8 with FP32 accumulation. Activations and gradients
are stored in BF16, and FP32 is also used for some internal
computations.

<figure class="wp-block-image aligncenter size-large">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/01/image.png?resize=960%2C289&amp;ssl=1"
class="wp-image-15060" data-recalc-dims="1" decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/01/image.png?resize=960%2C289&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/01/image.png?resize=480%2C145&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/01/image.png?resize=768%2C232&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/01/image.png?w=1101&amp;ssl=1 1101w"
sizes="(max-width: 960px) 100vw, 960px" width="960" height="289" />
<figcaption>Figure 6 from the DeepSeek-V3 paper, showing the variety of
float precisions that are used in their Linear layer.</figcaption>
</figure>

Why do we care about FP8 training? On NVIDIA GPUs, GEMM computations can
take advantage of hardware acceleration provided by the GPU's Tensor
Cores. On the Hopper architecture, FP8 GEMM is natively-supported and
achieves the highest possible compute throughput, [advertised at \~2
petaFLOPS on the H100 SXM
GPU](https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet).
In fact, NVIDIA finds low-precision computation so important that it's
expanding Tensor Core capabilities to FP4 and FP6 with Blackwell.
Storing model weights in low precision also reduces the overall size of
the model, placing less pressure on the memory and inter-GPU
communication channels, which are already being pushed to their limits
to keep up with the Tensor Cores.

Working in FP8 comes with several tradeoffs. First, to prevent overflow,
one typically scales a higher-precision weight or activation matrix down
to the FP8 representable range before quantizing it --- for example, by
dividing the whole tensor by its maximum element. That maximum element
is retained separately and used as a scaling factor in each matmul with
the quantized tensor. However, this makes the quantization process
extremely sensitive to outliers: the presence of a very large weight in
some layer could force all other weights to be quantized to 0. The
DeepSeek team handles this issue by introducing *blockwise* and
*tilewise* scaling, in which each 128×128 submatrix of a weight matrix,
respectively each 1×128 subvector of an activation vector, is scaled and
quantized separately. Then, due to the presence of varying scaling
factors along the "inner" or "contracting" dimension of the GEMM, the
rescaling computations need to be *fused* into the matmul mainloop. This
required the team to write a custom FP8-GEMM-with-rescaling kernel. We
also remark that blockwise quantization only (i.e., also for
activations) proved insufficient for their purposes due to training
instability; cf. the ablation study described in appendix B.2 of the
paper.

Furthermore, optimal GEMM on Hopper GPUs uses warpgroup-wide MMA
instructions (WGMMA), which we described in detail as [part of our GEMM
tutorial](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/).
Under these instructions, all of the Tensor Cores on a Hopper GPU's
Streaming Multiprocessor (SM) collaborate to compute fragments of the
matrix product. However, this brings us to the second issue with FP8.
The DeepSeek researchers found that the FP8 Tensor Cores were using a
certain "fixed-point accumulation" strategy that effectively used only
14 bits of precision as opposed to true FP32 precision; cf. section
3.5.2 of the paper. This led to training inaccuracies that grew for
large model sizes.

DeepSeek's solution was to move some of the accumulation outside the
Tensor Cores. Their GEMM kernel performs each series of 4 consecutive
WGMMA operations inside the Tensor Cores, accumulating in the
lower-precision format, but then adds the result into a separate
register-backed accumulator tensor in FP32. This second addition is
performed using CUDA Cores (the GPU's standard execution unit for
non-matmul FP32 arithmetic) and thus takes place in ordinary FP32
precision, mitigating the loss of accuracy. The dequantizing scaling
factor is also applied to this FP32 accumulator.

<figure class="wp-block-image aligncenter size-full">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/01/filled.png?resize=600%2C447&amp;ssl=1"
class="wp-image-15117" data-recalc-dims="1" decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/01/filled.png?w=600&amp;ssl=1 600w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/01/filled.png?resize=362%2C270&amp;ssl=1 362w"
sizes="(max-width: 600px) 100vw, 600px" width="600" height="447" />
<figcaption>Figure 7(b) of the paper: the mixed-precision matmul
technique used during training of DeepSeek-V3, in which lower-precision
WGMMA operations on Tensor Cores alternate with higher-precision
accumulation in CUDA Cores.</figcaption>
</figure>

The paper authors cite NVIDIA's [CUTLASS
library](https://github.com/NVIDIA/cutlass) for this technique. CUTLASS
has supported [promotion of FP8 matmul to FP32 accumulation in CUDA
Cores](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/collective/fp8_accumulation.hpp)
since version 3.2. Moreover, blockwise scaling was added in [this
PR](https://github.com/NVIDIA/cutlass/pull/1932){data-type="link"
data-id="https://github.com/NVIDIA/cutlass/pull/1932"} and merged to
main in version 3.7, and tilewise scaling will soon be supported thanks
to [this
PR](https://github.com/NVIDIA/cutlass/pull/2037){data-type="link"
data-id="https://github.com/NVIDIA/cutlass/pull/2037"} (which renames
the concept to *groupwise* scaling for clarity). As a CUTLASS user, you
can invoke Hopper FP8 GEMM with promoted FP32 accumulation and blockwise
scaling through the
[`CollectiveBuilder`](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl)
with the `KernelScheduleType` set to
`KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum` (cf. [example
67](https://github.com/NVIDIA/cutlass/blob/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling.cu){data-type="link"
data-id="https://github.com/NVIDIA/cutlass/blob/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling.cu"}).
In fact, CUTLASS's Hopper FP8 GEMM kernels use the CUDA Core
accumulation technique by default. Alternatively, you can accumulate
only in the Tensor Cores using schedules such as
[`KernelTmaWarpSpecializedFP8FastAccum`](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl#L491);
this trades better performance for lower accuracy, which may work better
for inference applications.

At Colfax, we're working to spread knowledge of these techniques so that
anyone can take advantage of the optimizations that were central to
DeepSeek's success. If you'd like to learn more about using the CUTLASS
library to build highly performant GEMM kernels, our [tutorial
series](https://research.colfax-intl.com/category/papers/tutorials/){data-type="link"
data-id="https://research.colfax-intl.com/blog/"} is a great place to
start. If you are interested in customized training or have a more
involved problem that could benefit from our expertise, please get in
touch with our research team at <services@colfax-intl.com>.

::::: {.sharedaddy .sd-sharing-enabled}
:::: {.robots-nocontent .sd-block .sd-social .sd-social-icon-text .sd-sharing}
### Share this: {#share-this .sd-title}

::: sd-content
- [[Share on LinkedIn (Opens in new window)]{#sharing-linkedin-15059
  hidden=""}
  LinkedIn](https://research.colfax-intl.com/deepseek-r1-and-fp8-mixed-precision-training/?share=linkedin){.share-linkedin
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-linkedin-15059" target="_blank"
  aria-labelledby="sharing-linkedin-15059"}
- [[Share on X (Opens in new window)]{#sharing-twitter-15059 hidden=""}
  X](https://research.colfax-intl.com/deepseek-r1-and-fp8-mixed-precision-training/?share=twitter){.share-twitter
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-twitter-15059" target="_blank"
  aria-labelledby="sharing-twitter-15059"}
- [[Share on Facebook (Opens in new window)]{#sharing-facebook-15059
  hidden=""}
  Facebook](https://research.colfax-intl.com/deepseek-r1-and-fp8-mixed-precision-training/?share=facebook){.share-facebook
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-facebook-15059" target="_blank"
  aria-labelledby="sharing-facebook-15059"}
- [[Share on Reddit (Opens in new window)]{#sharing-reddit-15059
  hidden=""}
  Reddit](https://research.colfax-intl.com/deepseek-r1-and-fp8-mixed-precision-training/?share=reddit){.share-reddit
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-reddit-15059" target="_blank"
  aria-labelledby="sharing-reddit-15059"}
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
January 27, 2025
:::

in

::: {.taxonomy-category .wp-block-post-terms}
[Article](https://research.colfax-intl.com/category/article/){rel="tag"}[,
]{.wp-block-post-terms__separator}[Blog](https://research.colfax-intl.com/category/blog/){rel="tag"}
:::
:::::
::::::
:::::::
::::::::
::::::::::

:::: {.section .wp-block-template-part}
::: {.wp-block-group .has-global-padding .is-layout-constrained .wp-container-core-group-is-layout-a666d811 .wp-block-group-is-layout-constrained style="padding-top:var(--wp--preset--spacing--40);padding-right:var(--wp--preset--spacing--40);padding-bottom:var(--wp--preset--spacing--40);padding-left:var(--wp--preset--spacing--40)"}
:::
::::
:::::::::::::::::::::::
