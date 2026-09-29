::::::::::::::::::::::::: {.wp-block-group .is-layout-flow .wp-block-group-is-layout-flow role="main" style="margin-top:var(--wp--preset--spacing--50)"}
::: {.wp-block-group .has-global-padding .is-layout-constrained .wp-block-group-is-layout-constrained}
# A Case Study in CUDA Kernel Fusion: Implementing FlashAttention-2 on NVIDIA Hopper Architecture using the CUTLASS Library {#a-case-study-in-cuda-kernel-fusion-implementing-flashattention-2-on-nvidia-hopper-architecture-using-the-cutlass-library .wp-block-post-title style="margin-bottom:var(--wp--preset--spacing--40);"}
:::

:::::: {.entry-content .wp-block-post-content .has-global-padding .is-layout-constrained .wp-block-post-content-is-layout-constrained}
We provide an optimized implementation of the forward pass of
FlashAttention-2, a popular memory-aware scaled dot-product attention
algorithm, as a custom fused CUDA^®^ kernel targeting NVIDIA Hopper^™^
architecture and written using the open-source CUTLASS library. In doing
so, we explain the challenges and techniques involved in fusing
online-softmax with back-to-back GEMM kernels, utilizing the
Hopper-specific Tensor Memory Accelerator (TMA) and Warpgroup
Matrix-Multiply-Accumulate (WGMMA) instructions, defining and
transforming CUTLASS Layouts and Tensors, overlapping copy and GEMM
operations, and choosing optimal tile sizes for the Q, K and V attention
matrices while balancing the register pressure and shared memory
utilization. In head-to-head benchmarks on a single NVIDIA^®^ H100
Tensor Core PCIe GPU for some common choices of hyperparameters, we
observe 20-50% higher FLOPs/s over a version of FlashAttention-2
optimized for last-generation NVIDIA Ampere architecture.

![](/wp-content/uploads/2023/10/PDF_32.png){.download-link}
[colfax-flashattention.pdf](https://research.colfax-intl.com/download/colfax-flashattention/?tmstv=1771876184){#download-link-9090
.download-link e-disable-page-transition="true" rel="nofollow"
redirect="false"}

::::: {.sharedaddy .sd-sharing-enabled}
:::: {.robots-nocontent .sd-block .sd-social .sd-social-icon-text .sd-sharing}
### Share this: {#share-this .sd-title}

::: sd-content
- [[Share on LinkedIn (Opens in new window)]{#sharing-linkedin-9050
  hidden=""}
  LinkedIn](https://research.colfax-intl.com/nvidia-hopper-flashattention-2/?share=linkedin){.share-linkedin
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-linkedin-9050" target="_blank"
  aria-labelledby="sharing-linkedin-9050"}
- [[Share on X (Opens in new window)]{#sharing-twitter-9050 hidden=""}
  X](https://research.colfax-intl.com/nvidia-hopper-flashattention-2/?share=twitter){.share-twitter
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-twitter-9050" target="_blank"
  aria-labelledby="sharing-twitter-9050"}
- [[Share on Facebook (Opens in new window)]{#sharing-facebook-9050
  hidden=""}
  Facebook](https://research.colfax-intl.com/nvidia-hopper-flashattention-2/?share=facebook){.share-facebook
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-facebook-9050" target="_blank"
  aria-labelledby="sharing-facebook-9050"}
- [[Share on Reddit (Opens in new window)]{#sharing-reddit-9050
  hidden=""}
  Reddit](https://research.colfax-intl.com/nvidia-hopper-flashattention-2/?share=reddit){.share-reddit
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-reddit-9050" target="_blank"
  aria-labelledby="sharing-reddit-9050"}
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
December 5, 2023
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
### Leave a Reply [[Cancel reply](/nvidia-hopper-flashattention-2/#respond){#cancel-comment-reply-link rel="nofollow" style="display:none;"}]{.small} {#reply-title .comment-reply-title}

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
