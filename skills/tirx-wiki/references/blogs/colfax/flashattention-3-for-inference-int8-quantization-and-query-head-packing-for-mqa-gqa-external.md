::::::::::::::::::::::::::::: {.wp-block-group .is-layout-flow .wp-block-group-is-layout-flow role="main" style="margin-top:var(--wp--preset--spacing--50)"}
::: {.wp-block-group .has-global-padding .is-layout-constrained .wp-block-group-is-layout-constrained}
# FlashAttention-3 for Inference: INT8 Quantization and Query Head Packing for MQA/GQA (External) {#flashattention-3-for-inference-int8-quantization-and-query-head-packing-for-mqagqa-external .wp-block-post-title style="margin-bottom:var(--wp--preset--spacing--40);"}
:::

:::::::::::: {.entry-content .wp-block-post-content .has-global-padding .is-layout-constrained .wp-block-post-content-is-layout-constrained}
In this [blog
post](https://blog.character.ai/optimizing-ai-inference-at-character-ai-part-deux-2/){data-type="link"
data-id="https://research.character.ai/optimizing-ai-inference-at-character-ai-part-deux"}
presented on the Character.AI research blog, we explain two techniques
that are important for using
[FlashAttention-3](https://research.colfax-intl.com/flashattention-3-fast-and-accurate-attention-with-asynchrony-and-low-precision/){data-type="post"
data-id="11531"} for inference:

1.  A general methodology for in-kernel pre-processing of tensors via
    warp specialization, applied to the case of a half INT8 attention
    kernel design that upcasts the V tensor in the producer warpgroup.
2.  Query head packing of the Q tile done for multi-query attention
    (MQA) or grouped query attention (GQA), which is needed to saturate
    bandwidth during the memory-bound decoding phase of inference.

We also give microbenchmark results for both prefill and decode-type
attention workloads, measured on an NVIDIA H100 SXM5 GPU.

:::::::: {.vlp-link-container .vlp-layout-compact .wp-block-visual-link-preview-link}
[](https://blog.character.ai/optimizing-ai-inference-at-character-ai-part-deux-2/ "Optimizing AI Inference at Character.AI (Part Deux)"){.vlp-link
rel="nofollow" target="_blank"}

:::: vlp-layout-zone-side
::: {.vlp-block-2 .vlp-link-image}
![](https://i0.wp.com/blog.character.ai/content/images/size/w1200/2025/08/character-deus-featured-image.png?ssl=1){recalc-dims="1"
decoding="async" style="max-width: 100px; max-height: 100px"}
:::
::::

::::: vlp-layout-zone-main
::: {.vlp-block-0 .vlp-link-title}
Optimizing AI Inference at Character.AI (Part Deux)
:::

::: {.vlp-block-1 .vlp-link-summary}
At Character.AI, we're building personalized AI entertainment. In order
to offer our users engaging, interactive experiences, it's critical we
achieve highly efficient inference, or the process by which LLMs
generate replies. Our last post on this topic looked at several
techniques that contribute to the performance and sustainability
:::
:::::
::::::::

*Joint work with Character.AI*.

::::: {.sharedaddy .sd-sharing-enabled}
:::: {.robots-nocontent .sd-block .sd-social .sd-social-icon-text .sd-sharing}
### Share this: {#share-this .sd-title}

::: sd-content
- [[Share on LinkedIn (Opens in new window)]{#sharing-linkedin-14704
  hidden=""}
  LinkedIn](https://research.colfax-intl.com/flashattention-3-for-inference-int8-quantization-and-query-head-packing-for-mqa-gqa-external/?share=linkedin){.share-linkedin
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-linkedin-14704" target="_blank"
  aria-labelledby="sharing-linkedin-14704"}
- [[Share on X (Opens in new window)]{#sharing-twitter-14704 hidden=""}
  X](https://research.colfax-intl.com/flashattention-3-for-inference-int8-quantization-and-query-head-packing-for-mqa-gqa-external/?share=twitter){.share-twitter
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-twitter-14704" target="_blank"
  aria-labelledby="sharing-twitter-14704"}
- [[Share on Facebook (Opens in new window)]{#sharing-facebook-14704
  hidden=""}
  Facebook](https://research.colfax-intl.com/flashattention-3-for-inference-int8-quantization-and-query-head-packing-for-mqa-gqa-external/?share=facebook){.share-facebook
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-facebook-14704" target="_blank"
  aria-labelledby="sharing-facebook-14704"}
- [[Share on Reddit (Opens in new window)]{#sharing-reddit-14704
  hidden=""}
  Reddit](https://research.colfax-intl.com/flashattention-3-for-inference-int8-quantization-and-query-head-packing-for-mqa-gqa-external/?share=reddit){.share-reddit
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-reddit-14704" target="_blank"
  aria-labelledby="sharing-reddit-14704"}
-
:::
::::
:::::
::::::::::::

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
November 27, 2024
:::

in

::: {.taxonomy-category .wp-block-post-terms}
[Benchmarks](https://research.colfax-intl.com/category/papers/benchmarks/){rel="tag"}[,
]{.wp-block-post-terms__separator}[Deep
Learning](https://research.colfax-intl.com/category/papers/deep-learning/){rel="tag"}[,
]{.wp-block-post-terms__separator}[Publications](https://research.colfax-intl.com/category/papers/){rel="tag"}
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
:::::::::::::::::::::::::::::
