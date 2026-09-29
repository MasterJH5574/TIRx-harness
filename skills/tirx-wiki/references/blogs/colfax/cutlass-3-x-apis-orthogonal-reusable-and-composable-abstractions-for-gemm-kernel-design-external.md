::::::::::::::::::::::::::::: {.wp-block-group .is-layout-flow .wp-block-group-is-layout-flow role="main" style="margin-top:var(--wp--preset--spacing--50)"}
::: {.wp-block-group .has-global-padding .is-layout-constrained .wp-block-group-is-layout-constrained}
# CUTLASS 3.x APIs: Orthogonal, Reusable, and Composable Abstractions for GEMM Kernel Design (External) {#cutlass-3.x-apis-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design-external .wp-block-post-title style="margin-bottom:var(--wp--preset--spacing--40);"}
:::

:::::::::::: {.entry-content .wp-block-post-content .has-global-padding .is-layout-constrained .wp-block-post-content-is-layout-constrained}
In this [blog
post](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/){data-type="link"
data-id="https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/"}
presented on the NVIDIA technical blog, we give a concise introduction
to the CUTLASS 3.x APIs, focusing on the collective, kernel, and device
layers and the functionality of the collective builders. This post was
authored in conjunction with members of the CUTLASS team.

:::::::: {.vlp-link-container .vlp-layout-basic .wp-block-visual-link-preview-link}
[](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/ "CUTLASS 3.x: Orthogonal, Reusable, and Composable Abstractions for GEMM Kernel Design | NVIDIA Technical Blog"){.vlp-link
rel="nofollow" target="_blank"}

:::: vlp-layout-zone-side
::: {.vlp-block-2 .vlp-link-image}
![](https://i0.wp.com/developer-blogs.nvidia.com/wp-content/uploads/2024/07/cutlass-featured.png?ssl=1){recalc-dims="1"
decoding="async" style="max-width: 150px; max-height: 150px"}
:::
::::

::::: vlp-layout-zone-main
::: {.vlp-block-0 .vlp-link-title}
CUTLASS 3.x: Orthogonal, Reusable, and Composable Abstractions for GEMM
Kernel Design \| NVIDIA Technical Blog
:::

::: {.vlp-block-1 .vlp-link-summary}
GEMM optimization on GPUs is a modular problem. Performant
implementations need to specify hyperparameters such as tile shapes,
math and copy instructions, and warp-specialization schemes.
:::
:::::
::::::::

::::: {.sharedaddy .sd-sharing-enabled}
:::: {.robots-nocontent .sd-block .sd-social .sd-social-icon-text .sd-sharing}
### Share this: {#share-this .sd-title}

::: sd-content
- [[Share on LinkedIn (Opens in new window)]{#sharing-linkedin-15980
  hidden=""}
  LinkedIn](https://research.colfax-intl.com/cutlass-3-x-apis-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design-external/?share=linkedin){.share-linkedin
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-linkedin-15980" target="_blank"
  aria-labelledby="sharing-linkedin-15980"}
- [[Share on X (Opens in new window)]{#sharing-twitter-15980 hidden=""}
  X](https://research.colfax-intl.com/cutlass-3-x-apis-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design-external/?share=twitter){.share-twitter
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-twitter-15980" target="_blank"
  aria-labelledby="sharing-twitter-15980"}
- [[Share on Facebook (Opens in new window)]{#sharing-facebook-15980
  hidden=""}
  Facebook](https://research.colfax-intl.com/cutlass-3-x-apis-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design-external/?share=facebook){.share-facebook
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-facebook-15980" target="_blank"
  aria-labelledby="sharing-facebook-15980"}
- [[Share on Reddit (Opens in new window)]{#sharing-reddit-15980
  hidden=""}
  Reddit](https://research.colfax-intl.com/cutlass-3-x-apis-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design-external/?share=reddit){.share-reddit
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-reddit-15980" target="_blank"
  aria-labelledby="sharing-reddit-15980"}
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
July 19, 2025
:::

in

::: {.taxonomy-category .wp-block-post-terms}
[Blog](https://research.colfax-intl.com/category/blog/){rel="tag"}
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
