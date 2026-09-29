:::::::::::::::::::::::: {.wp-block-group .is-layout-flow .wp-block-group-is-layout-flow role="main" style="margin-top:var(--wp--preset--spacing--50)"}
::: {.wp-block-group .has-global-padding .is-layout-constrained .wp-block-group-is-layout-constrained}
# GPU Mode: CUTLASS and FlashAttention-3 {#gpu-mode-cutlass-and-flashattention-3 .wp-block-post-title style="margin-bottom:var(--wp--preset--spacing--40);"}
:::

::::::: {.entry-content .wp-block-post-content .has-global-padding .is-layout-constrained .wp-block-post-content-is-layout-constrained}
In this GPU Mode lecture, Jay Shah presents his joint work on
FlashAttention-3 and how to implement the main compute loop in the
algorithm using CUTLASS.

The code discussed in this lecture can be found at [this
commit](https://github.com/Dao-AILab/flash-attention/blob/b2d3fe92ff43edbd650aeba2af5ce0af23515683/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp){data-type="link"
data-id="https://github.com/Dao-AILab/flash-attention/blob/b2d3fe92ff43edbd650aeba2af5ce0af23515683/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp"}
in the FlashAttention-3 codebase.

<figure
class="wp-block-embed is-type-video is-provider-youtube wp-block-embed-youtube wp-embed-aspect-16-9 wp-has-aspect-ratio">
<div class="wp-block-embed__wrapper">
<div class="iframe">
<div id="player">

</div>
<div class="player-unavailable">
<h1 id="an-error-occurred." class="message">An error occurred.</h1>
<div class="submessage">
Unable to execute JavaScript.
</div>
</div>
</div>
</div>
</figure>

::: {.wp-block-file wp-interactive="core/file"}
[cutlass-flashattn3-slides](https://research.colfax-intl.com/wp-content/uploads/2024/11/flash_attn_3_gpu_mode_talk.pdf){#wp-block-file--media-f0a81364-b300-42a8-80c1-7532d814ed83}
:::

**Note**: Slides adapted from a talk given by Tri Dao.

::::: {.sharedaddy .sd-sharing-enabled}
:::: {.robots-nocontent .sd-block .sd-social .sd-social-icon-text .sd-sharing}
### Share this: {#share-this .sd-title}

::: sd-content
- [[Share on LinkedIn (Opens in new window)]{#sharing-linkedin-14675
  hidden=""}
  LinkedIn](https://research.colfax-intl.com/gpu-mode-cutlass-and-flashattention-3/?share=linkedin){.share-linkedin
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-linkedin-14675" target="_blank"
  aria-labelledby="sharing-linkedin-14675"}
- [[Share on X (Opens in new window)]{#sharing-twitter-14675 hidden=""}
  X](https://research.colfax-intl.com/gpu-mode-cutlass-and-flashattention-3/?share=twitter){.share-twitter
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-twitter-14675" target="_blank"
  aria-labelledby="sharing-twitter-14675"}
- [[Share on Facebook (Opens in new window)]{#sharing-facebook-14675
  hidden=""}
  Facebook](https://research.colfax-intl.com/gpu-mode-cutlass-and-flashattention-3/?share=facebook){.share-facebook
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-facebook-14675" target="_blank"
  aria-labelledby="sharing-facebook-14675"}
- [[Share on Reddit (Opens in new window)]{#sharing-reddit-14675
  hidden=""}
  Reddit](https://research.colfax-intl.com/gpu-mode-cutlass-and-flashattention-3/?share=reddit){.share-reddit
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-reddit-14675" target="_blank"
  aria-labelledby="sharing-reddit-14675"}
-
:::
::::
:::::
:::::::

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
November 18, 2024
:::

in

::: {.taxonomy-category .wp-block-post-terms}
[Deep
Learning](https://research.colfax-intl.com/category/papers/deep-learning/){rel="tag"}[,
]{.wp-block-post-terms__separator}[Video](https://research.colfax-intl.com/category/video/){rel="tag"}
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
::::::::::::::::::::::::
