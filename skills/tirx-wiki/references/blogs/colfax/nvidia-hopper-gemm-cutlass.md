::::::::::::::::::::::::: {.wp-block-group .is-layout-flow .wp-block-group-is-layout-flow role="main" style="margin-top:var(--wp--preset--spacing--50)"}
::: {.wp-block-group .has-global-padding .is-layout-constrained .wp-block-group-is-layout-constrained}
# Developing CUDA Kernels for GEMM on NVIDIA Hopper Architecture using CUTLASS {#developing-cuda-kernels-for-gemm-on-nvidia-hopper-architecture-using-cutlass .wp-block-post-title style="margin-bottom:var(--wp--preset--spacing--40);"}
:::

:::::: {.entry-content .wp-block-post-content .has-global-padding .is-layout-constrained .wp-block-post-content-is-layout-constrained}
We explain how to develop NVIDIA CUDA^®^ kernels for optimized general
matrix multiplication (GEMM) on NVIDIA Hopper^™^ architecture using the
template collection CUTLASS and its core library CuTe. Our main
contribution is to provide an implementation of a GEMM kernel that uses
the Tensor Memory Accelerator (TMA) and Warp Group
Matrix-Multiply-Accumulate (WGMMA) operations introduced with NVIDIA
Hopper^™^ architecture.

![](/wp-content/uploads/2023/10/PDF_32.png){.download-link}
[colfax-gemm-kernels-hopper.pdf](https://research.colfax-intl.com/download/colfax-gemm-kernels-hopper-pdf/?tmstv=1771876185){#download-link-9103
.download-link e-disable-page-transition="true" rel="nofollow"
redirect="false"}

::::: {.sharedaddy .sd-sharing-enabled}
:::: {.robots-nocontent .sd-block .sd-social .sd-social-icon-text .sd-sharing}
### Share this: {#share-this .sd-title}

::: sd-content
- [[Share on LinkedIn (Opens in new window)]{#sharing-linkedin-9012
  hidden=""}
  LinkedIn](https://research.colfax-intl.com/nvidia-hopper-gemm-cutlass/?share=linkedin){.share-linkedin
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-linkedin-9012" target="_blank"
  aria-labelledby="sharing-linkedin-9012"}
- [[Share on X (Opens in new window)]{#sharing-twitter-9012 hidden=""}
  X](https://research.colfax-intl.com/nvidia-hopper-gemm-cutlass/?share=twitter){.share-twitter
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-twitter-9012" target="_blank"
  aria-labelledby="sharing-twitter-9012"}
- [[Share on Facebook (Opens in new window)]{#sharing-facebook-9012
  hidden=""}
  Facebook](https://research.colfax-intl.com/nvidia-hopper-gemm-cutlass/?share=facebook){.share-facebook
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-facebook-9012" target="_blank"
  aria-labelledby="sharing-facebook-9012"}
- [[Share on Reddit (Opens in new window)]{#sharing-reddit-9012
  hidden=""}
  Reddit](https://research.colfax-intl.com/nvidia-hopper-gemm-cutlass/?share=reddit){.share-reddit
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-reddit-9012" target="_blank"
  aria-labelledby="sharing-reddit-9012"}
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
October 16, 2023
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

#### One response to "Developing CUDA Kernels for GEMM on NVIDIA Hopper Architecture using CUTLASS" {#comments .wp-block-comments-title .has-medium-font-size}

1.  :::::::::::: {#comment-474}
    ::::::::::: {.wp-block-columns .is-layout-flex .wp-container-core-columns-is-layout-a5cf46d6 .wp-block-columns-is-layout-flex style="margin-bottom:var(--wp--preset--spacing--40)"}
    :::: {.wp-block-column .is-layout-flow .wp-block-column-is-layout-flow style="flex-basis:40px"}
    ::: wp-block-avatar
    ![Shiki
    Avatar](https://secure.gravatar.com/avatar/?s=40&r=g){.avatar
    .avatar-40 .photo .avatar-default .wp-block-avatar__image
    srcset="https://secure.gravatar.com/avatar/?s=80&r=g 2x" height="40"
    width="40" style="border-radius:20px;" decoding="async"}
    :::
    ::::

    :::::::: {.wp-block-column .is-layout-flow .wp-block-column-is-layout-flow}
    ::: wp-block-comment-author-name
    Shiki
    :::

    :::: {.wp-block-group .is-layout-flex .wp-block-group-is-layout-flex style="margin-top:0px;margin-bottom:0px"}
    ::: wp-block-comment-date
    [May 20,
    2025](https://research.colfax-intl.com/nvidia-hopper-gemm-cutlass/#comment-474)
    :::
    ::::

    ::: wp-block-comment-content
    Well organized and very clear tutorial.\
    I think it would be a good start for me to learn cutlass.\
    Thank you for your sharing.
    :::

    ::: wp-block-comment-reply-link
    [Reply](https://research.colfax-intl.com/nvidia-hopper-gemm-cutlass/?replytocom=474#respond){.comment-reply-link
    rel="nofollow" commentid="474" postid="9012"
    belowelement="comment-474" respondelement="respond"
    replyto="Reply to Shiki" aria-label="Reply to Shiki"}
    :::
    ::::::::
    :::::::::::
    ::::::::::::

::: {#respond .comment-respond .wp-block-post-comments-form}
### Leave a Reply [[Cancel reply](/nvidia-hopper-gemm-cutlass/#respond){#cancel-comment-reply-link rel="nofollow" style="display:none;"}]{.small} {#reply-title .comment-reply-title}

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
