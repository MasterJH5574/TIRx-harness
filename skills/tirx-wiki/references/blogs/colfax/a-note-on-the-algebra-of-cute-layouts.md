::::::::::::::::::::::::: {.wp-block-group .is-layout-flow .wp-block-group-is-layout-flow role="main" style="margin-top:var(--wp--preset--spacing--50)"}
::: {.wp-block-group .has-global-padding .is-layout-constrained .wp-block-group-is-layout-constrained}
# A note on the algebra of CuTe Layouts {#a-note-on-the-algebra-of-cute-layouts .wp-block-post-title style="margin-bottom:var(--wp--preset--spacing--40);"}
:::

:::::: {.entry-content .wp-block-post-content .has-global-padding .is-layout-constrained .wp-block-post-content-is-layout-constrained}
The core abstraction of NVIDIA's CUTLASS library for high-performance
linear algebra is the CuTe Layout. In this technical note, we give a
rigorous, mathematical treatment of the algebra of these layouts and
certain layout operations. Currently, the main goal is to lay down
conditions for when the operations of complementation, composition, and
logical division are well-defined, which may be of general use to
CUTLASS developers. This note should be read as complementary to the
discussion of these layout operations in the CuTe documentation.\
\
1/8/24: added a section on permutations expressible as layout functions.

![](/wp-content/uploads/2023/10/PDF_32.png){.download-link}
[layout_algebra.pdf](https://research.colfax-intl.com/download/cute-layout-algebra/?tmstv=1771876183){#download-link-9117
.download-link e-disable-page-transition="true" rel="nofollow"
redirect="false"}

::::: {.sharedaddy .sd-sharing-enabled}
:::: {.robots-nocontent .sd-block .sd-social .sd-social-icon-text .sd-sharing}
### Share this: {#share-this .sd-title}

::: sd-content
- [[Share on LinkedIn (Opens in new window)]{#sharing-linkedin-9071
  hidden=""}
  LinkedIn](https://research.colfax-intl.com/a-note-on-the-algebra-of-cute-layouts/?share=linkedin){.share-linkedin
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-linkedin-9071" target="_blank"
  aria-labelledby="sharing-linkedin-9071"}
- [[Share on X (Opens in new window)]{#sharing-twitter-9071 hidden=""}
  X](https://research.colfax-intl.com/a-note-on-the-algebra-of-cute-layouts/?share=twitter){.share-twitter
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-twitter-9071" target="_blank"
  aria-labelledby="sharing-twitter-9071"}
- [[Share on Facebook (Opens in new window)]{#sharing-facebook-9071
  hidden=""}
  Facebook](https://research.colfax-intl.com/a-note-on-the-algebra-of-cute-layouts/?share=facebook){.share-facebook
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-facebook-9071" target="_blank"
  aria-labelledby="sharing-facebook-9071"}
- [[Share on Reddit (Opens in new window)]{#sharing-reddit-9071
  hidden=""}
  Reddit](https://research.colfax-intl.com/a-note-on-the-algebra-of-cute-layouts/?share=reddit){.share-reddit
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-reddit-9071" target="_blank"
  aria-labelledby="sharing-reddit-9071"}
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
December 14, 2023
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

#### One response to "A note on the algebra of CuTe Layouts" {#comments .wp-block-comments-title .has-medium-font-size}

1.  :::::::::::: {#comment-513}
    ::::::::::: {.wp-block-columns .is-layout-flex .wp-container-core-columns-is-layout-a5cf46d6 .wp-block-columns-is-layout-flex style="margin-bottom:var(--wp--preset--spacing--40)"}
    :::: {.wp-block-column .is-layout-flow .wp-block-column-is-layout-flow style="flex-basis:40px"}
    ::: wp-block-avatar
    ![ Avatar](https://secure.gravatar.com/avatar/?s=40&r=g){.avatar
    .avatar-40 .photo .avatar-default .wp-block-avatar__image
    srcset="https://secure.gravatar.com/avatar/?s=80&r=g 2x" height="40"
    width="40" style="border-radius:20px;" decoding="async"}
    :::
    ::::

    :::::::: {.wp-block-column .is-layout-flow .wp-block-column-is-layout-flow}
    ::: wp-block-comment-author-name
    Anonymous
    :::

    :::: {.wp-block-group .is-layout-flex .wp-block-group-is-layout-flex style="margin-top:0px;margin-bottom:0px"}
    ::: wp-block-comment-date
    [June 9,
    2025](https://research.colfax-intl.com/a-note-on-the-algebra-of-cute-layouts/#comment-513)
    :::
    ::::

    ::: wp-block-comment-content
    Excellent notes, really save my day!!!
    :::

    ::: wp-block-comment-reply-link
    [Reply](https://research.colfax-intl.com/a-note-on-the-algebra-of-cute-layouts/?replytocom=513#respond){.comment-reply-link
    rel="nofollow" commentid="513" postid="9071"
    belowelement="comment-513" respondelement="respond"
    replyto="Reply to Anonymous" aria-label="Reply to Anonymous"}
    :::
    ::::::::
    :::::::::::
    ::::::::::::

::: {#respond .comment-respond .wp-block-post-comments-form}
### Leave a Reply [[Cancel reply](/a-note-on-the-algebra-of-cute-layouts/#respond){#cancel-comment-reply-link rel="nofollow" style="display:none;"}]{.small} {#reply-title .comment-reply-title}

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
