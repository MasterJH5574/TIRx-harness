::::::::::::::::::::::: {.wp-block-group .is-layout-flow .wp-block-group-is-layout-flow role="main" style="margin-top:var(--wp--preset--spacing--50)"}
::: {.wp-block-group .has-global-padding .is-layout-constrained .wp-block-group-is-layout-constrained}
# Categorical Foundations for CuTe Layouts {#categorical-foundations-for-cute-layouts .wp-block-post-title style="margin-bottom:var(--wp--preset--spacing--40);"}
:::

:::::: {.entry-content .wp-block-post-content .has-global-padding .is-layout-constrained .wp-block-post-content-is-layout-constrained}
In GPU programming, performance depends critically on how data is stored
and accessed in memory. While the data we care about is typically
multi-dimensional, the GPU's memory is fundamentally one-dimensional.
This means that when we want to load, store, or otherwise manipulate
data, we need to map its multi-dimensional logical coordinates to
one-dimensional physical coordinates. This mapping, known as a
**layout**, is essential for reading from and writing to memory
correctly and efficiently. Moreover, with respect to the GPU's SIMT
execution model, layouts are used to describe and manipulate
partitionings of threads over data. This is important to ensure
optimized memory access patterns and correct invocation of specialized
hardware instructions, such as those used to target tensor cores.

CUTLASS pioneered a novel approach to layouts that both features shape
and stride tuples of arbitrary nesting and depth, and a "layout algebra"
formed out of certain fundamental operations, such as composition,
complementation, logical division and logical product. These **CuTe
layouts** are incredibly expressive, allowing one to describe
partitioning patterns for all generations of tensor core instructions,
for example. They are also fascinating from a mathematical perspective,
since they feature an unusual and subtle notion of function composition
that demands a theoretical explanation.

In a new paper, we develop a robust mathematical theory underlying this
approach, connecting CuTe layouts and their algebra to the theory of
**categories** and **operads** and developing a new graphical calculus
of layout diagrams for computing their operations. An up-to-date pdf
version of our paper is linked
[here](https://research.colfax-intl.com/download/categories-of-layouts/){data-type="link"
data-id="https://research.colfax-intl.com/download/categories-of-layouts/"},
and the companion git repository may be found
[here](https://github.com/ColfaxResearch/layout-categories).

![](/wp-content/uploads/2023/10/PDF_32.png){.download-link} [Categories
of Layouts
09/24/25](https://research.colfax-intl.com/download/categories-of-layouts/?tmstv=1771876161){#download-link-16203
.download-link e-disable-page-transition="true" rel="nofollow"
redirect="false"}

In this blog post, we give a concise overview of the main ideas and
results of our paper, referring the reader to the paper for a
comprehensive treatment, many worked examples, and proofs. To state our
results, we assume the reader knows the basic idea of a
[category](https://en.wikipedia.org/wiki/Category_(mathematics)) and a
[functor](https://en.wikipedia.org/wiki/Functor); cf. Appendix A of our
paper for a quick introduction. We also assume that the reader is
familiar with the basics of CuTe layouts, which we henceforth simply
refer to as layouts.

## Tractable layouts {#tractable-layouts .wp-block-heading}

While defining and computing layout operations for arbitrary layouts is
difficult, we can develop an intuitive framework for working with
layouts by restricting to **tractable layouts**. These include almost
all layouts one encounters in practice, such as

- **row-major** and **column-major** layouts, which are ubiquitous,
- **compact** layouts, which store data in consecutive memory addresses,
- **projections**, which broadcast multiple copies of data, and
- **dilations**, which enable padded loads and stores.

For now, let's focus on the **flat** case, where the shape and stride of
our layouts are tuples [ (x_1, \\ldots, x_n) ]{.katex-eq
katex-display="false"}, rather than more general nested tuples.
Throughout this article, we always suppose that shape tuples are
comprised of positive integers and stride tuples are comprised of
non-negative integers.

First, we define an ordering [ \\preceq ]{.katex-eq
katex-display="false"} on integer pairs [s : d]{.katex-eq
katex-display="false"} by

[ s : d \\preceq s\^{\\prime} : d\^{\\prime} \\: \\text{ if and only if
} \\: d \< d\^{\\prime} \\: \\text{ or } \\: d = d\^{\\prime} \\text{
and } s \\leq s\^{\\prime}.]{.katex-eq katex-display="true"}

**Definition:** We say that a flat layout

[ L = (s_1, \\ldots, s_m) : (d_1, \\ldots, d_m) ]{.katex-eq
katex-display="true"}

is ***tractable*** if for all pairs of integers [1 \\leq i, j \\leq
m,]{.katex-eq katex-display="false"} the following condition holds:

[ \\text{if } s_i : d\_ i \\preceq s_j : d_j \\text{ and } d_i, d_j
\\neq 0, \\text{ then } s_i d_i \\text{ divides } d_j. ]{.katex-eq
katex-display="true"}

If *L* is tractable, then *L* can be encoded by a **diagram.** For
example,

<figure class="wp-block-image aligncenter size-large is-resized">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/01-examples-of-tuple-mors.png?resize=604%2C540&amp;ssl=1"
class="wp-image-16300" style="width:555px;height:auto"
data-recalc-dims="1" decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/01-examples-of-tuple-mors.png?resize=604%2C540&amp;ssl=1 604w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/01-examples-of-tuple-mors.png?resize=302%2C270&amp;ssl=1 302w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/01-examples-of-tuple-mors.png?resize=768%2C687&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/01-examples-of-tuple-mors.png?resize=1536%2C1374&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/01-examples-of-tuple-mors.png?w=1932&amp;ssl=1 1932w"
sizes="(max-width: 604px) 100vw, 604px" width="604" height="540" />
</figure>

The shape of the encoded layout is the tuple on the left, and the stride
of the encoded layout is determined by the arrows and the tuple on the
right by taking prefix products:

<figure class="wp-block-image aligncenter size-large is-resized">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/02-anatopy-of-tuple-mor.png?resize=960%2C439&amp;ssl=1"
class="wp-image-16301" style="width:606px;height:auto"
data-recalc-dims="1" decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/02-anatopy-of-tuple-mor-scaled.png?resize=960%2C439&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/02-anatopy-of-tuple-mor-scaled.png?resize=480%2C219&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/02-anatopy-of-tuple-mor-scaled.png?resize=768%2C351&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/02-anatopy-of-tuple-mor-scaled.png?resize=1536%2C702&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/02-anatopy-of-tuple-mor-scaled.png?resize=2048%2C936&amp;ssl=1 2048w"
sizes="(max-width: 960px) 100vw, 960px" width="960" height="439" />
</figure>

These diagrams are visual depictions of **morphisms** in a category
**Tuple**:

**Definition:** Let **Tuple** denote the category in which

1.  an object is a tuple [(s_1, \\ldots, s_m) ]{.katex-eq
    katex-display="false"} of positive integers, and
2.  a morphism [ f : (s_1, \\ldots, s_m) \\to (t_1, \\ldots, t_n)
    ]{.katex-eq katex-display="false"} is specified by a map of finite
    pointed sets

[ \\alpha: \\{ \\ast, 1, \\ldots, m \\} \\to \\{ \\ast, 1, \\ldots, n\\}
]{.katex-eq katex-display="true"}

subject to the conditions

a.  [\\alpha(\*) = \*]{.katex-eq katex-display="false"},
b.  if [\\alpha(i) \\neq \*]{.katex-eq katex-display="false"} and
    [\\alpha(i) = \\alpha(i\^{\\prime})]{.katex-eq
    katex-display="false"}, then [i = i\^{\\prime}]{.katex-eq
    katex-display="false"},
c.  if [\\alpha(i) = j \\neq \*]{.katex-eq katex-display="false"}, then
    [s_i = t_j]{.katex-eq katex-display="false"}.

We say such a morphism *f* lies over *α*, and refer to *f* as a ***tuple
morphism***.

Each of the previously depicted diagrams was obtained from a tuple
morphism *f* by depicting the domain [ (s_1, \\ldots, s_m) ]{.katex-eq
katex-display="false"} and codomain [(t_1, \\ldots, t_n)]{.katex-eq
katex-display="false"} of *f* vertically, and drawing an arrow from
[s_i]{.katex-eq katex-display="false"} to [t_j]{.katex-eq
katex-display="false"} if [ \\alpha(i) = j ]{.katex-eq
katex-display="false"}. We can now give a precise definition of the
layout encoded by a tuple morphism.

**Definition:** If *f* is a tuple morphism, then the ***layout encoded
by*** *f* is the layout

[ L_f = (s_1, \\ldots, s_m) : (d_1, \\ldots, d_m) ]{.katex-eq
katex-display="true"}

whose shape is the domain of *f*, and whose stride is given by

[ d_i = \\begin{cases} t_1 \\cdots t\_{j-1} & \\text{if } \\alpha(i) = j
\\\\ 0 & \\text{if } \\alpha(i) = \*. \\end{cases} ]{.katex-eq
katex-display="true"}

An important subtlety is that there are many different tuple morphisms
which encode the same layout. For example, each of the tuple morphisms

<figure class="wp-block-image aligncenter size-large is-resized">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/03-tuple-mors-with-same-layout.png?resize=960%2C275&amp;ssl=1"
class="wp-image-16302" style="width:628px;height:auto"
data-recalc-dims="1" loading="lazy" decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/03-tuple-mors-with-same-layout-scaled.png?resize=960%2C275&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/03-tuple-mors-with-same-layout-scaled.png?resize=480%2C137&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/03-tuple-mors-with-same-layout-scaled.png?resize=768%2C220&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/03-tuple-mors-with-same-layout-scaled.png?resize=1536%2C440&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/03-tuple-mors-with-same-layout-scaled.png?resize=2048%2C586&amp;ssl=1 2048w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="275" />
</figure>

encodes the layout *L* = (4, 5) : (1, 64). The morphism *f* is clearly
the simplest: it does not contain superfluous entries (like the "7" in
*g*), and the entries not hit by *f* are consolidated (unlike the "2"
and "8" in *h*). When a morphism *f* satisfies these properties, we say
*f* has **standard form**.

If we further assume that our layouts *L* and morphisms *f* are
**non-degenerate,** which corresponds to the conditions

[s_i = 1 \\quad \\Rightarrow \\quad d_i = 0, ]{.katex-eq
katex-display="true"}

[s_i = 1 \\quad \\Rightarrow \\quad \\alpha(\*) = \*, ]{.katex-eq
katex-display="true"}

on the layout side and morphism side, respectively, then we can prove
the following correspondence theorem.

**Theorem:** There is a one-to-one correspondence between
**non-degenerate tractable flat layouts** and **non-degenerate tuple
morphisms of standard form**.

If *L* is a non-degenerate tractable layout, we write [f_L]{.katex-eq
katex-display="false"} for the tuple morphism corresponding to *L*, and
refer to [f_L]{.katex-eq katex-display="false"} as the **standard
representation** of *L*.

## Layout functions and the realization functor {#layout-functions-and-the-realization-functor .wp-block-heading}

The most important invariant of a layout *L* is its **layout function**
[\\Phi_L]{.katex-eq katex-display="false"}. When *L* is tractable, its
layout function arises naturally from the category **Tuple** through a
**realization functor**

[ \| \\cdot \| : \\textbf{Tuple} \\to \\textbf{FinSet} ]{.katex-eq
katex-display="true"}

Let's recall the definition of layout functions. In order to do so, we
must first recall the definition of **colexicographic isomorphisms** and
their inverses. It will be convenient to use the notation [ \[0, N) =
\\{ 0, 1, \\ldots, N -- 1 \\} ]{.katex-eq katex-display="false"}.

**Definition**: If [ S = (s_1, \\ldots, s_m) ]{.katex-eq
katex-display="false"} is a tuple of positive integers of size *M*, then
the ***colexicographic isomorphism*** is the function

[ \\mathrm{colex}\_S: \[0, s_1) \\times \\cdots \\times \[0, s_m) \\to
\[0, M) ]{.katex-eq katex-display="true"}

given by

[ \\mathrm{colex}\_S(x_1, \\ldots, x_m) = \\sum\_{i=1}\^m x_i \\cdot s_1
\\cdots s\_{i-1}. ]{.katex-eq katex-display="true"}

The ***inverse colexicographic isomorphism*** is the function

[ \\mathrm{colex}\_S\^{-1}: \[0, M) \\to \[0, s_1) \\times \\cdots
\\times \[0, s_m) ]{.katex-eq katex-display="true"}

given by

[ \\mathrm{colex}\_S\^{-1}(x) = (x_1, \\ldots, x_m) ]{.katex-eq
katex-display="true"}

where

[ x_i = \\lfloor x / (s_1 \\cdots s\_{i-1} ) \\rfloor \\: \\mod s_1
\\cdots s_i. ]{.katex-eq katex-display="true"}

When *L* is tractable, we can recover its layout function by means of a
**realization functor** from **Tuple** to the category of finite sets.

**Theorem:** There is a functor

[ \| \\cdot \| : \\textbf{Tuple} \\to \\textbf{FinSet}, ]{.katex-eq
katex-display="true"}

which we call *realization*, satisfying the following properties:

1.  If *S* is a tuple of size *M*, then [ \|S\| = \[0, M) ]{.katex-eq
    katex-display="false"}.
2.  If *S* and *T* are tuples of size *M* and *N*, respectively, and [f:
    S \\to T]{.katex-eq katex-display="false"} is a tuple morphism, then
    its realization [ \|f\|: \[0, M) \\to \[0, N) \\subset \\mathbb{Z}
    ]{.katex-eq katex-display="false"} is the layout function of [ L_f
    ]{.katex-eq katex-display="false"}.

In particular, this result provides an easy proof that the composition
of tuple morphisms is compatible with composition of layouts, which we
will discuss below.

## Layout Operations {#layout-operations .wp-block-heading}

Many important layout operations, such as coalesce, complement, and
composition, have analogues in the category **Tuple**. Let's take a
closer look at each of these operations.

### Coalesce {#coalesce .wp-block-heading}

If [L = (s_1, \\ldots, s_m) : (d_1, \\ldots, d_m)]{.katex-eq
katex-display="false"} is a layout, then we can coalesce *L* by
iteratively replacing any instance of

[ (\\ldots, s_i, s\_{i+1}, \\ldots) : (\\ldots, d_i, s_i d_i, \\ldots)
]{.katex-eq katex-display="true"}

with

[ (\\ldots, s_i s\_{i+1}, \\ldots) : (\\ldots, d_i, \\ldots) ]{.katex-eq
katex-display="true"}

We denote the resulting layout by *coal*(*L*). For example, if

[ L = (2, 2, 5, 5, 5) : (1, 2, 8, 40, 200) ]{.katex-eq
katex-display="true"}

then

[ \\mathit{coal}(L) = (4, 125) : (1, 8). ]{.katex-eq
katex-display="true"}

This construction has a direct analogue in the category **Tuple**. If
*f* is a tuple morphism, then we can coalesce *f* by collapsing parallel
arrows, and multiplying the corresponding entries. For example:

<figure class="wp-block-image aligncenter size-large is-resized">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/04-coalesce-of-tuple-mor.png?resize=960%2C529&amp;ssl=1"
class="wp-image-16303" style="width:610px;height:auto"
data-recalc-dims="1" loading="lazy" decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/04-coalesce-of-tuple-mor.png?resize=960%2C529&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/04-coalesce-of-tuple-mor.png?resize=480%2C265&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/04-coalesce-of-tuple-mor.png?resize=768%2C423&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/04-coalesce-of-tuple-mor.png?resize=1536%2C847&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/04-coalesce-of-tuple-mor.png?resize=2048%2C1129&amp;ssl=1 2048w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="529" />
</figure>

We prove that the coalesce operation in **Tuple** is compatible with
layout coalesce.

**Theorem:** If *f* is a tuple morphism, then the layout encoded by
*coal*(*f*) is

[ L\_{\\mathit{coal}(f)} = \\mathit{coal}(L_f). ]{.katex-eq
katex-display="true"}

### Complement {#complement .wp-block-heading}

If *L* is a layout and *N* is a positive integer, then *comp*(*L*, *N*)
is a sorted, coalesced layout whose concatenation with *L* is compact.
This means that the layout function of the concatenation is an
isomorphism onto its image. There is a minimal integer *N* with respect
to which *L* admits a complement, and in this case, we write *comp*(*L*)
= *comp*(*L*, *N*). For example, if

[ L = (2, 2, 2):(1, 6, 60) ]{.katex-eq katex-display="true"}

then

[ comp(L) = (3, 5) : (2, 12). ]{.katex-eq katex-display="true"}

Again, there is an analogue of complements in the category **Tuple**. We
can compute the complement [f\^c]{.katex-eq katex-display="false"} of a
tuple morphism *f* by including the entries not hit by *f*. For example,

<figure class="wp-block-image aligncenter size-large is-resized">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/05-complement-of-tuple-mor.png?resize=960%2C452&amp;ssl=1"
class="wp-image-16307" style="width:569px;height:auto"
data-recalc-dims="1" loading="lazy" decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/05-complement-of-tuple-mor.png?resize=960%2C452&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/05-complement-of-tuple-mor.png?resize=480%2C226&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/05-complement-of-tuple-mor.png?resize=768%2C362&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/05-complement-of-tuple-mor.png?resize=1536%2C723&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/05-complement-of-tuple-mor.png?resize=2048%2C964&amp;ssl=1 2048w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="452" />
</figure>

We prove that complements in the category **Tuple** are compatible with
layout complements.

**Theorem:** If *f* is an injective tuple morphism of standard form,
then

[ L\_{f\^c} = \\mathit{comp}(L_f). ]{.katex-eq katex-display="true"}

### Composition {#composition .wp-block-heading}

If *A* and *B* are layouts, then the **composition** *B* ∘ *A* is a
layout such that for any *x* ∈ \[0, *size*(*B* ∘ *A*)), we have

[ \\Phi\_{B \\circ A}(x) = \\Phi_B ( \\Phi_A(x)). ]{.katex-eq
katex-display="true"}

There are other properties which uniquely characterize the layout
*B* ∘ *A*, and we refer the reader to Definition 2.3.7.1 of the paper
for full details. For example, if *A* = (2, 2) : (5, 50) and *B* = (5,
2, 5, 2) : (1, 25, 5, 50), then the composition of *A* and *B* is (2, 2)
: (25, 50).

If *f* and *g* are tuple morphisms with *codomain*(*f*) = *domain*(*g*),
then we can compose *f* and *g* to form the tuple morphism *g* ∘ *f*.
For example, the tuple morphisms

<figure class="wp-block-image aligncenter size-large is-resized">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/06-composable-tuple-mors.png?resize=960%2C376&amp;ssl=1"
class="wp-image-16311" style="width:580px;height:auto"
data-recalc-dims="1" loading="lazy" decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/06-composable-tuple-mors.png?resize=960%2C376&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/06-composable-tuple-mors.png?resize=480%2C188&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/06-composable-tuple-mors.png?resize=768%2C300&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/06-composable-tuple-mors.png?resize=1536%2C601&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/06-composable-tuple-mors.png?resize=2048%2C801&amp;ssl=1 2048w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="376" />
</figure>

are composable, and their composite is as depicted below.

<figure class="wp-block-image aligncenter size-large is-resized">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/07-composition-of-tuple-mors.png?resize=960%2C294&amp;ssl=1"
class="wp-image-16313" style="width:637px;height:auto"
data-recalc-dims="1" loading="lazy" decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/07-composition-of-tuple-mors-scaled.png?resize=960%2C294&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/07-composition-of-tuple-mors-scaled.png?resize=480%2C147&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/07-composition-of-tuple-mors-scaled.png?resize=768%2C235&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/07-composition-of-tuple-mors-scaled.png?resize=1536%2C470&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/07-composition-of-tuple-mors-scaled.png?resize=2048%2C626&amp;ssl=1 2048w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="294" />
</figure>

We prove that composition in the category **Tuple** is compatible with
layout composition.

**Theorem:** If *f* and *g* are composable tuple morphisms, then

[ L\_{g \\circ f} = L_g \\circ L_f. ]{.katex-eq katex-display="true"}

This theorem provides a tool for computing the composition of flat,
tractable layouts *A* and *B*. Namely, we can take the standard
representations [ f = f_A ]{.katex-eq katex-display="false"} and [g =
f_B ]{.katex-eq katex-display="false"} of *A* and *B*, and if these
morphisms happen to be composable, we can obtain the composite of *A*
and *B* via the formula

[ B \\circ A = L\_{g \\circ f}. ]{.katex-eq katex-display="true"}

However, it is not often the case that the standard representations *f*
and *g* of some arbitrarily chosen tractable layouts *A* and *B* are
composable, even if the layouts themselves are. Instead, we may aim to
compute the composition of *A* and *B* by

1.  starting with the standard representations *f* and *g* of *A* and
    *B*,
2.  modifying *f* and *g* to obtain composable morphisms *f'* and *g'*
    which realize to the same layout functions as for *f* and *g*, and
3.  composing *f'* and *g'* to obtain the morphism *g'* ∘ *f'*, which
    encodes *B* ∘ *A*.

In order to make this procedure both rigorous and general, we must
broaden our scope to consider *nested* or *hierarchical* layouts.

## Nested layouts and nested tuple morphisms {#nested-layouts-and-nested-tuple-morphisms .wp-block-heading}

Let's fix some notation and terminology. A ***profile*** is a nested
tuple each of whose entries is the symbol ∗. For example *P* = (∗, (∗,
∗)) and *Q* = ((∗, ∗), ∗, (∗, ∗)) are profiles. A ***nested tuple*** *S*
is uniquely determined by its ***flattening*** [(s_1, \\ldots,
s_m)]{.katex-eq katex-display="false"}, which is an ordinary tuple, and
its profile *P*. It is convenient to write

[ S = (s_1, \\ldots s_m)\_P ]{.katex-eq katex-display="true"}

when working with nested tuples. For example, if *S* = ((2, 2), (5, 5)),
then we can write

[ S = (2, 2, 5, 5)\_P ]{.katex-eq katex-display="true"}

where *P* = ((∗, ∗), (∗, ∗)). If *L* = *S* : *D* is a layout, then *S*
and *D* are required to have the same profile, so we may write a general
layout as

[ L = (s_1, \\ldots, s_m)\_P : (d_1, \\ldots, d_m)\_P. ]{.katex-eq
katex-display="true"}

We refer to the layout

[ L\^\\flat = (s_1, \\ldots, s_m) : (d_1, \\ldots, d_m) ]{.katex-eq
katex-display="true"}

as the ***flattening*** of *L*. Much of our story about flat layouts may
be easily ported to the nested case.

**Definition:** We say a layout *L* is ***tractable*** if its flattening
*L*^♭^ is tractable.

Again, if *L* is tractable, then *L* can be encoded by a **diagram.**
For example,

<figure class="wp-block-image aligncenter size-large is-resized">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/nested_tuple_morphism_1.png?resize=960%2C228&amp;ssl=1"
class="wp-image-16340" style="width:630px" data-recalc-dims="1"
loading="lazy" decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/nested_tuple_morphism_1-scaled.png?resize=960%2C228&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/nested_tuple_morphism_1-scaled.png?resize=480%2C114&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/nested_tuple_morphism_1-scaled.png?resize=768%2C183&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/nested_tuple_morphism_1-scaled.png?resize=1536%2C365&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/nested_tuple_morphism_1-scaled.png?resize=2048%2C487&amp;ssl=1 2048w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="228" />
</figure>

<figure class="wp-block-image aligncenter size-large is-resized">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/nested_tuple_morphism_2.png?resize=960%2C251&amp;ssl=1"
class="wp-image-16341" style="width:655px" data-recalc-dims="1"
loading="lazy" decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/nested_tuple_morphism_2-scaled.png?resize=960%2C251&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/nested_tuple_morphism_2-scaled.png?resize=480%2C125&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/nested_tuple_morphism_2-scaled.png?resize=768%2C201&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/nested_tuple_morphism_2-scaled.png?resize=1536%2C401&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/nested_tuple_morphism_2-scaled.png?resize=2048%2C535&amp;ssl=1 2048w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="251" />
</figure>

These diagrams represent morphisms in a category **Nest**.

**Definition:** Let **Nest** denote the category in which

1.  an object is a nested tuple [(s_1, \\ldots, s_m)\_P ]{.katex-eq
    katex-display="false"} of positive integers, and
2.  a morphism [ f : (s_1, \\ldots, s_m)\_P \\to (t_1, \\ldots, t_n)\_Q
    ]{.katex-eq katex-display="false"} is specified by a map of finite
    pointed sets

[ \\alpha: \\{ \\ast, 1, \\ldots, m \\} \\to \\{ \\ast, 1, \\ldots, n\\}
]{.katex-eq katex-display="true"}

satisfying the same conditions above as for a tuple morphism, namely:

a.  [\\alpha(\*) = \*]{.katex-eq katex-display="false"},
b.  if [\\alpha(i) \\neq \*]{.katex-eq katex-display="false"} and
    [\\alpha(i) = \\alpha(i\^{\\prime})]{.katex-eq
    katex-display="false"}, then [i = i\^{\\prime}]{.katex-eq
    katex-display="false"},
c.  if [\\alpha(i) = j \\neq \*]{.katex-eq katex-display="false"}, then
    [s_i = t_j]{.katex-eq katex-display="false"}.

We say such a morphism *f* lies over *α*, and refer to *f* as a
***nested tuple morphism.***

**Definition:** If *f* is a nested tuple morphism, then the ***layout
encoded by*** *f* is the layout

[ L_f = (s_1, \\ldots, s_m)\_P : (d_1, \\ldots, d_m)\_P ]{.katex-eq
katex-display="true"}

whose shape is the domain of *f*, and whose stride *entries* are given
by

[ d_i = \\begin{cases} t_1 \\cdots t\_{j-1} & \\text{if } \\alpha(i) = j
\\\\ 0 & \\text{if } \\alpha(i) = \*. \\end{cases} ]{.katex-eq
katex-display="true"}

We can define **standard form** and **non-degeneracy** in the nested
case, and we again have a correspondence theorem.

**Theorem:** There is a one-to-one correspondence between
**non-degenerate tractable layouts** and **non-degenerate nested tuple
morphisms of standard form**.

We can compare the categories **Nest** and **Tuple** through the
flattening functor

[ (-)\^\\flat : \\mathbf{Nest} \\to \\mathbf{Tuple}. ]{.katex-eq
katex-display="true"}

In particular, we can postcompose with the realization functor from
**Tuple** to **FinSet** to obtain a realization functor 

[ \| \\cdot \| : \\textbf{Nest} \\to \\textbf{FinSet} ]{.katex-eq
katex-display="true"}

defined on **Nest**, such that it enjoys the same properties as before.

**Theorem:** The realization functor from **Nest** to **FinSet**
satisfies the following properties:

1.  If *S* is a nested tuple of size *M*, then \|*S*\| = \[0, *M*).
2.  If *S* and *T* are nested tuples of size *M* and *N*, respectively,
    and [f : S \\to T ]{.katex-eq katex-display="false"} is a tuple
    morphism, then the realization [\|f\| : \[0, M) \\to \[0, N)
    \\subset \\mathbf{Z} ]{.katex-eq katex-display="false"} is the
    layout function of [ L_f ]{.katex-eq katex-display="false"}.

In particular, this theorem leads to an easy proof of the following
result:

**Theorem:** If *f* and *g* are composable nested tuple morphisms, then

[ L\_{g \\circ f} = L_g \\circ L_f ]{.katex-eq katex-display="true"}

The category **Nest** supports analogues of many important layout
operations such as coalesce, complement, logical division, and logical
product. We summarize these operations and their compatibility with the
corresponding layout operations below.

**Theorem:**

1.  We define a coalesce operation [coal(f)]{.katex-eq
    katex-display="false"} on nested tuple morphisms, which is
    compatible with layout coalesce, in that

[ L\_{\\mathit{coal}(f)} = \\mathit{coal}(L_f). ]{.katex-eq
katex-display="true"}

2.  We define a complement operation [f\^c]{.katex-eq
    katex-display="false"} on nested tuple morphisms, which is
    compatible with layout complements in that if *f* is an injective
    nested tuple morphism of standard form, then

[ L\_{f\^c} = \\mathit{comp}(L_f) ]{.katex-eq katex-display="true"}

3.  We define a notion of *divisibility* of nested tuple morphisms, and
    a logical division operation [f \\oslash g]{.katex-eq
    katex-display="false"} when *g* divides *f*. This operation is
    compatible with logical division of layouts, in that

[ \\mathit{coal}(L\_{f \\oslash g}) = \\mathit{coal}(L_f \\oslash L_g).
]{.katex-eq katex-display="true"}

4.  We define a notion of *product admissibility* for nested tuple
    morphisms, and a logical product operation [f \\otimes g]{.katex-eq
    katex-display="false"} when *f* and *g* are product admissible. This
    operation is compatible with logical products of layouts, in that

[ L\_{f \\otimes g} = L_f \\otimes L_g. ]{.katex-eq
katex-display="true"}

## The composition algorithm {#the-composition-algorithm .wp-block-heading}

Now that we have generalized our story to the nested case, we can
explain our **composition algorithm** which computes the composition
*B* ∘ *A* of tractable layouts *A* and *B* using our categorical
framework. There are several important constructions used in our
algorithm which we have not already discussed, namely **mutual
refinements**, **pullbacks**, and **pushforwards**. We will explain
these concepts in the context of our example, and refer readers to
sections 4.1.2 and 4.1.3 of the paper for full details.

Suppose we want to compute the composition of the layouts *A* = (6, 6) :
(1, 6) and *B* = (12, 3, 6) : (1, 72, 12). Since *A* and *B* are
tractable, we may represent them with the tuple morphisms

<figure class="wp-block-image aligncenter size-large is-resized">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/10-tuple-mors-to-compose.png?resize=960%2C292&amp;ssl=1"
class="wp-image-16319" style="width:625px;height:auto"
data-recalc-dims="1" loading="lazy" decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/10-tuple-mors-to-compose.png?resize=960%2C292&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/10-tuple-mors-to-compose.png?resize=480%2C146&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/10-tuple-mors-to-compose.png?resize=768%2C234&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/10-tuple-mors-to-compose.png?resize=1536%2C468&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/10-tuple-mors-to-compose.png?resize=2048%2C624&amp;ssl=1 2048w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="292" />
</figure>

These morphisms are not composable, since the codomain (6, 6) of *f* is
not equal to the domain (12, 3, 6) of *g*. This means that we can not
use the morphisms *f* and *g* to compute the composite *B* ∘ *A*
directly. We can, however, proceed with our computation by finding a
**mutual refinement** of (6, 6) and (12, 3, 6), as depicted below

<figure class="wp-block-image aligncenter size-large is-resized">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/mutual_refinement.png?resize=960%2C509&amp;ssl=1"
class="wp-image-16343" style="width:356px" data-recalc-dims="1"
loading="lazy" decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/mutual_refinement.png?resize=960%2C509&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/mutual_refinement.png?resize=480%2C254&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/mutual_refinement.png?resize=768%2C407&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/mutual_refinement.png?w=1170&amp;ssl=1 1170w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="509" />
</figure>

Intuitively, such a mutual refinement is a specification of how we can
break up the codomain of *f* and the domain of *g* in a compatible way.
We can use our mutual refinement to convert *f* and *g* into composable
morphisms *f'* and *g'*. In the case of *f*, our mutual refinement
indicates that we should factor the first 6 into (2, 3), and include an
extra 6 in the codomain of *f*:

<figure class="wp-block-image aligncenter size-large is-resized">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/pullback_f_to_f.png?resize=960%2C325&amp;ssl=1"
class="wp-image-16344" style="width:640px" data-recalc-dims="1"
loading="lazy" decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/pullback_f_to_f.png?resize=960%2C325&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/pullback_f_to_f.png?resize=480%2C162&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/pullback_f_to_f.png?resize=768%2C260&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/pullback_f_to_f.png?resize=1536%2C520&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/pullback_f_to_f.png?resize=2048%2C693&amp;ssl=1 2048w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="325" />
</figure>

Rigorously, the construction of *f'* from *f* is an instance of a
**pullback**.

In the case of *g*, our mutual refinement indicates that we should
factor 12 as (6, 2):

<figure class="wp-block-image aligncenter size-large is-resized">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/13-pushforward-of-g-to-g-prime-2.png?resize=960%2C381&amp;ssl=1"
class="wp-image-16328" style="width:510px" data-recalc-dims="1"
loading="lazy" decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/13-pushforward-of-g-to-g-prime-2.png?resize=960%2C381&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/13-pushforward-of-g-to-g-prime-2.png?resize=480%2C191&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/13-pushforward-of-g-to-g-prime-2.png?resize=768%2C305&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/13-pushforward-of-g-to-g-prime-2.png?resize=1536%2C610&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/13-pushforward-of-g-to-g-prime-2.png?resize=2048%2C814&amp;ssl=1 2048w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="381" />
</figure>

Rigorously, the construction of *g'* from *g* is an instance of a
**pushforward**.

The nested tuple morphisms *f'* and *g'* are composable, so we may form
the composite

<figure class="wp-block-image aligncenter size-large is-resized">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/nested_composition.png?resize=960%2C237&amp;ssl=1"
class="wp-image-16346" style="width:800px" data-recalc-dims="1"
loading="lazy" decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/nested_composition-scaled.png?resize=960%2C237&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/nested_composition-scaled.png?resize=480%2C118&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/nested_composition-scaled.png?resize=768%2C189&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/nested_composition-scaled.png?resize=1536%2C379&amp;ssl=1 1536w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/nested_composition-scaled.png?resize=2048%2C505&amp;ssl=1 2048w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="237" />
</figure>

and computing the encoded layout yields

[ B \\circ A = L\_{g\^{\\prime} \\circ f\^{\\prime}} = ((2, 3), 6) :
((6, 72), 1). ]{.katex-eq katex-display="true"}

We have walked through an example of the composition algorithm in
action. We refer the reader to section 4.1.3 of the paper for a full and
precise description of this algorithm, and to section 4.1.4 for further
examples. We would also like to emphasize that since **logical
division** and **logical products** are defined in terms of composition,
this algorithm may be used to compute these operations as well.

## Connections to the theory of operads {#connections-to-the-theory-of-operads .wp-block-heading}

As we hinted at in the introduction, there are some interesting
connections between the theory of layouts that we have developed and the
theory of operads. Explaining them is not at all necessary to understand
or use our results, but they are of independent mathematical interest,
and at any rate served to guide our approach to the subject. In this
final section, oriented towards mathematicians rather than the working
programmer, we tersely discuss some of these connections.

We first describe how the category **Tuple** naturally occurs as a
subcategory of the [categories of
operators](https://ncatlab.org/nlab/show/category+of+operators) of an
operad. We then introduce the operad of profiles and propose an
alternative definition of a category of nested tuples that builds in
refinements as "backwards" morphisms, which contextualizes many of the
maneuvers around refinement done in the composition algorithm. Following
common practice, we identify operads with their category of operators,
e.g., the commutative operad is the category of finite pointed sets.

Consider the partially ordered set **ℤ**~\>0~ of positive integers under
the relation of divisibility: [a \\leq b]{.katex-eq
katex-display="false"} if and only if [a]{.katex-eq
katex-display="false"} divides [b]{.katex-eq katex-display="false"}.
Like with every poset, we have an associated category whose objects are
the set's elements and where [a \\to b]{.katex-eq katex-display="false"}
if and only if [a \\leq b]{.katex-eq katex-display="false"}, and by
abuse of notation we also denote this category as **ℤ**~\>0~. Now,
consider this category as a symmetric monoidal category under the
operation of multiplication and apply the operadic nerve to produce the
operad **ℤ**~\>0~^⊗^, which comes equipped with a structure functor to
the category of finite pointed sets. (For a reference on the operadic
nerve, see Construction 2.1.1.7 in [Higher
Algebra](https://www.math.ias.edu/~lurie/papers/HA.pdf).) We have the
wide subcategory E~0~^⊗^ of finite pointed sets on those maps injective
away from the basepoint, which is the operad encoding a single unary
operation. Then, the pullback of **ℤ**~\>0~ to E~0~^⊗^ identifies with
the definition of **Tuple** excluding condition 2c), and imposing 2c)
defines **Tuple** as a subcategory of this pullback.

From this perspective, how do we then incorporate profiles? Profiles
themselves form a (single-colored, symmetric) operad, whose set of
*n*-ary operations consists of profiles of length *n*, and which we
endow with trivial symmetric group action. Under the operadic nerve,
denote this operad as P^⊗^. One can then consider profiles with labels
in any symmetric monoidal category C^⊗^ by forming the pullback of
operads; for **ℤ**~\>0~^⊗^, denote the resulting pullback as
P**ℤ**~\>0~^⊗^. Then, considering depth 1 profiles, we also get that
**Tuple** is a subcategory of the pullback of P**ℤ**~\>0~^⊗^ over
E~0~^⊗^ (and indeed, of P**ℤ**~\>0~^⊗^ itself).

Moreover, since P**ℤ**~\>0~^⊗^ contains both tuple morphisms and
refinements (because multiplying integers was the monoidal product), it
is a suitable ambient category in which to make more sophisticated
constructions. Specifically, as we saw with the diagrams appearing in
the composition algorithm, it is natural to consider factorizations
followed by tuple morphisms as themselves literally comprising morphisms
in some category. There is a standard construction in category theory
that can do this for us; namely, we can form a certain [category of
spans](https://ncatlab.org/nlab/show/span) in P**ℤ**~\>0~^⊗^, where the
class of forward morphisms are those in the wide subcategory **Tuple**
and the class of backward morphisms **Ref** consists of the [cocartesian
edges](https://ncatlab.org/nlab/show/Cartesian+morphism) over those maps
[ \\alpha: \\{ \\ast, 1, \\ldots, m \\} \\to \\{ \\ast, 1, \\ldots, n\\}
]{.katex-eq katex-display="false"} of finite pointed sets such that

1.  [\\alpha]{.katex-eq katex-display="false"} is **active**: if
    [\\alpha(i) = \\ast]{.katex-eq katex-display="false"}, then [i =
    \\ast]{.katex-eq katex-display="false"}.
2.  [\\alpha]{.katex-eq katex-display="false"} is **surjective**.
3.  [\\alpha]{.katex-eq katex-display="false"} is **non-decreasing**
    when restricted to [\\{ 1, \\ldots, m \\}]{.katex-eq
    katex-display="false"}.

Here, the point of taking cocartesian edges is to consider e.g. maps
[(a, b) \\to c]{.katex-eq katex-display="false"} where [ab =
c]{.katex-eq katex-display="false"} instead of the general case of
[ab]{.katex-eq katex-display="false"} dividing [c]{.katex-eq
katex-display="false"} (i.e., we get exactly the refinements).

Note that for the span construction to be well-defined, we need to check
that one can form pullbacks of morphisms in **Tuple** along those in
**Ref** in P**ℤ**~\>0~^⊗^. However, one can prove this.

Finally, we denote the resulting category of spans as **Span(Tuple,
Ref)**. A typical morphism in this category looks like

<figure class="wp-block-image aligncenter size-large is-resized">
<img
src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/15-ref-span-tuple-mor-example-960x481.png?resize=960%2C481&amp;ssl=1"
class="wp-image-16335" style="width:391px;height:auto"
data-recalc-dims="1" loading="lazy" decoding="async"
srcset="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/15-ref-span-tuple-mor-example.png?resize=960%2C481&amp;ssl=1 960w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/15-ref-span-tuple-mor-example.png?resize=480%2C240&amp;ssl=1 480w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/15-ref-span-tuple-mor-example.png?resize=768%2C385&amp;ssl=1 768w, https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/09/15-ref-span-tuple-mor-example.png?w=1522&amp;ssl=1 1522w"
sizes="auto, (max-width: 960px) 100vw, 960px" width="960"
height="481" />
</figure>

where, following standard notation for span diagrams, we have now drawn
the left grouping to have backwards-pointing arrows.

By definition, **Span(Tuple, Ref)** contains **Tuple** and **Ref^op^**
as subcategories. We can then extend the realization functor from
**Tuple** to **FinSet** over **Span(Tuple, Ref)** so that refinements
are sent to *inverse* colexicographic isomorphisms. Conceptually, this
provides an alternative viewpoint on a category of nested tuples since,
in contrast to **Nest**, the *objects* of the category **Span(Tuple,
Ref)** are flat tuples, but morphisms are nested; this is concordant
with viewing a layout as defining a map valued on the depth 1 reduction
of its shape.

## Revision History {#revision-history .wp-block-heading}

09/24/25: Fixed some typos and improved exposition.\
09/21/25: Initial release.

::::: {.sharedaddy .sd-sharing-enabled}
:::: {.robots-nocontent .sd-block .sd-social .sd-social-icon-text .sd-sharing}
### Share this: {#share-this .sd-title}

::: sd-content
- [[Share on LinkedIn (Opens in new window)]{#sharing-linkedin-16040
  hidden=""}
  LinkedIn](https://research.colfax-intl.com/categorical-foundations-for-cute-layouts/?share=linkedin){.share-linkedin
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-linkedin-16040" target="_blank"
  aria-labelledby="sharing-linkedin-16040"}
- [[Share on X (Opens in new window)]{#sharing-twitter-16040 hidden=""}
  X](https://research.colfax-intl.com/categorical-foundations-for-cute-layouts/?share=twitter){.share-twitter
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-twitter-16040" target="_blank"
  aria-labelledby="sharing-twitter-16040"}
- [[Share on Facebook (Opens in new window)]{#sharing-facebook-16040
  hidden=""}
  Facebook](https://research.colfax-intl.com/categorical-foundations-for-cute-layouts/?share=facebook){.share-facebook
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-facebook-16040" target="_blank"
  aria-labelledby="sharing-facebook-16040"}
- [[Share on Reddit (Opens in new window)]{#sharing-reddit-16040
  hidden=""}
  Reddit](https://research.colfax-intl.com/categorical-foundations-for-cute-layouts/?share=reddit){.share-reddit
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-reddit-16040" target="_blank"
  aria-labelledby="sharing-reddit-16040"}
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
September 21, 2025
:::

in

::: {.taxonomy-category .wp-block-post-terms}
[Blog](https://research.colfax-intl.com/category/blog/){rel="tag"}[,
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
:::::::::::::::::::::::
