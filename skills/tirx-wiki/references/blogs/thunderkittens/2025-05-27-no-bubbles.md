:::::::::: {.container__Wrapper-sc-1hfr0wd-0 .bMusaN}
::: {}
May 27, 2025 · 13 min read

# Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B {#look-ma-no-bubbles-designing-a-low-latency-megakernel-for-llama-1b .Post_title__H6bPg}

[Benjamin Spector\*](https://benjaminfspector.com/), [Jordan
Juravsky\*](https://www.jordanjuravsky.com/), [Stuart
Sul](https://stuartsul.com/), [Owen Dugan](https://druidowm.github.io/),
[Dylan Lim](https://www.linkedin.com/in/dylanllim/), [Dan
Fu](https://danfu.org/), [Simran Arora](https://arorasimran.com/),
[Chris Ré](https://cs.stanford.edu/people/chrismre/)

##
:::

::::::: Post_content__JuwtT
There are some applications that benefit from running LLMs really,
really fast. This low-latency regime encompasses applications like
chatbots and human-in-the-loop workflows, where users care a lot about
seeing responses come back immediately.

Given the importance of these low-latency workloads, we wanted to
explore just how fast we can run open-source models on modern GPUs. To
really stress-test existing systems, we consider an aggressive
low-latency scenario where we generate a single sequence with
Llama-3.2-1B. This workload is strongly memory bound -- our performance
is dominated by how fast we can load model weights from GPU global
memory.

It turns out that popular LLM inference engines -- vLLM and SGLang --
are only able to use at most 50% of available GPU bandwidth when running
this workload on an H100. The root of the problem, which we\'ll describe
more below, is that existing systems break down a model forward pass
into around **a hundred separate kernels** that each implement a few
operations (e.g. RMS norm, attention, an MLP layer + activation,
rotary). Each kernel comes with a setup and teardown period and during
this time no useful work gets done -- for instance, the all-important
task of loading model weights is stalled.

::: {style="text-align:center;padding-top:0"}
![Performance comparison
graph](/static/posts/2025-05-27-no-bubbles/result.png){style="width:100%"}

*Figure 1: Speed! Results generated with a 32-token prompt and 128
generated tokens, with no speculation*
:::

In this post, we show how we can bypass this problem by merging the
entire Llama-1B forward pass into a single \"megakernel\" that
eliminates kernel boundaries altogether. Doing this achieves brr -- on
an H100, we use 78% of memory bandwidth and outperform existing systems
by over 1.5x. (To our knowledge, this is the lowest-latency forward pass
for Llama-1B in bfloat16!) In the rest of this post, we\'ll walk through
how and why one would do this. Specifically:

- First, we\'ll talk about how small kernels lead to AI systems that
  underutilize the GPU\'s full bandwidth.
- Second, we\'ll describe three important points about how we built our
  megakernel: how we fused lots of kernels together, how we share
  hardware resources across them to minimize overhead, and how we
  synchronize them efficiently.

If you\'re interested in learning more of the details or using these
ideas yourself, we\'re [open-sourcing all of our code
here](https://github.com/HazyResearch/Megakernels).

## Separate Kernels Kill the Vibe

In general, the way one runs code on a GPU is by launching a \"kernel\"
-- a small program that does a well-defined operation (e.g. RMS norm,
MLP). Today, all AI workloads run as long sequences of relatively small
kernels. To get an initial sense, let\'s look at the operations in the
Llama-1B transformer block, and some example kernel boundaries of how
they might be divided up (Figure 2).

::: {style="text-align:center;padding-top:0"}
![Kernel boundaries
diagram](/static/posts/2025-05-27-no-bubbles/kernel_boundaries.png){style="width:100%"}

*Figure 2: An example set of kernel boundaries for the Llama-1B
transformer block. Red boxes delineate the work done by individual
kernels.*
:::

As we described earlier, decoding a single sequence with Llama-1B is a
purely memory-bound workload: our performance depends on being able to
**always** be loading weights from GPU global memory. So, why are
existing approaches so far from using the full bandwidth of the GPU?

When we dug into it, we noticed a key problem was that the current
kernel-based approach to running models introduces stalls that prevent
us from constantly loading memory:

- First: GPU kernels are launched with a strict ordering, so that a
  thread block in one kernel can\'t start until all thread blocks in
  previous kernels have completely finished. Consequently, every time we
  start a kernel, we have to wait for all the straggler thread blocks
  from the prior one to finish. For example, if a kernel runs 512 thread
  blocks (like our Llama-1B down projection), but we only have 148
  streaming multiprocessors (like on a B200), we end up with 80 empty
  SM\'s at the end.
- Second, as we\'ve [previously
  highlighted](https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla),
  each kernel launch and teardown incurs costs. In principle, NVIDIA\'s
  CUDA graphs can help hide costs, but by our measurements they still
  leave a lot on the table. For a simple dummy kernel (which dumps a
  start time, sleeps, and dumps an end time) on an H100, we find that
  running on a CUDA stream incurs a launch cost of about 2.1
  microseconds, and with CUDA graphs the launch cost only decreases to
  around 1.3 microseconds -- time spent with the GPU doing no useful
  work! We\'d like to have the GPU spend all of its time doing useful
  work.
- Finally, even after we start the next kernel, we still have to wait to
  load weights and activations before any compute can start. These
  latencies leave the GPU sitting idle for thousands of cycles! Ideally,
  we\'d start loading the next weights while the previous computations
  and stores are happening. NVIDIA has also built a mechanism for this
  called [Programmatic Dependent
  Launch](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization)
  (PDL), which allows the next kernel to start preparing while the
  previous kernel is running, but we found it still introduces
  unnecessary stalls because the PDL synchronization mechanism
  (cudaGridDependencySynchronize) is very coarse. For example, it means
  we have to wait for all queries, keys, and values to complete in order
  to start attention, as opposed to starting heads as soon as they are
  ready. We\'ll later show another specific case of where this is useful
  in Llama-1B.

Taken together, these form the \"memory pipeline bubbles\" our title
references -- and they represent a key reason that we\'re **not always
loading from memory**. For short operations, these pauses add up,
wasting a huge chunk of potential bandwidth. In part, this is because
Llama-1B (actually 1.24B parameters) in batch size 1 is just so\...
small: if each operation is really fast, then the time spent in-between
them really starts to matter.

To illustrate the magnitude of the problem: for single-sequence
generation in 16-bit precision on a single H100, the **memory limit** is
3.35TB/s / 2.48GB = \~1350 forward passes per second. But with 7 kernel
launches per layer, and 16 layers, even with an optimistic 5 us of
stalling per kernel (counting stragglers, kernel launch, and memory
latencies), generation would run at just \~770 forward passes per
second. In practice, it\'s often worse. On low-latency workloads, GPUs
spend only a fraction of their time actually doing any useful work!

So while CUDA does provide some existing features (e.g. graphs, streams,
PDL) to partially solve these problems, we wanted to see if a different
approach could solve all of these problems, where we just fuse the
entire model forward pass into a single kernel.

## How to Megakernel

Next, we\'ll show you how we fused a whole Llama forward pass into a
single kernel, and our methods for resolving three key problems:

1.  Fusing dozens of operations is hard to do from scratch. We need a
    mechanism for executing these operations within the megakernel.
2.  In order to overlap multiple operations on the same hardware, we
    need to prevent contention over limited resources, such as shared
    memory.
3.  The GPU synchronizes after each kernel in the traditional kernel
    model. Without kernels, we have to synchronize the GPU all by
    ourselves!

Let\'s start with the first issue:

#### Issue 1/3: Fusing Lots of Operations

Traditional kernel fusion generally merges just two or three operations
together. In contrast, we need to fuse about a hundred. Consequently, we
need to have a sensible abstraction for how we can actually program a
megakernel.

Our approach is built on an on-GPU interpreter -- essentially a more
sophisticated version of our infrastructure underlying
[ThunderMLA](https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla).
Our interpreter is designed such that each streaming multiprocessor (SM)
within the GPU receives a sequence of **instructions** (each implemented
using the same CUDA template) and executes them. We **schedule** each
SM\'s instruction sequence ahead of time on the Python side, and notably
we can reuse each schedule for hundreds of forward passes!

For our end-to-end Llama forwards pass megakernel, we define the
following set of instructions:

- A fused RMS norm & QKV & RoPE instruction.
- An attention computation instruction.
- An attention reduction instruction (for ThunderGQA on long sequences).
- An O-projection + residual instruction.
- A fused RMS norm & up-gate & SiLU instruction.
- A down-projection + residual instruction.
- An RMS norm & language modeling head instruction, for computing the
  final token logits.

We implement each of these instructions using a common [CUDA
template](https://github.com/HazyResearch/Megakernels/blob/main/util/mk_init/sources/src/%7B%7BPROJECT_NAME_LOWER%7D%7D.cu)
(with load, store, compute boilerplate functions), facilitating
interoperability within our interpreter framework.

#### Issue 2/3: [S]{style="color:#D3222E"}[h]{style="color:#FF7301"}[a]{style="color:#FECD05"}[r]{style="color:#74C419"}[i]{style="color:#009FA0"}[n]{style="color:#2A3992"}[g]{style="color:#5F42B0"} Shared Memory to Eliminate Memory Bubbles

The instruction-and-interpreter structure lets us cleanly organize our
megakernel. However, we haven\'t yet addressed the key issue: making
sure that model weights are always being loaded in order to maximize
memory bandwidth utilization.

The reason why a megakernel lets us solve this problem is that we can
pipeline memory loads across instructions: our interpreter will start
loading the model weights for an instruction as soon as it can, even if
a previous instruction is still finishing up (e.g. storing out its
results to global memory). It\'s this tight transitioning between
instructions that minimizes the memory bubbles that would otherwise
appear if we launched multiple kernels.

However, there\'s a catch: loading the weights from global memory for
the next instruction doesn\'t do you much good if you have no place to
put the data you loaded! More precisely, all of our weight matrices are
loaded from GPU global memory into our SM\'s \"shared memory\" --
NVIDIA\'s term for the fast memory on each SM. Shared memory is a scarce
resource on each SM, and we can\'t start a load for a new instruction if
a previous instruction is using all of it. This necessitates a way to
keep track of which instruction is using which piece of shared memory
and quickly transition shared memory to the next instruction when the
current instruction is done with it.

We accomplish this by **paging** shared memory. We first divide the
first 213kB of shared memory on an H100 into 13 16KiB pages, and use
remaining shared memory for special purposes, like storing instruction
parameters. To use one of these pages, instructions have to explicitly
request and release them from the interpreter. The interpreter
automatically passes released pages to the next instruction, allowing
them to start issuing memory loads as early as shared memory becomes
available.

#### Issue 3/3: Synchronization

::: {style="text-align:center;padding-top:0"}
![Thanos
illustration](/static/posts/2025-05-27-no-bubbles/thanos.png){style="width:80%"}
:::

While megakernels let us minimize pipeline bubbles, they also introduce
a new problem: synchronization. The performance limitation with the
normal many-kernel execution model is that no thread blocks in a kernel
can start until all thread blocks in previous kernels are finished.
However, it\'s precisely this property that makes it easy to manage data
dependencies. When a kernel launches, CUDA guarantees that all of the
kernel\'s input tensors have already been produced and are safe to read
from immediately.

With megakernels, we have no such guarantees: when an SM starts to
execute a new instruction, its inputs might not be ready! To address
this, we explicitly synchronize the instructions inside of our
megakernel. We accomplish this with a simple counter system. Before the
megakernel launches, we initialize an array of counters (i.e. integers)
in GPU global memory with a starting value of zero. Whenever an
instruction completes, it increments one of these counters. Similarly,
whenever a new instruction starts, it must wait for some of these
counters to reach a target value, indicating that all of its
dependencies have finished.

One optimization this enables is in the big multi-layer perceptrons
(MLPs) in Llama-1B.

- In a naive implementation using PDL, one must await completing the
  whole hidden state before beginning the down projection matrix
  multiply.
- We instead produce and consume the intermediate state in four chunks,
  each with their own counter. This way, an instruction for the down
  projection only needs to wait for its input chunk to finish.

## Putting It All Together

To our knowledge, our H100 megakernel represents the first time anyone
has run the forward pass for a 16-bit 1B+ parameter language model in
under one millisecond on a GPU. Our B200 implementation pushes this even
further to under 680 microseconds per forward pass!

As shown in Figure 1, our megakernel outperforms vLLM and SGLang
baselines (which use CUDA graphs and torch compilation):

- On an H100, our megakernel runs almost 2.5x faster than vLLM and over
  1.5x faster than SGLang.
- On a B200, the gap with vLLM rises to over 3.5x, and we remain more
  than 1.5x faster than SGLang, too.

We\'re still actually quite a ways off from the theoretical limit on a
B200, which is around \~3,000 forward passes per second. Part of this
gap is because this theoretical limit is based purely on memory
bandwidth -- but we still have to wait to load activations. And although
these activations are small (and don\'t cost a lot of bandwidth), there
are still latencies in loading them that we can\'t hide. A breakdown of
the runtime of our current B200 forward pass (total runtime 600
microseconds):

- 250 microseconds are spent storing activations, awaiting consistency,
  and loading them. This is about 20% higher than a simple model would
  suggest: since each instruction has a dependence on the last one, we
  need to pay two load latencies (check ready, and then load
  activations) and two store latencies (store activations, then mark
  ready) per instruction. Using \~500 nanoseconds latency per load /
  store, this would impose about 200 microseconds of overhead. (We
  suspect some of the remaining 50 microseconds comes from time spent
  processing atomics in global memory.)
- 200 microseconds are spent actually running RMS norm and matrix-vector
  computations. 95% of this portion is devoted to matrix-vector. On
  Blackwell, we find that using the tensor cores is marginally helpful
  for this; on Hopper, we find it better to simply run on the CUDA
  cores. This difference comes from the fact that both GPUs have
  relatively similar CUDA core performance, but Blackwell tensor cores
  are much faster.
- 30 microseconds are spent awaiting weights from global memory
  (pipelining works!) Of these, 40% are spent in the LM head, which is
  the best-pipelined part of the whole megakernel due to its homogeneity
  and huge size.
- 40 microseconds are spent on low-level synchronization overhead across
  warps. A key issue here is that CUDA\'s asynchronous barriers are
  relatively slow, even when they\'re already in the \"pass\" state,
  requiring about 60 nanoseconds each time.
- 80 microseconds are on setup and various other overheads (e.g. passing
  instruction barriers, marking pages as complete, etc.)

We think there\'s probably more to do on each of these, but that\'ll
have to wait for a future update!

## The Megakernel Cinematic Universe

In this blog, we focus narrowly on designing a megakernel for
low-latency, batch-size one LLM inference. However, we believe that the
ability to more precisely control GPU execution with megakernels can
more generally be applied to accelerate a much broader set of AI
workloads. Stay tuned!

::: {style="text-align:center;padding-top:0"}
![Sonic
illustration](/static/posts/2025-05-27-no-bubbles/sonic.png){style="width:100%"}

**The Main Message of this Blog Post**
:::

If you\'d like to learn more, please reach out to Ben or Jordan! Please
include a tribute of at least five pictures of kittens in your email.

- Ben: <bfs@stanford.edu>
- Jordan: <jbj@stanford.edu>

And many, many thanks to Together AI for generously providing us with
B200s and H100s to do this work, which would not have been possible
without them!

See also: [**pretty big
kernels**](https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla)
\| [**regular
kernels**](https://hazyresearch.stanford.edu/blog/2024-05-12-tk)
:::::::

::: {.spacer__Spacer-sc-16n8pus-0 .iHVKse}
:::
::::::::::
