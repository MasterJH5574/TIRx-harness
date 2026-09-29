:::::::::::::::::::::::::::::::::::::::::::: {.wp-block-group .is-layout-flow .wp-block-group-is-layout-flow role="main" style="margin-top:var(--wp--preset--spacing--50)"}
::: {.wp-block-group .has-global-padding .is-layout-constrained .wp-block-group-is-layout-constrained}
# Tutorial: Python bindings for CUDA libraries in PyTorch {#tutorial-python-bindings-for-cuda-libraries-in-pytorch .wp-block-post-title style="margin-bottom:var(--wp--preset--spacing--40);"}
:::

::::::::::::::::::::::::: {.entry-content .wp-block-post-content .has-global-padding .is-layout-constrained .wp-block-post-content-is-layout-constrained}
PyTorch today is one of the most popular AI frameworks. Developed by
Meta (then Facebook) and open-sourced in 2017, it features approachable,
"pythonic" interfaces. This ease-of-use makes it especially potent for
research and development, where a researcher might need to go through
multiple iterations of novel AI workloads that they are developing.
However, developing in pure Python can have some drawbacks as well, with
one common and major downside being performance. Python often is slower
than languages like C++, and this is especially pronounced if the Python
code does not leverage GPU hardware acceleration at all or only in a
naive way (e.g., not targeting any special characteristics of a
particular GPU architecture for optimization purposes).

In order to take full advantage of code optimizations for NVIDIA^®^ GPUs
from PyTorch, one of the easiest methods is to have PyTorch call
optimized GPU-accelerated libraries. While PyTorch already does this for
many common AI workloads, not all workloads are integrated. And for some
workloads, there may be CUDA^®^ C++ libraries that offer more
performance than what PyTorch uses by default.

Alternatively, a CUDA developer creating a new novel library may want to
make it more accessible by porting it to PyTorch. While there are
libraries like PyCUDA that make CUDA available from Python, C++ is still
the main language for CUDA development. So the CUDA developer might need
to bind their C++ function to a Python call that can be used with
PyTorch.

The PyTorch website already has a very helpful
[guide](https://pytorch.org/tutorials/advanced/cpp_extension.html){target="_blank"
rel="noreferrer noopener"} that walks through the process of writing a
C++ extension. In this article, we will present some additional and
complementary information that we found to be useful when working with
CUDA and CUDA libraries such as
[CUTLASS](https://github.com/NVIDIA/cutlass/){data-type="link"
data-id="https://github.com/NVIDIA/cutlass/"}. By way of explanation, we
will go over an example C++ extension for PyTorch that uses NVIDIA's
CUTLASS library to do GEneral Matrix Multiplication (GEMM). We will
model the Python side interface on
[torch.mm](https://pytorch.org/docs/stable/generated/torch.mm.html){data-type="link"
data-id="https://pytorch.org/docs/stable/generated/torch.mm.html"
target="_blank" rel="noreferrer noopener"}, so that it can be used as a
drop-in replacement. Our aim is to create a complete, working example
that can be used as a template for future development.

## Converting inputs from Torch to CUTLASS {#converting-inputs-from-torch-to-cutlass .wp-block-heading}

We will base our implementation on the CUTLASS `basic_gemm` [example
0](https://github.com/NVIDIA/cutlass/blob/main/examples/00_basic_gemm/basic_gemm.cu){data-type="link"
data-id="https://github.com/NVIDIA/cutlass/blob/main/examples/00_basic_gemm/basic_gemm.cu"}.
For those familiar with CUTLASS, note that this example uses the 2.X
syntax. We will also have a separate example using 3.X syntax that
specifically targets the NVIDIA Hopper^™^ architecture in the addendum
to this article.

First, to simplify let's take that example and wrap it in a single
function call:

::: wp-block-syntaxhighlighter-code
``` {.brush: .cpp; .gutter: .false; .title: .; .notranslate title=""}

template<typename DataType, typename OutputType>
void cutlass_gemm_wrapper(int M, int N, int K,
                          DataType const* ptrA,
                          DataType const* ptrB,
                          OutputType* ptrC);
```
:::

We will then focus our attention on getting the arguments that we need
for this call. Specifically, we need three things:

1.  The shape of the tensors,
2.  The data type of the tensors, and
3.  The pointer to the data.

Our goal is to create a function that takes inputs from PyTorch,
extracts the above information, and calls the CUTLASS wrapper function.

### Input Torch Tensors {#input-torch-tensors .wp-block-heading}

The input arguments for our new function `cutlass_gemm` will be in the
form of the `torch::Tensor` class, which is the C++ representation of
the `torch.Tensor` class in Python. For instance, the function could
have signature:

::: wp-block-syntaxhighlighter-code
``` {.brush: .cpp; .gutter: .false; .title: .; .notranslate title=""}

torch::Tensor cutlass_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C)
```
:::

Note that in the above code, the matrix `C` is left as a required
argument even though it is optional for `mm`. We will fix this in a
later section. 

### Tensor Shapes {#tensor-shapes .wp-block-heading}

In order to extract the data we need for the GEMM, we can leverage the
PyTorch [ATen
API](https://pytorch.org/cppdocs/notes/tensor_basics.html){target="_blank"
rel="noreferrer noopener"}. To start, we can get the shapes of the
tensors using the `.sizes() `method:

::: wp-block-syntaxhighlighter-code
``` {.brush: .cpp; .gutter: .false; .title: .; .notranslate title=""}

auto A_shape = A.sizes();
```
:::

This returns an array (specifically, Torch's `IntArrayRef`) that
contains the shape of the tensor.

### Tensor Data Type {#tensor-data-type .wp-block-heading}

Next we have the data type. Torch tensors have multiple possible data
types that can be recovered using the `.dtype()` method:

::: wp-block-syntaxhighlighter-code
``` {.brush: .cpp; .gutter: .false; .title: .; .notranslate title=""}

auto A_type = A.dtype();
```
:::

This can then be compared to the Torch data types:

::: wp-block-syntaxhighlighter-code
``` {.brush: .cpp; .gutter: .false; .title: .; .notranslate title=""}

bool is_half = (A.dtype() == torch::kHalf);
```
:::

The full list of the different data types can be found
[here](https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/include/torch/types.h).

### Tensor Data Pointers {#tensor-data-pointers .wp-block-heading}

Finally, we can extract the pointer to the data using the `.data_ptr()`
method of the tensors:

::: wp-block-syntaxhighlighter-code
``` {.brush: .cpp; .gutter: .false; .title: .; .notranslate title=""}

float* A_ptr = A.data_ptr<float>();
```
:::

Here the `.data_ptr()` is templated, allowing the developer to cast the
returned pointer to the data type of their choice. Note that this
templating is sufficient if your application only handles default data
types, but it doesn't support custom data types. For example, in CUTLASS
the FP16 data type is `cutlass::half_t`, while the corresponding FP16
data type for which `.data_ptr()` is templated is `torch::kFloat16`.

So instead of the templating, we use `reinterpret_cast` to convert to
the data type we need:

::: wp-block-syntaxhighlighter-code
``` {.brush: .cpp; .gutter: .false; .title: .; .notranslate title=""}

float* A_ptr = reinterpret_cast<float*>(a.data_ptr());
```
:::

For our example, we will have CUTLASS use whatever data type the user
inputted. So we can use the datatype we found in the previous step to
cast to the correct precision. To do this we put the `reinterpret_cast`
inside an intermediate function and use C++ templates to pass the
datatype.

::: wp-block-syntaxhighlighter-code
``` {.brush: .cpp; .title: .; .notranslate title=""}

template<typename DataType, typename OutputType>
void cutlass_gemm_unpack(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  // Get data sizes
  const int M = A.sizes()[0];
  const int K = B.sizes()[0];
  const int N = B.sizes()[1];

  // Casting to the data type of the input tensor
  DataType const *ptrA = reinterpret_cast<DataType*>(A.data_ptr());
  DataType const *ptrB = reinterpret_cast<DataType*>(B.data_ptr());
  DataType *ptrC = reinterpret_cast<OutputType*>(C.data_ptr());
  cutlass_gemm_wrapper<DataType, OutputType>(M, N, K, ptrA, ptrB, ptrC);
}
```
:::

Note that template parameters are resolved at compile-time, but here we
need to select the correct template instantiation of
`cutlass_gemm_unpack` based on the data types of `A` and `C`, which we
know at run-time. To do this, we can introduce some conditional logic,
for example like so:

::: wp-block-syntaxhighlighter-code
``` {.brush: .cpp; .gutter: .false; .title: .; .notranslate title=""}

if(A.dtype() == torch::kFloat16 && C.dtype() == torch::kFloat32)
    cutlass_gemm_unpack<cutlass::half_t,float>(A, B, C);
// ...
```
:::

Actually, we don't quite write the code in this way. We will exhibit the
complete program further down after discussing a few more important
points.

## Input validation {#input-validation .wp-block-heading}

Now that we have our input and information on them, let's check that
they are valid inputs. With access to tensor shapes and data types, some
of the more trivial checks (e.g., compatible dimensions for matrix
multiplications) should be self explanatory. So we will focus on topics
more specific to Torch and CUTLASS.

One restriction that CUTLASS puts on matrix multiplication is that it
must be contiguous, meaning that adjacent elements are also adjacent in
memory. As PyTorch tensors are row-major, a contiguous tensor is one
where elements in the same row and adjacent column are next to each
other in memory. We can check if a tensor is contiguous with
`.is_contiguous()` method.

::: wp-block-syntaxhighlighter-code
``` {.brush: .cpp; .gutter: .false; .title: .; .notranslate title=""}

bool a_contiguous = A.is_contiguous();
```
:::

If a tensor is not contiguous, they can be made contiguous using the
`.contiguous()` method.

::: wp-block-syntaxhighlighter-code
``` {.brush: .cpp; .gutter: .false; .title: .; .notranslate title=""}

torch::Tensor _A = A.contiguous();
```
:::

This method simply returns the original tensor if it is already
contiguous. However, it creates a new contiguous tensor if it is not.
This is not an issue for input matrices `A` and `B`, but for `C` matrix
this is an issue because `torch.mm` supports in-place operation. So for
the C matrix, we will copy the data back if necessary with `.copy_()`.

::: wp-block-syntaxhighlighter-code
``` {.brush: .cpp; .gutter: .false; .title: .; .notranslate title=""}

torch::Tensor _C = C.contiguous();

// ... GEMM operation ... //

if(!C.is_contiguous())
    C.copy_(_C);
return C
```
:::

Another restriction is that the data must be on the GPU device. We can
check this easily with:

::: wp-block-syntaxhighlighter-code
``` {.brush: .cpp; .gutter: .false; .title: .; .notranslate title=""}

bool is_cuda = A.device().is_cuda();
```
:::

Our library is only built for the GPU. If the data had to be allocated
on the host, we move it to the device in Python by using the `.to()`
method. While it is possible to automatically move the data to the
device using `.to()` in C++, this behavior is inconsistent with most
other PyTorch functions, so we will instead throw an error if device is
not GPU.

## Making C optional {#making-c-optional .wp-block-heading}

Like PyTorch's `mm`, our function will return the `C` tensor back to
PyTorch to be used there. We also need to update the function arguments
to mark `C` as being optional. The Torch C++ API provides a utility
`c10::optional<torch::Tensor>` in order to specify the Tensor argument
as optional. With this, we can check if the input was provided with the
`.has_value()` method. If this returns `true`, we can then get the value
with `.value()` method.

If `.has_value()` returns `false`, then we need to create a new tensor.
ATen has many options for creating a tensor, which are documented
[here](https://pytorch.org/cppdocs/notes/tensor_creation.html). For our
purposes, we just need an empty tensor. Combined, we get:

::: wp-block-syntaxhighlighter-code
``` {.brush: .cpp; .title: .; .notranslate title=""}

torch::Tensor cutlass_gemm(torch::Tensor A, torch::Tensor B, c10::optional<torch::Tensor> out) {

  // Handling the optional C matrix
  torch::Tensor C;
  if(out.has_value()) {  // Output tensor was provided. So we will use it.
    C = out.value();
  } else {               // Output tensor was not provided. Creating an empty tensor.
    const int M = A.sizes()[0];
    const int N = B.sizes()[1];

    // We will allocate the matrix on GPU and set the datatype to be the same as the input
    auto c_options = torch::TensorOptions().device(torch::kCUDA).dtype(A.dtype());
    C = torch::empty({M, N}, c_options);
  }

  // ... Rest of the GEMM workload ...//
}
```
:::

When creating a new matrix, we set the options to set the device to GPU
and the data type to be the same as the input Tensor. It is recommended
to use ATen library when creating new Tensors. Although it is possible
to create a new `torch::Tensor` from an existing pointer to data, this
will mean that ATen does not own the data. This can limit certain
operations like resizing once the Tensor is passed back to Python. So
while CUTLASS has special allocators like `HostTensor`, we will not be
using them.

## Putting it together {#putting-it-together .wp-block-heading}

Finally, putting everything together we have:

::: wp-block-syntaxhighlighter-code
``` {.brush: .cpp; .first-line: .1; .title: .; .notranslate title=""}

template<typename DataType, typename OutputType>
void cutlass_gemm_unpack(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  // Get data sizes
  const int M = A.sizes()[0];
  const int K = B.sizes()[0];
  const int N = B.sizes()[1];

  // Casting to the data type of the input tensor
  DataType const *ptrA = reinterpret_cast<DataType*>(A.data_ptr());
  DataType const *ptrB = reinterpret_cast<DataType*>(B.data_ptr());
  DataType *ptrC = reinterpret_cast<OutputType*>(C.data_ptr());
  cutlass_gemm_wrapper<DataType, OutputType>(M, N, K, ptrA, ptrB, ptrC);
}

// Intermediate function to get the output precision to use for the wrapper template.
template<typename DataType>
void cutlass_gemm_find_output_type(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  if(C.dtype() == torch::kFloat16)
    cutlass_gemm_unpack<DataType, cutlass::half_t>(A, B, C);
  else if(C.dtype() == torch::kFloat32)
    cutlass_gemm_unpack<DataType, float>(A, B, C);
  else
    throw std::invalid_argument("Unsupported precision type");
}

// This function is bound to "cutlass_gemm.mm". Takes torch::Tensors as inputs
torch::Tensor cutlass_gemm(torch::Tensor A,  // A matrix (m x k)
                           torch::Tensor B,  // B matrix (k x n)
                           c10::optional<torch::Tensor> out) {  // optional out matrix (m x n)
  // Handling the optional C matrix
  torch::Tensor C;
  if(out.has_value()) {  // Output tensor was provided. So we will use it.
    C = out.value();
  } else {               // Output tensor was not provided. Creating an empty tensor.
    const int M = A.sizes()[0];
    const int N = B.sizes()[1];
    // We will allocate the matrix on GPU and set the datatype to be the same as the input
    auto c_options = torch::TensorOptions().device(torch::kCUDA).dtype(A.dtype());
    C = torch::empty({M, N}, c_options);
  }

  // Check that all tensors are allocated on GPU device.
  if(!(A.device().is_cuda() && B.device().is_cuda() && C.device().is_cuda()))
    throw std::invalid_argument("cutlass_gemm only supports GPU device.
                                 Use .to(device=torch.device('cuda'))");

  // Ensuring that the matrices are contiguous.
  torch::Tensor _A = A.contiguous();
  torch::Tensor _B = B.contiguous();
  torch::Tensor _C = C.contiguous();

  // Select the CUTLASS precision type to use based on Torch input data type.
  if(A.dtype() == torch::kFloat16)
    cutlass_gemm_find_output_type<cutlass::half_t>(_A, _B, _C);
  else if(A.dtype() == torch::kFloat32)
    cutlass_gemm_find_output_type<float>(_A, _B, _C);
  else
    throw std::invalid_argument("Unsupported precision type");

  // If C was not contiguous, C != _C so copy the result back into C
  if(!C.is_contiguous())
    C.copy_(_C);

  // Return the Torch tensor back to PyTorch
  return C;
}
```
:::

In this code, we took an ad-hoc approach to the conditional logic needed
for dispatching to the appropriate templated function based on the data
types of `A` and `C`. Clearly, this wouldn't scale well to a large
number of template parameters. For an example of how you can use a
Python script to handle writing a wrapper for highly templated C++/CUDA
functions like those in CUTLASS, we suggest looking at the
[\_python_gemm](https://github.com/NVIDIA/cutlass/blob/main/python/cutlass/emit/pytorch.py#L704){data-type="link"
data-id="https://github.com/NVIDIA/cutlass/blob/main/python/cutlass/emit/pytorch.py#L704"}
method and the
[EmitGemmUniversalInstance3x](https://github.com/NVIDIA/cutlass/blob/main/python/cutlass/backend/gemm_operation.py#L1195){data-type="link"
data-id="https://github.com/NVIDIA/cutlass/blob/main/python/cutlass/backend/gemm_operation.py#L1195"}
class in the CUTLASS library.

## Binding and Compiling {#binding-and-compiling .wp-block-heading}

Now that we have our function, let's compile it and bind it to a Python
function. We will be using
[`PyBind11`](https://pybind11.readthedocs.io/en/stable/){target="_blank"
rel="noreferrer noopener"} in combination with
[`setuptools`](https://setuptools.pypa.io/en/latest/userguide/index.html){target="_blank"
rel="noreferrer noopener"} to do this step. Rather than aim for a
comprehensive discussion of these tools, we will only go over what is
directly relevant to us.

### PyBind11 {#pybind11 .wp-block-heading}

The binding for our function is:

::: wp-block-syntaxhighlighter-code
``` {.brush: .cpp; .title: .; .notranslate title=""}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mm",
        py::overload_cast<torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>>(
          &cutlass_gemm),
        py::arg("A"),
        py::arg("B"),
        py::arg("out") = py::none());
}
```
:::

We also specify the third argument as a keyword argument "out", in
keeping with `torch.mm`, and set it to default to Python `None`.

### setuptools {#setuptools .wp-block-heading}

Unfortunately, out of the box `setuptools` does not support `nvcc`, the
CUDA compiler. While there is a workaround, it can be rather
[complex](https://stackoverflow.com/questions/10034325/can-python-distutils-compile-cuda-code). Fortunately,
PyTorch comes with a utility called `CUDAExtension` that can compile
CUDA code.

::: wp-block-syntaxhighlighter-code
``` {.brush: .python; .title: .; .notranslate title=""}

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

### ... set up lists cutlass_include_dirs, nvcc_flags, and ld_flags ... ###
setup(
    name='cutlass_gemm',
    ext_modules=[
        CUDAExtension(name="cutlass_gemm",
                      sources=["cutlass_gemm.cu"],
                      include_dirs=cutlass_include_dirs,
                      extra_compile_args={'nvcc': nvcc_flags},
                      libraries=ld_flags)
    ],
    cmdclass={'build_ext': BuildExtension})
```
:::

The syntax for the arguments is identical to the base Extension class.
However, it will automatically add all the necessary flags for the Torch
library. So the only thing for us is to add the path for CUTLASS. And
because CUTLASS is a header-only library, we just need to set the
`include_dir`. Once you run the `setup.py`, we now have our new module
`cutlass_gemm` accessible from our PyTorch code. 

## Calling our new `mm` with PyTorch {#calling-our-new-mm-with-pytorch .wp-block-heading}

Here is a simple PyTorch script that does CUTLASS GEMM using our new
function.

::: wp-block-syntaxhighlighter-code
``` {.brush: .python; .title: .; .notranslate title=""}

import math
import cutlass_gemm

M = K = N = 4096
cuda = torch.device('cuda')
A = torch.normal(0,1,size=(M, K)).to(device=cuda).to(dtype=torch.float16)/math.sqrt(K)
B = torch.normal(0,1,size=(K, N)).to(device=cuda).to(dtype=torch.float16)/math.sqrt(K)

C1 = cutlass_gemm.mm(A,B)
print("cutlass_gemm.mm result:")
print(C1)
print()

C2 = torch.mm(A,B)
print("torch.mm result:")
print(C2)
print()
print("max deviation: {:.10f}".format(torch.max(torch.abs(C2-C1))))
```
:::

We specify `.to(device=cuda)` to make `A` and `B` be accessible to the
GPU, and we use FP16 precision for the two matrices. Furthermore, we
have a validation step against `torch.mm` that shows the maximum
deviation from the Torch version.

::: wp-block-syntaxhighlighter-code
``` {.brush: .plain; .gutter: .false; .title: .; .notranslate title=""}

cutlass_gemm.mm result:
tensor([[-0.0045, -0.0139,  0.0109,  ...,  0.0192, -0.0117,  0.0083],
        ...,
        [ 0.0110,  0.0005, -0.0079,  ...,  0.0106, -0.0012, -0.0083]],
       device='cuda:0', dtype=torch.float16)

torch.mm result:
tensor([[-0.0045, -0.0139,  0.0109,  ...,  0.0192, -0.0117,  0.0083],
        ...,
        [ 0.0110,  0.0005, -0.0079,  ...,  0.0106, -0.0012, -0.0083]],
       device='cuda:0', dtype=torch.float16)

max deviation: 0.0000610352
```
:::

Here we can see that the resultant matrix is in fact using the FP16
precision format, and we are getting the same result (within epsilon) to
`torch.mm`. So now we can use this optimized GEMM in place of
`torch.mm`.

## Code download {#code-download .wp-block-heading}

The source code for the full example can be found on the [Colfax
Research github](https://github.com/ColfaxResearch/cfx-article-src).

## Addendum A: AMP support {#addendum-a-amp-support .wp-block-heading}

PyTorch has a feature called [Automatic Mixed Precision
(AMP)](https://pytorch.org/docs/stable/amp.html){target="_blank"
rel="noreferrer noopener"} that can be used to simplify mixed precision
workload. It revolves around the `autocast` context, inside which
operations automatically use lower precision when appropriate. This can
lead to significant performance improvements.

Our example does not support this feature, however you can find more
information on AMP support in C++ packages
[here](https://pytorch.org/tutorials/advanced/dispatcher.html#autocast).

## Addendum B: CUTLASS 3.X and Hopper Architecture {#addendum-b-cutlass-3.x-and-hopper-architecture .wp-block-heading}

As mentioned earlier, the above example uses the 2.X syntax for CUTLASS.
On our repository, we have also provided a CUTLASS 3.X example based on
the `hopper_warp_specialized_gemm` [example
48](https://github.com/NVIDIA/cutlass/blob/main/examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu).
However, within the scope of this article there isn't a difference
between what is needed for 2.X and 3.X CUTLASS. Our 3.X example still
wraps all the CUTLASS code in the wrapper function. For more on CUTLASS
3.X and how to optimize for a specific architecture, refer to the
CUTLASS documentation.

## Addendum C: Build backends {#addendum-c-build-backends .wp-block-heading}

In this article, our focus was on writing an extension that can be used
with PyTorch. To this end, we used `setuptools` as the build backend in
conjunction with PyTorch's `CUDAExtension` utility class. However this
adds PyTorch as a dependency for our extension, which may not be ideal
if the extension was not being developed for PyTorch. It is possible to
use `setuptools` without having to rely on `CUDAExtension`.  For an
example, see the python installation for
[CUTLASS](https://github.com/NVIDIA/cutlass/tree/main){target="_blank"
rel="noreferrer noopener"}.

In addition, there are other build backends that are compatible with
`nvcc` that can be used to create C/C++ based Python extensions. For
example,
[scikit-build-core](https://github.com/scikit-build/scikit-build-core){target="_blank"
rel="noreferrer noopener"} is a cmake based backend that can be used in
place of `setuptools`. There is a guide for using `nvcc` in `cmake` on
the [Nvidia developer
forums](https://developer.nvidia.com/blog/building-cuda-applications-cmake/).

As a final note, build backends are typically specified in the
`pyproject.toml` files that are then used by python packaging software.
Details on `pyproject.toml` and its usage can be found
[here](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/){target="_blank"
rel="noreferrer noopener"}.

::::: {.sharedaddy .sd-sharing-enabled}
:::: {.robots-nocontent .sd-block .sd-social .sd-social-icon-text .sd-sharing}
### Share this: {#share-this .sd-title}

::: sd-content
- [[Share on LinkedIn (Opens in new window)]{#sharing-linkedin-9708
  hidden=""}
  LinkedIn](https://research.colfax-intl.com/tutorial-python-binding-for-cuda-libraries-in-pytorch/?share=linkedin){.share-linkedin
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-linkedin-9708" target="_blank"
  aria-labelledby="sharing-linkedin-9708"}
- [[Share on X (Opens in new window)]{#sharing-twitter-9708 hidden=""}
  X](https://research.colfax-intl.com/tutorial-python-binding-for-cuda-libraries-in-pytorch/?share=twitter){.share-twitter
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-twitter-9708" target="_blank"
  aria-labelledby="sharing-twitter-9708"}
- [[Share on Facebook (Opens in new window)]{#sharing-facebook-9708
  hidden=""}
  Facebook](https://research.colfax-intl.com/tutorial-python-binding-for-cuda-libraries-in-pytorch/?share=facebook){.share-facebook
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-facebook-9708" target="_blank"
  aria-labelledby="sharing-facebook-9708"}
- [[Share on Reddit (Opens in new window)]{#sharing-reddit-9708
  hidden=""}
  Reddit](https://research.colfax-intl.com/tutorial-python-binding-for-cuda-libraries-in-pytorch/?share=reddit){.share-reddit
  .sd-button .share-icon rel="nofollow noopener noreferrer"
  shared="sharing-reddit-9708" target="_blank"
  aria-labelledby="sharing-reddit-9708"}
-
:::
::::
:::::
:::::::::::::::::::::::::

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
March 13, 2024
:::

in

::: {.taxonomy-category .wp-block-post-terms}
[Article](https://research.colfax-intl.com/category/article/){rel="tag"}[,
]{.wp-block-post-terms__separator}[Blog](https://research.colfax-intl.com/category/blog/){rel="tag"}[,
]{.wp-block-post-terms__separator}[Tutorials](https://research.colfax-intl.com/category/papers/tutorials/){rel="tag"}
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

#### 2 responses to "Tutorial: Python bindings for CUDA libraries in PyTorch" {#comments .wp-block-comments-title .has-medium-font-size}

1.  :::::::::::: {#comment-256}
    ::::::::::: {.wp-block-columns .is-layout-flex .wp-container-core-columns-is-layout-a5cf46d6 .wp-block-columns-is-layout-flex style="margin-bottom:var(--wp--preset--spacing--40)"}
    :::: {.wp-block-column .is-layout-flow .wp-block-column-is-layout-flow style="flex-basis:40px"}
    ::: wp-block-avatar
    ![Johnny
    Avatar](https://secure.gravatar.com/avatar/765d03961960be26c8d5b20383d95e7048d87680c9a6dc1d5512765f0a9cccd8?s=40&r=g){.avatar
    .avatar-40 .photo .wp-block-avatar__image
    srcset="https://secure.gravatar.com/avatar/765d03961960be26c8d5b20383d95e7048d87680c9a6dc1d5512765f0a9cccd8?s=80&r=g 2x"
    height="40" width="40" style="border-radius:20px;" decoding="async"}
    :::
    ::::

    :::::::: {.wp-block-column .is-layout-flow .wp-block-column-is-layout-flow}
    ::: wp-block-comment-author-name
    [Johnny](http://johnnynunez.github.io){rel="external nofollow ugc"
    target="_self"}
    :::

    :::: {.wp-block-group .is-layout-flex .wp-block-group-is-layout-flex style="margin-top:0px;margin-bottom:0px"}
    ::: wp-block-comment-date
    [November 20,
    2024](https://research.colfax-intl.com/tutorial-python-binding-for-cuda-libraries-in-pytorch/#comment-256)
    :::
    ::::

    ::: wp-block-comment-content
    and how to use python bindings with nvcc runtime? I mean, not use
    pre-compiled kernels
    :::

    ::: wp-block-comment-reply-link
    [Reply](https://research.colfax-intl.com/tutorial-python-binding-for-cuda-libraries-in-pytorch/?replytocom=256#respond){.comment-reply-link
    rel="nofollow" commentid="256" postid="9708"
    belowelement="comment-256" respondelement="respond"
    replyto="Reply to Johnny" aria-label="Reply to Johnny"}
    :::
    ::::::::
    :::::::::::

    1.  :::::::::::: {#comment-257}
        ::::::::::: {.wp-block-columns .is-layout-flex .wp-container-core-columns-is-layout-a5cf46d6 .wp-block-columns-is-layout-flex style="margin-bottom:var(--wp--preset--spacing--40)"}
        :::: {.wp-block-column .is-layout-flow .wp-block-column-is-layout-flow style="flex-basis:40px"}
        ::: wp-block-avatar
        ![Ryo
        Avatar](https://secure.gravatar.com/avatar/e315c98859dc1ec24185bb5a3b1f1bb443f17be646c2ed9bb32224af5a0e9eb1?s=40&r=g){.avatar
        .avatar-40 .photo .wp-block-avatar__image
        srcset="https://secure.gravatar.com/avatar/e315c98859dc1ec24185bb5a3b1f1bb443f17be646c2ed9bb32224af5a0e9eb1?s=80&r=g 2x"
        height="40" width="40" style="border-radius:20px;"
        decoding="async"}
        :::
        ::::

        :::::::: {.wp-block-column .is-layout-flow .wp-block-column-is-layout-flow}
        ::: wp-block-comment-author-name
        Ryo
        :::

        :::: {.wp-block-group .is-layout-flex .wp-block-group-is-layout-flex style="margin-top:0px;margin-bottom:0px"}
        ::: wp-block-comment-date
        [November 21,
        2024](https://research.colfax-intl.com/tutorial-python-binding-for-cuda-libraries-in-pytorch/#comment-257)
        :::
        ::::

        ::: wp-block-comment-content
        Do you mean something like Just-In-Time (JIT) compilation? If
        so, good place to start is at PyTorch JIT. There is also an
        Nvidia library called nvJitLink ([nvidia post on
        nvJitLink](https://developer.nvidia.com/blog/cuda-12-0-compiler-support-for-runtime-lto-using-nvjitlink-library/){rel="nofollow ugc"})
        that might be worth looking at.\
        We may do a blog post in the future focusing more on JIT.
        :::

        ::: wp-block-comment-reply-link
        [Reply](https://research.colfax-intl.com/tutorial-python-binding-for-cuda-libraries-in-pytorch/?replytocom=257#respond){.comment-reply-link
        rel="nofollow" commentid="257" postid="9708"
        belowelement="comment-257" respondelement="respond"
        replyto="Reply to Ryo" aria-label="Reply to Ryo"}
        :::
        ::::::::
        :::::::::::
        ::::::::::::
    ::::::::::::

::: {#respond .comment-respond .wp-block-post-comments-form}
### Leave a Reply [[Cancel reply](/tutorial-python-binding-for-cuda-libraries-in-pytorch/#respond){#cancel-comment-reply-link rel="nofollow" style="display:none;"}]{.small} {#reply-title .comment-reply-title}

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
::::::::::::::::::::::::::::::::::::::::::::
