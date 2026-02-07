## Changes

This document serves the purpose of detailing the changes made to the project in further depth.

## How it works

To use the custom backend (referred to as the *shared backend*, as it comes from the `triton-shared` project), you need to set an environment variable:

```bash
export TRITON_USE_SHARED_BACKEND=1
```

This enables the shared backend instead of the default CPU backend. After setting the variable, you can simply run your Python code as usual.

How `triton-shared` is integrated into the project

`triton-shared` is designed to be installed as a Triton plugin. See [the official repo](https://github.com/microsoft/triton-shared?tab=readme-ov-file#usage) for details.

To integrate it, I added the `TRITON_PLUGIN_DIRS` environment variable in the `README.md`. This ensures that `triton-shared` is recognized as a backend when installing the `triton-cpu` Python package.

## Background on how the shared backend works

The shared backend works by taking the TTIR and converting it to Linalg using the `triton-shared-opt` binary, which must be specified via the `TRITON_SHARED_OPT_PATH` environment variable. It is then lowered to LLVM by calling `mlir-opt`, whose path should be set using the `LLVM_BINARY_DIR` environment variable.

## `triton-shared` version

Since the OpenEuler `triton-cpu` package is based on version 3.0.0, I needed to use a compatible version of `triton-shared`. In this case, the commit `89286b4` is the latest one that supports Triton 3.0.0.

## How the division by zero bug was solved

By comparing my version of `triton-shared` with the upstream version, I noticed that the MLIR code generated was nearly identical. This led me to rule out incorrect code generation as the cause.

However, upon inspecting how the launcher functions were generated in `driver.py`, I noticed a difference: the upstream version passed more parameters, including constants as arguments.

To fix the bug, I modified the shared backend's `driver.py` to also pass constants as arguments.

## Adding Support for ARM SVE

A new method called `_sve_transform` was added to the `compiler.py` file of the shared backend to add optimizations through the transform dialect that enable SVE. This implementation is based on the [LLVM example](https://github.com/llvm/llvm-project/blob/llvmorg-19.1.7/mlir/test/Integration/Dialect/Linalg/CPU/ArmSVE/matmul.mlir). Additionally, several passes were added to the pipeline executed by `mlir-opt`.

## Bug with SVE/SME on LLVM 19

LLVM 19 contains a bug where using SVE/SME in the transform dialect without first loading the dialect via an SVE/SME-related pass in `mlir-opt` causes a crash. One straightforward (though hacky) workaround is to include an SVE/SME-related pass in the `mlir-opt` pipeline, even if it's not strictly necessary.

In the process of investigating this issue, I gained a deep understanding of the underlying bug and was able to fix it for OpenEuler (branch `dev_19.1.7`). I identified the fix in LLVM 20 just for SVE [here](https://github.com/llvm/llvm-project/commit/b9d3a644c2716e651b388f9fff660b12fdba577c#diff-291d3da22331ded009a6178f86f517984d6ac0aeb27fe6907bbd793d41670340R110).

The root of the issue is that SVE was not registered as an extension in the transform dialect. Until the referenced commit, there were no native SVE transform ops in the transform dialect, so there was no reason to register the extension. That commit fixes the issue by introducing the first native SVE transform op, and as a result, it also registers SVE as a transform dialect extension. In our case, however, the crash still occurs even without native SVE/SME ops, because SVE/SME can be invoked indirectly via `apply_registered_pass`.

Therefore, the fix for the OpenEuler LLVM 19 branch consists of registering the SVE/SME extension in the transform dialect **without** adding any native transform ops. The transform op introduced in LLVM 20 is written using LLVM 20 features and is not compatible with LLVM 19, so it is intentionally omitted. The fix is in [this pull request](https://gitee.com/openeuler/llvm-project/pulls/236)


## Adding Support for ARM SME

Similar to SVE, a new method called `_sme_transform` was added to the `compiler.py` file of the shared backend to add optimizations through the transform dialect that enable SME. This is based on the [LLVM example](https://gitee.com/openeuler/llvm-project/pulls/234) provided by this pull request. Additional passes were also added to the pipeline run by `mlir-opt`.

## Adding multithread support to the shared backend

The shared backend did not support multithreading, but implementing it is fairly easy. In `driver.py`, for the C++ code, we just need to pass the number of threads as a parameter to the `_launch` function and modify that function to use OpenMP when launching the kernel calls. This means the kernel is launched in several threads at the same time (depending on the grid) but the kernel itself is not running with multithreading. 

## Running on SVE

SVE has a bug even on upstream LLVM where if the matrix is too big it will crash due to a invalid memory load/store, see the issue [here](https://github.com/llvm/llvm-project/issues/151679), The crash can be at least be alleviated with `vector-to-scf='full-unroll=true'`. To not rely on full unroll which can be unsuitable in some scenarios some hosting needs to take place during or after the `vector-to-scf` pass to remove the `allocas` from inside the loop

## Running on SME

Compiling to object code required a patch in llvm provided by Chenzheng (already merged).

## Passing test on SVE

Around 60\% of test work out of the box. The failing test are manily caused by two errors:

* Unsupported operations in triton shared specially some kinds of Load and Stores, see [this issue](https://github.com/microsoft/triton-shared/issues/311)
* CPU not supporting fp8 data type

these two account for almost all the failing test. 

## Performance optimizations

Triton shared creates unnecessary copies of memref, see [this issue](https://github.com/microsoft/triton-shared/issues/308) which introduces overhead, that can be fixed as seen in the changes to the file `StructuredToMemref.cpp` Where I commented the copy and passed the reinterpret_cast (variable named ptr in the C++ code) to `bufferization.to_tensor` instead.

## Adding support for fp8e5m2

With the help of upstream, see [this issue](https://github.com/llvm/llvm-project/issues/152287) I found the way on how to implement `arith.ext` and `arith.trunc` so we can handle fp8e5m2 by converting them to fp32 and back to fp8e5m2, I used the algorith that can be found [in this blog post](https://www.xyzzhangfan.tech/blog/2025/Convert_fp32_and_fp8_e5m2/)

## Adding support for fp8e4m3

Sames as with fp8e5m2 again using [this other blog post](https://www.xyzzhangfan.tech/blog/2025/Convert_fp32_and_fp8_e4m3/) although, this time the test are giving me an accuracy error, the algorithm is right but we can expect much precision from this datatype so I lowered the expected error on the tests.

## Adding support for more complex reduction operations

Triton-shared does not support reduce operations that have more than one op in the body of the reduction such as `argmin` or `argmax` reduction types, to solve this I changed the code of `triton-shared/include/triton-shared/Conversion/TritonArithToLinalg/ConversionPatterns.hpp` specifically the method called `convertToLinalgReduce`. With this implementation it now supports multiple ops in the body, and an arbitrary number of inputs and outputs.

## Addding support for the scan operation

The scan operation performs a cumulative reduction, simply that means a reduction but storing all of the intermediate results, for instance a scan operation with just and add operation in it's body is just a [cumsum](https://numpy.org/devdocs/reference/generated/numpy.cumsum.html). 

As there is no such operation in standard MLIR (though there is in other projects like IREE) it's not possible to just translate it to a high level equivalent operation. Because of that, the best solution I found was just to lower it to loops directly.

At a conceptual level, this implementation works as follows:

* We iterate over all elements of the input tensor(s), treating the scan axis specially.
* For the first element along the scan axis, the output is initialized directly from the input.
* For the remaining elements, we load the previous accumulator value, apply the scan operation (which gets clone from the original `tt.scan`), and store the updated result.


## Fixing tt.storeOp cannot be rewritten

This triton-shared bug is partially fixed in the lastest version as discussed in [this issue](https://github.com/microsoft/triton-shared/issues/311). The code gets lowered to `tptr` and `ptr` dialect however the is not a lowering to LLVM just yet.

To get as close as upstream triton-shared as I could I started looking at the commits and making changes. At the start I tought I would only need commits related to `tptr` but I later tought that including the other could be a good idea too (that's why in my commit history it does not match with the original order of triton-shared). In the end I just ended up skipping commits related to triton updates or related to other pieces of code I had already changed myself (like reduction) and made the necessary changes for it to work with the MLIR 19 API.

However, I found a bug where all of my code was getting deleted, after debugging I found the solution in [this upstream commit](https://github.com/llvm/llvm-project/commit/df0d249b6511289f1e8c1389f4fd33d7b4c083fa) so I made [this backport](https://gitee.com/openeuler/llvm-project/pulls/255) to the open euler llvm.


Now I used the code in [this PR](https://github.com/microsoft/triton-shared/pull/325) that lower `tptr` to llvm as the `ptr` dialect in MLIR is not developed enough even in upstream. However, adding a `lower-to-llvm` pass to triton-shared implies that we need to start using `triton-shared-opt` instead of `mlir-opt` in the middle of the pipeline as now we are mixing standard MLIR + `tptr` in this situation.

To do that I introduced the needed passes to the Python API in `triton_shared.cc`, I also needed to introduce changes to some of the `CMakeLists.txt`, I used the official LLVM `mlir-opt.cpp` as a reference. 

After that almost all of the code lowered but there are some `memref.dealloc` still in the final code. This happends because they have `!ptr.ptr` type and there is no know way to lower that to llvm.free as referenced in [this issue](https://github.com/llvm/llvm-project/issues/156006).

After [backporting some more LLVM code](https://gitee.com/openeuler/llvm-project/pulls/266) to be able to handle `memref.dealloc` I started getting `invalid free` error. Debugging it I found that the problem is not related to LLVM but rather to `tptr-to-llvm` that has `MemRefAllocConverter`, `MemRefLoadConverter` and `MemRefStoreConverter` which target memrefs with `ptr` type, those rewrites were generating wrong code as `memref.allocOp` was being rewritten to `llvm.alloca` which is wrong since they use heap and stack memory respectively, creating the `invalidad free` error when trying to free stack memory.

I removed those rewrites and made sure to use the `TypeConverter` I backported from LLVM making this error finally go away and passing another group of test. For a closer look there is [This comment I made](https://github.com/microsoft/triton-shared/pull/325#issuecomment-3245391988) on the triton-shared repo.

## Adding support for tt.atomic_rmw

To add support to this operation I did a conversion pass in the file `ReconcilePtrCastsPass.cpp` that made the conversion from `tt.atomic_rmw` to `memref.atomic_rmw` I also needed to [backport some commits](https://gitee.com/openeuler/llvm-project/pulls/269?source=dashboard) to fix a bug found in the `remove-dead-values` pass that was crashing the pipeline.

for the tests called `test_atomic_rmw` the conversion was quite straight forward as those test only read-modify-write a single value. But, for the tests called `test_tensor_atomic_rmw` the conversion was harder as triton enables read-modify-write for whole tensors and that is not possible in MLIR so I took the same approach as before but adding a loop and some other conversion around the `memref.atomic_rmw` for it to iterate trough each element.


## Adding support for tt.dot with 3d tensors

To add support to this operation I just checked the rank of the tensors in the `MatmulConverter` in `ConversionPatterns.hpp` and if it's 3d the it replaces the op for a `linalg.batch_matmul` instead of a normal matmul.

## Why use the TPtr dialect

Related to the part of [fixing tt.storeOp cannot be rewritten](#fixing-ttstoreOp-cannot-be-rewritten) we are using the TPtr dialect and some additions from a PR that was actually **rejected** which may seem a little off. However we need this for two reason.

* There is still to this day (25-09-2025) no way to lower TPtr outside of that PR
* The plan of the triton-shared devs is to lower TPtr to mlir ptr dialect but that is introduced on llvm22 and backporting that to llvm19 is not feasible at all (believe me, I've tried) 

Having to use TPtr like this may be a bit of a burden as we need to use the `tptr-to-llvm` pass in the middle of the pipeline but is the only practical way to be able to lower some pieces of code.