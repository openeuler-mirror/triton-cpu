# openEuler Triton-CPU

A Triton-CPU that adapts to the LLVM of the openEuler community.

# Quick Installation
If you are using openEuler-24.03-LTS-SP2 or later, you can install the latest stable release of Triton-CPU from yum:
```
yum install triton-cpu
```

# Install from source
1. Prepare the Environment
```
pip install setuptools>=40.8.0
pip install wheel
pip install cmake>=3.18,<4.0
pip install ninja>=1.11.1
pip install pybind11>=2.13.1
pip install lit
```
2. Build openEuler LLVM19
```
git clone https://gitee.com/openeuler/llvm-project.git -b dev_19.1.7
cd llvm-project
mkdir build
cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON ../llvm -DLLVM_ENABLE_PROJECTS="clang;mlir;llvm;lld" -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" -DMLIR_ENABLE_BINDINGS_PYTHON=ON -DPython3_EXECUTABLE=$(which python) -DLLVM_ENABLE_RTTI=ON
ninja
```
3. Build openEuler Triton-CPU
```
git clone https://gitee.com/openeuler/triton-cpu.git
cd triton-cpu
git submodule init
git submodule update
export LLVM_BUILD_DIR=$YOUR_WORKDIR/llvm-project/build
export LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include
export LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib
export LLVM_SYSPATH=$LLVM_BUILD_DIR
export PATH=$LLVM_BUILD_DIR/bin:$PATH
export TRITON_BUILD_WITH_CLANG_LLD=true
export TRITON_PLUGIN_DIRS=$(pwd)/triton-shared
pip install -e python
```

# How to use it?
```
cd triton-cpu
export TRITON_DISABLE_LINE_INFO=1
TRITON_CPU_BACKEND=1 python3 python/tutorials/01-vector-add.py
```

