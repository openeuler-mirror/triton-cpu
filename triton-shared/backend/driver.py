import hashlib
import tempfile
import sysconfig
import time
import triton

# Maps MD5 launcher key -> loaded mod.launch function.
# Avoids re-instantiating FileCacheManager (and its ~100µs of Path.home() +
# lstat syscalls) on every kernel dispatch once the .so is already loaded.
_LAUNCHER_MODULE_CACHE: dict = {}

import os, subprocess, tempfile, platform
import importlib
import importlib.util
import sys

from pathlib import Path

from triton.runtime.cache import get_cache_manager
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget
import shutil

def _get_llvm_bin_path(bin_name: str) -> str:
    path = os.getenv("LLVM_BINARY_DIR", "")
    if path == "":
        raise Exception("LLVM_BINARY_DIR is not set.")
    return os.path.join(path, bin_name)

# for locating libsleef
try:
    _triton_C_dir = importlib.resources.files(triton).joinpath("_C")
except AttributeError:
    # resources.files() doesn't exist for Python < 3.9
    _triton_C_dir = importlib.resources.path(triton, "_C").__enter__()

# -------------------- Launcher ----------------------------
def _ty_to_cpp(ty):
    if ty[0] == '*':
        return "void*"
    if ty == "constexpr":
        return "PyObject*"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]

def _extracted_type(ty):
    if ty[0] == '*':
        return "PyObject*"
    if ty == "constexpr":
        return "PyObject*"
    return _ty_to_cpp(ty)

def _format_of(ty):
    return {
      "PyObject*": "O",
      "constexpr": "O",
      "float": "f",
      "double": "d",
      "long": "l",
      "int8_t": "b",
      "int16_t": "h",
      "int32_t": "i",
      "int64_t": "l",
      "uint8_t": "B",
      "uint16_t": "H",
      "uint32_t": "I",
      "uint64_t": "K",
    }[ty]

def _generate_launcher(constants, signature, kernel_name):

    arg_decls = ', '.join(f"{_ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())
    args_format = ''.join([_format_of(_extracted_type(ty)) for ty in signature.values()])
    format = "iiiOOOO" + args_format
    args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''

    kernel_arg_decls = ', '.join(_ty_to_cpp(ty) if ty[0] != "*" else f"int64_t, void*" for i, ty in signature.items() if ty != "constexpr")
    kernel_arg_decls += ', ' if kernel_arg_decls else ''

    kernel_parameters = ', '.join(f"static_cast<{_ty_to_cpp(ty)}>(arg{i})" if ty[0] != "*" else f"0, &ptr_arg{i}" for i, ty in signature.items() if ty != "constexpr")
    kernel_parameters += ', ' if kernel_parameters else ''

    return f"""
#ifdef _OPENMP
#include <omp.h>
#endif
#include <assert.h>
#include <stdbool.h>
#include <Python.h>
#include "ExecutionEngine/CRunnerUtils.h"
#include "ExecutionEngine/CRunnerUtils.cpp"
#include <iostream>

extern "C" {{
  // Pointer type (=Memref) becomes int64_t + MemRef struct
  // FIXME: understand what this int64_t is used for.
  void {kernel_name}({kernel_arg_decls}
                       int, int, int, int, int, int);
}}

static void _launch(int gridX, int gridY, int gridZ, int num_threads {', ' + arg_decls if arg_decls else ''}) {{
int64_t N = (int64_t)gridX * gridY * gridZ;

#ifdef _OPENMP
int max_threads = (num_threads > 0) ? num_threads : omp_get_max_threads();

#pragma omp parallel for schedule(static) num_threads(max_threads)
#endif
for (int64_t i = 0; i < N; ++i) {{
  int x = i % gridX;
  int y = (i / gridX) % gridY;
  int z = i / (gridX * gridY);


  // Optional: declare memrefs like before
  {' '.join(f'StridedMemRefType<char, 0> ptr_arg{i} = {{static_cast<char *>(arg{i}), static_cast<char *>(arg{i}), 0}};' for i, ty in signature.items() if i not in constants and ty[0] == "*")}

  {kernel_name}({kernel_parameters}
                gridX, gridY, gridZ, x, y, z);
  }}
}}

typedef struct _DevicePtrInfo {{
  void *dev_ptr;
  bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(obj));
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(ret));
    if(!ptr_info.dev_ptr)
      return ptr_info;
    Py_DECREF(ret);  // Thanks ChatGPT!
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return ptr_info;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook {args_list})) {{
    return NULL;
  }}


  // [CPULauncher-specific]: We don't need the metadata below but just put them
  // here anyway to be consistent with others.
  // This will make updating the driver easier in the future.

  //  int num_warps, num_ctas, shared_memory, clusterDimX, clusterDimY, clusterDimZ;
  //  if (!PyArg_ParseTuple(kernel_metadata, \"iiiiii\", &num_warps, &num_ctas, &shared_memory, &clusterDimX, &clusterDimY, &clusterDimZ)) {{
  //    PyErr_SetString(PyExc_TypeError, "kernel_metadata must be a tuple");
  //    return NULL;
  //  }}

  
  // num_threads is at index 7 in the packed_metadata tuple (see pack_metadata in compiler.py)
  int num_threads = 0;
  if (kernel_metadata && kernel_metadata != Py_None) {{
    PyObject *num_threads_item = PySequence_GetItem(kernel_metadata, 7);
    if (num_threads_item) {{
      if (PyLong_Check(num_threads_item))
        num_threads = PyLong_AsLong(num_threads_item);
      Py_DECREF(num_threads_item);
    }} else {{
      PyErr_Clear(); // Avoid leaving a lingering Python exception
    }}
  }}
  
  
  
  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}


  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  _launch(gridX, gridY, gridZ, num_threads {', ' + ', '.join([f"ptr_info{i}.dev_ptr" if ty[0]=="*" else f"_arg{i}"for i, ty in signature.items()]) if signature else ""});

  if (PyErr_Occurred()) {{
    return NULL;
  }}
  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  // return None
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__triton_shared_ref_cpu_kernel_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_shared_ref_cpu_kernel_launcher(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""


def compile_module(launcher_src, kernel_placeholder_name):
    py_version = sys.version_info
    if platform.system() == "Windows":
        py_include_dir = os.path.join(sys.base_prefix, 'include')
        py_lib_dir = os.path.join(sys.base_prefix, 'libs')
        py_lib = '{name}{major}{minor}.lib'.format(name="python", major=py_version.major, minor=py_version.minor)
    else:
        py_include_dir = os.path.join(sys.base_prefix, 'include', f'python{sys.version_info.major}.{sys.version_info.minor}')
        py_lib_dir = os.path.join(sys.base_prefix, 'lib')
        py_lib = '{name}{major}.{minor}'.format(name="python", major=py_version.major, minor=py_version.minor)
    cpu_backend_path = Path(__file__).resolve().parent
    include_dir = os.path.join(cpu_backend_path, "include")

    # Mutable cell: _cached_key[0] holds the MD5 hex string after the first call.
    # kernel_obj (cu_function) and kernel_name are fixed for a given compiled kernel,
    # so src and therefore the key never change across calls to this closure.
    _cached_key = [None]

    def launch(
        gridX, gridY, gridZ, stream, cu_function,
        kernel_metadata, launch_metadata,
        launch_enter_hook, launch_exit_hook, *args):
        # Unlike CUDA/HIP, we cannot easily pass function pointer across different pybind libraries.
        # Let's compile one kernel every time.
        # The cu_function parameter actually contains our kernel obj.
        # See CPUUtils.load_binary method.
        kernel_obj = cu_function
        kernel_name = kernel_metadata[6] # see pack_metadata in compiler.py

        # Fast path: key and .so are both cached — skip MD5, src build, and all filesystem work.
        if _cached_key[0] is not None and _cached_key[0] in _LAUNCHER_MODULE_CACHE:
            return _LAUNCHER_MODULE_CACHE[_cached_key[0]](gridX, gridY, gridZ,
                                                          kernel_metadata, launch_metadata,
                                                          launch_enter_hook, launch_exit_hook,
                                                          *args)

        src = launcher_src.replace(kernel_placeholder_name, kernel_name)
        key = hashlib.md5(src.encode("utf-8") + kernel_obj).hexdigest()
        _cached_key[0] = key

        # Second fast path: module already loaded (e.g. from a prior process run that
        # populated _LAUNCHER_MODULE_CACHE) but _cached_key was not yet set.
        if key in _LAUNCHER_MODULE_CACHE:
            return _LAUNCHER_MODULE_CACHE[key](gridX, gridY, gridZ,
                                               kernel_metadata, launch_metadata,
                                               launch_enter_hook, launch_exit_hook,
                                               *args)

        cache = get_cache_manager(key)
        name = "__triton_shared_ref_cpu_kernel_launcher"

        if platform.system() == "Windows":
          filename = f"{name}.pyd"
        else:
          filename = f"{name}.so"
        cache_path = cache.get_file(filename)

        if cache_path is None:
          with tempfile.TemporaryDirectory() as tmpdir:
              if platform.system() == "Windows":
                  obj_path = os.path.join(tmpdir, "kernel.obj")
                  launcher_src_path = os.path.join(tmpdir, "main.cxx")
                  so_path = os.path.join(tmpdir, "kernel.pyd")
                  Path(obj_path).write_bytes(kernel_obj)
                  Path(launcher_src_path).write_text(src)
                  # Compile it together.
                  subprocess.check_call([
                    "cl", "/LD", "/std:c++17", launcher_src_path, obj_path,
                    f"-I{py_include_dir}", f"-I{include_dir}", "/link", f"/LIBPATH:{py_lib_dir}",
                    "/link", f"{py_lib}", f"/OUT:{so_path}"
                  ])
              else:
                  obj_path = os.path.join(tmpdir, "kernel.o")
                  launcher_src_path = os.path.join(tmpdir, "main.cxx")
                  so_path = os.path.join(tmpdir, "kernel.so")
                  Path(obj_path).write_bytes(kernel_obj)
                  Path(launcher_src_path).write_text(src)
                  # Compile it together.
                  ## dump main.cxx to dir in TRITON_SHARED_DUMP_PATH
                  if "TRITON_SHARED_DUMP_PATH" in os.environ:
                    dump_path = os.environ["TRITON_SHARED_DUMP_PATH"]
                    shutil.copy(launcher_src_path, dump_path)
                  subprocess.check_call([
                    _get_llvm_bin_path("clang++"), "-O3", "-g", "-std=c++17", launcher_src_path, obj_path,
                    f"-I{py_include_dir}", f"-I{include_dir}", f"-L{py_lib_dir}",
                    "-shared", f"-l{py_lib}", "-fPIC", "-fopenmp",
                    f"-L{_triton_C_dir}", "-lsleef", f"-Wl,-rpath,{_triton_C_dir}",
                    "-o", so_path, "-rtlib=compiler-rt"
                  ])

              with open(so_path, "rb") as f:
                cache_path = cache.put(f.read(), filename, binary=True)

        # Load and launch the compiled kernel.
        spec = importlib.util.spec_from_file_location(name, cache_path)
        if spec is None:
            raise RuntimeError(f"Cannot find {name} module in {cache_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _LAUNCHER_MODULE_CACHE[key] = mod.launch
        return mod.launch(gridX, gridY, gridZ,
                          kernel_metadata, launch_metadata,
                          launch_enter_hook, launch_exit_hook,
                          *args)

    return launch


class CPULauncher(object):

    def __init__(self, src, metadata):
        kernel_placeholder_name = "KERNEL_NAME_PLACEHOLDER"
        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        constants_aux = {i: v for i, v in constants.items()}
        for i, v in constants_aux.items():
          signature[i] = "constexpr"
        # Sort signature by original parameter index so that _generate_launcher's
        # PyArg_ParseTuple format matches the order produced by __call__'s insert
        # logic.  Without this, constexpr entries appended at the end of the dict
        # cause a positional mismatch when they have indices interleaved with
        # non-constexpr parameters.
        signature = dict(sorted(signature.items()))
        launcher_src = _generate_launcher(constants, signature, kernel_placeholder_name)
        # Save only the tl.constexpr constants that are NOT sent by jit.py
        # and must be inserted by __call__.  Specialized scalar constants (e.g.
        # strides specialized to 1) are already included in the non-constexpr
        # values that jit.py sends, and must not be inserted again.
        constexpr_indices = set(getattr(src.fn, 'constexpr_indices', []))
        self.constants = {k: v for k, v in constants.items() if k in constexpr_indices}
        # Later KERNEL_NAME_PLACEHOLDER will be used to assign the kernel name
        # in the following launch function.
        self.launch = compile_module(launcher_src, kernel_placeholder_name)

    def __call__(self, *args, **kwargs):
        ## Insert tl.constexpr values back into their original parameter positions.
        ## The non-constexpr values arrive contiguously after 9 preamble args
        ## (without gaps for constexprs).  We insert each constexpr value at its
        ## original index (processing in ascending order so earlier inserts shift
        ## later positions correctly), matching the sorted signature order that
        ## _generate_launcher's PyArg_ParseTuple format expects.
        args = list(args)
        offset = 9  ## skips first 9 preamble args
        for idx, val in sorted(self.constants.items()):
            args.insert(idx + offset, val)
        self.launch(*args, **kwargs)



class CPUUtils(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CPUUtils, cls).__new__(cls)
        return cls.instance

    # Note:
    # nvidia and amd backends have their corresponding driver.c file that exposes
    # get_device_properties and load_binary using python bindings.
    # (see third_party/nvidia/backend/driver.c)
    # These methods are then used in compiler.py to initialize handles before running
    # the triton kernels.
    # Since we recompile the kernel every time (see compile_module above),
    # and the metadata generated by these functions aren't applicable to the cpu
    # backend, just define the same functions with dummy implementation.
    @staticmethod
    def get_device_properties(device):
        return {
          "max_shared_mem": 2 ** 20,
          "multiprocessor_count": None,
          "sm_clock_rate": None,
          "mem_clock_rate": None,
          "mem_bus_width": None
        }

    # Important note:
    # Since we cannot easy pass function pointers around, we pass along the
    # obj of the kernel so that compile_module above can recompile the
    # module every time.
    @staticmethod
    def load_binary(name, kernel_obj, shared, device):
        return (
          None,       # module
          kernel_obj, # function
          None,       # n_regs
          None,        # n_spills
        )

## Copy and paste from the CPU backend just for compatibility.
class CPUDeviceInterface:

    class HooksTimeAccessor:

        def __init__(self, di):
            self.di = di
            self.record_idx = 0

        def elapsed_time(self, end_event) -> float:
            total_time = 0
            for i in range(self.record_idx, end_event.record_idx):
                total_time += self.di.kernel_times[i]
            return total_time * 1000

        def record(self):
            self.record_idx = len(self.di.kernel_times)

    class TimerEvent:

        def __init__(self):
            self.timer = 0

        def elapsed_time(self, end_event) -> float:
            return (end_event.timer - self.timer) * 1000

        def record(self):
            self.timer = time.perf_counter()

    def __init__(self):
        self.kernel_times = []
        self.last_start = 0
        self.use_hooks = False
        triton.compiler.CompiledKernel.launch_enter_hook = None
        triton.compiler.CompiledKernel.launch_exit_hook = None

    def enable_hook_timing(self):
        self.use_hooks = True
        triton.compiler.CompiledKernel.launch_enter_hook = lambda arg: self._enter_hook()
        triton.compiler.CompiledKernel.launch_exit_hook = lambda arg: self._exit_hook()

    def synchronize(self):
        pass

    def _enter_hook(self):
        self.last_start = time.perf_counter()

    def _exit_hook(self):
        self.kernel_times.append(time.perf_counter() - self.last_start)

    def Event(self, enable_timing=True):
        if self.use_hooks:
            return CPUDeviceInterface.HooksTimeAccessor(self)
        return CPUDeviceInterface.TimerEvent()



class CPUDriver(DriverBase):

    def __init__(self):
        super().__init__()
        self.utils = CPUUtils()
        self.launcher_cls = CPULauncher
        self.binary_ext = "obj"

    # CPU driver won't be automatically chosen unless explicitly set through
    # triton.runtime.driver.set_active(CPUDriver())
    @staticmethod
    def is_active():
        return False

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_device_capability(self):
        return ("cpu", 0)

    def get_current_stream(self, device):
        return 0

    def get_current_device(self):
        # CPU doesn't have a device to return. Return something.
        return "cpu"

    def set_current_device(self, device):
        # CPU doesn't have a device to set
        assert device == "cpu"
        return

    def get_current_target(self):
        return GPUTarget("cpu", 0, 0)

    def get_active_torch_device(self):
        import torch
        return torch.device("cpu")

    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        return args
    
    def map_python_to_cpp_type(self, ty: str) -> str:
        return _ty_to_cpp(ty)
  
    ## Copy and paste from the CPU backend just for compatibility.
    def get_device_interface(self):
        return CPUDeviceInterface()

    def get_empty_cache_for_benchmark(self):
        import torch

        # A typical LLC size for high-end server CPUs are ~400MB.
        cache_size = 512 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device="cpu")


    def clear_cache(self, cache):
        global tl
        import triton.language as tl

        class Pointer:

            def __init__(self, data):
                self.data = data
                self.dtype = data.dtype

            def data_ptr(self):
                return self.data.ctypes.data

        @triton.jit
        def clear_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr, TILE_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            for i in range(0, tl.cdiv(BLOCK_SIZE, TILE_SIZE)):
                offsets = block_start + i * TILE_SIZE + tl.arange(0, TILE_SIZE)
                mask = offsets < n_elements
                tl.store(x_ptr + offsets, 0, mask=mask)

        n_elements = len(cache)
        BLOCK_SIZE = 4096
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        clear_kernel[grid](Pointer(cache), n_elements, BLOCK_SIZE, TILE_SIZE=16)

